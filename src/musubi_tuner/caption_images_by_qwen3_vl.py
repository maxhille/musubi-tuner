#!/usr/bin/env python3
"""Generate captions for images using Qwen3-VL (e.g. the Krea 2 text encoder weights).

Reuses the local Qwen3-VL safetensors already present for Krea 2 training, so no
separate captioning model (GGUF/llama-server) is needed.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from musubi_tuner.dataset import image_video_dataset
from musubi_tuner.krea2.krea2_encoder import _load_qwen3_vl_model

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DEFAULT_MAX_SIZE = 1280

# Default prompt: the "subtraction strategy" used by this repository's dataset captioning
# (cf. caption.go): everything is captioned except the face, so the likeness binds to the
# trigger token. Used when --trigger is given; override with --prompt for full control.
DEFAULT_TRIGGER_PROMPT = (
    "Describe this image for AI dataset training using 1-2 natural, direct sentences. "
    "Always start the description directly with '{trigger}' followed by their action or pose "
    "(e.g., '{trigger} stands...', '{trigger} sits...'). "
    "Describe the shot framing, pose, clothing, background environment, and lighting. "
    "DO NOT use introductory filler like 'A shot of', 'Medium shot of', or '{trigger}, in a medium shot'. "
    "DO NOT describe the subject's face, facial features, skin tone, or eye color. "
    "Keep the entire description under 35 words."
)

# Qwen3 models may emit a thinking block before the actual answer; strip it if present.
_THINK_TAG = "think"


def _strip_thinking(text: str) -> str:
    open_tag = f"<{_THINK_TAG}>"
    close_tag = f"</{_THINK_TAG}>"
    if close_tag in text:
        text = text.split(close_tag, 1)[1]
    elif open_tag in text:
        # Unterminated thinking block: no answer was produced.
        text = ""
    return text.strip()



def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate captions for images using Qwen3-VL")

    parser.add_argument("--image_dir", type=str, required=True, help="Path to directory containing images")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to Qwen3-VL model safetensors (e.g. the Krea 2 text encoder qwen3vl_4b_bf16.safetensors)",
    )
    parser.add_argument("--output_file", type=str, required=False, help="Output JSONL file path (required for 'jsonl' format)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate (default: 512)")
    parser.add_argument(
        "--trigger",
        type=str,
        default=None,
        help="Trigger word (dataset subject name): uses the default identity-captioning prompt with this trigger",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for caption generation (supports \\n for newlines); overrides --trigger",
    )
    parser.add_argument(
        "--processor_path",
        type=str,
        required=True,
        help="Local directory with the Qwen3-VL processor/tokenizer files (preprocessor_config.json, tokenizer.json, "
        "tokenizer_config.json, chat_template.json); config files from any Qwen3-VL size are interchangeable",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=DEFAULT_MAX_SIZE,
        help=f"Maximum image size (default: {DEFAULT_MAX_SIZE}). Images are resized so the total pixel area stays within (max_size x max_size)",
    )
    parser.add_argument("--fp8_vl", action="store_true", help="Load the Qwen3-VL model in fp8 precision")
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["jsonl", "text"],
        default="text",
        help="Output format: 'jsonl' for JSONL file or 'text' for individual text files (default: text)",
    )

    args = parser.parse_args()
    if args.prompt is None and args.trigger is None:
        parser.error("either --prompt or --trigger is required")
    return args


def load_model_and_processor(
    model_path: str, processor_path: str, device: torch.device, max_size: int = DEFAULT_MAX_SIZE, fp8_vl: bool = False
):
    """Load Qwen3-VL model and processor (fully local: no Hub access)"""
    logger.info(f"Loading model from: {model_path}")

    # Image size is constrained via pixel-area budgets; the processor resizes internally.
    min_pixels = 256 * 256
    max_pixels = max_size * max_size
    processor = AutoProcessor.from_pretrained(processor_path, min_pixels=min_pixels, max_pixels=max_pixels)

    dtype = torch.float8_e4m3fn if fp8_vl else torch.bfloat16
    model = _load_qwen3_vl_model(model_path, dtype=dtype, device=device, disable_mmap=True)

    logger.info(f"Model loaded successfully on device: {model.device}")
    return processor, model


def generate_caption(
    processor,
    model,
    image_path: str,
    device: torch.device,
    max_new_tokens: int,
    prompt: str,
    fp8_vl: bool = False,
) -> str:
    """Generate caption for a single image"""
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Disable thinking mode if the chat template supports it (Qwen3 models).
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except (TypeError, KeyError):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    if fp8_vl:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    else:
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    caption = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Strip a thinking block if the model emitted one anyway.
    caption = _strip_thinking(caption)
    return caption if caption else ""


def process_images(args):
    """Main processing function"""
    if args.output_format == "jsonl" and not args.output_file:
        raise ValueError("--output_file is required when --output_format is 'jsonl'")

    if args.prompt is not None:
        prompt = args.prompt.replace("\\n", "\n")
    else:
        prompt = DEFAULT_TRIGGER_PROMPT.format(trigger=args.trigger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Output format: {args.output_format}")
    if args.fp8_vl:
        logger.info("Using fp8 precision for model")

    image_files = image_video_dataset.glob_images(args.image_dir)
    logger.info(f"Found {len(image_files)} image files")

    processor, model = load_model_and_processor(args.model_path, args.processor_path, device, args.max_size, args.fp8_vl)

    if args.output_format == "jsonl":
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(args.output_file, "w", encoding="utf-8") as f:
            for image_path in tqdm(image_files, desc="Generating captions"):
                caption = generate_caption(processor, model, image_path, device, args.max_new_tokens, prompt, args.fp8_vl)
                entry = {"image_path": image_path, "caption": caption}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()

        logger.info(f"Caption generation completed. Results saved to: {args.output_file}")
    else:
        for image_path in tqdm(image_files, desc="Generating captions"):
            caption = generate_caption(processor, model, image_path, device, args.max_new_tokens, prompt, args.fp8_vl)
            text_file_path = Path(image_path).with_suffix(".txt")
            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write(caption)

        logger.info("Caption generation completed. Text files saved alongside each image.")


def main():
    """Main function"""
    args = parse_args()
    process_images(args)


if __name__ == "__main__":
    main()
