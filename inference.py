"""
inference.py

Generates pixel landslide effect videos from a single mountain image using
a fine-tuned Stable Video Diffusion model with LoRA weights.

Usage:
    # Single image
    python inference.py \\
        --image my_mountain.jpg \\
        --lora-weights checkpoints/lora_final \\
        --output output.mp4

    # Batch inference on a folder
    python inference.py \\
        --input-dir my_images/ \\
        --lora-weights checkpoints/lora_final \\
        --output-dir outputs/
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import numpy as np
from PIL import Image
import cv2

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video


# ---------------------------------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (512, 512),
) -> Image.Image:
    """
    Load and preprocess an input image for the SVD model.

    Resizes and center-crops the image to the target size.

    Args:
        image_path: Path to the input image file
        target_size: (width, height) for the output image

    Returns:
        Preprocessed PIL Image
    """
    img = Image.open(image_path).convert("RGB")

    target_w, target_h = target_size
    aspect_ratio = img.width / img.height
    target_ratio = target_w / target_h

    if aspect_ratio > target_ratio:
        new_height = target_h
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = target_w
        new_height = int(new_width / aspect_ratio)

    img = img.resize((new_width, new_height), Image.BICUBIC)
    left = (img.width - target_w) // 2
    top = (img.height - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))

    return img


# ---------------------------------------------------------------------------
# Video Saving
# ---------------------------------------------------------------------------

def save_video_mp4(
    frames: list[Image.Image],
    output_path: str,
    fps: int = 14,
) -> None:
    """
    Save a list of PIL frames as an MP4 video using OpenCV.

    Args:
        frames: List of PIL Image frames
        output_path: Output video file path
        fps: Frames per second
    """
    h, w = frames[0].size[1], frames[0].size[0]
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for frame in frames:
        bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        out.write(bgr)
    out.release()


def save_webp_animation(
    frames: list[Image.Image],
    output_path: str,
    fps: int = 14,
) -> None:
    """
    Save a list of PIL frames as an animated WebP file.

    Args:
        frames: List of PIL Image frames
        output_path: Output WebP file path
        fps: Frames per second
    """
    duration_ms = int(1000 / fps)
    rgb_frames = [f.convert("RGB") for f in frames]
    rgb_frames[0].save(
        output_path,
        format="WEBP",
        save_all=True,
        append_images=rgb_frames[1:],
        duration=duration_ms,
        loop=0,
    )


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_pipeline(
    base_model: str,
    lora_weights: Optional[str],
    device: str,
    dtype: torch.dtype,
) -> StableVideoDiffusionPipeline:
    """
    Load the SVD pipeline with optional LoRA weights.

    Args:
        base_model: HuggingFace model ID or local path to the base SVD model
        lora_weights: Path to LoRA weights directory, or None to use base model
        device: Target device ('cuda', 'cpu', etc.)
        dtype: Model dtype (torch.float16 or torch.float32)

    Returns:
        Loaded and configured SVD pipeline
    """
    print(f"Loading base model: {base_model}")
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )

    if lora_weights is not None:
        lora_path = Path(lora_weights)
        if lora_path.exists():
            print(f"Loading LoRA weights from: {lora_weights}")
            pipeline.unet.load_attn_procs(lora_weights)
        else:
            print(f"WARNING: LoRA weights not found at '{lora_weights}', using base model")

    pipeline = pipeline.to(device)

    # Enable memory optimizations
    if hasattr(pipeline, "enable_model_cpu_offload") and device == "cuda":
        pipeline.enable_model_cpu_offload()

    pipeline.set_progress_bar_config(disable=False)

    return pipeline


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_video(
    pipeline: StableVideoDiffusionPipeline,
    image: Image.Image,
    num_frames: int = 14,
    num_inference_steps: int = 25,
    fps: int = 14,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02,
    seed: Optional[int] = None,
) -> list[Image.Image]:
    """
    Generate a landslide effect video from a single image.

    Args:
        pipeline: Loaded SVD pipeline
        image: Input PIL Image (preprocessed)
        num_frames: Number of frames to generate
        num_inference_steps: Denoising steps (more = better quality, slower)
        fps: Frames per second for timing conditioning
        motion_bucket_id: Controls motion intensity (higher = more motion)
        noise_aug_strength: Noise augmentation for conditioning frame
        seed: Random seed for reproducibility

    Returns:
        List of PIL Image frames
    """
    generator = None
    if seed is not None:
        generator = torch.manual_seed(seed)

    result = pipeline(
        image,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        fps=fps,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        generator=generator,
        decode_chunk_size=2,
    )

    return result.frames[0]


# ---------------------------------------------------------------------------
# Single Image Inference
# ---------------------------------------------------------------------------

def run_single(
    pipeline: StableVideoDiffusionPipeline,
    image_path: str,
    output_path: str,
    args: argparse.Namespace,
) -> None:
    """
    Run inference on a single image and save the output.

    Args:
        pipeline: Loaded SVD pipeline
        image_path: Path to input image
        output_path: Path to output video (MP4)
        args: Parsed CLI arguments
    """
    print(f"Processing: {image_path}")

    image = preprocess_image(image_path, target_size=(args.image_width, args.image_height))

    frames = generate_video(
        pipeline=pipeline,
        image=image,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        fps=args.fps,
        motion_bucket_id=args.motion_bucket_id,
        noise_aug_strength=args.noise_aug_strength,
        seed=args.seed,
    )

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save MP4
    save_video_mp4(frames, output_path, fps=args.fps)
    print(f"  Saved MP4: {output_path}")

    # Optionally save WebP animation
    if args.save_webp:
        webp_path = str(Path(output_path).with_suffix(".webp"))
        save_webp_animation(frames, webp_path, fps=args.fps)
        print(f"  Saved WebP: {webp_path}")


# ---------------------------------------------------------------------------
# Batch Inference
# ---------------------------------------------------------------------------

def run_batch(
    pipeline: StableVideoDiffusionPipeline,
    input_dir: str,
    output_dir: str,
    args: argparse.Namespace,
) -> None:
    """
    Run inference on all images in a directory.

    Args:
        pipeline: Loaded SVD pipeline
        input_dir: Directory containing input images
        output_dir: Directory to save output videos
        args: Parsed CLI arguments
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all image files
    extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
    image_files = sorted(set(image_files))

    if not image_files:
        print(f"No images found in '{input_dir}'")
        return

    print(f"Found {len(image_files)} images in '{input_dir}'")

    for i, image_file in enumerate(image_files):
        output_video = output_path / f"{image_file.stem}_landslide.mp4"

        # Skip if already processed
        if output_video.exists():
            print(f"[{i+1}/{len(image_files)}] Skipping (exists): {image_file.name}")
            continue

        print(f"[{i+1}/{len(image_files)}] ", end="")
        run_single(pipeline, str(image_file), str(output_video), args)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pixel landslide videos from mountain images using fine-tuned SVD"
    )

    # Input/output
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--image", type=str, default=None,
                             help="Path to a single input image")
    input_group.add_argument("--input-dir", type=str, default=None,
                             help="Directory of images for batch inference")

    parser.add_argument("--output", type=str, default="output.mp4",
                        help="Output video path for single image mode (default: output.mp4)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory for batch mode (default: outputs)")

    # Model
    parser.add_argument("--base-model", type=str,
                        default="stabilityai/stable-video-diffusion-img2vid",
                        help="Base SVD model ID or local path")
    parser.add_argument("--lora-weights", type=str, default=None,
                        help="Path to trained LoRA weights directory")

    # Generation settings
    parser.add_argument("--num-frames", type=int, default=14,
                        help="Number of frames to generate (default: 14)")
    parser.add_argument("--fps", type=int, default=14,
                        help="Output video FPS (default: 14)")
    parser.add_argument("--num-inference-steps", type=int, default=25,
                        help="Number of denoising steps (default: 25)")
    parser.add_argument("--motion-bucket-id", type=int, default=127,
                        help="Motion intensity control (default: 127)")
    parser.add_argument("--noise-aug-strength", type=float, default=0.02,
                        help="Noise augmentation strength (default: 0.02)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible generation")

    # Image preprocessing
    parser.add_argument("--image-width", type=int, default=512,
                        help="Input image width (default: 512)")
    parser.add_argument("--image-height", type=int, default=512,
                        help="Input image height (default: 512)")

    # Output options
    parser.add_argument("--save-webp", action="store_true",
                        help="Also save output as animated WebP")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "mps"],
                        help="Device to run inference on (default: cuda)")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use fp16 precision (default: True)")

    args = parser.parse_args()

    if args.image is None and args.input_dir is None:
        parser.error("Provide either --image or --input-dir")

    # Select device and dtype
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    dtype = torch.float16 if args.fp16 and device == "cuda" else torch.float32

    # Load model
    pipeline = load_pipeline(
        base_model=args.base_model,
        lora_weights=args.lora_weights,
        device=device,
        dtype=dtype,
    )

    if args.image is not None:
        run_single(pipeline, args.image, args.output, args)
    else:
        run_batch(pipeline, args.input_dir, args.output_dir, args)


if __name__ == "__main__":
    main()
