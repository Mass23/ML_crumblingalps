"""
create_landslide_traindata.py

Generates training data for the pixel landslide AI model.
For each input image, produces:
  - A preprocessed source image (source_{name}.png)
  - Individual video frames (frames_{name}/frame_XXXXX.png)
  - A video file (video_{name}.mp4)

All core landslide simulation logic is preserved from the original script.

Usage:
    python create_landslide_traindata.py \\
        --input-dir data/raw-images \\
        --output-dir data/training \\
        --duration 4 \\
        --fps 14 \\
        --target-width 1920 \\
        --target-height 1080 \\
        --num-workers 4
"""

import os
import argparse
import glob
import random
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from typing import Optional, Tuple

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Image Loading & Preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess_image(
    filepath: str,
    target_size: Tuple[int, int] = (2880, 1620),
) -> Optional[Image.Image]:
    """
    Load a single image file and apply preprocessing.

    Supports JPG, PNG, and HEIC formats.
    Resizes and center-crops to target_size, normalizes colors, and adds noise.

    Args:
        filepath: Path to the image file
        target_size: (width, height) tuple for the output image

    Returns:
        Preprocessed PIL Image, or None on failure
    """
    try:
        img = Image.open(filepath).convert("RGB")
    except Exception as e:
        print(f"  Could not open {filepath}: {e}")
        return None

    aspect_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]

    if aspect_ratio > target_ratio:
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)

    img = img.resize((new_width, new_height), Image.BICUBIC)
    left = (img.width - target_size[0]) // 2
    top = (img.height - target_size[1]) // 2
    img = img.crop((left, top, left + target_size[0], top + target_size[1]))

    img = normalize_colors(img)
    img = add_gaussian_noise(img)

    return img


def normalize_colors(img: Image.Image) -> Image.Image:
    """
    Normalize image colors by standardizing per-channel statistics.

    Args:
        img: Input PIL Image

    Returns:
        Color-normalized PIL Image
    """
    arr = np.asarray(img).astype(np.float32)
    mean = arr.mean(axis=(0, 1), keepdims=True)
    std = arr.std(axis=(0, 1), keepdims=True)
    arr = (arr - mean) / (std + 1e-5)
    arr = arr * 40 + 128
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_gaussian_noise(img: Image.Image, sigma: float = 10) -> Image.Image:
    """
    Add Gaussian noise to an image.

    Args:
        img: Input PIL Image
        sigma: Standard deviation of the Gaussian noise

    Returns:
        Noisy PIL Image
    """
    arr = np.asarray(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


# ---------------------------------------------------------------------------
# Landslide Simulation (original logic, fully preserved)
# ---------------------------------------------------------------------------

def simulate_landslides(
    img: Image.Image,
    duration: int = 4,
    fps: int = 14,
) -> list[Image.Image]:
    """
    Simulate pixel-art style landslide effect on an image.

    Generates a sequence of frames where blocks of pixels slide downward,
    creating a cascading landslide effect.

    Args:
        img: Input PIL Image (source frame)
        duration: Video duration in seconds
        fps: Frames per second

    Returns:
        List of PIL Image frames representing the animation
    """
    total_frames = duration * fps
    base_image = np.array(img).copy()
    frames = []

    h, w, _ = base_image.shape
    active_landslides = []
    MAX_ACTIVE = 1000

    def start_landslide(
        block_size: int = 400,
        x: Optional[int] = None,
        y: Optional[int] = None,
        generation: int = 0,
        max_fall_dist: int = 1000,
    ) -> dict:
        """
        Initialize a single landslide block.

        Args:
            block_size: Size of the sliding block in pixels
            x: Starting x-coordinate (random if None)
            y: Starting y-coordinate (random if None)
            generation: Cascade generation number (0 = initial)
            max_fall_dist: Maximum distance the block can fall

        Returns:
            Dict describing the landslide block state
        """
        if x is None:
            x = random.randint(0, w - block_size)
        if y is None:
            y = random.randint(0, h // 2 - block_size)
        fall_distance = random.randint(150, max_fall_dist)
        duration_frames = random.randint(20, 200)
        dy_per_frame = fall_distance / duration_frames
        dx_per_frame = random.uniform(-1.0, 1.0)

        block_h = min(block_size, h - y)
        block_w = min(block_size, w - x)
        block_pixels = base_image[y:y + block_h, x:x + block_w].copy()

        return {
            "x": x,
            "y": y,
            "block_size": block_size,
            "fall_distance": fall_distance,
            "duration_frames": duration_frames,
            "frames_moved": 0,
            "dy_per_frame": dy_per_frame,
            "dx_per_frame": dx_per_frame,
            "block_pixels": block_pixels,
            "generation": generation,
        }

    for frame_idx in range(total_frames):
        img_array = base_image.copy()

        if len(active_landslides) < MAX_ACTIVE:
            if random.random() < 0.05:
                active_landslides.append(start_landslide(random.randint(100, 500)))

        new_landslides = []
        for ls in active_landslides:
            x, y, bs = ls["x"], ls["y"], ls["block_size"]
            frames_moved = ls["frames_moved"] + 1
            dy = int(ls["dy_per_frame"])
            dx = int(ls["dx_per_frame"])
            new_y = y + dy
            new_x = x + dx

            block_h = min(bs, h - new_y)
            block_w = min(bs, w - new_x)

            new_x = max(0, min(w - block_w, new_x))
            new_y = max(0, min(h - block_h, new_y))

            if block_h <= 0 or block_w <= 0:
                continue

            block_old = ls["block_pixels"]

            img_array[new_y:new_y + block_h, new_x:new_x + block_w] = block_old[:block_h, :block_w]
            base_image[new_y:new_y + block_h, new_x:new_x + block_w] = block_old[:block_h, :block_w]

            if frames_moved < ls["duration_frames"]:
                ls["x"] = new_x
                ls["y"] = new_y
                ls["frames_moved"] = frames_moved

                if ls["generation"] < 5 and random.random() < 0.2:
                    if len(new_landslides) < MAX_ACTIVE:
                        size = random.randint(
                            int(ls["block_size"] * 0.5),
                            int(ls["block_size"] * 1.2),
                        )
                        angle = random.uniform(-0.7, 0.7)
                        offset = int(ls["block_size"] * angle)
                        sx = ls["x"] + offset
                        sy = ls["y"] + random.randint(
                            ls["block_size"] // 2, ls["block_size"]
                        )
                        sx = max(0, min(w - size, sx))
                        sy = max(0, min(h - size, sy))
                        new_landslides.append(
                            start_landslide(
                                size,
                                x=sx,
                                y=sy,
                                generation=ls["generation"] + 1,
                                max_fall_dist=500,
                            )
                        )

                new_landslides.append(ls)

        active_landslides = [ls for ls in new_landslides if ls["frames_moved"] < ls["duration_frames"]]
        frames.append(Image.fromarray(img_array.copy()))

    return frames


# ---------------------------------------------------------------------------
# Saving Utilities
# ---------------------------------------------------------------------------

def save_video(frames: list[Image.Image], filename: str, fps: int = 14) -> None:
    """
    Save a list of PIL frames as an MP4 video.

    Args:
        frames: List of PIL Image frames
        filename: Output video file path
        fps: Frames per second for the output video
    """
    h, w = frames[0].size[1], frames[0].size[0]
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        frame = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()


def save_frames(frames: list[Image.Image], frames_dir: Path) -> None:
    """
    Save individual video frames as PNG files.

    Args:
        frames: List of PIL Image frames
        frames_dir: Directory to save frame images
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        frame_path = frames_dir / f"frame_{i:05d}.png"
        frame.save(str(frame_path))


# ---------------------------------------------------------------------------
# Per-image Processing
# ---------------------------------------------------------------------------

def process_image(
    filepath: str,
    output_dir: str,
    duration: int,
    fps: int,
    target_width: int,
    target_height: int,
) -> None:
    """
    Process a single image: preprocess, simulate landslide, and save outputs.

    Args:
        filepath: Path to input image file
        output_dir: Root directory for training data output
        duration: Video duration in seconds
        fps: Frames per second
        target_width: Target image width in pixels
        target_height: Target image height in pixels
    """
    output_path = Path(output_dir)
    name = Path(filepath).stem

    # Check if already processed (skip)
    source_png = output_path / f"source_{name}.png"
    video_mp4 = output_path / f"video_{name}.mp4"
    frames_dir = output_path / f"frames_{name}"

    if source_png.exists() and video_mp4.exists() and frames_dir.exists():
        print(f"  Skipping (already processed): {name}")
        return

    print(f"  Processing: {name}")

    img = load_and_preprocess_image(filepath, target_size=(target_width, target_height))
    if img is None:
        print(f"  Failed to load: {filepath}")
        return

    # Save preprocessed source image
    img.save(str(source_png))

    # Simulate landslide and get frames
    frames = simulate_landslides(img, duration=duration, fps=fps)

    # Save individual frames
    save_frames(frames, frames_dir)

    # Save video
    save_video(frames, str(video_mp4), fps=fps)

    print(f"  Done: {name} -> {len(frames)} frames saved")


def process_image_worker(args: tuple) -> None:
    """
    Multiprocessing worker that unpacks arguments and calls process_image.

    Args:
        args: Tuple of (filepath, output_dir, duration, fps, target_width, target_height)
    """
    process_image(*args)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate landslide training data from mountain images"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Folder containing input images (JPG/PNG/HEIC)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save training data",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=4,
        help="Video duration in seconds (default: 4)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=14,
        help="Frames per second (default: 14, matching SVD)",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=512,
        help="Target image width in pixels (default: 512)",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=512,
        help="Target image height in pixels (default: 512)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Process only the first image (for testing)",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all supported image files
    input_dir = Path(args.input_dir)
    patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
    if HEIF_AVAILABLE:
        patterns += ["*.HEIC", "*.heic", "*.heif", "*.HEIF"]

    filepaths = []
    for pattern in patterns:
        filepaths.extend(glob.glob(str(input_dir / pattern)))
    filepaths = sorted(set(filepaths))

    if not filepaths:
        print(f"No supported images found in '{args.input_dir}'")
        return

    if args.test:
        filepaths = filepaths[:1]
        print(f"Test mode: processing 1 image")

    print(f"Found {len(filepaths)} images. Output dir: '{args.output_dir}'")
    print(f"Settings: {args.duration}s @ {args.fps}fps, {args.target_width}x{args.target_height}px")

    task_args = [
        (fp, args.output_dir, args.duration, args.fps, args.target_width, args.target_height)
        for fp in filepaths
    ]

    if args.num_workers > 1:
        print(f"Using {args.num_workers} parallel workers...")
        with Pool(processes=args.num_workers) as pool:
            pool.map(process_image_worker, task_args)
    else:
        for task in task_args:
            process_image_worker(task)

    print(f"\nAll done! Training data saved to '{args.output_dir}'")


if __name__ == "__main__":
    main()
