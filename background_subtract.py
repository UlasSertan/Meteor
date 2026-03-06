"""
Background subtraction pipeline for meteorological chart images.

Steps:
  1. build_background  - Compute per-pixel mean from a folder of .tif files.
  2. subtract           - Remove background from a single .tif, isolating the pen line.

Usage:
  # Build background from all .tif files in a folder and save it
  python background_subtract.py build --input_dir ./data --output background.tif

  # Subtract background from a single image
  python background_subtract.py subtract --background background.tif --target 1942_EKİM-03.tif --output result.tif

  # Do both in one shot: build background from folder, subtract from each image
  python background_subtract.py full --input_dir . --output_dir ./results
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image


def load_tif_files(input_dir: str) -> list[str]:
    """Return sorted list of .tif file paths in input_dir."""
    files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith((".tif", ".tiff"))
    )
    if not files:
        sys.exit(f"Error: no .tif files found in {input_dir}")
    return files


def determine_common_size(files: list[str]) -> tuple[int, int]:
    """Find the minimum width and height across all images."""
    min_w, min_h = float("inf"), float("inf")
    for f in files:
        with Image.open(f) as img:
            w, h = img.size
            min_w = min(min_w, w)
            min_h = min(min_h, h)
    return int(min_w), int(min_h)


def load_and_resize(path: str, target_size: tuple[int, int]) -> np.ndarray:
    """Load a .tif, resize to target_size (width, height), return as float64 array."""
    with Image.open(path) as img:
        if img.size != target_size:
            img = img.resize(target_size, Image.LANCZOS)
        return np.array(img, dtype=np.float64)


def build_background(input_dir: str) -> tuple[np.ndarray, tuple[int, int]]:
    """Compute per-pixel mean across all .tif files in input_dir."""
    files = load_tif_files(input_dir)
    common_size = determine_common_size(files)
    print(f"Found {len(files)} .tif files, common size: {common_size[0]}x{common_size[1]}")

    accumulator = None
    for i, f in enumerate(files):
        arr = load_and_resize(f, common_size)
        if accumulator is None:
            accumulator = arr
        else:
            accumulator += arr
        print(f"  [{i+1}/{len(files)}] Loaded {os.path.basename(f)}")

    background = accumulator / len(files)
    return background, common_size


def save_tif(array: np.ndarray, output_path: str):
    """Save a numpy array as a .tif file. Clips to [0, 255] uint8."""
    clipped = np.clip(array, 0, 255).astype(np.uint8)
    img = Image.fromarray(clipped)
    img.save(output_path)
    print(f"Saved: {output_path}")


def subtract_background(
    background: np.ndarray,
    target_path: str,
    common_size: tuple[int, int],
) -> np.ndarray:
    """
    Subtract background from target image to isolate the pen line.

    The pen line is darker than the background, so we compute:
        diff = background - target  (positive where pen line exists)

    We then amplify and invert so the pen line appears dark on white.
    """
    target = load_and_resize(target_path, common_size)

    # Raw difference: positive where target is darker than background (= pen line)
    diff = background - target

    # Clip negatives (areas brighter than background are not pen line)
    diff = np.clip(diff, 0, 255)

    # Convert to grayscale magnitude for cleaner output
    if diff.ndim == 3:
        gray_diff = np.mean(diff, axis=2)
    else:
        gray_diff = diff

    # Normalize to use full dynamic range
    max_val = gray_diff.max()
    if max_val > 0:
        gray_diff = (gray_diff / max_val) * 255.0

    # Invert: pen line becomes dark on white background (natural look)
    result = 255.0 - gray_diff

    # Also produce a 3-channel version preserving color info of the difference
    color_diff = np.clip(diff, 0, 255)
    color_max = color_diff.max()
    if color_max > 0:
        color_diff = (color_diff / color_max) * 255.0

    return result, color_diff


def cmd_build(args):
    background, _ = build_background(args.input_dir)
    save_tif(background, args.output)


def cmd_subtract(args):
    # Load pre-computed background
    with Image.open(args.background) as bg_img:
        background = np.array(bg_img, dtype=np.float64)
        common_size = bg_img.size  # (width, height)

    grayscale, color = subtract_background(background, args.target, common_size)
    save_tif(grayscale, args.output)

    # Also save color diff
    base, ext = os.path.splitext(args.output)
    color_path = f"{base}_color{ext}"
    save_tif(color, color_path)


def cmd_full(args):
    """Build background and subtract from every image in the folder."""
    background, common_size = build_background(args.input_dir)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save the computed background
    bg_path = os.path.join(output_dir, "background.tif")
    save_tif(background, bg_path)

    # Process each file
    files = load_tif_files(args.input_dir)
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        grayscale, color = subtract_background(background, f, common_size)

        save_tif(grayscale, os.path.join(output_dir, f"{name}_line.tif"))
        save_tif(color, os.path.join(output_dir, f"{name}_line_color.tif"))

    print(f"\nDone! Results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Background subtraction for meteorological chart TIFs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build background image from a folder of .tif files")
    p_build.add_argument("--input_dir", required=True, help="Folder with .tif files")
    p_build.add_argument("--output", default="background.tif", help="Output background .tif path")
    p_build.set_defaults(func=cmd_build)

    p_sub = sub.add_parser("subtract", help="Subtract background from a single .tif")
    p_sub.add_argument("--background", required=True, help="Path to background .tif")
    p_sub.add_argument("--target", required=True, help="Path to target .tif")
    p_sub.add_argument("--output", default="result.tif", help="Output path")
    p_sub.set_defaults(func=cmd_subtract)

    p_full = sub.add_parser("full", help="Build background + subtract from all images")
    p_full.add_argument("--input_dir", required=True, help="Folder with .tif files")
    p_full.add_argument("--output_dir", default="results", help="Output folder")
    p_full.set_defaults(func=cmd_full)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
