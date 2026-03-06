"""
Digitize the pen line from background-subtracted meteorological chart images.

Extracts the y-position of the pen line for each x-column and outputs CSV.

Usage:
  # Digitize a single image
  python digitize_line.py single --input results_1991_TEMMUZ/1991_TEMMUZ-15_line.tif --output line_15.csv

  # Digitize all _line.tif files in a results folder
  python digitize_line.py batch --input_dir results_1991_TEMMUZ --output_dir csv_1991_TEMMUZ
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image


def suppress_vertical_lines(signal: np.ndarray) -> np.ndarray:
    """
    Suppress vertical grid line artifacts.

    Strategy: vertical grid remnants elevate an entire column uniformly.
    The pen line only elevates a small band of rows. By subtracting each
    column's median, uniform vertical features vanish while the localized
    pen line peak survives (median is robust to the few bright pen pixels).
    """
    col_medians = np.median(signal, axis=0, keepdims=True)
    filtered = np.clip(signal - col_medians, 0, 255)
    return filtered


def suppress_horizontal_lines(signal: np.ndarray) -> np.ndarray:
    """
    Suppress horizontal grid line remnants the same way — subtract row medians.
    """
    row_medians = np.median(signal, axis=1, keepdims=True)
    filtered = np.clip(signal - row_medians, 0, 255)
    return filtered


def digitize_image(
    image_path: str,
    signal_threshold: float = 25.0,
    window: int = 30,
    min_column_signal: float = 400.0,
    margin_top: int = 30,
    margin_bottom: int = 30,
) -> tuple[np.ndarray, int]:
    """
    Extract pen line y-coordinates from a background-subtracted grayscale image.

    Returns:
        Tuple of (data array of shape (N, 3): [x_pixel, y_pixel, y_normalized],
                  image_height)
    """
    img = np.array(Image.open(image_path))
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    img = img.astype(np.float64)
    h, w = img.shape

    # Invert: pen line becomes bright (high signal)
    signal = 255.0 - img

    # Mask out top/bottom margins (text, labels, paper edges)
    signal[:margin_top, :] = 0
    signal[h - margin_bottom:, :] = 0

    # Suppress grid line remnants (vertical then horizontal)
    signal = suppress_vertical_lines(signal)
    signal = suppress_horizontal_lines(signal)

    y_coords = np.arange(h, dtype=np.float64)
    results = []

    for col in range(w):
        profile = signal[:, col]
        peak_y = int(np.argmax(profile))
        peak_val = profile[peak_y]

        if peak_val < signal_threshold:
            continue

        # Weighted centroid in a local window around the peak
        y_start = max(0, peak_y - window)
        y_end = min(h, peak_y + window + 1)
        local_signal = profile[y_start:y_end]
        local_y = y_coords[y_start:y_end]

        local_signal = np.where(local_signal > signal_threshold * 0.3, local_signal, 0)

        total = local_signal.sum()
        if total < min_column_signal:
            continue

        centroid_y = np.average(local_y, weights=local_signal)
        y_norm = 1.0 - (centroid_y / (h - 1))
        results.append((col, centroid_y, y_norm))

    data = np.array(results) if results else np.empty((0, 3))
    return data, h


def remove_outliers(data: np.ndarray, max_jump: float = 40.0, passes: int = 3) -> np.ndarray:
    """Remove points that jump too far from their neighbors. Multi-pass for thorough cleaning."""
    for _ in range(passes):
        if len(data) < 3:
            return data

        mask = np.ones(len(data), dtype=bool)
        y_vals = data[:, 1]

        for i in range(1, len(data) - 1):
            diff_prev = abs(y_vals[i] - y_vals[i - 1])
            diff_next = abs(y_vals[i] - y_vals[i + 1])
            if diff_prev > max_jump and diff_next > max_jump:
                mask[i] = False

        data = data[mask]

    return data


def median_smooth(data: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Apply median smoothing to the y values — more robust to remaining outliers than mean."""
    if len(data) < kernel_size:
        return data

    smoothed = data.copy()
    half = kernel_size // 2
    y_vals = data[:, 1]
    h_img = 1.0 / (1.0 - data[0, 2]) * data[0, 1] if data[0, 2] < 1.0 else 1.0

    for i in range(len(data)):
        start = max(0, i - half)
        end = min(len(data), i + half + 1)
        smoothed[i, 1] = np.median(y_vals[start:end])

    # Recalculate y_normalized from smoothed y_pixel
    # Use the image height derived from original data
    if len(data) > 0 and data[0, 2] < 1.0:
        img_h_approx = data[0, 1] / (1.0 - data[0, 2])
        smoothed[:, 2] = 1.0 - (smoothed[:, 1] / img_h_approx)

    return smoothed


def save_csv(data: np.ndarray, output_path: str):
    """Save digitized line data to CSV."""
    with open(output_path, "w") as f:
        f.write("x_pixel,y_pixel,y_normalized\n")
        for row in data:
            f.write(f"{int(row[0])},{row[1]:.2f},{row[2]:.6f}\n")
    print(f"Saved: {output_path} ({len(data)} points)")


def process_single(image_path: str, output_path: str):
    """Digitize a single image and save CSV."""
    data, h = digitize_image(image_path)
    if len(data) == 0:
        print(f"Warning: no line detected in {image_path}")
        return

    data = remove_outliers(data, max_jump=40.0, passes=3)
    data = median_smooth(data, kernel_size=7)

    save_csv(data, output_path)


def cmd_single(args):
    process_single(args.input, args.output)


def cmd_batch(args):
    """Digitize all _line.tif files in a directory."""
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith("_line.tif") and not f.endswith("_line_color.tif")
    )

    if not files:
        sys.exit(f"Error: no _line.tif files found in {input_dir}")

    print(f"Found {len(files)} files to digitize")

    for f in files:
        name = f.replace("_line.tif", "")
        input_path = os.path.join(input_dir, f)
        output_path = os.path.join(output_dir, f"{name}.csv")
        process_single(input_path, output_path)

    print(f"\nDone! CSVs saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Digitize pen lines from background-subtracted TIFs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_single = sub.add_parser("single", help="Digitize a single image")
    p_single.add_argument("--input", required=True, help="Path to _line.tif")
    p_single.add_argument("--output", default="line.csv", help="Output CSV path")
    p_single.set_defaults(func=cmd_single)

    p_batch = sub.add_parser("batch", help="Digitize all _line.tif files in a folder")
    p_batch.add_argument("--input_dir", required=True, help="Folder with _line.tif files")
    p_batch.add_argument("--output_dir", default="csv_output", help="Output CSV folder")
    p_batch.set_defaults(func=cmd_batch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
