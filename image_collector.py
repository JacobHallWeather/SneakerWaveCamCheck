import argparse
import time
from datetime import datetime
from pathlib import Path

import requests

from visual_review_tool import VisualReviewTool


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect periodic webcam images into KymoSlices and write one daily kymograph into KymoDay."
    )
    parser.add_argument(
        "--image-url",
        default="https://biz.parks.wa.gov/webcams/CapedNorthHead1.jpg",
        help="Source image URL",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=3600,
        help="Capture interval in seconds (3600 hourly, 1800 every 30 minutes)",
    )
    parser.add_argument("--slices-dir", default="KymoSlices", help="Folder for captured source images")
    parser.add_argument("--kymoday-dir", default="KymoDay", help="Folder for daily merged kymograph images")
    parser.add_argument(
        "--images-per-day",
        type=int,
        default=None,
        help="Expected captures per day; defaults to round(86400/interval-seconds)",
    )
    parser.add_argument(
        "--target-kymo-width",
        type=int,
        default=2400,
        help="Target daily kymograph width in pixels",
    )
    parser.add_argument(
        "--slice-angle-deg",
        type=float,
        default=0.0,
        help="Slice angle in degrees from vertical (default center vertical)",
    )
    parser.add_argument(
        "--slice-center-x-ratio",
        type=float,
        default=0.5,
        help="Slice center X position ratio (0.0 left, 1.0 right)",
    )
    parser.add_argument(
        "--slice-center-y-ratio",
        type=float,
        default=0.5,
        help="Slice center Y position ratio (0.0 top, 1.0 bottom)",
    )
    parser.add_argument("--slice-start", default=None, help="Optional explicit slice start point x,y")
    parser.add_argument("--slice-end", default=None, help="Optional explicit slice end point x,y")
    parser.add_argument("--timeout-seconds", type=int, default=20, help="HTTP request timeout")
    return parser.parse_args()


def infer_images_per_day(interval_seconds, explicit_value):
    if explicit_value is not None:
        return max(1, int(explicit_value))
    interval = max(1, int(interval_seconds))
    return max(1, int(round(86400 / float(interval))))


def capture_image(image_url, destination_path, timeout_seconds):
    response = requests.get(image_url, timeout=max(1, int(timeout_seconds)))
    response.raise_for_status()
    destination_path.write_bytes(response.content)


def day_image_paths(slices_dir, day_key):
    return sorted(slices_dir.glob(f"{day_key}_*.jpg"))


def build_daily_kymograph(day_key, slices_dir, kymoday_dir, tool, images_per_day, target_kymo_width):
    images = day_image_paths(slices_dir, day_key)
    if not images:
        print(f"No captures found for {day_key}; skipping daily kymograph.")
        return None

    output_path = kymoday_dir / f"{day_key}_kymograph.jpg"
    tool.create_kymograph_from_paths(
        image_paths=images,
        output_path=str(output_path),
        images_per_day=images_per_day,
        target_kymo_width=target_kymo_width,
    )
    print(f"Daily kymograph written: {output_path}")
    return output_path


def main():
    args = parse_args()

    slices_dir = Path(args.slices_dir)
    kymoday_dir = Path(args.kymoday_dir)
    slices_dir.mkdir(parents=True, exist_ok=True)
    kymoday_dir.mkdir(parents=True, exist_ok=True)

    images_per_day = infer_images_per_day(args.interval_seconds, args.images_per_day)
    tool = VisualReviewTool(
        slice_angle_deg=args.slice_angle_deg,
        slice_center_x_ratio=args.slice_center_x_ratio,
        slice_center_y_ratio=args.slice_center_y_ratio,
        slice_start=VisualReviewTool._parse_point(args.slice_start),
        slice_end=VisualReviewTool._parse_point(args.slice_end),
    )

    print(f"Collecting to: {slices_dir}")
    print(f"Daily outputs: {kymoday_dir}")
    print(f"Interval: {args.interval_seconds}s | images/day: {images_per_day}")

    active_day = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        while True:
            now = datetime.utcnow()
            day_key = now.strftime("%Y-%m-%d")

            if day_key != active_day:
                build_daily_kymograph(
                    day_key=active_day,
                    slices_dir=slices_dir,
                    kymoday_dir=kymoday_dir,
                    tool=tool,
                    images_per_day=images_per_day,
                    target_kymo_width=args.target_kymo_width,
                )
                active_day = day_key

            file_name = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
            destination = slices_dir / file_name

            try:
                capture_image(args.image_url, destination, args.timeout_seconds)
                print(f"Captured: {destination.name}")
            except Exception as exc:
                print(f"Capture failed at {now.isoformat()}Z: {exc}")

            time.sleep(max(1, int(args.interval_seconds)))
    except KeyboardInterrupt:
        print("\nStopping collector. Building current-day kymograph before exit...")
        build_daily_kymograph(
            day_key=active_day,
            slices_dir=slices_dir,
            kymoday_dir=kymoday_dir,
            tool=tool,
            images_per_day=images_per_day,
            target_kymo_width=args.target_kymo_width,
        )
        print("Stopped.")


if __name__ == "__main__":
    main()