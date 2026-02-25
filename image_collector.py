import argparse
import time
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from CreateKymograph import (
    DEFAULT_SLICE_END,
    DEFAULT_SLICE_ROW_ENDS,
    DEFAULT_SLICE_ROW_STARTS,
    DEFAULT_SLICE_START,
    VisualReviewTool,
)


PACIFIC_TZ = ZoneInfo("America/Los_Angeles")


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
        default=900,
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
        default=1440,
        help="Target daily kymograph width in pixels",
    )
    parser.add_argument(
        "--slice-preview-path",
        default=None,
        help="Output path for full-image slice overlay preview",
    )
    parser.add_argument(
        "--slice-preview-show-clip-guides",
        action="store_true",
        help="Show red top/bottom clip guide lines on preview image",
    )
    parser.add_argument(
        "--slice-vertical-stretch",
        type=float,
        default=1.0,
        help="Scale factor for kymograph height (use >1.0 to enlarge a short slice)",
    )
    parser.add_argument(
        "--slice-thickness-px",
        type=int,
        default=15,
        help="Source slice thickness in pixels sampled perpendicular to the slice line",
    )
    parser.add_argument(
        "--header-photo-max-height",
        type=int,
        default=260,
        help="Maximum height in pixels for the top header beach photo",
    )
    parser.add_argument(
        "--header-photo-width-ratio",
        type=float,
        default=1.0,
        help="Header beach photo width as a fraction of final output width (0.2-1.0)",
    )
    parser.add_argument(
        "--slice-rows",
        type=int,
        default=3,
        help="Number of parallel slice rows to stack vertically in the output",
    )
    parser.add_argument(
        "--slice-row-spacing-px",
        type=float,
        default=None,
        help="Optional spacing between adjacent slice rows in pixels (auto if omitted)",
    )
    parser.add_argument(
        "--slice-row-span-ratio",
        type=float,
        default=0.35,
        help="When auto spacing is used, total row span as fraction of min(image width,height)",
    )
    parser.add_argument(
        "--slice-row-starts",
        default=DEFAULT_SLICE_ROW_STARTS,
        help="Manual semicolon-separated start points for rows (example: 100,200;120,220;140,240)",
    )
    parser.add_argument(
        "--slice-row-ends",
        default=DEFAULT_SLICE_ROW_ENDS,
        help="Manual semicolon-separated end points for rows (example: 300,400;320,420;340,440)",
    )
    parser.add_argument(
        "--normalize-slice-brightness",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize each slice brightness to reduce cloud/daylight intensity swings (off by default)",
    )
    parser.add_argument(
        "--slice-angle-deg",
        type=float,
        default=0.0,
        help="Slice angle in degrees from vertical (default center vertical)",
    )
    parser.add_argument(
        "--slice-angle-horizontal-deg",
        type=float,
        default=None,
        help="Slice angle in degrees from horizontal (example: -30)",
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
    parser.add_argument(
        "--slice-top-ratio",
        type=float,
        default=0.0,
        help="Vertical clip top bound ratio for slice points (0.0 top, 1.0 bottom)",
    )
    parser.add_argument(
        "--slice-bottom-ratio",
        type=float,
        default=1.0,
        help="Vertical clip bottom bound ratio for slice points (0.0 top, 1.0 bottom)",
    )
    parser.add_argument("--slice-start", default=DEFAULT_SLICE_START, help="Optional explicit slice start point x,y")
    parser.add_argument("--slice-end", default=DEFAULT_SLICE_END, help="Optional explicit slice end point x,y")
    parser.add_argument("--timeout-seconds", type=int, default=20, help="HTTP request timeout")
    parser.add_argument(
        "--capture-start",
        default="07:00",
        help="Local start time for captures in HH:MM (default 09:00)",
    )
    parser.add_argument(
        "--capture-end",
        default="16:00",
        help="Local end time for captures in HH:MM (default 16:00)",
    )
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


def parse_hhmm(value):
    try:
        parsed = datetime.strptime(value, "%H:%M")
        return dt_time(hour=parsed.hour, minute=parsed.minute)
    except ValueError as exc:
        raise ValueError(f"Invalid time '{value}'. Expected HH:MM in 24-hour format.") from exc


def is_within_window(now_local, start_time, end_time):
    current_minutes = now_local.hour * 60 + now_local.minute
    start_minutes = start_time.hour * 60 + start_time.minute
    end_minutes = end_time.hour * 60 + end_time.minute
    return start_minutes <= current_minutes <= end_minutes


def seconds_until_next_window(now_local, start_time, end_time):
    today_start = now_local.replace(
        hour=start_time.hour,
        minute=start_time.minute,
        second=0,
        microsecond=0,
    )
    today_end = now_local.replace(
        hour=end_time.hour,
        minute=end_time.minute,
        second=0,
        microsecond=0,
    )

    if now_local < today_start:
        target = today_start
    elif now_local > today_end:
        target = today_start + timedelta(days=1)
    else:
        return 0

    return max(1, int((target - now_local).total_seconds()))


def build_daily_kymograph(
    day_key,
    slices_dir,
    kymoday_dir,
    tool,
    images_per_day,
    target_kymo_width,
    slice_preview_path=None,
    preview_show_clip_guides=False,
):
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
        preview_path=slice_preview_path,
        preview_show_clip_guides=preview_show_clip_guides,
    )
    print(f"Daily kymograph written: {output_path}")
    return output_path


def main():
    args = parse_args()

    manual_row_lines = None
    if args.slice_row_starts or args.slice_row_ends:
        if not args.slice_row_starts or not args.slice_row_ends:
            raise ValueError("Provide both --slice-row-starts and --slice-row-ends when defining manual row positions.")
        starts = VisualReviewTool._parse_point_list(args.slice_row_starts)
        ends = VisualReviewTool._parse_point_list(args.slice_row_ends)
        if not starts or not ends:
            raise ValueError("Manual row starts/ends were provided but no valid points were parsed.")
        if len(starts) != len(ends):
            raise ValueError("--slice-row-starts and --slice-row-ends must contain the same number of points.")
        manual_row_lines = list(zip(starts, ends))

    window_start = parse_hhmm(args.capture_start)
    window_end = parse_hhmm(args.capture_end)
    if window_start >= window_end:
        raise ValueError("capture-start must be earlier than capture-end (same day window).")

    slices_dir = Path(args.slices_dir)
    kymoday_dir = Path(args.kymoday_dir)
    slices_dir.mkdir(parents=True, exist_ok=True)
    kymoday_dir.mkdir(parents=True, exist_ok=True)

    images_per_day = infer_images_per_day(args.interval_seconds, args.images_per_day)
    tool = VisualReviewTool(
        slice_angle_deg=args.slice_angle_deg,
        slice_angle_from_horizontal_deg=args.slice_angle_horizontal_deg,
        slice_center_x_ratio=args.slice_center_x_ratio,
        slice_center_y_ratio=args.slice_center_y_ratio,
        slice_top_ratio=args.slice_top_ratio,
        slice_bottom_ratio=args.slice_bottom_ratio,
        slice_start=VisualReviewTool._parse_point(args.slice_start),
        slice_end=VisualReviewTool._parse_point(args.slice_end),
        slice_vertical_stretch=args.slice_vertical_stretch,
        normalize_slice_brightness=args.normalize_slice_brightness,
        slice_thickness_px=args.slice_thickness_px,
        slice_rows=args.slice_rows,
        slice_row_spacing_px=args.slice_row_spacing_px,
        slice_row_span_ratio=args.slice_row_span_ratio,
        slice_row_lines=manual_row_lines,
        header_photo_max_height=args.header_photo_max_height,
        header_photo_width_ratio=args.header_photo_width_ratio,
        capture_window_label=f"{window_start.strftime('%H:%M')}-{window_end.strftime('%H:%M')} PT",
    )

    print(f"Collecting to: {slices_dir}")
    print(f"Daily outputs: {kymoday_dir}")
    print(f"Interval: {args.interval_seconds}s | images/day: {images_per_day}")
    display_slice_width = VisualReviewTool._slice_width(images_per_day, args.target_kymo_width)
    if manual_row_lines is not None:
        row_config = f"manual row lines={len(manual_row_lines)}"
    else:
        row_config = f"slice rows={args.slice_rows}"
    print(
        "Kymograph settings: "
        f"target width={args.target_kymo_width}px | display slice width={display_slice_width}px | "
        f"source slice thickness={args.slice_thickness_px}px | {row_config}"
    )
    print(f"Capture window (Pacific time): {window_start.strftime('%H:%M')} to {window_end.strftime('%H:%M')}")

    active_day = datetime.now(PACIFIC_TZ).strftime("%Y-%m-%d")
    last_built_day = None

    try:
        while True:
            now = datetime.now(PACIFIC_TZ)
            day_key = now.strftime("%Y-%m-%d")

            if not is_within_window(now, window_start, window_end):
                if now.time() > window_end and active_day == day_key and last_built_day != active_day:
                    build_daily_kymograph(
                        day_key=active_day,
                        slices_dir=slices_dir,
                        kymoday_dir=kymoday_dir,
                        tool=tool,
                        images_per_day=images_per_day,
                        target_kymo_width=args.target_kymo_width,
                        slice_preview_path=args.slice_preview_path,
                        preview_show_clip_guides=args.slice_preview_show_clip_guides,
                    )
                    last_built_day = active_day

                if day_key != active_day:
                    active_day = day_key

                sleep_seconds = seconds_until_next_window(now, window_start, window_end)
                next_start = now + timedelta(seconds=sleep_seconds)
                print(f"Outside capture window. Sleeping until {next_start.strftime('%Y-%m-%d %H:%M:%S')}.")
                time.sleep(sleep_seconds)
                continue

            if day_key != active_day:
                if last_built_day != active_day:
                    build_daily_kymograph(
                        day_key=active_day,
                        slices_dir=slices_dir,
                        kymoday_dir=kymoday_dir,
                        tool=tool,
                        images_per_day=images_per_day,
                        target_kymo_width=args.target_kymo_width,
                        slice_preview_path=args.slice_preview_path,
                        preview_show_clip_guides=args.slice_preview_show_clip_guides,
                    )
                    last_built_day = active_day
                active_day = day_key

            file_name = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
            destination = slices_dir / file_name

            try:
                capture_image(args.image_url, destination, args.timeout_seconds)
                print(f"Captured: {destination.name}")
            except Exception as exc:
                print(f"Capture failed at {now.isoformat()}: {exc}")

            time.sleep(max(1, int(args.interval_seconds)))
    except KeyboardInterrupt:
        print("\nStopping collector. Building current-day kymograph before exit...")
        if last_built_day != active_day:
            build_daily_kymograph(
                day_key=active_day,
                slices_dir=slices_dir,
                kymoday_dir=kymoday_dir,
                tool=tool,
                images_per_day=images_per_day,
                target_kymo_width=args.target_kymo_width,
                slice_preview_path=args.slice_preview_path,
                preview_show_clip_guides=args.slice_preview_show_clip_guides,
            )
        print("Stopped.")


if __name__ == "__main__":
    main()