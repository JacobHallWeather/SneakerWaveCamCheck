import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from CreateKymograph import (
    DEFAULT_SLICE_END,
    DEFAULT_SLICE_ROW_ENDS,
    DEFAULT_SLICE_ROW_STARTS,
    DEFAULT_SLICE_START,
    VisualReviewTool,
)
from image_collector import (
    build_daily_kymograph,
    infer_images_per_day,
    parse_hhmm,
)


REQUIRED_R2_ENV_VARS = (
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download one UTC date or date range from Cloudflare R2 and build kymographs. "
            "Accepts one date for a single day, or two dates for an inclusive range."
        ),
        epilog=(
            "Examples:\n"
            "  python run_kymograph.py 2-26-2026\n"
            "  python run_kymograph.py 2-25-2026 2-26-2026\n"
            "  python run_kymograph.py 2-26-2026 --bucket beachkymographs\n"
            "  python run_kymograph.py 2-26-2026 --skip-build"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "dates",
        nargs="+",
        help="One date (MM-DD-YYYY) or two dates (MM-DD-YYYY MM-DD-YYYY)",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("R2_BUCKET", "beachkymographs"),
        help="R2 bucket name (default from R2_BUCKET env, else beachkymographs)",
    )
    parser.add_argument(
        "--prefix-root",
        default=os.getenv("R2_PATH_PREFIX", "captures"),
        help="Root key prefix used in R2 (default from R2_PATH_PREFIX env, else captures)",
    )
    parser.add_argument(
        "--slices-dir",
        default="KymoSlices",
        help="Local folder to write downloaded images",
    )
    parser.add_argument(
        "--kymoday-dir",
        default="KymoDay",
        help="Local folder for output kymograph",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=300,
        help="Capture interval in seconds used for labels/images per day",
    )
    parser.add_argument(
        "--images-per-day",
        type=int,
        default=None,
        help="Override expected images/day",
    )
    parser.add_argument(
        "--capture-start",
        default="07:00",
        help="Capture window start in HH:MM PT (default 07:00)",
    )
    parser.add_argument(
        "--capture-end",
        default="16:00",
        help="Capture window end in HH:MM PT (default 16:00)",
    )
    parser.add_argument(
        "--target-kymo-width",
        type=int,
        default=1200,
        help="Target output kymograph width",
    )
    parser.add_argument(
        "--download-limit",
        type=int,
        default=None,
        help="Optional max number of images to download per day",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they already exist locally",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Only sync images; do not build a kymograph",
    )
    return parser.parse_args()


def normalize_date_input(date_value):
    normalized = date_value.strip().replace("/", "-")
    for fmt in ("%m-%d-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(normalized, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(
        f"Invalid date '{date_value}'. Use MM-DD-YYYY (example: 2-26-2026) or YYYY-MM-DD."
    )


def expand_dates(raw_dates):
    if len(raw_dates) == 1 and " " in raw_dates[0].strip():
        raw_dates = [part for part in raw_dates[0].split() if part]
    if len(raw_dates) not in (1, 2):
        raise ValueError("Provide one date or two dates (start and end).")
    return raw_dates


def normalized_day_keys(raw_dates):
    parsed = [datetime.strptime(normalize_date_input(value), "%Y-%m-%d").date() for value in expand_dates(raw_dates)]
    start_dt = parsed[0]
    end_dt = parsed[-1]
    if start_dt > end_dt:
        raise ValueError("Start date must be on or before end date.")

    out = []
    cursor = start_dt
    while cursor <= end_dt:
        out.append(cursor.strftime("%Y-%m-%d"))
        cursor += timedelta(days=1)
    return out


def load_env_file(env_path):
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def ensure_required_env():
    missing = [name for name in REQUIRED_R2_ENV_VARS if not os.getenv(name)]
    if missing:
        missing_joined = ", ".join(missing)
        raise RuntimeError(
            "Missing required R2 environment values: "
            f"{missing_joined}.\n"
            "Copy .env.example to .env and fill in your Cloudflare R2 credentials, "
            "or export them in your shell."
        )


def build_r2_client():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required. Install with: pip install boto3") from exc

    endpoint_url = f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def r2_prefix(prefix_root, day_key):
    dt = datetime.strptime(day_key, "%Y-%m-%d")
    clean_root = prefix_root.strip("/")
    return f"{clean_root}/{dt.strftime('%Y/%m/%d')}/"


def local_filename_for_key(day_key, key):
    file_name = Path(key).name
    if not file_name:
        return None
    lowered = file_name.lower()
    if not lowered.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")):
        return None
    if file_name.startswith(f"{day_key}_"):
        return file_name
    return f"{day_key}_{file_name}"


def download_day_images(client, bucket, prefix, day_key, slices_dir, download_limit=None, overwrite=False):
    slices_dir.mkdir(parents=True, exist_ok=True)

    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    downloaded = 0
    skipped_existing = 0
    scanned = 0

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if not key:
                continue

            scanned += 1
            local_name = local_filename_for_key(day_key, key)
            if local_name is None:
                continue

            destination = slices_dir / local_name
            if destination.exists() and not overwrite:
                skipped_existing += 1
                continue

            client.download_file(bucket, key, str(destination))
            downloaded += 1
            print(f"Downloaded: {key} -> {destination.name}")

            if download_limit is not None and downloaded >= max(1, int(download_limit)):
                return scanned, downloaded, skipped_existing

    return scanned, downloaded, skipped_existing


def main():
    args = parse_args()

    try:
        day_keys = normalized_day_keys(args.dates)
        load_env_file(Path(__file__).with_name(".env"))
        ensure_required_env()

        window_start = parse_hhmm(args.capture_start)
        window_end = parse_hhmm(args.capture_end)
        if window_start >= window_end:
            raise ValueError("capture-start must be earlier than capture-end (same day window).")
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    client = build_r2_client()
    slices_dir = Path(args.slices_dir)
    kymoday_dir = Path(args.kymoday_dir)
    kymoday_dir.mkdir(parents=True, exist_ok=True)

    images_per_day = infer_images_per_day(
        interval_seconds=max(1, int(args.interval_seconds)),
        explicit_value=args.images_per_day,
        start_time=window_start,
        end_time=window_end,
    )

    row_starts = VisualReviewTool._parse_point_list(DEFAULT_SLICE_ROW_STARTS)
    row_ends = VisualReviewTool._parse_point_list(DEFAULT_SLICE_ROW_ENDS)
    manual_row_lines = (
        list(zip(row_starts, row_ends)) if row_starts and row_ends and len(row_starts) == len(row_ends) else None
    )

    tool = VisualReviewTool(
        slice_start=VisualReviewTool._parse_point(DEFAULT_SLICE_START),
        slice_end=VisualReviewTool._parse_point(DEFAULT_SLICE_END),
        slice_row_lines=manual_row_lines,
        slice_rows=(len(manual_row_lines) if manual_row_lines else 3),
        slice_thickness_px=15,
        header_photo_max_height=260,
        header_photo_width_ratio=1.0,
    )

    print(f"R2 bucket: {args.bucket}")
    print(f"Local slices dir: {slices_dir}")
    print(f"Days to process: {len(day_keys)}")

    total_scanned = 0
    total_downloaded = 0
    total_skipped_existing = 0
    built_count = 0

    for day_key in day_keys:
        prefix = r2_prefix(args.prefix_root, day_key)
        print(f"\nProcessing {day_key} | prefix={prefix}")

        scanned, downloaded, skipped_existing = download_day_images(
            client=client,
            bucket=args.bucket,
            prefix=prefix,
            day_key=day_key,
            slices_dir=slices_dir,
            download_limit=args.download_limit,
            overwrite=args.overwrite,
        )

        total_scanned += scanned
        total_downloaded += downloaded
        total_skipped_existing += skipped_existing

        print(
            f"Day sync summary ({day_key}): "
            f"scanned={scanned}, downloaded={downloaded}, skipped_existing={skipped_existing}"
        )

        if args.skip_build:
            continue

        output = build_daily_kymograph(
            day_key=day_key,
            slices_dir=slices_dir,
            kymoday_dir=kymoday_dir,
            tool=tool,
            images_per_day=images_per_day,
            target_kymo_width=args.target_kymo_width,
            slice_preview_path=None,
            preview_show_clip_guides=False,
        )
        if output is not None:
            built_count += 1

    print(
        "\nOverall sync summary: "
        f"scanned={total_scanned}, downloaded={total_downloaded}, skipped_existing={total_skipped_existing}"
    )
    if args.skip_build:
        print("Build skipped for all days (--skip-build).")
    else:
        print(f"Kymographs built: {built_count}/{len(day_keys)}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
