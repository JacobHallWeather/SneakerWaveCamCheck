import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

from CreateKymograph import VisualReviewTool
from image_collector import build_daily_kymograph, infer_images_per_day


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download UTC day(s) of webcam images from Cloudflare R2 into KymoSlices and optionally build kymographs."
    )
    parser.add_argument(
        "--date",
        default=None,
        help="UTC date to sync in YYYY-MM-DD format (example: 2026-02-26)",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Range start date in YYYY-MM-DD (inclusive)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Range end date in YYYY-MM-DD (inclusive)",
    )
    parser.add_argument(
        "--last-days",
        type=int,
        default=None,
        help="Sync this many UTC days ending today (example: 7)",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("R2_BUCKET", "beachkymograph"),
        help="R2 bucket name (default from R2_BUCKET env, else beachkymograph)",
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
        help="Capture interval in seconds used to estimate images/day for build settings",
    )
    parser.add_argument(
        "--images-per-day",
        type=int,
        default=None,
        help="Override expected images/day",
    )
    parser.add_argument(
        "--target-kymo-width",
        type=int,
        default=825,
        help="Target output kymograph width",
    )
    parser.add_argument(
        "--download-limit",
        type=int,
        default=None,
        help="Optional max number of images to download",
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


def parse_day_key(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("--date must be in YYYY-MM-DD format") from exc


def date_range_from_args(args):
    mode_count = sum(
        [
            args.date is not None,
            args.last_days is not None,
            (args.start_date is not None or args.end_date is not None),
        ]
    )
    if mode_count != 1:
        raise ValueError("Choose exactly one mode: --date, --last-days, or --start-date/--end-date")

    if args.date is not None:
        day_key = parse_day_key(args.date)
        return [day_key]

    if args.last_days is not None:
        days = max(1, int(args.last_days))
        end_dt = datetime.utcnow().date()
        start_dt = end_dt - timedelta(days=days - 1)
    else:
        if args.start_date is None or args.end_date is None:
            raise ValueError("--start-date and --end-date must be provided together")
        start_dt = datetime.strptime(parse_day_key(args.start_date), "%Y-%m-%d").date()
        end_dt = datetime.strptime(parse_day_key(args.end_date), "%Y-%m-%d").date()

    if start_dt > end_dt:
        raise ValueError("start date must be <= end date")

    out = []
    cursor = start_dt
    while cursor <= end_dt:
        out.append(cursor.strftime("%Y-%m-%d"))
        cursor += timedelta(days=1)
    return out


def require_env(var_name):
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    return value


def build_r2_client():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required. Install with: pip install boto3") from exc

    account_id = require_env("R2_ACCOUNT_ID")
    access_key_id = require_env("R2_ACCESS_KEY_ID")
    secret_access_key = require_env("R2_SECRET_ACCESS_KEY")

    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
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
    day_keys = date_range_from_args(args)

    client = build_r2_client()
    slices_dir = Path(args.slices_dir)
    kymoday_dir = Path(args.kymoday_dir)
    kymoday_dir.mkdir(parents=True, exist_ok=True)

    print(f"R2 bucket: {args.bucket}")
    print(f"Local slices dir: {slices_dir}")
    print(f"Days to process: {len(day_keys)}")

    total_scanned = 0
    total_downloaded = 0
    total_skipped_existing = 0
    built_count = 0

    images_per_day = infer_images_per_day(
        interval_seconds=max(1, int(args.interval_seconds)),
        explicit_value=args.images_per_day,
        start_time=None,
        end_time=None,
    )
    tool = VisualReviewTool()

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


if __name__ == "__main__":
    main()
