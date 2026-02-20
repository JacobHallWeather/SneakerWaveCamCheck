import argparse
import glob
import os
import shutil
import subprocess
import sys

import cv2


def run_command(command):
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}")


def require_tool(name, install_hint):
    if shutil.which(name):
        return
    raise RuntimeError(f"Missing required tool '{name}'. Install it with: {install_hint}")


def download_video(url, output_dir, base_name, cookie_file=None, cookies_from_browser=None):
    output_template = os.path.join(output_dir, f"{base_name}.%(ext)s")

    for path in glob.glob(os.path.join(output_dir, f"{base_name}.*")):
        if os.path.isfile(path):
            os.remove(path)

    command = [
        "yt-dlp",
        "-f",
        "best[ext=mp4][vcodec!=none][acodec!=none]/best[ext=mp4]/best",
        "-o",
        output_template,
        "--no-playlist",
        url,
    ]

    if cookie_file:
        command.extend(["--cookies", cookie_file])

    if cookies_from_browser:
        command.extend(["--cookies-from-browser", cookies_from_browser])

    run_command(command)

    matches = [
        path
        for path in glob.glob(os.path.join(output_dir, f"{base_name}.*"))
        if os.path.isfile(path)
    ]

    if not matches:
        raise RuntimeError("Video download completed but no output file was found.")

    matches.sort()
    return matches[0]


def extract_frames(video_path, output_dir, fps, prefix, clean):
    if clean:
        for old_frame in glob.glob(os.path.join(output_dir, f"{prefix}_*.jpg")):
            os.remove(old_frame)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open downloaded video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0.0:
        source_fps = 30.0

    target_fps = max(0.1, float(fps))
    frame_step = max(1, int(round(source_fps / target_fps)))

    frame_index = 0
    saved_count = 0

    while True:
        has_frame, frame = capture.read()
        if not has_frame:
            break

        if frame_index % frame_step == 0:
            saved_count += 1
            frame_name = os.path.join(output_dir, f"{prefix}_{saved_count:05d}.jpg")
            cv2.imwrite(frame_name, frame)

        frame_index += 1

    capture.release()

    if saved_count == 0:
        raise RuntimeError("No frames were extracted from the video.")

    return saved_count


def run_wave_detector():
    command = [sys.executable, "wave_detector.py"]
    run_command(command)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a YouTube video, extract image frames to test_images, and run wave detection."
    )
    parser.add_argument("url", help="YouTube URL to process")
    parser.add_argument("--output-dir", default="test_images", help="Folder where frames are written")
    parser.add_argument("--fps", type=float, default=1.0, help="Target extracted frames per second")
    parser.add_argument("--video-name", default="youtube_source", help="Base filename for downloaded video")
    parser.add_argument("--frame-prefix", default="yt_frame", help="Filename prefix for extracted frames")
    parser.add_argument(
        "--cookies",
        default=None,
        help="Path to Netscape-format cookies.txt file for authenticated downloads",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Browser name for yt-dlp cookie extraction (example: chrome, firefox)",
    )
    parser.add_argument(
        "--keep-existing-frames",
        action="store_true",
        help="Keep existing extracted frames instead of cleaning them first",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    require_tool("yt-dlp", "python3 -m pip install -U yt-dlp")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Downloading video...")
    video_path = download_video(
        args.url,
        args.output_dir,
        args.video_name,
        cookie_file=args.cookies,
        cookies_from_browser=args.cookies_from_browser,
    )
    print(f"Downloaded video to: {video_path}")

    print("Extracting frames...")
    frame_count = extract_frames(
        video_path=video_path,
        output_dir=args.output_dir,
        fps=args.fps,
        prefix=args.frame_prefix,
        clean=not args.keep_existing_frames,
    )
    print(f"Extracted {frame_count} frame(s) to {args.output_dir}")

    print("Running wave detector...")
    run_wave_detector()
    print("Wave detection complete.")


if __name__ == "__main__":
    main()