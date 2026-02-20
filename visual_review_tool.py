import argparse
from pathlib import Path

import cv2
import numpy as np


class VisualReviewTool:
    """Create kymograph views from webcam image sequences."""

    def __init__(
        self,
        slice_angle_deg=0.0,
        slice_center_x_ratio=0.5,
        slice_center_y_ratio=0.5,
        slice_start=None,
        slice_end=None,
    ):
        self.slice_angle_deg = float(slice_angle_deg)
        self.slice_center_x_ratio = float(slice_center_x_ratio)
        self.slice_center_y_ratio = float(slice_center_y_ratio)
        self.slice_start = slice_start
        self.slice_end = slice_end

    @staticmethod
    def _collect_images(image_folder, max_images=None):
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        images = sorted(
            [
                p
                for p in Path(image_folder).iterdir()
                if p.is_file() and p.suffix.lower() in extensions and not p.stem.endswith("_annotated")
            ]
        )
        if max_images is not None:
            images = images[: max(1, int(max_images))]
        return images

    @staticmethod
    def _normalize_paths(image_paths, max_images=None):
        files = [Path(p) for p in image_paths]
        files = [p for p in files if p.is_file()]
        files = sorted(files)
        if max_images is not None:
            files = files[: max(1, int(max_images))]
        return files

    @staticmethod
    def _parse_point(value):
        if value is None:
            return None
        pieces = [part.strip() for part in value.split(",")]
        if len(pieces) != 2:
            raise ValueError(f"Invalid point format '{value}'. Expected x,y")
        return int(float(pieces[0])), int(float(pieces[1]))

    @staticmethod
    def _line_segment_through_image(center, direction, width, height):
        x0, y0 = float(center[0]), float(center[1])
        dx, dy = float(direction[0]), float(direction[1])

        eps = 1e-9
        t_min = -float("inf")
        t_max = float("inf")

        if abs(dx) < eps:
            if x0 < 0 or x0 > width - 1:
                return None
        else:
            tx1 = (0.0 - x0) / dx
            tx2 = (float(width - 1) - x0) / dx
            t_min = max(t_min, min(tx1, tx2))
            t_max = min(t_max, max(tx1, tx2))

        if abs(dy) < eps:
            if y0 < 0 or y0 > height - 1:
                return None
        else:
            ty1 = (0.0 - y0) / dy
            ty2 = (float(height - 1) - y0) / dy
            t_min = max(t_min, min(ty1, ty2))
            t_max = min(t_max, max(ty1, ty2))

        if t_max < t_min:
            return None

        p1 = np.array([x0 + t_min * dx, y0 + t_min * dy], dtype=np.float32)
        p2 = np.array([x0 + t_max * dx, y0 + t_max * dy], dtype=np.float32)
        return p1, p2

    def _slice_points(self, width, height):
        if self.slice_start is not None and self.slice_end is not None:
            start = np.array(self.slice_start, dtype=np.float32)
            end = np.array(self.slice_end, dtype=np.float32)
        else:
            center = np.array(
                [
                    float(np.clip(self.slice_center_x_ratio, 0.0, 1.0)) * (width - 1),
                    float(np.clip(self.slice_center_y_ratio, 0.0, 1.0)) * (height - 1),
                ],
                dtype=np.float32,
            )
            angle_rad = np.deg2rad(self.slice_angle_deg)
            direction = np.array([np.sin(angle_rad), np.cos(angle_rad)], dtype=np.float32)
            segment = self._line_segment_through_image(center, direction, width, height)
            if segment is None:
                raise RuntimeError("Slice line does not intersect image bounds.")
            start, end = segment

        length = int(max(2, round(float(np.linalg.norm(end - start)))))
        xs = np.linspace(start[0], end[0], length)
        ys = np.linspace(start[1], end[1], length)

        xi = np.clip(np.round(xs).astype(np.int32), 0, width - 1)
        yi = np.clip(np.round(ys).astype(np.int32), 0, height - 1)
        return xi, yi, (int(round(start[0])), int(round(start[1]))), (int(round(end[0])), int(round(end[1])))

    @staticmethod
    def _slice_width(images_per_day, target_kymo_width):
        per_day = max(1, int(images_per_day))
        target = max(200, int(target_kymo_width))
        return max(1, int(round(target / per_day)))

    def _create_kymograph_from_paths(
        self, image_paths, output_path="kymograph.jpg", images_per_day=24, target_kymo_width=2400
    ):
        images = self._normalize_paths(image_paths)
        if not images:
            raise FileNotFoundError("No supported images found for kymograph generation")

        first = cv2.imread(str(images[0]))
        if first is None:
            raise RuntimeError(f"Could not load image: {images[0]}")

        height, width = first.shape[:2]
        xs, ys, start_pt, end_pt = self._slice_points(width, height)
        strip_width = self._slice_width(images_per_day, target_kymo_width)

        strips = []
        processed = []
        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                continue
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

            profile = image[ys, xs].reshape(len(xs), 1, 3)
            if strip_width > 1:
                profile = cv2.resize(profile, (strip_width, len(xs)), interpolation=cv2.INTER_LINEAR)

            strips.append(profile)
            processed.append(path.name)

        if not strips:
            raise RuntimeError("No valid images could be processed for kymograph generation.")

        kymograph = np.hstack(strips)
        cv2.imwrite(output_path, kymograph)

        print(f"✓ Kymograph saved: {output_path}")
        print(f"  Images used: {len(processed)}")
        print(f"  Slice width per image: {strip_width} px (images/day={images_per_day})")
        print(f"  Slice endpoints: {start_pt} -> {end_pt}")
        return output_path, (start_pt, end_pt)

    def create_kymograph(
        self,
        image_folder,
        output_path="kymograph.jpg",
        images_per_day=24,
        target_kymo_width=2400,
        max_images=None,
    ):
        images = self._collect_images(image_folder, max_images=max_images)
        if not images:
            raise FileNotFoundError(f"No supported images found in {image_folder}")
        return self._create_kymograph_from_paths(
            image_paths=images,
            output_path=output_path,
            images_per_day=images_per_day,
            target_kymo_width=target_kymo_width,
        )

    def create_kymograph_from_paths(
        self,
        image_paths,
        output_path="kymograph.jpg",
        images_per_day=24,
        target_kymo_width=2400,
        max_images=None,
    ):
        images = self._normalize_paths(image_paths, max_images=max_images)
        if not images:
            raise FileNotFoundError("No supported images found for kymograph generation")
        return self._create_kymograph_from_paths(
            image_paths=images,
            output_path=output_path,
            images_per_day=images_per_day,
            target_kymo_width=target_kymo_width,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate kymograph visualizations.")
    parser.add_argument("--image-folder", default="KymoSlices", help="Input folder containing images")
    parser.add_argument("--output-path", default="KymoDay/kymograph.jpg", help="Output kymograph image path")
    parser.add_argument("--input-pattern", default=None, help="Optional glob pattern to select specific images")

    parser.add_argument(
        "--images-per-day",
        type=int,
        default=24,
        help="Expected captures per day (24 for hourly, 48 for every 30 min)",
    )
    parser.add_argument(
        "--target-kymo-width",
        type=int,
        default=2400,
        help="Target kymograph width in pixels; slice width per image scales from this and images-per-day",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on images used")

    parser.add_argument(
        "--slice-angle-deg",
        type=float,
        default=0.0,
        help="Slice angle in degrees from vertical (0=center vertical slice)",
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
        "--slice-start",
        default=None,
        help="Optional explicit slice start point as x,y (pixels)",
    )
    parser.add_argument(
        "--slice-end",
        default=None,
        help="Optional explicit slice end point as x,y (pixels)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    tool = VisualReviewTool(
        slice_angle_deg=args.slice_angle_deg,
        slice_center_x_ratio=args.slice_center_x_ratio,
        slice_center_y_ratio=args.slice_center_y_ratio,
        slice_start=VisualReviewTool._parse_point(args.slice_start),
        slice_end=VisualReviewTool._parse_point(args.slice_end),
    )

    output_parent = Path(args.output_path).parent
    if output_parent:
        output_parent.mkdir(parents=True, exist_ok=True)

    if args.input_pattern:
        selected = sorted(Path(args.image_folder).glob(args.input_pattern))
        tool.create_kymograph_from_paths(
            image_paths=selected,
            output_path=args.output_path,
            images_per_day=args.images_per_day,
            target_kymo_width=args.target_kymo_width,
            max_images=args.max_images,
        )
    else:
        tool.create_kymograph(
            image_folder=args.image_folder,
            output_path=args.output_path,
            images_per_day=args.images_per_day,
            target_kymo_width=args.target_kymo_width,
            max_images=args.max_images,
        )


if __name__ == "__main__":
    main()