import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SLICE_START = "1080,300"
DEFAULT_SLICE_END = "1260,500"


class VisualReviewTool:
    """Create kymograph views from webcam image sequences."""

    def __init__(
        self,
        slice_angle_deg=0.0,
        slice_angle_from_horizontal_deg=None,
        slice_center_x_ratio=0.5,
        slice_center_y_ratio=0.5,
        slice_top_ratio=0.0,
        slice_bottom_ratio=1.0,
        slice_start=None,
        slice_end=None,
        slice_vertical_stretch=1.0,
        normalize_slice_brightness=False,
        slice_thickness_px=15,
    ):
        self.slice_angle_deg = float(slice_angle_deg)
        self.slice_angle_from_horizontal_deg = (
            None if slice_angle_from_horizontal_deg is None else float(slice_angle_from_horizontal_deg)
        )
        self.slice_center_x_ratio = float(slice_center_x_ratio)
        self.slice_center_y_ratio = float(slice_center_y_ratio)
        self.slice_top_ratio = float(np.clip(slice_top_ratio, 0.0, 1.0))
        self.slice_bottom_ratio = float(np.clip(slice_bottom_ratio, 0.0, 1.0))
        self.slice_start = slice_start
        self.slice_end = slice_end
        self.slice_vertical_stretch = max(1.0, float(slice_vertical_stretch))
        self.normalize_slice_brightness = bool(normalize_slice_brightness)
        self.slice_thickness_px = max(1, int(slice_thickness_px))

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
        top_y = int(round(min(self.slice_top_ratio, self.slice_bottom_ratio) * (height - 1)))
        bottom_y = int(round(max(self.slice_top_ratio, self.slice_bottom_ratio) * (height - 1)))

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
            if self.slice_angle_from_horizontal_deg is not None:
                angle_rad = np.deg2rad(self.slice_angle_from_horizontal_deg)
                direction = np.array([np.cos(angle_rad), -np.sin(angle_rad)], dtype=np.float32)
            else:
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

        vertical_mask = (yi >= top_y) & (yi <= bottom_y)
        xi = xi[vertical_mask]
        yi = yi[vertical_mask]
        if xi.size < 2:
            raise RuntimeError(
                "Slice clipping removed all points. Adjust --slice-top-ratio/--slice-bottom-ratio or slice angle/position."
            )

        return xi, yi, (int(round(start[0])), int(round(start[1]))), (int(round(end[0])), int(round(end[1])))

    @staticmethod
    def _save_slice_preview(
        image,
        preview_path,
        xs,
        ys,
        start_pt,
        end_pt,
        top_ratio,
        bottom_ratio,
        show_clip_guides=False,
    ):
        overlay = image.copy()
        height, width = overlay.shape[:2]

        for i in range(1, len(xs)):
            cv2.line(
                overlay,
                (int(xs[i - 1]), int(ys[i - 1])),
                (int(xs[i]), int(ys[i])),
                (0, 255, 255),
                2,
            )

        if show_clip_guides:
            top_y = int(round(min(top_ratio, bottom_ratio) * (height - 1)))
            bottom_y = int(round(max(top_ratio, bottom_ratio) * (height - 1)))
            cv2.line(overlay, (0, top_y), (width - 1, top_y), (0, 0, 255), 1)
            cv2.line(overlay, (0, bottom_y), (width - 1, bottom_y), (0, 0, 255), 1)

            cv2.putText(
                overlay,
                "Yellow: sampled slice | Red: top/bottom clip",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        preview_parent = Path(preview_path).parent
        if str(preview_parent) != "":
            preview_parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_path), overlay)

    @staticmethod
    def _slice_width(images_per_day, target_kymo_width):
        per_day = max(1, int(images_per_day))
        target = max(200, int(target_kymo_width))
        return max(1, int(round(target / per_day)))

    @staticmethod
    def _match_slice_brightness(strips):
        if not strips:
            return strips

        luminance_means = []
        for strip in strips:
            strip_float = strip.astype(np.float32)
            y_channel = (
                0.114 * strip_float[:, :, 0]
                + 0.587 * strip_float[:, :, 1]
                + 0.299 * strip_float[:, :, 2]
            )
            luminance_means.append(float(np.mean(y_channel)))

        target_luminance = float(np.median(np.array(luminance_means, dtype=np.float32)))
        normalized = []
        for strip, mean_luminance in zip(strips, luminance_means):
            scale = target_luminance / max(1e-6, mean_luminance)
            scale = float(np.clip(scale, 0.65, 1.45))
            adjusted = np.clip(strip.astype(np.float32) * scale, 0, 255).astype(np.uint8)
            normalized.append(adjusted)

        return normalized

    @staticmethod
    def _extract_profile(image, xs, ys, start_pt, end_pt, thickness_px):
        h, w = image.shape[:2]
        thickness = max(1, int(thickness_px))

        if thickness == 1:
            return image[ys, xs].reshape(len(xs), 1, 3)

        dx = float(end_pt[0] - start_pt[0])
        dy = float(end_pt[1] - start_pt[1])
        length = float(np.hypot(dx, dy))
        if length < 1e-6:
            return image[ys, xs].reshape(len(xs), 1, 3)

        nx = -dy / length
        ny = dx / length
        offsets = np.linspace(-(thickness - 1) / 2.0, (thickness - 1) / 2.0, thickness, dtype=np.float32)

        xi_stack = []
        yi_stack = []
        xs_float = xs.astype(np.float32)
        ys_float = ys.astype(np.float32)
        for offset in offsets:
            xi = np.clip(np.round(xs_float + nx * offset).astype(np.int32), 0, w - 1)
            yi = np.clip(np.round(ys_float + ny * offset).astype(np.int32), 0, h - 1)
            xi_stack.append(xi)
            yi_stack.append(yi)

        xi_arr = np.stack(xi_stack, axis=0)
        yi_arr = np.stack(yi_stack, axis=0)
        samples = image[yi_arr, xi_arr]
        averaged = np.mean(samples.astype(np.float32), axis=0).astype(np.uint8)
        return averaged.reshape(len(xs), 1, 3)

    def _create_kymograph_from_paths(
        self,
        image_paths,
        output_path="kymograph.jpg",
        images_per_day=24,
        target_kymo_width=2400,
        preview_path=None,
        preview_show_clip_guides=False,
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

        if preview_path is not None:
            self._save_slice_preview(
                image=first,
                preview_path=preview_path,
                xs=xs,
                ys=ys,
                start_pt=start_pt,
                end_pt=end_pt,
                top_ratio=self.slice_top_ratio,
                bottom_ratio=self.slice_bottom_ratio,
                show_clip_guides=preview_show_clip_guides,
            )

        strips = []
        processed = []
        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                continue
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

            profile = self._extract_profile(
                image=image,
                xs=xs,
                ys=ys,
                start_pt=start_pt,
                end_pt=end_pt,
                thickness_px=self.slice_thickness_px,
            )
            if strip_width > 1:
                profile = cv2.resize(profile, (strip_width, len(xs)), interpolation=cv2.INTER_LINEAR)

            strips.append(profile)
            processed.append(path.name)

        if not strips:
            raise RuntimeError("No valid images could be processed for kymograph generation.")

        if self.normalize_slice_brightness:
            strips = self._match_slice_brightness(strips)

        kymograph = np.hstack(strips)
        if self.slice_vertical_stretch > 1.0:
            stretched_height = int(round(kymograph.shape[0] * self.slice_vertical_stretch))
            kymograph = cv2.resize(kymograph, (kymograph.shape[1], max(2, stretched_height)), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(output_path, kymograph)

        print(f"✓ Kymograph saved: {output_path}")
        print(f"  Images used: {len(processed)}")
        print(f"  Slice width per image: {strip_width} px (images/day={images_per_day})")
        print(f"  Source slice thickness: {self.slice_thickness_px} px")
        print(f"  Vertical stretch: {self.slice_vertical_stretch:.2f}x")
        print(f"  Brightness normalized: {'yes' if self.normalize_slice_brightness else 'no'}")
        print(f"  Slice endpoints: {start_pt} -> {end_pt}")
        return output_path, (start_pt, end_pt)

    def create_kymograph(
        self,
        image_folder,
        output_path="kymograph.jpg",
        images_per_day=24,
        target_kymo_width=2400,
        max_images=None,
        preview_path=None,
        preview_show_clip_guides=False,
    ):
        images = self._collect_images(image_folder, max_images=max_images)
        if not images:
            raise FileNotFoundError(f"No supported images found in {image_folder}")
        return self._create_kymograph_from_paths(
            image_paths=images,
            output_path=output_path,
            images_per_day=images_per_day,
            target_kymo_width=target_kymo_width,
            preview_path=preview_path,
            preview_show_clip_guides=preview_show_clip_guides,
        )

    def create_kymograph_from_paths(
        self,
        image_paths,
        output_path="kymograph.jpg",
        images_per_day=24,
        target_kymo_width=2400,
        max_images=None,
        preview_path=None,
        preview_show_clip_guides=False,
    ):
        images = self._normalize_paths(image_paths, max_images=max_images)
        if not images:
            raise FileNotFoundError("No supported images found for kymograph generation")
        return self._create_kymograph_from_paths(
            image_paths=images,
            output_path=output_path,
            images_per_day=images_per_day,
            target_kymo_width=target_kymo_width,
            preview_path=preview_path,
            preview_show_clip_guides=preview_show_clip_guides,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate kymograph visualizations.")
    parser.add_argument("--image-folder", default="KymoSlices", help="Input folder containing images")
    parser.add_argument("--output-path", default="KymoDay/kymograph.jpg", help="Output kymograph image path")
    parser.add_argument(
        "--slice-preview-path",
        default=None,
        help="Optional output path for full-image slice overlay preview",
    )
    parser.add_argument(
        "--slice-preview-show-clip-guides",
        action="store_true",
        help="Show red top/bottom clip guide lines on preview image",
    )
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
        default=1440,
        help="Target kymograph width in pixels; slice width per image scales from this and images-per-day",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on images used")
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
        "--normalize-slice-brightness",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize each slice brightness to reduce cloud/daylight intensity swings (off by default)",
    )

    parser.add_argument(
        "--slice-angle-deg",
        type=float,
        default=0.0,
        help="Slice angle in degrees from vertical (0=center vertical slice)",
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
    parser.add_argument(
        "--slice-start",
        default=DEFAULT_SLICE_START,
        help="Optional explicit slice start point as x,y (pixels)",
    )
    parser.add_argument(
        "--slice-end",
        default=DEFAULT_SLICE_END,
        help="Optional explicit slice end point as x,y (pixels)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
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
            preview_path=args.slice_preview_path,
            preview_show_clip_guides=args.slice_preview_show_clip_guides,
        )
    else:
        tool.create_kymograph(
            image_folder=args.image_folder,
            output_path=args.output_path,
            images_per_day=args.images_per_day,
            target_kymo_width=args.target_kymo_width,
            max_images=args.max_images,
            preview_path=args.slice_preview_path,
            preview_show_clip_guides=args.slice_preview_show_clip_guides,
        )


if __name__ == "__main__":
    main()