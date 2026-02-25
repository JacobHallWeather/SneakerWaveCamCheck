import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SLICE_START = "1080,300"
DEFAULT_SLICE_END = "1260,500"
DEFAULT_SLICE_ROW_STARTS = "980,350;1080,300;1180,300"
DEFAULT_SLICE_ROW_ENDS = "1080,450;1180,450;1280,450"


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
        slice_rows=3,
        slice_row_spacing_px=None,
        slice_row_span_ratio=0.35,
        slice_row_lines=None,
        header_photo_max_height=500,
        header_photo_width_ratio=1,
        capture_window_label=None,
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
        self.slice_rows = max(1, int(slice_rows))
        self.slice_row_spacing_px = (
            None if slice_row_spacing_px is None else max(1.0, float(slice_row_spacing_px))
        )
        self.slice_row_span_ratio = max(0.0, float(slice_row_span_ratio))
        self.slice_row_lines = slice_row_lines
        self.header_photo_max_height = max(80, int(header_photo_max_height))
        self.header_photo_width_ratio = float(np.clip(header_photo_width_ratio, 0.2, 1.0))
        self.capture_window_label = capture_window_label

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
    def _parse_point_list(value):
        if value is None:
            return None
        parts = [part.strip() for part in value.split(";") if part.strip()]
        return [VisualReviewTool._parse_point(part) for part in parts]

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

    def _line_to_slice_points(self, width, height, start, end):
        top_y = int(round(min(self.slice_top_ratio, self.slice_bottom_ratio) * (height - 1)))
        bottom_y = int(round(max(self.slice_top_ratio, self.slice_bottom_ratio) * (height - 1)))

        start = np.array(start, dtype=np.float32)
        end = np.array(end, dtype=np.float32)

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

        return self._line_to_slice_points(width=width, height=height, start=start, end=end)

    def _slice_rows(self, width, height):
        if self.slice_row_lines is not None:
            rows = []
            for start_pt, end_pt in self.slice_row_lines:
                rows.append(
                    self._line_to_slice_points(
                        width=width,
                        height=height,
                        start=start_pt,
                        end=end_pt,
                    )
                )
            if not rows:
                raise RuntimeError("No valid manual slice rows were provided.")
            return rows

        base_xs, base_ys, base_start, base_end = self._slice_points(width, height)

        if self.slice_rows == 1:
            return [(base_xs, base_ys, base_start, base_end)]

        dx = float(base_end[0] - base_start[0])
        dy = float(base_end[1] - base_start[1])
        length = float(np.hypot(dx, dy))
        if length < 1e-6:
            return [(base_xs, base_ys, base_start, base_end)]

        nx = -dy / length
        ny = dx / length

        if self.slice_row_spacing_px is not None:
            spacing_px = float(self.slice_row_spacing_px)
        else:
            total_span = float(min(width, height)) * self.slice_row_span_ratio
            spacing_px = total_span / max(1, self.slice_rows - 1)

        center_index = 0.5 * (self.slice_rows - 1)
        offsets = [(row_idx - center_index) * spacing_px for row_idx in range(self.slice_rows)]

        rows = []
        for offset in offsets:

            xi = np.clip(np.round(base_xs.astype(np.float32) + nx * offset).astype(np.int32), 0, width - 1)
            yi = np.clip(np.round(base_ys.astype(np.float32) + ny * offset).astype(np.int32), 0, height - 1)
            if xi.size < 2:
                continue

            start_off = (
                int(np.clip(round(base_start[0] + nx * offset), 0, width - 1)),
                int(np.clip(round(base_start[1] + ny * offset), 0, height - 1)),
            )
            end_off = (
                int(np.clip(round(base_end[0] + nx * offset), 0, width - 1)),
                int(np.clip(round(base_end[1] + ny * offset), 0, height - 1)),
            )
            rows.append((xi, yi, start_off, end_off))

        if not rows:
            raise RuntimeError("Slice row generation produced no valid rows.")

        return rows

    @staticmethod
    def _draw_slice_overlay(
        image,
        slice_rows,
        top_ratio,
        bottom_ratio,
        show_clip_guides=False,
        detailed_labels=False,
    ):
        overlay = image.copy()
        height, width = overlay.shape[:2]

        for row_idx, (xs, ys, start_pt, end_pt) in enumerate(slice_rows, start=1):
            for i in range(1, len(xs)):
                cv2.line(
                    overlay,
                    (int(xs[i - 1]), int(ys[i - 1])),
                    (int(xs[i]), int(ys[i])),
                    (0, 255, 255),
                    2,
                )

            cv2.circle(overlay, (int(start_pt[0]), int(start_pt[1])), 5, (0, 255, 0), -1)
            cv2.circle(overlay, (int(end_pt[0]), int(end_pt[1])), 5, (0, 128, 255), -1)

            label = f"R{row_idx} S{start_pt} E{end_pt}" if detailed_labels else f"{row_idx}"
            if detailed_labels:
                text_x = max(8, min(width - 420, int(start_pt[0]) + 8))
                text_y = max(20, min(height - 10, int(start_pt[1]) - 8))
                font_scale = 0.5
                text_outline = 3
                text_thickness = 1
            else:
                mid_x = int(round((start_pt[0] + end_pt[0]) * 0.5))
                mid_y = int(round((start_pt[1] + end_pt[1]) * 0.5))
                text_x = max(8, min(width - 30, mid_x - 10))
                text_y = max(24, min(height - 8, mid_y - 8))
                font_scale = 0.9
                text_outline = 4
                text_thickness = 2
            cv2.putText(
                overlay,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                text_outline,
            )
            cv2.putText(
                overlay,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                text_thickness,
            )

        if show_clip_guides:
            top_y = int(round(min(top_ratio, bottom_ratio) * (height - 1)))
            bottom_y = int(round(max(top_ratio, bottom_ratio) * (height - 1)))
            cv2.line(overlay, (0, top_y), (width - 1, top_y), (0, 0, 255), 1)
            cv2.line(overlay, (0, bottom_y), (width - 1, bottom_y), (0, 0, 255), 1)

            cv2.putText(
                overlay,
                "Yellow: sampled slice | Green: start | Orange: end | Red: top/bottom clip",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        return overlay

    @staticmethod
    def _save_slice_preview(
        image,
        preview_path,
        slice_rows,
        top_ratio,
        bottom_ratio,
        show_clip_guides=False,
    ):
        overlay = VisualReviewTool._draw_slice_overlay(
            image=image,
            slice_rows=slice_rows,
            top_ratio=top_ratio,
            bottom_ratio=bottom_ratio,
            show_clip_guides=show_clip_guides,
            detailed_labels=False,
        )

        preview_parent = Path(preview_path).parent
        if str(preview_parent) != "":
            preview_parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_path), overlay)

    @staticmethod
    def _infer_date_label(image_paths):
        if not image_paths:
            return datetime.now().strftime("%Y-%m-%d")

        candidate = Path(image_paths[0]).stem.split("_")[0]
        try:
            parsed = datetime.strptime(candidate, "%Y-%m-%d")
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            return candidate

    @staticmethod
    def _build_row_labeled_kymograph(row_kymographs):
        separator_height = 4
        parts = []
        row_spans = []
        cursor_y = 0

        for row_idx, row_img in enumerate(row_kymographs, start=1):
            bordered = cv2.copyMakeBorder(row_img, 1, 1, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            parts.append(bordered)
            row_spans.append((row_idx, cursor_y, bordered.shape[0]))
            cursor_y += bordered.shape[0]

            if row_idx < len(row_kymographs):
                separator = np.zeros((separator_height, bordered.shape[1], 3), dtype=np.uint8)
                parts.append(separator)
                cursor_y += separator_height

        body = np.vstack(parts)

        label_width = 56
        label_col = np.full((body.shape[0], label_width, 3), 255, dtype=np.uint8)
        cv2.line(label_col, (label_width - 1, 0), (label_width - 1, label_col.shape[0] - 1), (0, 0, 0), 2)

        for row_idx, start_y, height in row_spans:
            label_text = str(row_idx)
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            text_x = max(4, (label_width - text_w) // 2)
            text_y = start_y + (height + text_h) // 2 - 2
            cv2.putText(label_col, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        return np.hstack([label_col, body])

    def _build_top_panel(
        self,
        reference_image,
        slice_rows,
        top_ratio,
        bottom_ratio,
        panel_width,
        date_label,
        images_used,
    ):
        overlay = VisualReviewTool._draw_slice_overlay(
            image=reference_image,
            slice_rows=slice_rows,
            top_ratio=top_ratio,
            bottom_ratio=bottom_ratio,
            show_clip_guides=False,
            detailed_labels=False,
        )
        overlay = VisualReviewTool._crop_image_around_slices(overlay, slice_rows)

        side_panel_width = max(130, int(round(panel_width * 0.12)))
        max_allowed_photo_width = max(120, panel_width - (2 * side_panel_width) - 20)
        max_photo_width = max(120, int(round(panel_width * self.header_photo_width_ratio)))
        max_photo_width = min(max_photo_width, max_allowed_photo_width)
        scale = max_photo_width / float(overlay.shape[1])
        resized_w = max(1, int(round(overlay.shape[1] * scale)))
        resized_h = max(1, int(round(overlay.shape[0] * scale)))

        max_photo_height = self.header_photo_max_height
        if resized_h > max_photo_height:
            photo_scale = max_photo_height / float(resized_h)
            resized_w = max(1, int(round(resized_w * photo_scale)))
            resized_h = max(1, int(round(resized_h * photo_scale)))

        photo = cv2.resize(overlay, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        top_padding = 20
        bottom_padding = 12
        panel_height = top_padding + resized_h + bottom_padding
        panel = np.full((panel_height, panel_width, 3), 255, dtype=np.uint8)

        photo_x = max(0, (panel_width - resized_w) // 2)
        photo_y = top_padding
        panel[photo_y : photo_y + resized_h, photo_x : photo_x + resized_w] = photo
        cv2.rectangle(panel, (photo_x, photo_y), (photo_x + resized_w - 1, photo_y + resized_h - 1), (0, 0, 0), 2)

        left_x = 12
        left_y = top_padding + 30
        cv2.putText(panel, f"Date: {date_label}", (left_x, left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2)
        cv2.putText(
            panel,
            f"Slices: {len(slice_rows)}",
            (left_x, left_y + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            panel,
            f"Images: {int(images_used)}",
            (left_x, left_y + 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 0, 0),
            2,
        )

        right_anchor_x = panel_width - side_panel_width + 8
        legend_y = top_padding + 24
        cv2.putText(panel, "Legend", (right_anchor_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        cv2.line(panel, (right_anchor_x, legend_y + 20), (right_anchor_x + 34, legend_y + 20), (0, 255, 255), 2)
        cv2.putText(panel, "slice", (right_anchor_x + 40, legend_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        cv2.circle(panel, (right_anchor_x + 9, legend_y + 44), 5, (0, 255, 0), -1)
        cv2.putText(panel, "start", (right_anchor_x + 24, legend_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        cv2.circle(panel, (right_anchor_x + 9, legend_y + 67), 5, (0, 128, 255), -1)
        cv2.putText(panel, "end", (right_anchor_x + 24, legend_y + 71), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        capture_label = self.capture_window_label if self.capture_window_label else "n/a"
        cv2.putText(
            panel,
            "Capture",
            (right_anchor_x, legend_y + 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            panel,
            str(capture_label),
            (right_anchor_x, legend_y + 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )
        return panel

    @staticmethod
    def _slice_width(images_per_day, target_kymo_width):
        per_day = max(1, int(images_per_day))
        target = max(200, int(target_kymo_width))
        return max(1, int(round(target / per_day)))

    @staticmethod
    def _fit_row_to_target_width(strips, target_kymo_width):
        if not strips:
            return None
        target_width = max(1, int(target_kymo_width))

        strip_count = len(strips)
        base_width = target_width // strip_count
        remainder = target_width % strip_count
        target_widths = [base_width + (1 if idx < remainder else 0) for idx in range(strip_count)]

        resized_strips = []
        for strip, strip_width in zip(strips, target_widths):
            width = max(1, int(strip_width))
            if strip.shape[1] == width:
                resized_strips.append(strip)
            else:
                resized_strips.append(cv2.resize(strip, (width, strip.shape[0]), interpolation=cv2.INTER_LINEAR))

        row = np.hstack(resized_strips)
        return row

    @staticmethod
    def _crop_image_around_slices(image, slice_rows, margin_ratio=0.12, min_margin_px=20):
        if image is None or not slice_rows:
            return image

        image_h, image_w = image.shape[:2]
        x_values = []
        y_values = []
        for xs, ys, start_pt, end_pt in slice_rows:
            x_values.append(xs.astype(np.int32))
            y_values.append(ys.astype(np.int32))
            x_values.append(np.array([start_pt[0], end_pt[0]], dtype=np.int32))
            y_values.append(np.array([start_pt[1], end_pt[1]], dtype=np.int32))

        if not x_values:
            return image

        x_min = int(np.min(np.concatenate(x_values)))
        x_max = int(np.max(np.concatenate(x_values)))
        y_min = int(np.min(np.concatenate(y_values)))
        y_max = int(np.max(np.concatenate(y_values)))

        box_w = max(1, x_max - x_min + 1)
        box_h = max(1, y_max - y_min + 1)
        margin = max(int(min_margin_px), int(round(max(box_w, box_h) * float(margin_ratio))))

        left = max(0, x_min - margin)
        right = min(image_w, x_max + margin + 1)
        top = max(0, y_min - margin)
        bottom = min(image_h, y_max + margin + 1)

        if right - left < 2 or bottom - top < 2:
            return image

        return image[top:bottom, left:right].copy()

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
        slice_rows = self._slice_rows(width, height)

        if preview_path is not None:
            self._save_slice_preview(
                image=first,
                preview_path=preview_path,
                slice_rows=slice_rows,
                top_ratio=self.slice_top_ratio,
                bottom_ratio=self.slice_bottom_ratio,
                show_clip_guides=preview_show_clip_guides,
            )

        row_strips = [[] for _ in slice_rows]
        processed = []
        middle_index = len(images) // 2
        middle_reference_image = None
        fallback_reference_image = None
        for image_idx, path in enumerate(images):
            image = cv2.imread(str(path))
            if image is None:
                continue
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

            if fallback_reference_image is None:
                fallback_reference_image = image.copy()
            if image_idx == middle_index:
                middle_reference_image = image.copy()

            for row_idx, (xs, ys, start_pt, end_pt) in enumerate(slice_rows):
                profile = self._extract_profile(
                    image=image,
                    xs=xs,
                    ys=ys,
                    start_pt=start_pt,
                    end_pt=end_pt,
                    thickness_px=self.slice_thickness_px,
                )
                row_strips[row_idx].append(profile)

            processed.append(path.name)

        if not any(row for row in row_strips):
            raise RuntimeError("No valid images could be processed for kymograph generation.")

        row_kymographs = []
        for strips in row_strips:
            if not strips:
                continue
            if self.normalize_slice_brightness:
                strips = self._match_slice_brightness(strips)
            fitted = self._fit_row_to_target_width(strips, target_kymo_width)
            if fitted is not None:
                row_kymographs.append(fitted)

        if not row_kymographs:
            raise RuntimeError("No kymograph rows were generated from the provided images.")

        kymograph = self._build_row_labeled_kymograph(row_kymographs)

        if self.slice_vertical_stretch > 1.0:
            stretched_height = int(round(kymograph.shape[0] * self.slice_vertical_stretch))
            kymograph = cv2.resize(kymograph, (kymograph.shape[1], max(2, stretched_height)), interpolation=cv2.INTER_LINEAR)

        reference_image = middle_reference_image if middle_reference_image is not None else fallback_reference_image
        if reference_image is not None:
            date_label = self._infer_date_label(images)
            top_panel = self._build_top_panel(
                reference_image=reference_image,
                slice_rows=slice_rows,
                top_ratio=self.slice_top_ratio,
                bottom_ratio=self.slice_bottom_ratio,
                panel_width=kymograph.shape[1],
                date_label=date_label,
                images_used=len(processed),
            )
            divider = np.zeros((4, kymograph.shape[1], 3), dtype=np.uint8)
            kymograph = np.vstack([top_panel, divider, kymograph])

        cv2.imwrite(output_path, kymograph)

        print(f"✓ Kymograph saved: {output_path}")
        print(f"  Images used: {len(processed)}")
        if processed:
            dynamic_slice_width = float(max(200, int(target_kymo_width))) / float(len(processed))
            print(f"  Slice width per image: {dynamic_slice_width:.2f} px (dynamic fit to {max(200, int(target_kymo_width))}px)")
        else:
            print("  Slice width per image: n/a")
        print(f"  Source slice thickness: {self.slice_thickness_px} px")
        print(f"  Slice rows: {len(row_kymographs)}")
        print(f"  Vertical stretch: {self.slice_vertical_stretch:.2f}x")
        print(f"  Brightness normalized: {'yes' if self.normalize_slice_brightness else 'no'}")
        center_row = slice_rows[len(slice_rows) // 2]
        center_start, center_end = center_row[2], center_row[3]
        print(f"  Center slice endpoints: {center_start} -> {center_end}")
        return output_path, (center_start, center_end)

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
        default=96,
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
        "--capture-window-label",
        default=None,
        help="Optional capture window label shown in the header side panel",
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
        capture_window_label=args.capture_window_label,
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