import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np


DEFAULT_SLICE_START = "1080,300"
DEFAULT_SLICE_END = "1260,500"
DEFAULT_SLICE_ROW_STARTS = "1000,390;1100,350;1200,310"
DEFAULT_SLICE_ROW_ENDS = "1100,470;1200,430;1300,390"
DEFAULT_ROW_LABEL_WIDTH_PX = 20
FINAL_IMAGE_PADDING_PX = 15
SLICE_PLOT_BORDER_PX = 2
PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
UTC_TZ = ZoneInfo("UTC")


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
        capture_interval_label=None,
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
        self.capture_interval_label = capture_interval_label

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
            return f"{datetime.now(UTC_TZ).strftime('%Y/%m/%d')} UTC"

        parsed_times = VisualReviewTool._parsed_capture_times(image_paths)
        if parsed_times:
            parsed_times = sorted(parsed_times)
            start_date = parsed_times[0].strftime("%Y/%m/%d")
            end_date = parsed_times[-1].strftime("%Y/%m/%d")
            if start_date == end_date:
                return f"{start_date} UTC"
            return f"{start_date} - {end_date} UTC"

        candidate = Path(image_paths[0]).stem.split("_")[0]
        try:
            parsed = datetime.strptime(candidate, "%Y-%m-%d")
            return f"{parsed.strftime('%Y/%m/%d')} UTC"
        except ValueError:
            return candidate

    @staticmethod
    def _infer_capture_label(image_paths):
        if not image_paths:
            return None

        parsed_times = VisualReviewTool._parsed_capture_times(image_paths)

        if not parsed_times:
            return None

        parsed_times = sorted(parsed_times)
        start = parsed_times[0]
        end = parsed_times[-1]
        return f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')} UTC"

    @staticmethod
    def _infer_interval_label(image_paths):
        if not image_paths:
            return None

        parsed_times = VisualReviewTool._parsed_capture_times(image_paths)
        if len(parsed_times) < 2:
            return None

        parsed_times = sorted(parsed_times)
        deltas = []
        for idx in range(1, len(parsed_times)):
            delta_seconds = int((parsed_times[idx] - parsed_times[idx - 1]).total_seconds())
            if delta_seconds > 0:
                deltas.append(delta_seconds)

        if not deltas:
            return None

        interval_seconds = int(round(float(np.median(np.array(deltas, dtype=np.float32)))))
        if interval_seconds % 3600 == 0:
            hours = interval_seconds // 3600
            return f"{hours}h"
        if interval_seconds >= 3600:
            hours = interval_seconds // 3600
            minutes = (interval_seconds % 3600) // 60
            return f"{hours}h{minutes}m"
        if interval_seconds % 60 == 0:
            return f"{interval_seconds // 60}m"
        return f"{interval_seconds}s"

    @staticmethod
    def _parse_capture_datetime(path):
        stem = Path(path).stem
        tokens = stem.split("_")
        for idx in range(max(0, len(tokens) - 1)):
            candidate = f"{tokens[idx]}_{tokens[idx + 1]}"
            try:
                parsed_local = datetime.strptime(candidate, "%Y-%m-%d_%H-%M-%S")
                return parsed_local.replace(tzinfo=PACIFIC_TZ).astimezone(UTC_TZ)
            except ValueError:
                continue
        return None

    @staticmethod
    def _parsed_capture_times(image_paths):
        parsed_times = []
        for path in image_paths:
            parsed = VisualReviewTool._parse_capture_datetime(path)
            if parsed is not None:
                parsed_times.append(parsed)
        return parsed_times

    @staticmethod
    def _build_row_labeled_kymograph(row_kymographs):
        separator_height = 2
        parts = []

        for row_idx, row_img in enumerate(row_kymographs, start=1):
            parts.append(row_img)

            if row_idx < len(row_kymographs):
                separator = np.zeros((separator_height, row_img.shape[1], 3), dtype=np.uint8)
                parts.append(separator)

        body = np.vstack(parts)
        border = max(0, int(SLICE_PLOT_BORDER_PX))
        if border > 0:
            body = cv2.copyMakeBorder(
                body,
                border,
                border,
                border,
                border,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        return body

    @staticmethod
    def _resize_to_fit(image, max_width, max_height):
        if image is None:
            return None
        src_h, src_w = image.shape[:2]
        if src_h < 1 or src_w < 1:
            return image

        width_scale = float(max(1, int(max_width))) / float(src_w)
        height_scale = float(max(1, int(max_height))) / float(src_h)
        scale = min(width_scale, height_scale)
        out_w = max(1, int(round(src_w * scale)))
        out_h = max(1, int(round(src_h * scale)))
        if out_w == src_w and out_h == src_h:
            return image
        return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _slot_widths(total_width, slot_count):
        width = max(1, int(total_width))
        count = max(1, int(slot_count))
        base_width = width // count
        remainder = width % count
        return [base_width + (1 if idx < remainder else 0) for idx in range(count)]

    @staticmethod
    def _put_text_supersampled(image, text, org, font, font_scale, color, thickness, supersample=2):
        if image is None or text is None or text == "":
            return

        ss = max(1, int(supersample))
        if ss == 1:
            cv2.putText(image, str(text), org, font, font_scale, color, thickness, cv2.LINE_AA)
            return

        text_value = str(text)
        hi_scale = float(font_scale) * float(ss)
        hi_thickness = max(1, int(round(float(thickness) * float(ss))))
        (text_w_hi, text_h_hi), baseline_hi = cv2.getTextSize(text_value, font, hi_scale, hi_thickness)

        pad_hi = 4 * ss
        patch_w_hi = max(1, text_w_hi + (2 * pad_hi))
        patch_h_hi = max(1, text_h_hi + baseline_hi + (2 * pad_hi))
        patch_hi = np.zeros((patch_h_hi, patch_w_hi), dtype=np.uint8)

        baseline_y_hi = pad_hi + text_h_hi
        cv2.putText(
            patch_hi,
            text_value,
            (pad_hi, baseline_y_hi),
            font,
            hi_scale,
            255,
            hi_thickness,
            cv2.LINE_AA,
        )

        patch_w = max(1, int(round(patch_w_hi / float(ss))))
        patch_h = max(1, int(round(patch_h_hi / float(ss))))
        patch = cv2.resize(patch_hi, (patch_w, patch_h), interpolation=cv2.INTER_AREA)

        baseline_y = int(round(baseline_y_hi / float(ss)))
        pad = int(round(pad_hi / float(ss)))

        x0 = int(org[0] - pad)
        y0 = int(org[1] - baseline_y)
        x1 = x0 + patch.shape[1]
        y1 = y0 + patch.shape[0]

        img_h, img_w = image.shape[:2]
        clip_x0 = max(0, x0)
        clip_y0 = max(0, y0)
        clip_x1 = min(img_w, x1)
        clip_y1 = min(img_h, y1)
        if clip_x1 <= clip_x0 or clip_y1 <= clip_y0:
            return

        patch_x0 = clip_x0 - x0
        patch_y0 = clip_y0 - y0
        patch_x1 = patch_x0 + (clip_x1 - clip_x0)
        patch_y1 = patch_y0 + (clip_y1 - clip_y0)

        alpha = patch[patch_y0:patch_y1, patch_x0:patch_x1].astype(np.float32) / 255.0
        if alpha.size == 0:
            return
        alpha_3 = alpha[:, :, None]

        roi = image[clip_y0:clip_y1, clip_x0:clip_x1].astype(np.float32)
        color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        blended = (roi * (1.0 - alpha_3)) + (color_arr * alpha_3)
        image[clip_y0:clip_y1, clip_x0:clip_x1] = np.clip(blended, 0, 255).astype(np.uint8)

    def _build_top_panel(
        self,
        reference_image,
        slice_rows,
        top_ratio,
        bottom_ratio,
        panel_width,
        date_label,
        images_used,
        capture_label,
        interval_label,
    ):
        overlay = VisualReviewTool._draw_slice_overlay(
            image=reference_image,
            slice_rows=slice_rows,
            top_ratio=top_ratio,
            bottom_ratio=bottom_ratio,
            show_clip_guides=False,
            detailed_labels=False,
        )

        crop_margin_ratio = 0.22
        crop_min_margin_px = 24
        crop_bounds = VisualReviewTool._slice_crop_bounds(
            image_shape=overlay.shape,
            slice_rows=slice_rows,
            margin_ratio=crop_margin_ratio,
            min_margin_px=crop_min_margin_px,
        )
        if crop_bounds is not None:
            left, right, top, bottom = crop_bounds
            cv2.rectangle(overlay, (left, top), (right - 1, bottom - 1), (255, 0, 255), 4)

        cropped_overlay = VisualReviewTool._crop_image_around_slices(
            overlay,
            slice_rows,
            margin_ratio=crop_margin_ratio,
            min_margin_px=crop_min_margin_px,
        )

        outer_pad = 12
        center_gap = 12
        available_w = max(220, panel_width - (2 * outer_pad))
        content_w = int(round(available_w * self.header_photo_width_ratio))
        content_w = max(220, min(available_w, content_w))
        content_left = outer_pad + max(0, (available_w - content_w) // 2)
        image_columns_w = max(200, content_w - center_gap)
        left_w = max(100, image_columns_w // 2)
        right_w = max(100, image_columns_w - left_w)

        info_height = 108
        right_image_max_height = max(120, int(self.header_photo_max_height))

        left_image = VisualReviewTool._resize_to_fit(
            overlay,
            max_width=left_w,
            max_height=right_image_max_height,
        )
        right_image = VisualReviewTool._resize_to_fit(
            cropped_overlay,
            max_width=right_w,
            max_height=right_image_max_height,
        )

        images_height = max(left_image.shape[0], right_image.shape[0])
        content_height = info_height + 10 + images_height
        panel_height = (2 * outer_pad) + content_height
        panel = np.full((panel_height, panel_width, 3), 255, dtype=np.uint8)

        display_capture = capture_label if capture_label else "n/a"
        display_interval = interval_label if interval_label else "n/a"

        title_text = "Caped North Head - Kymograph"
        title_y = outer_pad + 28
        (title_w, _), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.82, 2)
        title_x = max(6, (panel_width - title_w) // 2)
        VisualReviewTool._put_text_supersampled(
            panel,
            title_text,
            (title_x, title_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (0, 0, 0),
            2,
            supersample=2,
        )

        info_top_y = outer_pad + 66
        col_edges = np.linspace(outer_pad, panel_width - outer_pad, 4).astype(np.int32)
        columns = [
            f"Capture: {display_capture}",
            f"Date: {date_label}",
            f"Images: {int(images_used)} | Interval: {display_interval}",
        ]
        for idx, text in enumerate(columns):
            center_x = int(round((int(col_edges[idx]) + int(col_edges[idx + 1])) / 2.0))
            (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 2)
            text_x = max(6, center_x - (text_w // 2))
            VisualReviewTool._put_text_supersampled(
                panel,
                text,
                (text_x, info_top_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (0, 0, 0),
                2,
                supersample=2,
            )

        left_column_x = content_left
        left_x = left_column_x + max(0, (left_w - left_image.shape[1]) // 2)
        left_y = outer_pad + info_height + 10 + max(0, (images_height - left_image.shape[0]) // 2)
        panel[left_y : left_y + left_image.shape[0], left_x : left_x + left_image.shape[1]] = left_image
        cv2.rectangle(
            panel,
            (left_x, left_y),
            (left_x + left_image.shape[1] - 1, left_y + left_image.shape[0] - 1),
            (0, 0, 0),
            2,
        )

        right_column_x = left_column_x + left_w + center_gap
        right_x = right_column_x + max(0, (right_w - right_image.shape[1]) // 2)
        right_y = outer_pad + info_height + 10 + max(0, (images_height - right_image.shape[0]) // 2)
        panel[right_y : right_y + right_image.shape[0], right_x : right_x + right_image.shape[1]] = right_image
        cv2.rectangle(
            panel,
            (right_x, right_y),
            (right_x + right_image.shape[1] - 1, right_y + right_image.shape[0] - 1),
            (0, 0, 0),
            2,
        )
        return panel

    @staticmethod
    def _slice_width(images_per_day, target_kymo_width):
        per_day = max(1, int(images_per_day))
        target = max(200, int(target_kymo_width))
        label_width = min(DEFAULT_ROW_LABEL_WIDTH_PX, max(1, target - per_day))
        usable_width = max(per_day, target - label_width)
        return max(1, int(round(usable_width / per_day)))

    @staticmethod
    def _fit_row_to_target_width(strips, target_kymo_width, row_label_text=None):
        if not strips:
            return None
        target_width = max(1, int(target_kymo_width))

        include_label_slot = row_label_text is not None and str(row_label_text) != ""
        strip_count = len(strips)
        slot_count = strip_count + (1 if include_label_slot else 0)
        target_widths = VisualReviewTool._slot_widths(total_width=target_width, slot_count=slot_count)

        resized_strips = []
        if include_label_slot:
            label_width = min(DEFAULT_ROW_LABEL_WIDTH_PX, max(1, target_width - len(strips)))
            label_strip = np.full((strips[0].shape[0], label_width, 3), 235, dtype=np.uint8)
            text = str(row_label_text)
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            text_x = max(2, (label_width - text_w) // 2)
            text_y = max(text_h + 2, (label_strip.shape[0] + text_h) // 2)
            VisualReviewTool._put_text_supersampled(
                label_strip,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                supersample=2,
            )
            resized_strips.append(label_strip)
            strip_target_widths = VisualReviewTool._slot_widths(
                total_width=max(1, target_width - label_width),
                slot_count=len(strips),
            )
        else:
            strip_target_widths = target_widths

        for strip, strip_width in zip(strips, strip_target_widths):
            width = max(1, int(strip_width))
            if strip.shape[1] == width:
                resized_strips.append(strip)
            else:
                resized_strips.append(cv2.resize(strip, (width, strip.shape[0]), interpolation=cv2.INTER_LINEAR))

        row = np.hstack(resized_strips)
        return row

    @staticmethod
    def _add_time_axis(kymograph_image, image_paths, target_kymo_width):
        if kymograph_image is None or not image_paths:
            return kymograph_image

        parsed_times = []
        for path in image_paths:
            parsed_times.append(VisualReviewTool._parse_capture_datetime(path))

        valid_count = len([ts for ts in parsed_times if ts is not None])
        if valid_count < 2:
            return kymograph_image

        total_width = max(1, int(kymograph_image.shape[1]))
        target_width = max(1, int(target_kymo_width))
        side_border = max(0, int(SLICE_PLOT_BORDER_PX))
        if total_width < (2 * side_border) + 1:
            side_border = 0

        if total_width != target_width + (2 * side_border):
            side_border = 0

        content_left = side_border
        content_width = max(1, total_width - (2 * side_border))

        axis_h = 56
        axis = np.full((axis_h, total_width, 3), 255, dtype=np.uint8)
        baseline_y = 1
        tick_len = 8
        baseline_x0 = content_left
        baseline_x1 = max(content_left, content_left + content_width - 1)
        cv2.line(axis, (baseline_x0, baseline_y), (baseline_x1, baseline_y), (0, 0, 0), 1, cv2.LINE_AA)

        label_width = min(DEFAULT_ROW_LABEL_WIDTH_PX, max(1, content_width - len(image_paths)))
        slot_widths = VisualReviewTool._slot_widths(
            total_width=max(1, content_width - label_width),
            slot_count=len(image_paths),
        )
        left_edges = [content_left + label_width]
        for slot_w in slot_widths[:-1]:
            left_edges.append(left_edges[-1] + slot_w)

        previous_labeled_hour_key = None
        for idx in range(len(image_paths)):
            x0 = left_edges[idx]
            w = slot_widths[idx]
            center_x = int(round(x0 + (w / 2.0)))
            cv2.line(axis, (center_x, baseline_y), (center_x, baseline_y + tick_len), (0, 0, 0), 1, cv2.LINE_AA)

            ts = parsed_times[idx]
            if ts is None:
                continue
            current_hour_key = (ts.date(), ts.hour)
            if previous_labeled_hour_key is not None and current_hour_key == previous_labeled_hour_key:
                continue

            label = ts.strftime("%H")
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            text_x = int(np.clip(center_x - (text_w // 2), 0, max(0, total_width - text_w - 1)))
            preferred_y = baseline_y + tick_len + 8 + text_h
            text_y = int(min(axis_h - 4, preferred_y))
            VisualReviewTool._put_text_supersampled(
                axis,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 0, 0),
                2,
                supersample=2,
            )
            previous_labeled_hour_key = current_hour_key

        axis_title = "Time UTC"
        (title_w, title_h), _ = cv2.getTextSize(axis_title, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        title_x = max(2, (total_width - title_w) // 2)
        title_y = max(title_h + 2, axis_h - 4)
        VisualReviewTool._put_text_supersampled(
            axis,
            axis_title,
            (title_x, title_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            2,
            supersample=2,
        )

        return np.vstack([kymograph_image, axis])

    @staticmethod
    def _crop_image_around_slices(image, slice_rows, margin_ratio=0.12, min_margin_px=20):
        bounds = VisualReviewTool._slice_crop_bounds(
            image_shape=None if image is None else image.shape,
            slice_rows=slice_rows,
            margin_ratio=margin_ratio,
            min_margin_px=min_margin_px,
        )
        if image is None or bounds is None:
            return image

        left, right, top, bottom = bounds
        if right - left < 2 or bottom - top < 2:
            return image

        return image[top:bottom, left:right].copy()

    @staticmethod
    def _slice_crop_bounds(image_shape, slice_rows, margin_ratio=0.12, min_margin_px=20):
        if image_shape is None or not slice_rows:
            return None

        image_h, image_w = image_shape[:2]
        x_values = []
        y_values = []
        for xs, ys, start_pt, end_pt in slice_rows:
            x_values.append(xs.astype(np.int32))
            y_values.append(ys.astype(np.int32))
            x_values.append(np.array([start_pt[0], end_pt[0]], dtype=np.int32))
            y_values.append(np.array([start_pt[1], end_pt[1]], dtype=np.int32))

        if not x_values:
            return None

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
            return None
        return left, right, top, bottom

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
        target_kymo_width=1200,
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
        for row_idx, strips in enumerate(row_strips, start=1):
            if not strips:
                continue
            if self.normalize_slice_brightness:
                strips = self._match_slice_brightness(strips)
            fitted = self._fit_row_to_target_width(strips, target_kymo_width, row_label_text=str(row_idx))
            if fitted is not None:
                row_kymographs.append(fitted)

        if not row_kymographs:
            raise RuntimeError("No kymograph rows were generated from the provided images.")

        kymograph = self._build_row_labeled_kymograph(row_kymographs)
        kymograph = self._add_time_axis(kymograph, images, max(200, int(target_kymo_width)))

        if self.slice_vertical_stretch > 1.0:
            stretched_height = int(round(kymograph.shape[0] * self.slice_vertical_stretch))
            kymograph = cv2.resize(kymograph, (kymograph.shape[1], max(2, stretched_height)), interpolation=cv2.INTER_LINEAR)

        reference_image = middle_reference_image if middle_reference_image is not None else fallback_reference_image
        if reference_image is not None:
            date_label = self._infer_date_label(images)
            inferred_capture_label = self.capture_window_label if self.capture_window_label else self._infer_capture_label(images)
            interval_label = self.capture_interval_label if self.capture_interval_label else self._infer_interval_label(images)
            top_panel = self._build_top_panel(
                reference_image=reference_image,
                slice_rows=slice_rows,
                top_ratio=self.slice_top_ratio,
                bottom_ratio=self.slice_bottom_ratio,
                panel_width=kymograph.shape[1],
                date_label=date_label,
                images_used=len(processed),
                capture_label=inferred_capture_label,
                interval_label=interval_label,
            )
            divider = np.full((4, kymograph.shape[1], 3), 255, dtype=np.uint8)
            kymograph = np.vstack([top_panel, divider, kymograph])

        pad = max(0, int(FINAL_IMAGE_PADDING_PX))
        if pad > 0:
            kymograph = cv2.copyMakeBorder(
                kymograph,
                pad,
                pad,
                pad,
                pad,
                borderType=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )

        cv2.imwrite(output_path, kymograph)

        print(f"✓ Kymograph saved: {output_path}")
        print(f"  Images used: {len(processed)}")
        if processed:
            output_width = max(200, int(target_kymo_width))
            label_width = min(DEFAULT_ROW_LABEL_WIDTH_PX, max(1, output_width - len(processed)))
            dynamic_slice_width = float(max(1, output_width - label_width)) / float(len(processed))
            print(
                f"  Slice width per image: {dynamic_slice_width:.2f} px "
                f"(dynamic fit to {output_width}px, {label_width}px label slot)"
            )
            capture_range_label = self.capture_window_label if self.capture_window_label else self._infer_capture_label(images)
            if capture_range_label is not None:
                print(f"  Capture range: {capture_range_label}")
            if self.capture_interval_label is not None:
                print(f"  Configured interval: {self.capture_interval_label}")
            else:
                inferred_interval = self._infer_interval_label(images)
                if inferred_interval is not None:
                    print(f"  Inferred interval: {inferred_interval}")
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
        target_kymo_width=1200,
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
        target_kymo_width=1200,
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
        default=55,
        help="Expected captures per day (default 55 for 07:00-16:00 at 10-minute captures)",
    )
    parser.add_argument(
        "--target-kymo-width",
        type=int,
        default=1200,
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
        "--capture-interval-label",
        default=None,
        help="Optional capture interval label shown in header metadata (example: 10m)",
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
        capture_interval_label=args.capture_interval_label,
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