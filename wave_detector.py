import os
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt

class AngledTreeLineDetector:
    def __init__(self, boundary_points):
        """Initialize with boundary line coordinates."""
        self.boundary_points = boundary_points

    def detect(self, scan_length):
        """Scan for wet sand across the boundary line."""
        # Assuming boundary_points is a list of tuples [(x1, y1), (x2, y2)]
        detection_points = []
        for i in range(len(self.boundary_points) - 1):
            p1 = np.array(self.boundary_points[i])
            p2 = np.array(self.boundary_points[i + 1])
            # Calculate the direction of the boundary line
            line_dir = p2 - p1
            line_unit = line_dir / np.linalg.norm(line_dir)

            # Calculate perpendicular direction
            perp_dir = np.array([-line_unit[1], line_unit[0]])
            detection_point = p1 + (scan_length * perp_dir)
            detection_points.append(detection_point)

        return detection_points

    def visualize(self, detection_points):
        """Visualize the boundary line and detection points."""
        boundary_x, boundary_y = zip(*self.boundary_points)
        detection_x, detection_y = zip(*detection_points)

        plt.figure()
        plt.plot(boundary_x, boundary_y, 'g-', label='Boundary Line')  # boundary line
        plt.scatter(detection_x, detection_y, color='red', label='Detection Points')  # detection points
        plt.title('Angled Tree Line Detection')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid()
        plt.show()


class WaveDetector:
    def __init__(
        self,
        treeline_points=((820, 440), (930, 420), (1090, 370), (1280, 305)),
        transect_count=4,
        max_scan_percent=0.75,
        nearshore_search_fraction=0.58,
        per_point_max_distance_px=(260, 260, 220, 220),
        axis_step_px=100,
        treeline_segment_x_percent=(0.58, 0.95),
        transect_angle_offset_deg=-30.0,
    ):
        self.treeline_points = treeline_points
        self.transect_count = transect_count
        self.max_scan_percent = max_scan_percent
        self.nearshore_search_fraction = nearshore_search_fraction
        self.per_point_max_distance_px = per_point_max_distance_px
        self.axis_step_px = axis_step_px
        self.treeline_segment_x_percent = treeline_segment_x_percent
        self.transect_angle_offset_deg = transect_angle_offset_deg

    def _clipped_control_points(self, image_shape):
        height, width = image_shape[:2]
        clipped = []
        for x, y in self.treeline_points:
            clipped.append((int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))))
        return clipped

    def detect_wave(self, image, output_path="wave_detector_output.png"):
        working = image.copy()
        hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)

        transects = self._measure_transects(working, hsv, lab)
        self._visualize(working, transects, output_path)

        distances = [t["distance_px"] for t in transects if t.get("distance_px") is not None]
        mean_distance = float(np.mean(distances)) if distances else None

        return {
            "transects": transects,
            "mean_distance_px": mean_distance,
        }

    def _local_tangent_slope(self, points, index):
        if len(points) < 2:
            return 0.0

        if index <= 0:
            x1, y1 = points[0]
            x2, y2 = points[1]
        elif index >= len(points) - 1:
            x1, y1 = points[-2]
            x2, y2 = points[-1]
        else:
            x1, y1 = points[index - 1]
            x2, y2 = points[index + 1]

        dx = float(x2 - x1)
        if abs(dx) < 1e-6:
            return 0.0
        return float(y2 - y1) / dx

    def _estimate_water_centroid(self, hsv):
        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)

        water_mask = (
            ((h >= 88.0) & (h <= 132.0) & (s >= 18.0) & (v >= 40.0))
            | ((s <= 55.0) & (v >= 145.0))
        )

        ys, xs = np.where(water_mask)
        height, width = hsv.shape[:2]
        if xs.size < 100:
            return np.array([width * 0.85, height * 0.70], dtype=np.float32)

        return np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float32)

    def _sample_ray(self, image_shape, start_point, direction, max_distance):
        height, width = image_shape[:2]
        points = []
        for d in range(0, int(max_distance) + 1):
            p = start_point + direction * float(d)
            x = int(round(p[0]))
            y = int(round(p[1]))
            if x < 0 or x >= width or y < 0 or y >= height:
                break
            points.append((x, y))
        return points

    def _find_waterline_index(self, hsv_samples, lab_samples):
        if len(hsv_samples) < 20:
            return None

        h = hsv_samples[:, 0].astype(np.float32)
        s = hsv_samples[:, 1].astype(np.float32)
        v = hsv_samples[:, 2].astype(np.float32)
        l = lab_samples[:, 0].astype(np.float32)

        wetness = 0.6 * np.clip((145.0 - l) / 90.0, 0.0, 1.0) + 0.4 * np.clip((75.0 - s) / 75.0, 0.0, 1.0)
        wetness = cv2.GaussianBlur(wetness.reshape(-1, 1), (9, 1), 0).reshape(-1)

        dry_sand = (
            np.clip(1.0 - np.abs(h - 20.0) / 26.0, 0.0, 1.0)
            * np.clip((s - 35.0) / 120.0, 0.0, 1.0)
            * np.clip((160.0 - l) / 90.0, 0.0, 1.0)
        )
        dry_sand = cv2.GaussianBlur(dry_sand.reshape(-1, 1), (9, 1), 0).reshape(-1)

        wet_blue = np.maximum(
            np.clip(1.0 - np.abs(h - 105.0) / 28.0, 0.0, 1.0) * np.clip((s - 10.0) / 95.0, 0.0, 1.0),
            np.clip(1.0 - np.abs(h - 108.0) / 26.0, 0.0, 1.0) * np.clip((v - 70.0) / 120.0, 0.0, 1.0),
        )
        wet_blue = cv2.GaussianBlur(wet_blue.reshape(-1, 1), (9, 1), 0).reshape(-1)

        foam_like = ((s <= 55.0) & (v >= 155.0) & (l >= 155.0)).astype(np.float32)
        foam_like = cv2.GaussianBlur(foam_like.reshape(-1, 1), (7, 1), 0).reshape(-1)

        blue_water = (h >= 86.0) & (h <= 132.0) & (s >= 16.0) & (v >= 35.0)
        water_mask = blue_water.astype(np.float32)
        water_smooth = cv2.GaussianBlur(water_mask.reshape(-1, 1), (11, 1), 0).reshape(-1)

        grad_wet_blue = np.gradient(wet_blue)
        n = len(wetness)
        start_i = max(8, int(n * 0.08))
        end_i = min(max(start_i + 1, int(n * float(self.nearshore_search_fraction))), n - 1)
        if end_i - start_i < 14:
            return None

        window = 7
        for i in range(start_i + window, end_i - window):
            sand_before = float(np.mean(dry_sand[i - window : i]))
            wet_after = float(np.mean(wet_blue[i : i + window]))
            water_after = float(np.mean(water_smooth[i : i + window]))
            foam_after = float(np.mean(foam_like[i : i + window]))

            if (
                sand_before >= 0.22
                and wet_after >= 0.34
                and (wet_after - sand_before) >= 0.05
                and water_after >= 0.24
                and foam_after <= 0.50
                and grad_wet_blue[i] > 0.002
            ):
                return int(i)

        window = 8
        for i in range(start_i + window, end_i - window):
            before_water = float(np.mean(water_smooth[i - window : i]))
            after_water = float(np.mean(water_smooth[i : i + window]))
            before_wet = float(np.mean(wetness[i - window : i]))
            after_foam = float(np.mean(foam_like[i : i + window]))

            if (
                before_water <= 0.26
                and after_water >= 0.48
                and (after_water - before_water) >= 0.18
                and before_wet >= 0.18
                and after_foam <= 0.55
            ):
                return int(i)

        transition_candidates = np.where(
            (np.arange(n) >= start_i)
            & (np.arange(n) < end_i)
            & (wetness >= 0.30)
            & (wet_blue >= 0.28)
            & (grad_wet_blue > 0.003)
            & (water_smooth >= 0.26)
            & (foam_like <= 0.60)
        )[0]
        if transition_candidates.size > 0:
            return int(transition_candidates[0])

        sustained = np.where((water_smooth[start_i:end_i] >= 0.50) & (foam_like[start_i:end_i] <= 0.60))[0]
        if sustained.size > 0:
            return int(start_i + sustained[0])

        combined = 0.45 * wetness + 0.35 * water_smooth + 0.35 * wet_blue - 0.25 * foam_like
        if combined[start_i:end_i].size > 0:
            best = int(start_i + np.argmax(combined[start_i:end_i]))
            if float(combined[best]) >= 0.32:
                return best

        return None

    def _treeline_x_span(self, width):
        start_pct, end_pct = self.treeline_segment_x_percent
        start_pct = float(np.clip(start_pct, 0.0, 1.0))
        end_pct = float(np.clip(end_pct, 0.0, 1.0))
        if end_pct < start_pct:
            start_pct, end_pct = end_pct, start_pct

        x_start = int(round(start_pct * (width - 1)))
        x_end = int(round(end_pct * (width - 1)))
        if x_end <= x_start:
            x_end = min(width - 1, x_start + 1)
        return x_start, x_end

    def _control_point_x_span(self, width):
        xs = [int(np.clip(p[0], 0, width - 1)) for p in self.treeline_points]
        x_start = int(min(xs))
        x_end = int(max(xs))
        if x_end <= x_start:
            x_end = min(width - 1, x_start + 1)
        return x_start, x_end

    def _rotate_vector(self, vec, degrees):
        theta = np.deg2rad(float(degrees))
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        x, y = float(vec[0]), float(vec[1])
        return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)

    def _point_distance_limit(self, index_1_based):
        if self.per_point_max_distance_px is None:
            return None
        values = list(self.per_point_max_distance_px)
        if not values:
            return None
        idx = max(0, min(index_1_based - 1, len(values) - 1))
        return max(20.0, float(values[idx]))

    def _measure_transects(self, image, hsv, lab):
        height, width = image.shape[:2]
        max_scan = max(30.0, min(width, height) * self.max_scan_percent)
        water_centroid = self._estimate_water_centroid(hsv)

        control_points = sorted(self._clipped_control_points(image.shape), key=lambda p: p[0])

        transects = []
        for i, (cx, cy) in enumerate(control_points, start=1):
            dy_dx = self._local_tangent_slope(control_points, i - 1)

            normal = np.array([-dy_dx, 1.0], dtype=np.float32)
            norm = float(np.linalg.norm(normal))
            if norm < 1e-6:
                normal = np.array([0.0, 1.0], dtype=np.float32)
            else:
                normal /= norm

            normal = self._rotate_vector(normal, self.transect_angle_offset_deg)
            rotated_norm = float(np.linalg.norm(normal))
            if rotated_norm < 1e-6:
                normal = np.array([0.0, 1.0], dtype=np.float32)
            else:
                normal /= rotated_norm

            start = np.array([float(cx), float(cy)], dtype=np.float32)
            if float(np.dot(normal, water_centroid - start)) < 0.0:
                normal = -normal

            ray = self._sample_ray(image.shape, start, normal, max_scan)
            point_limit = self._point_distance_limit(i)
            if point_limit is not None:
                ray = ray[: max(2, int(point_limit) + 1)]

            if len(ray) < 20:
                transects.append(
                    {
                        "index": i,
                        "treeline_point": (int(round(cx)), int(round(cy))),
                        "waterline_point": None,
                        "distance_px": None,
                    }
                )
                continue

            hsv_samples = np.array([hsv[py, px, :] for px, py in ray], dtype=np.uint8)
            lab_samples = np.array([lab[py, px, :] for px, py in ray], dtype=np.uint8)
            water_idx = self._find_waterline_index(hsv_samples, lab_samples)

            if water_idx is None or water_idx >= len(ray):
                transects.append(
                    {
                        "index": i,
                        "treeline_point": (int(round(cx)), int(round(cy))),
                        "waterline_point": None,
                        "distance_px": None,
                    }
                )
                continue

            water_point = ray[water_idx]
            dist = float(np.linalg.norm(np.array(water_point, dtype=np.float32) - start))
            transects.append(
                {
                    "index": i,
                    "treeline_point": (int(round(cx)), int(round(cy))),
                    "waterline_point": (int(water_point[0]), int(water_point[1])),
                    "distance_px": round(dist, 2),
                }
            )

        return transects

    def _draw_axes(self, image):
        height, width = image.shape[:2]
        axis_color = (200, 200, 200)

        cv2.arrowedLine(image, (20, height - 20), (min(width - 20, 260), height - 20), axis_color, 2, tipLength=0.04)
        cv2.arrowedLine(image, (20, height - 20), (20, max(20, height - 260)), axis_color, 2, tipLength=0.04)
        cv2.putText(image, "x", (min(width - 28, 266), height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 2)
        cv2.putText(image, "y", (24, max(26, height - 266)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 2)

        step = max(50, int(self.axis_step_px))
        for x in range(20, width, step):
            cv2.line(image, (x, height - 24), (x, height - 16), axis_color, 1)
            if x > 20:
                cv2.putText(image, str(x), (x - 12, height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, axis_color, 1)

        for y in range(height - 20, -1, -step):
            cv2.line(image, (16, y), (24, y), axis_color, 1)
            if y < height - 20:
                cv2.putText(image, str(y), (28, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, axis_color, 1)

    def _visualize(self, image, transects, output_path):
        output = image.copy()
        height, width = output.shape[:2]

        self._draw_axes(output)

        control_points = sorted(self._clipped_control_points(image.shape), key=lambda p: p[0])
        for idx, point in enumerate(control_points, start=1):
            cv2.circle(output, point, 6, (255, 0, 255), -1)
            cv2.putText(
                output,
                f"P{idx}",
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 255),
                1,
            )

        if len(control_points) > 1:
            cv2.polylines(output, [np.array(control_points, dtype=np.int32)], False, (0, 255, 255), 2)

        cv2.putText(
            output,
            f"line_x_range={self.treeline_segment_x_percent}, angle_offset={self.transect_angle_offset_deg} deg",
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
        )

        for t in transects:
            tx, ty = t["treeline_point"]
            cv2.circle(output, (tx, ty), 4, (0, 255, 0), -1)

            water = t.get("waterline_point")
            if water is not None:
                wx, wy = water
                cv2.circle(output, (wx, wy), 4, (0, 0, 255), -1)
                cv2.line(output, (tx, ty), (wx, wy), (255, 255, 0), 2)
                label = f"{t['index']}: {t['distance_px']} px"
            else:
                label = f"{t['index']}: NA"

            ly = int(np.clip(ty - 8, 18, height - 6))
            cv2.putText(output, label, (tx + 8, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_path, output)


if __name__ == "__main__":
    input_dir = "test_images"
    output_dir = os.path.join(input_dir, "outputs")
    output_txt_path = os.path.join(output_dir, "wave_measurements.txt")
    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    os.makedirs(output_dir, exist_ok=True)

    image_paths = []
    for file_name in sorted(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        stem, ext = os.path.splitext(file_name)
        if ext.lower() in supported_extensions and not stem.endswith("_annotated"):
            image_paths.append(file_path)

    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in {input_dir}")

    detector = WaveDetector()
    point_count = len(detector.treeline_points)

    with open(output_txt_path, "w", encoding="utf-8") as txt_file:
        measurement_headers = [f"m{i}_px" for i in range(1, point_count + 1)]
        txt_file.write(f"DateTime, {', '.join(measurement_headers)}, mean_px\n")

        for input_path in image_paths:
            image = cv2.imread(input_path)
            if image is None:
                continue

            base_name = os.path.basename(input_path)
            stem, _ = os.path.splitext(base_name)
            output_path = os.path.join(output_dir, f"{stem}_annotated.png")

            result = detector.detect_wave(image, output_path=output_path)
            transects = result.get("transects", [])

            values = [None] * point_count
            for transect in transects:
                idx = int(transect.get("index", 0)) - 1
                if 0 <= idx < point_count:
                    values[idx] = transect.get("distance_px")

            mean_distance = result.get("mean_distance_px")

            date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            value_strs = ["NA" if v is None else f"{float(v):.2f}" for v in values]
            mean_str = "NA" if mean_distance is None else f"{float(mean_distance):.2f}"

            txt_file.write(f"{date_str}, {', '.join(value_strs)}, {mean_str}\n")
            print(f"Saved output image to: {output_path}")

    print(f"Saved measurements to: {output_txt_path}")
