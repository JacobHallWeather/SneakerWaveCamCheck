import cv2
import numpy as np

class WaveDetector:
    def __init__(self, left_region_start=0.05, left_region_end=0.35,
                 right_region_start=0.65, right_region_end=0.95,
                 vertical_percent=0.30, bottom_margin_px=60):
        self.left_region_start = left_region_start
        self.left_region_end = left_region_end
        self.right_region_start = right_region_start
        self.right_region_end = right_region_end
        self.vertical_percent = vertical_percent
        self.bottom_margin_px = bottom_margin_px

    def detect_wave(self, image):
        height, width, _ = image.shape

        # Define regions
        left_region = image[int(height * self.vertical_percent):height - self.bottom_margin_px,
                            int(width * self.left_region_start):int(width * self.left_region_end)]
        right_region = image[int(height * self.vertical_percent):height - self.bottom_margin_px,
                             int(width * self.right_region_start):int(width * self.right_region_end)]

        # Generate sample points
        left_sample_points = [(int(width * self.left_region_start + (i * (width * (self.left_region_end - self.left_region_start) / 4)), int(height * self.vertical_percent)) ) for i in range(4)]
        right_sample_points = [(int(width * self.right_region_start + (i * (width * (self.right_region_end - self.right_region_start) / 4)), int(height * self.vertical_percent)) ) for i in range(4)]

        # Perform gradient-based waterline detection
        left_waterline = self._detect_waterline(left_region)
        right_waterline = self._detect_waterline(right_region)

        # Check for sneaker wave conditions
        if left_waterline is None and right_waterline is None:
            return "No waterline detected in both regions. Sneaker wave alert!"

        # Visualization
        self.visualize(image, left_sample_points, right_sample_points, left_waterline, right_waterline)
        return left_waterline, right_waterline

    def _detect_waterline(self, region):
        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Calculate gradient
gradient = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        _, thresh = cv2.threshold(np.abs(gradient), 30, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        # Find the highest point in the contours as the beach waterline
        waterline = max(contours, key=lambda x: cv2.contourArea(x))
        return waterline

    def visualize(self, original_image, left_points, right_points, left_waterline, right_waterline):
        # Draw sample points
        for point in left_points:
            cv2.circle(original_image, point, 5, (0, 255, 0), -1)  # Green for left
        for point in right_points:
            cv2.circle(original_image, point, 5, (255, 0, 0), -1)  # Blue for right

        # Draw detected waterlines
        if left_waterline is not None:
            cv2.drawContours(original_image, [left_waterline], -1, (0, 0, 255), 2)  # Red for left waterline
        if right_waterline is not None:
            cv2.drawContours(original_image, [right_waterline], -1, (0, 0, 255), 2)  # Red for right waterline

        # Show the image
        cv2.imshow('Wave Detector', original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()