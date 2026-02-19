import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.ndimage import gaussian_filter1d

class BeachWaveDetector:
    """
    Detects wet sand waterline in beach photos to identify sneaker waves
    """
    
    def __init__(self, 
                 horizontal_percent=0.30,
                 vertical_percent=0.25,
                 right_margin_px=120,
                 bottom_margin_px=60):
        """
        Initialize detector with scan region parameters
        
        Args:
            horizontal_percent: Portion of image width to scan from right edge (0.0-1.0)
            vertical_percent: Portion of image height to scan from bottom edge (0.0-1.0)
            right_margin_px: Pixels to inset from right edge (to avoid timestamp)
            bottom_margin_px: Pixels to inset from bottom edge (to avoid timestamp)
        """
        self.horizontal_percent = horizontal_percent
        self.vertical_percent = vertical_percent
        self.right_margin_px = right_margin_px
        self.bottom_margin_px = bottom_margin_px
        self.num_sample_points = 4
        
    def detect_and_visualize(self, image_path, output_path=None):
        """
        Main function: Detect waterline and create detailed visualization
        
        Args:
            image_path: Path to beach image
            output_path: Optional path for output image (default: adds _analyzed suffix)
            
        Returns:
            Dictionary with detection results
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Could not load image: {image_path}")
            return None
            
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate scan region with margins to avoid timestamp
        scan_right = width - self.right_margin_px
        scan_width = int(width * self.horizontal_percent)
        scan_left = scan_right - scan_width
        
        scan_bottom = height - self.bottom_margin_px
        scan_height = int(height * self.vertical_percent)
        scan_top = scan_bottom - scan_height
        
        # Safety checks
        if scan_left < 0 or scan_top < 0 or scan_width < 50 or scan_height < 50:
            print(f"ERROR: Scan region too small or invalid!")
            print(f"  Image: {width}×{height}, Scan: {scan_width}×{scan_height}")
            print(f"  Try reducing margins or increasing percentages")
            return None
        
        print(f"\n{'='*60}")
        print(f"Analyzing: {Path(image_path).name}")
        print(f"{'='*60}")
        print(f"Image size: {width} × {height}px")
        print(f"Margins: right={self.right_margin_px}px, bottom={self.bottom_margin_px}px")
        print(f"Scan region: {scan_width} × {scan_height}px ({scan_width*scan_height} total pixels)")
        print(f"Scan area: ({scan_left}, {scan_top}) to ({scan_right}, {scan_bottom})")
        print(f"Image coverage: {(scan_width*scan_height)/(width*height)*100:.1f}% of total image")
        
        # Extract scan region
        scan_region = gray[scan_top:scan_bottom, scan_left:scan_right]
        
        # Calculate sample X positions (4 vertical slices within scan region)
        sample_x_offsets = [
            int(scan_width * 0.2),
            int(scan_width * 0.4),
            int(scan_width * 0.6),
            int(scan_width * 0.8)
        ]
        
        # Detect waterline at each sample point
        detections = []
        print(f"\nScanning {len(sample_x_offsets)} sample points:")
        
        for i, x_offset in enumerate(sample_x_offsets, 1):
            result = self._scan_vertical_slice(scan_region, x_offset, scan_height)
            
            if result:
                # Convert to full image coordinates
                result['x_full'] = scan_left + result['x_position']
                result['y_full'] = scan_top + result['y_position']
                detections.append(result)
                
                distance_from_scan_bottom = scan_height - result['y_position']
                print(f"  Point {i}: Waterline at {distance_from_scan_bottom}px from scan bottom "
                      f"(strength: {result['gradient_strength']:.1f})")
            else:
                print(f"  Point {i}: No waterline found (fully wet)")
        
        # Analyze results
        if len(detections) == 0:
            print(f"\n🌊 SNEAKER WAVE ALERT: Wet sand extends beyond entire scan region!")
            result_data = {
                'status': 'SNEAKER_WAVE_ALERT',
                'message': 'Wet sand extends beyond scan region',
                'distance_from_bottom': height - scan_top,
                'confidence': 'high',
                'detection_type': 'out_of_frame',
                'samples_found': 0,
                'total_samples': len(sample_x_offsets)
            }
        else:
            # Calculate median waterline position
            y_values = [d['y_full'] for d in detections]
            median_y = int(np.median(y_values))
            std_y = np.std(y_values)
            
            distance_from_bottom = height - median_y
            
            # Determine confidence
            if std_y > 20:
                confidence = 'low'
                message = f'High variation ({std_y:.1f}px) between sample points'
            elif len(detections) < 3:
                confidence = 'medium'
                message = f'Only {len(detections)}/4 sample points detected'
            else:
                confidence = 'high'
                message = 'Clean detection across all sample points'
            
            print(f"\n✓ Waterline detected:")
            print(f"  Distance from image bottom: {distance_from_bottom}px")
            print(f"  Confidence: {confidence}")
            print(f"  Samples detected: {len(detections)}/{len(sample_x_offsets)}")
            print(f"  Std deviation: {std_y:.1f}px")
            
            result_data = {
                'status': 'detected',
                'distance_from_bottom': distance_from_bottom,
                'wet_line_y': median_y,
                'confidence': confidence,
                'std_deviation': std_y,
                'samples_found': len(detections),
                'total_samples': len(sample_x_offsets),
                'message': message,
                'detection_type': 'normal'
            }
        
        # Create visualization
        viz_img = self._create_visualization(
            img, scan_left, scan_top, scan_right, scan_bottom,
            sample_x_offsets, detections, result_data
        )
        
        # Save visualization
        if output_path is None:
            output_path = Path(image_path).stem + '_analyzed.jpg'
        
        cv2.imwrite(output_path, viz_img)
        print(f"\n✓ Visualization saved: {output_path}")
        print(f"{'='*60}\n")
        
        return result_data
    
    def _scan_vertical_slice(self, region, x_offset, scan_height):
        """
        Scan a vertical slice to find wet/dry sand transition
        """
        column = region[:, x_offset]
        gradient = np.abs(np.diff(column.astype(float)))
        gradient_smooth = gaussian_filter1d(gradient, sigma=3)
        
        search_start = int(len(gradient_smooth) * 0.2)
        search_region = gradient_smooth[search_start:]
        
        if len(search_region) < 5:
            return None
        
        threshold = np.percentile(gradient_smooth, 70)
        
        for i in range(len(search_region) - 1, 0, -1):
            actual_idx = search_start + i
            if gradient_smooth[actual_idx] > threshold:
                return {
                    'y_position': actual_idx,
                    'gradient_strength': gradient_smooth[actual_idx],
                    'x_position': x_offset
                }
        
        bottom_brightness = np.mean(column[-30:])
        
        if bottom_brightness < 80:
            return None
        else:
            return {
                'y_position': len(column) - 5,
                'gradient_strength': 0,
                'x_position': x_offset
            }
    
    def _create_visualization(self, img, scan_left, scan_top, scan_right, 
                             scan_bottom, sample_x_offsets, detections, result_data):
        """
        Create comprehensive visualization
        """
        viz = img.copy()
        height, width = viz.shape[:2]
        
        # Highlight excluded timestamp region in red
        if self.right_margin_px > 0 or self.bottom_margin_px > 0:
            if self.right_margin_px > 0:
                timestamp_overlay = img.copy()
                cv2.rectangle(timestamp_overlay, 
                            (width - self.right_margin_px, 0),
                            (width, height),
                            (0, 0, 255), -1)
                cv2.addWeighted(timestamp_overlay, 0.15, viz, 0.85, 0, viz)
                cv2.line(viz, (width - self.right_margin_px, 0),
                        (width - self.right_margin_px, height),
                        (0, 0, 255), 2)
            
            if self.bottom_margin_px > 0:
                timestamp_overlay = img.copy()
                cv2.rectangle(timestamp_overlay,
                            (0, height - self.bottom_margin_px),
                            (width, height),
                            (0, 0, 255), -1)
                cv2.addWeighted(timestamp_overlay, 0.15, viz, 0.85, 0, viz)
                cv2.line(viz, (0, height - self.bottom_margin_px),
                        (width, height - self.bottom_margin_px),
                        (0, 0, 255), 2)
        
        # Draw scan region
        overlay = img.copy()
        cv2.rectangle(overlay, (scan_left, scan_top), 
                     (scan_right, scan_bottom), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.2, viz, 0.8, 0, viz)
        
        cv2.rectangle(viz, (scan_left, scan_top), 
                     (scan_right, scan_bottom), (255, 100, 0), 3)
        
        # Corner markers
        corner_size = 15
        cv2.line(viz, (scan_left, scan_top), 
                (scan_left + corner_size, scan_top), (255, 100, 0), 4)
        cv2.line(viz, (scan_left, scan_top), 
                (scan_left, scan_top + corner_size), (255, 100, 0), 4)
        cv2.line(viz, (scan_right, scan_top), 
                (scan_right - corner_size, scan_top), (255, 100, 0), 4)
        cv2.line(viz, (scan_right, scan_top), 
                (scan_right, scan_top + corner_size), (255, 100, 0), 4)
        cv2.line(viz, (scan_left, scan_bottom), 
                (scan_left + corner_size, scan_bottom), (255, 100, 0), 4)
        cv2.line(viz, (scan_left, scan_bottom), 
                (scan_left, scan_bottom - corner_size), (255, 100, 0), 4)
        cv2.line(viz, (scan_right, scan_bottom), 
                (scan_right - corner_size, scan_bottom), (255, 100, 0), 4)
        cv2.line(viz, (scan_right, scan_bottom), 
                (scan_right, scan_bottom - corner_size), (255, 100, 0), 4)
        
        # Sample lines
        for x_offset in sample_x_offsets:
            x_full = scan_left + x_offset
            for y in range(scan_top, scan_bottom, 20):
                cv2.line(viz, (x_full, y), (x_full, min(y + 10, scan_bottom)), 
                        (0, 255, 255), 2)
        
        # Detection points
        for i, detection in enumerate(detections, 1):
            x = detection['x_full']
            y = detection['y_full']
            cv2.circle(viz, (x, y), 12, (255, 255, 255), -1)
            cv2.circle(viz, (x, y), 10, (0, 165, 255), -1)
            cv2.circle(viz, (x, y), 12, (0, 0, 0), 2)
            cv2.putText(viz, str(i), (x - 5, y + 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Median waterline
        if result_data['status'] == 'detected':
            median_y = result_data['wet_line_y']
            cv2.line(viz, (scan_left, median_y), (scan_right, median_y), 
                    (0, 255, 0), 4)
            cv2.circle(viz, (scan_left, median_y), 8, (0, 255, 0), -1)
            cv2.circle(viz, (scan_right, median_y), 8, (0, 255, 0), -1)
            
            line_x = scan_left + 30
            cv2.arrowedLine(viz, (line_x, median_y), (line_x, height - 10),
                           (255, 0, 255), 3, tipLength=0.05)
            
            distance = result_data['distance_from_bottom']
            label = f"{distance}px"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            label_y = (median_y + height) // 2
            cv2.rectangle(viz, 
                         (line_x - text_width//2 - 5, label_y - text_height - 5),
                         (line_x + text_width//2 + 5, label_y + 5),
                         (0, 0, 0), -1)
            cv2.putText(viz, label, 
                       (line_x - text_width//2, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        # Information panel
        panel_height = 140
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        y_offset = 30
        status = result_data['status']
        if status == 'SNEAKER_WAVE_ALERT':
            title = "SNEAKER WAVE ALERT!"
            title_color = (0, 0, 255)
        else:
            title = "Waterline Detected"
            title_color = (0, 255, 0)
        
        cv2.putText(panel, title, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, title_color, 2)
        
        y_offset += 35
        cv2.putText(panel, f"Distance from bottom: {result_data['distance_from_bottom']}px", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(panel, f"Confidence: {result_data['confidence'].upper()}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_offset += 25
        samples_text = f"Samples: {result_data['samples_found']}/{result_data['total_samples']}"
        cv2.putText(panel, samples_text, 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_offset += 25
        scan_info = f"Scan: {scan_right-scan_left}x{scan_bottom-scan_top}px @ ({scan_left},{scan_top})"
        cv2.putText(panel, scan_info, 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Legend
        legend_x = width - 400
        legend_y = 25
        
        cv2.rectangle(panel, (legend_x, legend_y - 10), 
                     (legend_x + 20, legend_y + 5), (0, 0, 255), -1)
        cv2.putText(panel, "Excluded (timestamp)", (legend_x + 30, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        legend_y += 25
        cv2.rectangle(panel, (legend_x, legend_y - 10), 
                     (legend_x + 20, legend_y + 5), (255, 100, 0), 2)
        cv2.putText(panel, "Scan Region", (legend_x + 30, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        legend_y += 25
        cv2.line(panel, (legend_x, legend_y - 5), (legend_x + 20, legend_y - 5),
                (0, 255, 255), 2)
        cv2.putText(panel, "Sample Lines", (legend_x + 30, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        legend_y += 25
        cv2.circle(panel, (legend_x + 10, legend_y - 5), 8, (0, 165, 255), -1)
        cv2.putText(panel, "Detection Points", (legend_x + 30, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        legend_y += 25
        cv2.line(panel, (legend_x, legend_y - 5), (legend_x + 20, legend_y - 5),
                (0, 255, 0), 3)
        cv2.putText(panel, "Waterline", (legend_x + 30, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        final_viz = np.vstack([panel, viz])
        return final_viz
