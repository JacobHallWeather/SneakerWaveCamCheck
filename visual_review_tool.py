import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

class VisualReviewTool:
    """
    Creates visual review images for human analysis of sneaker waves
    Combines kymograph (time-slice) and grid comparison views
    """
    
    def __init__(self, scan_line_angle_deg=25, scan_position=0.6):
        """
        Args:
            scan_line_angle_deg: Angle of scan line following tree line (degrees from horizontal)
            scan_position: Vertical position of scan line (0.0=top, 1.0=bottom)
        """
        self.scan_angle = np.deg2rad(scan_line_angle_deg)
        self.scan_position = scan_position
        
    def create_kymograph(self, image_folder, output_path='kymograph.jpg'):
        """
        Creates a kymograph (time-slice visualization)
        Each vertical line is one time point, shows wet sand progression over time
        """
        print("\n" + "="*60)
        print("Creating Kymograph (Time-Slice Visualization)")
        print("="*60)
        
        # Get all images sorted by name (assumes timestamp in filename)
        images = sorted(Path(image_folder).glob('*.jpg'))
        
        if len(images) == 0:
            print(f"ERROR: No images found in {image_folder}")
            return None
            
        print(f"Found {len(images)} images")
        
        # Read first image to get dimensions
        first_img = cv2.imread(str(images[0]))
        height, width = first_img.shape[:2]
        
        # Define scan line (angled to follow tree line)
        y_pos = int(height * self.scan_position)
        
        # Calculate angled scan line endpoints
        # Start from left, angle downward to right
        x1, y1 = 0, y_pos
        x2 = width
        y2 = int(y_pos + width * np.tan(self.scan_angle))
        
        # Extract scan line from each image
        scan_lines = []
        valid_images = []
        
        for i, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  Skipping {img_path.name} (failed to load)")
                continue
                
            # Extract pixels along the angled scan line
            num_points = width
            x_coords = np.linspace(x1, x2, num_points).astype(int)
            y_coords = np.linspace(y1, y2, num_points).astype(int)
            
            # Clip to image bounds
            valid_mask = (y_coords >= 0) & (y_coords < height)
            x_coords = x_coords[valid_mask]
            y_coords = y_coords[valid_mask]
            
            # Extract pixel values along line
            scan_line = img[y_coords, x_coords]
            scan_lines.append(scan_line)
            valid_images.append(img_path.name)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(images)} images...")
        
        # Stack scan lines horizontally to create kymograph
        kymo = np.hstack(scan_lines)
        
        # Enhance contrast for better visibility
        kymo_enhanced = cv2.convertScaleAbs(kymo, alpha=1.5, beta=20)
        
        # Create visualization with labels
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
        
        # Original kymograph
        ax1.imshow(cv2.cvtColor(kymo, cv2.COLOR_BGR2RGB))
        ax1.set_title('Kymograph: Time-Series of Beach Scan Line (Original)', fontsize=16)
        ax1.set_xlabel('Time (each column = one image) →', fontsize=12)
        ax1.set_ylabel('← Left Beach | Right Beach →', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Enhanced kymograph
        ax2.imshow(cv2.cvtColor(kymo_enhanced, cv2.COLOR_BGR2RGB))
        ax2.set_title('Kymograph: Enhanced Contrast (Wet Sand Detection)', fontsize=16)
        ax2.set_xlabel('Time (each column = one image) →', fontsize=12)
        ax2.set_ylabel('← Left Beach | Right Beach →', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add time markers every N images
        marker_interval = max(1, len(scan_lines) // 10)
        for i in range(0, len(scan_lines), marker_interval):
            if i < len(valid_images):
                # Extract timestamp from filename if possible
                timestamp = valid_images[i]
                ax1.axvline(x=i, color='yellow', alpha=0.5, linewidth=0.5)
                ax2.axvline(x=i, color='yellow', alpha=0.5, linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Kymograph saved: {output_path}")
        print(f"  Dimensions: {kymo.shape[1]} time points × {kymo.shape[0]} pixels")
        return output_path
    
    def create_grid_comparison(self, image_folder, images_per_row=6, 
                              max_images=24, output_path='grid_comparison.jpg'):
        """
        Creates a grid of images with scan line overlay for visual comparison
        """
        print("\n" + "="*60)
        print("Creating Grid Comparison View")
        print("="*60)
        
        images = sorted(Path(image_folder).glob('*.jpg'))[:max_images]
        
        if len(images) == 0:
            print(f"ERROR: No images found in {image_folder}")
            return None
            
        print(f"Creating grid with {len(images)} images")
        
        # Calculate grid dimensions
        num_rows = int(np.ceil(len(images) / images_per_row))
        
        # Read first image to get dimensions
        first_img = cv2.imread(str(images[0]))
        height, width = first_img.shape[:2]
        
        # Resize for grid (smaller for overview)
        thumb_width = 400
        thumb_height = int(height * (thumb_width / width))
        
        # Define scan line
        y_pos = int(thumb_height * self.scan_position)
        x1, y1 = 0, y_pos
        x2 = thumb_width
        y2 = int(y_pos + thumb_width * np.tan(self.scan_angle))
        
        # Create grid
        grid_rows = []
        
        for row in range(num_rows):
            row_images = []
            
            for col in range(images_per_row):
                idx = row * images_per_row + col
                
                if idx >= len(images):
                    # Fill with blank if not enough images
                    blank = np.ones((thumb_height, thumb_width, 3), dtype=np.uint8) * 50
                    row_images.append(blank)
                    continue
                
                # Load and resize image
                img = cv2.imread(str(images[idx]))
                img_resized = cv2.resize(img, (thumb_width, thumb_height))
                
                # Draw scan line on image
                cv2.line(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add timestamp label
                timestamp = images[idx].stem
                cv2.rectangle(img_resized, (5, 5), (200, 30), (0, 0, 0), -1)
                cv2.putText(img_resized, timestamp, (10, 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                row_images.append(img_resized)
            
            # Concatenate row
            row_concat = np.hstack(row_images)
            grid_rows.append(row_concat)
        
        # Concatenate all rows
        grid = np.vstack(grid_rows)
        
        # Add title
        title_height = 60
        title_bar = np.ones((title_height, grid.shape[1], 3), dtype=np.uint8) * 40
        cv2.putText(title_bar, 
                   f"Grid Comparison: {len(images)} Images | Green Line = Scan Region",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        final_grid = np.vstack([title_bar, grid])
        
        # Save
        cv2.imwrite(output_path, final_grid)
        
        print(f"✓ Grid comparison saved: {output_path}")
        print(f"  Grid size: {num_rows} rows × {images_per_row} columns")
        return output_path
    
    def generate_all_visualizations(self, image_folder, output_folder='visualizations'):
        """
        Generate all visualization types
        """
        # Create output folder
        Path(output_folder).mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("GENERATING ALL VISUALIZATIONS")
        print("="*60)
        
        # 1. Kymograph
        kymo_path = Path(output_folder) / 'kymograph.jpg'
        self.create_kymograph(image_folder, str(kymo_path))
        
        # 2. Grid comparison
        grid_path = Path(output_folder) / 'grid_comparison.jpg'
        self.create_grid_comparison(image_folder, output_path=str(grid_path))
        
        print("\n" + "="*60)
        print("ALL VISUALIZATIONS COMPLETE!")
        print("="*60)
        print(f"Output folder: {output_folder}/")
        print(f"  - {kymo_path.name}")
        print(f"  - {grid_path.name}")
        
        return {
            'kymograph': str(kymo_path),
            'grid': str(grid_path)
        }


if __name__ == "__main__":
    # Example usage
    tool = VisualReviewTool(scan_line_angle_deg=25, scan_position=0.6)
    
    # Generate all visualizations from collected images
    results = tool.generate_all_visualizations(
        image_folder='test_images',
        output_folder='visualizations'
    )
    
    print("\nDone! Open the visualization folder to review images.")