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