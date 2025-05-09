import cv2
import numpy as np
import math
import time

class SimpleAR:
    def __init__(self):
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Could not open webcam")
            return
        
        # Get webcam dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create a cube vertices (8 corners)
        self.cube_vertices = np.array([
            [-1, -1, -1],  # 0: back-bottom-left
            [1, -1, -1],   # 1: back-bottom-right
            [1, 1, -1],    # 2: back-top-right
            [-1, 1, -1],   # 3: back-top-left
            [-1, -1, 1],   # 4: front-bottom-left
            [1, -1, 1],    # 5: front-bottom-right
            [1, 1, 1],     # 6: front-top-right
            [-1, 1, 1]     # 7: front-top-left
        ], dtype=float)
        
        # Define cube edges as pairs of vertex indices
        self.cube_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # back face
            (4, 5), (5, 6), (6, 7), (7, 4),  # front face
            (0, 4), (1, 5), (2, 6), (3, 7)   # connecting edges
        ]
        
        # Create a pyramid vertices (5 points)
        self.pyramid_vertices = np.array([
            [0, 0, 1],      # 0: apex
            [-1, -1, -1],   # 1: base front-left
            [1, -1, -1],    # 2: base front-right
            [1, 1, -1],     # 3: base back-right
            [-1, 1, -1]     # 4: base back-left
        ], dtype=float)
        
        # Define pyramid edges
        self.pyramid_edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),  # edges from apex to base
            (1, 2), (2, 3), (3, 4), (4, 1)   # base edges
        ]
        
        # Create sphere approximation (we'll use a geodesic sphere)
        self.sphere_vertices, self.sphere_edges = self.create_geodesic_sphere(2)
        
        # Scale and initial position for our objects
        self.cube_scale = 40
        self.pyramid_scale = 30
        self.sphere_scale = 35
        
        # Starting rotation angles
        self.angle_x = 0
        self.angle_y = 0
        self.angle_z = 0
        
        # Start time for animation
        self.start_time = time.time()
        
        # Colors in BGR format
        self.cube_color = (255, 0, 0)      # Blue
        self.pyramid_color = (0, 255, 0)   # Green
        self.sphere_color = (0, 0, 255)    # Red
    
    def create_geodesic_sphere(self, subdivisions=1):
        """Create a geodesic sphere by subdividing an icosahedron"""
        # Start with 12 vertices of an icosahedron
        t = (1.0 + math.sqrt(5.0)) / 2.0
        vertices = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
        ], dtype=float)
        
        # Normalize to make all vertices lie on a unit sphere
        for i in range(len(vertices)):
            norm = np.linalg.norm(vertices[i])
            vertices[i] = vertices[i] / norm
        
        # Define the 30 edges of an icosahedron
        edges = [
            (0, 5), (0, 11), (0, 1), (0, 7), (0, 10),
            (1, 5), (1, 9), (1, 8), (1, 7),
            (2, 3), (2, 4), (2, 6), (2, 10), (2, 11),
            (3, 4), (3, 6), (3, 8), (3, 9),
            (4, 5), (4, 9), (4, 11),
            (5, 9), (5, 11),
            (6, 7), (6, 8), (6, 10),
            (7, 8), (7, 10),
            (8, 9),
            (10, 11)
        ]
        
        return vertices, edges
    
    def project_point(self, point, angle_x, angle_y, angle_z, scale, offset_x, offset_y):
        """Project a 3D point onto the 2D screen with rotation"""
        # Apply rotation around x-axis
        x, y, z = point
        y_rot = y * math.cos(angle_x) - z * math.sin(angle_x)
        z_rot = y * math.sin(angle_x) + z * math.cos(angle_x)
        x, y, z = x, y_rot, z_rot
        
        # Apply rotation around y-axis
        x_rot = x * math.cos(angle_y) + z * math.sin(angle_y)
        z_rot = -x * math.sin(angle_y) + z * math.cos(angle_y)
        x, y, z = x_rot, y, z_rot
        
        # Apply rotation around z-axis
        x_rot = x * math.cos(angle_z) - y * math.sin(angle_z)
        y_rot = x * math.sin(angle_z) + y * math.cos(angle_z)
        x, y, z = x_rot, y_rot, z
        
        # Add a z-offset to move objects in front of the camera
        z += 3
        
        # Perspective projection
        f = 200  # focal length
        if z > 0:
            x_proj = int(f * x / z * scale + offset_x)
            y_proj = int(f * y / z * scale + offset_y)
            return (x_proj, y_proj), 1/z  # Return depth for z-sorting
        else:
            return None, 0
    
    def draw_3d_object(self, frame, vertices, edges, angle_x, angle_y, angle_z, scale, offset_x, offset_y, color, thickness=2):
        """Draw a 3D object on the frame"""
        # Project all vertices
        projected_points = []
        for v in vertices:
            proj, depth = self.project_point(v, angle_x, angle_y, angle_z, scale, offset_x, offset_y)
            if proj:
                projected_points.append((proj, depth))
            else:
                projected_points.append((None, 0))
        
        # Draw all edges
        for e in edges:
            if projected_points[e[0]][0] and projected_points[e[1]][0]:
                # Get the points
                p1 = projected_points[e[0]][0]
                p2 = projected_points[e[1]][0]
                
                # Get average depth for edge coloring (closer is brighter)
                depth = (projected_points[e[0]][1] + projected_points[e[1]][1]) / 2
                brightness = min(255, int(200 * depth + 55))  # Scale and clamp brightness
                
                # Create color with brightness adjustment
                edge_color = tuple(min(255, int(c * brightness / 255)) for c in color)
                
                # Draw the edge
                cv2.line(frame, p1, p2, edge_color, thickness)
    
    def run(self):
        while True:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Get frame center
            center_x = self.width // 2
            center_y = self.height // 2
            
            # Calculate time-based rotation and orbital positions
            current_time = time.time() - self.start_time
            
            # Rotation angles (continuously increasing)
            self.angle_x = current_time * 0.7
            self.angle_y = current_time * 1.3
            self.angle_z = current_time * 0.5
            
            # Orbital positioning (objects orbit around center)
            orbit_radius = 150
            orbit_speed = 0.5
            
            # Cube position
            cube_orbit = current_time * orbit_speed
            cube_x = center_x + int(orbit_radius * math.cos(cube_orbit))
            cube_y = center_y + int(orbit_radius * 0.5 * math.sin(cube_orbit))  # Flatten orbit vertically
            
            # Pyramid position - offset by 120 degrees
            pyramid_orbit = current_time * orbit_speed + (2 * math.pi / 3)
            pyramid_x = center_x + int(orbit_radius * math.cos(pyramid_orbit))
            pyramid_y = center_y + int(orbit_radius * 0.5 * math.sin(pyramid_orbit))
            
            # Sphere position - offset by 240 degrees
            sphere_orbit = current_time * orbit_speed + (4 * math.pi / 3)
            sphere_x = center_x + int(orbit_radius * math.cos(sphere_orbit))
            sphere_y = center_y + int(orbit_radius * 0.5 * math.sin(sphere_orbit))
            
            # Draw objects
            self.draw_3d_object(frame, self.cube_vertices, self.cube_edges, 
                               self.angle_x, self.angle_y, self.angle_z, 
                               self.cube_scale, cube_x, cube_y, self.cube_color)
            
            self.draw_3d_object(frame, self.pyramid_vertices, self.pyramid_edges, 
                               self.angle_x + 0.5, self.angle_y, self.angle_z, 
                               self.pyramid_scale, pyramid_x, pyramid_y, self.pyramid_color)
            
            self.draw_3d_object(frame, self.sphere_vertices, self.sphere_edges, 
                               self.angle_x, self.angle_y + 0.3, self.angle_z, 
                               self.sphere_scale, sphere_x, sphere_y, self.sphere_color)
            
            # Add text
            cv2.putText(frame, "OpenCV 3D AR - Press 'Q' to quit", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # Display the frame
            cv2.imshow('Augmented Reality', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ar_app = SimpleAR()
    ar_app.run()