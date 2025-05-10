import cv2
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import sys
import time
import math
import mediapipe as mp

class IronManARHandTracker:
    def __init__(self):
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Could not open webcam")
            sys.exit()
            
        # Get webcam dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("Iron Man AR Interface - Press ESC to quit")
        self.display = (800, 600)
        self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Set up OpenGL
        self.setup_opengl()
        
        # Initialize 3D objects
        self.init_objects()
        
        # Start time for animation
        self.start_time = time.time()
        
        # Hand position tracking
        self.hand_positions = []
        self.hand_gestures = []
        self.is_grabbing = False
        
        # For hybrid rendering
        self.clock = pygame.time.Clock()
        
        # Iron Man HUD elements
        self.hud_font = pygame.font.SysFont('Arial', 16)
        self.hud_color = (0, 164, 237)  # Iron Man blue
        self.highlight_color = (255, 140, 0)  # Orange highlight
        
        # FIX: We'll use a different approach for the webcam background instead of OpenGL textures
        
    def setup_opengl(self):
        """Set up OpenGL rendering"""
        # Set up perspective
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 50.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5)
        
        # Enable depth test
        glEnable(GL_DEPTH_TEST)
        
        # Set up lighting for holographic effect
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Light position and properties for holographic look
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 2, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.1, 0.1, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.3, 0.6, 0.8, 1])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.5, 1.0, 1])
        
        # Enable alpha blending for transparent effects
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
    def init_objects(self):
        """Initialize the 3D objects"""
        # Define cube vertices
        self.vertices = (
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, -1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
            (-1, -1, 1)
        )
        
        # Define cube edges
        self.edges = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        )
        
        # Define cube surfaces
        self.surfaces = (
            (0, 1, 2, 3),  # back
            (3, 2, 6, 7),  # left
            (7, 6, 5, 4),  # front
            (4, 5, 1, 0),  # right
            (1, 5, 6, 2),  # top
            (4, 0, 3, 7)   # bottom
        )
        
        # Define Iron Man holographic colors with alpha (transparency)
        self.colors = (
            (0, 0.64, 0.93, 0.4),  # Blue back
            (0, 0.64, 0.93, 0.4),  # Blue left
            (0, 0.64, 0.93, 0.4),  # Blue front
            (0, 0.64, 0.93, 0.4),  # Blue right
            (0, 0.64, 0.93, 0.4),  # Blue top
            (0, 0.64, 0.93, 0.4)   # Blue bottom
        )
        
        # Edge color (iron man highlight)
        self.edge_color = (1, 0.55, 0, 0.8)  # Orange with alpha
        
        # Object position and rotation
        self.obj_position = [0, 0, 0]
        self.obj_rotation = [0, 0, 0]
        self.obj_scale = 0.5
    
    def process_frame_to_surface(self, frame):
        """Convert OpenCV frame to a Pygame surface"""
        # Flip the image horizontally (mirror effect)
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB for proper color display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands with MediaPipe
        frame_with_landmarks = self.process_hands(frame_rgb)
        
        # Resize to match the display dimensions
        frame_resized = cv2.resize(frame_with_landmarks, self.display)
        
        # Create a pygame surface from the numpy array
        pygame_surface = pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1))
        
        return pygame_surface
    
    def draw_background(self, surface):
        """Draw the webcam feed as a background"""
        # Save the current matrices
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display[0], self.display[1], 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth test and lighting for background
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Draw the background (using pygame's native drawing)
        self.screen.blit(surface, (0, 0))
        
        # Restore matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        # Re-enable depth test and lighting for 3D rendering
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def draw_holographic_cube(self, position, rotation, scale=1.0):
        """Draw a holographic cube with Iron Man style"""
        glPushMatrix()
        
        # Position and scale
        glTranslatef(position[0], position[1], position[2])
        
        # Apply rotation
        glRotatef(rotation[0], 1, 0, 0)  # X-axis
        glRotatef(rotation[1], 0, 1, 0)  # Y-axis
        glRotatef(rotation[2], 0, 0, 1)  # Z-axis
        
        # Apply scaling
        glScalef(scale, scale, scale)
        
        # Draw translucent faces first
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Draw each face with Iron Man blue tint
        glBegin(GL_QUADS)
        for i, surface in enumerate(self.surfaces):
            glColor4fv(self.colors[i])
            for vertex in surface:
                glVertex3fv(self.vertices[vertex])
        glEnd()
        
        # Draw edges with orange highlight
        glLineWidth(2.0)
        glColor4fv(self.edge_color)
        glBegin(GL_LINES)
        for edge in self.edges:
            for vertex in edge:
                glVertex3fv(self.vertices[vertex])
        glEnd()
        
        # Add a subtle glow effect (additional slightly larger transparent cube)
        glPushMatrix()
        glScalef(1.05, 1.05, 1.05)
        glColor4f(0, 0.64, 0.93, 0.15)  # Very transparent blue
        glBegin(GL_QUADS)
        for surface in self.surfaces:
            for vertex in surface:
                glVertex3fv(self.vertices[vertex])
        glEnd()
        glPopMatrix()
        
        # Disable blending
        glDisable(GL_BLEND)
        
        glPopMatrix()
    
    def draw_hud_elements(self):
        """Draw Iron Man HUD elements"""
        # Switch to 2D ortho projection for HUD elements
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display[0], self.display[1], 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth test and lighting for HUD
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Create text surfaces
        hud_title = self.hud_font.render("JARVIS AR INTERFACE v1.0.4", True, self.hud_color)
        
        # Show hand tracking status
        status = "HAND DETECTED: YES" if self.hand_positions else "HAND DETECTED: NO"
        status_color = self.highlight_color if self.hand_positions else self.hud_color
        status_text = self.hud_font.render(status, True, status_color)
        
        # Show grabbing status
        if self.is_grabbing:
            grab_text = self.hud_font.render("OBJECT CONTROL: ACTIVE", True, self.highlight_color)
            self.screen.blit(grab_text, (20, 60))
        
        # Show coordinates if hand is detected
        if self.hand_positions:
            pos = self.hand_positions[0]
            coord_text = f"X: {pos[0]:.2f} Y: {pos[1]:.2f} Z: {pos[2]:.2f}"
            coord_surface = self.hud_font.render(coord_text, True, self.hud_color)
            self.screen.blit(coord_surface, (20, 80))
        
        # Show FPS
        fps = int(self.clock.get_fps())
        fps_text = self.hud_font.render(f"FPS: {fps}", True, self.hud_color)
        
        # Blit text to screen
        self.screen.blit(hud_title, (20, 20))
        self.screen.blit(status_text, (20, 40))
        self.screen.blit(fps_text, (self.display[0] - 80, 20))
        
        # Draw some decorative HUD elements (Iron Man style)
        # Top-right corner elements
        pygame.draw.line(self.screen, self.hud_color, (self.display[0] - 100, 10), (self.display[0] - 10, 10), 1)
        pygame.draw.line(self.screen, self.hud_color, (self.display[0] - 10, 10), (self.display[0] - 10, 50), 1)
        # Bottom left corner bracket
        pygame.draw.line(self.screen, self.hud_color, (10, self.display[1] - 10), (10, self.display[1] - 50), 1)
        pygame.draw.line(self.screen, self.hud_color, (10, self.display[1] - 10), (100, self.display[1] - 10), 1)
        
        # Restore matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        # Re-enable depth test and lighting for 3D
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def process_hands(self, frame):
        """Process hand landmarks using MediaPipe"""
        # Process the image to find hands
        results = self.hands.process(frame)
        
        # Reset hand positions
        self.hand_positions = []
        self.hand_gestures = []
        
        # Check if hands were found
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the 3D coordinates of the hand landmarks
                landmarks_3d = []
                for landmark in hand_landmarks.landmark:
                    x = landmark.x * self.width
                    y = landmark.y * self.height
                    z = landmark.z * self.width  # Scale z to match x scale
                    landmarks_3d.append((x, y, z))
                
                # Store hand position (using index finger tip)
                index_finger_pos = landmarks_3d[8]  # Index finger tip
                thumb_pos = landmarks_3d[4]  # Thumb tip
                
                # Convert to normalized coordinates (-1 to 1)
                norm_x = (index_finger_pos[0] / self.width) * 2 - 1
                norm_y = -((index_finger_pos[1] / self.height) * 2 - 1)  # Flip y-axis
                norm_z = index_finger_pos[2] / self.width
                
                # Store hand position
                self.hand_positions.append((norm_x * 3, norm_y * 3, norm_z))
                
                # Detect grabbing gesture (thumb and index finger close together)
                distance = np.sqrt(
                    (thumb_pos[0] - index_finger_pos[0])**2 + 
                    (thumb_pos[1] - index_finger_pos[1])**2 + 
                    (thumb_pos[2] - index_finger_pos[2])**2
                )
                
                # If distance is less than a threshold, consider it a grabbing gesture
                is_grabbing = distance < 50  # Adjust threshold as needed
                self.hand_gestures.append(is_grabbing)
                
                # Draw hand landmarks with Iron Man style
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return frame
    
    def update_object_position(self):
        """Update object position based on hand tracking"""
        if len(self.hand_positions) > 0:
            # Get the first hand position
            hand_pos = self.hand_positions[0]
            
            # If grabbing, move the object
            if len(self.hand_gestures) > 0 and self.hand_gestures[0]:
                self.obj_position = [hand_pos[0], hand_pos[1], hand_pos[2]]
                self.is_grabbing = True
            else:
                self.is_grabbing = False
    
    def run(self):
        """Main loop"""
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            try:
                # Capture webcam frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame from webcam")
                    continue
                
                # Update the webcam texture as background
                # FIX: Changed from update_background_texture to process_frame_to_surface
                pygame_surface = self.process_frame_to_surface(frame)
                
                # Update object position based on hand tracking
                self.update_object_position()
                
                # Clear the screen
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                
                # Draw the webcam feed as background
                self.draw_background(pygame_surface)
                
                # Set up 3D projection for objects
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 50.0)
                
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                glTranslatef(0.0, 0.0, -5)
                
                # Calculate time-based rotation
                current_time = time.time() - self.start_time
                rotation_speed = 20
                
                if not self.is_grabbing:
                    # Only auto-rotate when not being grabbed
                    self.obj_rotation = [
                        (current_time * rotation_speed) % 360,
                        (current_time * rotation_speed * 0.7) % 360,
                        (current_time * rotation_speed * 0.5) % 360
                    ]
                
                # Clear only the depth buffer to draw 3D objects
                glClear(GL_DEPTH_BUFFER_BIT)
                
                # Draw the holographic cube
                self.draw_holographic_cube(self.obj_position, self.obj_rotation, self.obj_scale)
                
                # Draw Iron Man HUD elements
                self.draw_hud_elements()
                
                # Update the display
                pygame.display.flip()
                
                # Control frame rate
                self.clock.tick(60)
                
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Release resources
        self.cap.release()
        self.hands.close()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    try:
        app = IronManARHandTracker()
        app.run()
    except Exception as e:
        print(f"Application error: {str(e)}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)