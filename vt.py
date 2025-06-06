"""
3D Object Creation and Manipulation using ModernGL
Fixed version with proper vector operations and error handling
"""

import moderngl
import pygame
import numpy as np
import math
from pyrr import Matrix44

class Vector3:
    """Simple Vector3 class to replace pyrr Vector3 with proper operations"""
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)
    
    def __add__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.data + other.data)
        return Vector3(self.data + other)
    
    def __sub__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.data - other.data)
        return Vector3(self.data - other)
    
    def __mul__(self, scalar):
        return Vector3(self.data * scalar)
    
    def __rmul__(self, scalar):
        return Vector3(self.data * scalar)
    
    def cross(self, other):
        return Vector3(np.cross(self.data, other.data))
    
    def normalize(self):
        norm = np.linalg.norm(self.data)
        if norm == 0:
            return Vector3([0, 0, 0])
        return Vector3(self.data / norm)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __setitem__(self, index, value):
        self.data[index] = value

class GL3DRenderer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.running = True
        self.time = 0.0
        
        # Initialize Pygame and OpenGL context
        pygame.init()
        pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("3D Object Manipulation")
        
        # Create ModernGL context
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        
        # Clock for timing
        self.clock = pygame.time.Clock()
        
        # Initialize 3D objects
        self.init_shaders()
        self.create_cube()
        self.create_pyramid()
        self.setup_camera()
        
        print("3D Renderer initialized successfully!")
        print("Controls:")
        print("- Mouse: Rotate camera")
        print("- WASD: Move camera")
        print("- Q/E: Move up/down")
        print("- ESC: Exit")
    
    def init_shaders(self):
        """Initialize vertex and fragment shaders"""
        
        # Vertex shader with MVP matrices
        vertex_shader = """
        #version 330 core
        
        in vec3 in_position;
        in vec3 in_color;
        in vec3 in_normal;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform vec3 light_pos;
        
        out vec3 frag_color;
        out vec3 frag_normal;
        out vec3 frag_pos;
        out vec3 light_direction;
        
        void main() {
            vec4 world_pos = model * vec4(in_position, 1.0);
            gl_Position = projection * view * world_pos;
            
            frag_color = in_color;
            frag_normal = mat3(transpose(inverse(model))) * in_normal;
            frag_pos = world_pos.xyz;
            light_direction = normalize(light_pos - frag_pos);
        }
        """
        
        # Fragment shader with basic lighting
        fragment_shader = """
        #version 330 core
        
        in vec3 frag_color;
        in vec3 frag_normal;
        in vec3 frag_pos;
        in vec3 light_direction;
        
        out vec4 out_color;
        
        void main() {
            // Basic diffuse lighting
            vec3 normal = normalize(frag_normal);
            float diff = max(dot(normal, light_direction), 0.0);
            
            // Ambient + diffuse lighting
            vec3 ambient = 0.3 * frag_color;
            vec3 diffuse = diff * frag_color;
            
            out_color = vec4(ambient + diffuse, 1.0);
        }
        """
        
        # Create shader program
        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
    
    def create_cube(self):
        """Create a 3D cube with vertices, colors, and normals"""
        
        # Cube vertices (position, color, normal)
        vertices = np.array([
            # Front face (red)
            [-1, -1,  1,  1, 0, 0,  0, 0, 1],
            [ 1, -1,  1,  1, 0, 0,  0, 0, 1],
            [ 1,  1,  1,  1, 0, 0,  0, 0, 1],
            [-1,  1,  1,  1, 0, 0,  0, 0, 1],
            
            # Back face (green)
            [-1, -1, -1,  0, 1, 0,  0, 0, -1],
            [ 1, -1, -1,  0, 1, 0,  0, 0, -1],
            [ 1,  1, -1,  0, 1, 0,  0, 0, -1],
            [-1,  1, -1,  0, 1, 0,  0, 0, -1],
            
            # Left face (blue)
            [-1, -1, -1,  0, 0, 1,  -1, 0, 0],
            [-1, -1,  1,  0, 0, 1,  -1, 0, 0],
            [-1,  1,  1,  0, 0, 1,  -1, 0, 0],
            [-1,  1, -1,  0, 0, 1,  -1, 0, 0],
            
            # Right face (yellow)
            [ 1, -1, -1,  1, 1, 0,  1, 0, 0],
            [ 1, -1,  1,  1, 1, 0,  1, 0, 0],
            [ 1,  1,  1,  1, 1, 0,  1, 0, 0],
            [ 1,  1, -1,  1, 1, 0,  1, 0, 0],
            
            # Top face (cyan)
            [-1,  1, -1,  0, 1, 1,  0, 1, 0],
            [ 1,  1, -1,  0, 1, 1,  0, 1, 0],
            [ 1,  1,  1,  0, 1, 1,  0, 1, 0],
            [-1,  1,  1,  0, 1, 1,  0, 1, 0],
            
            # Bottom face (magenta)
            [-1, -1, -1,  1, 0, 1,  0, -1, 0],
            [ 1, -1, -1,  1, 0, 1,  0, -1, 0],
            [ 1, -1,  1,  1, 0, 1,  0, -1, 0],
            [-1, -1,  1,  1, 0, 1,  0, -1, 0],
        ], dtype=np.float32)
        
        # Cube indices for triangles
        indices = np.array([
            # Front face
            0, 1, 2,  0, 2, 3,
            # Back face
            4, 6, 5,  4, 7, 6,
            # Left face
            8, 9, 10,  8, 10, 11,
            # Right face
            12, 14, 13,  12, 15, 14,
            # Top face
            16, 17, 18,  16, 18, 19,
            # Bottom face
            20, 22, 21,  20, 23, 22,
        ], dtype=np.uint32)
        
        # Create vertex buffer and vertex array
        self.cube_vbo = self.ctx.buffer(vertices.tobytes())
        self.cube_ibo = self.ctx.buffer(indices.tobytes())
        
        self.cube_vao = self.ctx.vertex_array(
            self.program,
            [(self.cube_vbo, '3f 3f 3f', 'in_position', 'in_color', 'in_normal')],
            self.cube_ibo
        )
        
        self.cube_indices_count = len(indices)
    
    def create_pyramid(self):
        """Create a 3D pyramid"""
        
        # Pyramid vertices
        vertices = np.array([
            # Base (square)
            [-1, -1, -1,  1, 0.5, 0,  0, -1, 0],  # 0
            [ 1, -1, -1,  1, 0.5, 0,  0, -1, 0],  # 1
            [ 1, -1,  1,  1, 0.5, 0,  0, -1, 0],  # 2
            [-1, -1,  1,  1, 0.5, 0,  0, -1, 0],  # 3
            
            # Apex
            [ 0,  1,  0,  0, 1, 1,  0, 1, 0],     # 4
        ], dtype=np.float32)
        
        indices = np.array([
            # Base
            0, 1, 2,  0, 2, 3,
            # Sides
            0, 4, 1,  1, 4, 2,  2, 4, 3,  3, 4, 0,
        ], dtype=np.uint32)
        
        self.pyramid_vbo = self.ctx.buffer(vertices.tobytes())
        self.pyramid_ibo = self.ctx.buffer(indices.tobytes())
        
        self.pyramid_vao = self.ctx.vertex_array(
            self.program,
            [(self.pyramid_vbo, '3f 3f 3f', 'in_position', 'in_color', 'in_normal')],
            self.pyramid_ibo
        )
        
        self.pyramid_indices_count = len(indices)
    
    def setup_camera(self):
        """Setup camera and projection matrices"""
        self.camera_pos = Vector3([0.0, 0.0, 5.0])
        self.camera_front = Vector3([0.0, 0.0, -1.0])
        self.camera_up = Vector3([0.0, 1.0, 0.0])
        
        self.yaw = -90.0
        self.pitch = 0.0
        self.last_x = self.width // 2
        self.last_y = self.height // 2
        self.first_mouse = True
        
        # Projection matrix
        self.projection = Matrix44.perspective_projection(
            45.0, self.width / self.height, 0.1, 100.0
        )
    
    def process_input(self):
        """Handle keyboard and mouse input"""
        keys = pygame.key.get_pressed()
        
        camera_speed = 0.1
        if keys[pygame.K_w]:
            self.camera_pos = self.camera_pos + (self.camera_front * camera_speed)
        if keys[pygame.K_s]:
            self.camera_pos = self.camera_pos - (self.camera_front * camera_speed)
        if keys[pygame.K_a]:
            right = self.camera_front.cross(self.camera_up).normalize()
            self.camera_pos = self.camera_pos - (right * camera_speed)
        if keys[pygame.K_d]:
            right = self.camera_front.cross(self.camera_up).normalize()
            self.camera_pos = self.camera_pos + (right * camera_speed)
        if keys[pygame.K_q]:
            self.camera_pos = self.camera_pos - (self.camera_up * camera_speed)
        if keys[pygame.K_e]:
            self.camera_pos = self.camera_pos + (self.camera_up * camera_speed)
        
        # Mouse look
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        if self.first_mouse:
            self.last_x = mouse_x
            self.last_y = mouse_y
            self.first_mouse = False
        
        x_offset = mouse_x - self.last_x
        y_offset = self.last_y - mouse_y  # Reversed since y-coordinates go from bottom to top
        self.last_x = mouse_x
        self.last_y = mouse_y
        
        sensitivity = 0.1
        x_offset *= sensitivity
        y_offset *= sensitivity
        
        self.yaw += x_offset
        self.pitch += y_offset
        
        # Constrain pitch
        if self.pitch > 89.0:
            self.pitch = 89.0
        if self.pitch < -89.0:
            self.pitch = -89.0
        
        # Update camera front vector
        front_x = math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        front_y = math.sin(math.radians(self.pitch))
        front_z = math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        self.camera_front = Vector3([front_x, front_y, front_z]).normalize()
    
    def get_view_matrix(self):
        """Calculate view matrix"""
        eye = self.camera_pos.data
        center = (self.camera_pos + self.camera_front).data
        up = self.camera_up.data
        return Matrix44.look_at(eye, center, up)
    
    def manipulate_object_matrix(self, position, rotation, scale):
        """Create transformation matrix for object manipulation"""
        # Translation
        translation = Matrix44.from_translation(position)
        
        # Rotation (Euler angles: X, Y, Z)
        rotation_x = Matrix44.from_x_rotation(rotation[0])
        rotation_y = Matrix44.from_y_rotation(rotation[1])
        rotation_z = Matrix44.from_z_rotation(rotation[2])
        rotation_matrix = rotation_z @ rotation_y @ rotation_x
        
        # Scale
        scale_matrix = Matrix44.from_scale(scale)
        
        # Combine transformations: Translation * Rotation * Scale
        model_matrix = translation @ rotation_matrix @ scale_matrix
        
        return model_matrix
    
    def render_object(self, vao, indices_count, model_matrix):
        """Render a 3D object with given transformation"""
        
        # Set uniforms
        self.program['model'].write(model_matrix.astype(np.float32).tobytes())
        self.program['view'].write(self.get_view_matrix().astype(np.float32).tobytes())
        self.program['projection'].write(self.projection.astype(np.float32).tobytes())
        self.program['light_pos'].write(np.array([2.0, 2.0, 2.0], dtype=np.float32).tobytes())
        
        # Render
        vao.render()
    
    def render_frame(self):
        """Render a single frame"""
        # Clear screen
        self.ctx.clear(0.1, 0.1, 0.1)  # Dark gray background
        self.ctx.clear(depth=1.0)
        
        # Animate objects
        cube_rotation = [self.time * 0.5, self.time * 0.3, 0]
        cube_position = [2.0, 0, 0]
        cube_scale = [1.0, 1.0, 1.0]
        
        pyramid_rotation = [0, self.time * 0.7, self.time * 0.4]
        pyramid_position = [-2.0, math.sin(self.time) * 0.5, 0]
        pyramid_scale = [0.8 + 0.2 * math.sin(self.time * 2), 1.2, 0.8 + 0.2 * math.sin(self.time * 2)]
        
        # Render cube
        cube_model = self.manipulate_object_matrix(cube_position, cube_rotation, cube_scale)
        self.render_object(self.cube_vao, self.cube_indices_count, cube_model)
        
        # Render pyramid
        pyramid_model = self.manipulate_object_matrix(pyramid_position, pyramid_rotation, pyramid_scale)
        self.render_object(self.pyramid_vao, self.pyramid_indices_count, pyramid_model)
    
    def run(self):
        """Main render loop"""
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        
        try:
            while self.running:
                dt = self.clock.tick(60) / 1000.0  # Delta time in seconds
                self.time += dt
                
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                
                # Process input
                self.process_input()
                
                # Render frame
                self.render_frame()
                
                # Swap buffers
                pygame.display.flip()
        
        except Exception as e:
            print(f"Runtime error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)
        pygame.quit()

# Example usage with more complex object manipulation
class Advanced3DScene(GL3DRenderer):
    def __init__(self, width=800, height=600):
        super().__init__(width, height)
        self.objects = []
        self.create_multiple_objects()
    
    def create_multiple_objects(self):
        """Create multiple objects with different properties"""
        # Create a grid of cubes with different animations
        for i in range(-2, 3):
            for j in range(-2, 3):
                obj = {
                    'type': 'cube' if (i + j) % 2 == 0 else 'pyramid',
                    'position': [i * 3, 0, j * 3],
                    'rotation': [0, 0, 0],
                    'scale': [0.5, 0.5, 0.5],
                    'animation_offset': (i + j) * 0.3,
                    'base_position': [i * 3, 0, j * 3]
                }
                self.objects.append(obj)
    
    def update_objects(self):
        """Update object transformations based on time"""
        for obj in self.objects:
            # Animate position (wave motion)
            wave_height = math.sin(self.time * 2 + obj['animation_offset']) * 0.5
            obj['position'][1] = obj['base_position'][1] + wave_height
            
            # Animate rotation
            obj['rotation'][1] = self.time * 0.5 + obj['animation_offset']
            
            # Animate scale (pulsing)
            pulse = 0.8 + 0.3 * math.sin(self.time * 3 + obj['animation_offset'])
            obj['scale'] = [pulse * 0.5, pulse * 0.5, pulse * 0.5]
    
    def render_frame(self):
        """Render frame with multiple animated objects"""
        self.ctx.clear(0.05, 0.05, 0.1)  # Dark blue background
        self.ctx.clear(depth=1.0)
        
        # Update all objects
        self.update_objects()
        
        # Render all objects
        for obj in self.objects:
            model_matrix = self.manipulate_object_matrix(
                obj['position'], obj['rotation'], obj['scale']
            )
            
            if obj['type'] == 'cube':
                self.render_object(self.cube_vao, self.cube_indices_count, model_matrix)
            else:
                self.render_object(self.pyramid_vao, self.pyramid_indices_count, model_matrix)

# Usage example
if __name__ == "__main__":
    print("Choose rendering mode:")
    print("1. Basic 3D Scene (2 objects)")
    print("2. Advanced 3D Scene (grid of animated objects)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    try:
        if choice == "2":
            renderer = Advanced3DScene()
        else:
            renderer = GL3DRenderer()
        
        renderer.run()
        
    except ModuleNotFoundError as e:
        print(f"Missing dependency: {e}")
        print("\nInstall required dependencies:")
        print("pip install moderngl pygame pyrr numpy")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        print("\nExiting...")