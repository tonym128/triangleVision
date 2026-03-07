import moderngl
import numpy as np
import cv2

class GPURenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # Initialize moderngl in standalone mode (using glcontext)
        self.ctx = moderngl.create_standalone_context()
        
        # Vertex Shader: Handles 2D points and color pass-through
        self.v_shader = """
            #version 330
            in vec2 in_vert;
            in vec3 in_color;
            out vec3 v_color;
            void main() {
                // Normalize screen coordinates from [0, W/H] to [-1, 1]
                gl_Position = vec4((in_vert.x / float(WIDTH) * 2.0) - 1.0, 
                                   (in_vert.y / float(HEIGHT) * -2.0) + 1.0, 0.0, 1.0);
                v_color = in_color / 255.0;
            }
        """.replace("WIDTH", str(width)).replace("HEIGHT", str(height))

        # Fragment Shader: Simple color output
        self.f_shader = """
            #version 330
            in vec3 v_color;
            out vec4 f_color;
            void main() {
                f_color = vec4(v_color, 1.0);
            }
        """
        
        self.prog = self.ctx.program(vertex_shader=self.v_shader, fragment_shader=self.f_shader)
        
        # Renderbuffers for off-screen rendering
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.renderbuffer((width, height))],
        )

    def render(self, points, simplices, colors, clear=True):
        """
        Renders triangles to a numpy array using the GPU.
        """
        self.fbo.use()
        if clear:
            self.ctx.clear(0, 0, 0, 1) # Black background
        
        # simplices are indices into points. We need to create a flat vertex array for moderngl
        # A more efficient way is using an Element Buffer (IBO)
        
        # Flattened Vertices and Colors for each vertex of each triangle
        # OpenGL expects 3 vertices per triangle if we don't use IBO
        # To make it fastest, we'll use an Index Buffer
        
        # 1. Create Vertex Buffer (Points)
        vbo_pts = self.ctx.buffer(points.astype('f4').tobytes())
        # 2. Create Index Buffer (Simplices)
        ibo = self.ctx.buffer(simplices.astype('u4').tobytes())
        # 3. Create Color Buffer (Repeat each color 3 times? No, colors are per-triangle)
        # In standard OpenGL, colors are per-vertex. To get flat shading per triangle,
        # we need to duplicate vertices or use a special shader technique.
        # Fastest way: Duplicate points for each triangle to have per-triangle colors.
        
        # Let's use the "duplicate points" method for simplicity in this prototype
        # since simplices are relatively small (< 10k)
        tri_pts = points[simplices].astype('f4') # (N, 3, 2)
        tri_colors = np.repeat(colors[:, np.newaxis, :], 3, axis=1).astype('f4') # (N, 3, 3)
        
        # Pack into interleaved buffer: [x, y, r, g, b, x, y, r, g, b, ...]
        v_data = np.dstack([tri_pts, tri_colors]).astype('f4').tobytes()
        vbo = self.ctx.buffer(v_data)
        
        vao = self.ctx.simple_vertex_array(self.prog, vbo, 'in_vert', 'in_color')
        vao.render(moderngl.TRIANGLES)
        
        # Read back pixels
        raw = self.fbo.read(components=3, dtype='u1')
        img = np.frombuffer(raw, dtype='u1').reshape((self.height, self.width, 3))
        
        # BGR for OpenCV
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def close(self):
        self.ctx.release()
