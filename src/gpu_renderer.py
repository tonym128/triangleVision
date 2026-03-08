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
        
        # To make it fastest and handle per-triangle colors, we duplicate vertices
        tri_pts = points[simplices].astype('f4') # (N, 3, 2)
        # colors are BGR from OpenCV. We pass them as-is.
        tri_colors = np.repeat(colors[:, np.newaxis, :], 3, axis=1).astype('f4') # (N, 3, 3)
        
        # Pack into interleaved buffer: [x, y, b, g, r] per vertex
        v_data = np.concatenate([tri_pts, tri_colors], axis=2).astype('f4').tobytes()
        vbo = self.ctx.buffer(v_data)
        
        vao = self.ctx.simple_vertex_array(self.prog, vbo, 'in_vert', 'in_color')
        vao.render(moderngl.TRIANGLES)
        
        # Ensure all rendering commands are finished before reading back
        self.ctx.finish()
        
        # Read back pixels. ModernGL reads are bottom-up.
        raw = self.fbo.read(components=3, dtype='u1')
        img = np.frombuffer(raw, dtype='u1').reshape((self.height, self.width, 3))
        
        # Flip vertically to match OpenCV coordinate system (top-down)
        # We use .copy() because flipud returns a view, which might be read-only
        img = np.flipud(img).copy()
        
        # Cleanup GPU resources to prevent memory leaks
        vbo.release()
        vao.release()
        
        # Since we passed BGR to the shader and read back the same bytes,
        # 'img' is already in BGR format for OpenCV. No conversion needed.
        return img

    def close(self):
        self.ctx.release()
