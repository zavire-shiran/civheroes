from OpenGL.GL import *
import numpy
from ctypes import c_void_p

currentworld = None
screensize = None


def setscreensize(size):
    global screensize
    screensize = size


def getworld():
    global currentworld
    return currentworld


def transitionto(world):
    global currentworld
    currentworld = world(currentworld)


def createshader(filename, shadertype):
    shader = glCreateShader(shadertype)
    source = open(filename).read()
    glShaderSource(shader, source)
    glCompileShader(shader)
    ok = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not ok:
        print 'Shader compile failed:', filename
        print glGetShaderInfoLog(shader)
        # this should probably be an exception, but I can't be arsed right now.
        return -1
    return shader


def createprogram(*shaders):
    program = glCreateProgram()
    for shader in shaders:
        glAttachShader(program, shader)
    glLinkProgram(program)
    ok = glGetProgramiv(program, GL_LINK_STATUS)
    if not ok:
        print 'Could not link program:'
        print glGetProgramInfoLog(program)
        return -1
    return program

def make_ortho_matrix(left, right, bottom, top, near, far):
    return numpy.array([[2 / float(right - left), 0, 0, -float(right + left) / (right - left)],
                        [0, 2 / float(top - bottom), 0, -float(top + bottom) / (top - bottom)],
                        [0, 0, 2 / float(far - near), -float(far + near) / (far - near)],
                        [0, 0, 0, 1]], numpy.float32)


class World:
    def __init__(self, previous = None):
        pass

    def keydown(self, key):
        pass

    def keyup(self, key):
        pass

    def click(self, pos):
        pass

    def draw(self):
        pass

    def step(self, dt):
        pass

rawbufferdata = [0, 0,   1, 0, 0, 1,
                 0, 0.5, 0, 1, 0, 1,
                 0.5, 0, 0, 0, 1, 1]
bufferdata = numpy.array(rawbufferdata, numpy.float32)

class Game(World):
    def __init__(self, previous = None):
        vertshader = createshader('color_vertex.shader', GL_VERTEX_SHADER)
        fragshader = createshader('color_fragment.shader', GL_FRAGMENT_SHADER)
        self.shaderprogram = createprogram(vertshader, fragshader)

        self.world_to_camera_uniform = glGetUniformLocation(self.shaderprogram, 'WorldToCameraTransform')
        self.camera_to_clip_uniform = glGetUniformLocation(self.shaderprogram, 'CameraToClipTransform')

        self.vertexbuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexbuffer)
        glBufferData(GL_ARRAY_BUFFER, bufferdata, GL_STATIC_DRAW)

    def draw(self):
        glUseProgram(self.shaderprogram)

        identitymatrix = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]

        screenratio = float(screensize[0]) / screensize[1]

        glUniformMatrix4fv(self.world_to_camera_uniform, 1, False, identitymatrix)
        glUniformMatrix4fv(self.camera_to_clip_uniform, 1, False, make_ortho_matrix(-2 * screenratio, 2 * screenratio, -2, 2, 10, -10))

        glBindBuffer(GL_ARRAY_BUFFER, self.vertexbuffer)
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6*4, None)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6*4, c_void_p(8))

        glDrawArrays(GL_TRIANGLES, 0, 3)

