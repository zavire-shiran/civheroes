from OpenGL.GL import *
import numpy
import texture
import math
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


def hexpos(pos, hexsize):
    x, y = pos
    if x % 2 == 0:
        return (x * hexsize * 0.75,
                y * hexsize * math.sqrt(3)/2 + math.sqrt(3)/4 * hexsize)
    else:
        return (x * hexsize * 0.75,
                y * hexsize * math.sqrt(3)/2)


def hexcorners(pos, hexsize, scale=0.98):
    pos = hexpos(pos, hexsize)
    size = float(hexsize)/2 * scale
    halfsize = size/2
    top = size*math.sqrt(3)/2
    return [[pos[0]+size, pos[1]], 
            [pos[0]+halfsize, pos[1]+top],
            [pos[0]-halfsize, pos[1]+top],
            [pos[0]-size, pos[1]],
            [pos[0]-halfsize, pos[1]-top],
            [pos[0]+halfsize, pos[1]-top]]


class Primitives:
    def __init__(self, primtype, pos_attrib_loc, texcoord_attrib_loc):
        self.buffer = []
        self.glbuffer = glGenBuffers(1)
        self.primtype = primtype
        self.pos_attrib_loc = pos_attrib_loc
        self.texcoord_attrib_loc = texcoord_attrib_loc
        self.numverts = 0
        self.possize = 2
        self.texcoordsize = 2

    def addvertex(self, pos, texcoord):
        self.buffer += pos + texcoord
        self.numverts += 1

    def finalize_buffer(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.glbuffer)
        glBufferData(GL_ARRAY_BUFFER, numpy.array(self.buffer, numpy.float32), GL_STATIC_DRAW)

    def draw(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.glbuffer)

        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(self.pos_attrib_loc, self.possize, GL_FLOAT, GL_FALSE, (self.possize + self.texcoordsize) * 4, None)
        glVertexAttribPointer(self.texcoord_attrib_loc, self.texcoordsize, GL_FLOAT, GL_FALSE, (self.possize + self.texcoordsize) * 4, c_void_p(self.possize * 4))

        glDrawArrays(self.primtype, 0, self.numverts)


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



class Game(World):
    def __init__(self, previous = None):
        vertshader = createshader('color_vertex.shader', GL_VERTEX_SHADER)
        fragshader = createshader('color_fragment.shader', GL_FRAGMENT_SHADER)
        self.shaderprogram = createprogram(vertshader, fragshader)

        self.camera_center_uniform = glGetUniformLocation(self.shaderprogram, 'CameraCenter')
        self.camera_to_clip_uniform = glGetUniformLocation(self.shaderprogram, 'CameraToClipTransform')
        self.texture_uniform = glGetUniformLocation(self.shaderprogram, 'tex')

        self.squares = Primitives(GL_QUADS, 0, 1)
        self.squares.addvertex([-1,-1], [0,1])
        self.squares.addvertex([-1,1], [0,0])
        self.squares.addvertex([1,1], [1,0])
        self.squares.addvertex([1,-1], [1,1])
        self.squares.finalize_buffer()

        self.tex = texture.Texture('image.png')

        self.hexes = Primitives(GL_TRIANGLES, 0, 1)
        corners = hexcorners((0,0), 1.0)
        for i in xrange(len(corners)-2):
            self.hexes.addvertex(corners[0], corners[0])
            self.hexes.addvertex(corners[i+1], corners[i+1])
            self.hexes.addvertex(corners[i+2], corners[i+2])
        self.hexes.finalize_buffer()

        texsamplers = ctypes.c_uint(0)
        glGenSamplers(1, texsamplers)
        self.texsampler = texsamplers.value
        glSamplerParameteri(self.texsampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glSamplerParameteri(self.texsampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glSamplerParameteri(self.texsampler, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glSamplerParameteri(self.texsampler, GL_TEXTURE_WRAP_T, GL_REPEAT)

        self.camerapos = [0,0]

        self.time = 0

    def draw(self):
        glUseProgram(self.shaderprogram)

        screenratio = float(screensize[0]) / screensize[1]

        glUniform2fv(self.camera_center_uniform, 1, self.camerapos)
        glUniformMatrix4fv(self.camera_to_clip_uniform, 1, False, make_ortho_matrix(-3 * screenratio, 3 * screenratio, -3, 3, 10, -10))

        # glBindBuffer(GL_ARRAY_BUFFER, self.vertexbuffer)
        # glEnableVertexAttribArray(0)
        # glEnableVertexAttribArray(1)
        # glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6*4, None)
        # glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6*4, c_void_p(8))

        # glDrawArrays(GL_TRIANGLES, 0, 3)

        # glBindBuffer(GL_ARRAY_BUFFER, self.backgroundbuffer)
        # glEnableVertexAttribArray(0)
        # glEnableVertexAttribArray(1)
        # glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6*4, None)
        # glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6*4, c_void_p(8))

        # glDrawArrays(GL_QUADS, 0, 12)

        glBindSampler(1, self.texsampler)

        glActiveTexture(GL_TEXTURE0 + 1)
        self.tex()
        glUniform1i(self.texture_uniform, 1)
        
        #self.squares.draw()
        self.hexes.draw()

    def step(self, dt):
        self.time += dt
        #self.camerapos = [math.cos(self.time*5/2.0), math.cos(self.time * 3/2.0)]
