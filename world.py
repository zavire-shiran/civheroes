from OpenGL.GL import *
import numpy
import texture
import math
from collections import defaultdict
import random
from ctypes import c_void_p
import pprint
import pygame

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
                y * hexsize * math.sqrt(3)/2 - math.sqrt(3)/4 * hexsize)
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
        self.scale_uniform = glGetUniformLocation(self.shaderprogram, 'scale')

        self.tex = texture.Texture('terrain.png')

        self.map = initworldmap((50, 50))

        self.hexes = Primitives(GL_TRIANGLES, 0, 1)
        for x, row in enumerate(self.map):
            for y, tile in enumerate(row):
                corners = hexcorners((x-7,y-7), 0.5)
                texcorners = hexcorners((0.65, 0.5), 0.5)
                terrainpos = terrainlookup(tile['terrain'])
                texcorners = [[s/4.0 + (terrainpos[0]/4.0),
                               t/3.0 + (terrainpos[1] / 3.0)]
                              for (s, t) in texcorners]
                for i in xrange(len(corners)-2):
                    self.hexes.addvertex(corners[0], texcorners[0])
                    self.hexes.addvertex(corners[i+1], texcorners[i+1])
                    self.hexes.addvertex(corners[i+2], texcorners[i+2])
        self.hexes.finalize_buffer()

        texsamplers = ctypes.c_uint(0)
        glGenSamplers(1, texsamplers)
        self.texsampler = texsamplers.value
        glSamplerParameteri(self.texsampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glSamplerParameteri(self.texsampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glSamplerParameteri(self.texsampler, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glSamplerParameteri(self.texsampler, GL_TEXTURE_WRAP_T, GL_REPEAT)

        self.camerapos = [0,0]
        self.scale = 1.0

        self.time = 0

        self.camcontrols = {'left': False, 'right': False, 'up': False, 'down': False, 'zoomin':False, 'zoomout':False}

    def keydown(self, key):
        if key == pygame.K_RIGHT:
            self.camcontrols['right'] = True
        if key == pygame.K_LEFT:
            self.camcontrols['left'] = True
        if key == pygame.K_UP:
            self.camcontrols['up'] = True
        if key == pygame.K_DOWN:
            self.camcontrols['down'] = True
        if key == pygame.K_MINUS:
            self.camcontrols['zoomout'] = True
        if key == pygame.K_EQUALS:
            self.camcontrols['zoomin'] = True

    def keyup(self, key):
        if key == pygame.K_RIGHT:
            self.camcontrols['right'] = False
        if key == pygame.K_LEFT:
            self.camcontrols['left'] = False
        if key == pygame.K_UP:
            self.camcontrols['up'] = False
        if key == pygame.K_DOWN:
            self.camcontrols['down'] = False
        if key == pygame.K_MINUS:
            self.camcontrols['zoomout'] = False
        if key == pygame.K_EQUALS:
            self.camcontrols['zoomin'] = False

    def draw(self):
        glUseProgram(self.shaderprogram)

        screenratio = float(screensize[0]) / screensize[1]

        glUniform2fv(self.camera_center_uniform, 1, self.camerapos)
        glUniformMatrix4fv(self.camera_to_clip_uniform, 1, False, make_ortho_matrix(-3 * screenratio, 3 * screenratio, -3, 3, 10, -10))
        glUniform1f(self.scale_uniform, self.scale)

        glBindSampler(1, self.texsampler)

        glActiveTexture(GL_TEXTURE0 + 1)
        self.tex()
        glUniform1i(self.texture_uniform, 1)
        
        #self.squares.draw()
        self.hexes.draw()

    def step(self, dt):
        self.time += dt
        if self.camcontrols['right']:
            self.camerapos[0] += 2 * dt / self.scale
        if self.camcontrols['left']:
            self.camerapos[0] -= 2 * dt / self.scale
        if self.camcontrols['up']:
            self.camerapos[1] += 2 * dt / self.scale
        if self.camcontrols['down']:
            self.camerapos[1] -= 2 * dt / self.scale
        if self.camcontrols['zoomin']:
            self.scale += dt/2
        if self.camcontrols['zoomout']:
            self.scale -= dt/2
            self.scale = max(self.scale, 0.2)


def terrainlookup(terrain):
    if terrain == 'hill':
        return (0, 0)
    if terrain == 'desert':
        return (1, 0)
    if terrain == 'forest':
        return (2, 0)
    if terrain == 'tundra':
        return (3, 0)
    if terrain == 'grassland':
        return (0, 1)
    if terrain == 'mountain':
        return (1, 1)
    if terrain == 'ocean':
        return (2, 1)
    if terrain == 'coast':
        return (3, 1)
    return (2, 3)


def worldpos2gridpos(pos, hexsize):
    pos = [x/hexsize for x in pos]
    pos[0] = (pos[0]) / 0.75
    pos[0] = int(math.floor(pos[0] + 0.5))
    pos[1] /= math.sqrt(3)/2
    if pos[0] % 2 == 0:
        pos[1] -= 0.5
    pos[1] = int(math.floor(pos[1] + 0.5))
    return pos

def adjacenthexes(pos):
    ret = [(pos[0],   pos[1]+1),
           (pos[0],   pos[1]-1)]
    if pos[0] % 2 == 0:
        ret += [(pos[0]+1, pos[1]+1), (pos[0]-1, pos[1]+1)]
    ret += [(pos[0]+1, pos[1]),
            (pos[0]-1, pos[1])]
    if pos[0] % 2 == 1:
        ret += [(pos[0]+1, pos[1]-1), (pos[0]-1, pos[1]-1)]
    return ret


def adjacenthexesbounded(pos, size):
    return [h for h in adjacenthexes(pos) if h[0] >= 0 and h[0] < size[0] and h[1] >= 0 and h[1] < size[1]]


def initworldmap(mapsize):
    worldmap  = [[{'terrain':'ocean', 'unit':None, 'improvement':None, 'owner':None, 'influence':defaultdict(lambda:0.0)}
            for y in xrange(mapsize[1])] for x in xrange(mapsize[0])]

    generateContinents(worldmap, mapsize)

    return worldmap


def generateContinents(ret, mapsize):
    mincontinents = 1
    minsize = 20
    maxsize = 30
    minlandratio = 0.3
    totalsquares = mapsize[0] * mapsize[1]

    numcontinents = 0
    landsquares = 0
    availablehexes = set((x,y) for x in xrange(mapsize[0]) for y in xrange(mapsize[1]))
    # Grow continents
    while numcontinents < mincontinents or float(landsquares) / totalsquares < minlandratio:
        seed = random.sample(availablehexes, 1)[0]
        availablehexes.remove(seed)
        ret[seed[0]][seed[1]]['terrain'] = 'grassland'
        numcontinents += 1
        landsquares += 1
        continentsize = random.randint(minsize, maxsize)
        growablearea = set(h for h in adjacenthexesbounded(seed, mapsize) if h in availablehexes)
        for i in xrange(continentsize):
            if len(growablearea) == 0:
                continue
            growinto = random.sample(growablearea, 1)[0]
            growablearea.remove(growinto)
            ret[growinto[0]][growinto[1]]['terrain'] = 'grassland'
            availablehexes.remove(growinto)
            landsquares += 1
            for h in adjacenthexesbounded(growinto, mapsize):
                if h in availablehexes and ret[h[0]][h[1]]['terrain'] == 'ocean':
                    growablearea.add(h)
        availablehexes -= growablearea
    # distribute other terrain
    for x, row in enumerate(ret):
        for y, tile in enumerate(row):
            if tile['terrain'] == 'ocean':
                continue
            lat = float(abs((mapsize[1] / 2) - y)) / (mapsize[1]/2)
            tundrachance = max((lat - 0.8) * 5, 0)
            desertchance = max((0.2 - lat) * 5, 0)
            roll = random.random()
            if roll < tundrachance:
                tile['terrain'] = 'tundra'
            if roll > 1.0 - desertchance:
                tile['terrain'] = 'desert'
    for x, row in enumerate(ret):
        for y, tile in enumerate(row):
            if tile['terrain'] == 'ocean':
                for adj in [ret[adjx][adjy] for (adjx, adjy) in adjacenthexesbounded((x,y), mapsize)]:
                    if adj['terrain'] != 'ocean' and adj['terrain'] != 'coast':
                        tile['terrain'] = 'coast'
