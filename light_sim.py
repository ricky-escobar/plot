from math import cos, pi, acos
from math import sin
from time import time
from plot import init, writegif, rgb
from random import uniform, randrange
from wav2rgb import wav2rgb

H = 0.001


class Vec(object):
    def __init__(self, x, y):
        (self.x, self.y) = (float(x), float(y))

    def __mul__(self, other):
        return self.x * other.x + self.y * other.y

    def __rmul__(self, other):
        return Vec(self.x * other, self.y * other)

    def __add__(self, other):
        return Vec(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec(self.x - other.x, self.y - other.y)

    def __neg__(self):
        return Vec(-self.x, -self.y)

    def __div__(self, other):
        return Vec(self.x / other, self.y / other)

    def __abs__(self):
        return (self.x ** 2 + self.y ** 2) ** .5

    def __str__(self):
        return "<{x}, {y}>".format(x=self.x, y=self.y)

    def tuple(self):
        return self.x, self.y

    def unit(self):
        div = 1.0 / abs(self)
        return Vec(self.x * div, self.y * div)

    def rot(self, theta):
        return Vec(self.x * cos(theta) - self.y * sin(theta),
                   self.x * sin(theta) + self.y * cos(theta))

    def reflect(self, surface):
        return -self + 2 * (self * surface.unit()) * surface.unit()


class Material(object):
    def __init__(self, func, index):
        self.func = func
        self.index = index

    def __contains__(self, point):
        if isinstance(point, Vec):
            return self.func(point.x, point.y)
        elif isinstance(point, Photon):
            return self.func(point.pos.x, point.pos.y)
        return self.func(*point)

    def find_intersect(self, pos1, v):
        pos2 = pos1 + v
        exiting = pos1 in self
        isect = (pos1 + pos2) / 2
        for i in range(10):
            if (self.func(isect.x, isect.y)) == exiting:
                pos1 = isect
                isect = (isect + pos2) / 2
            else:
                pos2 = isect
                isect = (isect + pos1) / 2
        return isect

    def surface(self, pos):
        pts = []
        for i in (0, 1):
            theta = 0
            dtheta = pi / 2 - i * pi
            while abs(dtheta) > pi / 2 ** 10:
                # while pos + H * Vec(cos(theta), sin(theta)) in self:
                while self.func(pos.x + H * cos(theta), pos.y + H * sin(theta)):
                    theta += dtheta
                dtheta /= 2
                # while pos + H * Vec(cos(theta), sin(theta)) not in self:
                while not self.func(pos.x + H * cos(theta), pos.y + H * sin(theta)):
                    theta -= dtheta
                dtheta /= 2
            pts.append(Vec(pos.x + H * cos(theta), pos.y + H * sin(theta)))
        return pts[1] - pts[0]

    def calc_index(self, wavelength):
        low = max(1.0, self.index * .9)
        return self.index + (self.index - low) * (580 - wavelength) / 200

    @staticmethod
    def empty():
        return Material(lambda x, y: False, 1)


class Photon(object):
    def __init__(self, pos, v, wavelength):
        self.pos = pos
        self.v = v
        self.wavelength = wavelength
        self.color = wav2rgb(wavelength)

    def move(self, mat):
        effindex = mat.calc_index(self.wavelength)
        if self in mat:
            v = self.v / effindex
            if self.pos + v not in mat:
                isect = mat.find_intersect(self.pos, v)
                s = mat.surface(isect)
                u = self.pos - isect
                cos_theta2 = effindex * (s * u) / (abs(s) * abs(u))
                if abs(cos_theta2) > 1:
                    v = v.reflect(s)
                    self.v = effindex * v
                    self.pos = isect + (abs(v) - abs(u)) * v.unit()
                else:
                    theta2 = acos(cos_theta2)
                    self.v = abs(self.v) * (-s).rot(theta2).unit()
                    self.pos = isect + (abs(self.v) - effindex * abs(u)) * self.v.unit()
            else:
                self.pos += v
        else:
            if self.pos + self.v in mat:
                isect = mat.find_intersect(self.pos, self.v)
                s = mat.surface(isect)
                u = self.pos - isect
                cos_theta2 = (1 / effindex) * (s * u) / (abs(s) * abs(u))
                theta2 = acos(cos_theta2)
                self.v = abs(self.v) * (-s).rot(-theta2).unit()
                self.pos = isect + (abs(self.v) - abs(u)) * self.v.unit() / effindex
            else:
                self.pos += self.v


def light_sim(vw, max_run_time=100, spawn_time=100, origin=Vec(0, 0), v=Vec(0, .05), mat=Material.empty(),
              density=20, perp_width=.5, parallel_width=.6, segregation=False, colormin=380, colormax=780,
              color_add=True, save=True, gif=True, gifres=1):
    data = init(vw)
    data0 = init(vw)
    mat_color = rgb(20, 20, 20)
    l = []
    imgs = []
    offset1 = parallel_width * v.unit() / 2
    offset2 = perp_width * v.unit().rot(pi / 2) / 2
    for px in vw.pxiterator():
        if vw.px2cart(px) in mat:
            data.putpixel(px, mat_color)
            data0.putpixel(px, mat_color)
    for t in xrange(max_run_time):
        for i in xrange(len(l) - 1, -1, -1):
            photon = l[i]
            if photon.pos.tuple() not in vw:
                del l[i]
            elif gif:
                (m, n) = vw.cart2px(photon.pos.tuple())
                data.putpixel((m, n), data0[(m, n)])  # , add=True, avg=True)
        for photon in l:
            photon.move(mat)
            if gif or t == max_run_time - 1:
                data.putpoint(photon.pos.tuple(), photon.color, add=color_add)
        if t < spawn_time:
            for i in range(density):
                r1, r2 = uniform(-1, 1), uniform(-1, 1)
                pos0 = origin + r1 * offset1 + r2 * offset2
                if segregation:
                    wav = r2 * 200 + 580
                else:
                    wav = randrange(colormin, colormax + 1)
                photon = Photon(pos0, v, wav)
                l.append(photon)
                if gif or t == max_run_time - 1:
                    data.putpoint(pos0.tuple(), photon.color, add=color_add)
        if gif and t % gifres == 0:
            imgs.append(data.save("", False))
        if not l:
            break
    if save:
        data.save("light_sim" + str(int(time())))
    if gif:
        writegif("light_sim" + str(int(time())), imgs)
