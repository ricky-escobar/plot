from plot import *
import random
from wav2rgb import wav2rgb

H = 0.001


class Vec(object):
    def __init__(self, x, y):
        (self.x, self.y) = (float(x), float(y))

    def __mul__(self, other):
        if isinstance(other, Vec):
            return self.x * other.x + self.y * other.y
        else:
            return Vec(self.x * other, self.y * other)

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

    def scale(self, c):
        return Vec(self.x * c, self.y * c)

    def sqrmagn(self):
        return self.x ** 2 + self.y ** 2

    def unit(self):
        div = 1 / abs(self)
        return Vec(self.x * div, self.y * div)

    def normal(self):
        return Vec(-self.y, self.x)

    def rot(self, theta):
        return Vec(self.x * cos(theta) - self.y * sin(theta),
                   self.x * sin(theta) + self.y * cos(theta))

    def reflect(self, surface):
        s = surface.unit()
        return -self + 2 * (self * s) * s


class Material(object):
    def __init__(self, func, index):
        self.func = func
        self.index = index

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __contains__(self, point):
        if isinstance(point, Vec):
            return self(point.x, point.y)
        elif isinstance(point, Photon):
            return self(point.pos.x, point.pos.y)
        return self(*point)

    def find_intersect(self, pos1, v):
        pos2 = pos1 + v
        exiting = pos1 in self
        isect = (pos1 + pos2) / 2
        for i in range(20):
            if (isect in self) == exiting:
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
                while pos + H * Vec(cos(theta), sin(theta)) in self:
                    theta += dtheta
                dtheta /= 2
                while pos + H * Vec(cos(theta), sin(theta)) not in self:
                    theta -= dtheta
                dtheta /= 2
            pts.append(pos + H * Vec(cos(theta), sin(theta)))
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
            if self.pos + self.v not in mat:
                isect = mat.find_intersect(self.pos, self.v)
                s = mat.surface(isect)
                u = self.pos - isect
                cos_theta2 = effindex * (s * u) / (abs(s) * abs(u))
                if abs(cos_theta2) > 1:
                    self.v = self.v.reflect(s)
                    self.pos = isect + (abs(self.v) - abs(u)) * self.v.unit()
                else:
                    theta2 = math.acos(cos_theta2)
                    self.v = (-s).rot(theta2).unit() * abs(self.v) * effindex
                    self.pos = isect + (abs(self.v) - effindex * abs(u)) * self.v.unit()
            else:
                self.pos += self.v

        else:
            if self.pos + self.v in mat:
                isect = mat.find_intersect(self.pos, self.v)
                s = mat.surface(isect)
                u = self.pos - isect
                cos_theta2 = (1 / effindex) * (s * u) / (abs(s) * abs(u))
                theta2 = math.acos(cos_theta2)
                self.v = (-s).rot(-theta2).unit() * abs(self.v) / effindex
                self.pos = isect + (abs(self.v) - abs(u) / effindex) * self.v.unit()
            else:
                self.pos += self.v


def light_sim(vw, run_time=100, spawn_time=100, origin=Vec(0, 0), v=Vec(0, .05), mat=Material.empty(),
              density=20, perp_width=.5, parallel_width=.6, segregation=False, colormin=380,
              colormax=780, color_add=True, gif=True, gifres=1):
    data = init(vw)
    l = []
    imgs = []
    offset1 = parallel_width * v.unit() / 2
    offset2 = perp_width * v.unit().rot(pi / 2) / 2
    for px in vw.pxiterator():
        if vw.px2cart(px) in mat:
            data.putpixel(px, rgb(20, 20, 20))
    for t in xrange(run_time):
        for i in xrange(len(l) - 1, -1, -1):
            photon = l[i]
            data.putpoint(photon.pos.tuple(), (photon in mat) * rgb(20, 20, 20))  # , add=True, avg=True)
            if photon.pos.tuple() not in vw:
                del l[i]
        for photon in l:
            photon.move(mat)
            data.putpoint(photon.pos.tuple(), photon.color, add=color_add)
        if t < spawn_time:
            for i in range(density):
                r1, r2 = random.uniform(-1, 1), random.uniform(-1, 1)
                pos0 = origin + offset1 * r1 + offset2 * r2
                if segregation:
                    wav = r2 * 200 + 580
                else:
                    wav = random.randrange(colormin, colormax + 1)
                photon = Photon(pos0, v, wav)
                l.append(photon)
                data.putpoint(photon.pos.tuple(), photon.color, add=color_add)
        if gif and t % gifres == 0:
            imgs.append(data.save("", False))
    data.save("light_sim" + str(int(time.time())))
    if gif:
        writegif("light_sim" + str(int(time.time())), imgs)
