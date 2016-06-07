from math import sqrt, e, log, cos, pi, sin, ceil, atan, atan2
from random import randint, choice
from time import time
from itertools import product

from PIL import Image
import pygame

from graph import Graph
from polynomial import Polynomial
from plot_help import Color, PlotData, ViewWindow, colorcube, hilbert, lp, direct, argcmp, diffquo, newton, frange, \
    writegif, closest, PYGAME_PIXEL



try:
    import numpy as np
except ImportError:
    np = None


def funcgraph(funclist, vw=ViewWindow(), colorlist=None, auto=False, filename="plot" + str(int(time())), gif=False,
              add=False, avg=False, save=True):
    images = []
    if auto:
        vw.autograph(funclist)
    if callable(funclist):
        funclist = [funclist]
    data = PlotData(vw)
    if colorlist is None:
        colorlist = Color.colors(len(funclist))
    for j in range(len(funclist)):
        f = funclist[j]
        it = vw.xiterator()
        x0 = it.next()
        for x1 in it:
            if colorlist == "rainbow":
                color = Color.rbow(x1, 2)
            else:
                color = colorlist[j]
            data.line((x0, f(x0)), (x1, f(x1)), color, add, avg)
            x0 = x1
            if gif:
                images.append(data.save("", False))
    if gif:
        writegif(filename, images, 0.01)
    return data.save(filename, save)


def polar(rlist, vw=ViewWindow(), colorlist=None, auto=False, gif=False, add=False, avg=False,
          filename="polar" + str(int(time())), save=True):
    if callable(rlist):
        xlist = lambda t: rlist(t) * cos(t)
        ylist = lambda t: rlist(t) * sin(t)
    else:
        xlist = [(lambda r: lambda t: r(t) * cos(t))(r0) for r0 in rlist]
        ylist = [(lambda r: lambda t: r(t) * sin(t))(r0) for r0 in rlist]
    return param(xlist, ylist, vw, colorlist, auto, gif, add, avg, filename, save)


def param(xlist, ylist, vw=ViewWindow(), colorlist=None, auto=False, gif=False, add=False, avg=False,
          filename="param" + str(int(time())), save=True):
    if callable(xlist):
        xlist = [xlist]
    if callable(ylist):
        ylist = [ylist]
    if auto:
        vw.autoparam(xlist, ylist)
    data = PlotData(vw)
    images = []
    color = 0
    if colorlist is None:
        colorlist = Color.colors(len(xlist))
    for i in range(len(xlist)):
        x = xlist[i]
        y = ylist[i]
        t = vw.tmin
        if colorlist != 'rainbow':
            color = colorlist[i]
        while t <= vw.tmax:
            if colorlist == 'rainbow':
                color = Color.rbow(vw.tmax - t, 2)
            u = t + vw.tstep
            data.line0((x(t), y(t)), (x(u), y(u)), color, add, avg)
            t = u
            if gif:
                images.append(data.save("", False))
    if gif:
        writegif(filename, images, 0.01)
    return data.save(filename, save)


def polarshade(r1, r2, vw=ViewWindow(), color=None, auto=False, filename="polarshade" + str(int(time())), save=True):
    if auto:
        vw.autoparam([lambda t0: r1(t0) * cos(t0), lambda t0: r2(t0) * cos(t0)],
                     [lambda t0: r1(t0) * sin(t0), lambda t0: r2(t0) * sin(t0)])
    data = PlotData(vw)
    th = vw.thick * (vw.xmax - vw.xmin) / vw.dimx
    if color is None:
        c = Color(255, 0, 0)
    else:
        c = color
    for i in range(vw.dimx):
        x = vw.xpxcart(i)
        for j in range(vw.dimy):
            y = vw.ypxcart(j)
            t = atan2(y, x)
            r = sqrt(x ** 2 + y ** 2)
            if color == 'rainbow':
                c = Color.rbow(t)
            if r1(t) - th <= r <= r2(t) + th or r2(t) - th <= r <= r1(t) + th:
                data.putpixel((i, j), c)
    return data.save(filename, save)


def basin(f, rootlist, colorlist=None, vw=ViewWindow(), df=None, n=10, dots=True,
          filename="basin" + str(int(time())), save=True):
    if df is None:
        df = diffquo(f, .0001)
    if colorlist is None:
        colorlist = Color.colors(len(rootlist))
    data = PlotData(vw)
    for i in range(vw.dimx):
        for j in range(vw.dimy):
            data.putpixel((i, j), colorlist[closest(rootlist, newton(f, vw.xpxcart(i) + vw.ypxcart(j) * 1j, df, n))])
    if dots:
        for r in rootlist:
            data.putpoint((r.real, r.imag), 0)
    return data.save(filename, save)


def basin2(f, rootlist, colorlist=None, vw=ViewWindow(), df=None, n=10, dots=True,
           filename="basin2" + str(int(time())), dist=0.3, save=True):
    if df is None:
        df = diffquo(f, .0001)
    if colorlist is None:
        colorlist = Color.colors(len(rootlist))
    data = PlotData(vw)
    for i in range(vw.dimx):
        for j in range(vw.dimy):
            z = vw.xpxcart(i) + vw.ypxcart(j) * 1j
            for k in range(n):
                z = newton(f, z, df, 1)
                index = closest(rootlist, z)
                d = abs(z - rootlist[index])
                if d < dist or k == n - 1:
                    data.putpixel((i, j), max(.2, 1 - float(k) / n) * colorlist[index])
                    break
    if dots:
        for r in rootlist:
            data.putpoint((r.real, r.imag), 0)
    return data.save(filename, save)


def basin3(f, rootlist, colorlist=None, vw=ViewWindow(), df=None, n=10, dots=True,
           filename="basin" + str(int(time())), save=True):
    if df is None:
        df = diffquo(f, .0001)
    if colorlist is None:
        colorlist = Color.colors(len(rootlist))
    data = PlotData(vw)
    for i in range(vw.dimx):
        for j in range(vw.dimy):
            z = vw.xpxcart(i) + vw.ypxcart(j) * 1j
            for k in range(n + 1):
                data.putpixel((i, j), colorlist[closest(rootlist, z)], add=True, avg=True)
                if k != n:
                    z = newton(f, z, df, 1)
    if dots:
        for r in rootlist:
            data.putpoint((r.real, r.imag), 0)
    return data.save(filename, save)


def newtondist(f, vw=ViewWindow(), df=None, n=1, pxdist=1, filename="newtondist" + str(int(time()))):
    images = []
    if df is None:
        df = diffquo(f, .0001)
    col = Color.colors(vw.dimx * vw.dimy)
    for k in range(6):
        data = PlotData(vw)
        for i in range(vw.dimx):
            for j in range(vw.dimy):
                if i % pxdist == 0 and j % pxdist == 0:
                    z0 = vw.xpxcart(i) + 1j * vw.ypxcart(j)
                    z = newton(f, z0, df, n, 1e-1)
                    z0 = newton(f, z0, df, n - 1, 1e-1)
                    data.putpoint(((k * z.real + (5 - k) * z0.real) / 5, (k * z.imag + (5 - k) * z0.imag) / 5),
                                  col[i + j * vw.dimx])
        images.append(data.save("", False))
    writegif(filename, images, 1)
    return images[-1]


def newtondiff(f, vw=ViewWindow(), df=None, n=1, pxdist=1, filename="newtondiff" + str(int(time())), save=True):
    if df is None:
        df = diffquo(f, .0001)
    data = PlotData(vw)
    color1 = Color(0, 255, 0)
    color2 = Color(255, 0, 0)
    for i in range(vw.dimx):
        for j in range(vw.dimy):
            if i % pxdist == 0 and j % pxdist == 0:
                z0 = vw.xpxcart(i) + vw.ypxcart(j) * 1j
                z = newton(f, z0, df, n, 1e-1)
                if abs(z) < abs(z0):
                    color = color1
                else:
                    color = color2
                data.putpixel((i, j), color)
    return data.save(filename, save)


def rk4(f, initlist, vw=ViewWindow(), h=.001, colorlist=None, sf=False, sfspread=20, sfcolor=0x0000FF,
        filename="rk4" + str(int(time())), save=True):
    data = PlotData(vw)
    if sf:
        data.data = list(slopefield(f, vw, sfspread, sfcolor, save=False).getdata())
    if colorlist is None:
        colorlist = Color.colors(len(initlist))
    for i in range(len(initlist)):
        c = colorlist[i]
        for h1 in [h, -h]:
            x0, y0 = initlist[i]
            while vw.xmin < x0 < vw.xmax:
                if colorlist == "rainbow":
                    c = Color.rbow(2 * pi * (x0 - vw.xmin) / (vw.xmax - vw.xmin), 2)
                k1 = f(x0, y0)
                k2 = f(x0 + h1 / 2, y0 + h1 * k1 / 2)
                k3 = f(x0 + h1 / 2, y0 + h1 * k2 / 2)
                k4 = f(x0 + h1, y0 + h1 * k3)
                x1, y1 = x0 + h1, y0 + (h1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                data.line((x0, y0), (x1, y1), c)
                x0, y0 = x1, y1
    return data.save(filename, save)


def slopefield(f, vw=ViewWindow(), spread=20, color=0x0000FF, filename="slopefield" + str(int(time())), save=True):
    tmp = vw.thick
    vw.thick = 0
    data = PlotData(vw)
    cartspreadx = vw.xpxcart(spread) - vw.xpxcart(0)
    cartspready = vw.ypxcart(spread) - vw.ypxcart(0)
    for i in range(vw.dimx):
        for j in range(vw.dimy):
            if i % spread == 0 and j % spread == 0:
                theta = atan(f(*vw.xpxcart((i, j))))
                data.line((vw.xpxcart(i) + cartspreadx * cos(theta) / 4, vw.ypxcart(j) - cartspready * sin(theta) / 4),
                          (vw.xpxcart(i) - cartspreadx * cos(theta) / 4, vw.ypxcart(j) + cartspready * sin(theta) / 4),
                          color)
    vw.thick = tmp
    return data.save(filename, save)


def implicit(flist, vw=ViewWindow(), decay=100, colorlist=None, filename="implicit" + str(int(time())), save=True):
    data = PlotData(vw)
    color = 0
    if colorlist is None:
        colorlist = Color.colors(len(flist))
    for k in range(len(flist)):
        f = flist[k]
        if colorlist != "rainbow":
            color = colorlist[k]
        for i in range(vw.dimx):
            for j in range(vw.dimy):
                if colorlist == "rainbow":
                    color = Color.rbow(vw.xpxcart(i) + vw.ypxcart(j), 2)
                color0 = e ** (-decay * abs(f(*vw.px2cart((i, j))))) * color
                color0 += data[(i, j)]
                data.putpixel((i, j), color0)
    return data.save(filename, save)


def levelcurves(f, vw=ViewWindow(), decay=100, minval=-1, maxval=1, step=.1,
                filename="levelcurves" + str(int(time())), save=True):
    flist = [(lambda k: lambda x, y: f(x, y) - k)(i) for i in frange(minval, maxval + step, step)]
    implicit(flist, vw, decay, Color.colors(len(flist)), filename, save)


def color3d(f, vw=ViewWindow(), base=1.0, pos=Color(255, 0, 0), neg=Color(0, 0, 255),
            filename="color3d" + str(int(time())), save=True):
    data = PlotData(vw)
    for i in range(vw.dimx):
        for j in range(vw.dimy):
            (x, y) = vw.px2cart((i, j))
            color = Color()
            val = f((x, y))
            if val > 0:
                color = abs(val / base) * pos
            if val < 0:
                color = abs(val / base) * neg
            data.putpixel((i, j), color)
    return data.save(filename, save)


def zrotate(xlist, ylist, vw=ViewWindow(), segs=4, colorlist=None, auto=False, filename=None, gif=False, add=True,
            avg=False, save=True):
    if auto:
        vw.autoparam(xlist, ylist)
    if filename is None:
        filename = "zrotate" + str(int(time()))
    if callable(xlist):
        xlist = [xlist]
    if callable(ylist):
        ylist = [ylist]
    data = PlotData(vw, "rainbow" if colorlist == "rainbow" else Color(0, 0, 0))
    images = []
    if colorlist is None:
        colorlist = Color.colors(len(xlist))
    tlist = [vw.tmin + (vw.tmax - vw.tmin) * j / segs for j in range(segs + 1)]
    while tlist[0] < vw.tmax:
        for (x, y, color) in zip(xlist, ylist, colorlist):
            if colorlist == 'rainbow':
                color = Color.rbow(tlist[0], segs)
            for j in range(len(tlist) - 1):
                data.line0((x(tlist[j]), y(tlist[j])), (x(tlist[j + 1]), y(tlist[j + 1])), color, add, avg)
        if gif:
            images.append(data.save("", False))
        tlist = [t + vw.tstep for t in tlist]
    if gif:
        writegif(filename, images, 0.01)
    return data.save(filename, save)


def gridspanningtree(length, height, d=2, r=0, color=Color(255, 0, 0), g="grid", filename=None, treefunc="dfs", gif=False,
                     save=True):
    images = []
    if not isinstance(g, Graph):
        (x, y) = (randint(0, length - 1), randint(0, height - 1))
        g = getattr(Graph, g)(length, height).treefunc(treefunc, (x, y))
    else:
        (x, y) = list(g.dict.keys)[0]
    vw = ViewWindow(dimx=d * length + d - 1, dimy=d * height + d - 1)
    data = PlotData(vw)
    color2 = 0.8 * color
    for v1 in g.dict:
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                data.putpixel((d * (v1[0] + 1) - 1 + i, d * (v1[1] + 1) - 1 + j), color)
    s = {(x, y)}
    q = [(x, y)]
    while q:
        v1 = q.pop(0)
        for v2 in g.dict[v1]:
            if v2 not in s:
                for i in range(1 + r, d - r):
                    data.putpixel(
                        (d * (v1[0] + 1) - 1 + (v2[0] - v1[0]) * i, d * (v1[1] + 1) - 1 + (v2[1] - v1[1]) * i), color2)
                q.append(v2)
                s.add(v2)
                if gif:
                    images.append(data.save("", False))
    if filename is None:
        filename = "gst " + str((length, height)) + " " + str(int(time()))
    ret = data.save(filename, save)
    if gif:
        writegif(filename, images, 0.01)
    return ret


def gridspanningtree2(length, height, d=2, r=0, g="grid", color=Color(255, 0, 0), filename=None, treefunc=None, func="dfs",
                      gif=False,
                      save=True):
    images = []
    if not isinstance(g, Graph):
        g = getattr(Graph, g)(length, height)
    if treefunc is not None:
        (x, y) = (randint(0, length - 1), randint(0, height - 1))
        g = g.treefunc(treefunc, (x, y))
    (x, y) = (randint(0, length - 1), randint(0, height - 1))
    g = g.func(func, (x, y))
    vw = ViewWindow(dimx=d * length + d - 1, dimy=d * height + d - 1)
    data = PlotData(vw)
    color2 = 0.8 * color
    for p in g:
        try:
            for i in range(-r, r + 1):
                for j in range(-r, r + 1):
                    data.putpixel((d * (p[0] + 1) - 1 + i, d * (p[1] + 1) - 1 + j), color)
        except TypeError:
            for i in range(-r, r + 1):
                for j in range(-r, r + 1):
                    data.putpixel((d * (p[1][0] + 1) - 1 + i, d * (p[1][1] + 1) - 1 + j), color)
            for i in range(1 + r, d - r):
                data.putpixel(
                    (d * (p[0][0] + 1) - 1 + (p[1][0] - p[0][0]) * i, d * (p[0][1] + 1) - 1 + (p[1][1] - p[0][1]) * i),
                    color2)
        if gif:
            images.append(data.save("", False))
    if filename is None:
        filename = "gst2 " + str((length, height)) + " " + str(time())
    ret = data.save(filename, save)
    if gif:
        writegif(filename, images, 0.01)
    return ret


def graphpict(length, height, graphtype="grid", treefunc="dfs", func="bfs", colorfunc="hilbert", gif=False, gifres=None, v0=None, v1=None,
              rand=True, save=True, scales=None):
    images = []
    clist = []
    if isinstance(graphtype, Graph):
        area = len(graphtype.dict)
    else:
        area = length * height
    if not colorfunc.startswith("rbow"):
        clist = colorcube(colorfunc)
    if func == "hilbert":
        l = hilbert(int(ceil(log(max(length, height), 2))))
        gstr = "None"
    else:
        if not isinstance(graphtype, Graph):
            gstr = graphtype
            graphtype = getattr(Graph, graphtype)(length, height)
        else:
            gstr = "Graph"
        if v0 is None:
            v0 = choice(graphtype.dict.keys())
        if treefunc is not None:
            if v1 is None:
                v1 = choice(graphtype.dict.keys())
            graphtype = graphtype.treefunc(treefunc, v1, rand)
            print "Computed spanning tree"
        l = graphtype.func(func, v0, rand)
    if gif and gifres is None:
        gifres = ceil(area / 150.0)
    vw = ViewWindow(dimx=length, dimy=height)
    data = PlotData(vw, scales=scales)
    i = 0
    for p in l:
        if colorfunc.startswith("rbow"):
            data.putpixel(p, getattr(Color, colorfunc)(2 * pi * i / area, 1))
        else:
            data.putpixel(p, Color(*[c * 4 for c in clist[min(64 ** 3 * i / area, 64 ** 3 - 1)]]))
        if gif and (i + 1) % gifres == 0:
            images.append(data.save("", False))
        i += 1
    filename = str((length, height)) + " " + gstr + " spanningtree" + str(treefunc) + " " + func + " " + colorfunc + " " + str(
        rand) + " " + str((v0, v1)) + " " + str(int(time()))
    ret = data.save(filename, save)
    if gif:
        writegif(filename, images, 0.001)
    return ret


def voronoi(rootlist, colorlist=None, vw=ViewWindow(), dots=True, metric=lp(2), select=1,
            filename="voronoi" + str(int(time())), save=True):
    if colorlist is None:
        colorlist = Color.colors(len(rootlist))
    data = PlotData(vw)
    for i in range(vw.dimx):
        for j in range(vw.dimy):
            data.putpixel((i, j), colorlist[closest(rootlist, vw.xpxcart(i) + vw.ypxcart(j) * 1j, metric, select)])
    if dots:
        for r in rootlist:
            data.putpoint((r.real, r.imag), 0)
    return data.save(filename, save)


def complexroots(deg, vw, color=Color(5, 0, 0), coeffs=(-1, 1), filename=None, save=True):
    if np is None:
        raise Exception()
    data = PlotData(vw)
    for i, l in enumerate(product(coeffs, repeat=deg + 1)):
        if i % len(coeffs) ** (deg - 4) == 0:
            print float(i) / 2 ** (deg + 1)
        for root in np.roots(l):
            data.putpoint((root.real, root.imag), color, True)
    if filename is None:
        filename = "complexroots " + str(deg) + " " + str(int(time()))
    return data.save(filename, save)


def polynomialbasin(coeffs, vw, auto=False, n=10, dots=True, mode=1, filename=None, save=True):
    p = Polynomial(*coeffs)
    rootlist = list(p.roots())
    rootlist.sort(cmp=argcmp)
    if auto:
        vw.autocomplex(*rootlist)
    if filename is None:
        filename = "polynomialbasin {poly} {n} {dots} {time}".format(poly=p, n=n, dots=dots, time=str(int(time())))
    return globals()["basin" + str(mode) * (mode in (2, 3))](p, rootlist, Color.colors(len(rootlist)), vw, p.deriv(), n,
                                                             dots, filename, save)


def mandelbrot(vw=ViewWindow(xmin=-2.3, xmax=0.7, ymin=-1.5, ymax=1.5, dimx=700, dimy=701), n=50, reflect=True,
               color=Color(255, 0, 0), filename=None, save=True):
    data = PlotData(vw)
    color0 = color
    if callable(color):
        colorfunc = color
    else:
        colorfunc = lambda t: t * color0
    for p in vw.pxiterator():
        c = vw.xpxcart(p[0]) + vw.ypxcart(p[1]) * 1j
        if c.imag > -0.1 or not reflect:
            i, z0 = 0, 0
            q = (c.real - 0.25) ** 2 + c.imag ** 2
            if q * (q + (c.real - 0.25)) < c.imag ** 2 / 4 or 16 * ((c.real + 1) ** 2 + c.imag ** 2) < 1:
                data.putpixel(p, colorfunc(1.0), flip=False)
                if reflect:
                    data.putpixel((p[0], vw.ycartpx(-c.imag)), colorfunc(1.0), flip=False)
            else:
                while abs(z0) < 2 and i < n:
                    z0 = z0 * z0 + c
                    i += 1
                data.putpixel(p, colorfunc(float(i) / n), flip=False)
                if reflect:
                    data.putpixel((p[0], vw.ycartpx(-c.imag)), colorfunc(float(i) / n), flip=False)
        else:
            break
        if PYGAME_PIXEL and p[0] == 0:
            pygame.display.flip()
    if filename is None:
        filename = "mandelbrot " + str(n) + " " + str(int(time()))
    return data.save(filename, save)


def julia(f, vw, n=50, zmax=2, color=Color(255, 0, 0), filename=None, save=True):
    data = PlotData(vw)
    if callable(color):
        colorfunc = color
    else:
        colorfunc = lambda t: t * color
    for p in vw.pxiterator():
        z = vw.xpxcart(p[0]) + vw.ypxcart(p[1]) * 1j
        i = 0
        while abs(z) < zmax and i < n:
            z = f(z)
            i += 1
        data.putpixel(p, colorfunc(float(i) / n), flip=False)
        if PYGAME_PIXEL and p[0] == 0:
            pygame.display.flip()
    if filename is None:
        filename = "julia " + str(n) + " " + str(zmax) + " " + str(int(time()))
    return data.save(filename, save)


def juliazoom(f, vw, z0=0, factor=1.5, frames=20, n=50, duration=0.5, zmax=2, color=Color(255, 0, 0), filename=None):
    images = []
    for i in range(frames):
        vw = ViewWindow(z0.real - factor ** -i, z0.real + factor ** -i, z0.imag - factor ** -i, z0.imag + factor ** -i,
                        dimx=vw.dimx, dimy=vw.dimy)
        images.append(julia(f, vw, n, zmax, color, filename, False))
    if filename is None:
        filename = "julia " + str(n) + " " + str(zmax) + " " + str(int(time()))
    writegif(filename, images, duration)


def pict2graphpict(img, condition=None, treefunc="dfs", func="bfs", colorfunc="hilbert", gif=False, gifres=None,
                   v0=None, v1=None, rand=True, save=True, scales=None):
    if condition is None:
        condition = lambda x: x == 0
    if isinstance(img, (str, Image.Image)):
        if isinstance(img, str):
            try:
                img = Image.open(img)
            except IOError:
                img = Image.open(direct + img)
        (length, height) = img.size
        seq = img.getdata()
        img = [[condition(seq[j * length + i]) for j in range(height)] for i in range(length)]
    else:
        (length, height) = (len(img), len(img[0]))
    g = Graph.gridmatrix(img)
    return graphpict(length, height, g, treefunc, func, colorfunc, gif, gifres, v0, v1, rand, save, scales)
