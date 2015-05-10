from math import sin
from time import asctime

from light_sim import light_sim, Material, Vec
from plot_help import vws, mkdir


def triangle(x, y):
    return 0 < y < 1.732 * (1 - abs(x))


def disk(x, y):
    return -1 < x < 1 and -1 < y < 1 and x ** 2 + y ** 2 < 1


def lens(x, y):
    return (x - .9) ** 2 + y ** 2 < 1 and (x + .9) ** 2 + y ** 2 < 1


def ring(x, y):
    return .81 < x * x + y * y < 1


def ring2(x, y):
    return .81 < x * x + y * y < 1 or 0 <= y <= .4 and -1 <= x <= -.94


def div_lens(x, y):
    return -.4 < x < .4 and -.75 < y < .75 and (x - 1.625) ** 2 + y ** 2 > 2.25 and (x + 1.625) ** 2 + y ** 2 > 2.25


def sine(x, y):
    return sin(2 * x) - 0.5 < y < sin(2 * x) + 0.5


def test1():
    light_sim(vw=vws[11], max_run_time=500, spawn_time=500, origin=Vec(0, 1.9), v=Vec(0, -1).unit() / 20,
              mat=Material(triangle, 1.3), density=30, perp_width=.09, parallel_width=.09, save=False, gif=False, gifres=1)


def test2():
    light_sim(vw=vws[11], max_run_time=200, origin=Vec(-.5, 1.0), v=Vec(0, -1).unit() / 20, mat=Material(triangle, 1.7),
              density=30, perp_width=.09, parallel_width=.09, gif=True, gifres=1)


def test2a():
    light_sim(vw=vws[11], max_run_time=200, origin=Vec(-.5, 1.0), v=Vec(0, -1).unit() / 40, mat=Material(triangle, 1.5),
              density=50, perp_width=.05, parallel_width=.05, gif=True, gifres=2)


def test3():
    light_sim(vw=vws[6], max_run_time=100, origin=Vec(0, 1.2), v=Vec(0, -1).unit() / 20, mat=Material(disk, 1.6),
              density=50, perp_width=2, parallel_width=.06)


def test3a():
    light_sim(vw=vws[6], max_run_time=130, spawn_time=5, origin=Vec(0, 1.2), v=Vec(0, -1) / 20, color_add=False,
              mat=Material(disk, 2.0), density=500, perp_width=2, parallel_width=.06, segregation=False)


def test4():
    light_sim(vw=vws[6], max_run_time=100, origin=Vec(-1, 0), v=Vec(1, 0).unit() / 20, mat=Material(lens, 1.5),
              density=50, perp_width=.75, parallel_width=.06, segregation=True)


def test5():
    light_sim(vw=vws[6], max_run_time=300, spawn_time=1, origin=Vec(.05, 1.1), v=Vec(-1.3, -1).unit() / 10,
              mat=Material(ring, 2.3), density=200, perp_width=.03, parallel_width=.06)


def test5a():
    light_sim(vw=vws[6], max_run_time=500, spawn_time=1, origin=Vec(-.97, .5), v=Vec(0, -1).unit() / 20,
              mat=Material(ring2, 2.3), density=200, perp_width=.02, parallel_width=.06, gif=False)


def test6():
    light_sim(vw=vws[6], max_run_time=70, spawn_time=8, origin=Vec(-1, .5), v=Vec(1, 0).unit() / 20,
              mat=Material(div_lens, 1.5), density=2000, perp_width=.01, parallel_width=.06, segregation=False)


def test7():
    light_sim(vw=vws[10], max_run_time=200, spawn_time=2, origin=Vec(-2.6, 0), v=Vec(1, 0).unit() / 20,
              mat=Material(sine, 1.7), density=50, perp_width=.5, parallel_width=.06)

mkdir()

print "Execution began at " + asctime()

test1()

print "Execution ended at " + asctime()
