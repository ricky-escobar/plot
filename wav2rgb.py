# adapted from http://codingmess.blogspot.com/2009/05/conversion-of-wavelength-in-nanometers.html
from plot_help import Color


def wav2rgb(wavelength):
    w = int(wavelength)

    # colour
    if 380 <= w < 440:
        r = -(w - 440.) / (440. - 350.)
        g = 0.0
        b = 1.0
    elif 440 <= w < 490:
        r = 0.0
        g = (w - 440.) / (490. - 440.)
        b = 1.0
    elif 490 <= w < 510:
        r = 0.0
        g = 1.0
        b = -(w - 510.) / (510. - 490.)
    elif 510 <= w < 580:
        r = (w - 510.) / (580. - 510.)
        g = 1.0
        b = 0.0
    elif 580 <= w < 645:
        r = 1.0
        g = -(w - 645.) / (645. - 580.)
        b = 0.0
    elif 645 <= w <= 780:
        r = 1.0
        g = 0.0
        b = 0.0
    else:
        r = 0.0
        g = 0.0
        b = 0.0

    # intensity correction
    c = 1.0
    if 380 <= w < 420:
        c = 0.3 + 0.7 * (w - 350) / (420 - 350)
    elif 700 < w <= 780:
        c = 0.3 + 0.7 * (780 - w) / (780 - 700)
    c *= 255

    return Color(c * r, c * g, c * b)
