import random as r


def shuffle(seq, rand=True):
    if rand:
        r.shuffle(seq)


def choice(seq, rand=True):
    if rand:
        return r.choice(seq)
    else:
        for elem in seq:
            return elem


def random(rand=True):
    return int(rand) * r.random()
