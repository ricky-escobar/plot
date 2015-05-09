import itertools

import random


class Graph(object):
    def __init__(self, init_dict=None):
        if init_dict is None:
            self.dict = {}
        else:
            self.dict = init_dict

    def addvert(self, *verts):
        for v in verts:
            self.dict[v] = set()

    def addedge(self, *pairs):
        for v1, v2 in pairs:
            self.dict[v1] |= {v2}
            self.dict[v2] |= {v1}

    def removeedge(self, *pairs):
        for v1, v2 in pairs:
            self.dict[v1] -= {v2}
            self.dict[v2] -= {v1}

    def removevert(self, *verts):
        for v1 in verts:
            for v2 in self.dict[v1]:
                self.dict[v2] -= {v1}
            del self.dict[v1]

    def copy(self):
        g = Graph()
        for v in self.dict:
            g.dict[v] = self.dict[v].copy()
        return g

    def __str__(self):
        s = ""
        for key in sorted(self.dict):
            s += str(key) + ":  "
            for elem in sorted(self.dict[key]):
                s += str(elem) + ", "
            s = s[:-2]
            s += "\n"
        return s

    @staticmethod
    def c(n):
        g = Graph()
        g.addvert(*(i for i in range(n)))
        g.addedge(*((i, (i + 1) % n) for i in range(n)))
        return g

    @staticmethod
    def k(n):
        g = Graph()
        g.addvert(*(i for i in range(n)))
        g.addedge(*((i, j) for i in range(n) for j in range(n) if i != j))
        return g

    @staticmethod
    def k(n, m):
        g = Graph()
        g.addvert(*(i for i in range(n + m)))
        g.addedge(*((i, j) for i in range(n) for j in range(n, n + m)))
        return g

    @staticmethod
    def w(n):
        g = Graph.c(n)
        g.addvert(n)
        g.addedge(*((i, n) for i in range(n)))
        return g

    @staticmethod
    def grid(n, m):
        gdict = {}
        for i in range(n):
            for j in range(m):
                if 0 < i < n - 1:
                    if 0 < j < m - 1:
                        gdict[(i, j)] = {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}
                    elif j == 0:
                        gdict[(i, j)] = {(i - 1, j), (i + 1, j), (i, j + 1)}
                    else:
                        gdict[(i, j)] = {(i - 1, j), (i + 1, j), (i, j - 1)}
                elif i == 0:
                    if 0 < j < m - 1:
                        gdict[(i, j)] = {(i + 1, j), (i, j - 1), (i, j + 1)}
                    elif j == 0:
                        gdict[(i, j)] = {(i + 1, j), (i, j + 1)}
                    else:
                        gdict[(i, j)] = {(i + 1, j), (i, j - 1)}
                else:
                    if 0 < j < m - 1:
                        gdict[(i, j)] = {(i - 1, j), (i, j - 1), (i, j + 1)}
                    elif j == 0:
                        gdict[(i, j)] = {(i - 1, j), (i, j + 1)}
                    else:
                        gdict[(i, j)] = {(i - 1, j), (i, j - 1)}
        return Graph(gdict)

    @staticmethod
    def griddiag(n, m):
        g = Graph()
        for i in range(n):
            for j in range(m):
                if 0 < i < n - 1:
                    if 0 < j < m - 1:
                        g.dict[(i, j)] = {(i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1), (i, j + 1),
                                          (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)}
                    elif j == 0:
                        g.dict[(i, j)] = {(i - 1, j), (i - 1, j + 1), (i, j + 1), (i + 1, j), (i + 1, j + 1)}
                    else:
                        g.dict[(i, j)] = {(i - 1, j), (i - 1, j - 1), (i, j - 1), (i + 1, j), (i + 1, j - 1)}
                elif i == 0:
                    if 0 < j < m - 1:
                        g.dict[(i, j)] = {(i, j - 1), (i, j + 1), (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)}
                    elif j == 0:
                        g.dict[(i, j)] = {(i, j + 1), (i + 1, j), (i + 1, j + 1)}
                    else:
                        g.dict[(i, j)] = {(i, j - 1), (i + 1, j), (i + 1, j - 1)}
                else:
                    if 0 < j < m - 1:
                        g.dict[(i, j)] = {(i, j - 1), (i, j + 1), (i - 1, j - 1), (i - 1, j), (i - 1, j + 1)}
                    elif j == 0:
                        g.dict[(i, j)] = {(i, j + 1), (i - 1, j), (i - 1, j + 1)}
                    else:
                        g.dict[(i, j)] = {(i, j - 1), (i - 1, j), (i - 1, j - 1)}
        return g

    @staticmethod
    def griddiagwrap(n, m):
        g = Graph()
        for i in range(n):
            for j in range(m):
                g.dict[(i, j)] = {((i - 1) % n, j), ((i + 1) % n, j), (i, (j - 1) % m), (i, (j + 1) % m),
                                  ((i - 1) % n, (j - 1) % m), ((i + 1) % n, (j - 1) % m), ((i + 1) % n, (j + 1) % m),
                                  ((i - 1) % n, (j + 1) % m)}
        return g

    @staticmethod
    def gridwrap(n, m):
        gdict = {}
        for i in range(n):
            for j in range(m):
                gdict[(i, j)] = {((i - 1) % n, j), ((i + 1) % n, j), (i, (j - 1) % m), (i, (j + 1) % m)}
        return Graph(gdict)

    @staticmethod
    def grid3d(n, m, p):
        gdict = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if 0 < i < n - 1:
                        if 0 < j < m - 1:
                            if 0 < k < p - 1:
                                gdict[(i, j, k)] = {(i - 1, j, k), (i + 1, j, k), (i, j - 1, k), (i, j + 1, k),
                                                    (i, j, k - 1), (i, j, k + 1)}
                            elif k == 0:
                                gdict[(i, j, k)] = {(i - 1, j, k), (i + 1, j, k), (i, j - 1, k), (i, j + 1, k),
                                                    (i, j, k + 1)}
                            else:
                                gdict[(i, j, k)] = {(i - 1, j, k), (i + 1, j, k), (i, j - 1, k), (i, j + 1, k),
                                                    (i, j, k - 1)}
                        elif j == 0:
                            if 0 < k < p - 1:
                                gdict[(i, j, k)] = {(i - 1, j, k), (i + 1, j, k), (i, j + 1, k), (i, j, k - 1),
                                                    (i, j, k + 1)}
                            elif k == 0:
                                gdict[(i, j, k)] = {(i - 1, j, k), (i + 1, j, k), (i, j + 1, k), (i, j, k + 1)}
                            else:
                                gdict[(i, j, k)] = {(i - 1, j, k), (i + 1, j, k), (i, j + 1, k), (i, j, k - 1)}
                        else:
                            if 0 < k < p - 1:
                                gdict[(i, j, k)] = {(i - 1, j, k), (i + 1, j, k), (i, j - 1, k), (i, j, k - 1),
                                                    (i, j, k + 1)}
                            elif k == 0:
                                gdict[(i, j, k)] = {(i - 1, j, k), (i + 1, j, k), (i, j - 1, k), (i, j, k + 1)}
                            else:
                                gdict[(i, j, k)] = {(i - 1, j, k), (i + 1, j, k), (i, j - 1, k), (i, j, k - 1)}
                    elif i == 0:
                        if 0 < j < m - 1:
                            if 0 < k < p - 1:
                                gdict[(i, j, k)] = {(i + 1, j, k), (i, j - 1, k), (i, j + 1, k), (i, j, k - 1),
                                                    (i, j, k + 1)}
                            elif k == 0:
                                gdict[(i, j, k)] = {(i + 1, j, k), (i, j - 1, k), (i, j + 1, k), (i, j, k + 1)}
                            else:
                                gdict[(i, j, k)] = {(i + 1, j, k), (i, j - 1, k), (i, j + 1, k), (i, j, k - 1)}
                        elif j == 0:
                            if 0 < k < p - 1:
                                gdict[(i, j, k)] = {(i + 1, j, k), (i, j + 1, k), (i, j, k - 1), (i, j, k + 1)}
                            elif k == 0:
                                gdict[(i, j, k)] = {(i + 1, j, k), (i, j + 1, k), (i, j, k + 1)}
                            else:
                                gdict[(i, j, k)] = {(i + 1, j, k), (i, j + 1, k), (i, j, k - 1)}
                        else:
                            if 0 < k < p - 1:
                                gdict[(i, j, k)] = {(i + 1, j, k), (i, j - 1, k), (i, j, k - 1), (i, j, k + 1)}
                            elif k == 0:
                                gdict[(i, j, k)] = {(i + 1, j, k), (i, j - 1, k), (i, j, k + 1)}
                            else:
                                gdict[(i, j, k)] = {(i + 1, j, k), (i, j - 1, k), (i, j, k - 1)}
                    else:
                        if 0 < j < m - 1:
                            if 0 < k < p - 1:
                                gdict[(i, j, k)] = {(i - 1, j, k), (i, j - 1, k), (i, j + 1, k), (i, j, k - 1),
                                                    (i, j, k + 1)}
                            elif k == 0:
                                gdict[(i, j, k)] = {(i - 1, j, k), (i, j - 1, k), (i, j + 1, k), (i, j, k + 1)}
                            else:
                                gdict[(i, j, k)] = {(i - 1, j, k), (i, j - 1, k), (i, j + 1, k), (i, j, k - 1)}
                        elif j == 0:
                            if 0 < k < p - 1:
                                gdict[(i, j, k)] = {(i - 1, j, k), (i, j + 1, k), (i, j, k - 1), (i, j, k + 1)}
                            elif k == 0:
                                gdict[(i, j, k)] = {(i - 1, j, k), (i, j + 1, k), (i, j, k + 1)}
                            else:
                                gdict[(i, j, k)] = {(i - 1, j, k), (i, j + 1, k), (i, j, k - 1)}
                        else:
                            if 0 < k < p - 1:
                                gdict[(i, j, k)] = {(i - 1, j, k), (i, j - 1, k), (i, j, k - 1), (i, j, k + 1)}
                            elif k == 0:
                                gdict[(i, j, k)] = {(i - 1, j, k), (i, j - 1, k), (i, j, k + 1)}
                            else:
                                gdict[(i, j, k)] = {(i - 1, j, k), (i, j - 1, k), (i, j, k - 1)}
        return Graph(gdict)

    @staticmethod
    def gridnd(*dims):
        gdict = {}
        for t in itertools.product(*[tuple(range(dim)) for dim in dims]):
            gdict[t] = set()
            for i in range(len(t)):
                if 0 < t[i] < dims[i] - 1:
                    gdict[t].update({tuple([t[j] + (i == j) for j in range(len(t))]),
                                     tuple([t[j] - (i == j) for j in range(len(t))])})
                elif t[i] == 0:
                    gdict[t].update({tuple([t[j] + (i == j) for j in range(len(t))])})
                else:
                    gdict[t].update({tuple([t[j] - (i == j) for j in range(len(t))])})
        return Graph(gdict)

    @staticmethod
    def gridmatrix(matrix):
        m = len(matrix)
        n = len(matrix[0])
        gdict = {}
        for i in range(m):
            for j in range(n):
                if matrix[i][j]:
                    gdict[(i, j)] = set()
                    if i > 0 and matrix[i - 1][j]:
                        gdict[(i, j)].add((i - 1, j))
                    if j > 0 and matrix[i][j - 1]:
                        gdict[(i, j)].add((i, j - 1))
                    if i < m - 1 and matrix[i + 1][j]:
                        gdict[(i, j)].add((i + 1, j))
                    if j < n - 1 and matrix[i][j + 1]:
                        gdict[(i, j)].add((i, j + 1))
        return Graph(gdict)

    def dfs(self, vert, rand=True, mode=False):
        yield vert
        stack = [vert]
        s = {vert}
        while stack:
            parent = stack[-1]
            if self.dict[parent]:
                children = sorted(self.dict[parent])
                child = children[int(rand) * int(random.random() ** 1.1 * len(children))]
                self.dict[parent].remove(child)
                if child not in s:
                    if mode:
                        yield (parent, child)
                    else:
                        yield child
                    stack.append(child)
                    s.add(child)
            else:
                del stack[-1]

    def bfs(self, vert, rand=True, mode=False):
        yield vert
        s = {vert}
        q = [vert]
        while q:
            parent = q.pop(0)
            children = list(self.dict[parent])
            random.shuffle(children, rand)
            for child in children:
                if child not in s:
                    if mode:
                        yield (parent, child)
                    else:
                        yield child
                    q.append(child)
                    s.add(child)

    def prim(self, vert, rand=True, mode=False):
        yield vert
        g0dict = self.copy().dict
        s = {vert}
        for child in g0dict[vert]:
            g0dict[child].discard(vert)
        while s:
            if rand:
                parent = random.choice(s)
                child = random.choice(g0dict[parent])
                g0dict[parent].discard(child)
            else:
                parent = s.pop()
                s.add(parent)
                child = s.pop()
            if mode:
                yield (parent, child)
            else:
                yield child
            g0dict[child].discard(parent)
            if g0dict[child]:
                s.add(child)
            if not g0dict[parent]:
                s.discard(parent)
            for grandchild in self.dict[child]:
                g0dict[grandchild].discard(child)
                if not g0dict[grandchild]:
                    s.discard(grandchild)

    def esco(self, vert, rand=True, mode=False):
        yield vert
        l = [vert]
        s = {vert}
        while l:
            parent = l.pop(int(rand) * int(random.random() ** 3 * len(l)))
            children = self.dict[parent]
            if rand:
                random.shuffle(list(children))
            for child in children:
                if child not in s:
                    if mode:
                        yield (parent, child)
                    else:
                        yield child
                    s.add(child)
                    l.append(child)

    @staticmethod
    def spanningtree(f, vert, rand=True):
        gen = f(vert, rand, True)
        gdict = {gen.next(): set()}
        for (parent, child) in gen:
            gdict[parent].add(child)
            gdict[child] = {parent}
        return Graph(gdict)

    def func(self, f, (x, y), rand=True):
        return getattr(self, f)((x, y), rand)

    def treefunc(self, tf, (x, y), rand=True):
        return Graph.spanningtree(getattr(self, tf), (x, y), rand)
