from sys import argv
from inspect import getargspec

import plot



FUNCS = ["graphpict", "voronoi", "basin", "basin2", "basin3", "newtondist",
         "newtondiff", "gridspanningtree", "gridspanningtree2", "complexroots",
         "polynomialbasin", "mandelbrot", "julia", "juliazoom", "pict2graphpict"]

class Function(object):
    def __init__(self, name):
        if name not in FUNCS:
            error("Unknown function: " + name)
        self.name = name
        self.func = getattr(plot, name)
        self.argspec = getargspec(self.func)
        self.args = {}

    def add_arg(self, name, value):
        self.args[self.resolve_param(name)] = value

    def resolve_param(self, name):
        if name in self.argspec.args:
            return name
        candidates = filter(lambda arg: arg.startswith(name), self.argspec.args)
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) == 0:
            error("Unknown parameter: " + name)
        else:
            error("Ambiguous parameter: " + name)

    def call(self):
        try:
            return self.func(**self.args)
        except TypeError:
            print_function_help(self)
            exit(1)


def main():
    if len(argv) == 1:
        print_usage()
    else:
        function = Function(argv[1])
        if "--help" in argv:
            print_function_help(function)
        else:
            for arg, value in parse_args(argv[2:]):
                function.add_arg(arg, value)
            function.call()
    exit(0)

def parse_args(args):
    i = 0
    while i < len(args):
        if args[i].startswith("-"):
            param = args[i].strip("-")
            i += 1
            if args[i].startswith("["):
                lst = []
                while not args[i].endswith("]"):
                    lst.append(convert(args[i].strip("[")))
                    i += 1
                lst.append(convert(args[i].strip("[]")))
                yield param, lst
            else:
                yield param, convert(args[i])
        else:
            error("Unexpected argument: " + args[i])
        i += 1


def print_usage():
    print "usage: " + argv[0] + " <function name> <arguments>"
    print "functions: " + ", ".join(FUNCS)

def print_function_help(f):
    required_params = f.argspec.args[:-len(f.argspec.defaults)]
    default_params = zip(f.argspec.args[-len(f.argspec.defaults):], f.argspec.defaults)

    if required_params:
        print "Required arguments:"
    for param in required_params:
        print "\t--" + param

    if default_params:
        print "Optional arguments:"
    for param, default in default_params:
        print "\t--" + param + " <{default}>".format(default=str(default))

def convert(arg):
    if arg == "True":
        return True
    elif arg == "False":
        return False
    else:
        try:
            return int(arg)
        except ValueError:
            try:
                return float(arg)
            except ValueError:
                return arg

def error(msg):
    print msg
    exit(1)

if __name__ == '__main__':
    main()
