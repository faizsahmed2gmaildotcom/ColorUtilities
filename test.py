from time import time
from typing import Callable, NamedTuple
import numpy as np
from random import randint


def funcTimer(f: Callable) -> Callable:
    def decoratedFunc():
        start = time()
        f()
        print(f"Function {f.__name__} took {time() - start}s")

    return decoratedFunc


my_set: set = set()
my_set.update({str(list(map(float, [1, 2, 3])))})
print(my_set)

print("test col ahn~".replace(" ", "_"))
