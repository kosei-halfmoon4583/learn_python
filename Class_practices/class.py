# __coding: UTF-8 __

import math
from unittest import result

class Dog:
    name = ""
    def bark(self):
        m = self.name + ": Bow-wow!"
        print(m)

pochi = Dog()
pochi.name = "Pochi"
pochi.bark()

hachi = Dog()
hachi.name = "Hachi"
hachi.bark()