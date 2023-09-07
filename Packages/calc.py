#!/usr/bin/env python
# __coding: UTF-8 __

from datetime_decorator import printDatetimes

@printDatetimes
def main(a, b):
    print(a + b)
    
main("Hello", " Kosei!")
