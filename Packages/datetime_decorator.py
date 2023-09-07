#!/usr/bin/env python
# __coding: UTF-8 __

import datetime

def printDatetimes(f):
    def wrapper(*args, **kwargs):
        print(f'開始: {datetime.datetime.now()}')
        f(*args, **kwargs)
        print(f'終了: {datetime.datetime.now()}')
    return wrapper

if __name__ == "__main__":
    printDatetimes        