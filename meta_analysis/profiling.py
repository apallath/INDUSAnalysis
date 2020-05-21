"""
Helper function definitions for code profiling

@Author: Akash Pallath
"""
import time

"""Decorators"""
def timefunc(func):
    def timedfunc(*args, **kwargs):
        tstart = time.time()
        output = func(*args, **kwargs)
        tend = time.time()

        print("%r %.2f s " % (func.__name__, tend - tstart))

        return output
    return timedfunc
