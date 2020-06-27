"""
Helper function definitions for code profiling

@Author: Akash Pallath
"""
import time
from functools import wraps

"""
Skip function exection
"""
def skipfunc(func):
    @wraps(func)
    def skippedfunc(*args, **kwargs):
        return True
    return skippedfunc

"""
Report function execution time to stdout
"""
def timefunc(func):
    @wraps(func)
    def timedfunc(*args, **kwargs):
        tstart = time.time()
        output = func(*args, **kwargs)
        tend = time.time()

        print("%r %.2f s " % (func.__name__, tend - tstart))

        return output
    return timedfunc

"""
Report function execution time to file
"""
def timefuncfile(fname):
    def timefunc(func):
        @wraps(func)
        def timedfunc(*args, **kwargs):
            tstart = time.time()
            output = func(*args, **kwargs)
            tend = time.time()

            with open(fname, "a+") as f:
                f.write("%r %.2f s\n" % (func.__name__, tend - tstart))

            return output
        return timedfunc
    return timefunc
