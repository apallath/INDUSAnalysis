"""
Helper function definitions for code profiling
"""
from functools import wraps
import logging
import time

logger = logging.getLogger(__name__)


def skipfunc(func):
    """
    Bypasses function execution [decorator]
    """
    @wraps(func)
    def skippedfunc(*args, **kwargs):
        return True
    return skippedfunc


def timefunc(func):
    """
    Reports function execution time to stdout [decorator]
    """
    @wraps(func)
    def timedfunc(*args, **kwargs):
        tstart = time.time()
        output = func(*args, **kwargs)
        tend = time.time()

        logger.debug("%r %.2f s " % (func.__name__, tend - tstart))

        return output
    return timedfunc


def timefuncfile(fname):
    """
    Reports function execution time to file [decorator]
    """
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
