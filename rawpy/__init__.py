from __future__ import absolute_import

from ._version import __version__

import rawpy._rawpy
globals().update({k:v for k,v in rawpy._rawpy.__dict__.items() if not k.startswith('_')})

import os
import sys
import warnings

# Check for potential multiprocessing deadlock on first import
_warned_about_multiprocessing = False

def _check_multiprocessing_fork():
    """
    Check if we're in a child process created by fork() with OpenMP enabled.
    Issue a warning if this combination is detected, as it can cause deadlocks.
    """
    global _warned_about_multiprocessing
    
    # Only warn once per process
    if _warned_about_multiprocessing:
        return
    
    # Only relevant on Unix-like systems where fork is available
    if sys.platform == 'win32':
        return
        
    # Only warn if OpenMP is enabled
    if not (flags and flags.get('OPENMP', False)):
        return
    
    # Check if we're in a forked child process by looking at multiprocessing context
    try:
        import multiprocessing as mp
        # Only warn if the current start method is 'fork'
        # Note: get_start_method() returns the current effective method
        if mp.get_start_method(allow_none=True) == 'fork':
            # Only warn if we're actually in a child process
            # We detect this by checking if multiprocessing has been initialized
            # and if we're not in the main process
            try:
                if mp.current_process().name != 'MainProcess':
                    warnings.warn(
                        "rawpy is being used in a process created by multiprocessing with the 'fork' start method. "
                        "This can cause deadlocks when OpenMP is enabled (which it is in this build). "
                        "Please use mp.set_start_method('spawn') or mp.set_start_method('forkserver') "
                        "at the start of your main module. "
                        "See https://github.com/letmaik/rawpy#faq for more information.",
                        RuntimeWarning,
                        stacklevel=3
                    )
                    _warned_about_multiprocessing = True
            except (RuntimeError, ValueError):
                # Multiprocessing context not initialized yet, or other error - skip warning
                pass
    except ImportError:
        # multiprocessing not available
        pass

def imread(pathOrFile):
    """
    Convenience function that creates a :class:`rawpy.RawPy` instance, opens the given file,
    and returns the :class:`rawpy.RawPy` instance for further processing.
    
    :param str|file pathOrFile: path or file object of RAW image that will be read
    :rtype: :class:`rawpy.RawPy`
    """
    _check_multiprocessing_fork()
    d = RawPy()
    if hasattr(pathOrFile, 'read'):
        d.open_buffer(pathOrFile)
    else:
        d.open_file(pathOrFile)
    return d