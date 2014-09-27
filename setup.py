from __future__ import print_function

from setuptools import setup, Extension, find_packages
import numpy
import subprocess
import errno
import os
import shutil
import sys
import zipfile
try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

if sys.version_info < (2, 7):
    raise NotImplementedError('Minimum supported Python version is 2.7')

isWindows = os.name == 'nt'
is64Bit = sys.maxsize > 2**32

# adapted from cffi's setup.py
# the following may be overridden if pkg-config exists
libraries = ['libraw']
include_dirs = []
library_dirs = []
extra_compile_args = []
extra_link_args = []

def _ask_pkg_config(resultlist, option, result_prefix='', sysroot=False):
    pkg_config = os.environ.get('PKG_CONFIG','pkg-config')
    try:
        p = subprocess.Popen([pkg_config, option, 'libraw'],
                             stdout=subprocess.PIPE)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise
    else:
        t = p.stdout.read().decode().strip()
        if p.wait() == 0:
            res = t.split()
            # '-I/usr/...' -> '/usr/...'
            for x in res:
                assert x.startswith(result_prefix)
            res = [x[len(result_prefix):] for x in res]

            sysroot = sysroot and os.environ.get('PKG_CONFIG_SYSROOT_DIR', '')
            if sysroot:
                # old versions of pkg-config don't support this env var,
                # so here we emulate its effect if needed
                res = [path if path.startswith(sysroot)
                            else sysroot + path
                         for path in res]
            resultlist[:] = res

def use_pkg_config():
    _ask_pkg_config(include_dirs,       '--cflags-only-I', '-I', sysroot=True)
    _ask_pkg_config(extra_compile_args, '--cflags-only-other')
    _ask_pkg_config(library_dirs,       '--libs-only-L', '-L', sysroot=True)
    _ask_pkg_config(extra_link_args,    '--libs-only-other')
    _ask_pkg_config(libraries,          '--libs-only-l', '-l')

# Some thoughts on bundling LibRaw in Linux installs:
# Compiling and bundling libraw.so like in the Windows wheels is likely not
# easily possible for Linux. This is due to the fact that the dynamic linker ld
# doesn't search for libraw.so in the directory where the Python extension is in.
# The -rpath with $ORIGIN method can not be used in this case as $ORIGIN is always
# relative to the executable and not the shared library, 
# see https://stackoverflow.com/q/6323603.
# But note that this was never tested and may actually still work somehow.
# matplotlib works around such problems by including external libraries as pure
# Python extensions, partly rewriting their sources and removing any dependency
# on a configure script, or cmake or other build infrastructure. 
# A possible work-around could be to statically link against libraw.

include_dirs += [numpy.get_include()]

if isWindows:
    cmake_build = 'external/LibRaw/cmake_build'
    librawh_dir = 'external/LibRaw/libraw'
    librawlib_dir = cmake_build
    
    include_dirs += [librawh_dir]
    library_dirs += [librawlib_dir]
    libraries = ['raw_r']
    extra_compile_args += ['/DWIN32']
else:
    use_pkg_config()

def windows_libraw_compile():
    # check that lensfun git submodule is cloned
    if not os.path.exists('external/LibRaw/README'):
        print('LibRaw git submodule is not cloned yet, will invoke "git submodule update --init" now')
        if os.system('git submodule update --init') != 0:
            raise Exception('git failed')
    
    # download cmake to compile libraw
    # the cmake zip contains a cmake-3.0.1-win32-x86 folder when extracted
    cmake_url = 'http://www.cmake.org/files/v3.0/cmake-3.0.1-win32-x86.zip'
    cmake = os.path.abspath('external/cmake-3.0.1-win32-x86/bin/cmake.exe')
    files = [(cmake_url, 'external', cmake)]
    for url, extractdir, extractcheck in files:
        if not os.path.exists(extractcheck):
            path = 'external/' + os.path.basename(url)
            if not os.path.exists(path):
                print('Downloading', url)
                urlretrieve(url, path)
        
            with zipfile.ZipFile(path) as z:
                print('Extracting', path, 'into', extractdir)
                z.extractall(extractdir)
            
            if not os.path.exists(path):
                raise RuntimeError(path + ' not found!')
    
    # configure and compile libraw
    cwd = os.getcwd()
    if not os.path.exists(cmake_build):
        os.mkdir(cmake_build)
    os.chdir(cmake_build)
    # TODO get OpenMP support working
    # see http://msdn.microsoft.com/en-us/library/0h7x01y0.aspx
    # see http://blog.codekills.net/2007/09/20/openmp-and-visual-c++-the-free-way-%28sorta%29/
    # http://www.metaintegration.net/Products/License/Microsoft-VisualCpp2010-RedistributionLicense.txt
    # http://blogs.msdn.com/b/vcblog/archive/2007/10/12/how-to-redistribute-the-visual-c-libraries-with-your-application.aspx?PageIndex=2
    # -> Visual Studio must be installed and contains the dlls in <Visual Studio install dir>\VC\redist
    cmds = [cmake + ' .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ' +\
                    '-DENABLE_EXAMPLES=OFF -DENABLE_OPENMP=ON -DENABLE_RAWSPEED=OFF ' +\
                    '-DENABLE_DEMOSAIC_PACK_GPL2=ON -DDEMOSAIC_PACK_GPL2_RPATH=../LibRaw-demosaic-pack-GPL2 ' +\
                    '-DENABLE_DEMOSAIC_PACK_GPL3=ON -DDEMOSAIC_PACK_GPL3_RPATH=../LibRaw-demosaic-pack-GPL3',
            'dir',
            'nmake raw_r' # build only thread-safe version ('raw'=non-thread-safe)
            ]
    for cmd in cmds:
        print(cmd)
        if os.system(cmd) != 0:
            sys.exit()   
    os.chdir(cwd)
        
    # bundle runtime dlls
    dll_runtime_libs = [('raw_r.dll', 'external/LibRaw/cmake_build')]
    
    # openmp dll
    isVS2008 = sys.version_info < (3, 3)
    isVS2010 = (3, 3) <= sys.version_info < (3, 5)
    isVS2014 = (3, 5) <= sys.version_info
    
    libraw_configh = 'external/LibRaw/cmake_build/libraw_config.h'
    match = '#define LIBRAW_USE_OPENMP 1'
    hasOpenMpSupport = match in open(libraw_configh).read()
    
    if isVS2008:
        if not hasOpenMpSupport:
            raise Exception('OpenMP not available but should be, see error messages above')
        if is64Bit:
            omp = [r'C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\redist\amd64\microsoft.vc90.openmp\vcomp90.dll',
                   r'C:\Windows\winsxs\amd64_microsoft.vc90.openmp_1fc8b3b9a1e18e3b_9.0.21022.8_none_a5325551f9d85633\vcomp90.dll']
        else:
            omp = [r'C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\redist\x86\microsoft.vc90.openmp\vcomp90.dll',
                   r'C:\Windows\winsxs\x86_microsoft.vc90.openmp_1fc8b3b9a1e18e3b_9.0.21022.8_none_ecdf8c290e547f39\vcomp90.dll']
    elif isVS2010:
        # Visual Studio 2010 Express and the free SDKs don't support OpenMP
        if hasOpenMpSupport:
            if is64Bit:
                omp = [r'C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\redist\x86\microsoft.vc100.openmp\vcomp100.dll']
            else:
                omp = [r'C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\redist\amd64\microsoft.vc100.openmp\vcomp100.dll']
    elif isVS2014:
        raise NotImplementedError('Python 3.5 will likely target MSVC 2014, not supported yet')
    
    if hasOpenMpSupport:
        try:
            omp_dir = os.path.dirname(list(filter(os.path.exists, omp))[0])
            dll_runtime_libs += [(os.path.basename(omp[0]), omp_dir)]
        except KeyError:
            raise Exception('OpenMP DLL not found, please read WINDOWS_COMPILE')
        
    for filename, folder in dll_runtime_libs:
        src = os.path.join(folder, filename)
        dest = 'rawpy/' + filename
        print('copying', src, '->', dest)
        shutil.copyfile(src, dest)
        
package_data = {}

# evil hack, check cmd line for relevant commands
# custom cmdclasses didn't work out in this case
cmdline = ''.join(sys.argv[1:])
if isWindows and any(s in cmdline for s in ['bdist', 'build_ext', 'nosetests']):
    windows_libraw_compile()
        
    package_data['rawpy'] = ['*.dll']
    
if any(s in cmdline for s in ['clean', 'sdist']):
    # When running sdist after a previous run of bdist or build_ext
    # then even with the 'clean' command the .egg-info folder stays.
    # This folder contains SOURCES.txt which in turn is used by sdist
    # to include package data files, but we don't want .dll's and .xml
    # files in our source distribution. Therefore, to prevent accidents,
    # we help a little...
    egg_info = 'rawpy.egg-info'
    print('removing', egg_info)
    shutil.rmtree(egg_info, ignore_errors=True)

pyx_path = '_rawpy.pyx'
c_path = '_rawpy.cpp'
if not os.path.exists(pyx_path):
    # we are running from a source dist which doesn't include the .pyx
    use_cython = False
else:
    try:
        from Cython.Build import cythonize
    except ImportError:
        use_cython = False
    else:
        use_cython = True

source_path = pyx_path if use_cython else c_path

extensions = [Extension("rawpy._rawpy",
              include_dirs=include_dirs,
              sources=[source_path],
              libraries=libraries,
              library_dirs=library_dirs,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
             )]

if use_cython:
    extensions = cythonize(extensions)

setup(
      name = 'rawpy',
      version = '0.1.0',
      description = 'Python wrapper for the LibRaw library',
      long_description = open('README.rst').read(),
      author = 'Maik Riechert',
      author_email = 'maik.riechert@arcor.de',
      url = 'https://github.com/neothemachine/rawpy',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Software Development :: Libraries',
      ],
      packages = find_packages(),
      ext_modules = extensions,
      package_data = package_data,
      install_requires=['enum34'],
)
