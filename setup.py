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
    files = [(cmake_url, 'cmake-3.0.1-win32-x86.zip', 'external', cmake)]
    for url, path, extractdir, extractcheck in files:
        if not os.path.exists(extractcheck):
            if not os.path.exists(path):
                print('Downloading', url)
                urlretrieve(url, path)
        
            with zipfile.ZipFile(path) as z:
                print('Extracting', path, 'into', extractdir)
                z.extractall(extractdir)
    
    # configure and compile libraw
    cwd = os.getcwd()
    if not os.path.exists(cmake_build):
        os.mkdir(cmake_build)
    os.chdir(cmake_build)
    cmds = [cmake + ' .. -G "NMake Makefiles" -DENABLE_EXAMPLES=OFF -DENABLE_OPENMP=OFF -DENABLE_RAWSPEED=OFF ' +\
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
    dll_runtime_libs = [('raw_r.dll', 'external/LibRaw/cmake_build'),
                        ]    
    
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
        'Programming Language :: Python :: 3',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Software Development :: Libraries',
      ],
      packages = find_packages(),
      ext_modules = extensions,
      package_data = package_data,
      install_requires=['enum34'],
)
