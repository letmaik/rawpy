from __future__ import print_function

from setuptools import setup, Extension, find_packages
import numpy
import subprocess
import errno
import os
import shutil
import sys
import zipfile
import re
try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

if sys.version_info < (2, 7):
    raise NotImplementedError('Minimum supported Python version is 2.7')

# As rawpy is distributed under the MIT license, it cannot use or distribute
# GPL'd code. This is relevant for the Windows and Mac binary wheels which
# include a self-compiled version of LibRaw that is dynamically linked against.
# Under Linux, the source distribution dynamically links against whatever
# variant of LibRaw is installed on the users system. Common Linux distributions
# such as Ubuntu don't compile LibRaw with the GPL'd code parts, therefore
# it is safe to assume that the standard installed version is LGPL.
# If a user compiles his own version of LibRaw including the GPL'd code
# (either by flipping the switch below for Mac or Windows, or by manually
# compiling under Linux) then the software produced by the user would have to
# be released under GPL as well.
buildGPLCode = False

isWindows = os.name == 'nt'
isMac = sys.platform == 'darwin'
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

if isWindows or isMac:
    external_dir = os.path.abspath('external')
    cmake_build = os.path.join(external_dir, 'LibRaw', 'cmake_build')
    install_dir = os.path.join(cmake_build, 'install')
    
    include_dirs += [os.path.join(install_dir, 'include', 'libraw')]
    library_dirs += [os.path.join(install_dir, 'lib')]
    libraries = ['raw_r']
    
    # for Windows and Mac we use cmake, so libraw_config.h will always exist
    libraw_config_found = True
else:
    use_pkg_config()
    
    # check if libraw_config.h exists
    # this header is only installed when using cmake
    libraw_config_found = False
    for include_dir in include_dirs:
        if 'libraw_config.h' in os.listdir(include_dir):
            libraw_config_found = True
            break

define_macros = [('_HAS_LIBRAW_CONFIG_H', '1' if libraw_config_found else '0')]

if isWindows:
    extra_compile_args += ['/DWIN32']
    
# this must be after use_pkg_config()!
include_dirs += [numpy.get_include()]

def clone_submodules():
    if not os.path.exists('external/LibRaw/README'):
        print('LibRaw git submodule is not cloned yet, will invoke "git submodule update --init" now')
        if os.system('git submodule update --init') != 0:
            raise Exception('git failed')
    
    # copy cmake files into LibRaw root directory
    if not os.path.exists('external/LibRaw/CMakeLists.txt'):
        print('copying CMake scripts from LibRaw-cmake repository')
        shutil.copy('external/LibRaw-cmake/CMakeLists.txt', 'external/LibRaw/CMakeLists.txt')
        shutil.copytree('external/LibRaw-cmake/cmake', 'external/LibRaw/cmake')
        
def patch_cmakelists():
    """Makes 'raw' target OPTIONAL during installation."""
    cmakelists_path = os.path.join(external_dir, 'LibRaw', 'CMakeLists.txt')
    def add_optional(m):
        if 'OPTIONAL' in m.group(1):
            return m.group(0)
        else:
            return 'INSTALL(TARGETS raw OPTIONAL {})'.format(m.group(1))
    with open(cmakelists_path, 'r') as fp:
        cmakelists = fp.read()
        cmakelists_patched = re.sub(r'INSTALL\(TARGETS raw(.*?)\)', add_optional, cmakelists, count=1, flags=re.DOTALL)
    if cmakelists != cmakelists_patched:
        with open(cmakelists_path, 'w') as fp:
            fp.write(cmakelists_patched)

def windows_libraw_compile():
    clone_submodules()
    
    # download cmake to compile libraw
    # the cmake zip contains a cmake-3.7.2-win32-x86 folder when extracted
    cmake_url = 'https://cmake.org/files/v3.7/cmake-3.7.2-win32-x86.zip'
    cmake = os.path.abspath('external/cmake-3.7.2-win32-x86/bin/cmake.exe')
    
    files = [(cmake_url, 'external', cmake)]
    
    # libraw's rawspeed support is based on the master branch which still requires libxml2
    # the develop branch has this dependency removed
    # -> let's wait until rawspeed develop is merged into master and libraw catches up
    # see https://github.com/LibRaw/LibRaw/issues/40
    use_rawspeed = False
    if use_rawspeed:
        # FIXME probably have to apply rawspeed patches 
        
        # dependencies for rawspeed
        pthreads_url = 'http://mirrors.kernel.org/sourceware/pthreads-win32/pthreads-w32-2-9-1-release.zip'
        files.extend([(pthreads_url, 'external/pthreads', 'external/pthreads/Pre-built.2')])
    
    for url, extractdir, extractcheck in files:
        if not os.path.exists(extractcheck):
            path = 'external/' + os.path.basename(url)
            if not os.path.exists(path):
                print('Downloading', url)
                try:
                    urlretrieve(url, path)
                except:
                    # repeat once in case of network issues
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
    
    # We only want to build and install the raw_r target (to make builds faster).
    # To do that we set CMAKE_SKIP_INSTALL_ALL_DEPENDENCY=1 so that 'install' doesn't depend
    # on raw and raw_r targets.
    # For the install target to finish successfully, we need to set the raw target to optional,
    # which unfortunately can only be done in the CMakeLists.txt file. Therefore we patch it.
    patch_cmakelists()
    
    # Important: always use Release build type, otherwise the library will depend on a
    #            debug version of OpenMP which is not what we bundle it with, and then it would fail
    ext = lambda p: os.path.join(external_dir, p)
    pthreads_dir = ext('pthreads/Pre-built.2')
    arch = 'x64' if is64Bit else 'x86'
    cmds = [cmake + ' .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ' +\
                    '-DENABLE_EXAMPLES=OFF -DENABLE_OPENMP=ON -DENABLE_RAWSPEED=OFF ' +\
                    ('-DENABLE_DEMOSAIC_PACK_GPL2=ON -DDEMOSAIC_PACK_GPL2_RPATH=../LibRaw-demosaic-pack-GPL2 ' +\
                     '-DENABLE_DEMOSAIC_PACK_GPL3=ON -DDEMOSAIC_PACK_GPL3_RPATH=../LibRaw-demosaic-pack-GPL3 '
                     if buildGPLCode else '') +\
                    ('-DENABLE_RAWSPEED=ON -DRAWSPEED_RPATH=../rawspeed/RawSpeed ' +\
                     '-DPTHREADS_INCLUDE_DIR=' + os.path.join(pthreads_dir, 'include') + ' ' +\
                     '-DPTHREADS_LIBRARY=' + os.path.join(pthreads_dir, 'lib', arch, 'pthreadVC2.lib') + ' '
                     if use_rawspeed else '') +\
                    '-DCMAKE_SKIP_INSTALL_ALL_DEPENDENCY=ON ' +\
                    '-DCMAKE_INSTALL_PREFIX:PATH=install',
            'nmake raw_r',
            'nmake install'
            ]
    for cmd in cmds:
        print(cmd)
        code = os.system(cmd)
        if code != 0:
            sys.exit(code)
    os.chdir(cwd)
    
    # bundle runtime dlls
    dll_runtime_libs = [('raw_r.dll', os.path.join(install_dir, 'bin'))]
    if use_rawspeed:
        dll_runtime_libs.extend([
            ('pthreadVC2.dll', os.path.join(pthreads_dir, 'lib', arch))
            ])
    
    # openmp dll
    isVS2008 = sys.version_info < (3, 3)
    isVS2010 = (3, 3) <= sys.version_info < (3, 5)
    isVS2015 = (3, 5) <= sys.version_info
    
    libraw_configh = os.path.join(install_dir, 'include', 'libraw', 'libraw_config.h')
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
    elif isVS2015:
        if not hasOpenMpSupport:
            raise Exception('OpenMP not available but should be, see error messages above')
        if is64Bit:
            omp = [r'C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\redist\x64\Microsoft.VC140.OpenMP\vcomp140.dll']
        else:
            omp = [r'C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\redist\x86\Microsoft.VC140.OPENMP\vcomp140.dll']
    
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
 
def mac_libraw_compile():
    clone_submodules()
        
    # configure and compile libraw
    cwd = os.getcwd()
    if not os.path.exists(cmake_build):
        os.mkdir(cmake_build)
    os.chdir(cmake_build)
    
    patch_cmakelists()
    
    install_name_dir = os.path.join(install_dir, 'lib')
    cmds = ['cmake .. -DCMAKE_BUILD_TYPE=Release ' +\
                    '-DENABLE_EXAMPLES=OFF -DENABLE_RAWSPEED=OFF ' +\
                    ('-DENABLE_DEMOSAIC_PACK_GPL2=ON -DDEMOSAIC_PACK_GPL2_RPATH=../LibRaw-demosaic-pack-GPL2 ' +\
                     '-DENABLE_DEMOSAIC_PACK_GPL3=ON -DDEMOSAIC_PACK_GPL3_RPATH=../LibRaw-demosaic-pack-GPL3 '
                     if buildGPLCode else '') +\
                    '-DCMAKE_SKIP_INSTALL_ALL_DEPENDENCY=ON ' +\
                    '-DCMAKE_INSTALL_PREFIX:PATH=install -DCMAKE_MACOSX_RPATH=0 -DCMAKE_INSTALL_NAME_DIR=' + install_name_dir,
            'make raw_r',
            'make install'
            ]
    for cmd in cmds:
        print(cmd)
        code = os.system(cmd)
        if code != 0:
            sys.exit(code)
    os.chdir(cwd)
        
package_data = {}

# evil hack, check cmd line for relevant commands
# custom cmdclasses didn't work out in this case
cmdline = ''.join(sys.argv[1:])
needsCompile = any(s in cmdline for s in ['install', 'bdist', 'build_ext', 'nosetests'])
if isWindows and needsCompile:
    windows_libraw_compile()        
    package_data['rawpy'] = ['*.dll']

elif isMac and needsCompile:
    mac_libraw_compile()        
    
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

pyx_path = os.path.join('rawpy', '_rawpy.pyx')
c_path = os.path.join('rawpy', '_rawpy.c')
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
              define_macros=define_macros,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
             )]

if use_cython:
    extensions = cythonize(extensions)

# version handling from https://stackoverflow.com/a/7071358
VERSIONFILE="rawpy/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

install_requires = ['numpy']
if sys.version_info < (3, 4):
    # Backport of Python 3.4 enums to earlier versions
    install_requires.append('enum34')

setup(
      name = 'rawpy',
      version = verstr,
      description = 'Python wrapper for the LibRaw library',
      long_description = open('README.rst').read(),
      author = 'Maik Riechert',
      author_email = 'maik.riechert@arcor.de',
      url = 'https://github.com/letmaik/rawpy',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Software Development :: Libraries',
      ],
      packages = find_packages(),
      ext_modules = extensions,
      package_data = package_data,
      install_requires=install_requires
)
