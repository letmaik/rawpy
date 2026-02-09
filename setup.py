from setuptools import setup, Extension, find_packages
import subprocess
import errno
import os
import shutil
import sys
import glob

import numpy

from Cython.Build import cythonize

# --- Configuration ---

# As rawpy is distributed under the MIT license, it cannot use or distribute
# GPL'd code. This is relevant only for the binary wheels which would have to
# bundle the GPL'd code/algorithms (extra demosaic packs).
buildGPLCode = os.getenv("RAWPY_BUILD_GPL_CODE") == "1"
useSystemLibraw = os.getenv("RAWPY_USE_SYSTEM_LIBRAW") == "1"

# Platform detection
isWindows = os.name == "nt" and "GCC" not in sys.version
isMac = sys.platform == "darwin"
isLinux = sys.platform.startswith("linux")
is64Bit = sys.maxsize > 2**32

# --- Compiler/Linker Flags ---

libraries = ["libraw_r"]
include_dirs = [numpy.get_include()]  # Always include numpy headers
library_dirs = []
extra_compile_args = []
extra_link_args = []
define_macros = []

if isWindows:
    extra_compile_args += ["/DWIN32"]

if isLinux:
    # On Linux, we want the extension to find the bundled libraw_r.so in the same directory
    extra_link_args += ["-Wl,-rpath,$ORIGIN"]

if isMac:
    # On macOS, @loader_path is the equivalent of $ORIGIN — it resolves to
    # the directory containing the binary that references the dylib.
    extra_link_args += ["-Wl,-rpath,@loader_path"]

# --- Helper Functions ---


def _ask_pkg_config(resultlist, option, result_prefix="", sysroot=False):
    pkg_config = os.environ.get("PKG_CONFIG", "pkg-config")
    try:
        p = subprocess.Popen([pkg_config, option, "libraw_r"], stdout=subprocess.PIPE)
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
            res = [x[len(result_prefix) :] for x in res]

            sysroot = sysroot and os.environ.get("PKG_CONFIG_SYSROOT_DIR", "")
            if sysroot:
                res = [
                    path if path.startswith(sysroot) else sysroot + path for path in res
                ]
            resultlist[:] = res


def use_pkg_config():
    pkg_config = os.environ.get("PKG_CONFIG", "pkg-config")
    if subprocess.call([pkg_config, "--atleast-version=0.21", "libraw_r"]) != 0:
        raise SystemExit("ERROR: System LibRaw is too old or not found. rawpy requires LibRaw >= 0.21.")
    _ask_pkg_config(include_dirs, "--cflags-only-I", "-I", sysroot=True)
    _ask_pkg_config(extra_compile_args, "--cflags-only-other")
    _ask_pkg_config(library_dirs, "--libs-only-L", "-L", sysroot=True)
    _ask_pkg_config(extra_link_args, "--libs-only-other")
    _ask_pkg_config(libraries, "--libs-only-l", "-l")


def clone_submodules():
    if not os.path.exists("external/LibRaw/libraw/libraw.h"):
        print(
            'LibRaw git submodule is not cloned yet, will invoke "git submodule update --init" now'
        )
        if os.system("git submodule update --init") != 0:
            raise Exception("git failed")


def get_cmake_build_dir():
    external_dir = os.path.abspath("external")
    return os.path.join(external_dir, "LibRaw-cmake", "build")


def get_install_dir():
    return os.path.join(get_cmake_build_dir(), "install")


def windows_libraw_compile():
    clone_submodules()

    cmake = "cmake"

    # openmp dll
    # VS 2017 and higher
    vc_redist_dir = os.getenv("VCToolsRedistDir")
    vs_target_arch = os.getenv("VSCMD_ARG_TGT_ARCH")
    if not vc_redist_dir:
        # VS 2015
        if "VCINSTALLDIR" in os.environ:
            vc_redist_dir = os.path.join(os.environ["VCINSTALLDIR"], "redist")
            vs_target_arch = "x64" if is64Bit else "x86"
        else:
            vc_redist_dir = None

    if vc_redist_dir and vs_target_arch:
        omp_glob = os.path.join(
            vc_redist_dir, vs_target_arch, "Microsoft.VC*.OpenMP", "vcomp*.dll"
        )
        omp_dlls = glob.glob(omp_glob)
    else:
        omp_dlls = []

    if len(omp_dlls) == 1:
        has_openmp_dll = True
        omp = omp_dlls[0]
    elif len(omp_dlls) > 1:
        print("WARNING: disabling OpenMP because multiple runtime DLLs were found:")
        for omp_dll in omp_dlls:
            print(omp_dll)
        has_openmp_dll = False
    else:
        print("WARNING: disabling OpenMP because no runtime DLLs were found")
        has_openmp_dll = False

    # configure and compile libraw
    cwd = os.getcwd()

    cmake_build = get_cmake_build_dir()
    install_dir = get_install_dir()
    libraw_dir = os.path.join(os.path.abspath("external"), "LibRaw")

    shutil.rmtree(cmake_build, ignore_errors=True)
    os.makedirs(cmake_build, exist_ok=True)
    os.chdir(cmake_build)

    # Important: always use Release build type, otherwise the library will depend on a
    #            debug version of OpenMP which is not what we bundle it with, and then it would fail
    enable_openmp_flag = "ON" if has_openmp_dll else "OFF"
    cmds = [
        cmake
        + ' .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release '
        + "-DCMAKE_PREFIX_PATH="
        + os.environ["CMAKE_PREFIX_PATH"]
        + " "
        + "-DLIBRAW_PATH="
        + libraw_dir.replace("\\", "/")
        + " "
        + "-DENABLE_X3FTOOLS=ON -DENABLE_6BY9RPI=ON "
        + "-DENABLE_EXAMPLES=OFF -DENABLE_OPENMP="
        + enable_openmp_flag
        + " -DENABLE_RAWSPEED=OFF "
        + (
            "-DENABLE_DEMOSAIC_PACK_GPL2=ON -DDEMOSAIC_PACK_GPL2_RPATH=../../LibRaw-demosaic-pack-GPL2 "
            + "-DENABLE_DEMOSAIC_PACK_GPL3=ON -DDEMOSAIC_PACK_GPL3_RPATH=../../LibRaw-demosaic-pack-GPL3 "
            if buildGPLCode
            else ""
        )
        + "-DCMAKE_INSTALL_PREFIX=install",
        cmake + " --build . --target install",
    ]
    for cmd in cmds:
        print(cmd)
        code = os.system(cmd)
        if code != 0:
            sys.exit(code)
    os.chdir(cwd)

    # bundle runtime dlls
    dll_runtime_libs = [("raw_r.dll", os.path.join(install_dir, "bin"))]

    if has_openmp_dll:
        # Check if OpenMP was enabled in the CMake build, independent of the flag we supplied.
        # If not, we don't have to bundle the DLL.
        libraw_configh = os.path.join(
            install_dir, "include", "libraw", "libraw_config.h"
        )
        match = "#define LIBRAW_USE_OPENMP 1"
        has_openmp_support = match in open(libraw_configh).read()
        if has_openmp_support:
            dll_runtime_libs.append((os.path.basename(omp), os.path.dirname(omp)))
        else:
            print(
                'WARNING: "#define LIBRAW_USE_OPENMP 1" not found even though OpenMP was enabled'
            )
            print("Will not bundle OpenMP runtime DLL")

    for filename, folder in dll_runtime_libs:
        src = os.path.join(folder, filename)
        dest = "rawpy/" + filename
        print("copying", src, "->", dest)
        shutil.copyfile(src, dest)


def unix_libraw_compile():
    """Compiles LibRaw using CMake on macOS and Linux."""
    clone_submodules()

    external_dir = os.path.abspath("external")
    libraw_dir = os.path.join(external_dir, "LibRaw")
    cmake_build = get_cmake_build_dir()
    install_dir = get_install_dir()

    cwd = os.getcwd()
    if not os.path.exists(cmake_build):
        os.makedirs(cmake_build, exist_ok=True)
    os.chdir(cmake_build)

    # Use @rpath so the dylib's install name becomes @rpath/libraw_r.<ver>.dylib.
    # Combined with -rpath @loader_path on the extension, dyld will find the
    # bundled dylib next to the .so at runtime. delocate (used in CI wheel
    # builds) rewrites these paths anyway, so this is compatible with both
    # plain pip installs and CI wheel builds.
    install_name_dir = "@rpath" if isMac else os.path.join(install_dir, "lib")

    # CMake arguments
    cmake_args = [
        "cmake",
        "..",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLIBRAW_PATH=" + libraw_dir,
        "-DENABLE_X3FTOOLS=ON",
        "-DENABLE_6BY9RPI=ON",
        "-DENABLE_OPENMP=OFF",
        "-DENABLE_EXAMPLES=OFF",
        "-DENABLE_RAWSPEED=OFF",
        "-DCMAKE_INSTALL_PREFIX=install",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        "-DCMAKE_INSTALL_NAME_DIR=" + install_name_dir,
    ]

    if buildGPLCode:
        cmake_args.extend(
            [
                "-DENABLE_DEMOSAIC_PACK_GPL2=ON",
                "-DDEMOSAIC_PACK_GPL2_RPATH=../../LibRaw-demosaic-pack-GPL2",
                "-DENABLE_DEMOSAIC_PACK_GPL3=ON",
                "-DDEMOSAIC_PACK_GPL3_RPATH=../../LibRaw-demosaic-pack-GPL3",
            ]
        )

    cmds = [" ".join(cmake_args), "cmake --build . --target install"]

    for cmd in cmds:
        print(f"Running: {cmd}")
        if os.system(cmd) != 0:
            sys.exit(f"Error executing: {cmd}")

    os.chdir(cwd)

    if isLinux or isMac:
        # When compiling LibRaw from source (not using system libraw), we
        # copy the shared libraries into the package directory so they get
        # bundled with the installed package (via package_data globs).
        # The extension uses rpath ($ORIGIN on Linux, @loader_path on macOS)
        # to find them at runtime.
        #
        # In CI, auditwheel (Linux) and delocate (macOS) further repair the
        # wheel, but for editable installs and plain `pip install .` we need
        # the libraries in-tree.
        lib_dir = os.path.join(install_dir, "lib")
        if isLinux:
            libs = glob.glob(os.path.join(lib_dir, "libraw_r.so*"))
        else:  # macOS
            libs = glob.glob(os.path.join(lib_dir, "libraw_r*.dylib"))
        for lib in libs:
            dest = os.path.join("rawpy", os.path.basename(lib))
            if os.path.islink(lib):
                if os.path.lexists(dest):
                    os.remove(dest)
                linkto = os.readlink(lib)
                os.symlink(linkto, dest)
            else:
                shutil.copyfile(lib, dest)
            print(f"Bundling {lib} -> {dest}")


# --- Main Logic ---

# Determine if we need to compile LibRaw from source
# If using system libraw (e.g. installed via apt), we check pkg-config
libraw_config_found = False

if (isWindows or isMac or isLinux) and not useSystemLibraw:
    # Build from source
    install_dir = get_install_dir()
    include_dirs += [os.path.join(install_dir, "include", "libraw")]
    library_dirs += [os.path.join(install_dir, "lib")]
    libraries = ["raw_r"]
    # If building from source, we know we have the config header
    libraw_config_found = True
else:
    # Use system library
    use_pkg_config()
    for include_dir in include_dirs:
        if "libraw_config.h" in os.listdir(include_dir):
            libraw_config_found = True
            break

# Ensure numpy headers are always included (use_pkg_config replaces the list)
if numpy.get_include() not in include_dirs:
    include_dirs.insert(0, numpy.get_include())

define_macros.append(("_HAS_LIBRAW_CONFIG_H", "1" if libraw_config_found else "0"))

# Package Data
package_data = {"rawpy": ["py.typed", "*.pyi"]}

# Evil hack to detect if we are building/installing
# (We don't want to compile libraw just for 'python setup.py --version')
cmdline = "".join(sys.argv[1:])
needsCompile = (
    any(s in cmdline for s in ["install", "bdist", "build_ext", "wheel", "develop"])
    and not useSystemLibraw
)

if needsCompile:
    if isWindows:
        windows_libraw_compile()
        package_data["rawpy"].append("*.dll")
    elif isMac or isLinux:
        unix_libraw_compile()
        if isLinux:
            package_data["rawpy"].append("*.so*")
        elif isMac:
            package_data["rawpy"].append("*.dylib")

# Clean up egg-info if needed
if any(s in cmdline for s in ["clean", "sdist"]):
    egg_info = "rawpy.egg-info"
    if os.path.exists(egg_info):
        print("removing", egg_info)
        shutil.rmtree(egg_info, ignore_errors=True)

# Extensions
extensions = cythonize(
    [
        Extension(
            "rawpy._rawpy",
            include_dirs=include_dirs,
            sources=[os.path.join("rawpy", "_rawpy.pyx")],
            libraries=libraries,
            library_dirs=library_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]
)

# Version
exec(open("rawpy/_version.py").read())

setup(
    version=__version__,
    packages=find_packages(),
    ext_modules=extensions,
    package_data=package_data,
)
