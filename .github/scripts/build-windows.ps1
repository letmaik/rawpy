$ErrorActionPreference = 'Stop'

function exec {
    [CmdletBinding()]
    param([Parameter(Position=0,Mandatory=1)][scriptblock]$cmd)
    Write-Host "$cmd"
    # https://stackoverflow.com/q/2095088
    $ErrorActionPreference = 'Continue'
    & $cmd
    $ErrorActionPreference = 'Stop'
    if ($lastexitcode -ne 0) {
        throw ("ERROR exit code $lastexitcode")
    }
}

function Initialize-VS {
    param([Parameter(Mandatory=$true)][string]$Architecture)

    # https://wiki.python.org/moin/WindowsCompilers
    # setuptools automatically selects the right compiler for building
    # the extension module. The following is mostly for building any
    # native dependencies, here via CMake.
    # https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line
    # https://docs.microsoft.com/en-us/cpp/porting/binary-compat-2015-2017

    $VS_ROOTS = @(
        "C:\Program Files\Microsoft Visual Studio",
        "C:\Program Files (x86)\Microsoft Visual Studio"
    )
    $VS_VERSIONS = @("2017", "2019", "2022")
    $VS_EDITIONS = @("Enterprise", "Professional", "Community")
    $VS_INIT_CMD_SUFFIX = "Common7\Tools\vsdevcmd.bat"

    $VS_INIT_ARGS = "-arch=$Architecture -no_logo"

    $found = $false
    :outer foreach ($VS_ROOT in $VS_ROOTS) {
        foreach ($version in $VS_VERSIONS) {
            foreach ($edition in $VS_EDITIONS) {
                $VS_INIT_CMD = "$VS_ROOT\$version\$edition\$VS_INIT_CMD_SUFFIX"
                if (Test-Path $VS_INIT_CMD) {
                    $found = $true
                    break outer
                }
            }
        }
    }

    if (!$found) {
        throw ("No suitable Visual Studio installation found")
    }

    Write-Host "Executing: $VS_INIT_CMD $VS_INIT_ARGS"

    # https://github.com/Microsoft/vswhere/wiki/Start-Developer-Command-Prompt
    & "${env:COMSPEC}" /s /c "`"$VS_INIT_CMD`" $VS_INIT_ARGS && set" | foreach-object {
        $name, $value = $_ -split '=', 2
        try {
            set-content env:\"$name" $value
        } catch {
        }
    }
}

if (!$env:PYTHON_VERSION) {
    throw "PYTHON_VERSION env var missing, must be x.y"
}
switch ($env:PYTHON_ARCH) {
    'x86' {
        $VS_ARCH = 'x86'
        $VCPKG_TRIPLET = 'x86-windows-static'
        $PYTHON_PLATFORM = 'win32'
    }
    'x86_64' {
        $VS_ARCH = 'x64'
        $VCPKG_TRIPLET = 'x64-windows-static'
        $PYTHON_PLATFORM = 'win-amd64'
    }
    'arm64' {
        $VS_ARCH = 'arm64'
        $VCPKG_TRIPLET = 'arm64-windows-static'
        $PYTHON_PLATFORM = 'win-arm64'
    }
    default {
        throw "PYTHON_ARCH env var must be x86, x86_64, or arm64"
    }
}

Initialize-VS -Architecture $VS_ARCH

# Check Python version and architecture
exec { python -c "import platform, sysconfig; assert platform.python_version().startswith('$env:PYTHON_VERSION'); assert sysconfig.get_platform() == '$PYTHON_PLATFORM', sysconfig.get_platform()" }

# Prefer binary packages over building from source
$env:PIP_PREFER_BINARY = 1

Get-ChildItem env:

# Install vcpkg and build dependencies
if (!(Test-Path ./vcpkg)) {
    exec { git clone https://github.com/microsoft/vcpkg -b 2026.07.29 --depth 1}
    exec { ./vcpkg/bootstrap-vcpkg }
}
exec { ./vcpkg/vcpkg install zlib libjpeg-turbo[jpeg8] jasper lcms --triplet=$VCPKG_TRIPLET --recurse }
$env:CMAKE_PREFIX_PATH = $pwd.Path + "\vcpkg\installed\$VCPKG_TRIPLET"

# Build the wheel in a virtual environment
exec { python -m venv env\build }
& .\env\build\scripts\activate
exec { python -m pip install --upgrade pip }
exec { python -m pip wheel . --wheel-dir dist --no-deps }
deactivate
