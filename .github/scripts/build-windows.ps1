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

function Initialize-Python {
    if ($env:USE_CONDA -eq 1) {
        $env:CONDA_ROOT = $pwd.Path + "\external\miniconda_$env:PYTHON_ARCH"
        & .\.github\scripts\install-miniconda.ps1
        & $env:CONDA_ROOT\shell\condabin\conda-hook.ps1
        exec { conda update --yes -n base -c defaults conda }
    }
    # Check Python version
    exec { python -c "import platform; assert platform.python_version().startswith('$env:PYTHON_VERSION')" }
}

function Create-VEnv {
    [CmdletBinding()]
    param([Parameter(Position=0,Mandatory=1)][string]$name)
    if ($env:USE_CONDA -eq 1) {
        exec { conda create --yes --name $name -c defaults --strict-channel-priority python=$env:PYTHON_VERSION --force }
    } else {
        exec { python -m venv env\$name }
    }
}

function Enter-VEnv {
    [CmdletBinding()]
    param([Parameter(Position=0,Mandatory=1)][string]$name)
    if ($env:USE_CONDA -eq 1) {
        conda activate $name
    } else {
        & .\env\$name\scripts\activate
    }
}

function Create-And-Enter-VEnv {
    [CmdletBinding()]
    param([Parameter(Position=0,Mandatory=1)][string]$name)
    Create-VEnv $name
    Enter-VEnv $name
}

function Exit-VEnv {
    if ($env:USE_CONDA -eq 1) {
        conda deactivate
    } else {
        deactivate
    }
}

function Initialize-VS {
    # https://wiki.python.org/moin/WindowsCompilers
    # setuptools automatically selects the right compiler for building
    # the extension module. The following is mostly for building any
    # native dependencies, here via CMake.
    # https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line
    # https://docs.microsoft.com/en-us/cpp/porting/binary-compat-2015-2017

    $VS_ROOT = "C:\Program Files (x86)\Microsoft Visual Studio"
    $VS_VERSIONS = @("2017", "2019", "2022")
    $VS_EDITIONS = @("Enterprise", "Professional", "Community")
    $VS_INIT_CMD_SUFFIX = "Common7\Tools\vsdevcmd.bat"

    $VS_ARCH = if ($env:PYTHON_ARCH -eq 'x86') { 'x86' } else { 'x64' }
    $VS_INIT_ARGS = "-arch=$VS_ARCH -no_logo"

    $found = $false
    :outer foreach ($version in $VS_VERSIONS) {
        foreach ($edition in $VS_EDITIONS) {
            $VS_INIT_CMD = "$VS_ROOT\$version\$edition\$VS_INIT_CMD_SUFFIX"
            if (Test-Path $VS_INIT_CMD) {
                $found = $true
                break outer
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
if ($env:PYTHON_ARCH -ne 'x86' -and $env:PYTHON_ARCH -ne 'x86_64') {
    throw "PYTHON_ARCH env var must be x86 or x86_64"
}
if (!$env:NUMPY_VERSION) {
    throw "NUMPY_VERSION env var missing"
}

Initialize-VS
Initialize-Python

Get-ChildItem env:

# Install vcpkg and build dependencies
if (!(Test-Path ./vcpkg)) {
    exec { git clone https://github.com/microsoft/vcpkg -b 2025.01.13 --depth 1}
    exec { ./vcpkg/bootstrap-vcpkg }
}
exec { ./vcpkg/vcpkg install zlib libjpeg-turbo[jpeg8] jasper lcms --triplet=x64-windows-static --recurse }
$env:CMAKE_PREFIX_PATH = $pwd.Path + "\vcpkg\installed\x64-windows-static"


# Build the wheel.
Create-And-Enter-VEnv build
exec { python -m pip install --upgrade pip wheel setuptools }
exec { python -m pip install --only-binary :all: numpy==$env:NUMPY_VERSION cython }
exec { python -u setup.py bdist_wheel }
Exit-VEnv
