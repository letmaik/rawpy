$ErrorActionPreference = 'Stop'

function not-exist { -not (Test-Path $args) }
Set-Alias !exists not-exist -Option "Constant, AllScope"
Set-Alias exists Test-Path -Option "Constant, AllScope"

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

function Init-VS {
    # https://wiki.python.org/moin/WindowsCompilers
    # setuptools automatically selects the right compiler for building
    # the extension module. The following is mostly for building any
    # dependencies like libraw.
    # FIXME choose matching VC++ compiler, maybe using -vcvars_ver
    #   -> dependencies should not be built with newer compiler than Python itself
    # https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line
    # https://docs.microsoft.com/en-us/cpp/porting/binary-compat-2015-2017

    $VS2015_ROOT = "C:\Program Files (x86)\Microsoft Visual Studio 14.0"
    $VS2017_ROOT = "C:\Program Files (x86)\Microsoft Visual Studio\2017"
    $VS2019_ROOT = "C:\Program Files (x86)\Microsoft Visual Studio\2019"

    $VS_ARCH = if ($env:PYTHON_ARCH -eq '32') { 'x86' } else { 'x64' }

    $PYTHON_VERSION_TUPLE = $env:PYTHON_VERSION.split('.')
    $PYTHON_VERSION_MAJOR = $PYTHON_VERSION_TUPLE[0]
    $PYTHON_VERSION_MINOR = $PYTHON_VERSION_TUPLE[1]

    if ($PYTHON_VERSION_MAJOR -eq '3') {
        if ($PYTHON_VERSION_MINOR -le '4') {
            throw ("Python <= 3.4 unsupported: $env:PYTHON_VERSION")
        }
        if (exists $VS2015_ROOT) {
            $VS_VERSION = "2015"
            $VS_ROOT = $VS2015_ROOT
		    $VS_INIT_CMD = "$VS_ROOT\VC\vcvarsall.bat"
		    $VS_INIT_ARGS = "$VS_ARCH"
        } elseif (exists $VS2017_ROOT) {
            $VS_VERSION = "2017"
            if (exists "$VS2017_ROOT\Enterprise") {
                $VS_ROOT = "$VS2017_ROOT\Enterprise"
            } else {
                $VS_ROOT = "$VS2017_ROOT\Community"
            }
            $VS_INIT_CMD = "$VS_ROOT\Common7\Tools\vsdevcmd.bat"
            $VS_INIT_ARGS = "-arch=$VS_ARCH -no_logo"
        } elseif (exists $VS2019_ROOT) {
            $VS_VERSION = "2019"
            if (exists "$VS2019_ROOT\Enterprise") {
                $VS_ROOT = "$VS2019_ROOT\Enterprise"
            } else {
                $VS_ROOT = "$VS2019_ROOT\Community"
            }
            $VS_INIT_CMD = "$VS_ROOT\Common7\Tools\vsdevcmd.bat"
            $VS_INIT_ARGS = "-arch=$VS_ARCH -no_logo"
        } else {
            throw ("No suitable Visual Studio installation found")
        }
    } else {
        throw ("Unsupported Python version: $PYTHON_VERSION_MAJOR")
    }
    Write-Host "Configuring VS$VS_VERSION for Python $env:PYTHON_VERSION on a $env:PYTHON_ARCH bit architecture"
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
if ($env:PYTHON_ARCH -ne '32' -and $env:PYTHON_ARCH -ne '64') {
    throw "PYTHON_ARCH env var must be 32 or 64"
}
if (!$env:NUMPY_VERSION) {
    throw "NUMPY_VERSION env var missing"
}

Init-VS

Get-ChildItem env:

$env:CONDA_ROOT = $pwd.Path + "\external\miniconda_$env:PYTHON_ARCH"
& .\.github\scripts\install-miniconda.ps1

& $env:CONDA_ROOT\shell\condabin\conda-hook.ps1

exec { conda update --yes -n base -c defaults conda }

exec { conda create --yes --name pyenv_build python=$env:PYTHON_VERSION numpy=$env:NUMPY_VERSION cython jpeg zlib --force }
exec { conda activate pyenv_build }

# Check that we have the expected version and architecture for Python
exec { python --version }
exec { python -c "import struct; assert struct.calcsize('P') * 8 == $env:PYTHON_ARCH" }

# output what's installed
exec { pip freeze }

# Build the compiled extension.
# -u disables output buffering which caused intermixing of lines
# when the external tools were started  
exec { python -u setup.py bdist_wheel }

# Test
exec { conda create --yes --name pyenv_test python=$env:PYTHON_VERSION numpy scikit-image --force }
exec { conda activate pyenv_test }
ls dist\*.whl | % { exec { pip install $_ } }
exec { pip install -r dev-requirements.txt }
mkdir tmp_for_test | out-null
cd tmp_for_test
exec { nosetests --verbosity=3 --nocapture ../test }
cd ..
