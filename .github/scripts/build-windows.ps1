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
        throw ("ERROR exit code " -f $lastexitcode)
    }
}

Get-ChildItem env:

$env:CONDA_ROOT = $pwd.Path + "\external\miniconda_$env:PYTHON_ARCH"
& .\.github\scripts\install-miniconda.ps1

& $env:CONDA_ROOT\shell\condabin\conda-hook.ps1

conda update --yes -n base -c defaults conda

conda env remove -n pyenv_build
exec { conda create --yes --name pyenv_build python=$env:PYTHON_VERSION numpy=$env:NUMPY_VERSION cython jpeg zlib }
exec { conda activate pyenv_build }

# Check that we have the expected version and architecture for Python
exec { python --version }
exec { python -c "import struct; assert struct.calcsize('P') * 8 == $env:PYTHON_ARCH" }

# output what's installed
exec { pip freeze }

# Build the compiled extension.
# -u disables output buffering which caused intermixing of lines
# when the external tools were started  
exec { cmd /E:ON /V:ON /C .\.github\scripts\run_with_env.cmd python -u setup.py bdist_wheel }

# Test
exec { conda create --yes --name pyenv_test python=$env:PYTHON_VERSION numpy scikit-image }
exec { conda activate pyenv_test }
ls dist\*.whl | % { exec { pip install $_ } }
exec { pip install -r dev-requirements.txt }
mkdir tmp_for_test
cd tmp_for_test
exec { nosetests --verbosity=3 --nocapture ../test }
cd ..
