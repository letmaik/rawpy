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

if (!$env:PYTHON_VERSION) {
    throw "PYTHON_VERSION env var missing, must be x.y"
}
if ($env:PYTHON_ARCH -ne 'x86' -and $env:PYTHON_ARCH -ne 'x86_64') {
    throw "PYTHON_ARCH env var must be x86 or x86_64"
}
if (!$env:NUMPY_VERSION) {
    throw "NUMPY_VERSION env var missing"
}

$PYVER = ($env:PYTHON_VERSION).Replace('.', '')

# Check Python version/arch
exec { python -c "import platform; assert platform.python_version().startswith('$env:PYTHON_VERSION')" }

# Upgrade pip and prefer binary packages
exec { python -m pip install --upgrade pip }
$env:PIP_PREFER_BINARY = 1

Get-ChildItem env:

# Install and import in an empty environment.
# This is to catch DLL issues that may be hidden with dependencies.
exec { python -m venv env\import-test }
& .\env\import-test\scripts\activate
python -m pip uninstall -y rawpy
ls dist\*cp${PYVER}*win*.whl | % { exec { python -m pip install $_ } }

# Avoid using in-source package during tests
mkdir -f tmp_for_test | out-null
pushd tmp_for_test
exec { python -c "import rawpy" }
popd

deactivate

# Run test suite with all required and optional dependencies
exec { python -m venv env\testsuite }
& .\env\testsuite\scripts\activate
python -m pip uninstall -y rawpy
ls dist\*cp${PYVER}*win*.whl | % { exec { python -m pip install $_ } }
exec { python -m pip install -r dev-requirements.txt numpy==$env:NUMPY_VERSION }

# Avoid using in-source package during tests
mkdir -f tmp_for_test | out-null
pushd tmp_for_test
exec { pytest --verbosity=3 -s ../test }
popd

deactivate
