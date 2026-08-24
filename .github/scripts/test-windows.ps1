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
switch ($env:PYTHON_ARCH) {
    'x86' {
        $PYTHON_PLATFORM = 'win32'
        $WHEEL_PLATFORM = 'win32'
    }
    'x86_64' {
        $PYTHON_PLATFORM = 'win-amd64'
        $WHEEL_PLATFORM = 'win_amd64'
    }
    'arm64' {
        $PYTHON_PLATFORM = 'win-arm64'
        $WHEEL_PLATFORM = 'win_arm64'
    }
    default {
        throw "PYTHON_ARCH env var must be x86, x86_64, or arm64"
    }
}
if (!$env:NUMPY_VERSION) {
    throw "NUMPY_VERSION env var missing"
}

$PYVER = ($env:PYTHON_VERSION).Replace('.', '')

# Check Python version and architecture
exec { python -c "import platform, sysconfig; assert platform.python_version().startswith('$env:PYTHON_VERSION'); assert sysconfig.get_platform() == '$PYTHON_PLATFORM', sysconfig.get_platform()" }

$wheels = @(Get-ChildItem "dist\*cp${PYVER}*${WHEEL_PLATFORM}.whl")
if ($wheels.Count -ne 1) {
    throw "Expected exactly one CPython $env:PYTHON_VERSION $WHEEL_PLATFORM wheel, found $($wheels.Count)"
}
$wheel = $wheels[0].FullName

# Upgrade pip and prefer binary packages
exec { python -m pip install --upgrade pip }
$env:PIP_PREFER_BINARY = 1

Get-ChildItem env:

# Install and import in an empty environment.
# This is to catch DLL issues that may be hidden with dependencies.
exec { python -m venv env\import-test }
& .\env\import-test\scripts\activate
python -m pip uninstall -y rawpy
exec { python -m pip install $wheel }

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
exec { python -m pip install $wheel }
exec { python -m pip install -r dev-requirements.txt numpy==$env:NUMPY_VERSION }

# Avoid using in-source package during tests
mkdir -f tmp_for_test | out-null
pushd tmp_for_test
exec { pytest --verbosity=3 -s ../test }
popd

deactivate
