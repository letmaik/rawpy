$ErrorActionPreference = 'Stop'

function exec {
    [CmdletBinding()]
    param([Parameter(Position=0,Mandatory=1)][scriptblock]$cmd)
    Write-Host "$cmd"
    $ErrorActionPreference = 'Continue'
    & $cmd
    $ErrorActionPreference = 'Stop'
    if ($lastexitcode -ne 0) {
        throw ("ERROR exit code $lastexitcode")
    }
}

function Initialize-VS {
    $VS_ROOTS = @(
        "C:\Program Files\Microsoft Visual Studio",
        "C:\Program Files (x86)\Microsoft Visual Studio"
    )
    $VS_VERSIONS = @("2017", "2019", "2022")
    $VS_EDITIONS = @("Enterprise", "Professional", "Community")
    $VS_INIT_CMD_SUFFIX = "Common7\Tools\vsdevcmd.bat"

    $VS_ARCH = if ($env:PYTHON_ARCH -eq 'x86') { 'x86' } else { 'x64' }
    $VS_INIT_ARGS = "-arch=$VS_ARCH -no_logo"

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

Initialize-VS

# Check Python version
exec { python -c "import platform; assert platform.python_version().startswith('$env:PYTHON_VERSION')" }

# Install vcpkg dependencies (needed for building from source)
if (!(Test-Path ./vcpkg)) {
    exec { git clone https://github.com/microsoft/vcpkg -b 2025.01.13 --depth 1 }
    exec { ./vcpkg/bootstrap-vcpkg }
}
exec { ./vcpkg/vcpkg install zlib libjpeg-turbo[jpeg8] jasper lcms --triplet=x64-windows-static --recurse }
$env:CMAKE_PREFIX_PATH = $pwd.Path + "\vcpkg\installed\x64-windows-static"

# Create a clean venv and install the sdist
exec { python -m venv sdist-test-env }
& .\sdist-test-env\scripts\activate
exec { python -m pip install --upgrade pip }

$sdist = Get-ChildItem dist\rawpy-*.tar.gz | Select-Object -First 1
exec { pip install "$($sdist.FullName)[test]" }

# Run tests from a temp directory to avoid importing from the source tree
mkdir -f tmp_for_test | out-null
pushd tmp_for_test
exec { pytest --verbosity=3 -s ../test }
popd

deactivate
