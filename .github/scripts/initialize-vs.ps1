# Discover the installed toolchain rather than assuming a VS version or edition.
function Initialize-VS {
    param([Parameter(Mandatory=$true)][string]$Architecture)

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (!(Test-Path $vswhere)) {
        throw "Visual Studio locator not found: $vswhere"
    }
    $VS_INIT_CMD = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find Common7\Tools\VsDevCmd.bat
    if ($LASTEXITCODE -ne 0 -or !$VS_INIT_CMD) {
        throw "No suitable Visual Studio installation found"
    }

    $VS_INIT_ARGS = "-arch=$Architecture -no_logo"
    Write-Host "Executing: $VS_INIT_CMD $VS_INIT_ARGS"

    # Import the developer environment for CMake and setuptools.
    & "${env:COMSPEC}" /s /c "`"$VS_INIT_CMD`" $VS_INIT_ARGS && set" | ForEach-Object {
        $name, $value = $_ -split '=', 2
        if ($name -and $null -ne $value) {
            Set-Content -LiteralPath "env:\$name" -Value $value
        }
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Visual Studio environment initialization failed with exit code $LASTEXITCODE"
    }
}
