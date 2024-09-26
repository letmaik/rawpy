Set-Location $PSScriptRoot
Set-Location ".."
pixi install
./glatzel/init_build.ps1
pixi run -e vfx2024 python -u setup.py bdist_wheel