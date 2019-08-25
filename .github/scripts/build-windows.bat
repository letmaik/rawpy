dir "C:\"
dir "C:\Program Files\Microsoft SDKs\Windows"

rem Install Python and pip
rem pwsh .\appveyor\install.ps1 || goto :error

rem CALL "%PYTHON%\\Scripts\\activate.bat"

:error
exit /b %errorlevel%