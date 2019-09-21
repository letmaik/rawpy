@ECHO ON

dir "C:\"
dir "C:\Program Files"
dir "C:\Program Files (x86)"

rem Install Python and pip
rem pwsh .\appveyor\install.ps1 || goto :error

rem CALL "%PYTHON%\\Scripts\\activate.bat"

:error
exit /b %errorlevel%