:: To build extensions for Python 3.5, you need:
:: MS Visual Studio 2015 Community
:: (Note: Windows SDK for Windows 10 does NOT contain compilers!)
::
:: To build extensions for Python 3.4, you need:
:: MS Windows SDK for Windows 7 and .NET Framework 4 (SDK v7.1)
:: (or MS Visual Studio 2010 C++)
::
:: To build extensions for Python 2.7, you need:
:: MS Windows SDK for Windows 7 and .NET Framework 3.5 (SDK v7.0)
:: (or MS Visual Studio 2008 C++)
::
:: Note: this script needs to be run with the /E:ON and /V:ON flags for the
:: cmd interpreter
::
:: More details at:
:: http://stackoverflow.com/a/13751649
::
:: Authors: Olivier Grisel, Maik Riechert
:: License: CC0 1.0 Universal: http://creativecommons.org/publicdomain/zero/1.0/
@ECHO OFF

SET COMMAND_TO_RUN=%*
SET WIN_SDK_ROOT=C:\Program Files\Microsoft SDKs\Windows
SET VS2015_ROOT=C:\Program Files (x86)\Microsoft Visual Studio 14.0

:: Extract the major and minor versions, and allow for the minor version to be
:: more than 9.  This requires the version number to have two dots in it.
SET MAJOR_PYTHON_VERSION=%PYTHON_VERSION:~0,1%
IF "%PYTHON_VERSION:~3,1%" == "." (
    SET MINOR_PYTHON_VERSION=%PYTHON_VERSION:~2,1%
) ELSE (
    SET MINOR_PYTHON_VERSION=%PYTHON_VERSION:~2,2%
)

IF %MAJOR_PYTHON_VERSION% == 2 (
    SET WINDOWS_SDK_VERSION="v7.0"
) ELSE IF %MAJOR_PYTHON_VERSION% == 3 (
  IF %MINOR_PYTHON_VERSION% LEQ 4 (
    SET WINDOWS_SDK_VERSION="v7.1"
  ) ELSE (
    SET WINDOWS_SDK_VERSION="2015"
  )
) ELSE (
    ECHO Unsupported Python version: "%MAJOR_PYTHON_VERSION%"
    EXIT 1
)

IF "%PYTHON_ARCH%"=="64" (
	IF %WINDOWS_SDK_VERSION%=="2015" (
	    ECHO Configuring VS2015 for Python %MAJOR_PYTHON_VERSION%.%MINOR_PYTHON_VERSION% on a 64 bit architecture
	    "%VS2015_ROOT%\VC\vcvarsall.bat" x64
	    ECHO Executing: %COMMAND_TO_RUN%
		call %COMMAND_TO_RUN% || EXIT 1
	) ELSE (
		ECHO Configuring Windows SDK %WINDOWS_SDK_VERSION% for Python %MAJOR_PYTHON_VERSION% on a 64 bit architecture
	    SET DISTUTILS_USE_SDK=1
	    SET MSSdk=1
	    "%WIN_SDK_ROOT%\%WINDOWS_SDK_VERSION%\Setup\WindowsSdkVer.exe" -q -version:%WINDOWS_SDK_VERSION%
	    "%WIN_SDK_ROOT%\%WINDOWS_SDK_VERSION%\Bin\SetEnv.cmd" /x64 /release
	    ECHO Executing: %COMMAND_TO_RUN%
		call %COMMAND_TO_RUN% || EXIT 1
	)
) ELSE (
	IF %WINDOWS_SDK_VERSION%=="2015" (
	    ECHO Configuring VS2015 for Python %MAJOR_PYTHON_VERSION%.%MINOR_PYTHON_VERSION% on a 32 bit architecture
	    "%VS2015_ROOT%\VC\vcvarsall.bat" x86
	    ECHO Executing: %COMMAND_TO_RUN%
		call %COMMAND_TO_RUN% || EXIT 1
	) ELSE (
	    ECHO Configuring Windows SDK %WINDOWS_SDK_VERSION% for Python %MAJOR_PYTHON_VERSION% on a 32 bit architecture
	    "%WIN_SDK_ROOT%\%WINDOWS_SDK_VERSION%\Setup\WindowsSdkVer.exe" -q -version:%WINDOWS_SDK_VERSION%
	    "%WIN_SDK_ROOT%\%WINDOWS_SDK_VERSION%\Bin\SetEnv.cmd" /x86 /release
	    ECHO Executing: %COMMAND_TO_RUN%
		call %COMMAND_TO_RUN% || EXIT 1
	)
)
