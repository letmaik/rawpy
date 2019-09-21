:: To build extensions for Python 3.5 and higher, you need:
:: MS Visual Studio 2015 Community
:: (Note: Windows SDK for Windows 10 does NOT contain compilers!)
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

where python
python --version

SET COMMAND_TO_RUN=%*
SET VS2015_ROOT=C:\Program Files (x86)\Microsoft Visual Studio 14.0
SET VS2017_ROOT=C:\Program Files (x86)\Microsoft Visual Studio\2017

:: Extract the major and minor versions, and allow for the minor version to be
:: more than 9.  This requires the version number to have two dots in it.
SET MAJOR_PYTHON_VERSION=%PYTHON_VERSION:~0,1%
IF "%PYTHON_VERSION:~3,1%" == "." (
    SET MINOR_PYTHON_VERSION=%PYTHON_VERSION:~2,1%
) ELSE (
    SET MINOR_PYTHON_VERSION=%PYTHON_VERSION:~2,2%
)

IF %MAJOR_PYTHON_VERSION% == 3 (
	IF EXIST %VS2017_ROOT% (
		SET VS_VERSION="2017"
		IF EXIST %VS2017_ROOT%\Enterprise (
			SET VS2017_ROOT=%VS2017_ROOT%\Enterprise
		) ELSE (
			SET VS2017_ROOT=%VS2017_ROOT%\Community
		)
	) ELSE (
		SET VS_VERSION="2015"
	)
) ELSE (
    ECHO Unsupported Python version: "%MAJOR_PYTHON_VERSION%"
    EXIT 1
)

IF "%PYTHON_ARCH%"=="64" (
	IF %VS_VERSION%=="2015" (
	    ECHO Configuring VS2015 for Python %MAJOR_PYTHON_VERSION%.%MINOR_PYTHON_VERSION% on a 64 bit architecture
	    "%VS2015_ROOT%\VC\vcvarsall.bat" x64
	    ECHO Executing: %COMMAND_TO_RUN%
		call %COMMAND_TO_RUN% || EXIT 1
	) ELSE IF %VS_VERSION%=="2017" (
	    ECHO Configuring VS2017 for Python %MAJOR_PYTHON_VERSION%.%MINOR_PYTHON_VERSION% on a 64 bit architecture
	    "%VS2017_ROOT%\Common7\Tools\vsdevcmd.bat" -arch=x64
	    ECHO Executing: %COMMAND_TO_RUN%
		call %COMMAND_TO_RUN% || EXIT 1
	)
) ELSE (
	IF %VS_VERSION%=="2015" (
	    ECHO Configuring VS2015 for Python %MAJOR_PYTHON_VERSION%.%MINOR_PYTHON_VERSION% on a 32 bit architecture
	    "%VS2015_ROOT%\VC\vcvarsall.bat" x86
	    ECHO Executing: %COMMAND_TO_RUN%
		call %COMMAND_TO_RUN% || EXIT 1
	) ELSE IF %VS_VERSION%=="2017" (
	    ECHO Configuring VS2017 for Python %MAJOR_PYTHON_VERSION%.%MINOR_PYTHON_VERSION% on a 32 bit architecture
	    "%VS2017_ROOT%\Common7\Tools\vsdevcmd.bat" -arch=x86
	    ECHO Executing: %COMMAND_TO_RUN%
		call %COMMAND_TO_RUN% || EXIT 1
	)
)
