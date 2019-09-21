@ECHO ON

dir "C:\"
dir "C:\Program Files"
dir "C:\Program Files\Microsoft SDKs"
dir "C:\Program Files (x86)"
dir "C:\Program Files (x86)\Microsoft SDKs"

CALL "%CONDA%\Scripts\activate.bat"

conda create --yes --name pyenv_build python=%PYTHON_VERSION% numpy=%NUMPY_VERSION% cython jpeg zlib || goto :error
conda activate pyenv_build || goto :error

rem Check that we have the expected version and architecture for Python
python --version
python -c "import struct; print(struct.calcsize('P') * 8)"
  
rem output what's installed
pip freeze
    
rem Build the compiled extension.
rem -u disables output buffering which caused intermixing of lines
rem when the external tools were started  
cmd /E:ON /V:ON /C .\appveyor\run_with_env.cmd python -u setup.py bdist_wheel || goto :error

:error
exit /b %errorlevel%