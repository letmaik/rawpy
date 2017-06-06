echo "APPVEYOR_REPO_TAG: " + $env:APPVEYOR_REPO_TAG
echo "APPVEYOR_REPO_TAG_NAME: " + $env:APPVEYOR_REPO_TAG_NAME

if (($env:APPVEYOR_REPO_TAG -eq "True") -and ($env:APPVEYOR_REPO_TAG_NAME.StartsWith("v"))) {
  # save credentials in ~\.pypirc
  (Get-Content appveyor\.pypirc) | Foreach-Object {$_ -replace '%PASS%',$env:PYPI_PASS} | Set-Content $env:userprofile\.pypirc
  
  # build and upload binary wheel
  Invoke-Expression "$env:CMD_IN_ENV python setup.py bdist_wheel upload"
  
  # build and upload docs
  if ($env:PYTHON -eq "C:\Python36_64") {
    Invoke-Expression "$env:CMD_IN_ENV python setup.py build_ext --inplace"
    Invoke-Expression "python setup.py build_sphinx"
    Invoke-Expression "python setup.py upload_docs"
  }
}
