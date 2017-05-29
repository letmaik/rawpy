echo "APPVEYOR_REPO_TAG: " + $env:APPVEYOR_REPO_TAG
echo "APPVEYOR_REPO_TAG_NAME: " + $env:APPVEYOR_REPO_TAG_NAME

if (($env:APPVEYOR_REPO_TAG -eq "True") -and ($env:APPVEYOR_REPO_TAG_NAME.StartsWith("v"))) {
  (Get-Content appveyor\.pypirc) | Foreach-Object {$_ -replace '%PASS%',$env:PYPI_PASS} | Set-Content $env:userprofile\.pypirc
  Invoke-Expression "$env:CMD_IN_ENV python setup.py bdist_wheel upload"
}
