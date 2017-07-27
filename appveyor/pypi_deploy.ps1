Write-Host "APPVEYOR_REPO_TAG: $env:APPVEYOR_REPO_TAG"
Write-Host "APPVEYOR_REPO_TAG_NAME: $env:APPVEYOR_REPO_TAG_NAME" 

if (($env:APPVEYOR_REPO_TAG -eq "True") -and ($env:APPVEYOR_REPO_TAG_NAME.StartsWith("v"))) {
  # save credentials in ~\.pypirc
  (Get-Content appveyor\.pypirc) `
    | Foreach-Object {$_ -replace '%USER%',$env:PYPI_USER -replace '%PASS%',$env:PYPI_PASS} `
    | Set-Content $env:userprofile\.pypirc
  
  # build and upload binary wheel
  Invoke-Expression "$env:CMD_IN_ENV python setup.py bdist_wheel upload"
}
