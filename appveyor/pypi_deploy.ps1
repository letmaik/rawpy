if (($env:appveyor_repo_tag -eq "True") -and ($env:appveyor_repo_branch.StartsWith("v"))) {
  (Get-Content appveyor\.pypirc) | Foreach-Object {$_ -replace '%PASS%',$env:PYPI_PASS} | Set-Content $env:userprofile\.pypirc
  Invoke-Expression "$env:CMD_IN_ENV python setup.py bdist_wheel upload"
}