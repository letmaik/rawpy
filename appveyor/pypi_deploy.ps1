$gitinfo = git log -n 1 --pretty=format:'%d'
echo "git log: $gitinfo"
if ($gitinfo -match "tag: v") {
  (Get-Content appveyor\.pypirc) | Foreach-Object {$_ -replace '%PASS%',$env:PYPI_PASS} | Set-Content $env:userprofile\.pypirc
  Invoke-Expression "$env:CMD_IN_ENV python setup.py bdist_wheel upload"
}