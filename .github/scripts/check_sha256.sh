#!/bin/bash
set -e

path=$1
expected=$2

actual=$(sha256sum "$1" | cut -d ' ' -f 1)
if [ "$expected" != "$actual" ]; then 
  echo "CHECKSUM MISMATCH: $path"
  echo "Expected: $expected"
  echo "Actual: $actual"
  exit 1
fi
