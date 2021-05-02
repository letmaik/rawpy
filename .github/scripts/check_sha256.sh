#!/bin/bash
set -e

path=$1
expected=$2

if [ "$(uname)" == "Darwin" ]; then
    actual=$(shasum -a 256 "$path" | cut -d ' ' -f 1)
elif [ "$(uname)" == "Linux" ]; then
    actual=$(sha256sum "$path" | cut -d ' ' -f 1)
else
    echo "Unknown system: $(uname)"
    exit 1
fi

if [ "$expected" != "$actual" ]; then 
  echo "CHECKSUM MISMATCH: $path"
  echo "Expected: $expected"
  echo "Actual: $actual"
  exit 1
fi
