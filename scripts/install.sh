#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash scripts/install.sh /path/to/TOPECL" >&2
    exit 2
fi

target="$(cd "$1" && pwd)"
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

git -C "$target" apply --whitespace=nowarn \
    "$source_root/patches/topecl-caar.patch"
cp -R "$source_root/overlay/." "$target/"

echo "CAAR installed into $target"
