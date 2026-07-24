#!/usr/bin/env bash
# sync_binaries.sh -- vendor CI build artifacts into the wrapper bundle dirs.
#
# The wrappers already find libraries directly under bindings/binaries/ when run
# from a repo checkout (the loaders search ancestor `binaries/` trees). This
# script is for the *installed* case: it copies each platform's shared object
# out of binaries/libstiles-<ci>/lib/ into the locations that ship inside a pip
# wheel / R package, so an installed pysTiles or sTiles is self-contained:
#
#     python/pysTiles/_libs/<os>-<arch>/libstiles.{so,dylib}
#     R/sTiles/inst/libs/<os>-<arch>/libstiles.{so,dylib}
#
# Usage:
#   ./sync_binaries.sh          # only this machine's platform
#   ./sync_binaries.sh --all    # every platform present in binaries/
#
# The copied binaries are large and .gitignore'd; re-run after refreshing
# binaries/. For R, re-run `R CMD INSTALL sTiles` afterwards to pick them up.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bin_dir="$here/binaries"
py_libs="$here/python/sTiles/_libs"
r_libs="$here/R/sTiles/inst/libs"

all=0
[ "${1:-}" = "--all" ] && all=1

# Map a CI artifact folder name -> "<os>-<arch> filename".
map() {
    case "$1" in
        libstiles-linux-x86_64)       echo "linux-x86_64 libstiles.so" ;;
        libstiles-linux-arm64)        echo "linux-arm64 libstiles.so" ;;
        libstiles-macos-apple-arm64)  echo "macos-arm64 libstiles.dylib" ;;
        libstiles-macos-intel-x86_64) echo "macos-x86_64 libstiles.dylib" ;;
        libstiles-windows-x86_64)     echo "windows-x86_64 libstiles.dll" ;;
        *) echo "" ;;
    esac
}

# This machine's CI folder name (used when --all is not given).
this_ci() {
    local s m; s="$(uname -s)"; m="$(uname -m)"
    case "$m" in x86_64|amd64) m=x86_64 ;; arm64|aarch64) m=arm64 ;; esac
    if [ "$s" = "Darwin" ]; then
        [ "$m" = arm64 ] && echo "libstiles-macos-apple-arm64" || echo "libstiles-macos-intel-x86_64"
    else
        echo "libstiles-linux-$m"
    fi
}

synced=0
for src_dir in "$bin_dir"/libstiles-*/; do
    [ -d "$src_dir" ] || continue
    ci="$(basename "$src_dir")"
    if [ "$all" -eq 0 ] && [ "$ci" != "$(this_ci)" ]; then continue; fi

    read -r tag fname <<<"$(map "$ci")"
    [ -n "$tag" ] || { echo "skip (unknown platform): $ci"; continue; }

    src="$src_dir/lib/$fname"
    if [ ! -f "$src" ]; then echo "skip (no $fname): $ci"; continue; fi

    for dest in "$py_libs/$tag" "$r_libs/$tag"; do
        mkdir -p "$dest"
        cp -f "$src" "$dest/$fname"
        echo "copied  $ci/lib/$fname  ->  ${dest#$here/}/$fname"
    done
    synced=$((synced + 1))
done

echo ""
if [ "$synced" -eq 0 ]; then
    echo "Nothing synced. Put unzipped CI artifacts in binaries/ (e.g."
    echo "binaries/libstiles-linux-x86_64/lib/libstiles.so), or pass --all."
else
    echo "Synced $synced platform(s). For R: re-run 'R CMD INSTALL sTiles'."
fi
