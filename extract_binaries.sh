#!/usr/bin/env bash
# extract_binaries.sh -- unzip the CI artifact zips in binaries/ into place.
#
# GitHub Actions artifacts download as `binaries/libstiles-<platform>.zip`, each
# containing `lib/` + `include/`. The wrappers read EXTRACTED directories
# (binaries/libstiles-<platform>/lib/libstiles.{so,dylib}), not zips -- so after
# dropping fresh artifact zips, run this once. Then Python and R pick up the new
# build automatically (they search the binaries/ tree; no bundling needed for
# dev use).
#
#   cd bindings && ./extract_binaries.sh
#
# Idempotent: re-extracts (overwrites) whatever zips are present.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bin_dir="$here/binaries"

shopt -s nullglob
zips=("$bin_dir"/libstiles-*.zip)
if [ ${#zips[@]} -eq 0 ]; then
    echo "No binaries/libstiles-*.zip found. Drop the CI artifact zips in"
    echo "  $bin_dir"
    exit 1
fi

for z in "${zips[@]}"; do
    name="$(basename "${z%.zip}")"
    unzip -o -q "$z" -d "$bin_dir/$name"
    lib="$(ls "$bin_dir/$name"/lib/libstiles.* 2>/dev/null | grep -vE '\.a$' | head -1)"
    echo "extracted  $name  ->  ${lib#$here/}"
done

echo ""
echo "Done. The wrappers now use these when run from the repo (zero config)."
echo "For self-contained pip/R installs, also run: ./sync_binaries.sh"
