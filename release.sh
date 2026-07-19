#!/usr/bin/env bash
# release.sh -- date-based (CalVer) release helper for the sTiles Python package.
#
# Stamps today's date as the version (YYYY.M.D), builds the wheel + sdist, and
# prints the publish steps. The GitHub Release tag matches the version (v<date>).
#
#   ./release.sh          -> version 2026.7.19
#   ./release.sh 1        -> version 2026.7.19.1   (a second release the same day)
#
# The four platform binaries are shipped as Release assets (not in git). Point
# $STILES_BINARIES at the folder holding the libstiles-<platform>.zip files.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py="$here/python"
bins="${STILES_BINARIES:-$here/../../ideas/adv_sTiles/bindings/binaries}"

sfx="${1:-}"
VER="$(python3 -c "import datetime as d; t=d.date.today(); print(f'{t.year}.{t.month}.{t.day}')")"
[ -n "$sfx" ] && VER="$VER.$sfx"

echo "==> stamping version $VER"
cat > "$py/sTiles/_version.py" <<EOF
# Single source of truth for the sTiles Python package version.
# Date-based (CalVer): YYYY.M.D. Bump with ../release.sh (stamps today's date).
__version__ = "$VER"
EOF

echo "==> syncing the download page's release line"
if [ -f "$here/docs/download.html" ]; then
    sed -i -E "s#(Latest release <strong>v)[0-9][0-9.]*(</strong>)#\1$VER\2#" "$here/docs/download.html"
fi

echo "==> building the Python distribution"
rm -rf "$py/build" "$py"/*.egg-info "$py/dist"
python3 -m build "$py"
ls "$py/dist"

cat <<STEPS

Version $VER built. Next (you run these):

  1. Commit + tag:
       cd "$here"
       git add -A && git commit -m "Release $VER"
       git tag v$VER
       git push && git push --tags

  2. GitHub Release v$VER with the four platform binaries
     (from \$STILES_BINARIES = $bins):
       gh release create v$VER \\
         "$bins"/libstiles-linux-x86_64.zip \\
         "$bins"/libstiles-linux-arm64.zip \\
         "$bins"/libstiles-macos-apple-arm64.zip \\
         "$bins"/libstiles-windows-x86_64.zip \\
         --title "sTiles $VER" --notes "sTiles $VER"
     (or drag the four zips onto the Releases web page)

  3. Publish to PyPI:
       python -m twine upload "$py"/dist/*
STEPS
