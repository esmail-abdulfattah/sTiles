#!/usr/bin/env bash
# release.sh -- date-based (CalVer) release helper for the sTiles Python package.
#
# Stamps today's date as the version, builds the wheel + sdist, and prints the
# publish steps. TWO spellings of the same date, on purpose:
#   package version  YYYY.M.D   (pip orders numerically; switching format would
#                                make every new release sort BELOW the old ones
#                                and pip would never upgrade anyone again)
#   GitHub tag       vYY.MM.DD  (zero padded, the same way INLA writes its
#                                releases, so the two projects' pages read
#                                consistently side by side)
# Nothing parses the tag: every loader resolves releases/latest by asset name.
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
TAG="v$(python3 -c "import datetime as d; t=d.date.today(); print(f'{t.year%100:02d}.{t.month:02d}.{t.day:02d}')")"
[ -n "$sfx" ] && VER="$VER.$sfx" && TAG="$TAG.$sfx"

echo "==> stamping version $VER"
cat > "$py/sTiles/_version.py" <<EOF
# Single source of truth for the sTiles Python package version.
# Date-based (CalVer): YYYY.M.D. Bump with ../release.sh (stamps today's date).
__version__ = "$VER"
EOF

echo "==> syncing the download page's release line"
if [ -f "$here/docs/download.html" ]; then
    # Version AND build date, both by id, so neither can drift: the old script
    # rewrote the number only and the hand-written "(built ...)" beside it went
    # three weeks stale without anyone noticing.
    sed -i -E "s#(<strong id=\"rel-ver\">v)[0-9][0-9.]*(</strong>)#\1$VER\2#" "$here/docs/download.html"
    sed -i -E "s#(<span id=\"rel-date\">)[0-9-]*(</span>)#\1$(date -u +%Y-%m-%d)\2#" "$here/docs/download.html"
    # Legacy form, in case the ids are ever removed again.
    sed -i -E "s#(Latest release <strong>v)[0-9][0-9.]*(</strong>)#\1$VER\2#" "$here/docs/download.html"
    echo "==> download page: $(grep -oE 'rel-ver\">v[0-9.]+' "$here/docs/download.html" | head -1), built $(date -u +%Y-%m-%d)"
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
       git tag $TAG
       git push && git push --tags

  2. GitHub Release $TAG with the four platform binaries
     (from \$STILES_BINARIES = $bins):
       gh release create $TAG \\
         "$bins"/libstiles-linux-x86_64.zip \\
         "$bins"/libstiles-linux-arm64.zip \\
         "$bins"/libstiles-macos-apple-arm64.zip \\
         "$bins"/libstiles-windows-x86_64.zip \\
         --title "sTiles $VER" --notes "sTiles $VER"
     (or drag the four zips onto the Releases web page)

  3. Publish to PyPI:
       python -m twine upload "$py"/dist/*
STEPS
