#!/usr/bin/env python3
"""Generate 4 sparsity-pattern SVGs for the hero density spectrum.

Each tile is a 7x7 grid (49 cells) inside a 56x56 viewBox.
- Empty cells render in a light tint so the grid is always visible.
- Filled cells use the tile's accent color.

Patterns are matrix-realistic (diagonal-dominant / banded), since these
tiles represent fill-density levels inside a tile-based sparse SPD solver.
"""
from pathlib import Path

OUT = Path("/home/abdulfe/Documents/sites/stiles/docs/images")

# 7x7 grid; each cell 7x7 with 1px gap -> stride 8.
GRID = 7
CELL = 7
STRIDE = 8
SIZE = GRID * STRIDE  # 56

# Filled-cell positions (row, col) for each density tier.
SPARSE     = [(1, 1), (3, 4), (5, 2)]                                 # 3 dots, 6%
SEMI_SPARSE = (
    [(i, i) for i in range(GRID)]                                     # diagonal (7)
    + [(0, 1), (1, 2), (5, 4)]                                        # 3 off-diag
)                                                                     # 10 dots, 20%
# Banded: main + 2 off-diags on each side, clipped to grid bounds.
SEMI_DENSE = []
for r in range(GRID):
    for c in range(GRID):
        if abs(r - c) <= 2:
            SEMI_DENSE.append((r, c))                                 # 29 dots, 59%
DENSE = [(r, c) for r in range(GRID) for c in range(GRID)]            # 49 dots, 100%

TILES = [
    ("sparse",      "Sparse",      "#c5d86d", SPARSE),       # lime
    ("semi-sparse", "Semi-Sparse", "#e08c5a", SEMI_SPARSE),  # orange
    ("semi-dense",  "Semi-Dense",  "#4a9a8f", SEMI_DENSE),   # teal
    ("dense",       "Dense",       "#1a1a1a", DENSE),        # near-black
]

EMPTY_FILL = "#f0f0f0"
BG_FILL    = "#fafafa"

def render(positions, color):
    cells = []
    filled = set(positions)
    for r in range(GRID):
        for c in range(GRID):
            x = c * STRIDE
            y = r * STRIDE
            fill = color if (r, c) in filled else EMPTY_FILL
            cells.append(
                f'    <rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" rx="1" fill="{fill}"/>'
            )
    return "\n".join(cells)

template = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {sz} {sz}" role="img" aria-label="{label} density tile">
  <title>{label}</title>
  <rect width="{sz}" height="{sz}" rx="3" fill="{bg}"/>
{cells}
</svg>
'''

for slug, label, color, positions in TILES:
    svg = template.format(
        sz=SIZE, label=label, bg=BG_FILL, cells=render(positions, color)
    )
    out_path = OUT / f"tile-{slug}.svg"
    out_path.write_text(svg)
    print(f"  {out_path.name}  ({len(positions)} cells, {len(positions)/49*100:.0f}%)")

print("Done.")
