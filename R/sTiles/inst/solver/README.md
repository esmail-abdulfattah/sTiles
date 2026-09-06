# Bundled solver libraries (optional)

A prebuilt libstiles can be dropped here, in a `<os>-<arch>/` sub-directory
matching `sTiles:::.sTiles_platform_tag()`:

```
inst/solver/
  linux-x86_64/libstiles.so
  macos-x86_64/libstiles.dylib      # Intel Macs
  macos-arm64/libstiles.dylib       # Apple Silicon
  windows-x86_64/libstiles.dll
```

At install these become `<pkg>/solver/<os>-<arch>/...`, and the package prefers
them over the download cache. This is for building a self-contained install of
your own: the released package ships no binary, and these files are
intentionally git-ignored. During development set `STILES_LIB` /
`STILES_LIB_DIR` instead, or rely on the loader's fallback to a repo
`lib/libstiles.so`.

Do not use the name `libs/` here: R reserves `<pkg>/libs` for the package's own
compiled code, and a non-empty `inst/libs` makes R CMD check complain.
