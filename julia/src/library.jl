# Locating and loading the prebuilt libstiles shared object.
#
# This file does exactly what pysTiles' _ffi.py and the R package's loader do,
# with the same search order, the same environment variables and the same cache
# layout -- so the three bindings share one downloaded solver on a machine.
#
# Search order (first hit wins)
#   1. $STILES_LIB           -- full path to the shared object.
#   2. $STILES_LIB_DIR       -- directory containing it.
#   3. $STILES_BINARIES_DIR  -- a CI-artifact tree with
#                               libstiles-<plat>/lib/libstiles.{so,dylib}.
#   4. a binaries/ (or bindings/binaries/) directory in any ancestor of the
#      package or the working directory, same CI-artifact layout -- so unzipped
#      GitHub Actions artifacts are picked up with zero configuration.
#   5. bundled deps/<os>-<arch>/  -- vendored into the package by sync_binaries.sh.
#   6. the download cache, newest release first.
#   7. repo dev fallback: lib/libstiles.{so,dylib} in an ancestor.
#   8. GitHub Release download, cached under ~/.cache/sTiles/<tag>/<plat>/.
#      Disable with $STILES_NO_DOWNLOAD=1.

const RELEASE_REPO = get(ENV, "STILES_RELEASE_REPO", "esmail-abdulfattah/sTiles")

# Filled in by __init__.
const LIBHANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const LIBPATH   = Ref{String}("")

"""
    library_path() -> String

Absolute path of the loaded `libstiles` shared object.
"""
library_path() = LIBPATH[]

function _lib_filename()
    Sys.isapple()   && return "libstiles.dylib"
    Sys.iswindows() && return "libstiles.dll"
    return "libstiles.so"
end

"""Architecture tag used by the bundled `deps/<os>-<arch>/` directory."""
function _platform_tag()
    arch = Sys.ARCH === :x86_64 ? "x86_64" :
           Sys.ARCH in (:aarch64, :arm64) ? "arm64" : String(Sys.ARCH)
    Sys.isapple()   && return "macos-$arch"
    Sys.islinux()   && return "linux-$arch"
    Sys.iswindows() && return "windows-$arch"
    return "$(Sys.KERNEL)-$arch"
end

"""
Name of the CI build-artifact directory for this platform. The GitHub Actions
`build` workflow uploads one directory per target as
`libstiles-<name>/lib/libstiles.{so,dylib}`; those names differ from the bundle
`<os>-<arch>` tag (e.g. `libstiles-macos-apple-arm64`).

`STILES_VARIANT` selects an opt-in build (e.g. `v3-mkl` on Linux x86_64,
`armv82-armpl` on Linux arm64); the default is the most compatible build.
"""
function _ci_folder()
    arch = Sys.ARCH === :x86_64 ? "x86_64" :
           Sys.ARCH in (:aarch64, :arm64) ? "arm64" : String(Sys.ARCH)
    base = if Sys.isapple()
        arch == "arm64" ? "libstiles-macos-apple-arm64" : "libstiles-macos-intel-x86_64"
    elseif Sys.islinux()
        "libstiles-linux-$arch"
    elseif Sys.iswindows()
        "libstiles-windows-$arch"
    else
        "libstiles-$(Sys.KERNEL)-$arch"
    end
    variant = strip(get(ENV, "STILES_VARIANT", ""))
    return isempty(variant) ? base : "$base-$variant"
end

function _cache_dir()
    env = get(ENV, "STILES_CACHE_DIR", "")
    isempty(env) || return env
    if Sys.iswindows()
        return joinpath(get(ENV, "LOCALAPPDATA", joinpath(homedir(), "AppData", "Local")), "sTiles")
    end
    return joinpath(get(ENV, "XDG_CACHE_HOME", joinpath(homedir(), ".cache")), "sTiles")
end

"""Every ancestor of `p`, `p` itself first."""
function _ancestors(p::AbstractString)
    out = String[]
    cur = abspath(p)
    while true
        push!(out, cur)
        parent = dirname(cur)
        parent == cur && break
        cur = parent
    end
    return out
end

"""Cached solvers for this platform, newest release first (the offline path)."""
function _cached_libs(ci::AbstractString, fname::AbstractString)
    root = _cache_dir()
    isdir(root) || return String[]
    hits = String[]
    for tag in sort!(readdir(root); rev = true)
        p = joinpath(root, tag, ci, fname)
        isfile(p) && push!(hits, p)
    end
    flat = joinpath(root, ci, fname)          # pre-versioning layout
    isfile(flat) && push!(hits, flat)
    return hits
end

function _candidate_paths()
    fname = _lib_filename()
    ci    = _ci_folder()
    cands = String[]

    env_lib = get(ENV, "STILES_LIB", "")
    isempty(env_lib) || push!(cands, env_lib)

    env_dir = get(ENV, "STILES_LIB_DIR", "")
    isempty(env_dir) || push!(cands, joinpath(env_dir, fname))

    env_bin = get(ENV, "STILES_BINARIES_DIR", "")
    isempty(env_bin) || push!(cands, joinpath(env_bin, ci, "lib", fname))

    # CI-artifact trees above the package and above the working directory.
    roots = _ancestors(@__DIR__)
    try
        append!(roots, _ancestors(pwd()))
    catch
        # pwd() can fail if the directory was deleted underneath us; ignore.
    end
    for parent in roots
        push!(cands, joinpath(parent, "binaries", ci, "lib", fname))
        push!(cands, joinpath(parent, "bindings", "binaries", ci, "lib", fname))
    end

    # Bundled inside the package: deps/<os>-<arch>/ then a flat fallback.
    pkgroot = dirname(@__DIR__)
    push!(cands, joinpath(pkgroot, "deps", _platform_tag(), fname))
    push!(cands, joinpath(pkgroot, "deps", fname))

    # Previously downloaded solvers, newest release first.
    append!(cands, _cached_libs(ci, fname))

    # Repo dev fallback.
    for parent in roots
        push!(cands, joinpath(parent, "lib", fname))
    end

    return cands
end

"""
Fail with a clear message on CPUs the prebuilt library cannot run on. The
x86_64 builds are compiled for AVX2 (Intel Haswell 2013+ / AMD Excavator+);
without this check an unsupported machine loads the library fine and then dies
with an uninformative illegal instruction at the first factorization.
"""
function _check_cpu_supported()
    Sys.ARCH === :x86_64 || return nothing
    have_avx2 = true
    try
        if Sys.islinux()
            have_avx2 = occursin(" avx2", read("/proc/cpuinfo", String))
        end
        # macOS Intel: every machine Apple still supports has AVX2.
    catch
        return nothing        # never block loading on a failed detection
    end
    have_avx2 || throw(sTilesError(
        "the prebuilt sTiles library requires a CPU with AVX2 (Intel Haswell " *
        "2013+ or AMD Excavator+); this machine does not report it. Build " *
        "sTiles from source for this CPU instead."))
    return nothing
end

"""Current release tag, so the cache can be keyed by it. `nothing` when offline."""
function _latest_tag()
    tag = get(ENV, "STILES_RELEASE_TAG", "")
    isempty(tag) || return tag
    try
        io = IOBuffer()
        Downloads.download("https://api.github.com/repos/$RELEASE_REPO/releases/latest",
                           io; timeout = 10)
        m = match(r"\"tag_name\"\s*:\s*\"([^\"]+)\"", String(take!(io)))
        return m === nothing ? nothing : String(m.captures[1])
    catch
        return nothing        # offline is normal, not an error
    end
end

function _unzip(zipfile::AbstractString, dest::AbstractString)
    try
        run(pipeline(`$(p7zip_jll.p7zip()) x -y -o$dest $zipfile`;
                     stdout = devnull, stderr = devnull))
        return true
    catch
    end
    try                        # p7zip_jll unavailable: fall back to the system tool
        run(pipeline(`unzip -o -q $zipfile -d $dest`; stdout = devnull, stderr = devnull))
        return true
    catch
    end
    return false
end

"""
Download the matching libstiles into the cache; return its path or `nothing`.

The cache is keyed by RELEASE, not by platform alone: keyed by platform the
first download would become permanent, so later releases were never fetched and
users kept a months-old solver. Same layout as the Python and R bindings.
"""
function _download_from_release(; force::Bool = false)
    isempty(get(ENV, "STILES_NO_DOWNLOAD", "")) || return nothing
    ci    = _ci_folder()
    fname = _lib_filename()

    tag = _latest_tag()
    if tag === nothing                        # offline: use whatever is cached
        have = _cached_libs(ci, fname)
        return isempty(have) ? nothing : have[1]
    end
    dest = joinpath(_cache_dir(), tag, ci)
    libpath = joinpath(dest, fname)
    (isfile(libpath) && !force) && return libpath

    base = get(ENV, "STILES_RELEASE_BASE_URL",
               "https://github.com/$RELEASE_REPO/releases/download/$tag")
    url = "$base/$ci.zip"
    try
        mkpath(dest)
        @info "sTiles: fetching libstiles for $ci from $url"
        tmpdir = mktempdir()
        try
            zipfile = joinpath(tmpdir, "$ci.zip")
            Downloads.download(url, zipfile)
            staging = joinpath(tmpdir, "x")
            mkpath(staging)
            _unzip(zipfile, staging) || return nothing
            # Everything shipped under lib/: the library itself plus, on the
            # platforms that are not fully self-contained (macOS Intel,
            # Windows), the sibling runtime it loads by a loader-relative path.
            srcdir = joinpath(staging, "lib")
            isdir(srcdir) || (srcdir = staging)
            for f in readdir(srcdir)
                src = joinpath(srcdir, f)
                isfile(src) || continue
                # Move into place rather than writing over the destination: if
                # that file is already mapped into this process, truncating it
                # turns the mapping to garbage, while a rename only swaps the
                # directory entry and leaves the running process its old inode.
                staged = joinpath(dest, f * ".new")
                cp(src, staged; force = true)
                mv(staged, joinpath(dest, f); force = true)
            end
        finally
            rm(tmpdir; recursive = true, force = true)
        end
        # Superseded solvers are ~20 MB each and serve nobody once a newer one
        # loads; drop them, including any pre-versioning copy.
        for entry in readdir(_cache_dir())
            entry == tag && continue
            old = joinpath(_cache_dir(), entry)
            isdir(old) || continue
            if entry == ci || startswith(entry, "v") || isdigit(first(entry))
                rm(old; recursive = true, force = true)
            end
        end
    catch exc
        @warn "sTiles: release download failed" exception = exc
        return nothing
    end
    return isfile(libpath) ? libpath : nothing
end

"""
    clear_cache() -> Int

Delete every downloaded solver, returning how many were removed. The compiled
solver is downloaded separately from this package and cached, so reinstalling
the package does not replace it; use this to start fresh.
"""
function clear_cache()
    root = _cache_dir()
    isdir(root) || return 0
    n = 0
    for entry in readdir(root)
        rm(joinpath(root, entry); recursive = true, force = true)
        n += 1
    end
    return n
end

"""
    update_library() -> String

Download the current released solver, replacing any cached copy, and return its
path. Restart Julia afterwards: the solver is loaded once per process.
"""
function update_library()
    clear_cache()
    lib = _download_from_release(force = true)
    lib === nothing && throw(sTilesError(
        "could not download a solver; check the network, or set STILES_LIB to " *
        "a local libstiles"))
    return lib
end

function _load_library()
    _check_cpu_supported()
    flags = Libdl.RTLD_LAZY | Libdl.RTLD_LOCAL
    tried = String[]
    for path in _candidate_paths()
        isfile(path) || continue
        try
            return Libdl.dlopen(path, flags), abspath(path)
        catch exc
            push!(tried, "$path  (load failed: $exc)")
        end
    end

    downloaded = _download_from_release()
    if downloaded !== nothing
        try
            return Libdl.dlopen(downloaded, flags), abspath(downloaded)
        catch exc
            push!(tried, "$downloaded  (load failed: $exc)")
        end
    end

    # Last resort: let the loader resolve a bare SONAME via LD_LIBRARY_PATH.
    try
        h = Libdl.dlopen(_lib_filename(), flags)
        return h, Libdl.dlpath(h)
    catch
    end

    throw(sTilesError(string(
        "could not locate libstiles for this platform.\n",
        "The automatic download from the GitHub Release failed or was disabled.\n",
        "Set STILES_LIB to the shared object, point STILES_BINARIES_DIR at a CI\n",
        "artifact tree ($(_ci_folder())/lib/$(_lib_filename())), or drop it in\n",
        "deps/$(_platform_tag())/$(_lib_filename()) inside this package.",
        isempty(tried) ? "" : "\nFailed to load:\n  " * join(tried, "\n  "))))
end
