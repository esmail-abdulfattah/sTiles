## sTiles -- R interface to the sTiles sparse Cholesky / selected-inverse
## framework.  See ?sTiles for the entry point.

# Package-private state: the located libstiles path, the glue DLLInfo, and a
# cache of resolved native symbols.
.sTiles <- new.env(parent = emptyenv())

# Tile factorization regimes (mirror stiles.h tile_type_mode values).
.sTiles_modes <- c(dense = 0L, semisparse = 1L, semi = 1L,
                   sparse = 2L, auto = 3L)

# ---------------------------------------------------------------------------
# Library resolution -- mirrors the Python binding's search order.
# ---------------------------------------------------------------------------
.sTiles_lib_filename <- function() {
    if (Sys.info()[["sysname"]] == "Darwin") "libstiles.dylib" else "libstiles.so"
}

.sTiles_platform_tag <- function() {
    sysname <- Sys.info()[["sysname"]]
    machine <- Sys.info()[["machine"]]
    arch <- switch(machine,
                   "x86_64" = "x86_64", "amd64" = "x86_64",
                   "arm64" = "arm64", "aarch64" = "arm64", machine)
    os <- if (sysname == "Darwin") "macos" else "linux"
    paste0(os, "-", arch)
}

# Name of the CI build-artifact directory for this platform, e.g.
# "libstiles-linux-x86_64" or "libstiles-macos-apple-arm64".
.sTiles_ci_folder <- function() {
    sysname <- Sys.info()[["sysname"]]
    machine <- Sys.info()[["machine"]]
    arch <- switch(machine,
                   "x86_64" = "x86_64", "amd64" = "x86_64",
                   "arm64" = "arm64", "aarch64" = "arm64", machine)
    if (sysname == "Darwin")
        return(if (arch == "arm64") "libstiles-macos-apple-arm64"
               else "libstiles-macos-intel-x86_64")
    paste0("libstiles-linux-", arch)
}

# Walk up from `start`, collecting CI-artifact candidates
#   <ancestor>/binaries/<ci>/lib/<fname>  and  <ancestor>/bindings/binaries/...
.sTiles_binaries_candidates <- function(start, ci, fname) {
    out <- character(0)
    here <- start
    for (i in seq_len(12)) {
        out <- c(out,
                 file.path(here, "binaries", ci, "lib", fname),
                 file.path(here, "bindings", "binaries", ci, "lib", fname))
        parent <- dirname(here)
        if (parent == here) break
        here <- parent
    }
    out
}

# ---------------------------------------------------------------------------
# Fetch the matching prebuilt libstiles from the GitHub Release.
#
# When the package is install_github()'d there is no binary in the tree, so on
# first use we download the platform library from the project's Release assets
# and cache it. The Linux/macOS builds are self-contained (BLAS embedded).
# Overrides: STILES_NO_DOWNLOAD, STILES_RELEASE_REPO, STILES_RELEASE_BASE_URL,
# STILES_CACHE_DIR.
# ---------------------------------------------------------------------------
.sTiles_cache_dir <- function() {
    env <- Sys.getenv("STILES_CACHE_DIR", "")
    if (nzchar(env)) return(env)
    tools::R_user_dir("sTiles", which = "cache")
}

.sTiles_download_from_release <- function() {
    if (nzchar(Sys.getenv("STILES_NO_DOWNLOAD", ""))) return(NA_character_)
    ci <- .sTiles_ci_folder()
    fname <- .sTiles_lib_filename()
    dest <- file.path(.sTiles_cache_dir(), ci)
    lib <- file.path(dest, fname)
    if (file.exists(lib)) return(lib)   # already downloaded on a previous run

    repo <- Sys.getenv("STILES_RELEASE_REPO", "esmail-abdulfattah/sTiles")
    base <- Sys.getenv(
        "STILES_RELEASE_BASE_URL",
        sprintf("https://github.com/%s/releases/latest/download", repo))
    url <- sprintf("%s/%s.zip", base, ci)
    ok <- tryCatch({
        dir.create(dest, recursive = TRUE, showWarnings = FALSE)
        tmp <- tempfile(fileext = ".zip")
        message(sprintf("sTiles: fetching libstiles for %s from %s", ci, url))
        utils::download.file(url, tmp, mode = "wb", quiet = TRUE)
        entries <- utils::unzip(tmp, list = TRUE)$Name
        want <- entries[basename(entries) == fname]   # the lib itself, flat
        utils::unzip(tmp, files = want, exdir = dest, junkpaths = TRUE)
        unlink(tmp)
        TRUE
    }, error = function(e) {
        message(sprintf("sTiles: release download failed (%s)", conditionMessage(e)))
        FALSE
    })
    if (ok && file.exists(lib)) lib else NA_character_
}

.sTiles_find_lib <- function(libname, pkgname) {
    fname <- .sTiles_lib_filename()
    ci <- .sTiles_ci_folder()
    cands <- character(0)

    env_lib <- Sys.getenv("STILES_LIB", "")
    if (nzchar(env_lib)) cands <- c(cands, env_lib)

    env_dir <- Sys.getenv("STILES_LIB_DIR", "")
    if (nzchar(env_dir)) cands <- c(cands, file.path(env_dir, fname))

    # CI binaries tree: <root>/libstiles-<ci>/lib/libstiles.{so,dylib}.
    env_bin <- Sys.getenv("STILES_BINARIES_DIR", "")
    if (nzchar(env_bin)) cands <- c(cands, file.path(env_bin, ci, "lib", fname))

    pkgdir <- if (!missing(libname) && !missing(pkgname))
        file.path(libname, pkgname) else system.file(package = "sTiles")

    # Search a `binaries/` tree above the package AND above the working dir
    # (covers an installed package run from inside the repo checkout).
    cands <- c(cands,
               .sTiles_binaries_candidates(pkgdir, ci, fname),
               .sTiles_binaries_candidates(getwd(), ci, fname))

    # Bundled inside the installed package: inst/libs/<plat>/ -> libs/<plat>/.
    cands <- c(cands,
               file.path(pkgdir, "libs", .sTiles_platform_tag(), fname),
               file.path(pkgdir, "libs", fname))

    # Development checkout: search ancestors for lib/libstiles.{so,dylib}.
    here <- pkgdir
    for (i in seq_len(10)) {
        cands <- c(cands, file.path(here, "lib", fname))
        parent <- dirname(here)
        if (parent == here) break
        here <- parent
    }

    hit <- cands[file.exists(cands)]
    if (length(hit) > 0) return(normalizePath(hit[1]))

    # Nothing local: fetch the prebuilt library from the GitHub Release.
    dl <- .sTiles_download_from_release()
    if (!is.na(dl) && file.exists(dl)) return(normalizePath(dl))

    stop("Could not locate ", fname, ".\nThe automatic download from the GitHub ",
         "Release failed or was disabled. Set STILES_LIB to the shared object, ",
         "STILES_LIB_DIR to its directory, or STILES_BINARIES_DIR to a CI-artifact ",
         "tree (", ci, "/lib/", fname, ").\nSearched:\n  ",
         paste(cands, collapse = "\n  "), call. = FALSE)
}

.onLoad <- function(libname, pkgname) {
    # Quiet libstiles' one-time banner unless the user opted in.
    if (Sys.getenv("STILES_NO_BANNER", "") == "")
        Sys.setenv(STILES_NO_BANNER = "1")

    # Defer locating/loading libstiles + the glue to first use, so that
    # install (R CMD INSTALL test-load) and library(sTiles) never fail merely
    # because the binary hasn't been downloaded from the Release yet. The
    # download happens on the first actual sTiles call (see .sTiles_ensure_loaded).
    .sTiles$libname <- libname
    .sTiles$pkgname <- pkgname
    .sTiles$sym <- new.env(parent = emptyenv())
}

# Locate/download libstiles and load it + the glue DLL. Idempotent; called from
# every native entry point via .sc().
.sTiles_ensure_loaded <- function() {
    if (!is.null(.sTiles$dll)) return(invisible())
    libpath <- .sTiles_find_lib(.sTiles$libname, .sTiles$pkgname)
    # Preload libstiles with GLOBAL symbol visibility (local = FALSE) so the
    # glue's undefined sTiles_* symbols resolve against it.
    dyn.load(libpath, local = FALSE, now = TRUE)
    .sTiles$libpath <- libpath

    gluepath <- file.path(.sTiles$libname, .sTiles$pkgname, "libs",
                          .Platform$r_arch,
                          paste0(.sTiles$pkgname, .Platform$dynlib.ext))
    .sTiles$dll <- dyn.load(gluepath)
    invisible()
}

.onUnload <- function(libpath) {
    if (!is.null(.sTiles$dll)) try(dyn.unload(.sTiles$dll[["path"]]), silent = TRUE)
}

# Resolve (and cache) a registered native routine from the glue DLL.
.sc <- function(name) {
    .sTiles_ensure_loaded()
    s <- .sTiles$sym[[name]]
    if (is.null(s)) {
        s <- getNativeSymbolInfo(name, PACKAGE = .sTiles$dll)$address
        assign(name, s, envir = .sTiles$sym)
    }
    s
}

#' Absolute path of the loaded libstiles shared object.
#' @export
sTiles_library_path <- function() { .sTiles_ensure_loaded(); .sTiles$libpath }

#' sTiles library version string.
#' @export
sTiles_version <- function() .Call(.sc("sTiles_version_R"))

# ---------------------------------------------------------------------------
# Matrix -> lower-triangle COO (0-based, canonical (row, col) order).
# ---------------------------------------------------------------------------
.sTiles_lower_coo <- function(Q) {
    if (is.matrix(Q)) Q <- methods::as(Q, "CsparseMatrix")
    if (!methods::is(Q, "sparseMatrix"))
        Q <- methods::as(methods::as(Q, "matrix"), "CsparseMatrix")
    if (nrow(Q) != ncol(Q)) stop("matrix must be square")
    n <- nrow(Q)

    L <- Matrix::tril(methods::as(Q, "CsparseMatrix"))
    L <- methods::as(L, "TsparseMatrix")   # triplet form: @i, @j, @x (0-based)
    i <- L@i; j <- L@j; x <- L@x

    ord <- order(i, j)                     # canonical, stable across refactors
    list(n = n, i = as.integer(i[ord]), j = as.integer(j[ord]),
         x = as.double(x[ord]))
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#' Preprocess (analyze) a matrix: ordering bake-off + tile layout ONLY.
#'
#' This is the symbolic phase -- it depends only on the sparsity pattern, not
#' the numeric values, and does no Cholesky. Follow it with sTiles_factorize()
#' to run the numeric factorization; timing the two separately isolates the
#' preprocessing cost from the numeric cost.
#'
#' @inheritParams sTiles
#' @return An object of class "sTiles" that is analyzed but not yet factorized.
#' @export
sTiles_analyze <- function(Q, cores = 1L, mode = "auto", tile_size = 40L,
                           inverse = FALSE, group = 0L, log_level = -1L) {
    m <- if (is.character(mode)) {
        code <- .sTiles_modes[tolower(mode)]
        if (is.na(code)) stop("unknown mode '", mode, "'")
        code
    } else as.integer(mode)

    coo <- .sTiles_lower_coo(Q)
    ptr <- .Call(.sc("sTiles_analyze_R"), coo$i, coo$j, coo$n,
                 as.integer(cores), as.integer(m), as.integer(tile_size),
                 as.logical(inverse), as.integer(group), as.integer(log_level))

    obj <- list(ptr = ptr, n = coo$n, nnz = length(coo$i),
                mode = as.integer(m), cores = as.integer(cores),
                inverse = isTRUE(inverse), factored = FALSE,
                values = coo$x, pattern = list(i = coo$i, j = coo$j))
    class(obj) <- "sTiles"
    obj
}

#' Numeric Cholesky factorization (reuses the preprocessing from analyze).
#'
#' @param x  An "sTiles" object from sTiles_analyze() (or sTiles()).
#' @param Q  Optional: a matrix with the SAME sparsity pattern whose values to
#'   factor. If omitted, the values captured at analyze time are used.
#' @return The (invisibly returned) "sTiles" object, now factorized.
#' @export
sTiles_factorize <- function(x, Q = NULL) {
    vals <- if (is.null(Q)) x$values else {
        coo <- .sTiles_lower_coo(Q)
        if (length(coo$i) != x$nnz ||
            !identical(coo$i, x$pattern$i) || !identical(coo$j, x$pattern$j))
            stop("sTiles_factorize(Q=): Q must share this object's sparsity ",
                 "pattern; build a new sTiles_analyze() otherwise", call. = FALSE)
        coo$x
    }
    .Call(.sc("sTiles_factorize_R"), x$ptr, as.double(vals))
    invisible(x)
}

#' Factorize a symmetric positive-definite matrix with sTiles.
#'
#' One-shot: runs preprocessing (sTiles_analyze) then the numeric factorization
#' (sTiles_factorize). To time the two phases apart, or to reuse preprocessing
#' across many value-sets, call sTiles_analyze() + sTiles_factorize() yourself.
#'
#' @param Q  A symmetric positive-definite matrix (Matrix::sparseMatrix or a
#'   base matrix).  Only the lower triangle is used.
#' @param cores  Worker threads (default 1).
#' @param mode  "auto", "dense", "semisparse" or "sparse" (default "auto").
#' @param tile_size  Tile size, or -1 for auto (default 40).
#' @param inverse  Reserve selected-inverse storage; required for
#'   sTiles_selinv()/_diag()/_elm()/_row() (default FALSE).
#' @param group  libstiles group slot; use distinct values for concurrent
#'   handles (default 0).
#' @param log_level  libstiles verbosity: -1 silent (default), 0 timing,
#'   1 info, 2 debug, 3 trace.
#' @return An object of class "sTiles" wrapping a live factorization.
#' @export
sTiles <- function(Q, cores = 1L, mode = "auto", tile_size = 40L,
                   inverse = FALSE, group = 0L, log_level = -1L) {
    s <- sTiles_analyze(Q, cores = cores, mode = mode, tile_size = tile_size,
                        inverse = inverse, group = group, log_level = log_level)
    sTiles_factorize(s)
    s
}

#' Log-determinant of Q ( = 2 * sum(log diag(L)) ).
#' @export
sTiles_logdet <- function(x) .Call(.sc("sTiles_logdet_R"), x$ptr)

#' Compute the selected inverse, reusing the current numeric factorization.
#'
#' Z = Q^-1 restricted to the pattern of the Cholesky factor (pattern(L+L^T)).
#' Requires the handle to have been built with inverse = TRUE. Idempotent, and
#' otherwise computed lazily on the first sTiles_selinv_*() query. Call it
#' explicitly to time the selected inverse on its own, and re-call it after each
#' sTiles_factorize() to refresh Z for new values.
#' @return The (invisibly returned) "sTiles" object.
#' @export
sTiles_selinv <- function(x) {
    .Call(.sc("sTiles_selinv_R"), x$ptr)
    invisible(x)
}

#' Diagonal of the selected inverse, diag(Q^-1) -- the marginal variances.
#' @export
sTiles_selinv_diag <- function(x) .Call(.sc("sTiles_selinv_diag_R"), x$ptr)

#' Selected-inverse entry (Q^-1)[i, j] at ANY position (1-based, original order).
#'
#' Returns the selected inverse at (i, j) when that position lies in the factor
#' pattern (pattern(L+L^T)), and exactly 0 outside it. Both triangles work (Z is
#' symmetric). Triggers the selected-inverse computation on first use.
#' @export
sTiles_selinv_elm <- function(x, i, j)
    .Call(.sc("sTiles_selinv_elm_R"), x$ptr, as.integer(i), as.integer(j))

#' Selected-inverse values (Q^-1)[node, k] for each k in `neighbors` (1-based).
#' @export
sTiles_selinv_row <- function(x, node, neighbors)
    .Call(.sc("sTiles_selinv_row_R"), x$ptr, as.integer(node),
          as.integer(neighbors))

#' Solve with the factorization.
#'
#' @param x  A factorized "sTiles" object.
#' @param b  Right-hand side: a length-n vector or an n x nrhs matrix.
#' @param system  Which system to solve: "A" for Q x = b (default), "L" for
#'   L y = b (forward), "Lt" for L^T x = b (backward).
#' @return The solution, same shape as `b`.
#' @export
sTiles_solve <- function(x, b, system = c("A", "L", "Lt")) {
    which <- switch(match.arg(system), A = 0L, L = 1L, Lt = 2L)
    .Call(.sc("sTiles_solve_R"), x$ptr, as.double(b), which)
}

#' Structured summary of a factorization: dimensions, fill, mode, phase state,
#' and library-measured timings. Returns a list (e.g. sTiles_summary(s)$chol_time)
#' and prints a short report.
#' @export
sTiles_summary <- function(x) {
    modes <- c("dense", "semisparse", "sparse", "auto")
    fac <- as.logical(.Call(.sc("sTiles_is_factored_R"), x$ptr))
    out <- list(
        n           = x$n,
        nnz         = x$nnz,
        nnz_factor  = .Call(.sc("sTiles_nnz_factor_R"), x$ptr),
        mode        = modes[x$mode + 1L],
        cores       = x$cores,
        inverse     = x$inverse,
        factored    = fac,
        chol_time   = if (fac) .Call(.sc("sTiles_chol_time_R"), x$ptr) else NA_real_,
        selinv_time = tryCatch(.Call(.sc("sTiles_selinv_time_R"), x$ptr),
                               error = function(e) NA_real_),
        version     = sTiles_version(),
        library     = .sTiles$libpath)
    class(out) <- "sTiles_summary"
    out
}

#' Free the factorization now (otherwise freed at garbage collection).
#' @export
sTiles_close <- function(x) invisible(.Call(.sc("sTiles_free_R"), x$ptr))

#' @export
print.sTiles <- function(x, ...) {
    modes <- c("dense", "semisparse", "sparse", "auto")
    fac <- as.logical(.Call(.sc("sTiles_is_factored_R"), x$ptr))
    cat(sprintf("<sTiles: %d x %d, nnz=%d, mode=%s, cores=%d, inverse=%s, %s>\n",
                x$n, x$n, x$nnz, modes[x$mode + 1L], x$cores, x$inverse,
                if (fac) "factorized" else "analyzed"))
    invisible(x)
}

#' @export
print.sTiles_summary <- function(x, ...) {
    fmt_t <- function(t) if (is.na(t)) "-" else sprintf("%.4g s", t)
    cat(sprintf(paste0(
        "sTiles factorization\n",
        "  dimension  : %d x %d\n",
        "  input nnz  : %d      factor nnz(L) : %d\n",
        "  tile mode  : %s   cores : %d   inverse : %s\n",
        "  state      : %s\n",
        "  chol time  : %s      selinv time : %s\n",
        "  library    : %s\n"),
        x$n, x$n, x$nnz, x$nnz_factor, x$mode, x$cores, x$inverse,
        if (x$factored) "factorized" else "analyzed (not factorized)",
        fmt_t(x$chol_time), fmt_t(x$selinv_time), x$library))
    invisible(x)
}
