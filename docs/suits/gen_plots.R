#!/usr/bin/env Rscript
# Generate high-res sparsity (spy) plots for every matrix in the suite using
# the streaming inla.stiles.spy.mtx renderer (consistent format, 2400x2400
# @300dpi, grey log-density, never loads the full matrix). Output goes to
# plots/<name>_sparsity.png. Idempotent: existing plots are skipped, so the
# job is resumable. Matrices are processed smallest-file-first so the quick
# ones land immediately.

suppressWarnings(suppressMessages(
  source(Sys.getenv("STILES_UTILS_R", ""))   # external R helper; set STILES_UTILS_R
))

MTX_ROOT <- Sys.getenv("STILES_MTX_DIR", "")  # raw .mtx corpus; set STILES_MTX_DIR
OUT_DIR  <- "plots"   # relative: run this script from docs/suits
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

files <- list.files(MTX_ROOT, pattern = "\\.mtx$", recursive = TRUE,
                    full.names = TRUE)
sizes <- file.info(files)$size
files <- files[order(sizes)]                 # smallest first

cat(sprintf("[gen_plots] %d matrices to consider\n", length(files)))
done <- 0L; skipped <- 0L; failed <- character(0)

for (f in files) {
    name <- sub("\\.mtx$", "", basename(f))
    out  <- file.path(OUT_DIR, paste0(name, "_sparsity.png"))
    if (file.exists(out)) {
        skipped <- skipped + 1L
        next
    }
    t0 <- Sys.time()
    ok <- tryCatch({
        inla.stiles.spy.mtx(f, out = out, GRID = 1000L,
                            text = FALSE, colorbar = FALSE, palette = "grey")
        TRUE
    }, error = function(e) {
        cat(sprintf("[gen_plots] FAIL %-32s : %s\n", name, conditionMessage(e)))
        FALSE
    })
    if (ok) {
        dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
        done <- done + 1L
        cat(sprintf("[gen_plots] OK   %-32s  %6.1fs  (%d done)\n",
                    name, dt, done))
        flush.console()
    } else {
        failed <- c(failed, name)
    }
}

cat(sprintf("\n[gen_plots] finished: %d generated, %d already present, %d failed\n",
            done, skipped, length(failed)))
if (length(failed) > 0) cat("[gen_plots] failed:", paste(failed, collapse = ", "), "\n")
