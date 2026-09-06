## Runs everywhere, solver or not: these are the paths that decide WHETHER a
## solver is used, so they must not need one (and must not touch the network).

test_that("sTiles_available() answers without loading anything", {
    ans <- sTiles_available()
    expect_type(ans, "logical")
    expect_length(ans, 1L)
    expect_false(is.na(ans))
})

test_that("the candidate list is well formed", {
    cands <- sTiles:::.sTiles_lib_candidates()
    expect_type(cands, "character")
    expect_gt(length(cands), 0L)
    expect_false(any(is.na(cands)))
    expect_true(all(nzchar(cands)))
    # Every candidate names the platform's library file.
    expect_true(all(basename(cands) == sTiles:::.sTiles_lib_filename()))
})

test_that("STILES_NO_DOWNLOAD blocks the installer", {
    withr_env <- Sys.getenv("STILES_NO_DOWNLOAD", unset = NA)
    Sys.setenv(STILES_NO_DOWNLOAD = "1")
    on.exit(if (is.na(withr_env)) Sys.unsetenv("STILES_NO_DOWNLOAD")
            else Sys.setenv(STILES_NO_DOWNLOAD = withr_env))

    expect_error(sTiles_install_library(), "STILES_NO_DOWNLOAD")
    # And the internal fetch is a no-op rather than a network call.
    expect_true(is.na(sTiles:::.sTiles_download_from_release()))
})

test_that("a missing solver reports how to install one", {
    skip_if(sTiles_available(), "a solver is installed on this machine")
    expect_error(sTiles_analyze(spd_band(10)), "sTiles_install_library")
})
