/**
 * @file stiles.h
 * @brief Public API for the sTiles framework.
 *
 * The sTiles framework enables tile-based symbolic and numeric factorization
 * for large sparse systems, designed for high-performance computing environments
 * with optional GPU acceleration and tree-based parallelism.
 *
 * Developed by Esmail Abdul Fattah and the sTiles team at KAUST.
 *
 * @date June 16, 2025
 * @author
 *   Esmail Abdul Fattah<br>
 *   esmail.abdulfattah@kaust.edu.sa
 */

#ifndef _STILES_MAIN_HEADER_H_
#define _STILES_MAIN_HEADER_H_

#include <stdio.h>
#include <stdbool.h>

#ifdef STILES_GPU
    #include <cuda_runtime.h>
    #include <cusolverDn.h>
    #include <cublas_v2.h>
#endif

//==============================================================================
// Named Parameter Indices
//==============================================================================
#define STILES_PARAM_CORRECTION_MODE       0
#define STILES_PARAM_TILE_SIZE             1
#define STILES_PARAM_ORDERING_MODE         2   /* bake-off candidate digit list */
#define STILES_PARAM_TILE_TYPE             3
#define STILES_PARAM_TILE_ORDERING_MODE    4
#define STILES_PARAM_TILE_ORDERING_SIZE    5
#define STILES_PARAM_FORCE_ND              6
#define STILES_PARAM_INVERSE_STORAGE       7
#define STILES_PARAM_USE_OMP               8
#define STILES_PARAM_SEMISPARSE_IMPL       9
#define STILES_PARAM_GPU_COMPARE          10
#define STILES_PARAM_GPU_ENABLE           11
#define STILES_PARAM_MEMORY_ESTIMATE      12
#define STILES_PARAM_TILE_ORD_MIN_DIM     13   /* DEPRECATED: stored, never read */
#define STILES_PARAM_SERIAL_MODE          14
#define STILES_PARAM_BW_MODE              15
/* 16..19, 23, 24 reserved */
#define STILES_PARAM_FORCE_SCOTCH         20
#define STILES_PARAM_SCOTCH_PADDING       21
#define STILES_PARAM_PATH2_DEPTH          22
#define STILES_PARAM_TREE_PATH_ENABLE     25
#define STILES_PARAM_TREE_PATH_FORCE      26
#define STILES_PARAM_COUNT                27   /* defined slots (0..26)          */
#define STILES_PARAM_ARRAY_SIZE           50   /* full ABI array incl. reserved  */

//==============================================================================
// Named Parameter Values
//==============================================================================

/* STILES_PARAM_CORRECTION_MODE */
#define STILES_CORRECTION_NONE             0   /* No pruning                    */
#define STILES_CORRECTION_SINGLE           1   /* Prune single zero active col  */
#define STILES_CORRECTION_TILES            2   /* Prune zero semisparse tiles   */
#define STILES_CORRECTION_COLUMNS          3   /* Prune zero semisparse columns */

/* STILES_PARAM_TILE_SIZE */
#define STILES_TILE_AUTO                  -1   /* Auto-detect tile size         */

/* STILES_PARAM_ORDERING_MODE — digits for sTiles_set_ordering_mode.
 * 0 = adaptive selection (default). Otherwise concatenate digits to name the
 * exact bake-off candidate set (best fill wins), e.g. 167 = RCM+AMD+CAMD,
 * a single digit forces that ordering alone.                                */
#define STILES_ORD_RCM                     1   /* Reverse Cuthill-McKee        */
#define STILES_ORD_METIS                   2   /* METIS nested dissection      */
#define STILES_ORD_SCOTCH                  3   /* SCOTCH nested dissection     */
#define STILES_ORD_ASCOTCH                 4   /* SCOTCH variant A             */
#define STILES_ORD_FSCOTCH                 5   /* SCOTCH variant F             */
#define STILES_ORD_AMD                     6   /* Approximate minimum degree   */
#define STILES_ORD_CAMD                    7   /* Constrained AMD              */
#define STILES_ORD_COLAMD                  8   /* Column AMD                   */

/* STILES_PARAM_TILE_TYPE */
#define STILES_TILE_DENSE                  0   /* Dense tiles                   */
#define STILES_TILE_SEMISPARSE             1   /* Semisparse tiles              */
#define STILES_TILE_SPARSE                 2   /* Non-uniform (sparse) tiles    */
#define STILES_TILE_AUTO_SELECT            3   /* Auto: resolve 0/1/2 after symbolic */
#define STILES_TILE_DENSE_SEMISPARSE       3   /* DEPRECATED alias of AUTO_SELECT */

/* STILES_PARAM_FORCE_ND */
#define STILES_ND_AUTO                     0   /* Auto-detect ordering          */
#define STILES_ND_FORCE                    1   /* Force nested dissection       */

/* STILES_PARAM_INVERSE_STORAGE */
#define STILES_INV_OVERWRITE               0   /* Overwrite factor in-place     */
#define STILES_INV_SEPARATE                1   /* Separate inverse storage      */

/* STILES_PARAM_USE_OMP */
#define STILES_THREAD_PTHREADS             0   /* Use pthreads (default)        */
#define STILES_THREAD_OMP                  1   /* Use OpenMP                    */

/* STILES_PARAM_SEMISPARSE_IMPL */
#define STILES_SEMI_ORIGINAL               0   /* Original implementation       */
#define STILES_SEMI_IMPROVED               1   /* Improved implementation       */
#define STILES_SEMI_VECTORIZED             2   /* Vectorized implementation     */
#define STILES_SEMI_SERIAL_SPARSE          3   /* Vectorized + 1-core sparse-aware path */

/* STILES_PARAM_GPU_COMPARE */
#define STILES_GPU_ONLY                    0   /* GPU results only              */
#define STILES_GPU_WITH_CPU                1   /* GPU with CPU validation       */

/* STILES_PARAM_GPU_ENABLE */
#define STILES_GPU_DISABLED                0   /* GPU disabled                  */
#define STILES_GPU_ENABLED                 1   /* GPU enabled                   */

/* STILES_PARAM_MEMORY_ESTIMATE */
#define STILES_MEMEST_SKIP                 0   /* Skip memory estimation        */
#define STILES_MEMEST_PRINT                1   /* Compute & print estimate      */

/* STILES_PARAM_SERIAL_MODE */
#define STILES_SERIAL_AUTO                 0   /* Serial when 1 core, parallel otherwise */
#define STILES_SERIAL_ALWAYS_PARALLEL      1   /* Always use parallel kernels   */

/* STILES_PARAM_BW_MODE */
#define STILES_BW_CONSERVATIVE             0   /* Bandwidth = tile width - 1    */
#define STILES_BW_TIGHT                    1   /* Bandwidth = la - fa           */

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// User Parallel Execution Callback
//==============================================================================
/**
 * @brief User-defined function type for parallel execution.
 *
 * @param rank       Thread rank (0 to nthreads-1)
 * @param nthreads   Total number of threads in the bound team
 * @param user_data  Pointer to user-provided data (passed through from sTiles_parallel_exec)
 */
typedef void (*sTiles_user_func_t)(int rank, int nthreads, void* user_data);

//==============================================================================
// Core Interface
//==============================================================================
const char* sTiles_get_version(void);
void sTiles_print_version(void);
void sTiles_set_log_level(int);
void sTiles_set_tile_size(int);
int  sTiles_return_tile_size(void);
int  sTiles_get_auto_tile_size(void);
void sTiles_set_control_param(int index, int value);

/**
 * @brief Get a specific control parameter value.
 *
 * Retrieves the current value of an internal control parameter.
 * Use this to query configuration state set by other functions.
 *
 * @param index The parameter index (0-26; use sTiles_get_param_description
 *               for the per-value meaning of each slot):
 *   - 0: Semisparse pruning mode (set via sTiles_set_correction_mode)
 *   - 1: Tile size (set via sTiles_set_tile_size; -1 = auto, the default)
 *   - 2: Ordering candidate list (set via sTiles_set_ordering_mode):
 *        0 = adaptive (default); digits 1..8 = exact candidate set
 *        (1=RCM 2=METIS 3=SCOTCH 4=ASCOTCH 5=FSCOTCH 6=AMD 7=CAMD 8=COLAMD)
 *   - 3: Tile type mode (set via sTiles_set_tile_type_mode):
 *        0=dense, 1=semisparse, 2=non-uniform, 3=auto
 *        Default: 1 (semisparse) on all platforms
 *   - 4: Tile ordering mode (set via sTiles_set_tile_ordering_mode)
 *   - 5: Tile ordering size (set via sTiles_set_tile_ordering_size;
 *        -1 = auto: tile_size/2, the default)
 *   - 6: Force nested dissection mode (set via sTiles_force_ND)
 *   - 7: Inverse storage mode (set via sTiles_set_inverse_storage_mode)
 *   - 8: Parallelization mode (set via sTiles_set_use_omp): 0=pthreads, 1=OMP
 *        Default: pthreads on Linux, OMP on macOS
 *   - 9: Semisparse implementation (set via sTiles_set_semisparse_impl):
 *        0=original, 1=improved, 2=vectorized (default), 3=vectorized+sparse-aware
 *   - 10-11: GPU compare mode / GPU enable
 *   - 12: Memory estimate mode (set via sTiles_set_memory_estimate): 0=skip (default), 1=compute & print
 *   - 13: DEPRECATED (was tile-ordering min dim; stored, never read)
 *   - 14-15: Serial mode / Bandwidth mode
 *   - 16-19: Reserved
 *   - 20: Force SCOTCH ordering (set via sTiles_set_force_scotch_ordering)
 *   - 21: SCOTCH padding (set via sTiles_set_scotch_padding)
 *   - 22: Path 2 ND scheduling depth (set via sTiles_set_path2_depth)
 *   - 23-24: Reserved
 *   - 25: Tree-reduction path enable (set via sTiles_set_tree_path_enable; default 1)
 *   - 26: Tree-reduction path force (set via sTiles_set_tree_path_force)
 * @return The parameter value, or -1 if index is out of range (a warning is
 *         logged; note -1 is also a valid stored value for slots 1 and 5).
 */
int  sTiles_get_control_param(int index);
const char* sTiles_get_param_description(int index);
void sTiles_reset_control_param(int index);
void sTiles_reset_all_params(void);

/** Bake-off candidate list: 0 = adaptive (default); digits 1..8 select the
 *  exact set of orderings to evaluate (STILES_ORD_*: 1=RCM 2=METIS 3=SCOTCH
 *  4=ASCOTCH 5=FSCOTCH 6=AMD 7=CAMD 8=COLAMD). A single digit pins one
 *  ordering, e.g. sTiles_set_ordering_mode(STILES_ORD_AMD). */
void sTiles_set_ordering_mode(int value);
void sTiles_set_tile_ordering_mode(int value);
/** DEPRECATED, no effect (slot 16 is never read). */
void sTiles_set_tile_first_ordering_mode(int value);
void sTiles_set_tile_ordering_size(int value);
/** DEPRECATED, no effect (slot 13 is never read). */
void sTiles_set_tile_ordering_min_dim(int value);
void sTiles_set_correction_mode(int value);
void sTiles_set_tile_type_mode(int value);
void sTiles_force_ND(int enable);
void sTiles_set_inverse_storage_mode(int mode);
int  sTiles_get_inverse_storage_mode(void);
void sTiles_set_use_omp(int mode);
int  sTiles_get_use_omp(void);
void sTiles_set_semisparse_impl(int impl);
int  sTiles_get_semisparse_impl(void);
/** Force SCOTCH as winner in main symbolic_phase (0=off default, 1=on).
 *  Replaces STILES_FORCE_ORDERING env var. */
void sTiles_set_force_scotch_ordering(int on);
int  sTiles_get_force_scotch_ordering(void);
/** Enable top-level SCOTCH block padding + populate partition_sizes so
 *  collect_tasks Path 2 engages. 0=off default, 1=on. Replaces
 *  STILES_ENABLE_SCOTCH_PADDING env var. */
void sTiles_set_scotch_padding(int on);
int  sTiles_get_scotch_padding(void);
/** Path 2 ND scheduling depth. 0 or 1 = legacy 3-way (P1|P2|Sep); >=2 uses
 *  the SCOTCH tree at the requested depth to produce 2^(D+1)-1 regions
 *  (2^D leaves + 2^D-1 separators). Only engages when scotch_padding is on. */
void sTiles_set_path2_depth(int depth);
int  sTiles_get_path2_depth(void);
void sTiles_set_neighbor_lookup_method(int value);
int  sTiles_get_neighbor_lookup_method(void);
void sTiles_set_serial_mode(int mode);
int  sTiles_get_serial_mode(void);
void sTiles_set_memory_estimate(int enable);
int  sTiles_get_memory_estimate(void);
void sTiles_set_bw_mode(int mode);
int  sTiles_get_bw_mode(void);
// Global enable for the corner-probe tree-reduction activation. 0 = off (default), 1 = on.
// Call sTiles_expert_user() first. Env var STILES_CORNER_PROBE_ACTIVATE=1 is also honored.
void sTiles_set_tree_path_enable(int on);
int  sTiles_get_tree_path_enable(void);
// Bypass the 6-gate activation predicate and force the tree-reduction path
// on, regardless of corner thickness / heavy-slot count / aggregate work.
// 0 = honor gate (default), 1 = force on. Used by kernel-correctness tests.
// Env var STILES_CORNER_PROBE_FORCE=1 is also honored (either trigger forces).
void sTiles_set_tree_path_force(int on);
int  sTiles_get_tree_path_force(void);
void sTiles_set_params(const int* params, int n);
void sTiles_get_all_params(int* params, int* size);
void sTiles_print_params(void);
/** Print every STILES_* environment variable the library reads (current
 *  value, default, one-line description). Env vars are runtime overrides and
 *  debug hooks; prefer the sTiles_set_* API for stable configuration. */
void sTiles_print_env(void);
void sTiles_set_dense_export_file(const char* filepath);
const char* sTiles_get_dense_export_file(void);
void sTiles_set_rescale_cores(const int* rescale_list, int num_counts);
void sTiles_set_user_permutation(int group_index, const int* perm, int n, bool force);
void sTiles_clear_user_permutation(int group_index);
void sTiles_clear_all_user_permutations(void);
void sTiles_set_partition_sizes(int group_index, int p1_size, int p2_size, int sep_size, bool force);
void sTiles_clear_partition_sizes(int group_index);
const int* sTiles_get_partition_sizes(int group_index, void** stile);

//==============================================================================
// GPU Control (requires STILES_GPU)
//==============================================================================
/**
 * @brief Get the number of available CUDA GPUs.
 * @return Number of GPUs, or 0 if no GPU support or no devices found.
 */
int sTiles_get_num_gpus(void);

/**
 * @brief Enable or disable GPU acceleration globally.
 *
 * Call this BEFORE sTiles_create() or sTiles_init_group() to control
 * whether GPU acceleration is used. When disabled, all computations
 * run on CPU regardless of available GPUs.
 *
 * @param enable 1 to enable GPU (default), 0 to disable GPU
 */
void sTiles_use_gpu(int enable);

/**
 * @brief Get the current GPU enable/disable setting.
 * @return 1 if GPU is enabled (default), 0 if disabled
 */
int sTiles_get_gpu_enabled(void);

/**
 * @brief Set GPU device ID for a specific call.
 *
 * Call this AFTER sTiles_create() but BEFORE sTiles_init_group() to assign
 * a specific GPU to a call. Use for multi-GPU testing.
 *
 * @param group_index Group index (typically 0)
 * @param call_index Call index within the group
 * @param gpu_id GPU device ID (0, 1, 2, ...)
 * @param stile Pointer to sTiles handle
 * @return 0 on success, -1 on error
 */
int sTiles_set_gpu_for_call(int group_index, int call_index, int gpu_id, void** stile);

/**
 * @brief Check if a call is using GPU acceleration.
 * @param group_index Group index
 * @param call_index Call index
 * @param stile Pointer to sTiles handle
 * @return 1 if GPU is used, 0 otherwise
 */
int sTiles_is_using_gpu(int group_index, int call_index, void** stile);

//==============================================================================
// Object Creation and Initialization
//==============================================================================
int sTiles_create(void**, int, const int*, const int*, const int*, const bool*);
int sTiles_create_expert(void**, int, const int*, const int*, const int*, const bool*, const int*, const int*, const int*);
int sTiles_init(void**);
int sTiles_init_group(int, void**);
/* Ordering + symbolic only: populate scheme->element_perm for every group with
 * NO numeric factor arena (covers all numeric modes). Read the perm via
 * sTiles_return_perm_vec; complete the factor later with sTiles_init (numeric),
 * forcing that perm via sTiles_set_user_permutation. */
int  sTiles_init_symbolic(void**);
void sTiles_set_symbolic_only(int);
int  sTiles_get_symbolic_only(void);
/* Register a callback fired the instant group 0's ordering+symbolic is done
 * (element_perm ready), BEFORE the numeric arena -- so a caller can ship the
 * perm mid-init and overlap the rest. Pass NULL to clear. cb(group,call,perm,n). */
void sTiles_set_symbolic_done_callback(void (*cb)(int, int, const int *, int));
void sTiles_map_group_call_to_group_call(void**, int, int, int, int);

/** Pack L into flat CSC layout for one call. Must be called *between*
 *  sTiles_chol and sTiles_unbind, on the same (group_index, call_index)
 *  the chol used. The pack runs on the bound thread pool exactly like
 *  sTiles_chol — same param[8] dispatch (OMP vs pthreads), same workers,
 *  same bindings. This makes parallel-call workloads safe: each pack
 *  touches only its own (group, call)'s pool.
 *  After packing, single-core nrhs==1 solves on this call take the
 *  csc_dtrsm shortcut; multi-core / multi-RHS solves are unchanged.
 *  Returns 0 on success. */
int sTiles_packing(int group_index, int call_index, void** obj);

/** Free the packed L_values for one call. Symmetric to sTiles_packing.
 *  Subsequent solves revert to the parallel tile path. */
int sTiles_unpacking(int group_index, int call_index, void** obj);

/** Bench-only: time csc_dtrsm_multi (column-major X) vs csc_dtrsm_multi_row
 *  (row-major X) on the packed scheme. Skips the permute pipeline. Best
 *  per-iteration wall-clock written to *out_col_s and *out_row_s. */
int sTiles_bench_csc_layouts(int group_index, int call_index, void** obj,
                             int nrhs, int repeats,
                             double* out_col_s, double* out_row_s);

//==============================================================================
// Graph and Data Assignment
//==============================================================================
int sTiles_assign_graph(int, void**, int, int, int*, int*);
int sTiles_assign_graph_one_call(int, int, void**, int, int, int*, int*);
int sTiles_assign_values(int, int, void**, double*);

//==============================================================================
// Binding and Execution
//==============================================================================
int sTiles_bind(int, int, void**);
int sTiles_unbind(int, int, void**);
int sTiles_chol(int, int, void**);
int sTiles_selinv(int, int, void**);

/**
 * @brief Execute a user-defined function in parallel on the bound thread team.
 *
 * Must be called between sTiles_bind() and sTiles_unbind(). The user function
 * is dispatched to all threads in the bound team and executed in parallel.
 *
 * @param group_index  Group index (same as used in sTiles_bind)
 * @param call_index   Call index within the group
 * @param obj          Pointer to the sTiles handle
 * @param user_func    User function to execute (receives rank, nthreads, user_data)
 * @param user_data    Pointer to user-provided data (passed through to user_func)
 * @return 0 on success, non-zero on error
 */
int sTiles_parallel_exec(int group_index, int call_index, void** obj,
                         sTiles_user_func_t user_func, void* user_data);

int sTiles_rescale_cores(int group_index, int call_index, int new_cores, void** obj);
int sTiles_turn_on_rescale(int group_index, void** obj);
int sTiles_turn_off_rescale(int group_index, void** obj);

double sTiles_get_selinv_elm(int, int, int, int, void**);
double* sTiles_get_selinv_row(int, int, int, int*, int, void**);
double sTiles_get_chol_elm(int, int, int, int, void**);
int sTiles_clear_selinv(int, int, void**);

//==============================================================================
// Linear Solvers
//==============================================================================
int sTiles_solve_LLT(int, int, void**, double*, int);
int sTiles_solve_L(int, int, void**, double*, int);
int sTiles_solve_LT(int, int, void**, double*, int);

// Rescale variants (use different thread count than factorization)
int sTiles_solve_LLT_rescale(int, int, void**, double*, int, int, int);
int sTiles_solve_L_rescale(int, int, void**, double*, int, int, int);
int sTiles_solve_LT_rescale(int, int, void**, double*, int, int, int);

//==============================================================================
// Query Utilities
//==============================================================================
int* sTiles_return_perm_vec(int, void**);
int* sTiles_return_iperm_vec(int, void**);
int  sTiles_get_num_calls(void**, int);
long long sTiles_get_nnz_factor(int, int, void**);
double* sTiles_get_L_values(int, int, void**);
void sTiles_set_pack_cache_threshold_bytes(long long bytes);
long long sTiles_get_pack_cache_threshold_bytes(void);
const int* sTiles_get_element_perm(int, int, void**);
/* Length of the array returned by sTiles_get_element_perm. This is the PADDED
 * scheme dimension (original n + ND padding), which can exceed the input matrix
 * dimension when ND block padding is applied. Returns -1 on error. */
int sTiles_get_element_perm_length(int, int, void**);
/* Logical element permutation over the ORIGINAL nodes only, rank-compressed to
 * [0, n_logical) with any ND-padding slots removed. Writes n_logical entries into
 * caller-allocated out_perm (allocate at least the input matrix dimension) and
 * returns n_logical (= padded dim - nd_padding = the input matrix dimension), or
 * -1 on error. Prefer this over reading the raw sTiles_get_element_perm when you
 * want a permutation of the input matrix: truncating the raw (padded) perm to the
 * input dimension is NOT a valid permutation. */
int sTiles_get_logical_element_perm(int, int, void**, int* out_perm);

double sTiles_get_logdet(int, int, void**);
double sTiles_get_chol_timing(int, int, void**);
double sTiles_get_selinv_timing(int, int, void**);
void sTiles_print_chol_timings(int, void**);
void sTiles_print_selinv_timings(int, void**);
void sTiles_print_logdets(int, void**);
double sTiles_debug_matrix(int, int, void** );

/**
 * @brief Enable expert mode for sTiles configuration.
 *
 * This function MUST be called before using any configuration functions.
 * Without calling this first, the following functions will print a warning
 * and return without making changes:
 *   - sTiles_set_tile_size()
 *   - sTiles_set_ordering_mode()
 *   - sTiles_set_correction_mode()
 *   - sTiles_set_tile_type_mode()
 *   - sTiles_set_tile_ordering_mode()
 *   - sTiles_set_tile_ordering_size()
 *   - sTiles_force_ND()
 *   - sTiles_set_inverse_storage_mode()
 *   - sTiles_set_control_param()
 *
 * Call once at program startup if you need to customize sTiles behavior.
 */
void sTiles_expert_user(void);

//==============================================================================
// Memory Management
//==============================================================================
double sTiles_GetGroupMemoryUsage(int);
double sTiles_GetGroupsMemoryUsage(void);
void   sTiles_freeGroup(int);
void   sTiles_quit(void);
int    sTiles_check(void**);

//==============================================================================
// Memory Estimation (call before sTiles_create to check if matrix fits in RAM)
//==============================================================================
/**
 * @brief Estimate memory required for sparse Cholesky factorization.
 *
 * Call this BEFORE sTiles_create() to check if your matrix will fit in RAM.
 * This is an early estimate based on matrix parameters and typical fill-in ratios.
 *
 * @param n                    Matrix dimension (number of rows/columns)
 * @param nnz                  Number of non-zeros in the input matrix
 * @param tile_size            Tile size (use 0 for auto-detection, typically 40 for CPU)
 * @param variant              Factorization variant — must be 0, 1, or 2:
 *                             0 = Sparse, 1 = full dense, 2 = scaled dense.
 *                             Other values are clamped to 0 with a warning.
 * @param compute_inverse      Whether inverse will be computed (1=yes, 0=no) - triples memory
 * @param use_nested_dissection Whether ND ordering will be used (1=yes, 0=no) - affects fill-in
 * @return Estimated total memory usage in GB
 */
double sTiles_estimate_memory(int n, int nnz, int tile_size, int variant,
                              int compute_inverse, int use_nested_dissection);

/**
 * @brief Estimate memory with detailed breakdown printed to console.
 *
 * Same as sTiles_estimate_memory() but also prints a detailed breakdown showing:
 * - Dense tiles memory
 * - Inverse tiles memory (if compute_inverse)
 * - Sparse tiles memory (variant 0)
 * - Metadata and lookup tables
 * - Thread workspace memory
 *
 * @return Estimated total memory usage in GB
 */
double sTiles_estimate_memory_verbose(int n, int nnz, int tile_size, int variant,
                                      int compute_inverse, int use_nested_dissection);

/**
 * @brief Check if matrix will fit in available RAM.
 *
 * @param n                    Matrix dimension
 * @param nnz                  Number of non-zeros
 * @param tile_size            Tile size (0 for auto)
 * @param variant              Factorization variant — must be 0, 1, or 2
 *                             (0 = Sparse, 1 = full dense, 2 = scaled dense).
 *                             Other values are clamped to 0 with a warning.
 * @param compute_inverse      Whether inverse will be computed
 * @param use_nested_dissection Whether ND ordering will be used
 * @param available_ram_gb     Available RAM in GB
 * @return 1 if matrix fits (with 10% safety margin), 0 if it doesn't fit
 */
int sTiles_memory_fits(int n, int nnz, int tile_size, int variant,
                       int compute_inverse, int use_nested_dissection,
                       double available_ram_gb);

#ifdef __cplusplus
}
#endif

#endif /* _STILES_MAIN_HEADER_H_ */
