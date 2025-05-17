/**
 * @file stiles.h
 *
 * Header file for sTiles matrix and tree descriptor structures and auxiliary routines.
 * sTiles is an advanced extension of the PLASMA software package, originally developed by:
 * - University of Tennessee
 * - University of California, Berkeley
 * - University of Colorado, Denver
 *
 * Redesigned and significantly improved for sTiles by Esmail Abdul Fattah, 
 * King Abdullah University of Science and Technology (KAUST), and the sTiles team.
 *
 * This file defines essential data structures, constants, and utility functions
 * used in the sTiles framework for efficient linear algebra computations and 
 * symbolic factorization routines.
 *
 * @version 1.0.0
 * @redesigned_by Esmail Abdul Fattah
 * @original_authors Jakub Kurzak, Mathieu Faverge
 * @contact esmail.abdulfattah@kaust.edu.sa
 * @date 2025-01-30
 */


#ifndef _STILES_STRUCTS_H_
#define _STILES_STRUCTS_H_

#include <stdio.h>
#include <stdbool.h>


#ifdef STILES_GPU
    #include <cuda_runtime.h>
    #include <cusolverDn.h>
    #include <cublas_v2.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** ****************************************************************************
 * Debugging and configuration macros
 ******************************************************************************/
#define DEBUG 
#define E_STILES_SYMBOLIC_FACTORIZATION /**< Boolean-based symbolic factorization */
//#define E_STILES_BIT_SYMBOLIC_FACTORIZATION // E_STILES_SYMBOLIC_FACTORIZATION should not be defined and bit is used if this is defined
//#define TECH_DEBUG

/** ****************************************************************************
 * Structure representing a single node within a tree leaf
 ******************************************************************************/
typedef struct NodeLeaf {
    int surviving_level; /**< Level at which this node survives */
    int index;           /**< Node index */
    int level;           /**< Current level of the node */
    int leafheight;      /**< Height of the leaf */
    int leafwidth;       /**< Width of the leaf */
    double *x;           /**< Data buffer for GEMM operations */
} NodeLeaf;

/** ****************************************************************************
 * Structure representing a tree of leaves for symbolic factorization
 ******************************************************************************/
typedef struct TreeLeaf {
    NodeLeaf *nodes;       /**< Array of nodes */
    int *max_nodes;        /**< Maximum number of nodes per level */
    int *gold_number;      /**< Array of "gold" numbers for calculations */
    int *half_gold;        /**< Auxiliary array for intermediate values */
    int silver_number;     /**< "Silver" number for calculations */
    int num_splits;
    int num_tasks;
    int num_nodes;         /**< Total number of nodes */
    int counter_nodes;     /**< Counter of nodes at each level */
    int max_levels;        /**< Maximum number of levels in the tree */
    int *dependency;       /**< Dependency vector */
} TreeLeaf;

/** ****************************************************************************
 * Structure representing a matrix tile
 ******************************************************************************/
typedef struct {
    int row, col;       /**< Row and column indices of the tile */
    int width, height;  /**< Dimensions of the tile */
    double *elements;   /**< Pointer to the tile's data */
} TILE;

/** ****************************************************************************
 * Structure representing a matrix tile for GPU
 ******************************************************************************/
typedef struct {
    int width, height;  /**< Dimensions of the tile */
    double *x;  // Pointer to a 2D array
} TILE_GPU;

/** ****************************************************************************
 * Structure representing a configuration for specific solution types
 ******************************************************************************/
typedef struct {
    int STYLE_NRHS_SOLVE;      /**< Number of right-hand sides */
    int **e_trick_solve;       /**< Pointer to solution tricks */
    int *e_trick_size_solve;   /**< Sizes of solution tricks */
} E_TRICK_SOLVE;

/**
 * @brief Structure representing a matrix descriptor in the sTiles framework.
 *
 * This structure provides metadata and additional parameters for managing and
 * operating on tile-based matrices within the sTiles library. It includes 
 * information about matrix dimensions, tiling, submatrices, and optimization-specific
 * configurations.
 */
typedef struct stiles_desc_t {

    /** Matrix data pointers and offsets **/
    void *mat;          /**< Pointer to the beginning of the matrix data. */
    size_t A21;         /**< Offset to the A21 block in the matrix. */
    size_t A12;         /**< Offset to the A12 bloc#ifdef __cplusplus*/
    size_t A22;         /**< Offset to the A22 block in the matrix. */

    /** Matrix metadata **/
    int dtyp;   /**< Precision or data type of the matrix (e.g., float, double). */
    int mb;             /**< Number of rows in a single tile. */
    int nb;             /**< Number of columns in a single tile. */
    int bsiz;           /**< Total size of a tile, including padding (in elements). */
    int lm;             /**< Total number of rows in the full matrix. */
    int ln;             /**< Total number of columns in the full matrix. */

    /** Tiling and submatrix details **/
    int lm1;            /**< Number of tile rows in the A11 block (derived parameter). */
    int ln1;            /**< Number of tile columns in the A11 block (derived parameter). */
    int lmt;            /**< Total number of tile rows in the entire matrix. */
    int lnt;            /**< Total number of tile columns in the entire matrix. */
    int i;              /**< Row index of the starting point for a submatrix. */
    int j;              /**< Column index of the starting point for a submatrix. */
    int m;              /**< Number of rows in the submatrix. */
    int n;              /**< Number of columns in the submatrix. */
    int mt;             /**< Number of tile rows in the submatrix (derived parameter). */
    int nt;             /**< Number of tile columns in the submatrix (derived parameter). */

    /** Tiling and performance optimization details **/
    bool **on_off_tiles;       /**< Boolean matrix indicating active/inactive tiles. */
    double stiles_call;        /**< Timing or metadata related to tile calls. */
    TILE **tiles;              /**< Array of pointers to individual tiles. */
    TILE *dense_style;               /**< Pointer to the primary matrix tile representation. */
    TILE *dense_style_B;
    TILE *inv_dense_style;           /**< Pointer to the inverse of the matrix tile representation. */
    int call_index;            /**< Index for keeping track of matrix call operations. */
    int *separators;           /**< Array of separators for specific tree structures. */
    double *stiles;            /**< Auxiliary array for specific tile computations. */
    int *magic_perm1;          /**< Array for storing optimized permutations. */
    bool *of_perm;             /**< Flags indicating specific permutations. */
    bool activated_nd;         /**< Flag indicating if nested dissection ordering is active. */

    /** Trick-based optimizations **/
    int **e_trick;                  /**< Matrix of tricks for optimization. */
    int *e_trick_size;              /**< Sizes corresponding to trick entries. */
    int **e_trick_partition1;       /**< Tricks for the first partition. */
    int *e_trick_size_partition1;   /**< Sizes for the first partition tricks. */
    int **e_trick_partition2;       /**< Tricks for the second partition. */
    int *e_trick_size_partition2;   /**< Sizes for the second partition tricks. */
    int **e_trick_inv;              /**< Inverse tricks for optimization. */
    int *e_trick_size_inv;          /**< Sizes corresponding to inverse tricks. */

    /** Tree structures **/
    TreeLeaf **trees;         /**< Array of trees used for factorization and optimization. */
    int tree_sep;             /**< Tree separator level. */
    int tree_stgy;            /**< Strategy indicator for tree-based optimizations. */
    bool boosted_e_trick;     /**< Flag indicating if boosted tricks are enabled. */

    /** Additional metadata **/
    int original_N;           /**< Original matrix size before transformations. */
    int *flops_mat;           /**< Array tracking floating-point operations (FLOPs) for each tile. */
    double *B;                /**< Pointer to auxiliary matrix/vector used in computations. */
    int sindex;               /**< Index indicating a specific solution phase or step. */
    int sversion;             /**< Version identifier for internal configurations. */

    /** Solution-specific structures **/
    E_TRICK_SOLVE *E_TRICK_TYPE_0; /**< Trick configuration for type 0 solutions. */
    E_TRICK_SOLVE *E_TRICK_TYPE_1; /**< Trick configuration for type 1 solutions. */
    int total_tiles;          /**< Total number of tiles in the matrix. */

    TILE_GPU *dense_style_gpu;                       /**< Matrix tiles */
    TILE_GPU *inv_dense_style_gpu;                   /**< Inverse matrix tiles */
    TILE_GPU *gpu_trees;                       /**< Matrix tiles */
    int GPU_ID;

#ifdef STILES_GPU
    cudaStream_t *streams;  /**< Pointer to dynamically allocated CUDA streams */
    cudaEvent_t *events;
#endif

} STILES_desc;


/** ****************************************************************************
 * Structure defining the main sTiles computational scheme
 ******************************************************************************/
typedef struct {
    int STYLE_N;                       /**< Number of rows/columns */
    int STYLE_NNZ;                     /**< Number of non-zeros */
    int STYLE_ORIGINAL_N;              /**< Original matrix row/column count */
    int STYLE_ORIGINAL_NNZ;            /**< Original matrix non-zero count */
    int STYLE_TILE_SIZE;               /**< Size of a tile */
    int STYLE_LAST_TILE_SIZE;          /**< Size of the last tile */
    int STYLE_TILES_SIZE;              /**< Size of all tiles */
    int STYLE_TOTAL_NUM_USED_TILES;    /**< Total number of used tiles */
    int STYLE_FIXED_COL;               /**< Fixed column size */
    int STYLE_OF_SIZE;                 /**< Original factor size */
    int STYLE_NUM_CORES;               /**< Number of cores used */
    int STYLE_INTERNAL_VERSION;        /**< Version of internal processing */
    int STYLE_ORDERING;                /**< Matrix ordering type */
    int STYLE_ND_NNZ;                  /**< Number of non-zeros for ND ordering */
    int STYLE_ND_N;                    /**< Number of nodes for ND ordering */
    int STYLE_RED_TREE_SEP;            /**< Separation level for red tree */

    int *STYLE_NEWSIZES;               /**< New sizes array */
    int *STYLE_SIZES;                  /**< Sizes array */
    int *STYLE_CSC_MAPPING;            /**< Mapping for compressed sparse columns */
    int *STYLE_ELE_PERM;               /**< Element permutation */
    int *STYLE_ELE_IPERM;              /**< Inverse element permutation */
    int *STYLE_NEIGHBORS;              /**< Neighbor relationships */
    int *STYLE_NEIGHBORS_SIZES;        /**< Sizes of neighbor relationships */
    int *REVSTYLE_NEIGHBORS;           /**< Reverse neighbors */
    int *REVSTYLE_NEIGHBORS_SIZES;     /**< Reverse neighbor sizes */
    int *STYLE_TREE_COUNTER;

    int **STYLE_CALL_MATRIX;           /**< Matrix for function calls */
    unsigned char *STYLE_BIT_ARRAY;    /**< Bit array for operations */

    bool STYLE_BOOST_E_TRICK;          /**< Flag for boosting tricks */
    bool STYLE_GET_INVERSE;            /**< Flag for computing the inverse */
    bool STYLE_ON_GPU;           
    bool STYLE_GET_INVERSE_GPU;            
    bool STYLE_COPY_PREPROCESS;
    bool *STYLE_OF_PERM;               /**< Permutation flags */
    bool **on_off_tiles;               /**< On/off status of tiles */

    TILE *dense_style;                       /**< Matrix tiles */
    TILE *dense_style_B;
    TILE *saved_style;                 /**< Backup of matrix tiles */
    TILE *inv_dense_style;                   /**< Inverse matrix tiles */
    TreeLeaf **trees;                  /**< Array of tree structures */

    int **e_trick;                     /**< Trick data */
    int *e_trick_size;                 /**< Sizes of tricks */
    int **e_trick_inv;                 /**< Inverse tricks */
    bool *e_trick_copy_ind;                 /**< Inverse tricks */
    int *e_trick_size_inv;             /**< Sizes of inverse tricks */
    int *t_indicies;                   /**< Tile indices */
    int *e_indicies;                   /**< Element indices */

    E_TRICK_SOLVE *E_TRICK_TYPE_0;     /**< Trick type 0 */
    E_TRICK_SOLVE *E_TRICK_TYPE_1;     /**< Trick type 1 */

    TILE_GPU *dense_style_gpu;                       /**< Matrix tiles */
    TILE_GPU *inv_dense_style_gpu;                   /**< Inverse matrix tiles */
    TILE_GPU *gpu_trees;                       /**< Matrix tiles */
    int GPU_ID;

#ifdef STILES_GPU
    cudaStream_t *streams;  /**< Pointer to dynamically allocated CUDA streams */
    cudaEvent_t *events;
#endif

} STILES_CONFIGURATION;


typedef struct {

    int STILES_GLOBAL_INDEX;
    int STILES_CALL_INDEX;   // Index of the call within the group
    int STILES_CALL_INDEX_MAPPED;
    int STILES_GROUP_INDEX_MAPPED;
    int STILES_NUM_CORES;        // Number of STILES_CORES allocated for this call
    int *STILES_CORE_BIND_IDS;  // Array to store the IDs of the STILES_NUM_CORES bound to this call
    int STILES_LOCAL_OFFSET;
    bool STILES_SAVE_FACTOR; 
    bool STILES_GET_INVERSE; 
    bool STILES_GET_LOGDET; 
    int STILES_ORDERING; 
    int STILES_TICK; 
    int STILES_N;
    int STILES_NNZ;
    int* STILES_ROW_INDICES;  // <-- updated from row_indicies
    int* STILES_COL_INDICES;  // <-- updated from col_indicies
    double* STILES_X_VALUES;
    int STILES_RED_TREE_SEP; // Array for fixed columns per group 
    int STILES_TILE_SIZE; // Array for fixed columns per group 
    int STILES_CHOL_TYPE; // Array for fixed columns per group 
    int STILES_ARROWHEAD_THICK; // Array for fixed columns per group
    int STILES_PREPROCESS_LEVEL; // Array for fixed columns per group
    int* STILES_PARAMETERS; // Array for fixed columns per group
    bool STILES_ND;
    int STILES_BANDWIDTH;
    int STILES_NRHS;
} CALL_INFO;

typedef struct {

    int group_index;      // Index of the group
    int group_offset;      // Index of the group
    int num_calls;        // Number of STILES_CALLS in this group
    int arrowhead_size_per_group; // Array for fixed columns per group
    CALL_INFO *STILES_CALLS;     // Array of CALL_INFO structures for STILES_CALLS in this group
    bool STILES_SAME_GROUP; 
} GROUP_INFO;

typedef struct {

    //global parameters:
    
    int num_call_groups;          // Number of STILES_GROUPS
    int *num_calls_per_group;     // Array for number of STILES_CALLS per group
    int *num_cores_per_group;     // Array for number of STILES_NUM_CORES per group
    int *factorization_type_per_group;      // dense or sparse chol
    int max_cores_sys;       // Maximum number of STILES_NUM_CORES
    int num_total_indices;         // Total number of Cholesky indices (sum of all STILES_CALLS)
    bool numa_enabled;
    int* global_indicies;
    int** call_matrix;
    const int* rhs;

    GROUP_INFO *STILES_GROUPS;           // Array of GROUP_INFO to store information for each group and its STILES_CALLS
    STILES_CONFIGURATION** schemes;

} sTiles_object;

 
int sTiles_assign_graph(int group_index, sTiles_object* stile, int N, int NNZ, int* row_indices, int* col_indices);
int sTiles_init(sTiles_object **obj);
int sTiles_init_group(int group_index, sTiles_object **obj);
int sTiles_assign_values(int group_index, int call_index, sTiles_object **obj, double *x);
int sTiles_chol(int group_index, int call_index, sTiles_object **obj);
int sTiles_selinv(int group_index, int call_index, sTiles_object **obj);
double sTiles_get_selinv_elm(int group_index, int call_index, int irow, int icol, sTiles_object** obj);
double* sTiles_get_selinv_row(int group_index, int call_index, int node, int* node_neighbors, int size, sTiles_object** obj);
double sTiles_get_logdet(int group_index, int call_index, sTiles_object** obj);
int sTiles_solve_LLT(int group_index, int call_index, sTiles_object** obj, double *B, int sindex);
int sTiles_solve_L(int group_index, int call_index, sTiles_object** obj, double *B, int sindex);
int sTiles_solve_LT(int group_index, int call_index, sTiles_object** obj, double *B, int sindex);
int* sTiles_return_perm_vec(int group_index, sTiles_object **obj);
int* sTiles_return_iperm_vec(int group_index, sTiles_object **obj);
int sTiles_clear_selinv(int group_index, int call_index, sTiles_object **obj);






//___________________________MEMORY MANAGEMENT
double sTiles_GetGroupMemoryUsage(int group_ID);
double sTiles_GetGroupsMemoryUsage();
void sTiles_freeGroup(int group_ID);
void sTiles_quit();

//___________________________USER TO sTiles Object
int sTiles_create(sTiles_object **obj, int num_call_groups, const int *calls_per_group, const int *cores_per_group, const int *factor_type_per_group, const bool *get_inverse, const int *rhs);
int sTiles_create_expert(sTiles_object **obj, int num_call_groups, const int *calls_per_group, const int *cores_per_group, const int *factor_type_per_group, const bool *get_inverse, const int *rhs, const int *arrowhead_size, const int *arrowhead_size_per_group, const int *user_params); 
void sTiles_set_tile_size(int tile_size);
int sTiles_return_tile_size();
void sTiles_map_group_call_to_group_call(sTiles_object** obj, int group_index1, int call_index1, int group_index2, int call_index2);

//___________________________BINDING
int sTiles_bind(int group_index, int call_index, sTiles_object** obj);
int sTiles_unbind(int group_index, int call_index, sTiles_object** obj);





#ifdef __cplusplus
}
#endif


#endif /* _STILES_STRUCTS_H_ */



