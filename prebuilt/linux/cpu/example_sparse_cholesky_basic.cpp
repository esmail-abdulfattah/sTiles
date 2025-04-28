/**
 * @file example_sparse_cholesky_basic.cpp
 * @brief A simple testing program for the sTiles library demonstrating sparse Cholesky factorization and selected inverse computation.
 *
 * This program constructs a small sparse lower triangular matrix,
 * sets up the sTiles environment, assigns the graph structure and values,
 * performs Cholesky factorization, computes the log-determinant,
 * and demonstrates how to compute selected entries of the matrix inverse.
 * 
 * Key concepts demonstrated:
 * - Separating matrix structure (graph) assignment from numerical value assignment
 * - Optional tile size configuration
 * - Group and matrix ID mapping
 * - Preprocessing phase before value assignment
 * - Performing sparse Cholesky factorization using sTiles
 * - Computing selected entries of the inverse efficiently after factorization
 * 
 * Author: Esmail Abdul Fattah
 * Date: 27-04-2025
 */


 #include "./include/stiles.h"
 #include <iostream>
 #include <cstdio>
 #include <cstdlib>
 #include <vector>
 #include <omp.h>
 
 /**
  * @struct SparseMatrix
  * @brief A simple struct to hold sparse matrix data in COO (Coordinate) format.
  *
  * Contains:
  * - Matrix size (N)
  * - Number of nonzeros (NNZ)
  * - Row indices
  * - Column indices
  * - Nonzero values
  */
 struct SparseMatrix {
     int     N = 0;               ///< Number of rows/columns (square matrix)
     int     NNZ = 0;             ///< Number of nonzero entries
     int*    row_indices = nullptr; ///< Array of row indices
     int*    col_indices = nullptr; ///< Array of column indices
     double* values = nullptr;      ///< Array of nonzero values
 };
 
 int main() {
 
     // -------------------------------------------------------------------------
     // 1. Group and call configuration
     // -------------------------------------------------------------------------
 
     int num_groups             = 1;       // Number of groups
     int calls_per_group[]      = {1};      // One matrix call per group
     int cores_per_group[]      = {1};      // 6 cores assigned to Group 0
     int chol_type_per_group[]  = {0};      // 0 = Sparse Cholesky factorization
     bool get_inverse[]         = {true};  // Do not compute the inverse
 
     // -------------------------------------------------------------------------
     // 2. Create a simple sparse matrix (5x5 lower triangular part)
     // -------------------------------------------------------------------------
 
     // Matrix layout (lower triangular 5x5):
     //
     // [5,  0,  0,  0,  0]
     // [-1, 7,  0,  0,  0]
     // [0, -2, 9,  0,  0]
     // [0,  0, -3, 11, 0]
     // [0,  0,  0, -4, 13]
     //
     SparseMatrix mat;
     mat.N = 5;
     mat.NNZ = 9;
 
     mat.row_indices = (int*)malloc(mat.NNZ * sizeof(int));
     mat.col_indices = (int*)malloc(mat.NNZ * sizeof(int));
     mat.values      = (double*)malloc(mat.NNZ * sizeof(double));
 
     int row_vals[9] =   {0, 1, 1, 2, 2, 3, 3, 4, 4};
     int col_vals[9] =   {0, 0, 1, 1, 2, 2, 3, 3, 4};
     double val[9]   =   {5, -1,
                               7, -2,    
                                   9, -3, 
                                       11, -4, 
                                            13};
 
     for (int i = 0; i < mat.NNZ; i++) {
         mat.row_indices[i] = row_vals[i];
         mat.col_indices[i] = col_vals[i];
         mat.values[i]      = val[i];
     }
 
     // -------------------------------------------------------------------------
     // 3. Initialize the sTiles object
     // -------------------------------------------------------------------------
 
     sTiles_object* stile = nullptr;
     sTiles_create(&stile, num_groups, calls_per_group, cores_per_group, chol_type_per_group, get_inverse, nullptr);
 
     // -------------------------------------------------------------------------
     // 4. Assign graph and values, and perform Cholesky factorization
     // -------------------------------------------------------------------------
 
     // Step 4.1: Assign the sparsity pattern (graph structure) of the matrix.
     //
     // Syntax:
     //   sTiles_assign_graph(group_id, stile, N, NNZ, row_indices, col_indices)
     //
     // Here:
     // - group_id = 0 → Refers to Group 0.
     //
     // Purpose:
     // Defines the nonzero structure of the matrix (rows, columns),
     // without yet providing the numerical values.
     // Multiple matrices in the same group can reuse the structure, saving preprocessing time.
     sTiles_assign_graph(0, stile, mat.N, mat.NNZ, mat.row_indices, mat.col_indices);
 
     // Step 4.2: (Optional) Set the tile size for decomposition.
     //
     // Syntax:
     //   sTiles_set_tile_size(tile_size)
     //
     // - tile_size = 2 → Matrix will be divided into 2x2 tiles.
     // If not called, a default tile size will be used.
     sTiles_set_tile_size(2);
 
     // Step 4.3: Initialize internal data structures after assigning graph.
     //
     // Syntax:
     //   sTiles_init(&stile)
     //
     // Purpose:
     // - Symbolic analysis
     // - Task graph generation
     // - Allocating memory for numerical phase
     sTiles_init(&stile);
 
     // Step 4.4: Assign the numerical values to the matrix.
     //
     // Syntax:
     //   sTiles_assign_values(group_id, matrix_id, &stile, values)
     //
     // Here:
     // - group_id  = 0 → Group 0
     // - matrix_id = 0 → First matrix inside Group 0
     //
     // Now the numerical values are provided, after the structure was fixed.
     sTiles_assign_values(0, 0, &stile, mat.values);
 
     // Step 4.5: Bind the matrix before performing computations.
     //
     // Syntax:
     //   sTiles_bind(group_id, matrix_id, &stile)
     //
     // Here:
     // - group_id  = 0 → Group 0
     // - matrix_id = 0 → Matrix 0 in Group 0
     //
     // Binding activates the specific matrix for computation.
     sTiles_bind(0, 0, &stile);
 
     // Step 4.6: Perform Cholesky factorization on the bound matrix.
     //
     // Syntax:
     //   sTiles_chol(group_id, matrix_id, &stile)
     //
     // - group_id  = 0 → Group 0
     // - matrix_id = 0 → Matrix 0 in Group 0
     //
     // Computes the Cholesky factorization A = L * Lᵀ.
     int status = sTiles_chol(0, 0, &stile);
     if (status != 0) {
         printf("Error in sTiles_chol (%d, %d): status = %d\n", 0, 0, status);
         sTiles_quit();
         return EXIT_FAILURE;
     }
 
     // Step 4.7: (Optional) Retrieve the log-determinant after factorization.
     double logdet = sTiles_get_logdet(0, 0, &stile);
     std::cout << "LogDet = " << logdet << "\n";
 
    // -------------------------------------------------------------------------
    // 5. (Optional) Compute the selected inverse
    // -------------------------------------------------------------------------

    // Step 5.1: Compute selected inverse entries (diagonal or specific entries) after Cholesky.
    //
    // Syntax:
    //   sTiles_selinv(group_id, matrix_id, &stile)
    //
    // Here:
    // - group_id  = 0 → Group 0
    // - matrix_id = 0 → Matrix 0 in Group 0
    //
    // Purpose:
    // - Computes selected entries of the inverse matrix without forming the full dense inverse.
    // - Efficient when only a small subset (e.g., diagonal entries) of the inverse is needed.
    sTiles_selinv(0, 0, &stile);

    // Step 5.2: Access specific entries from the selected inverse.
    //
    // Syntax:
    //   sTiles_get_selinv_elm(group_id, matrix_id, row, col, &stile)
    //
    // Example: Print the first 5 diagonal elements (i,i) of the inverse matrix.
    for (int i = 0; i < 5; i++) {
        std::cout << "Inverse(" << i << "," << i << ") = " << sTiles_get_selinv_elm(0, 0, i, i, &stile) << "\n";
    }

    sTiles_unbind(0, 0, &stile); // Unbind the matrix after computation.

    // -------------------------------------------------------------------------
    // 6. Cleanup
    // -------------------------------------------------------------------------

    sTiles_quit();  // Finalize the sTiles environment
    free(mat.row_indices);
    free(mat.col_indices);
    free(mat.values);

    std::cout << "\nProcess completed successfully :)\n";
    return EXIT_SUCCESS;
 }
 