/**
 * @file example_sparse_cholesky_from_binary.cpp
 * @brief Testing program for the sTiles library: sparse Cholesky factorization from a binary file.
 *
 * This program reads a sparse matrix stored in binary format (COO) from disk,
 * configures the sTiles environment, assigns graph and values, 
 * performs Cholesky factorization, and prints the log-determinant.
 *
 * Author: Esmail Abdul Fattah
 * Date: 27-04-2025
 */

#include "./include/stiles.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <omp.h>

/**
 * @struct SparseMatrix
 * @brief Structure to hold a simple sparse matrix in COO format.
 */
struct SparseMatrix {
    int     N = 0;
    int     NNZ = 0;
    int*    row_indices = nullptr;
    int*    col_indices = nullptr;
    double* values = nullptr;
};

/**
 * @brief Loads a sparse matrix from a binary file.
 * 
 * Binary file format:
 * - N (int): Matrix dimension
 * - NNZ (int): Number of non-zeros
 * - row_indices (int array of size NNZ)
 * - col_indices (int array of size NNZ)
 * - values (double array of size NNZ)
 */
bool loadSparseMatrix(const char* filename, SparseMatrix& mat);

int main() {

    // -------------------------------------------------------------------------
    // 1. Group and call configuration
    // -------------------------------------------------------------------------

    int num_groups             = 1;
    int calls_per_group[]      = {1};
    int cores_per_group[]      = {6};
    int chol_type_per_group[]  = {0};    // 0 = Sparse Cholesky
    bool get_inverse[]         = {true}; // No inverse computation

    // -------------------------------------------------------------------------
    // 2. Load sparse matrix from binary file
    // -------------------------------------------------------------------------

    SparseMatrix mat;
    const char* matrix_filename = "./matrices/debug/mat1.bin"; // <-- Change path if needed
    if (!loadSparseMatrix(matrix_filename, mat)) {
        std::cerr << "Failed to load matrix from binary file.\n";
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------------------
    // 3. Initialize the sTiles object
    // -------------------------------------------------------------------------

    sTiles_object* stile = nullptr;
    sTiles_create(&stile, num_groups, calls_per_group, cores_per_group, chol_type_per_group, get_inverse, nullptr);

    // -------------------------------------------------------------------------
    // 4. Assign graph and values, and perform Cholesky factorization
    // -------------------------------------------------------------------------

    // Assign graph structure (row indices, column indices)
    sTiles_assign_graph(0, stile, mat.N, mat.NNZ, mat.row_indices, mat.col_indices);

    // (Optional) Set tile size. If not set, a default tile size will be used.
    sTiles_set_tile_size(40);

    // Initialize preprocessing structures (symbolic factorization, task graph).
    sTiles_init(&stile);

    // Assign numerical values (after structure is fixed).
    sTiles_assign_values(0, 0, &stile, mat.values);

    // Bind the matrix (Group 0, Matrix 0) before factorization.
    sTiles_bind(0, 0, &stile);

    // Perform Cholesky factorization
    int status = sTiles_chol(0, 0, &stile);
    if (status != 0) {
        printf("Error in sTiles_chol (%d, %d): status = %d\n", 0, 0, status);
        sTiles_quit();
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------------------
    // 5. Compute the selected inverse (optional)
    // -------------------------------------------------------------------------

    // Step 5.1: Compute the selected inverse of the factorized matrix.
    //
    // Syntax:
    //   sTiles_selinv(group_id, matrix_id, &stile)
    //
    // Here:
    // - group_id  = 0 → Group 0
    // - matrix_id = 0 → Matrix 0 in Group 0
    //
    // Purpose:
    // After Cholesky factorization, this function computes selected entries
    // of the inverse of the matrix, typically only a subset (e.g., diagonal entries),
    // without explicitly forming the full dense inverse.
    //
    // Note:
    // - This is faster and more memory-efficient than computing the full inverse.
    // - Useful when only specific elements of the inverse are needed (e.g., variances).
    sTiles_selinv(0, 0, &stile);

    // Step 5.2: Access elements of the selected inverse.
    //
    // Syntax:
    //   sTiles_get_selinv_elm(group_id, matrix_id, row, col, &stile)
    //
    // Here:
    // - (row, col) = position of the requested element in the inverse matrix.
    //
    // In this example, we print the first 5 diagonal elements (i,i) of the inverse.
    for (int i = 0; i < 5; i++) {
        std::cout << "Inverse(" << i << "," << i << ") = " << sTiles_get_selinv_elm(0, 0, i, i, &stile) << "\n";
    }
    
    sTiles_unbind(0, 0, &stile);

    // -------------------------------------------------------------------------
    // 5. Cleanup
    // -------------------------------------------------------------------------

    sTiles_quit();
    free(mat.row_indices);
    free(mat.col_indices);
    free(mat.values);

    std::cout << "\nProcess completed successfully :)\n";
    return EXIT_SUCCESS;
}

bool loadSparseMatrix(const char* filename, SparseMatrix& mat){

    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening the file");
        return false;  // Error code for failure
    }

    double tmp1 = 0.0, tmp2 = 0.0;
    if (fread(&tmp1, sizeof(double), 1, file) != 1) {
        fprintf(stderr, "Error reading tmp1 from file.\n");
        fclose(file);
        return false;
    }
    if (fread(&tmp2, sizeof(double), 1, file) != 1) {
        fprintf(stderr, "Error reading tmp2 from file.\n");
        fclose(file);
        return false;
    }
    mat.N = (int)tmp1;
    mat.NNZ = (int)tmp2;

    // If *N is zero, rewind and read as int
    if (mat.N == 0) {
        rewind(file);  // Move file pointer back to the beginning

        int tmp1_int = 0, tmp2_int = 0;
        if (fread(&tmp1_int, sizeof(int), 1, file) != 1 || fread(&tmp2_int, sizeof(int), 1, file) != 1) {
            fprintf(stderr, "Error reading tmp1_int from file.\n");
            fclose(file);
            return false;
        }
        mat.N = tmp1_int;
        mat.NNZ = tmp2_int;
    }

    int *csr_p = (int *)malloc(((mat.N) + 1) * sizeof(int));
    if (!csr_p) {
        fprintf(stderr, "Memory allocation failed for csr_p.\n");
        fclose(file);
        return false;
    }
    mat.row_indices = (int *)malloc(mat.NNZ * sizeof(int));
    mat.values = (double *)malloc(mat.NNZ * sizeof(double));
    if (!mat.row_indices || !mat.values) {
        fprintf(stderr, "Memory allocation failed for CSR arrays.\n");
        free(csr_p);
        if (mat.row_indices) free(mat.row_indices);
        if (mat.values) free(mat.values);
        fclose(file);
        return false;
    }

    std::cout << mat.NNZ << std::endl;
    
    // Read CSR matrix data
    if (fread(mat.row_indices, sizeof(int), (mat.NNZ), file) != (mat.NNZ)) {
        fprintf(stderr, "Error reading row_indices from file.\n");
        free(csr_p);
        free(mat.row_indices);
        free(mat.values);
        fclose(file);
        return false;
    }
    if (fread(csr_p, sizeof(int), (mat.N) + 1, file) != ((mat.N) + 1)) {
        fprintf(stderr, "Error reading csr_p from file.\n");
        free(csr_p);
        free(mat.row_indices);
        free(mat.values);
        fclose(file);
        return false;
    }
    if (fread(mat.values, sizeof(double), (mat.NNZ), file) != (mat.NNZ)) {
        fprintf(stderr, "Error reading values from file.\n");
        free(csr_p);
        free(mat.row_indices);
        free(mat.values);
        fclose(file);
        return false;
    }

    fclose(file);

    mat.col_indices = (int *)malloc((mat.NNZ) * sizeof(int));
    if (!mat.col_indices) {
        fprintf(stderr, "Memory allocation failed for col_indices.\n");
        free(csr_p);
        free(mat.row_indices);
        free(mat.values);
        return false;
    }

    int csr_index = 0;
    for (int index_j = 1; index_j <= (mat.N); ++index_j) {
        for (int count = 0; count < (csr_p[index_j] - csr_p[index_j - 1]); ++count) {
            (mat.row_indices)[csr_index] -= 1;
            (mat.col_indices)[csr_index] = index_j - 1;
            if ((mat.row_indices)[csr_index] < (mat.col_indices)[csr_index])
                printf("check: (%d, %d): %f \n", (mat.row_indices)[csr_index], (mat.col_indices)[csr_index], (mat.values)[csr_index]);
            csr_index++;
        }
    }

    free(csr_p);  // Free the temporary allocation

    return true;
}
