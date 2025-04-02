/*
 * This C++ file is a testing program for the sTiles library.
 * Author: Esmail Abdul Fattah
 * Date: 2-4-2025
 */

 #include "stiles.h"
 #include <iostream>
 #include <cstdio>
 #include <cstdlib>
 #include <vector>
 #include <omp.h>
 
 struct SparseMatrix {
     int     N = 0;
     int     NNZ = 0;
     int*    row_indices = nullptr;
     int*    col_indices = nullptr;
     double* values = nullptr;
 };
 
 int main() {

    // -------------------------------------------------------------------------
    // 1. Group and call configuration
    // -------------------------------------------------------------------------
    int num_groups             = 1;
    int calls_per_group[]      = {1};
    int cores_per_group[]      = {6};
    int chol_type_per_group[]  = {0};    // 0 = sparse chol
    bool get_inverse[]         = {false};

    // -------------------------------------------------------------------------
    // 2. Simple sparse matrix (5x5 lower triangle)
    // -------------------------------------------------------------------------
    SparseMatrix mat;
    mat.N = 5;
    mat.NNZ = 9;

    mat.row_indices = (int*)malloc(mat.NNZ * sizeof(int));
    mat.col_indices = (int*)malloc(mat.NNZ * sizeof(int));
    mat.values      = (double*)malloc(mat.NNZ * sizeof(double));

    int row_vals[9] =   {0, 1, 1, 2, 2, 3, 3, 4, 4};
    int col_vals[9] =   {0, 0, 1, 1, 2, 2, 3, 3, 4};
    double val[9]   =   {5, -1, 7, -2, 9, -3, 11, -4, 13};

    for (int i = 0; i < mat.NNZ; i++) {
        mat.row_indices[i] = row_vals[i];
        mat.col_indices[i] = col_vals[i];
        mat.values[i]      = val[i];
    }

    /*std::cout << "Matrix Entries:\n";
    for (int i = 0; i < mat.NNZ; i++) {
        std::cout << mat.row_indices[i] << " - "
                << mat.col_indices[i] << " - "
                << mat.values[i] << std::endl;
    }*/

    // -------------------------------------------------------------------------
    // 3. Initialize the sTiles object
    // -------------------------------------------------------------------------
    sTiles_object* stile = nullptr;
    sTiles_create(&stile, num_groups, calls_per_group, cores_per_group, chol_type_per_group, get_inverse);

    // -------------------------------------------------------------------------
    // 4. Assign graph and values
    // -------------------------------------------------------------------------
    sTiles_assign_graph(0, stile, mat.N, mat.NNZ, mat.row_indices, mat.col_indices);
    sTiles_init(&stile, 0, nullptr);
    sTiles_assign_values(0, 0, &stile, mat.values);

    sTiles_bind(0, 0, &stile);
    int status = sTiles_chol(0, 0, &stile);
    if (status != 0) {
        printf("Error in sTiles_chol (%d, %d): status = %d\n", 0, 0, status);
        sTiles_quit();
        return EXIT_FAILURE;
    }

    double logdet = sTiles_get_logdet(0, 0, &stile);
    std::cout << "LogDet = " << logdet << "\n";

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
 