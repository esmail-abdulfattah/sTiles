
/*
 * This C++ file is a testing program for the sTiles library, specifically for
 * performing Cholesky factorization and computing the selected inverse of a sparse matrix.
 *
 * The program:
 * 1. Reads a sparse matrix stored in CSR format from a binary file.
 * 2. Initializes the STiles library and its data structures.
 * 3. Assigns the matrix data to the STiles object.
 * 4. Performs preprocessing of the matrix.
 * 5. Executes Cholesky factorization using STiles.
 * 6. Computes the selected inverse of the factorized matrix.
 * 7. Outputs an element from the computed inverse for verification.
 *
 * This code is designed for testing and verifying the functionality of the STiles library
 * and serves as a reference for using its API for Cholesky factorization.
 *
 * Author: Esmail Abdul Fattah
 * Date: 2-1-2025
 */
 


#include "./include/stiles.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <iomanip>
#include <omp.h>

// Forward declarations
struct SparseMatrix {
    int     N = 0;
    int     NNZ = 0;
    int*    row_indices = nullptr;
    int*    col_indices = nullptr;
    double* values = nullptr;
};

bool    loadSparseMatrix(const char* filename, SparseMatrix& mat);
double* read_array(const std::string& filename, int size);
void    read_indices(const std::string& filename, std::vector<std::pair<int, int>>& indices);
void    print_selected_indices_call(int group_index, 
                                   int call_index, 
                                   const std::vector<std::pair<int, int>>& indices, 
                                   sTiles_object* stile, 
                                   STILES_CONFIGURATION** schemes);


int main(int /*argc*/, char** /*argv*/)
{

    #ifdef STILES_GPU
        printf("GPU support is enabled in sTiles!\n");
    #else
        printf("GPU support is NOT enabled (CPU-only build).\n");
    #endif

    omp_set_nested(1); // Enable nested parallelism
    omp_set_max_active_levels(2); // Allow multiple levels of parallelism

    // -------------------------------------------------------------------------
    // 1. Group/call configurations
    // -------------------------------------------------------------------------
    // 
    int num_groups            =  1;
    int calls_per_group[]     = {1};
    int cores_per_group[]     = {1};
    int chol_type_per_group[] = {0}; // 0 => sparse, 1 => dense, etc., depending on your definition
    bool get_inverse[]        = {true, true};

    // -------------------------------------------------------------------------
    // 2. Load sparse matrices (9 in total)
    // -------------------------------------------------------------------------
    std::vector<std::vector<SparseMatrix>> mat(num_groups);
    // Resize vector to fit each group's calls
    for (int i = 0; i < num_groups; i++) {
        mat[i].resize(calls_per_group[i]);
    }

    // Load matrices dynamically
    const char* filenames[1][1] = {
        {"./mat12.bin"}  // No third call in group 1
    };

    if (!loadSparseMatrix(filenames[0][0], mat[0][0])) {
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------------------
    // 3. Initialize the sTiles object and configuration
    // -------------------------------------------------------------------------
    sTiles_object*         stile   = nullptr;
    sTiles_create(&stile, num_groups, calls_per_group, cores_per_group, chol_type_per_group, get_inverse);
                    
    // -------------------------------------------------------------------------
    // 4. Assign matrix structure dynamically
    // -------------------------------------------------------------------------
    sTiles_assign_graph(0, stile, mat[0][0].N, mat[0][0].NNZ, mat[0][0].row_indices, mat[0][0].col_indices);


    // -------------------------------------------------------------------------
    // 5. Preprocess each matrix call
    // -------------------------------------------------------------------------
    sTiles_init(&stile, 0, NULL);

    // -------------------------------------------------------------------------
    // 6. Update numerical values in each call before factorization
    // -------------------------------------------------------------------------
    sTiles_assign_values(0, 0, &stile, mat[0][0].values);

    // Print all results after computations
    std::printf("\n");
    std::cout << std::setw(10) << "Number"
              << std::setw(10) << "Group"
              << std::setw(10) << "Call"
              << std::setw(20) << "Time Spent (s)"
              << std::setw(13) << "Logdet"
              << "\n----------------------------------------------------------------------\n";

    int i =0;
    #pragma omp parallel for
    for (int j = 0; j < 1; j++) {

        sTiles_bind(i, j, &stile);

        int status = sTiles_chol(i, j, &stile);
        if(status!=0){
            printf("error in  sTiles_chol %d \n", status);
            sTiles_quit();
        }
        std::cout << "Group: " << i << ", Call: " << j << ": " << std::setw(20) << std::fixed << std::setprecision(6) << sTiles_get_logdet(i, j, &stile) << std::endl;
        sTiles_selinv(i, j, &stile);
        
        sTiles_unbind(i, j, &stile);

    }
    


    // -------------------------------------------------------------------------
    // 12. Cleanup
    // -------------------------------------------------------------------------

    sTiles_quit();

    // After your computations and before returning from main():
    // Free matrix memory for the first group:
    for (int i = 0; i < num_groups; i++) {
        for (int j = 0; j < calls_per_group[i]; j++) {
            free(mat[i][j].row_indices);
            free(mat[i][j].col_indices);
            free(mat[i][j].values);
        }
    }


    std::cout << "\nProcess completed successfully :)\n";
    return EXIT_SUCCESS;
}

/***************************************************************************
 *                          Function Definitions                            *
 ***************************************************************************/

/**
 * @brief Reads a sparse matrix (CSR) from a binary file.
 */
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

/**
 * @brief Example implementation for reading an array from a text file.
 *        Adjust to match your file format and error handling.
 */
double* read_array(const std::string& filename, int size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return nullptr;
    }

    double* array = new double[size];
    for (int i = 0; i < size; ++i) {
        if (!(file >> array[i])) {
            std::cerr << "Error: Insufficient data in file " << filename << std::endl;
            delete[] array; // Clean up if error occurs
            return nullptr;
        }
    }

    file.close();
    return array;
}

/**
 * @brief Reads a list of (row, col) indices from a CSV file.
 *        Adjust to match your file format.
 */

void read_indices(const std::string& filename, std::vector<std::pair<int, int>>& indices) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {

        std::istringstream iss(line);
        std::string row_str, col_str;

        // Split line by comma
        if (std::getline(iss, row_str, ',') && std::getline(iss, col_str, ',')) {
            int row = std::stoi(row_str);
            int col = std::stoi(col_str);
            indices.emplace_back(row, col);
        } else {
            std::cerr << "Warning: Skipping invalid line: " << line << std::endl;
        }
    }

    if (indices.empty()) {
        std::cerr << "Error: No indices found in the file." << std::endl;
    } else {
        std::cout << "\n" << std::endl;
        std::cout << "Successfully read " << indices.size() << " indices." << std::endl;
    }

    file.close();
}

// Function to print results for selected indices
void print_selected_indices_call(int group_index, int call_index, const std::vector<std::pair<int, int>>& indices, sTiles_object* stile, STILES_CONFIGURATION** schemes) {

    // Set precision for floating-point values
    std::cout << std::fixed << std::setprecision(10);

    // Print the header
    std::cout << std::left << std::setw(15) << "Index (i, j)"
              << " | " << std::setw(20) << "L[i, j]"
              << " | " << std::setw(20) << "INV[i, j]" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;

    // Iterate over indices and print values
    for (const auto& [i, j] : indices) {
        double L_val = STILES_GET_CHOL_VALUE_CALL(group_index, call_index, i-1, j-1, schemes); // Replace with actual STiles API to fetch L[i, j]
        double INV_val = STILES_GET_SELECTED_INVERSE_ELEMENTWISE_CALL(group_index, call_index, i-1, j-1, schemes); // Fetch INV[i, j]

        std::cout << std::left << std::setw(15) << "(" + std::to_string(i) + ", " + std::to_string(j) + ")"
                  << " | " << std::setw(20) << L_val
                  << " | " << std::setw(20) << INV_val << std::endl;
    }
}

