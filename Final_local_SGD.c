#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"
// #include "data_processor.h"
#include <mpi.h>


void alloc_by_size(int size, double** u){
     *u = (double*) malloc(sizeof(double)*size);
}

//*********************************************
void count_rows_and_columns(const char *filename, int *numRows, int *numCols) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Cannot open the file.\n");
        return;
    }
    char line[256];
    *numRows = 0;
    *numCols = 0;

    while (fgets(line, sizeof(line), file)) {
        if (*numCols == 0) {
            for (int i = 0; i < strlen(line); i++) {
                if (line[i] == ',') {
                    (*numCols)++;
                }
            }
            (*numCols)++;
        }
        (*numRows)++;
    }
    fclose(file);
}

void read_first_elements(const char *filename, int numRows, double **label) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Cannot open the file.\n");
        // return NULL;
    }
    alloc_by_size(numRows, label);
    // double *firstElements = (double *)malloc(numRows * sizeof(double));
    double v;
    for (int i = 0; i < numRows; i++) {
        // fscanf(file, "%lf,", &(*label)[i]);
        fscanf(file, "%lf,", &v);
        if(v==0){
            (*label)[i]=-1;
        }
        else{
            (*label)[i]=1;
        }
        // Skip the rest of the line
        char c;
        while ((c = fgetc(file)) != '\n' && c != EOF);
    }

    fclose(file);

    // return firstElements;
}


void read_rest_of_table(const char *filename, int numRows, int numCols, double** matrix) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Cannot open the file.\n");
        // return NULL;
    }
    alloc_by_size(numRows*(numCols-1), matrix);
    
    int dummy;
    double temp;
    for (int i = 0; i < numRows; i++) {
        fscanf(file, "%d", &dummy); // Read and discard the first element
        for (int j = 0; j < numCols - 1; j++) {
            fscanf(file, ",%lf", &temp);
            // printf("%f\n", temp);
            (*matrix)[i*(numCols - 1)+j] = temp;
        }
    }

    fclose(file);
}
//*********************************************

void gen_matrix(int m, int n, double** matrix){
    alloc_by_size(m*n, matrix);
    
    for(int i = 0; i < m*n; ++i){
        (*matrix)[i] = drand48();
    }
    // printf("\nfinished generating matrix...\n");
}



void gen_vector(int m, double** vector, int init_zero){
    alloc_by_size(m, vector);
    if(init_zero==0){
        srand48(42);
        for(int i = 0; i < m; ++i){
            (*vector)[i] = drand48();
        }
    }
    else{
        for(int i = 0; i < m; ++i){
            (*vector)[i] = 0;
        }
    }
}

void print_matrix(int m, int n, double* matrix){
    for(int i = 0; i < m; ++i){
        for(int j = 0; j <n ; ++j){
            printf("%.3f ", matrix[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_vector(int m, double* vector){
    for(int i = 0; i < m; ++i){
        printf("%.3f \n", vector[i]);
    }
}

void generate_labels(int m, double** labels){
    alloc_by_size(m, labels);
    srand48(42);
    for(int i = 0; i < m; ++i){
        (*labels)[i] = (drand48() < 0.5) ? (-1.0) : (1.0);
        // printf("%f", (*labels)[i]);
    }
    // printf("\nfinished generating labels...\n");
}

// computes the vector operation: y = y + alpha*x
void vector_addition(int n, double alpha, double* x, double *y){
    // assume that input args are sane.
    for(int i = 0; i < n; ++i){
        y[i] += alpha*x[i];
    }
}

// A is assumed to be square, n by n and stored in 1D-format: row-major.
void matrix_vector_product(int m, int n, int transpose_A, double* A, double* u, double *v){
    // assume that input args are sane.
    if(transpose_A >= 1){
        // parallelize outer loop with omp.
        for(int i = 0; i < n; ++i){
            v[i] = 0.;
            for(int j = 0; j < m; ++j){
                v[i] += A[j*n + i]*u[j];
            }
        }
    }
    else{
        // parallelize outer loop with omp.
        for(int i = 0; i < m; ++i){
            v[i] = 0.;
            for(int j = 0; j < n; ++j){
                v[i] += A[i*n + j]*u[j];
            }
        }
    }
}


void sigmoid_function(int m, double alpha, double* u){
    // apply the nonlinear sigmoid function using exp() from math.h
    // alpha is used to compute sigmoid(alpha*u). Different alpha's required for training_accuracy and during gradient descent.
    for(int i = 0; i < m; ++i){
        u[i] = 1./ (1. + exp(alpha*u[i]));
    }
}

double training_accuracy(int m, int n, double*A , double* weights, double* labels){
    double* u = (double*) malloc(sizeof(double)*m);
    matrix_vector_product(m, n, 0, A, weights, u);
    sigmoid_function(m, -1.0, u);
    double acc=0;
    for(int i=0; i< m; i++){
        if((u[i]>=0.5 && labels[i]==1)||(u[i]<0.5 && labels[i]==-1)){
            acc++;
        } 
    }
    double total = (double)m*1.0;
    double acc_op = (acc/total)*100;
    // printf("TP: %f\t total:%f \t ACC:%f\n", acc, total, acc_op);
    // printf("\nAcc_op: %.2f\n", acc_op);
    return acc_op;
}

void scale_A_by_labels(int m, int n, double* A, double* labels, double** scaled_A){
    // logistic regression multiplies A by labels, might as well do it all at once up front.
    *scaled_A = (double*) malloc(sizeof(double)*m*n);
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            (*scaled_A)[i*n + j] = A[i*n + j] *labels[i];
        }
    }
}

double objective_function(int m, int n, double* A, double* weights, double* labels){
    double objval = 0.;
    double *scaled;
    double* u = (double*) malloc(sizeof(double)*m);
    // matrix_vector_product(m, n, 0, A, weights, u);
    scale_A_by_labels(m, n, A, labels, &scaled);
    matrix_vector_product(m, n, 0, scaled, weights, u);
    // parallel using omp with reduction on objval, later.
    for(int i = 0; i < m; ++i){
        objval += log(1. + exp(-u[i]));
    }
    free(u);
    free(scaled);
    return objval;
}

void logistic_regression_gd(int m, int n, int local_iter, double learning_rate, double* local_matrix, double* local_weights){
    int niters = 0;
    for(int it=0;it<local_iter;it++){
        // double *u = (double*) malloc(sizeof(double)*m);
        // double *gradient = (double*) malloc(sizeof(double)*n);
        double *u, *gradient;
        alloc_by_size(m, &u);
        alloc_by_size(n, &gradient);
        matrix_vector_product(m, n, 0, local_matrix, local_weights, u);
        sigmoid_function(m, 1., u);
        matrix_vector_product(m, n, 1, local_matrix, u, gradient);
        vector_addition(n, learning_rate, gradient, local_weights); // x  = x - eta*grad
        free(u);
        free(gradient);
    }
}

void sample_data(int m, int n, int batch, int iter, double* data_matrix, double** sampled_data){
    alloc_by_size(batch*n, sampled_data);
    int b_num = (int)(m/batch);
    int pos = iter%b_num;
    int i_start = ((pos)*batch);
    int i,j;
    int row_count=0;
    for(i=i_start;i<(i_start+batch);i++){
        for(j=0; j<n; j++){
            (*sampled_data)[row_count*n+j]=data_matrix[i*n+j];
        }
        row_count++;
    }
}

int main(int argc, char** argv) {
    srand48(42);
    int rank, size;
   
    int maxiters = atoi(argv[1]);
    int local_iter = atoi(argv[2]);
    double lr = atof(argv[3]);
    int batch_size = atoi(argv[4]);
    int proc = atoi(argv[5]);

    int m, n;
    const char *filename = "output.csv";
    count_rows_and_columns(filename, &m, &n);
    double **results;
    double start_comm=0., end_comm=0.;
    double start_comp=0., end_comp=0.;
    double start_tot=0., end_tot=0.;
    double comm_diff=0., comp_diff=0., tot_diff=0.;

    start_tot = MPI_Wtime();
    double objval = 0.;
    double *matrix, *local_matrix, *local_scaled_matrix, *labels, *local_label,  *local_weights;
    
    int line = m/proc;
    m = line*proc;
    read_rest_of_table(filename, m, n, &matrix);
    read_first_elements(filename, m, &labels);
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (m % size != 0) {
        if (rank == 0) {
            printf("Error: number of processes must divide matrix size evenly.\n");
        }
        MPI_Finalize();
        return 1;
    }
    const int blocksize = m / size;
    alloc_by_size(blocksize*n, &local_matrix);
    alloc_by_size(blocksize, &local_label);
    
    start_comm = MPI_Wtime();
    MPI_Scatter(matrix, blocksize*n, MPI_DOUBLE, local_matrix, blocksize*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(labels, blocksize, MPI_DOUBLE, local_label, blocksize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    end_comm = MPI_Wtime();
    comm_diff += end_comm - start_comm;
    
    start_comp = MPI_Wtime();
    scale_A_by_labels(blocksize , n, local_matrix, local_label, &local_scaled_matrix);
    end_comp = MPI_Wtime();
    comp_diff += end_comp - start_comp;
    
    if (rank == 0) {
        results = (double **)malloc(maxiters * sizeof(double *));
        for (int i = 0; i < maxiters; i++) {
            results[i] = (double *)malloc(3 * sizeof(double));
        }
    }
    //*******************************************************
    //                GLOBAL LOOP
    //*******************************************************
    double *sampled_data;
    double acc;
    for(int iter = 0; iter < maxiters; ++iter){ 
            //***********************************************
            //             LOCAL COMPUTATIONS
            // //***********************************************
            start_comp = MPI_Wtime();
            sample_data(blocksize, n, batch_size, iter, local_scaled_matrix, &sampled_data);
            if (iter==0){
                gen_vector(n, &local_weights, 1);
            }
            end_comp = MPI_Wtime();
            comp_diff += end_comp - start_comp;
            start_comp = MPI_Wtime();
            logistic_regression_gd(batch_size, n, local_iter, lr, sampled_data, local_weights);
            end_comp = MPI_Wtime();
            comp_diff += end_comp - start_comp;
            
            start_comm = MPI_Wtime();
            MPI_Allreduce(MPI_IN_PLACE, local_weights, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            end_comm = MPI_Wtime();
            
            comm_diff += end_comm - start_comm;
            //***********************************************
            start_comp = MPI_Wtime();
            for (int i = 0; i < n; i++) {
                local_weights[i] /= ((double)(1.0*size));
            }
            end_comp = MPI_Wtime();
            comp_diff += end_comp - start_comp;
            //***********************************************
            if (rank==0){
                start_comp = MPI_Wtime();
                 objval = objective_function(m, n, matrix, local_weights, labels);
                  end_comp = MPI_Wtime();
                comp_diff += end_comp - start_comp;
                 acc = training_accuracy(m, n, matrix, local_weights, labels);
                 printf("[iter %d]: objval = %.7f , Acc: %.7f\n", iter, objval, acc);
                 if (results[iter] != NULL) {
                    results[iter][0] = (double)iter;
                    results[iter][1] = objval;
                    results[iter][2] = acc;
                }
            }
            //***********************************************
            free(sampled_data);
    }
    //*******************************************************
    // SAVE RESULT
    if (rank == 0) {
        // Write the results to a CSV file
        FILE *file = fopen("Local_SGD_FinalMay4_final.csv", "w");
        if (file == NULL) {
            printf("Error: Cannot open the results file.\n");
            return 1;
        }

        for (int i = 0; i < maxiters; i++) {
            fprintf(file, "%.0f,%.7f,%.7f\n", results[i][0], results[i][1], results[i][2]);
            free(results[i]);
        }
        fclose(file);
        free(results);
    }
    //*******************************************************
    free(matrix);
    free(local_matrix);
    free(local_scaled_matrix);
    free(labels);
    free(local_label);
    free(local_weights);
    MPI_Finalize();
    end_tot = MPI_Wtime();
    tot_diff = end_tot - start_tot;
    if(rank==0){
        printf("Comm cost: %f\n", comm_diff);
        printf("Comp cost: %f\n", comp_diff);
        printf("Tot Time: %f\n", tot_diff);
    }
    return 0;
}