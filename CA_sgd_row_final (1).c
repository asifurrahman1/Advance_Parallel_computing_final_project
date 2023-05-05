#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"
// #include "data_processor.h"
#include <mpi.h>


void alloc_by_size(int size, double** u){
     *u = (double*) malloc(sizeof(double)*size);
}
void alloc_by_size_int(int size, int** u){
     *u = (int*) malloc(sizeof(int)*size);
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
    srand48(42);
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

void mult_elementwise(int m, int n, double* V, double scaler, double* u){
    for(int i = 0; i < m; ++i){
        for(int j=0;j<n;j++){
            u[i*n+j] = V[i*n+j] *scaler;
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

// void logistic_regression_gd(int m, int n, double learning_rate, double* local_matrix, double* local_weights){
// // void logistic_regression_gd(int m, int n, double learning_rate, double* local_matrix, double* local_weights){
//     // TODO.
//     // GD algorithm steps:
//     int niters = 0;
//     double *u = (double*) malloc(sizeof(double)*m);
//     double *gradient = (double*) malloc(sizeof(double)*n);
//     double objval = 0.;
//     // Compute gradient.
//     matrix_vector_product(m, n, 0, local_matrix, local_weights, u);
//     sigmoid_function(m, 1., u);
//     matrix_vector_product(m, n, 1, local_matrix, u, gradient);
//     vector_addition(n, learning_rate, gradient, local_weights); // x  = x - eta*grad
//     free(u);
//     free(gradient);
// }

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

void get_s_step_batch_data(int blocksize, int n, int batch_size, int s_step, double *local_scaled_matrix, double** s_step_sampled_data){
    gen_matrix(s_step*batch_size, n, s_step_sampled_data);
    int row_count = 0; 
    for(int s=0; s<s_step; s++){
        double* sampled_data;
        sample_data(blocksize, n, batch_size, s, local_scaled_matrix, &sampled_data);
        for(int i=0;i<batch_size;i++){
            for (int j = 0; j < n; j++)
            {
                (*s_step_sampled_data)[row_count*n+j]= sampled_data[i*n+j];
            }
            row_count++;
        }
        free(sampled_data);
    }
}

void add_row_to_matrix(int m, int n, double *matrix, double *vec, double **appended_matrix){
    alloc_by_size((m+1)*n, appended_matrix);
    int i,j;
    for( i=0; i<m; i++){
        for(j=0; j<n; j++){
            (*appended_matrix)[i*n+j] = matrix[i*n+j];
        }
    }
    for(j=0; j<n; j++){
        (*appended_matrix)[i*n+j] = vec[j];
    }
}

void Transpose_mat(int m, int n, double* A, double** transA){
     alloc_by_size(m*n, transA);
     for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
            (*transA)[j*m + i] = A[i*n+j];
        }
    }
}
void matrix_matrix_multiply(int m, int n, int l, double *A, double *B, double **result) {
    int i, j, k;
    alloc_by_size(m*l, result);
    for(i=0; i<m; i++) {
        for(j=0; j<l; j++) {
            double sum = 0;
            for(k=0; k<n; k++) {
                sum += A[i*n + k] * B[k*l + j];
            }
            (*result)[i*l + j] = sum;
        }
    }
}

void get_all_reduce_data(int m, int n, int batch, double* concat_gram_matrix, int comm_size, double** comm_data){
    int i, j, k, pos=0;
    alloc_by_size(comm_size, comm_data);
    int idx = 0;
    for (i = 0; i < m/batch; i++) {
        for (j = i+1; j < m/batch; j++) {
            // printf("[%d][%d]\t", i, j);
            for(int x=0;x<batch;x++){
                for(int y=0;y<batch;y++){
                    // printf("[%d]\t", (batch*i*m)+(j*batch)+(x*m)+y);
                    (*comm_data)[pos]= concat_gram_matrix[(batch*i*n)+(j*batch)+(x*n)+y];
                    // printf("\n[%f]\t", (*comm_data)[pos]);
                    pos++;
                }
                    //  printf("\n");
            }
                // printf("\n"); 
        }
        // printf("\n");
    }
    for(j = 0; j < m; j++){
        // printf("%d\t", (j*n)+(m));
        (*comm_data)[pos]= concat_gram_matrix[(j*n)+(m)];
        // printf("\n[%f]\t", (*comm_data)[pos]);
        pos++;
    }
    printf("\n"); 
}

void get_gram_matrix(int m, int batch, double* comm_vec, int upp_triag_size, double** gram_matrix, double** Ax){
    alloc_by_size(upp_triag_size, gram_matrix);
    alloc_by_size(m, Ax);
    int pos=0, i, j;
    for (i = 0; i < m/batch; i++) {
        for (j = i+1; j < m/batch; j++) {
            // printf("[%d][%d]\n", i, j);
            for(int x=0;x<batch;x++){
                for(int y=0;y<batch;y++){
                    (*gram_matrix)[pos] =  comm_vec[pos];
                    //  printf("[%f]\t", (*gram_matrix)[pos]);
                     pos++;
                    // printf("[%f]\t", (*gram_matrix)[(batch*i*m)+(j*batch)+(x*m)+y]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
    }
    //--------Ax------------
    for(j = 0; j < m; j++){
        // printf("%d\t", (j*n)+(m));
        (*Ax)[j]= comm_vec[pos++];
    }
}

void get_Ak_Ax(int n, int batch, int pos, double* matrix, double* vec, double **Ak_mat, double **Ax_vec){
     alloc_by_size(n*batch, Ak_mat);
     alloc_by_size(batch, Ax_vec);
     int start = pos*n;
     int loc;
     for(int i=0;i<batch;i++){
        loc=0;
        for(int j=start; j<start+n;j++){
            (*Ak_mat)[i*n+loc]= matrix[j];
            loc++;
        }
        start = start+n;
     }
     int start_vec = pos*batch;
     int idx = 0;
     for(int i=start_vec;i<start_vec+batch;i++){
        (*Ax_vec)[idx++] = vec[i];
     }
}


void get_gram_matrix_block(int s, int s_step, int batch, double *upper_tri, double **block){
    alloc_by_size(batch*batch, block);
    int row=0;
    int pos=s*(batch*batch);
    // printf(" POS %d\n", pos);
    for(int x=0;x<batch;x++){
        int col=0;
        for(int y=0;y<batch;y++){
            // (*block)[row*batch+(col++)]= upper_tri[(batch*i*m)+(j*batch)+(x*m)+y];
            (*block)[row*batch+(col++)]= upper_tri[pos++];
        }
        row++;
    }
}

void calc_s_step(int m, int n, int s_step, int batch_size, double lr, int upper_tri_size, double* upper_tri, double* Ax, double* local_weight, double* s_step_sampled_data){
    // double **grad_vec = (double**) malloc((s_step)*sizeof(double*));
    double **vk = (double**) malloc((s_step)*sizeof(double*));
    double **uk = (double**) malloc((s_step)*sizeof(double*));
    double *grad_sum;
    gen_vector(n, &grad_sum, 1);
    double *A_k, *Ax_k, *grad, *block, *block_T;
    int c=0;
    // printf("### GRAM DATA %d ###\n", upper_tri_size);
    // print_vector(upper_tri_size, upper_tri);
    for(int s=0; s<s_step; s++){
        // printf("############ [s = %d] #############\n", s);
        alloc_by_size(batch_size, &grad);
        if(s==0){
            get_Ak_Ax(n, batch_size, s, s_step_sampled_data, Ax, &A_k, &Ax_k);
            uk[s] = Ax_k;
            sigmoid_function(batch_size, 1.0, uk[s]);
            matrix_vector_product(batch_size, n, 1, A_k, uk[s], grad);
            vector_addition(n, 1.0, grad, grad_sum);
            free(Ax_k);
            free(A_k);
        }
        else{
            get_Ak_Ax(n, batch_size, s, s_step_sampled_data, Ax, &A_k, &Ax_k);
            double *temp_vec;
            double *A_T;
            gen_vector(batch_size, &temp_vec, 1);
            double *temp1;
            alloc_by_size(batch_size, &temp1);
            int step = 0;
            for(int k=0; k<s;k++){
                get_gram_matrix_block(c, s_step, batch_size, upper_tri, &block);
                // Transpose_mat(batch_size, batch_size, block, &block_T);
                // matrix_vector_product(batch_size, batch_size, 0, block_T, uk[step++], temp1);
                matrix_vector_product(batch_size, batch_size, 0, block, uk[step++], temp1);
                vector_addition(batch_size, 1.0, temp1, temp_vec);
                c++;
            }
            uk[s] = Ax_k;
            vector_addition(batch_size, lr, temp_vec, uk[s]);  // Ak_X+ eta AAT uk
            sigmoid_function(batch_size, 1.0, uk[s]);
            
            matrix_vector_product(batch_size, n, 1, A_k, uk[s], grad);
            vector_addition(n, 1.0, grad, grad_sum);
            free(temp_vec);
            free(temp1);
            free(Ax_k);
            free(A_k);
        }
    }
    //************ WEIGHT UPDATE ********************
    vector_addition(n, lr, grad_sum, local_weight); 

    free(grad); 
    free(vk);
    free(uk);
    free(grad_sum);
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

int main(int argc, char** argv) {
    srand48(42);
    int rank, size;
    int proc = atoi(argv[1]);
    int maxiters = atoi(argv[2]);
    double lr = atof(argv[3]);
    int batch_size = atoi(argv[4]);
    int s_step = atoi(argv[5]);

    int m, n;
    const char *filename = "output.csv";
    count_rows_and_columns(filename, &m, &n);
    double **results;

    double start_comm=0., end_comm=0.;
    double start_comp=0., end_comp=0.;
    double start_tot=0., end_tot=0.;
    double comm_diff=0., comp_diff=0., tot_diff=0.;
    
    // #############################################
    start_tot = MPI_Wtime();
    double objval = 0.;
    double *matrix, *local_matrix, *local_scaled_matrix, *labels, *local_label,  *local_weights,  *communication_data, *global_avg_data;

    int line = m/proc;
    m = line*proc;
    read_rest_of_table(filename, m, n, &matrix);
    read_first_elements(filename, m, &labels);
   
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double acc;
    if (rank == 0) {
        results = (double **)malloc(maxiters * sizeof(double *));
        for (int i = 0; i < maxiters; i++) {
            results[i] = (double *)malloc(3 * sizeof(double));
        }
    }
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

    int sz = batch_size*s_step;
    int comm_size, upper_tri_size;

    upper_tri_size = (batch_size*batch_size)*s_step;
    comm_size = upper_tri_size+sz;
    alloc_by_size(comm_size, &global_avg_data);
    //*******************************************************
    //                GLOBAL LOOP
    //*******************************************************
    int *triangle_index;
    double *s_step_sampled_data, *appended_matrix, *append_mat_T, *concat_matrix, *gram_matrix, *Ax;
    for(int iter = 0; iter <  maxiters; ++iter){ 
            start_comp = MPI_Wtime();
            if (iter==0){
                gen_vector(n, &local_weights, 1);

                /*
                alloc_by_size(n, &local_weights);
                FILE *file = fopen("x_val.txt", "r");
                if (file == NULL) {
                    printf("Error: Cannot open the file.\n");
                    //return;
                }
                for(int j=0; j<n; j++){
                    fscanf(file, "%lf", &local_weights[j]);
                }
                */
                print_vector(n, local_weights);
            }
            // if(rank==0){
            get_s_step_batch_data(blocksize, n, batch_size, s_step, local_scaled_matrix, &s_step_sampled_data);
            end_comp = MPI_Wtime();
            comp_diff += end_comp - start_comp;
            start_comp = MPI_Wtime();
            add_row_to_matrix(batch_size*s_step, n, s_step_sampled_data, local_weights, &appended_matrix);
            end_comp = MPI_Wtime();
            comp_diff += end_comp - start_comp;
            start_comp = MPI_Wtime();
            // print_matrix((batch_size*s_step)+1, n, appended_matrix);
            Transpose_mat(batch_size*s_step+1, n, appended_matrix, &append_mat_T);
            end_comp = MPI_Wtime();
            comp_diff += end_comp - start_comp;
            start_comp = MPI_Wtime();
            // print_matrix(n,(batch_size*s_step)+1, append_mat_T);
            matrix_matrix_multiply(batch_size*s_step, n, (batch_size*s_step)+1, s_step_sampled_data, append_mat_T, &concat_matrix);
            end_comp = MPI_Wtime();
            comp_diff += end_comp - start_comp;
            // print_matrix(batch_size*s_step,(batch_size*s_step)+1, concat_matrix);
            
            get_all_reduce_data(batch_size*s_step, (batch_size*s_step)+1, batch_size, concat_matrix, comm_size, &communication_data);
            start_comm = MPI_Wtime();
            MPI_Allreduce(MPI_IN_PLACE, communication_data, comm_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            for (int i = 0; i < comm_size; i++) {
                communication_data[i] /= ((double)(1.0*size));
            }
            end_comm = MPI_Wtime();
            comm_diff += end_comm - start_comm;
            get_gram_matrix(s_step*batch_size, batch_size, communication_data, upper_tri_size, &gram_matrix, &Ax);

            start_comp = MPI_Wtime();
            calc_s_step(blocksize, n, s_step, batch_size, lr, upper_tri_size, gram_matrix, Ax, local_weights, s_step_sampled_data);
            end_comp = MPI_Wtime();
            comp_diff += end_comp - start_comp;
            // }
            if(rank==0){
                start_comp = MPI_Wtime();
                objval = objective_function(m, n, matrix, local_weights, labels);
                acc = training_accuracy(m, n, matrix, local_weights, labels);
                end_comp = MPI_Wtime();
                comp_diff += end_comp - start_comp;
                printf("[rank%d iter %d]: objval = %.7f , Acc: %.7f\n",rank, iter, objval, acc);
                if (results[iter] != NULL) {
                    results[iter][0] = (double)iter;
                    results[iter][1] = objval;
                    results[iter][2] = acc;
                }
            }
    free(s_step_sampled_data);
    free(appended_matrix);
    free(append_mat_T);
    free(concat_matrix);
    free(gram_matrix);
    free(Ax);
    }
    //*******************************************************
    //*******************************************************
    if (rank == 0) {
        // Write the results to a CSV file
        FILE *file = fopen("CA_sgd_row_FINALs3_4th.csv", "w");
        FILE *file2 = fopen("x_val.txt", "w");
        if (file == NULL) {
            printf("Error: Cannot open the results file.\n");
            return 1;
        }
        if (file2 == NULL) {
            printf("Error: Cannot open the results file.\n");
            return 1;
        }

        for (int i = 0; i < maxiters; i++) {
            fprintf(file, "%.0f,%.7f,%.7f\n", results[i][0], results[i][1], results[i][2]);
            free(results[i]);
        }
        print_vector(n, local_weights);
        for (int j = 0; j < n; j++) {
            fprintf(file2, "%lf\n", local_weights[j]);
        }


        fclose(file);
        free(results);
    }

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