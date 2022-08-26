#include <iostream>
#include <cstdlib>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <curand.h>
#include<cuda.h>
/*Check for safe return of all calls to the device */
void CUDA_SAFE_CALL(cudaError_t call)
{
        cudaError_t ret = call;
        //printf("RETURN FROM THE CUDA CALL:%d\t:",ret);                                        
        switch(ret)
        {
                case cudaSuccess:
                //              printf("Success\n");                    
                                break;
        /*      case cudaErrorInvalidValue:                             
                                {
                                printf("ERROR: InvalidValue:%i.\n",__LINE__);
                                exit(-1);
                                break;  
                                }                       
                case cudaErrorInvalidDevicePointer:                     
                                {
                                printf("ERROR:Invalid Device pointeri:%i.\n",__LINE__);
                                exit(-1);
                                break;
                                }                       
                case cudaErrorInvalidMemcpyDirection:                   
                                {
                                printf("ERROR:Invalid memcpy direction:%i.\n",__LINE__);        
                                exit(-1);
                                break;
                                }                       */
                default:
                        {
                                printf(" ERROR at line :%i.%d' ' %s\n",__LINE__,ret,cudaGetErrorString(ret));
                                exit(-1);
                                break;
                        }
        }
}
// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
 
    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
 
    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

double calculate_gflops(float &Tsec, int m, int k, int n, int nIter)
{
    float gflops=(1.0e-9 * (( 1.0 * m*k*n )*nIter/Tsec));
	return gflops;
}

/*prints the result in screen*/
void print_on_screen(char * program_name,float tsec,double gflops,int row, int col, int flag)//flag=1 if gflops has been calculated else flag =0
{
    printf("\n---------------%s----------------\n",program_name);
    printf("\tSIZE\t TIME_SEC\t Gops\n");
    if(flag==1)
        printf("\t%d,%d\t%f\t%lf\t",row, col,tsec,gflops);
    else
        printf("\t%d,%d\t%lf\t%lf\t",row, col,"---","---");
}

// Multiply the arrays A and B on GPU and save the result in C
 // C(m,n) = A(m,k) * B(k,n)
    void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
 
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
 
    // Do the actual multiplication
    cudaEvent_t start,stop;
    CUDA_SAFE_CALL(cudaEventCreate (&start));
    CUDA_SAFE_CALL(cudaEventCreate (&stop));
    float elapsedTime;
    CUDA_SAFE_CALL(cudaEventRecord (start, 0));

    int nIter = 300;
    for(int i=0; i<nIter; i++)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    CUDA_SAFE_CALL(cudaEventRecord (stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize (stop));
    
    // Destroy the handle
    cublasDestroy(handle);
    CUDA_SAFE_CALL(cudaEventElapsedTime ( &elapsedTime, start, stop));
    printf("elapsed time = %f\n", elapsedTime);
    float Tsec= 1.0e-3*elapsedTime;	
	//printing the result on screen  
    print_on_screen("GEMM",Tsec/(float)nIter,calculate_gflops(Tsec, m,k,n, nIter),m, k,1);
}
//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
 void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {
 
     for(int i = 0; i < nr_rows_A; ++i){
         for(int j = 0; j < nr_cols_A; ++j){
             std::cout << A[j * nr_rows_A + i] << " ";
         }
         std::cout << std::endl;
     }
     std::cout << std::endl;
 }
int main(int argc, char* argv[]) {
     // Allocate 3 arrays on CPU
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
  
      // for simplicity we are going to use square arrays
    nr_rows_A = atoi(argv[1]);
    nr_cols_A = atoi(argv[2]);
    nr_rows_B = nr_cols_A;
    nr_cols_B = atoi(argv[3]);
    nr_rows_C = nr_rows_A;
    nr_cols_C = nr_cols_B;
  
    float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
    float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
    float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));
 
     // Allocate 3 arrays on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
    cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
    cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));
 
     // Fill the arrays A and B on GPU with random numbers
    GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
    GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);
 
     // Optionally we can copy the data back on CPU and print the arrays
    cudaMemcpy(h_A,d_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B,d_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyDeviceToHost);
    // std::cout << "A =" << std::endl;
    // print_matrix(h_A, nr_rows_A, nr_cols_A);
    // std::cout << "B =" << std::endl;
    // print_matrix(h_B, nr_rows_B, nr_cols_B);
 
    // Multiply A and B on GPU
    gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
 
     // Copy (and print) the result on host memory
    cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
    // std::cout << "C =" << std::endl;
    // print_matrix(h_C, nr_rows_C, nr_cols_C);
 
     //Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
 
     // Free CPU memory
    free(h_A);
    free(h_B);
    free(h_C);
 
    return 0;
}