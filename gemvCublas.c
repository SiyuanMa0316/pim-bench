//Example 2. Application Using C and cuBLAS: 0-based indexing
//-----------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define M 6
#define N 5
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

static __inline__ void gemvCublas (cublasHandle_t handle, const float*A, const float*x, float* y, int M, int N){
    float alpha = 1;
    float beta = 0;

    cublasSgemv( handle, CUBLAS_OP_N,
                           M, N,
                           &alpha,
                           A, M,
                           x, 1,
                           &beta,
                           y, 1);
}

int main (void){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    float* devMat;
    float* mat = 0;
    mat = (float *)malloc (M * N * sizeof (*mat));
    if (!mat) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            mat[IDX2C(i,j,M)] = (float)(i * N + j + 1);
        }
    }

    float* devX;
    float* x=0;
    x = (float *)malloc(N * sizeof(*x));
    if(!x){
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    for(int i=0; i<N; i++){
        x[i]=(float)(i+1);
    }

    float* devY;
    float* y=0;
    y = (float *)malloc(N * sizeof(*y));
    
    cudaStat = cudaMalloc ((void**)&devMat, M*N*sizeof(*mat));
    if (cudaStat != cudaSuccess) {
        printf ("matrix device memory allocation failed");
        return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc ((void**)&devX, N*sizeof(*x));
    if (cudaStat != cudaSuccess) {
        printf ("x device memory allocation failed");
        return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc ((void**)&devY, N*sizeof(*y));
    if (cudaStat != cudaSuccess) {
        printf ("y device memory allocation failed");
        return EXIT_FAILURE;
    }

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix (M, N, sizeof(*mat), mat, M, devMat, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("matrix data download failed");
        cudaFree (devMat);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    stat = cublasSetVector(N, sizeof(*x), x, 1, devX, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("vector data download failed");
        cudaFree (devX);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    gemvCublas(handle, devMat, devX, devY, M, N);
    stat = cublasGetVector (N, sizeof(*y), devY, 1, y, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devY);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    cudaFree (devY);
    cudaFree (devX);
    cudaFree (devMat);
    cublasDestroy(handle);
    for (i = 0; i < N; i++) {
        printf ("%7.0f", y[i]);
    }
    printf ("\n");
    free(y);
    free(x);
    free(mat);
    return EXIT_SUCCESS;
}