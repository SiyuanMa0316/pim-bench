/*****************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example     :  cuda-matrix-vector-multiplication.cu 
 
  Objective   : Write CUDA program to compute Matrix-Vector multiplication.

  Input       : None 

  Output      : Execution time in seconds , Gflops achieved
                                                                                                                            
  Created     : August-2013

  E-mail      : hpcfte@cdac.in     
                                 
****************************************************************************/


#include<stdio.h>
#include<cuda.h>

#define BLOCKSIZE 256
#define SIZE 32
#define EPS 1.0e-15

cudaDeviceProp deviceProp;	


double *host_Mat,*host_Vect,*host_ResVect,*cpu_ResVect;
double *device_Mat,*device_Vect,*device_ResVect;
int     vlength ,matRowSize , matColSize;
int     device_Count;
int     size = SIZE;

/*mem error*/
void mem_error(char *arrayname, char *benchmark, int len, char *type)
{
        printf("\nMemory not sufficient to allocate for array %s\n\tBenchmark : %s  \n\tMemory requested = %d number of %s elements\n",arrayname, benchmark, len, type);
        exit(-1);
}

/*calculate Gflops*/
double calculate_gflops(float &Tsec, int row, int col, int nIter)
{
        float gflops=(1.0e-9 * (( 1.0 * row*col )*nIter/Tsec));
	return gflops;
}

/*sequential function for mat vect multiplication*/
void CPU_MatVect()
{
	cpu_ResVect = (double *)malloc(matRowSize*sizeof(double));
	if(cpu_ResVect==NULL)
                mem_error("cpu_ResVect","vectmatmul",size,"double");

	int i,j;
	for(i=0;i<matRowSize;i++)
	{cpu_ResVect[i]=0;
	for(j=0;j<matColSize;j++)
	cpu_ResVect[i]+=host_Mat[i*vlength+j]*host_Vect[j];
	}
}

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
 

/*free memory*/
void dfree(double * arr[],int len)
{
        for(int i=0;i<len;i++)
                CUDA_SAFE_CALL(cudaFree(arr[i]));
        printf("mem freed\n");
}

/* function to calculate relative error*/
void relError(double* dRes,double* hRes,int size)
{
        double relativeError=0.0,errorNorm=0.0;
        int flag=0;
        int i;
        

        for( i = 0; i < size; ++i) {
                if (fabs(hRes[i]) > fabs(dRes[i]))
                        relativeError = fabs((hRes[i] - dRes[i]) / hRes[i]);
                else
                        relativeError = fabs((dRes[i] - hRes[i]) / dRes[i]);

                if (relativeError > EPS && relativeError != 0.0e+00 )
                {
                        if(errorNorm < relativeError)
                        {
                                errorNorm = relativeError;
                                flag=1;
                        }
                }

        }
        if( flag == 1)
        {
                printf(" \n Results verfication : Failed");
                printf(" \n Considered machine precision : %e", EPS);
                printf(" \n Relative Error                  : %e\n", errorNorm);

        }
        else
                printf("\n Results verfication : Success\n");

}


/*prints the result in screen*/
void print_on_screen(char * program_name,float tsec,double gflops,int row, int col, int flag)//flag=1 if gflops has been calculated else flag =0
{
        printf("\n---------------%s----------------\n",program_name);
        printf("\tSIZE\t TIME_SEC\t Gflops\n");
        if(flag==1)
        printf("\t%d,%d\t%f\t%lf\t",row, col,tsec,gflops);
        else
        printf("\t%d,%d\t%lf\t%lf\t",row, col,"---","---");

}

/*funtion to check blocks per grid and threads per block*/
void check_block_grid_dim(cudaDeviceProp devProp,dim3 blockDim,dim3 gridDim)
{

        if( blockDim.x >= devProp.maxThreadsDim[0] || blockDim.y >= devProp.maxThreadsDim[1] || blockDim.z >= devProp.maxThreadsDim[2] )
        {
                printf("\nBlock Dimensions exceed the maximum limits:%d * %d * %d \n",devProp.maxThreadsDim[0],devProp.maxThreadsDim[1],devProp.maxThreadsDim[2]);
               exit(-1);
        }

        if( gridDim.x >= devProp.maxGridSize[0] || gridDim.y >= devProp.maxGridSize[1] || gridDim.z >= devProp.maxGridSize[2] )
        {
                printf("\nGrid Dimensions exceed the maximum limits:%d * %d * %d \n",devProp.maxGridSize[0],devProp.maxGridSize[1],devProp.maxGridSize[2]);
               exit(-1);
        }
}


/*Get the number of GPU devices present on the host */
int get_DeviceCount()
{
        int count;
        cudaGetDeviceCount(&count);
        return count;
}


/*Fill in the vector with double precision values */
void fill_dp_vector(double* vec,int size)
{
        int ind;
        for(ind=0;ind<size;ind++)
                vec[ind]=drand48();
}


/////////////////////////////////////////////////////////////////////////////////////////
//
// MatVect : this kernel will perform actual MatrixVector Multiplication 
//
/////////////////////////////////////////////////////////////////////////////////////////
__global__ void MatVectMultiplication(double *device_Mat, double *device_Vect,int matRowSize, int vlength,double *device_ResVect)
  {
        //int tidx = blockIdx.x*blockDim.x + threadIdx.x;
        //int tidy = blockIdx.y*blockDim.y + threadIdx.y;
        //int tindex=tidx+gridDim.x*BLOCKSIZE*tidy;
        int tindex = blockIdx.x*blockDim.x + threadIdx.x;


        //if(tindex<matRowSize)
	    //{
                int i;int m=tindex*vlength;
	        device_ResVect[tindex]=0;
	        for(i=0;i<vlength;i++)
	             device_ResVect[tindex]+=device_Mat[m+i]*device_Vect[i];
	    //}

     __syncthreads();

  }//end of MatVect device function



/*function to launch kernel*/
void launch_Kernel_MatVectMul()
{
/*          threads_per_block, blocks_per_grid  */


    

}


/*main function*/
int main(int argc, char* argv[])
{
	// Vector length , Matrix Row and Col sizes..............
       	vlength = matColSize = atoi(argv[2]);
       	matRowSize =atoi(argv[1]);
    
     	//  printf("this programs does computation of square matrix only\n");
	float elapsedTime,Tsec;
	cudaEvent_t start,stop;

	device_Count=get_DeviceCount();
        printf("\n\nNUmber of Devices : %d\n\n", device_Count);

        // Device Selection, Device 1: Tesla C1060
        cudaSetDevice(0);
      
        int device;
        // Current Device Detection
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&deviceProp,device);
        printf("Using device %d: %s \n", device, deviceProp.name);


	
  	/*allocating the memory for each matrix */
	host_Mat =new double[matRowSize*matColSize];
	host_Vect = new double[vlength];
	host_ResVect = new double[matRowSize];

	
	// ---------------checking host memory  for error..............................
	 if(host_Mat==NULL)
                mem_error("host_Mat","vectmatmul",matRowSize*matColSize,"double");

         if(host_Vect==NULL)
                mem_error("host_Vect","vectmatmul",vlength,"double");

         if(host_ResVect==NULL)
                mem_error("host_ResVect","vectmatmul",matRowSize,"double");

	//--------------Initializing the input arrays..............
    srand48(time(nullptr));
	fill_dp_vector(host_Mat,matRowSize*matColSize);
	fill_dp_vector(host_Vect,vlength);

    // //print
    //     printf("host_mat:");
    //     for(int k=0; k<matRowSize*matColSize; k++){
    //         printf("%f ",host_Mat[k]);
    //     }
    //     printf("\n");

    //     printf("host_Vect:");
    //     for(int k=0; k<vlength; k++){
    //         printf("%f ",host_Vect[k]);
    //     }
    //     printf("\n");

 	/* allocate memory for GPU events 
        start = (cudaEvent_t) malloc (sizeof(cudaEvent_t));
        stop = (cudaEvent_t) malloc (sizeof(cudaEvent_t));
        if(start==NULL)
                mem_error("start","vectvectmul",1,"cudaEvent_t");
        if(stop==NULL)
                mem_error("stop","vectvectmul",1,"cudaEvent_t");*/
  	
	//event creation...
        CUDA_SAFE_CALL(cudaEventCreate (&start));
        CUDA_SAFE_CALL(cudaEventCreate (&stop));

  	//allocating memory on GPU
	CUDA_SAFE_CALL(cudaMalloc( (void**)&device_Mat, matRowSize*matColSize* sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc( (void**)&device_Vect, vlength* sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc( (void**)&device_ResVect, matRowSize* sizeof(double)));

 	//moving data from CPU to GPU
	CUDA_SAFE_CALL(cudaMemcpy((void*)device_Mat, (void*)host_Mat, matRowSize*matColSize*sizeof(double) ,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy((void*)device_Vect, (void*)host_Vect,vlength*sizeof(double),cudaMemcpyHostToDevice));

	// Launching kernell..........	
	CUDA_SAFE_CALL(cudaEventRecord (start, 0));
	
	int BlocksPerGrid=(matRowSize+BLOCKSIZE-1)/BLOCKSIZE;
        dim3 dimBlock(BLOCKSIZE);
        dim3 dimGrid(BlocksPerGrid);
        //check_block_grid_dim(deviceProp,dimBlock,dimGrid);

        int nIter = 300;
        for(int i=0; i<nIter; i++)
                MatVectMultiplication<<<(matRowSize+BLOCKSIZE-1)/BLOCKSIZE,BLOCKSIZE>>>(device_Mat,device_Vect,matRowSize,vlength,device_ResVect);
	
	CUDA_SAFE_CALL(cudaEventRecord (stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize (stop));
	CUDA_SAFE_CALL(cudaEventElapsedTime ( &elapsedTime, start, stop));

	Tsec= 1.0e-3*elapsedTime;
 	
	// calling funtion for measuring Gflops
        //calculate_gflops(Tsec, matRowSize, matColSize, nIter);
	
	//printing the result on screen
    	print_on_screen("MAT VECT MULTIPLICATION",Tsec,calculate_gflops(Tsec, matRowSize, matColSize, nIter),matRowSize, matColSize,1); 

 
	//retriving result from device
  	CUDA_SAFE_CALL(cudaMemcpy((void*)host_ResVect, (void*)device_ResVect,matRowSize*sizeof(double),cudaMemcpyDeviceToHost));

	// CPU calculation..and checking error deviation....
	CPU_MatVect();
    // //print
    //     printf("\n");
    //     printf("host:");
    //     for(int k=0; k<size; k++){
    //         printf("%f ",host_ResVect[k]);
    //     }
    //     printf("\n");

    //     printf("cpu:");
    //     for(int k=0; k<size; k++){
    //         printf("%f ",cpu_ResVect[k]);
    //     }
    //     printf("\n");
  	relError(cpu_ResVect,host_ResVect,size);
   	printf("\n ----------------------------------------------------------------------\n");

	/*free the memory from GPU */
	double *array[3];
        array[0]=device_Mat;
        array[1]=device_Vect;
        array[2]=device_ResVect;
        dfree(array,3);

	//free host memory----------
        free(host_Mat);
        free(host_Vect);
        free(host_ResVect);
        free(cpu_ResVect);

	return 0;
}// end of main