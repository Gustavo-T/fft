
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include "utl.cuh"

#include <math.h>
#include <cstring>

#include <windows.h>

#define SIZE 1024*1024*2
#define THREADS_PER_BLOCK 256
#define BLOCKS 32


//-----------------Complex multiply------------------
static __device__ __host__ inline cufftComplex complexMul(cufftComplex a, cufftComplex b)
{
	float tmp = 0.0f;
	tmp = a.x * b.x - a.y*(-1)*b.y;
	a.y = a.x *(-1)* b.y + b.x * a.y;
	a.x = tmp;

	return a;
}
//-------------------------------------
static __global__ void multiply(cufftComplex* a, cufftComplex* b, int size)
{

	const int numThreads = blockDim.x * gridDim.x;
	const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	for (size_t i = threadId;  i < size; i+= numThreads ) {
		a[i] = complexMul(a[i],b[i]);

	}
}
//-------------------------------------

bool memcpCheck(cufftComplex* a, cufftComplex* d_a, size_t size ) 
{
	checkCudaErrors(cudaMemcpy(d_a, a,sizeof(cufftComplex) * size, cudaMemcpyHostToDevice));
	cufftComplex* tData = new cufftComplex[size];
	checkCudaErrors(cudaMemcpy(tData, d_a,sizeof(cufftComplex) * size, cudaMemcpyDeviceToHost));
	return  (memcmp(a,tData,sizeof(cufftComplex) * size) == 0) ? true : false;

}

int main()
{
	int dev = 0;
	printf("Input device number ");
	scanf("%d", &dev);

	cudaSetDevice(dev);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp,dev);
	printf("\nName %s\n", deviceProp.name);


	//////////////

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	float timeStartMem, timeMem = 0; 
	
	int logSize = log(2.0*SIZE)/log(2.0) + 1;
	size_t TSIZE = pow(2.0, logSize);
	size_t maxLag = SIZE - 1;


	cufftComplex  *IndataX = new cufftComplex[SIZE];
	cufftComplex  *IndataY = new cufftComplex[SIZE];

	cufftComplex *dataX = new cufftComplex[TSIZE];
	cufftComplex *dataY = new cufftComplex[TSIZE];

	cufftComplex *resultX = new cufftComplex[TSIZE];
	cufftComplex *result = new cufftComplex[TSIZE];

	cufftHandle forwardPlanX, forwardPlanY, backwardPlan;
	
	// create a simple plan for 1D transform
	if(cufftPlan1d(&forwardPlanX,TSIZE,CUFFT_C2C,1) != CUFFT_SUCCESS) { printf("Cuda: cufftPlan1d CUFFT_C2C failed\n"); return 1; }
	if(cufftPlan1d(&forwardPlanY,TSIZE,CUFFT_C2C,1) != CUFFT_SUCCESS) { printf("Cuda: cufftPlan1d CUFFT_C2C failed\n"); return 1; }
	if (cufftPlan1d(&backwardPlan,TSIZE,CUFFT_C2C,1) != CUFFT_SUCCESS) { printf("Cuda: cufftPlan1d CUFFT_C2C failed\n"); return 1; }
	///
	// file open
	FILE* inFileX = fopen("file1.sig","rb");

	if(!inFileX)
	{
		printf("Can't open file %s\n","file1.sig");
		exit(0);
	}


	FILE* inFileY = fopen("file2.sig","rb");

	if(!inFileY)
	{
		printf("Can't open file %s\n","file2.sig");
		exit(0);
	}

		short int xreal = 0,yreal = 0;
		short int xim = 0,yim = 0;

	//download data from files
	for(size_t i = 0; i < SIZE; ++i)
	{
		
		fread(&xreal, sizeof( unsigned short ), 1, inFileX);
		fread(&yreal, sizeof(unsigned short), 1, inFileY);

		fread(&xim, sizeof( unsigned short ), 1, inFileX);
		fread(&yim, sizeof( unsigned short ), 1, inFileY);

		IndataX[i].x = (float) xreal;
		IndataX[i].y = (float) xim;

		IndataY[i].x = (float) yreal;
		IndataY[i].y = (float) yim;

	}

	fclose(inFileX);
	fclose(inFileY);

	cufftComplex XMean,YMean;
	memset(&XMean,0,sizeof(cufftComplex));
	memset(&YMean,0,sizeof(cufftComplex));

	for(size_t i = 1; i < SIZE; ++i)
	{

		dataX[i-1].x = IndataX[i].x * IndataX[i-1].x - IndataX[i].y * (-1)* IndataX[i-1].y; 
		dataX[i-1].y = IndataX[i].x * (-1) * IndataX[i-1].y + IndataX[i-1].x * IndataX[i].y;

		dataY[i-1].x = IndataY[i].x * IndataY[i-1].x - IndataY[i].y * (-1) * IndataY[i-1].y;
		dataY[i-1].y = IndataY[i].x * (-1) * IndataY[i-1].y + IndataY[i-1].x * IndataY[i].y;

		XMean.x += dataX[i-1].x; 
		XMean.y += dataX[i-1].y;

		YMean.x += dataY[i-1].x; 
		YMean.y += dataY[i-1].y;

	}

	XMean.x = XMean.x/(SIZE-1);
	XMean.y = XMean.y/(SIZE-1);

	YMean.x = YMean.x/(SIZE-1);
	YMean.y = YMean.y/(SIZE-1);


	for(size_t i = 0; i < maxLag; ++i)
	{
		dataX[i].x-= XMean.x; 
		dataX[i].y-= XMean.y;

		dataY[i].x-= YMean.x; 
		dataY[i].y-= YMean.y;
	}

	float SigmaX = 0.0f;
	float SigmaY = 0.0f;

	//compute sigma
	for(size_t i = 0; i < maxLag; ++i)
	{
		SigmaX += (dataX[i].x*dataX[i].x) + (dataX[i].y*dataX[i].y);
		SigmaY += (dataY[i].x*dataY[i].x) + (dataY[i].y*dataY[i].y);
	}

	SigmaX = sqrt(SigmaX);
	SigmaY = sqrt(SigmaY);


	//pad vector with 0's
	for(size_t i = maxLag; i < TSIZE; ++i)
	{
		dataX[i].x = 0;
		dataX[i].y = 0;

		dataY[i].x = 0; 
		dataY[i].y = 0;

	}

	cufftComplex *d_dataX, *d_dataY;

	long sizeData = 0;
	checkCudaErrors(cudaMalloc(&d_dataX, sizeof(cufftComplex) * TSIZE));
	
	cudaEventRecord(start, 0);
	// Copy host memory to device
	checkCudaErrors(cudaMemcpy(d_dataX, dataX,sizeof(cufftComplex) * TSIZE, cudaMemcpyHostToDevice));
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&timeStartMem, start, stop);
	timeMem = timeStartMem;
	
	sizeData = sizeof(cufftComplex) * TSIZE;


	checkCudaErrors(cudaMalloc(&d_dataY, sizeof(cufftComplex) * TSIZE));

	if(memcpCheck(dataX,d_dataX,TSIZE)) 
	{
		cudaEventRecord(start, 0);
		checkCudaErrors(cudaMemcpy(d_dataY, dataY,sizeof(cufftComplex) * TSIZE, cudaMemcpyHostToDevice));
		cudaEventRecord(stop, 0); 
		cudaEventSynchronize(stop); 
		cudaEventElapsedTime(&timeStartMem, start, stop);
		timeMem += timeStartMem;
		
	}
	
	float elapsedTime, execTime = 0; // time execute cufftExec

	cudaEventRecord(start, 0);
	// execute plan for forward fft
	if ( cufftExecC2C(forwardPlanX,d_dataX,d_dataX,CUFFT_FORWARD) != CUFFT_SUCCESS) { printf("Cuda: cufftExecC2C failed\n"); return 1; }
	
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	
	cudaEventElapsedTime(&elapsedTime, start, stop);
	execTime = elapsedTime;
	
	

	//////////////////////////////////////////////////////

	cudaEventRecord(start, 0);

	checkCudaErrors(cudaMemcpy(dataX, d_dataX,sizeof(cufftComplex) * TSIZE, cudaMemcpyDeviceToHost));

	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&timeStartMem, start, stop);
	timeMem += timeStartMem;

	cudaEventRecord(start, 0);
	// execute plan for forward fft
	if ( cufftExecC2C(forwardPlanY,d_dataY,d_dataY,CUFFT_FORWARD) != CUFFT_SUCCESS) { printf("Cuda: cufftExecC2C failed\n"); return 1; }
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	execTime += elapsedTime;
	
	cudaEventRecord(start, 0);
	checkCudaErrors(cudaMemcpy(dataY, d_dataY,sizeof(cufftComplex) * TSIZE, cudaMemcpyDeviceToHost));
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&timeStartMem, start, stop);
	timeMem += timeStartMem;
	
	cufftComplex *d_result;
	
	checkCudaErrors(cudaMalloc(&d_result, sizeof(cufftComplex) * TSIZE));

	cudaEventRecord(start, 0);
	checkCudaErrors(cudaMemcpy(d_result, dataX,sizeof(cufftComplex) * TSIZE, cudaMemcpyHostToDevice));
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&timeStartMem, start, stop);
	timeMem += timeStartMem;
	
	printf("sizeof(dataX)= %d bytes\n", TSIZE);
	
	multiply<<<BLOCKS,THREADS_PER_BLOCK>>>(d_dataX,d_dataY,TSIZE);

	cudaEventRecord(start, 0);
	// execute plan for inverse fft
	if ( cufftExecC2C(backwardPlan,d_dataX,d_result,CUFFT_INVERSE) != CUFFT_SUCCESS) { printf("Cuda: cufftExecC2C failed\n"); return 1; }
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	execTime += elapsedTime;

	cudaDeviceSynchronize();

	cudaEventRecord(start, 0);
	checkCudaErrors(cudaMemcpy(result,d_result,sizeof(cufftComplex) * TSIZE,cudaMemcpyDeviceToHost));
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&timeStartMem, start, stop);
	timeMem += timeStartMem;
	
	for(size_t i = 0; i < TSIZE; ++i)
	{
		result[i].x = result[i].x/SigmaX;
		result[i].x = result[i].x/SigmaY;
		result[i].x = result[i].x/TSIZE;

		result[i].y = result[i].y/SigmaX;
		result[i].y = result[i].y/SigmaY;
		result[i].y = result[i].y/TSIZE;

	}

	cufftComplex *Result =  new cufftComplex[TSIZE];

	//swap positive and negative lags
	for(unsigned int i = 0; i < maxLag; ++i)
	{
		memcpy(&Result[i].x,&result[TSIZE - maxLag + 1 + i].x,sizeof(cufftComplex) );
		memcpy(&Result[i].y,&result[TSIZE - maxLag + 1 + i].y,sizeof(cufftComplex) );
	}

	for(size_t i = 0; i < maxLag; ++i)
	{
		Result[i + maxLag].x = result[i].x; 
		Result[i + maxLag].y = result[i].y;
	}

	float max = 0;
	size_t maxindex;

	for(size_t i = 0; i < 2*maxLag; ++i)
	{
		float temp = sqrt(Result[i].x*Result[i].x + Result[i].y*Result[i].y);
		
		if (temp > max)
		{
			max = temp;
			maxindex = i;
		}
		
	}

	double res = 3*sizeData/(timeMem*0.001);
	long r = res;
	printf("Max element %1.7f, index %d", max , maxindex+1);

	// free
	cudaFree(d_dataX);
	cudaFree(d_dataY);
	cudaFree(d_result);

	delete []IndataX;
	delete []IndataY;
	delete []dataX;
	delete []dataY;
	delete []resultX;
	delete []result;
	delete []Result;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cufftDestroy(forwardPlanX);
	cufftDestroy(forwardPlanY);
	cufftDestroy(backwardPlan);

	// print results
	int te = execTime, tm = timeMem;
	
	printf("\nTime mem, ms %d\n", tm);
	printf("Time exec,ms = %d\n", te);
	printf("Velocity, mb/s %ul \n", r/(1024*1024));
	///

	system("pause"); // if necessary
	return 0;
}

