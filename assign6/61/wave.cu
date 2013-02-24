/*******************************************************************************
 * 
 * wave.cu
 * Concurrency and parallel programming, assignment 6.1
 * 
 * by David van Schoorisse and Ben Witzen
 * University of Amsterdam, 12-12-12 (nice date!)
 * 
 * Calculates the well-known wave equation using CUDA. This program fills the
 * first 25% of the arrays (t0 and t1) with the sinus function, and calculates
 * the wave progression from there.
 *
 * Skeleton code was provided by Robert G. Belleman (University of Amsterdam).
 *
 * Usage: ./wave [i_max t_max blocksize]
 * i_max      (unsigned int) : amount of array elements;    (default 1024)
 * t_max      (unsigned int) : amount of timesteps;         (default 1000)
 * blocksize  (unsigned int) : amount of threads per block. (default  512)
 * 
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "timer.h"
#include <iostream>

double * end;

using namespace std;

/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}


__global__ void waveKernel(int n, double* prev, double* curr, double* next) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // shared, 1 element per thread
    __shared__ double s_prev[1024];
    __shared__ double s_curr[1024];
    
    // each thread reads 1 device memory element
    s_prev[threadIdx.x] = prev[i];
    s_curr[threadIdx.x] = curr[i];
    
    // avoid race condition
    __syncthreads();
    
    // nonborders
    if (threadIdx.x > 0 && threadIdx.x < blockDim.x - 1)
        next[i] = 2* s_curr[threadIdx.x] - s_prev[threadIdx.x] + 0.2 * ( s_curr[i-1] - ( 2 * s_curr[i] - s_curr[i+1] ));
    // left border
    else if (threadIdx.x == 0)
        next[i] = 2* s_curr[threadIdx.x] - s_prev[threadIdx.x] + 0.2 * ( 0 - ( 2 * s_curr[i] - s_curr[i+1] ));
    // right border
    else
        next[i] = 2* s_curr[threadIdx.x] - s_prev[threadIdx.x] + 0.2 * ( s_curr[i-1] - ( 2 * s_curr[i] - 0 ));
}

void waveCuda(int blocksize, int n, int t_max, double* a, double* b, double* result) {
    int threadBlockSize = blocksize;

    // allocate the vectors on the GPU
    double* deviceA = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceA, n * sizeof(double)));
    if (deviceA == NULL) {
        cout << "could not allocate memory!" << endl;
        return;
    }
    double* deviceB = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceB, n * sizeof(double)));
    if (deviceB == NULL) {
        checkCudaCall(cudaFree(deviceA));
        cout << "could not allocate memory!" << endl;
        return;
    }
    double* deviceResult = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceResult, n * sizeof(double)));
    if (deviceResult == NULL) {
        checkCudaCall(cudaFree(deviceA));
        checkCudaCall(cudaFree(deviceB));
        cout << "could not allocate memory!" << endl;
        return;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // copy the original vectors to the GPU
    checkCudaCall(cudaMemcpy(deviceA, a, n*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceB, b, n*sizeof(double), cudaMemcpyHostToDevice));
    // copy the result vector (it's zero'd), just to be sure
    checkCudaCall(cudaMemcpy(deviceResult, result, n*sizeof(double), cudaMemcpyHostToDevice));
        
    // execute kernel, t_max amount of times.
    cudaEventRecord(start, 0);
    for (int t = 1; t < t_max; t++) {
        // execute kernel
        waveKernel<<<ceil((double)n/threadBlockSize), threadBlockSize>>>(n, deviceA, deviceB, deviceResult);
        // rotate buffers
        double * tmp = deviceA;  // tmp = prev
        deviceA = deviceB;       // prev = cur
        deviceB = deviceResult;  // cur = next
        deviceResult = tmp;      // next = tmp (= prev)
    }     
    cudaEventRecord(stop, 0);

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    checkCudaCall(cudaMemcpy(result, deviceResult, n * sizeof(double), cudaMemcpyDeviceToHost));

    checkCudaCall(cudaFree(deviceA));
    checkCudaCall(cudaFree(deviceB));
    checkCudaCall(cudaFree(deviceResult));

    // print the time the kernel invocation took, without the copies!
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    cout << "kernel invocation took " << elapsedTime << " milliseconds" << endl;
}


/*
 * Quite a handy function that fills an array with samples of a given function.
 * Previously used in assignments 1.1, 2.1, and 3.1 for this course. Thanks
 * Koen!
 */

typedef double (*func_t)(double x);

void fill(double *array, int offset, int range, double sample_start,
          double sample_end, func_t f) {
    int i;
    double dx;

    dx = (sample_end - sample_start) / range;
    for (i = 0; i < range; i++)
        array[i + offset] = f(sample_start + i * dx);
}


int main(int argc, char* argv[]) {

    // default values for variables
    int n = 1024;
    int t_max = 1000;
    int blocksize = 512;

    // process command-line arguments
    if (argc == 1) {
        cout << "\033[;33m" << endl;
        cout << "Remember you can use command-line arguments:" << endl;
        cout << "./wave [i_max t_max blocksize]" << endl;
        cout << "Now I'm using default values." << endl;
        cout << "\033[0m" << endl;
    }
    else if (argc == 4) {
        n = atoi(argv[1]);
        t_max = atoi(argv[2]);
        blocksize = atoi(argv[3]);
    }
    else {
        cout << "\033[;33m" << endl;
        cout << "Invalid amount of arguments." << endl;
        cout << "./wave [INT i_max INT t_max INT blocksize]" << endl;
        cout << "i_max = array length    t_max = timesteps" << endl;
        cout << "\033[0m" << endl;
        return -2;
    }
    
    // validate arguments
    if (n <= 0 || t_max <= 0 || blocksize <= 0) {
        cout << "Argument error: each argument must be >0." << endl;
        return -2;
    }

    // print values being used
    cout << "\033[;36m" << endl;
    cout << "Using values:" << endl;
    cout << "i_max = " << n << endl;
    cout << "t_max = " << t_max << endl;
    cout << "blocksize = " << blocksize << endl;
    cout << "\033[0m" << endl;

    // start timer, prepare arrays
    timer vectorAddTimer("vector add timer");
    double* a = new double[n];
    double* b = new double[n];
    double* result = new double[n];

    // initialize the vectors
    fill(a, 1, n/4, 0, 2*3.14, sin);
    fill(b, 2, n/4, 0, 2*3.14, sin);
    // set the result vector to 0, just to be sure
    for (int i = 0; i < n; i++)
        result[i] = 0;

    vectorAddTimer.start();
    waveCuda(blocksize, n, t_max, a, b, result);
    vectorAddTimer.stop();

    cout << vectorAddTimer;
    
    // write results to file (might be useful)
    FILE * fp = fopen("results.txt", "w");
    if (fp == NULL)
        cout << "Could not write away results.txt" << endl;
    else {
        for (int i = 0; i < n; i++)
            fprintf(fp, "%f\n", result[i]);
        cout << "Results written to results.txt" << endl;
        fclose(fp);
    }
            
    delete[] a;
    delete[] b;
    delete[] result;
    
    return 0;
}
