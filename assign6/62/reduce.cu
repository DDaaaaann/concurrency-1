/*******************************************************************************
 * 
 * reduce.cu
 * Concurrency and parallel programming, assignment 6.1
 * 
 * by David van Schoorisse and Ben Witzen
 * University of Amsterdam, 12-12-12 (nice date!)
 * 
 * Finds the maximum value of an array. Parallizes the search by CUDA to speed
 * up the process.
 *
 * Skeleton code was provided by Robert G. Belleman (University of Amsterdam).
 *
 * Usage: ./wave [i_max t_max blocksize]
 * i_max      (unsigned int) : amount of array elements;    (default  512)
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


__global__ void reduceKernel(int n, double* a, double* b) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    
    /*
        note to self:
            a is the input vector
            b is the write-to vector
            n is the current problemsize (arraysize)
    */
    
    // shared, 1 element per thread
    __shared__ double sa[1024];
    
    // each thread reads 1 device memory element
    sa[threadIdx.x] = a[i];
    
    // avoid race condition
    __syncthreads();
    
    // do comparison, copy biggest to result array
    if ((i % 2 == 0) && i < n) {
        if (sa[i] > sa[i+1])
            b[i/2] = sa[i];
        else
            b[i/2] = sa[i+1];    
    }
    
}

void reduceCuda(int blocksize, int n, double* a, double* b) {
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // copy the original vectors to the GPU
    checkCudaCall(cudaMemcpy(deviceA, a, n*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceB, b, n*sizeof(double), cudaMemcpyHostToDevice));
        
    // execute kernel
    // each execution halves the problem size
    cudaEventRecord(start, 0);
    for (int psize = n; psize > 1; psize /= 2) {
        // execute kernel
        reduceKernel<<<ceil((double)n/threadBlockSize), threadBlockSize>>>(n, deviceA, deviceB);
        // rotate arrays
        double * tmp = deviceA;
        deviceA = deviceB;
        deviceB = tmp;
    }
    
    cudaEventRecord(stop, 0);

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    checkCudaCall(cudaMemcpy(b, deviceB, n * sizeof(double), cudaMemcpyDeviceToHost));

    checkCudaCall(cudaFree(deviceA));
    checkCudaCall(cudaFree(deviceB));

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
    int blocksize = 512;

    // process command-line arguments
    if (argc == 1) {
        cout << "\033[;33m" << endl;
        cout << "Remember you can use command-line arguments:" << endl;
        cout << "./wave [i_max blocksize]" << endl;
        cout << "Now I'm using default values." << endl;
        cout << "\033[0m" << endl;
    }
    else if (argc == 3) {
        n = atoi(argv[1]);
        blocksize = atoi(argv[2]);
    }
    else {
        cout << "\033[;33m" << endl;
        cout << "Invalid amount of arguments." << endl;
        cout << "./wave [i_max blocksize]" << endl;
        cout << "i_max = array length    t_max = timesteps" << endl;
        cout << "\033[0m" << endl;
        return -2;
    }
    
    // validate arguments
    if (n <= 0 || blocksize <= 0 || blocksize > 1024) {
        cout << "Argument error: each argument must be >0, and blocksize must be <=1024." << endl;
        return -2;
    }

    // print values being used
    cout << "\033[;36m" << endl;
    cout << "Using values:" << endl;
    cout << "i_max = " << n << endl;
    cout << "blocksize = " << blocksize << endl;
    cout << "\033[0m" << endl;

    // start timer, prepare arrays
    timer cudaTime("CUDA Timer");
    timer seqTime("Sequential Timer");
    double* a = new double[n];
    double* b = new double[n];

    // initialize the vector
    fill(a, 1, n, 0, 2*3.14, sin);

    // cuda implementation
    cudaTime.start();
    reduceCuda(blocksize, n, a, b);
    cudaTime.stop();
    
    // sequential implementation
    seqTime.start();    
    double max;
    for (int i = 0; i < n; i++)
        if (a[i] > max || i == 0)
            max = a[i];
    seqTime.stop();
    
    cout << cudaTime;
    cout << seqTime;
    
    cout << "Cuda's answer = " << b[0] << endl;
    cout << "Seq's answer  = " << max << endl;
    
    
    
    /*
        FILE * fp = fopen("stuff.txt", "w");
    if (fp == NULL)
        cout << "Could not write away results.txt" << endl;
    else {
        for (int i = 0; i < n; i++)
            fprintf(fp, "%f\n", a[i]);
        cout << "Results written to results.txt" << endl;
        fclose(fp);
    }
    
    FILE * fpd = fopen("results.txt", "w");
    if (fpd == NULL)
        cout << "Could not write away results.txt" << endl;
    else {
        for (int i = 0; i < n; i++)
            fprintf(fpd, "%f\n", b[i]);
        cout << "Results written to results.txt" << endl;
        fclose(fpd);
    }
    */
    
    // test result with sequential implementation
    
    // don't forget to time sequential implementation
            
    delete[] a;
    delete[] b;
    
    return 0;
}