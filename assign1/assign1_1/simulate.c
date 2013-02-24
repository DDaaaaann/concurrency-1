/*******************************************************************************
 * 
 * Concurrency and parallel programming - Assignment 1.2
 * simulate.c
 * THREADED VERSION
 * 
 * by Ben Witzen and David van Schoorisse
 * University of Amsterdam, 07-11-2012
 * 
 * Implements the requested simulate function using pthreads.
 * 
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "simulate.h"

void usleep(int);

typedef struct thread_info {
    int b, e, c, right_bound;
    double *t0, *t1, *t2;
} thread_info;


/*
 * Debug function that prints detailed thread information.
 */

void getThreadInfo(thread_info * t) {
    printf("%d %d [%d]\n", t->b, t->e, t->c);
}


/*
 * Asks a thread if it is done with its task. Returns 0 if done, otherwise
 * returns the number of calculations it still has to do.
 */

int threadDone(thread_info * t) {
    return t->e - t->c;
}


/*
 * Asks whether all threads are done with their work. Returns 0 if all are done,
 * otherwise returns the number of calculations they still have to perform. This
 * function calls threadDone().
 */

int allThreadsDone(thread_info * t, int n){
    int val = 0;
    for(int i = 0; i < n; i++) {
        val += threadDone(&t[i]);
    }
    return val;
}


/*
 * Thread function that performs the calculations. Never returns.
 */

void * worker(void * v) {
    thread_info * t = (thread_info *)v;
    
    // keep running until joined
    while(1) {
        // somehow, it refuses to run without this usleep
        usleep(1);
        // check if the thread has work to do, otherwise remain idle
        while(t->c < t->e) {
            // the formula provided, taking borders into account
            if (t->c == 0)
              t->t2[t->c] = 2 * t->t1[t->c] - t->t0[t->c] + 0.2 * ( 0 - (2 * t->t1[t->c] - t->t1[t->c + 1] ) );
            else if (t->c == t->right_bound - 1)
              t->t2[t->c] = 2 * t->t1[t->c] - t->t0[t->c] + 0.2 * ( t->t1[t->c - 1] - (2 * t->t1[t->c] - 0 ) );
            else
              t->t2[t->c] = 2 * t->t1[t->c] - t->t0[t->c] + 0.2 * ( t->t1[t->c - 1] - (2 * t->t1[t->c] - t->t1[t->c + 1] ) );
            t->c += 1;
        }
    }

    // keeps the compiler happy - we won't actually reach this
    return NULL;
}


/*
 * Executes the entire simulation.
 *
 * i_max        : how many data points are on a single wave
 * t_max        : how many iterations the simulation should run
 * num_threads  : how many threads to use (excluding the main threads)
 * old_array    : array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array   : array of size i_max. You should fill this with t+1
 *
 * returns      : pointer to double array filled with data of time t_max
 */

double *simulate(const int i_max, const int t_max, const int num_threads,
        double *old_array, double *current_array, double *next_array) {

    // determine how to divide the work to be done
    int amt_per_thread = i_max / num_threads;
    int amt_leftover = i_max % num_threads;
    int t_current = 0;

    // arrays store thread_info and thread_id's
    thread_info * t_info = calloc(num_threads, sizeof(thread_info));
    pthread_t * t_id = malloc(num_threads * sizeof(pthread_t));
    
    // prepares t_info structs
    for (int i = 0; i < num_threads; i++) {
        t_info[i].t0 = old_array;
        t_info[i].t1 = current_array;
        t_info[i].t2 = next_array;
        t_info[i].b = i * amt_per_thread;
        t_info[i].c = t_info[i].b;
        t_info[i].right_bound = i_max;
        if (i != (num_threads - 1))
            t_info[i].e = (i + 1) * amt_per_thread;
        else
            t_info[i].e = (i + 1) * amt_per_thread + amt_leftover;
    
        // all information is know, create threads now
        pthread_create(&t_id[i], NULL, &worker, (void *)(&t_info[i]));   
    }
    
    // calculate each t until we've reached t_max
    while(t_current < t_max) {
        
        // ask if all threads are done, if so, prepare for the next iteration
        if(!allThreadsDone(t_info, num_threads)) {
            double * tmp_array = old_array;
            old_array = current_array;
            current_array = next_array;
            next_array = tmp_array;
    
            // reset all thread's information
            for (int i = 0; i < num_threads; i++) {
                t_info[i].t0 = old_array;
                t_info[i].t1 = current_array;
                t_info[i].t2 = next_array;
                // wake threads up
                t_info[i].c = t_info[i].b;
            }

            // goto next time step
            t_current++;
        }
    }

    // we're done - join pthreads
    /*void * thread_status = NULL;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(t_id[i], thread_status);
    }*/
    
    // all done!
    return next_array;
}
