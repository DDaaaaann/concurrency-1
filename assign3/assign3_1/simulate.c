/*******************************************************************************
 * 
 * Concurrency and parallel programming - Assignment 1.2
 * simulate.c
 * SEQUENTIAL VERSION
 * 
 * by Ben Witzen and David van Schoorisse
 * University of Amsterdam, 07-11-2012
 * 
 * Implements the requested simulate function SEQUENTIAL, without using any
 * threading at all. Used to test speed against our threaded implementation.
 * 
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include "simulate.h"


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

double *simulate(const int i_max, const int t_max,
       double *old, double *current, double *next) {

  printf("Note that this is the sequential version.\n");

  double * tmp;

  for (int t = 0; t < t_max; t++) {
    for (int i = 0; i < i_max; i++) {
      // border case (left)
      if (i == 0)
        next[i] = 2 * current[i] - old[i] + 0.2 * (0 - (2 * current[i] - current[i+1]));
      // border case (right)
      else if (i == i_max - 1)
        next[i] = 2 * current[i] - old[i] + 0.2 * (current[i-1] - (2 * current[i] - 0));
      // all other cases
      else
        next[i] = 2 * current[i] - old[i] + 0.2 * (current[i-1] - (2 * current[i] - current[i+1]));
    }
    // rotate array pointers
    tmp = old;
    old = current;
    current = next;
    next = tmp;
  }
  // return last filled array
  return next;
}
