/*
 * assign3_1.c
 *
 * Contains code for setting up and finishing the simulation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "file.h"
#include "timer.h"

typedef double (*func_t)(double x);


/*
 * Fills a given array with samples of a given function. This is used to fill
 * the initial arrays with some starting data, to run the simulation on.
 *
 * The first sample is placed at array index `offset'. `range' samples are
 * taken, so your array should be able to store at least offset+range doubles.
 * The function `f' is sampled `range' times between `sample_start' and
 * `sample_end'.
 */
 
void fill(double *array, int offset, int range, double sample_start,
          double sample_end, func_t f) {
    int i;
    float dx;

    dx = (sample_end - sample_start) / range;
    for (i = 0; i < range; i++) {
        array[i + offset] = f(sample_start + i * dx);
    }
}


int main(int argc, char *argv[]) {


  // ensure correct usage
  if (argc != 5) {
    printf("Usage: %s i_max t_max thread_amt chunk_size\n", argv[0]);
    return EXIT_FAILURE;
  }

  // convert args
  int i_max = atoi(argv[1]);
  int t_max = atoi(argv[2]);
  int t_cnt = atoi(argv[3]);
  int chunk = atoi(argv[4]);

  // set amount of threads
  omp_set_num_threads(t_cnt);

  // malloc arrays
  double * prev = malloc(i_max * sizeof(double));
  double * curr = malloc(i_max * sizeof(double));
  double * next = malloc(i_max * sizeof(double));
  double * temp;
  if (prev == NULL || curr == NULL || next == NULL) {
    printf("\033[;31merror:\033[0m Couldn't malloc enough space.\n\n");
    return EXIT_FAILURE;
  }

  // START STATIC
  printf("Scheduler : STATIC\n");
  printf("Threads   : %d\n", t_cnt);
  printf("Chunk size: %d\n", chunk);

  fill(prev, 1, i_max/4, 0, 2*3.14, sin);
  fill(curr, 2, i_max/4, 0, 2*3.14, sin);

  double time;
  timer_start();

  printf("(performing calculations...)\n");

  for (int t = 0; t < t_max; t++) {
    #pragma omp parallel for schedule(static, chunk)
    for (int i = 1; i < i_max - 1; i++) {
      next[i] = 2 * curr[i] - prev[i] + 0.2 * ( curr[i - 1] - ( 2 * curr[i] - curr[i + 1] ));
    }
    temp = prev;
    prev = curr;
    curr = next;
    next = temp;
  }

  time = timer_end();
  printf("Took      : %g seconds\n", time);
  printf("Normalized: %g seconds\n", time / (1. * i_max * t_max));
  printf("Dumped array in 'result_static.txt'\n");
  file_write_double_array("result.txt_static", curr, i_max);
  // END STATIC

  // START DYNAMIC
  printf("\n");
  printf("Scheduler : DYNAMIC\n");
  printf("Threads   : %d\n", t_cnt);
  printf("Chunk size: %d\n", chunk);

  fill(prev, 1, i_max/4, 0, 2*3.14, sin);
  fill(curr, 2, i_max/4, 0, 2*3.14, sin);

  timer_start();

  printf("(performing calculations...)\n");

  for (int t = 0; t < t_max; t++) {
    #pragma omp parallel for schedule(dynamic, chunk)
    for (int i = 1; i < i_max - 1; i++) {
      next[i] = 2 * curr[i] - prev[i] + 0.2 * ( curr[i - 1] - ( 2 * curr[i] - curr[i + 1] ));
    }
    temp = prev;
    prev = curr;
    curr = next;
    next = temp;
  }

  time = timer_end();
  printf("Took      : %g seconds\n", time);
  printf("Normalized: %g seconds\n", time / (1. * i_max * t_max));
  printf("Dumped array in 'result_dynamic.txt'\n");
  file_write_double_array("result.txt_dynamic", curr, i_max);
  // END DYNAMIC

  // START GUIDED
  printf("\n");
  printf("Scheduler : GUIDED\n");
  printf("Threads   : %d\n", t_cnt);
  printf("Chunk size: %d\n", chunk);

  fill(prev, 1, i_max/4, 0, 2*3.14, sin);
  fill(curr, 2, i_max/4, 0, 2*3.14, sin);

  timer_start();

  printf("(performing calculations...)\n");

  for (int t = 0; t < t_max; t++) {
    #pragma omp parallel for schedule(guided, chunk)
    for (int i = 1; i < i_max - 1; i++) {
      next[i] = 2 * curr[i] - prev[i] + 0.2 * ( curr[i - 1] - ( 2 * curr[i] - curr[i + 1] ));
    }
    temp = prev;
    prev = curr;
    curr = next;
    next = temp;
  }

  time = timer_end();
  printf("Took      : %g seconds\n", time);
  printf("Normalized: %g seconds\n", time / (1. * i_max * t_max));
  printf("Dumped array in 'result_guided.txt'\n");
  file_write_double_array("result.txt_guided", curr, i_max);
  // END GUIDED

  return EXIT_SUCCESS;
}

