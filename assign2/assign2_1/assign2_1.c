/*
 * assign1_1.c
 *
 * Contains code for setting up and finishing the simulation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "file.h"
#include "timer.h"
#include "simulate.h"

#define TAG_1 1
#define TAG_2 2
#define TAG_3 3
#define TAG_4 4

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

    // start MPI
    int rank, amt;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &amt);

    // check for correct amount of arguments
    if (argc < 3 || argc > 4) {
        MPI_Finalize();
        if (rank == 0)
            printf("Usage: %s i_max t_max\n", argv[0]);
        return EXIT_FAILURE;
    }

    // check for valid arguments
    if (atoi(argv[1]) < 3 || atoi(argv[2]) < 1) {
        MPI_Finalize();
        if (rank == 0)
            printf("i_max should be >2 and t_max should be >=1.\n");
        return EXIT_FAILURE;
    }

    // MASTER CODE
    if (rank == 0) {

        // parse command-line arguments
        int i_max = atoi(argv[1]);
        int t_max = atoi(argv[2]);

        // allocate full-sized buffers...
        double * old     = malloc(i_max * sizeof(double));
        double * current = malloc(i_max * sizeof(double));
        double * next    = malloc(i_max * sizeof(double));
        if (old == NULL || current == NULL || next == NULL) {
            fprintf(stderr, "Could not allocate enough memory, aborting.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return EXIT_FAILURE;
        }

        // ...and fill them - we'll stick with sinus for now
        fill(old, 1, i_max/4, 0, 2*3.14, sin);
        fill(current, 2, i_max/4, 0, 2*3.14, sin);

        // all pre-work is done - start the clock
        double time;
        timer_start();

        // calculate array sizes and create buffers
        int split_array_size = i_max / amt;
        int split_array_modu = i_max % amt;
        double buffer_old[split_array_size];
        double buffer_cur[split_array_size];
        double buffer_next[split_array_size];

        // send each worker the info they need to get started
        for (int recv = 1; recv < amt; recv++) {
            MPI_Send(&split_array_size, 1, MPI_INT, recv, TAG_1, MPI_COMM_WORLD);
            MPI_Send(&t_max, 1, MPI_INT, recv, TAG_2, MPI_COMM_WORLD);
            for (int i = 0; i < split_array_size; i++) {
                buffer_old[i] = old[i + (recv - 1) * split_array_size];
                buffer_cur[i] = current[i + (recv - 1) * split_array_size];
            }
            MPI_Send(&buffer_old, split_array_size, MPI_DOUBLE, recv, TAG_3, MPI_COMM_WORLD);
            MPI_Send(&buffer_cur, split_array_size, MPI_DOUBLE, recv, TAG_3, MPI_COMM_WORLD);
        }
        
        MPI_Send(&old[i_max - split_array_size - split_array_modu]    , 1, MPI_DOUBLE, amt - 1, TAG_4, MPI_COMM_WORLD);
        MPI_Recv(&old[i_max - split_array_size - split_array_modu - 1], 1, MPI_DOUBLE, amt - 1, TAG_4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&current[i_max - split_array_size - split_array_modu]    , 1, MPI_DOUBLE, amt - 1, TAG_4, MPI_COMM_WORLD);
        MPI_Recv(&current[i_max - split_array_size - split_array_modu - 1], 1, MPI_DOUBLE, amt - 1, TAG_4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // start master's own calculations - it's stored directly in the result array
        for (int t = 0; t < t_max; t++) {
            MPI_Send(&current[i_max - split_array_size - split_array_modu]    , 1, MPI_DOUBLE, amt - 1, TAG_4, MPI_COMM_WORLD);
            MPI_Recv(&current[i_max - split_array_size - split_array_modu - 1], 1, MPI_DOUBLE, amt - 1, TAG_4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = i_max - split_array_size - split_array_modu; i < i_max; i++)
                next[i] = 2 * current[i] - old[i] + 0.2 * ( current[i - 1] - ( 2 * current[i] - current[i + 1] ));
            // rotate buffers
            memcpy(old, current, i_max * sizeof(double));
            memcpy(current, next, i_max * sizeof(double));
        }
        
        // master is done with his work, collect all the work done by workers
        for (int i = 1; i < amt; i++) {
            MPI_Recv(buffer_next, split_array_size, MPI_DOUBLE, i, TAG_3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int t = 0; t < split_array_size; t++) {
                next[t + (i - 1) * split_array_size] = buffer_next[t];
            }
        }

        // our new_array is now complete - stop the clock
        time = timer_end();

        // our new_array is now completely built
        // we probably want to replace this with a file write or sumat. Meh.
        printf("New Array:\n");
        for (int i = 0; i < i_max; i++) {
            printf("%lf \n", next[i]);
        }
        printf("\n");

        
        printf("Took %g seconds\n", time);
        printf("Normalized: %g seconds\n", time / (1. * i_max * t_max));
        
        // some other stuff we might want to do
        free(old);
        free(current);
        free(next);

    }

    // WORKER CODE
    else {

        // receive all our required info
        int array_size, t_max;

        MPI_Recv(&array_size, 1, MPI_INT, 0, TAG_1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&t_max,      1, MPI_INT, 0, TAG_2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double buffer_old[array_size + 2], buffer_cur[array_size + 2], buffer_next[array_size + 2];

        MPI_Recv(buffer_old + 1, array_size, MPI_DOUBLE, 0, TAG_3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(buffer_cur + 1, array_size, MPI_DOUBLE, 0, TAG_3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // determine neighbors
        int left_neighbor = rank - 1;
        int right_neighbor = rank + 1;
        if (right_neighbor >= amt)
            right_neighbor = 0;
        double * my_left_ele = buffer_cur + 1;
        double * my_right_ele = buffer_cur + array_size;
        double * my_left_old = buffer_old + 1;
        double * my_right_old = buffer_old + array_size;
        
        // communicate with neighbors to fill the missing slot in our old buffers
        if (rank != 1)
            MPI_Send(my_left_old, 1, MPI_DOUBLE, left_neighbor, TAG_4, MPI_COMM_WORLD);
        MPI_Recv(my_right_old + 1, 1, MPI_DOUBLE, right_neighbor, TAG_4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(my_right_old    , 1, MPI_DOUBLE, right_neighbor, TAG_4, MPI_COMM_WORLD);
        if (rank != 1)
            MPI_Recv(my_left_old - 1, 1, MPI_DOUBLE, left_neighbor, TAG_4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // communicate with neighbors to fill the missing slot in our cur buffers
        if (rank != 1)
            MPI_Send(my_left_ele, 1, MPI_DOUBLE, left_neighbor, TAG_4, MPI_COMM_WORLD);
        MPI_Recv(my_right_ele + 1, 1, MPI_DOUBLE, right_neighbor, TAG_4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(my_right_ele    , 1, MPI_DOUBLE, right_neighbor, TAG_4, MPI_COMM_WORLD);
        if (rank != 1)
            MPI_Recv(my_left_ele - 1, 1, MPI_DOUBLE, left_neighbor, TAG_4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for (int t = 0; t < t_max; t++) {
            // communicate with neighbors
            if (rank != 1)
                MPI_Send(my_left_ele, 1, MPI_DOUBLE, left_neighbor, TAG_4, MPI_COMM_WORLD);
            MPI_Recv(my_right_ele + 1, 1, MPI_DOUBLE, right_neighbor, TAG_4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(my_right_ele    , 1, MPI_DOUBLE, right_neighbor, TAG_4, MPI_COMM_WORLD);
            if (rank != 1)
                MPI_Recv(my_left_ele - 1, 1, MPI_DOUBLE, left_neighbor, TAG_4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
            // perform the calculation
            for (int i = 1; i < array_size + 1; i++)
                buffer_next[i] = 2 * buffer_cur[i] - buffer_old[i] + 0.2 * ( buffer_cur[i - 1] - ( 2 * buffer_cur[i] - buffer_cur[i + 1] ));
            // rotate buffers
            memcpy(buffer_old, buffer_cur, (array_size + 2) * sizeof(double));
            memcpy(buffer_cur, buffer_next, (array_size + 2) * sizeof(double));
        }

        // send results to master
        MPI_Send(buffer_next + 1, array_size, MPI_DOUBLE, 0, TAG_3, MPI_COMM_WORLD);

    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
