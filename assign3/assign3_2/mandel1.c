/*
 * assignment 3.2
 * David van Schoorisse, Ben Witzen
 *
 * Mandelval, edited and optimized with OpenMP.
 * This is the version that does output.
 *
 * NOTE: SPECIFY AN OUTFILE AS PER "> OUT" OR RISK HAVING YOUR TERMINAL NUKED.
 *
 */



/*
  A program to generate an image of the Mandelbrot set. 

  Usage: ./mandelbrot > output
         where "output" will be a binary image, 1 byte per pixel
         The program will print instructions on stderr as to how to
         process the output to produce a JPG file.

  Michael Ashley / UNSW / 13-Mar-2003
*/

// Define the range in x and y here:

double yMin = -1.0;
double yMax = +1.0;
double xMin = -2.0;
double xMax = +0.5;

// And here is the resolution:

double dxy = 0.0001;

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
#include "timer.h"

int main(int argc, char ** argv) {

    // enhanced usage, useful for testing
    if (argc != 1 && argc != 2 && argc != 7) {
        fprintf(stderr, "Usage: %s [[threads] yMin yMax xMin xMax dxy]\n", argv[0]);
        fprintf(stderr, "Either specify no args, or only threads, or all args.\n");
        return -2;
    }

    // determine amount of threads
    if (argc > 1)
        omp_set_num_threads(atoi(argv[1]));
    else
        omp_set_num_threads(4);

    // set constants if supplied
    if (argc == 7) {
        yMin = atof(argv[2]);
        yMax = atof(argv[3]);
        xMin = atof(argv[4]);
        xMax = atof(argv[5]);
        dxy  = atof(argv[6]);
    }
    
    double time;
    timer_start();
    
    double cx, cy;
    double zx, zy, new_zx;
    unsigned char n;
    int nx, ny;

    // The Mandelbrot calculation is to iterate the equation
    // z = z*z + c, where z and c are complex numbers, z is initially
    // zero, and c is the coordinate of the point being tested. If
    // the magnitude of z remains less than 2 for ever, then the point
    // c is in the Mandelbrot set. We write out the number of iterations
    // before the magnitude of z exceeds 2, or UCHAR_MAX, whichever is
    // smaller.

    nx = 0;
    ny = 0;
    nx = (xMax - xMin) / dxy;
    ny = (yMax - yMin) / dxy;
        
    int i, j;
    unsigned char * buffer = malloc(nx * ny * sizeof(unsigned char));
    if (buffer == NULL) {
      fprintf (stderr, "Couldn't malloc buffer!\n");
      return EXIT_FAILURE;
    }
    
    // do the calculations parallel
    #pragma omp parallel for private(i, j, cx, zx, zy, n, new_zx, cy)
    for (i = 0; i < ny; i++) {
        cy = yMin - dxy + i * dxy;
        for (j = 0; j < nx; j++) {
            cx = xMin - dxy + j * dxy;
            zx = 0.0; 
            zy = 0.0; 
            n = 0;
            
            while ((zx*zx + zy*zy < 4.0) && (n != UCHAR_MAX)) {
                new_zx = zx*zx - zy*zy + cx;
                zy = 2.0*zx*zy + cy;
                zx = new_zx;
                n++;
            }
            buffer[i * nx + j] = n;
        }
    }
    
    time = timer_end();
    
    fprintf (stderr, "Took %g seconds.\nNow writing file...\n", time);
    fwrite(buffer, sizeof(unsigned char), nx * ny, stdout);

    fprintf (stderr, "All done! To process the image: convert -depth 8 -size " \
             "%dx%d gray:output out.jpg\n", nx, ny);
    return 0;
}

