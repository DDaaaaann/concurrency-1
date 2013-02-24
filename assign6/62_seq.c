/*******************************************************************************
 *
 * Concurrency and parallel programming
 * Assignment 6.2
 *
 * David van Schoorisse and Ben Witzen
 * University of Amsterdam, 07-12-12
 *
 * Can determine the maximum of a given float array. Reads this array from a
 * file, one float per line. Program can also generate float arrays to test
 * with.
 *
 * Usage:
 * -max filename        returns maximum value in a given float array file
 * -gen filename amt    generates a float array file containing amt floats
 *
 ******************************************************************************/

#define MAX_FLOAT_LEN 32  // maximum float length, in characters
                          // affects both the -max and -gen operations

#include <float.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


/*
 * Function finds the maximum value in a given float array. Returns the amount
 * of elements read upon success, or -1 upon failure.
 */

int findMaximum (char * argv) {

  // open file
  FILE * file = fopen(argv, "r");
  if (file == NULL)
    return false;
  
  char * buffer = malloc(MAX_FLOAT_LEN * sizeof(char));
  int cnt = 0;
  float tmp, max;
  bool seperator = false, firstval = true;
  
  // read file, char by char
  for (int c = fgetc(file); c != EOF; c = fgetc(file)) {
    seperator = false;
  
    if (c == '.' || c == ',') {
      if (seperator)
        return false;
      buffer[cnt++] = '.';
      seperator = true;      
    }
    
    else if (c >= '0' && c <= '9' && cnt < MAX_FLOAT_LEN - 1)
      buffer[cnt++] = c;
    
    else if (c == '\n') {
      // fix buffer
      buffer[cnt] = '\0';
      cnt = 0;
      // get value
      tmp = atof(buffer);
      // check if biggest
      if (tmp > max || firstval) {
        max = tmp;
        firstval = false;
      }
    }
    
    else
      return false;

  }
  
  fclose(file);
  printf("The maximum value in that there array is: %lf\n", max);
  return true;

}

/*
 * Function creates a file containing a float array. Generates amt random floats
 * in range [0, 1). Returns amount of floats generated upon success, or -1 upon
 * failure.
 */
 
int generateFile (char * argv, int amt) {

  float min = 0.0, max = 1.0;
  float tmp, tmp2;

  if (amt < 1 || argv == NULL)
    return false;

  FILE * file = fopen(argv, "w");
  if (file == NULL)
    return false;

  // set seed
  srand((unsigned int) time(NULL));

  for (int i = 0; i < amt; i++) {
    tmp = min + (max - min) * rand() / (float)RAND_MAX;
    fprintf(file, "%f\n", tmp);
    if (tmp > tmp2 || i == 0) {
      tmp2 = tmp;
    }
  }

  printf("Done. Biggest float written = %f\n", tmp2);

}


int main (int argc, char ** argv) {

  if (argc == 3 && strcmp(argv[1], "-max") == 0) {
    if (findMaximum(argv[2]))
      return EXIT_SUCCESS;
    else
      printf("Something went wrong... :(\n");
  }
  else if (argc == 4 && strcmp(argv[1], "-gen") == 0) {
    if (generateFile(argv[2], atoi(argv[3])))
      return EXIT_SUCCESS;
    else
      printf("Something went wrong... :(\n");
  }
  else {
    printf("Usage:\n");
    printf("-max filename     | prints maximum value of a given float array\n");
    printf("-gen filename amt | generates a test file containing amt floats\n");
    return EXIT_FAILURE;
  }


}
