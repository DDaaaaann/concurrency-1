/*******************************************************************************
 * 
 * sieve_seq.c
 * 
 * by Ben Witzen
 * University of Amsterdam, 05-11-2012
 * 
 * Implements the Sieve of Eratosthenes sequentially.
 * http://en.wikipedia.org/wiki/Sieve_of_Eratosthenes 
 * 
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>

long long n = 2;            // the first number main will feed to sieve
long long sieves[10000000]; // contains all the sieves
long long amt = 0;          // amount of sieves found so far


/*
 * This checks if IN is prima by consulting the sieve array. If IN is prime,
 * returns IN and adds IN to sieve array. If IN is not prime, returns -1.
 */

long long sieve(long long in) {
  // checks in the sieve if this number is prime
  for (int i = 0; i < amt; i++)
    if (in % sieves[i] == 0)
      return -1;
  // not found in sieve, this number must be prime, add it to sieve...
  sieves[amt] = in;
  amt++;
  // ...and signal main to print it
  return in; 
}


/*
 * Feeds sieve its numbers (from 2 to "infinite"). Also prints to terminal.
 * Note that it never returns.
 */

int main(int argc, char ** argv) {
  while (1) {    
    if (sieve(n) > 0)
      printf("%lld\n", n);
    n++;
  }
  // this never happens...
  return 0;
}

