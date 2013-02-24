/*******************************************************************************
 * 
 * Concurrency and parallel programming - Assignment 1.2
 * sieve.c
 * 
 * by Ben Witzen and David van Schoorisse
 * University of Amsterdam, 07-11-2012
 * 
 * Implements the Sieve of Eratosthenes using pthreads. Generates an "infinite"
 * amount of prime numbers. Program must be terminated using Ctrl+C.
 * 
 ******************************************************************************/

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

#define BUFFER_SIZE 256

typedef struct buffer {
  long long       array[BUFFER_SIZE];
  int             cnt, in, out;
  pthread_mutex_t lock;
} buffer;

buffer * newBuffer(void) {
  buffer * new = calloc(1, sizeof(buffer));
  pthread_mutex_init(&new->lock, NULL);
  return new;
} 


/*
 * Function attempts to add VAL to array of buffer B, thread safe. Returns the
 * array index it was added into upon success, or -1 upon failure.
 */

int addToBuffer(long long val, buffer * b) {
  int rv = -1;
  pthread_mutex_lock(&b->lock);
  
  //Check if there's space left in buffer
  if(b->cnt < BUFFER_SIZE) {
    rv = b->in;
    b->array[b->in] = val;
    b->in = (b->in + 1) % BUFFER_SIZE;
    b->cnt++;
  }
  
  pthread_mutex_unlock(&b->lock);
  return rv; 
}


/*
 * Function attemps to remove the first added element from the array of buffer
 * B, thread safe. Returns that value upon success, or -1 upon failure.
 */

long long takeFromBuffer(buffer * b) {
  long long rv = -1;
  pthread_mutex_lock(&b->lock);
  
  if(b->cnt > 0){
    rv = b->array[b->out];
    b->out = (b->out + 1) % BUFFER_SIZE;
    b->cnt--;
  }
  
  pthread_mutex_unlock(&b->lock);
  return rv;
  
}


/*
 * Thread function that reads from an IN-buffer (provided as argument B) and
 * sieves potentially prime numbers. The first number it reads from the IN
 * buffer is taken as this function's sieve. All numbers that arive afterwards
 * are checked for divisibility by this function's sieve. It pipes numbers that
 * are not divisible to an OUT-buffer, and discards all other numbers.
 */

void * sieve(void * b) {
  pthread_t thread_id;
  long long s = -1;
  long long n;
  long long a = -1;
  buffer * in = (buffer * )b;
  buffer * out = newBuffer();

  // set this function's sieve
  while (s == -1)
    s = takeFromBuffer(in);

  // create a new thread to feed the OUT-buffer to
  if (pthread_create(&thread_id, NULL, &sieve, (void *)out))
    printf("Couldnt make thread :(\n");    
  
  // print value of this sieve
  printf("%lld\n", s);
  
  // initiate (and never stop) sieving
  while(1) {
    n = takeFromBuffer(in);
    if (n != -1 && n % s != 0) {
      while(a == -1) {
        a = addToBuffer(n, out);
      }
      a = -1;
    }
  }
  // this keeps compilers happy
  return NULL;
}


/*
 * Creates a sieve thread, then starts feeding that thread integer numbers
 * starting with 2. It will "never stop" doing this.
 */

int main(void) {
  buffer * out = newBuffer();
  
  long long n = 2;
  int a;
  
  // create a new sieve thread
  pthread_t thread_id;
  pthread_create(&thread_id, NULL, &sieve, (void *)out);
  
  // add n to buffer, once succesful, increase n and repeat
  while (1) {
    a = -1;
    while (a == -1)
      a = addToBuffer(n, out);
    n++;
  }
  
  // this keeps compilers happy
  return 0;
}
