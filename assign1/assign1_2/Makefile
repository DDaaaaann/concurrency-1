CC=gcc
CFLAGS=-c -std=gnu99
LDFLAGS=
SOURCES=forestfire.c malaria.c predator.c
OBJECTS=$(SOURCES:.c=.o)

all: sieve sieve_seq
	
sieve: sieve.o 
	$(CC) $(LDFLAGS) sieve.o -o $@ -lpthread

sieve_seq: sieve_seq.o 
	$(CC) $(LDFLAGS) sieve_seq.o -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@ -lm

clean:
	rm -rf *.o $(EXECUTABLE)
