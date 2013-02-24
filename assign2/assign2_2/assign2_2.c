/******************************************************************************
 *
 * Concurrency and paralel programming
 * assignment 2.2
 *
 * by Ben Witzen and David van Schoorisse
 * University of Amsterdam, 15-11-2012
 *
 * Implements the MYMPI_Bcast function and a simple main to test it with.
 * The Bcast function allows one node (root) to send its buffer to each other
 * node. We assume a ring structure; the root node first sends its buffer to
 * its left and right neighbours, and then the nodes keep forwarding the
 * buffer until it has reached the node(s) with the greatest distance to the
 * root. This method guarantees that each node has received the buffer.
 *
 * Usage: ./assign2_2 [ROOT]
 *
 * About the testing interface (main): Each node receives a buffer containg one
 * INT matching their rank. The ROOT argument determines which node will broad-
 * cast, each other node will listen. The result is thus that each node will
 * end up with a buffer matching the rank id of the sender, ROOT.
 *
 * ROOT will default to 0 if not provided. ROOT should be less than the amount
 * of processes.
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


/*
 * Broadcast function. Depending on who invokes it, performs different actions.
 * If called by ROOT, sends message to left and right neighbor if able. If
 * called by node(s) furthest away from ROOT, waits for incoming messages. If
 * called by intermediate nodes, forwards messages it receives.
 *
 * In each case, ROOT's buffer is copied to the caller's buffer.
 */

int MYMPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
                MPI_Comm communicator) {
    // retrieve rank and amt
    int tag = 55;
    int rank, amt;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &amt);

    // nobody to broadcast to, abort
    if (amt == 1)
        return -2;

    // determine neighbors
    int right = (rank + 1) % amt;
    int left  = (rank - 1);
    if (left < 0)
        left = amt - 1;

    // ROOT CODE
    if (rank == root) {
        // send to right
        MPI_Send(buffer, count, datatype, right, tag, communicator);

        // if there's more than one other node, also send to left
        if (right != left)
            MPI_Send(buffer, count, datatype, left, tag, communicator);
    }

    // NONROOT CODE
    else {
        MPI_Status status;
        status.MPI_SOURCE = amt;

        // catch a message from left or right neighbor
        do {
            MPI_Probe(MPI_ANY_SOURCE, tag, communicator, &status);
        } while (status.MPI_SOURCE != left && status.MPI_SOURCE != right);
        MPI_Recv(buffer, count, datatype, status.MPI_SOURCE, tag,
                 communicator, &status);

        // if I'm the only node that's the furthest away
        if ( ( amt / 2 + root ) % amt == rank && amt / 2 == 0 ) {
            // I'm expecting another message; catch it
            do {
                MPI_Probe(MPI_ANY_SOURCE, tag, communicator, &status);
            } while (status.MPI_SOURCE != left && status.MPI_SOURCE != right);
            MPI_Recv(buffer, count, datatype, left, tag, communicator,
                     MPI_STATUS_IGNORE);
            
        }
        // if I'm an intermediate node
        else if ( ( amt / 2 + root ) % amt != rank ) {
            // forward message (from left to right, or from right to left)
            if (status.MPI_SOURCE == left)
                MPI_Send(buffer, count, datatype, right, tag, communicator);
            else
                MPI_Send(buffer, count, datatype, left, tag, communicator);
        }
    }
    return 0;
}


/*
 * Simple main to test MYMPI_Bcast with. See program description for usage
 * details.
 */

int main(int argc, char *argv[]) {
    // start MPI
    int rank, amt;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &amt);

    // abort if arguments invalid
    if (argc > 2) {
        if (rank == 0)
            printf("Usage: ./assign2_2 [root]\n");
        MPI_Finalize();
        return -2;
    }

    // default root is 0 unless set by argument
    int root = 0;
    if (argc == 2)
        root = atoi(argv[1]);

    // abort if chosen root is invalid
    if (root >= amt) {
        if (rank == 0)
            printf("Error: Root should be less than amount of processes.\n");
        MPI_Finalize();
        return -2;
    }

    // set buffer content
    int x = rank;

    // what value do we start with
    printf("Hi, my rank is %d and my start value is %d.\n", rank, x);

    // call broadcast
    MYMPI_Bcast(&x, 1, MPI_INT, root, MPI_COMM_WORLD);

    // what value did we get back
    printf("Hi, my rank is %d and I ended up with value %d.\n", rank, x);

    // exit
    MPI_Finalize();
    return 0;
}

