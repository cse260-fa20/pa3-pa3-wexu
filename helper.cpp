/* 
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
// Needed for memalign
#include <malloc.h>
#include "cblock.h"

#ifdef _MPI_
#include <mpi.h>
#endif

using namespace std;

void printMat(const char mesg[], double *E, int m, int n);

//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//

int MASTER = 0;      /* taskid of first task */
int FROM_MASTER = 1; /* setting a message type */
int FROM_WORKER = 2; /* setting a message type */

extern control_block cb;

int BLOCK_M;
int BLOCK_N;
int BLOCK_SIZE;
int PACK_SIZE;

int node_m, node_n;
int p_m, p_n;
int myrank;

int border_node_row, border_node_col;
int node_start_row_ind, node_row_num, node_col_num;

double *buffer_in_west;
double *buffer_in_east;
double *buffer_in_north;
double *buffer_in_south;

double *buffer_out_west;
double *buffer_out_east;
double *buffer_out_north;
double *buffer_out_south;

double *buffer_E;
double *buffer_R;

double *alloc1D(int m, int n)
{
    int nx = n, ny = m;
    double *E;
    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E = (double *)memalign(16, sizeof(double) * nx * ny));
    return (E);
}

void init(double *E, double *E_prev, double *R, int m, int n)
{

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int i, j, source, dest;
    int mtype;

    p_m = cb.py;
    p_n = cb.px;

    int numworkers = p_m * p_n;

    BLOCK_M = cb.m % p_m ? cb.m / p_m + 1 : cb.m / p_m;
    BLOCK_N = cb.n % p_n ? cb.n / p_n + 1 : cb.n / p_n;

    node_m = myrank / p_n;
    node_n = myrank % p_n;

    border_node_row = BLOCK_M - (BLOCK_M * p_m - cb.m);
    border_node_col = BLOCK_N - (BLOCK_N * p_n - cb.n);
    node_start_row_ind = node_m * (n + 2) * BLOCK_M + (n + 2) + 1 + node_n * BLOCK_N;

    if (node_m == p_m - 1)
        node_row_num = border_node_row;
    else
        node_row_num = BLOCK_M;

    if (node_n == p_n - 1)
        node_col_num = border_node_col;
    else
        node_col_num = BLOCK_N;

    BLOCK_SIZE = BLOCK_M * BLOCK_N;
    PACK_SIZE = (BLOCK_M + 2) * (BLOCK_N + 2);

    buffer_in_west = alloc1D(1, node_row_num);
    buffer_in_east = alloc1D(1, node_row_num);
    buffer_in_north = alloc1D(1, node_col_num);
    buffer_in_south = alloc1D(1, node_col_num);

    buffer_out_west = alloc1D(1, node_row_num);
    buffer_out_east = alloc1D(1, node_row_num);
    buffer_out_north = alloc1D(1, node_col_num);
    buffer_out_south = alloc1D(1, node_col_num);

    buffer_E = alloc1D(BLOCK_M, BLOCK_N);
    buffer_R = alloc1D(BLOCK_M, BLOCK_N);

    if (myrank == MASTER)
    {
        for (i = 0; i < (m + 2) * (n + 2); i++)
            E_prev[i] = R[i] = 0;

        for (i = (n + 2); i < (m + 1) * (n + 2); i++)
        {
            int colIndex = i % (n + 2); // gives the base index (first row's) of the current index

            // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
            if (colIndex == 0 || colIndex == (n + 1) || colIndex < ((n + 1) / 2 + 1))
                continue;

            E_prev[i] = 1.0;
        }

        for (i = 0; i < (m + 2) * (n + 2); i++)
        {
            int rowIndex = i / (n + 2); // gives the current row number in 2D array representation
            int colIndex = i % (n + 2); // gives the base index (first row's) of the current index

            // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
            if (colIndex == 0 || colIndex == (n + 1) || rowIndex < ((m + 1) / 2 + 1))
                continue;

            R[i] = 1.0;
        }
    }

    double *buffer_E_prev = alloc1D(BLOCK_M, BLOCK_N);
    double *buffer_R = alloc1D(BLOCK_M, BLOCK_N);
    /*********** Initialization from the master node ***********/
    if (p_m * p_n != 1)
    {
        if (myrank == MASTER)
        {
            for (dest = 1; dest < numworkers; dest++)
            {
                /* Packing and Send*/
                int dest_node_m = dest / p_n;
                int dest_node_n = dest % p_n;

                int dest_node_start_row_ind = dest_node_m * (n + 2) * BLOCK_M + (n + 2) + 1 + dest_node_n * BLOCK_N;
                int dest_node_col_num, dest_node_row_num;

                if (dest_node_m == p_m - 1)
                    dest_node_row_num = border_node_row;
                else
                    dest_node_row_num = BLOCK_M;

                if (dest_node_n == p_n - 1)
                    dest_node_col_num = border_node_col;
                else
                    dest_node_col_num = BLOCK_N;

                for (i = 0; i < dest_node_row_num; i++)
                {
                    for (j = 0; j < dest_node_col_num; j++)
                    {
                        buffer_E_prev[i * dest_node_col_num + j] = E_prev[dest_node_start_row_ind + i * (n + 2) + j];
                        buffer_R[i * dest_node_col_num + j] = R[dest_node_start_row_ind + i * (n + 2) + j];
                    }
                }

                mtype = FROM_MASTER;
                MPI_Send(buffer_E_prev, BLOCK_SIZE, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
                MPI_Send(buffer_R, BLOCK_SIZE, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            }
        }
        else
        {
            mtype = FROM_MASTER;
            MPI_Recv(buffer_E_prev, BLOCK_SIZE, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(buffer_R, BLOCK_SIZE, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (i = 0; i < node_row_num; i++)
            {
                for (j = 0; j < node_col_num; j++)
                {
                    E_prev[node_start_row_ind + i * (n + 2) + j] = buffer_E_prev[i * node_col_num + j];
                    R[node_start_row_ind + i * (n + 2) + j] = buffer_R[i * node_col_num + j];
                }
            }
        }
    }
    /*************************************/

    // We only print the meshes if they are small enough
#if 1
    // printf("\nRank %d After initialization\n", myrank);
    // printMat("E_prev", E_prev, m, n);
    // printMat("R", R, m, n);
#endif
}

void printMat(const char mesg[], double *E, int m, int n)
{
    int i;
#if 0
    if (m>8)
      return;
#else
    if (m > 34)
        return;
#endif
    printf("%s\n", mesg);
    for (i = 0; i < (m + 2) * (n + 2); i++)
    {
        int rowIndex = i / (n + 2);
        int colIndex = i % (n + 2);
        if ((colIndex >= 0) && (colIndex <= n + 1))
            if ((rowIndex >= 0) && (rowIndex <= m + 1))
                printf("%6.3f ", E[i]);
        if (colIndex == n + 1)
            printf("\n");
    }
}
