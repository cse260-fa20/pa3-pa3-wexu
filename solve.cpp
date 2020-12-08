/* 
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD
 * 
 */

// #include <assert.h>
#include <stdlib.h>
// #include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h>
// #include "helper.cpp"

#ifdef _MPI_
#include <mpi.h>
#endif

using namespace std;

int m, n;

const int NORTH = 0;
const int EAST = 1;
const int SOUTH = 2;
const int WEST = 3;

extern int BLOCK_M, BLOCK_N, BLOCK_SIZE, PACK_SIZE;
extern control_block cb;

extern int p_m, p_n, node_m, node_n;
extern int myrank;

extern double *buffer_in_west;
extern double *buffer_in_east;
extern double *buffer_in_north;
extern double *buffer_in_south;

extern double *buffer_out_west;
extern double *buffer_out_east;
extern double *buffer_out_north;
extern double *buffer_out_south;

extern double *buffer_E;
extern double *buffer_R;

extern int MASTER;
extern int FROM_MASTER;
extern int FROM_WORKER;

void repNorms(double l2norm, double mx, double dt, int m, int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);

#ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
__attribute__((optimize("no-tree-vectorize")))
#endif

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double
L2Norm(double sumSq)
{
    double l2norm = sumSq / (double)((cb.m) * (cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

void exchange_value(double *E_prev)
{
    int pack_start_ind;

    /*  MPI variables  */
    int mtype, /* message type */
        op_count = 0,
        i, j; /* misc */

    MPI_Status status[8];
    MPI_Request reqs[8];

    /* Exchange data with neibors */
    // West
    if (node_n == 0)
    {
        pack_start_ind = node_m * (n + 2) * BLOCK_M + (n + 2) + node_n * BLOCK_N;
        for (i = 0; i < BLOCK_M; i++)
        {
            E_prev[pack_start_ind] = E_prev[pack_start_ind + 2];
            pack_start_ind += n + 2;
        }
    }
    else
    {
        mtype = FROM_WORKER;
        MPI_Irecv(buffer_in_west, BLOCK_M, MPI_DOUBLE, myrank - 1, mtype, MPI_COMM_WORLD, reqs + op_count);

        pack_start_ind = node_m * (n + 2) * BLOCK_M + (n + 2) + 1 + node_n * BLOCK_N;
        for (i = 0; i < BLOCK_M; i++)
        {
            buffer_out_west[i] = E_prev[pack_start_ind];
            pack_start_ind += n + 2;
        }

        mtype = FROM_WORKER;
        MPI_Isend(buffer_out_west, BLOCK_M, MPI_DOUBLE, myrank - 1, mtype, MPI_COMM_WORLD, reqs + op_count + 1);

        op_count += 2;
    }

    // East
    if (node_n == p_n - 1)
    {
        pack_start_ind = node_m * (n + 2) * BLOCK_M + 2 * (n + 2) - 1 + node_n * BLOCK_N;
        for (i = 0; i < BLOCK_M; i++)
        {
            E_prev[pack_start_ind] = E_prev[pack_start_ind - 2];
            pack_start_ind += n + 2;
        }
    }
    else
    {
        mtype = FROM_WORKER;
        MPI_Irecv(buffer_in_east, BLOCK_M, MPI_DOUBLE, myrank + 1, mtype, MPI_COMM_WORLD, reqs + op_count);

        pack_start_ind = node_m * (n + 2) * BLOCK_M + (n + 2) - 1 + BLOCK_N + node_n * BLOCK_N;
        for (i = 0; i < BLOCK_M; i++)
        {
            buffer_out_east[i] = E_prev[pack_start_ind];
            pack_start_ind += n + 2;
        }

        mtype = FROM_WORKER;
        MPI_Isend(buffer_out_east, BLOCK_M, MPI_DOUBLE, myrank + 1, mtype, MPI_COMM_WORLD, reqs + op_count + 1);

        op_count += 2;
    }

    // North
    if (node_m == 0)
    {
        pack_start_ind = node_m * (n + 2) * BLOCK_M + 1 + node_n * BLOCK_N;
        for (i = 0; i < BLOCK_N; i++)
        {
            E_prev[pack_start_ind] = E_prev[pack_start_ind + 2 * (n + 2)];
            pack_start_ind += 1;
        }
    }
    else
    {
        mtype = FROM_WORKER;
        pack_start_ind = node_m * (n + 2) * BLOCK_M + 1 + node_n * BLOCK_N;
        MPI_Irecv(E_prev + pack_start_ind, BLOCK_N, MPI_DOUBLE, myrank - p_n, mtype, MPI_COMM_WORLD, reqs + op_count);

        mtype = FROM_WORKER;
        pack_start_ind = node_m * (n + 2) * BLOCK_M + (n + 2) + 1 + node_n * BLOCK_N;
        MPI_Isend(E_prev + pack_start_ind, BLOCK_N, MPI_DOUBLE, myrank - p_n, mtype, MPI_COMM_WORLD, reqs + op_count + 1);

        op_count += 2;
    }

    // South
    if (node_m == p_m - 1)
    {
        pack_start_ind = (m + 1) * (n + 2) + 1 + node_n * BLOCK_N;
        for (i = 0; i < BLOCK_N; i++)
        {
            E_prev[pack_start_ind] = E_prev[pack_start_ind - 2 * (n + 2)];
            pack_start_ind += 1;
        }
    }
    else
    {
        mtype = FROM_WORKER;
        pack_start_ind = node_m * (n + 2) * BLOCK_M + BLOCK_M * (n + 2) + 1 + node_n * BLOCK_N;
        MPI_Irecv(E_prev + pack_start_ind, BLOCK_N, MPI_DOUBLE, myrank + p_n, mtype, MPI_COMM_WORLD, reqs + op_count);

        mtype = FROM_WORKER;
        pack_start_ind = node_m * (n + 2) * BLOCK_M + (BLOCK_M - 1) * (n + 2) + 1 + node_n * BLOCK_N;
        MPI_Isend(E_prev + pack_start_ind, BLOCK_N, MPI_DOUBLE, myrank + p_n, mtype, MPI_COMM_WORLD, reqs + op_count + 1);

        op_count += 2;
    }

    // Sync
    MPI_Waitall(op_count, reqs, status);

    // Unpack West
    if (node_n != 0)
    {
        // pack_start_ind = BLOCK_N + 2;
        pack_start_ind = node_m * (n + 2) * BLOCK_M + (n + 2) + node_n * BLOCK_N;
        for (i = 0; i < BLOCK_M; i++)
        {
            E_prev[pack_start_ind] = buffer_in_west[i];
            pack_start_ind += n + 2;
        }
    }

    // Unpack East
    if (node_n != p_n - 1)
    {
        pack_start_ind = node_m * (n + 2) * BLOCK_M + (n + 2) + BLOCK_N + node_n * BLOCK_N;
        for (i = 0; i < BLOCK_M; i++)
        {
            E_prev[pack_start_ind] = buffer_in_east[i];
            pack_start_ind += n + 2;
        }
    }
}

void gather_result(double *E, double *R)
{
    /*  MPI variables  */
    int mtype, /* message type */
        dest, source,
        op_count = 0,
        numworkers = p_m * p_n,
        i, j; /* misc */

    /* Send results back to the master node */

    int start_ind;
    if (myrank == MASTER)
    {
        mtype = FROM_WORKER;
        for (source = 1; source < numworkers; source++)
        {
            MPI_Recv(buffer_E, BLOCK_SIZE, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(buffer_R, BLOCK_SIZE, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int src_node_m = source / p_n;
            int src_node_n = source % p_n;

            start_ind = src_node_m * (n + 2) * BLOCK_M + (n + 2) + 1 + src_node_n * BLOCK_N;
            // printf("\n\nGather results: Rank %d,\tind = %d,\tn=%d\n", myrank, start_ind, n);

            for (i = 0; i < BLOCK_M; i++)
            {
                for (j = 0; j < BLOCK_N; j++)
                {
                    E[start_ind + i * (n + 2) + j] = buffer_E[i * BLOCK_N + j];
                    R[start_ind + i * (n + 2) + j] = buffer_R[i * BLOCK_N + j];
                }
            }
        }
    }
    else
    {
        // Slave nodes
        start_ind = node_m * (n + 2) * BLOCK_M + (n + 2) + 1 + node_n * BLOCK_N;
        for (i = 0; i < BLOCK_M; i++)
        {
            // start_ind = (i + 1) * (BLOCK_N + 2) + 1;
            for (j = 0; j < BLOCK_N; j++)
            {
                buffer_E[i * BLOCK_N + j] = E[start_ind + i * (n + 2) + j];
                buffer_R[i * BLOCK_N + j] = R[start_ind + i * (n + 2) + j];
            }
        }

        dest = MASTER;
        mtype = FROM_WORKER;
        MPI_Send(buffer_E, BLOCK_SIZE, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
        MPI_Send(buffer_R, BLOCK_SIZE, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
    }
}

void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf)
{
    // Simulated time is different from the integer timestep number
    double t = 0.0;

    double *E = *_E, *E_prev = *_E_prev;
    double *R_tmp = R;
    double *E_tmp = *_E;
    double *E_prev_tmp = *_E_prev;
    double mx, sumSq;
    int niter;

    m = cb.m, n = cb.n;

    int innerBlockRowStartIndex = node_m * (n + 2) * BLOCK_M + (n + 2) + 1 + node_n * BLOCK_N;
    int innerBlockRowEndIndex = innerBlockRowStartIndex + (BLOCK_M - 1) * (n + 2);

    /*  MPI variables  */
    int dest,  /* task id of message destination */
        mtype, /* message type */
        i, j;  /* misc */

    // We continue to sweep over the mesh until the simulation has reached
    // the desired number of iterations
    for (niter = 0; niter < cb.niters; niter++)
    {
        // if (cb.debug && (niter == 0))
        if (cb.stats_freq && (niter == 0))
        {
            stats(E_prev, m, n, &mx, &sumSq);
            double l2norm = L2Norm(sumSq);
            repNorms(l2norm, mx, dt, m, n, -1, cb.stats_freq);
            if (cb.plot_freq)
                plotter->updatePlot(E, -1, m + 1, n + 1);

            // printf("\nRank %d at Iter %d\n", myrank, niter);
            // printMat2("Mat E_prev0", E_prev, m, n);
        }

        /* 
    * Copy data from boundary of the computational box to the
    * padding region, set up for differencing computational box's boundary
    *
    * These are physical boundary conditions, and are not to be confused
    * with ghost cells that we would use in an MPI implementation
    *
    * The reason why we copy boundary conditions is to avoid
    * computing single sided differences at the boundaries
    * which increase the running time of solve()
    *
    */

        // 4 FOR LOOPS set up the padding needed for the boundary conditions
        // Fills in the TOP Ghost Cells

        if (!cb.noComm)
            exchange_value(E_prev);

//////////////////////////////////////////////////////////////////////////////
#define OPT 1
#define FUSED 1

#ifdef FUSED

#ifndef OPT
        // Solve for the excitation, a PDE
        for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += (n + 2))
        {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            R_tmp = R + j;
            for (i = 0; i < BLOCK_N; i++)
            {
                E_tmp[i] = E_prev_tmp[i] + alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] + E_prev_tmp[i + (n + 2)] + E_prev_tmp[i - (n + 2)]);
                E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
                R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) * (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
            }
        }

#else
        // Solve for the excitation, a PDE
        for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += (n + 2))
        {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            R_tmp = R + j;
            for (i = 0; i < BLOCK_N; i++)
            {
                register double tmp_E_prev = E_prev_tmp[i];
                register double tmp_E_prev_left = E_prev_tmp[i - 1];
                register double tmp_E_prev_right = E_prev_tmp[i + 1];
                register double tmp_E_prev_up = E_prev_tmp[i - (n + 2)];
                register double tmp_E_prev_down = E_prev_tmp[i + (n + 2)];
                register double tmp_R = R_tmp[i];
                register double tmp_E = E_tmp[i];

                tmp_E = tmp_E_prev + alpha * (tmp_E_prev_right + tmp_E_prev_left - 4 * tmp_E_prev + tmp_E_prev_down + tmp_E_prev_up);
                E_tmp[i] = tmp_E - dt * (kk * tmp_E_prev * (tmp_E_prev - a) * (tmp_E_prev - 1) + tmp_E_prev * tmp_R);
                R_tmp[i] = tmp_R + dt * (epsilon + M1 * tmp_R / (tmp_E_prev + M2)) * (-tmp_R - kk * tmp_E_prev * (tmp_E_prev - b - 1));
            }
        }
#endif

#else
        // Solve for the excitation, a PDE
        for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += (BLOCK_N + 2))
        {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            for (i = 0; i < BLOCK_N; i++)
            {
                E_tmp[i] = E_prev_tmp[i] + alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] + E_prev_tmp[i + (BLOCK_N + 2)] + E_prev_tmp[i - (BLOCK_N + 2)]);
            }
        }

        /*
                 * Solve the ODE, advancing excitation and recovery variables
                 *     to the next timtestep
                 */

        for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += (BLOCK_N + 2))
        {
            E_tmp = E + j;
            R_tmp = R + j;
            E_prev_tmp = E_prev + j;
            for (i = 0; i < BLOCK_N; i++)
            {
                E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
                R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) * (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
            }
        }
#endif
        /////////////////////////////////////////////////////////////////////////////////

        if (cb.stats_freq)
        {
            if (!(niter % cb.stats_freq))
            {
                stats(E, m, n, &mx, &sumSq);
                double l2norm = L2Norm(sumSq);
                repNorms(l2norm, mx, dt, m, n, niter, cb.stats_freq);

                // printf("\nRank %d at Iter %d\n", myrank, niter);
                // printMat2("Mat E_prev1", E_prev, m, n);
            }
        }

        if (cb.plot_freq)
        {
            if (!(niter % cb.plot_freq))
            {
                plotter->updatePlot(E, niter, m, n);
            }
        }

        // Swap current and previous meshes
        double *tmp = E;
        E = E_prev;
        E_prev = tmp;

    } //end of 'niter' loop at the beginning

    if (p_m * p_n != 1 && !cb.noComm)
        gather_result(E_prev, R);

    // if (myrank == MASTER)
    // printMat2("\n\nRank 0 Final matrix E_prev\n\n", E_prev, m, n); // return the L2 and infinity norms via in-out parameters

    stats(E_prev, m, n, &Linf, &sumSq);
    L2 = L2Norm(sumSq);

    // Swap pointers so we can re-use the arrays
    *_E = E;
    *_E_prev = E_prev;
}

void printMat2(const char mesg[], double *E, int m, int n)
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
