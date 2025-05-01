#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define BMP_HEADER_SIZE 54
#define ALPHA 0.01 // Thermal diffusivity
#define L 0.2      // Length (m) of the square domain
#define DX 0.02    // grid spacing in x-direction
#define DY 0.02    // grid spacing in y-direction
#define DT 0.0005  // Time step
#define T 1500     // Temperature on ºk of the heat source

// Function to print the grid (optional, for debugging or visualization)
void print_grid(double *grid, int nx, int ny)
{
    int i, j;
    for (i = 0; i < nx; i++)
    {
        for (j = 0; j < ny; j++)
        {
            printf("%.2f ", grid[i * ny + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Function to initialize the grid
void initialize_grid(double *grid, int nx, int ny, int temp_source)
{
    int i, j;
#pragma omp parallel for private(i, j) collapse(2)
    for (i = 0; i < nx; i++)
    {
        for (j = 0; j < ny; j++)
        {
            int inyj = i * ny + j;
            if (i == j)
            {
                grid[inyj] = 1500.0;
            }
            else if (i == nx - 1 - j)
            {
                grid[inyj] = 1500.0;
            }
            else
            {
                grid[inyj] = 0.0;
            }
        }
    }
}

void solve_heat_equation(double *grid, double *new_grid,
                         int steps, double r,
                         int nx, int ny,
                         int rank, int size)
{
    int step, i, j;
    double *temp;
    for (step = 0; step < steps; step++)
    {
        if (rank > 0)
            MPI_Sendrecv(&grid[1 * ny], ny, MPI_DOUBLE, rank - 1, 0,
                         &grid[0 * ny], ny, MPI_DOUBLE, rank - 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else
            for (j = 0; j < ny; j++)
                grid[0 * ny + j] = 0.0;
        if (rank < size - 1)
            MPI_Sendrecv(&grid[(nx - 2) * ny], ny, MPI_DOUBLE, rank + 1, 1,
                         &grid[(nx - 1) * ny], ny, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else
            for (j = 0; j < ny; j++)
                grid[(nx - 1) * ny + j] = 0.0;

#pragma omp parallel for private(i, j) collapse(2)
        for (i = 1; i < nx - 1; i++)
        {
            for (j = 1; j < ny - 1; j++)
            {
                int inyj = i * ny + j;
                new_grid[inyj] = grid[inyj] + r * (grid[(i + 1) * ny + j] + grid[(i - 1) * ny + j] - 2 * grid[inyj]) + r * (grid[inyj + 1] + grid[inyj - 1] - 2 * grid[inyj]);
            }
        }

#pragma omp parallel for private(i)
        for (i = 0; i < nx; i++)
        {
            new_grid[0 * ny + i] = 0.0;
            new_grid[ny * (nx - 1) + i] = 0.0;
        }
#pragma omp parallel for private(j)
        for (j = 0; j < ny; j++)
        {
            new_grid[0 + j * nx] = 0.0;
            new_grid[(ny - 1) + j * nx] = 0.0;
        }

        temp = grid;
        grid = new_grid;
        new_grid = temp;
    }
}

// Function to write BMP file header
void write_bmp_header(FILE *file, int width, int height)
{
    unsigned char header[BMP_HEADER_SIZE] = {0};

    int file_size = BMP_HEADER_SIZE + 3 * width * height;
    header[0] = 'B';
    header[1] = 'M';
    header[2] = file_size & 0xFF;
    header[3] = (file_size >> 8) & 0xFF;
    header[4] = (file_size >> 16) & 0xFF;
    header[5] = (file_size >> 24) & 0xFF;
    header[10] = BMP_HEADER_SIZE;

    header[14] = 40; // Info header size
    header[18] = width & 0xFF;
    header[19] = (width >> 8) & 0xFF;
    header[20] = (width >> 16) & 0xFF;
    header[21] = (width >> 24) & 0xFF;
    header[22] = height & 0xFF;
    header[23] = (height >> 8) & 0xFF;
    header[24] = (height >> 16) & 0xFF;
    header[25] = (height >> 24) & 0xFF;
    header[26] = 1;  // Planes
    header[28] = 24; // Bits per pixel

    fwrite(header, 1, BMP_HEADER_SIZE, file);
}

void get_color(double value, unsigned char *r, unsigned char *g, unsigned char *b)
{
    if (value >= 500.0)
    {
        *r = 255;
        *g = 0;
        *b = 0; // Red
    }
    else if (value >= 100.0)
    {
        *r = 255;
        *g = 128;
        *b = 0; // Orange
    }
    else if (value >= 50.0)
    {
        *r = 171;
        *g = 71;
        *b = 188; // Lilac
    }
    else if (value >= 25)
    {
        *r = 255;
        *g = 255;
        *b = 0; // Yellow
    }
    else if (value >= 1)
    {
        *r = 0;
        *g = 0;
        *b = 255; // Blue
    }
    else if (value >= 0.1)
    {
        *r = 5;
        *g = 248;
        *b = 252; // Cyan
    }
    else
    {
        *r = 255;
        *g = 255;
        *b = 255; // white
    }
}

// Function to write the grid matrix into the file
void write_grid(FILE *file, double *grid, int nx, int ny)
{
    int i, j, padding;
    // Write pixel data to BMP file
    for (i = nx - 1; i >= 0; i--)
    { // BMP format stores pixels bottom-to-top
        for (j = 0; j < ny; j++)
        {
            int inyj = i * ny + j;
            unsigned char color[3];
            get_color(grid[inyj], &color[2], &color[1], &color[0]);
            fwrite(color, 1, 3, file); // Write color channel
        }
        // Row padding for 4-byte alignment (if necessary)
        for (padding = 0; padding < (4 - (nx * 3) % 4) % 4; padding++)
        {
            fputc(0, file);
        }
    }
}

// Main function
int main(int argc, char **argv)
{
    double time_begin, time_end;
    char car;
    double r;   // constant of the heat equation
    int nx, ny; // Grid size in x-direction and y-direction
    int steps;  // Number of time steps
    int rank, size;
    // double DT;
    if (argc != 4)
    {
        printf("Command line wrong\n");
        printf("Command line should be: heat_serial size steps name_output_file.bmp. \n");
        printf("Try again!!!!\n");
        return 1;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    nx = ny = atoi(argv[1]);
    r = ALPHA * DT / (DX * DY);
    steps = atoi(argv[2]);
    time_begin = omp_get_wtime();

    // Static descomposition of the grid
    int rows_per_proc = nx / size;
    int local_nx = rows_per_proc;

    // Allocate memory for the grid
    double *grid = NULL;
    double *new_grid = NULL;
    if (rank == 0)
    {
        grid = calloc(nx * ny, sizeof(double));
        new_grid = calloc(nx * ny, sizeof(double));
        initialize_grid(grid, nx, ny, T);
    }

    double *local_grid = (double *)calloc((rows_per_proc + 2) * ny, sizeof(double));
    double *local_new_grid = (double *)calloc((rows_per_proc + 2) * ny, sizeof(double));

    MPI_Scatter(
        grid, rows_per_proc * ny, MPI_DOUBLE,
        &local_grid[ny], rows_per_proc * ny, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // Solve heat equation
    solve_heat_equation(
        local_grid,
        local_new_grid,
        steps,
        r,
        local_nx,
        ny, rank, size);

    MPI_Gather(
        local_grid, rows_per_proc * ny, MPI_DOUBLE,
        grid, rows_per_proc * ny, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // Write grid into a bmp file
        FILE *file = fopen(argv[3], "wb");
        if (!file)
        {
            printf("Error opening the output file.\n");
            return 1;
        }

        write_bmp_header(file, nx, ny);
        write_grid(file, grid, nx, ny);

        fclose(file);
        // Function to visualize the values of the temperature. Use only for debugging
        //  print_grid(grid, nx, ny);
        //  Free allocated memory
        free(grid);
        free(new_grid);
    }

    free(local_grid);
    free(local_new_grid);

    time_end = omp_get_wtime();
    printf("The Execution Time=%fs with a matrix size of %dx%d and %d steps\n", time_end - time_begin, nx, nx, steps);

    MPI_Finalize();
    return 0;
}
