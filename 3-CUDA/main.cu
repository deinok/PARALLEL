#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BMP_HEADER_SIZE 54
#define ALPHA 0.01 // Thermal diffusivity
#define L 0.2      // Length (m) of the square domain
#define DX 0.02    // grid spacing in x-direction
#define DY 0.02    // grid spacing in y-direction
#define DT 0.0005  // Time step
#define T 1500     // Temperature on ºk of the heat source

// Function to initialize the grid
__global__ void initialize_grid(double *grid, int nx, int ny, int temp_source)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nx && j < ny)
    {
        int inyj = i * ny + j;
        if (i == j || i == nx - 1 - j)
        {
            grid[inyj] = 1500.0;
        }
        else
        {
            grid[inyj] = 0.0;
        }
    }
}

__global__ void solve_heat_equation(double *grid, double *new_grid, double r, int nx, int ny)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1)
    {
        int inyj = i * ny + j;
        new_grid[inyj] = grid[inyj] + r * (grid[(i + 1) * ny + j] + grid[(i - 1) * ny + j] - 2 * grid[inyj]) + r * (grid[i * ny + (j + 1)] + grid[i * ny + (j - 1)] - 2 * grid[inyj]);
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

__global__ void apply_boundary_conditions(double *grid, int nx, int ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Top and bottom rows
    if (idx < ny)
    {
        grid[idx] = 0.0;                       // Top row
        grid[(nx - 1) * ny + idx] = 0.0;       // Bottom row
    }

    // Left and right columns
    if (idx < nx)
    {
        grid[idx * ny] = 0.0;                  // Left column
        grid[idx * ny + (ny - 1)] = 0.0;       // Right column
    }
}

// Main function
int main(int argc, char **argv)
{

    double r;   // constant of the heat equation
    int nx, ny; // Grid size in x-direction and y-direction
    int steps;  // Number of time steps

    if (argc != 4)
    {
        printf("Command line wrong\n");
        printf("Command line should be: heat_serial size steps name_output_file.bmp. \n");
        printf("Try again!!!!\n");
        return 1;
    }
    nx = ny = atoi(argv[1]);
    r = ALPHA * DT / (DX * DY);
    steps = atoi(argv[2]);

    size_t size = nx * ny * sizeof(double);
    double *grid, *new_grid;
    cudaMallocManaged(&grid, size);
    cudaMallocManaged(&new_grid, size);

    // Grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((ny + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (nx + threadsPerBlock.y - 1) / threadsPerBlock.y);

    dim3 oneDimBlocks((nx > ny ? nx : ny + 255) / 256);

    // Initialize grid
    initialize_grid<<<numBlocks, threadsPerBlock>>>(grid, nx, ny, T);
    cudaDeviceSynchronize();

    // Time steps
    for (int step = 0; step < steps; step++)
    {
        solve_heat_equation<<<numBlocks, threadsPerBlock>>>(grid, new_grid, r, nx, ny);
        cudaDeviceSynchronize();

        apply_boundary_conditions<<<oneDimBlocks, 256>>>(new_grid, nx, ny);
        cudaDeviceSynchronize();

        // Swap pointers
        double *temp = grid;
        grid = new_grid;
        new_grid = temp;
    }

    // Write BMP file
    FILE *file = fopen(argv[3], "wb");
    if (!file)
    {
        printf("Error opening file\n");
        return 1;
    }

    write_bmp_header(file, nx, ny);
    write_grid(file, grid, nx, ny);
    fclose(file);

    cudaFree(grid);
    cudaFree(new_grid);

    printf("Simulation complete! Output written to %s\n", argv[3]);
    return 0;
}
