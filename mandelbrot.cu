// opengl stuff
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// cuda stuff
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// visual studio is annoying stuff
#include "device_launch_parameters.h"

// performs mandelbrot fractal math, taken from in class assignment
__global__ void calculateMandelbrotKernel(const int Nx,
    const int Ny,
    const int Nt,
    const float minx,
    const float miny,
    const float hx,
    const float hy,
    float* __restrict__ h_count) {

    int tx = threadIdx.x; // x-coord in thread block
    int ty = threadIdx.y; // y-coord in thread block

    int bx = blockIdx.x; // x-coord of block
    int by = blockIdx.y; // y-coord of block

    int dx = blockDim.x; // x-dim of thread block
    int dy = blockDim.y; // y-dim of thread block

    int nx = dx * bx + tx; // global x index of thread
    int ny = dy * by + ty; // global y index of thread

    int n = nx + ny * Nx;

    if (nx < Nx && ny < Ny) {
        // Do fractal stuff here
        float cx = minx + nx * hx;
        float cy = miny + ny * hy;
        float zx = 0;
        float zy = 0;

        int t, cnt = 0;

        for (t = 0;t < Nt;++t) {

            // z = z^2 + c
            //   = (zx + i*zy)*(zx + i*zy) + (cx + i*cy)
            //   = zx^2 - zy^2 + 2*i*zy*zx + cx + i*cy
            float zxTmp = zx * zx - zy * zy + cx;
            zy = 2.f * zy * zx + cy;
            zx = zxTmp;

            cnt += (zx * zx + zy * zy < 4.f);
        }

        h_count[n] = cnt;
    }

}

// draws output of our mandelbrot kernel to the cuda surface
__global__ void drawMandelbrotKernel(cudaSurfaceObject_t surface, int Nx, int Ny, int Nt, float* data, float multiplier) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int dx = blockDim.x;
    int dy = blockDim.y;

    int nx = dx * bx + tx;
    int ny = dy * by + ty;

    int n = nx + ny * Nx;

    if (nx < Nx && ny < Ny) {

        float dataf[] = {
            0.0f,
            0.0f,
            0.0f,
            255.0f
        };

        float iterations = data[n];
        if (iterations < Nt) {

            if (iterations < float(Nt) / 3.0) {
                dataf[0] = ((iterations * multiplier) / 200.0f) * 255.0f;
            }
            else if (iterations >= float(Nt) / 3.0 && iterations < 2 * (float(Nt) / 3.0)) {
                dataf[1] = ((iterations * multiplier) / 200.0f) * 255.0f;
            }
            else {
                dataf[2] = ((iterations * multiplier) / 200.0f) * 255.0f;
            }

        }

        uchar4 data = make_uchar4(dataf[0], dataf[1], dataf[2], dataf[3]);
        surf2Dwrite(data, surface, nx * sizeof(data), ny);
    }
}

// wrapper for calling kernel to launch mandelbrot calculation
void calculateMandelbrot(int Nx, int Ny, int Nt, float centx, 
    float centy, float diam, float* c_count) {

    const float minx = centx - 0.5 * diam;
    const float remax = centx + 0.5 * diam;
    const float miny = centy - 0.5 * diam;
    const float immax = centy + 0.5 * diam;

    const float dx = (remax - minx) / (Nx - 1.f);
    const float dy = (immax - miny) / (Ny - 1.f);

    int D = 16;
    dim3 B(D, D);
    dim3 G((Nx + D - 1) / D, (Ny + D - 1) / D);

    calculateMandelbrotKernel << < G, B >> > (Nx, Ny, Nt, minx, miny, dx, dy, c_count);

}

// wrapper for kernel to draw mandelbrot results
void drawMandelbrot(int Nx, int Ny, int Nt, float* c_count, float multiplier, 
    cudaGraphicsResource_t* textureResource, cudaArray_t* textureArray, 
    struct cudaResourceDesc* resourceDesc, cudaSurfaceObject_t surfaceObj) {

    int D = 16;

    dim3 B(D, D);
    dim3 G((Nx + D - 1) / D, (Ny + D - 1) / D);

    cudaGraphicsMapResources(1, textureResource);

    cudaGraphicsSubResourceGetMappedArray(textureArray, *textureResource, 0, 0);

    resourceDesc->res.array.array = *textureArray;
    cudaCreateSurfaceObject(&surfaceObj, resourceDesc);

    drawMandelbrotKernel << < gridDim, blockDim >> > (surfaceObj, Nx, Ny, Nt, c_count, multiplier);

    cudaGraphicsUnmapResources(1, textureResource);

}
