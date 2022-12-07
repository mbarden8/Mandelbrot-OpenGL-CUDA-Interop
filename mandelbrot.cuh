#ifndef MANDELBROT_H
#define MANDELBROT_H

void calculateMandelbrot(int Nx, int Ny, int Nt, float centx, float centy, float diam, float* c_count);

void drawMandelbrot(int Nx, int Ny, int Nt, float* c_count, float multiplier, cudaGraphicsResource_t* textureResource, cudaArray_t* textureArray, struct cudaResourceDesc* resourceDesc, cudaSurfaceObject_t surfaceObj);

#endif
