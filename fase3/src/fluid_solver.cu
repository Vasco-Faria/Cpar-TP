#include "fluid_solver.h"
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x) { float *tmp = x0; x0 = x; x = tmp; }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

void add_source(int M, int N, int O, float *x, float *s, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2);
    for (int i = 0; i < size; i++) {
        x[i] += dt * s[i];
    }
}

__global__ void set_bnd_kernel(int M, int N, int O, int b, float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Bordas em k = 0 e k = O+1
    if (i >= 1 && i <= M && j >= 1 && j <= N) {
        if (k == 0) x[IX(i, j, 0)] = (b == 3) ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
        if (k == O + 1) x[IX(i, j, O + 1)] = (b == 3) ? -x[IX(i, j, O)] : x[IX(i, j, O)];
    }

    // Bordas em i = 0 e i = M+1
    if (j >= 1 && j <= N && k >= 1 && k <= O) {
        if (i == 0) x[IX(0, j, k)] = (b == 1) ? -x[IX(1, j, k)] : x[IX(1, j, k)];
        if (i == M + 1) x[IX(M + 1, j, k)] = (b == 1) ? -x[IX(M, j, k)] : x[IX(M, j, k)];
    }

    // Bordas em j = 0 e j = N+1
    if (i >= 1 && i <= M && k >= 1 && k <= O) {
        if (j == 0) x[IX(i, 0, k)] = (b == 2) ? -x[IX(i, 1, k)] : x[IX(i, 1, k)];
        if (j == N + 1) x[IX(i, N + 1, k)] = (b == 2) ? -x[IX(i, N, k)] : x[IX(i, N, k)];
    }

    // Cálculo explícito dos cantos
    if (i == 0 && j == 0 && k == 0) 
        x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
    if (i == M + 1 && j == 0 && k == 0) 
        x[IX(M + 1, 0, 0)] = 0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
    if (i == 0 && j == N + 1 && k == 0) 
        x[IX(0, N + 1, 0)] = 0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
    if (i == M + 1 && j == N + 1 && k == 0) 
        x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] + x[IX(M + 1, N + 1, 1)]);
}

void set_bnd(int M, int N, int O, int b, float *x) {
    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= M; i++) {
            x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
            x[IX(i, j, O + 1)] = b == 3 ? -x[IX(i, j, O)] : x[IX(i, j, O)];
        }
    }

    for (int j = 1; j <= O; j++) {
        for (int i = 1; i <= N; i++) {
            x[IX(0, i, j)] = b == 1 ? -x[IX(1, i, j)] : x[IX(1, i, j)];
            x[IX(M + 1, i, j)] = b == 1 ? -x[IX(M, i, j)] : x[IX(M, i, j)];
        }
    }

    for (int j = 1; j <= O; j++) {
        for (int i = 1; i <= M; i++) {
            x[IX(i, 0, j)] = b == 2 ? -x[IX(i, 1, j)] : x[IX(i, 1, j)];
            x[IX(i, N + 1, j)] = b == 2 ? -x[IX(i, N, j)] : x[IX(i, N, j)];
        }
    }
  
  x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
  x[IX(M + 1, 0, 0)] = 0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
  x[IX(0, N + 1, 0)] = 0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
  x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] + x[IX(M + 1, N + 1, 1)]);
}

__global__ void lin_solve_red_kernel(int M, int N, int O, int b, float* x, const float* x0, float a, float inv_c, float* max_change) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        if ((i + j + k) % 2 == 1) {  // Red
            int index = IX(i, j, k);
            float old_x = x[index];
            x[index] = (x0[index] +
                        a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                             x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                             x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) * inv_c;
            float change = fabsf(x[index] - old_x);
            atomicMaxFloat(max_change, change);
        }
    }
}

__global__ void lin_solve_black_kernel(int M, int N, int O, int b, float* x, const float* x0, float a, float inv_c, float* max_change) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        if ((i + j + k) % 2 == 0) {  // Black
            int index = IX(i, j, k);
            float old_x = x[index];
            x[index] = (x0[index] +
                        a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                             x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                             x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) * inv_c;
            float change = fabsf(x[index] - old_x);
            atomicMaxFloat(max_change, change);
        }
    }
}

void lin_solve(int M, int N, int O, int b, float* x, const float* x0, float a, float c) {
    float tol = 1e-7f;
    float* d_x = nullptr;
    float* d_x0 = nullptr;
    float* d_max_change = nullptr;
    float max_change;
    int size = ((M + 2) * (N + 2) * (O + 2)) * sizeof(float);

    // Allocate GPU memory
    if (cudaMalloc((void**)&d_x, size) != cudaSuccess) {
        std::cerr << "Error allocating memory for d_x\n";
    }
    if (cudaMalloc((void**)&d_x0, size) != cudaSuccess) {
        std::cerr << "Error allocating memory for d_x0\n";
    }
    if (cudaMalloc((void**)&d_max_change, sizeof(float)) != cudaSuccess) {
        std::cerr << "Error allocating memory for d_max_change\n";
    }

    // Copy data to GPU
    if (cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Error copying x to d_x\n";
    }
    if (cudaMemcpy(d_x0, x0, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Error copying x0 to d_x0\n";
    }

    // Configuração dos kernels
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    float inv_c = 1.0f / c;
    int iterations = 0;

    // Iterar até atingir a tolerância
    do {
        max_change = 0.0f;
        cudaMemcpy(d_max_change, &max_change, sizeof(float), cudaMemcpyHostToDevice);

        // Fase Red
        lin_solve_red_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_x, d_x0, a, inv_c, d_max_change);
        cudaDeviceSynchronize();

        // Fase Black
        lin_solve_black_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_x, d_x0, a, inv_c, d_max_change);
        cudaDeviceSynchronize();

        // Copiar `max_change` de volta para o host
        cudaMemcpy(&max_change, d_max_change, sizeof(float), cudaMemcpyDeviceToHost);

        // Aplicar condições de contorno
        set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_x);
        cudaDeviceSynchronize();

    } while (max_change > tol && ++iterations < 20);

    // Copiar resultados de volta para o host
    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);

    // Liberar memória na GPU
    cudaFree(d_x);
    cudaFree(d_x0);
    cudaFree(d_max_change);
}

void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
    int max = MAX(M, MAX(N, O));
    float a = dt * diff * max * max;
    lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

    for (int k = 1; k <= O; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= M; i++) {
                int index = IX(i, j, k);
                float u_val = u[index];
                float v_val = v[index];
                float w_val = w[index];

                float x = i - dtX * u_val;
                float y = j - dtY * v_val;
                float z = k - dtZ * w_val;

                x = fmaxf(0.5f, fminf(x, M + 0.5f));
                y = fmaxf(0.5f, fminf(y, N + 0.5f));
                z = fmaxf(0.5f, fminf(z, O + 0.5f));

                int i0 = (int)x, i1 = i0 + 1;
                int j0 = (int)y, j1 = j0 + 1;
                int k0 = (int)z, k1 = k0 + 1;

                float s1 = x - i0, s0 = 1 - s1;
                float t1 = y - j0, t0 = 1 - t1;
                float u1 = z - k0, u0 = 1 - u1;

                d[index] = s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
                                 t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
                           s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
                                 t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
            }
        }
    }

    set_bnd(M, N, O, b, d);
}

void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
  // Calculate divergence and initialize pressure field
  float inverso_MNO = 1.0f / (MAX(M, MAX(N, O)));
  for (int k = 1; k <= O; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= M; i++) {
        div[IX(i, j, k)] =
            -0.5f *
            (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] +
             v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)] +
             w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) *
             inverso_MNO;
        p[IX(i, j, k)] = 0;
      }
    }
  }

  set_bnd(M, N, O, 0, div);
  set_bnd(M, N, O, 0, p);
  lin_solve(M, N, O, 0, p, div, 1, 6);

  // Update velocity fields based on pressure
  for (int k = 1; k <= O; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= M; i++) {
        u[IX(i, j, k)] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
        v[IX(i, j, k)] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
        w[IX(i, j, k)] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
      }
    }
  }

  set_bnd(M, N, O, 1, u);
  set_bnd(M, N, O, 2, v);
  set_bnd(M, N, O, 3, w);
}

void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt) {
    add_source(M, N, O, x, x0, dt);
    SWAP(x0, x);
    diffuse(M, N, O, 0, x, x0, diff, dt);
    SWAP(x0, x);
    advect(M, N, O, 0, x, x0, u, v, w, dt);
}

void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0, float *v0, float *w0, float visc, float dt) {
    add_source(M, N, O, u, u0, dt);
    add_source(M, N, O, v, v0, dt);
    add_source(M, N, O, w, w0, dt);
    SWAP(u0, u);
    diffuse(M, N, O, 1, u, u0, visc, dt);
    SWAP(v0, v);
    diffuse(M, N, O, 2, v, v0, visc, dt);
    SWAP(w0, w);
    diffuse(M, N, O, 3, w, w0, visc, dt);
    project(M, N, O, u, v, w, u0, v0);
    SWAP(u0, u);
    SWAP(v0, v);
    SWAP(w0, w);
    advect(M, N, O, 1, u, u0, u0, v0, w0, dt);
    advect(M, N, O, 2, v, v0, u0, v0, w0, dt);
    advect(M, N, O, 3, w, w0, u0, v0, w0, dt);
    project(M, N, O, u, v, w, u0, v0);
}
