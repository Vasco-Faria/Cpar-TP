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

static int size;

// variaveis kernels
// set_bnd 
float* d_x = nullptr;
// lin_solve
float* d_xx = nullptr;
float* d_x0 = nullptr;
float* d_max_change = nullptr;
// advect
float* d_d = nullptr;
float* d_d0 = nullptr;
float* d_u = nullptr;
float* d_v = nullptr;
float* d_w = nullptr;
// project
float* d_uu = nullptr;
float* d_vv = nullptr;
float* d_ww = nullptr;
float* d_pp = nullptr;
float* d_div = nullptr;

void init_cuda_mallocs(int M, int N, int O) {
    size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);
    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_xx, size);
    cudaMalloc((void**)&d_x0, size);
    cudaMalloc((void**)&d_max_change, size);
    cudaMalloc((void**)&d_d, size);
    cudaMalloc((void**)&d_d0, size);
    cudaMalloc((void**)&d_u, size);
    cudaMalloc((void**)&d_v, size);
    cudaMalloc((void**)&d_w, size);
    cudaMalloc((void**)&d_uu, size);
    cudaMalloc((void**)&d_vv, size);
    cudaMalloc((void**)&d_ww, size);
    cudaMalloc((void**)&d_pp, size);
    cudaMalloc((void**)&d_div, size);
}

void free_cuda_mallocs() {
    cudaFree(d_x);
    cudaFree(d_xx);
    cudaFree(d_x0);
    cudaFree(d_max_change);
    cudaFree(d_d);
    cudaFree(d_d0);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_uu);
    cudaFree(d_vv);
    cudaFree(d_ww);
    cudaFree(d_pp);
    cudaFree(d_div);
}

void add_source(int M, int N, int O, float *x, float *s, float dt) {
    int size_ = (M + 2) * (N + 2) * (O + 2);
    for (int i = 0; i < size_; i++) {
        x[i] += dt * s[i];
    }
}

__global__ void set_bnd_kernel(int M, int N, int O, int b, float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= 1 && i <= M && j >= 1 && j <= N && k >= 1 && k <= O) {
        // k = 0 and k = O+1
        if (k == 0) x[IX(i, j, 0)] = (b == 3) ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
        if (k == O + 1) x[IX(i, j, O + 1)] = (b == 3) ? -x[IX(i, j, O)] : x[IX(i, j, O)];

        // i = 0 and i = M+1
        if (i == 0) x[IX(0, j, k)] = (b == 1) ? -x[IX(1, j, k)] : x[IX(1, j, k)];
        if (i == M + 1) x[IX(M + 1, j, k)] = (b == 1) ? -x[IX(M, j, k)] : x[IX(M, j, k)];

        // j = 0 and j = N+1
        if (j == 0) x[IX(i, 0, k)] = (b == 2) ? -x[IX(i, 1, k)] : x[IX(i, 1, k)];
        if (j == N + 1) x[IX(i, N + 1, k)] = (b == 2) ? -x[IX(i, N, k)] : x[IX(i, N, k)];
    }
    
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
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_x);
    cudaDeviceSynchronize();

    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
}

__global__ void lin_solve_red_kernel(int M, int N, int O, int b, float* x, const float* x0, float a, float inv_c, float* max_change) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        if ((i + j + k) % 2 == 1) { 
            int index = IX(i, j, k);
            float old_x = x[index];
            x[index] = (x0[index] +
                        a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                             x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                             x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) * inv_c;
            float change = fabsf(x[index] - old_x);
            if (change > *max_change) *max_change = change;
        }
    }
}

__global__ void lin_solve_black_kernel(int M, int N, int O, int b, float* x, const float* x0, float a, float inv_c, float* max_change) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        if ((i + j + k) % 2 == 0) { 
            int index = IX(i, j, k);
            float old_x = x[index];
            x[index] = (x0[index] +
                        a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                             x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                             x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) * inv_c;
            float change = fabsf(x[index] - old_x);
            if (change > *max_change) *max_change = change;
        }
    }
}

void lin_solve(int M, int N, int O, int b, float* x, const float* x0, float a, float c) {
    float tol = 1e-7f;
    float max_change;

    cudaMemcpy(d_xx, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x0, x0, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    float inv_c = 1.0f / c;
    int iterations = 0;

    do {
        max_change = 0.0f;
        cudaMemcpy(d_max_change, &max_change, sizeof(float), cudaMemcpyHostToDevice);

        lin_solve_red_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_xx, d_x0, a, inv_c, d_max_change);
        cudaDeviceSynchronize();

        lin_solve_black_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_xx, d_x0, a, inv_c, d_max_change);
        cudaDeviceSynchronize();

        cudaMemcpy(&max_change, d_max_change, sizeof(float), cudaMemcpyDeviceToHost);

        set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_xx);
        cudaDeviceSynchronize();

    } while (max_change > tol && ++iterations < 20);

    cudaMemcpy(x, d_xx, size, cudaMemcpyDeviceToHost);
}

void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
    int max = MAX(M, MAX(N, O));
    float a = dt * diff * max * max;
    lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

__global__ void advect_kernel(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // +1 para evitar bordas
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > M || j > N || k > O) return;

    int index = IX(i, j, k);
    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

    float u_val = u[index], v_val = v[index], w_val = w[index];
    float x = i - dtX * u_val, y = j - dtY * v_val, z = k - dtZ * w_val;

    x = (x < 0.5f) ? 0.5f : (x > M + 0.5f) ? M + 0.5f : x;
    y = (y < 0.5f) ? 0.5f : (y > N + 0.5f) ? N + 0.5f : y;
    z = (z < 0.5f) ? 0.5f : (z > O + 0.5f) ? O + 0.5f : z;

    int i0 = (int)x, i1 = i0 + 1, j0 = (int)y, j1 = j0 + 1, k0 = (int)z, k1 = k0 + 1;
    float s1 = x - i0, s0 = 1 - s1, t1 = y - j0, t0 = 1 - t1, u1 = z - k0, u0 = 1 - u1;

    d[index] = 
        s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) + 
              t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
        s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) + 
              t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
}

void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    cudaMemcpy(d_d, d, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d0, d0, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    advect_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_d, d_d0, d_u, d_v, d_w, dt);
    cudaDeviceSynchronize();

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_d);
    cudaDeviceSynchronize();

    cudaMemcpy(d, d_d, size, cudaMemcpyDeviceToHost);
}

__global__ void compute_div_and_init_p(int M, int N, int O, float *u, float *v, float *w, float *p, float *div, float inverso_MNO) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        int index = IX(i, j, k);
        div[index] = -0.5f * (
            u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] +
            v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)] +
            w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]
        ) * inverso_MNO;

        p[index] = 0.0f;
    }
}

__global__ void update_velocities(int M, int N, int O, float *u, float *v, float *w, float *p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        int index = IX(i, j, k);
        u[index] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
        v[index] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
        w[index] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
    }
}

void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
    float inverso_MNO = 1.0f / (MAX(M, MAX(N, O)));

    cudaMemcpy(d_uu, u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vv, v, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ww, w, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pp, p, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_div, div, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    compute_div_and_init_p<<<numBlocks, threadsPerBlock>>>(M, N, O, d_uu, d_vv, d_ww, d_pp, d_div, inverso_MNO);
    cudaDeviceSynchronize();

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 0, d_div);
    cudaDeviceSynchronize();

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 0, d_pp);
    cudaDeviceSynchronize();

    lin_solve(M, N, O, 0, d_pp, d_div, 1 ,6);

    update_velocities<<<numBlocks, threadsPerBlock>>>(M, N, O, d_uu, d_vv, d_ww, d_pp);
    cudaDeviceSynchronize();

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 1, d_uu);
    cudaDeviceSynchronize();

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 2, d_vv);
    cudaDeviceSynchronize();

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 3, d_ww);
    cudaDeviceSynchronize();

    cudaMemcpy(u, d_uu, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(v, d_vv, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(w, d_ww, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(p, d_pp, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(div, d_div, size, cudaMemcpyDeviceToHost);
}

void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt) {
    add_source(M, N, O, x, x0, dt);
    SWAP(x0, x);
    diffuse(M, N, O, 0, x, x0, diff, dt);
    SWAP(x0, x);
    advect(M, N, O, 0, x, x0, u, v, w, dt);
    free_cuda_mallocs();
}

void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0, float *v0, float *w0, float visc, float dt) {
    init_cuda_mallocs(M, N, O);
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
