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

float* max_change = nullptr;

float* v_u = nullptr;
float* v_v = nullptr;
float* v_w = nullptr;
float* v_u0 = nullptr;
float* v_v0 = nullptr;
float* v_w0 = nullptr;

float* d_x = nullptr;
float* d_x0 = nullptr;

void init_cuda_mallocs_vel(int M, int N, int O, float* u, float* v, float* w, float* u0, float* v0, float* w0) {
    size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);

    cudaMallocManaged((void**)&max_change, size);

    cudaMallocManaged((void**)&v_u, size);
    cudaMallocManaged((void**)&v_v, size);
    cudaMallocManaged((void**)&v_w, size);
    cudaMallocManaged((void**)&v_u0, size);
    cudaMallocManaged((void**)&v_v0, size);
    cudaMallocManaged((void**)&v_w0, size);

    cudaMemcpy(v_u, u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(v_v, v, size, cudaMemcpyHostToDevice);
    cudaMemcpy(v_w, w, size, cudaMemcpyHostToDevice);
    cudaMemcpy(v_u0, u0, size, cudaMemcpyHostToDevice);
    cudaMemcpy(v_v0, v0, size, cudaMemcpyHostToDevice);
    cudaMemcpy(v_w0, w0, size, cudaMemcpyHostToDevice);
}

void free_cuda_mallocs_vel() {
    cudaFree(v_u0);
    cudaFree(v_v0);
    cudaFree(v_w0);
}

void init_cuda_mallocs_dens(float* x, float* x0) {
    cudaMallocManaged((void**)&d_x, size);
    cudaMallocManaged((void**)&d_x0, size);

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x0, x0, size, cudaMemcpyHostToDevice);
}

void free_cuda_mallocs_dens() {
    cudaFree(max_change);

    cudaFree(v_u);
    cudaFree(v_v);
    cudaFree(v_w);

    cudaFree(d_x0);
}

__global__ void add_source_kernel(int M, int N, int O, float *x, float *s, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int size_ = (M + 2) * (N + 2) * (O + 2);
    if (i < size_) x[i] += dt * s[i];
}

void add_source(int M, int N, int O, float *x, float *s, float dt) {
    int threadsPerBlock = 256;
    int size_ = (M + 2) * (N + 2) * (O + 2);
    int numBlocks = (size_ + threadsPerBlock - 1) / threadsPerBlock;
    add_source_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, x, s, dt);
    cudaDeviceSynchronize();
}

__global__ void set_bnd_kernel(int M, int N, int O, int b, float* x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > M + 1 || j > N + 1 || k > O + 1) return;

    if (k == 0)
        x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
    else if (k == O + 1)
        x[IX(i, j, O + 1)] = b == 3 ? -x[IX(i, j, O)] : x[IX(i, j, O)];
    
    if (i == 0)
        x[IX(0, j, k)] = b == 1 ? -x[IX(1, j, k)] : x[IX(1, j, k)];
    else if (i == M + 1)
        x[IX(M + 1, j, k)] = b == 1 ? -x[IX(M, j, k)] : x[IX(M, j, k)];
    
    if (j == 0)
        x[IX(i, 0, k)] = b == 2 ? -x[IX(i, 1, k)] : x[IX(i, 1, k)];
    else if (j == N + 1)
        x[IX(i, N + 1, k)] = b == 2 ? -x[IX(i, N, k)] : x[IX(i, N, k)];

    if (k == 0) {
        if (i == 0 && j == 0)
            x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
        else if (i == M + 1 && j == 0)
            x[IX(M + 1, 0, 0)] = 0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
        else if (i == 0 && j == N + 1)
            x[IX(0, N + 1, 0)] = 0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
        else if (i == M + 1 && j == N + 1)
            x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] + x[IX(M + 1, N + 1, 1)]);
    }
}

__global__ void lin_solve_red_kernel(int M, int N, int O, int b, float* x, const float* x0, float a, float inv_c, float* max_change) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > M || j > N || k > O) return;

    if ((i + j + k) % 2 != 1) return;

    int index = IX(i, j, k);
    float old_x = x[index];
    x[index] = (x0[index] +
                a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                        x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                        x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) * inv_c;
    float change = fabsf(x[index] - old_x);
    if (change > *max_change) *max_change = change;
}

__global__ void lin_solve_black_kernel(int M, int N, int O, int b, float* x, const float* x0, float a, float inv_c, float* max_change) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > M || j > N || k > O) return;

    if ((i + j + k) % 2 != 0) return;

    int index = IX(i, j, k);
    float old_x = x[index];
    x[index] = (x0[index] +
                a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                        x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                        x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) * inv_c;
    float change = fabsf(x[index] - old_x);
    if (change > *max_change) *max_change = change;
}

void lin_solve(int M, int N, int O, int b, float* x, const float* x0, float a, float c) {
    float tol = 1e-7f;

    dim3 threadsPerBlock(64, 8, 2);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    float inv_c = 1.0f / c;
    int iterations = 0;
    
    do {
        *max_change = 0.0f;

        lin_solve_red_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, x, x0, a, inv_c, max_change);
        cudaDeviceSynchronize();

        lin_solve_black_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, x, x0, a, inv_c, max_change);
        cudaDeviceSynchronize();

        set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, x);
        cudaDeviceSynchronize();

    } while (*max_change > tol && ++iterations < 20);
}

void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
    int max = MAX(M, MAX(N, O));
    float a = dt * diff * max * max;
    lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

__global__ void advect_kernel(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > M || j > N || k > O) return;

    int index = IX(i, j, k);
    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

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

    int i0_j0_k0 = IX(i0, j0, k0);
    int i0_j0_k1 = IX(i0, j0, k1);
    int i0_j1_k0 = IX(i0, j1, k0);
    int i0_j1_k1 = IX(i0, j1, k1);
    int i1_j0_k0 = IX(i1, j0, k0);
    int i1_j0_k1 = IX(i1, j0, k1);
    int i1_j1_k0 = IX(i1, j1, k0);
    int i1_j1_k1 = IX(i1, j1, k1);

    d[index] = s0 * (t0 * (u0 * d0[i0_j0_k0] + u1 * d0[i0_j0_k1]) + t1 * (u0 * d0[i0_j1_k0] + u1 * d0[i0_j1_k1])) +
               s1 * (t0 * (u0 * d0[i1_j0_k0] + u1 * d0[i1_j0_k1]) + t1 * (u0 * d0[i1_j1_k0] + u1 * d0[i1_j1_k1]));
}

void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    dim3 threadsPerBlock(64, 8, 2);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    advect_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d, d0, u, v, w, dt);
    cudaDeviceSynchronize();

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d);
    cudaDeviceSynchronize();
}

__global__ void compute_div_and_init_p(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
    float inverso_MNO = 1.0f / (MAX(M, MAX(N, O)));

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > M || j > N || k > O) return;

    int index = IX(i, j, k);
    div[index] = -0.5f * (
        u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] +
        v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)] +
        w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]
    ) * inverso_MNO;

    p[index] = 0.0f;
}

__global__ void update_velocities(int M, int N, int O, float *u, float *v, float *w, float *p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > M || j > N || k > O) return;

    int index = IX(i, j, k);
    u[index] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
    v[index] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
    w[index] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
}

void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
    dim3 threadsPerBlock(64, 8, 2);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    compute_div_and_init_p<<<numBlocks, threadsPerBlock>>>(M, N, O, u, v, w, p, div);
    cudaDeviceSynchronize();

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 0, div);
    cudaDeviceSynchronize();

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 0, p);
    cudaDeviceSynchronize();

    lin_solve(M, N, O, 0, p, div, 1 ,6);

    update_velocities<<<numBlocks, threadsPerBlock>>>(M, N, O, u, v, w, p);
    cudaDeviceSynchronize();

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 1, u);
    cudaDeviceSynchronize();

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 2, v);

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 3, w);
    cudaDeviceSynchronize();
}

void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt) {
    add_source(M, N, O, d_x, d_x0, dt);
    SWAP(d_x0, d_x);
    diffuse(M, N, O, 0, d_x, d_x0, diff, dt);
    SWAP(d_x0, d_x);
    advect(M, N, O, 0, d_x, d_x0, v_u, v_v, v_w, dt);
}

void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0, float *v0, float *w0, float visc, float dt) {
    add_source(M, N, O, v_u, v_u0, dt);
    add_source(M, N, O, v_v, v_v0, dt);
    add_source(M, N, O, v_w, v_w0, dt);
    SWAP(v_u0, v_u);
    diffuse(M, N, O, 1, v_u, v_u0, visc, dt);
    SWAP(v_v0, v_v);
    diffuse(M, N, O, 2, v_v, v_v0, visc, dt);
    SWAP(v_w0, v_w);
    diffuse(M, N, O, 3, v_w, v_w0, visc, dt);
    project(M, N, O, v_u, v_v, v_w, v_u0, v_v0);
    SWAP(v_u0, v_u);
    SWAP(v_v0, v_v);
    SWAP(v_w0, v_w);
    advect(M, N, O, 1, v_u, v_u0, v_u0, v_v0, v_w0, dt);
    advect(M, N, O, 2, v_v, v_v0, v_u0, v_v0, v_w0, dt);
    advect(M, N, O, 3, v_w, v_w0, v_u0, v_v0, v_w0, dt);
    project(M, N, O, v_u, v_v, v_w, v_u0, v_v0);
}
