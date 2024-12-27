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

__global__ void add_source_kernel(int size, float *x, const float *s, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        x[i] += dt * s[i];
    }
}

void add_source(int M, int N, int O, float *x, float *s, float dt) {
    int size = ((M + 2) * (N + 2) * (O + 2)) * sizeof(float);

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Alocar memória no dispositivo
    float *d_x, *d_s;
    CUDA_CHECK(cudaMalloc(&d_x, size));
    CUDA_CHECK(cudaMalloc(&d_s, size));

    // Copiar dados para o dispositivo
    CUDA_CHECK(cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s, s, size, cudaMemcpyHostToDevice));

    // Executar o kernel
    add_source_kernel<<<numBlocks, threadsPerBlock>>>(size, d_x, d_s, dt);

    // Verificar erros no kernel
    CUDA_CHECK(cudaPeekAtLastError());

    // Copiar resultados de volta para o host
    CUDA_CHECK(cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost));

    // Liberar memória no dispositivo
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_s));
}

__global__ void set_bnd_kernel(int M, int N, int O, int b, float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= 1 && i <= M && j >= 1 && j <= N && k >= 1 && k <= O) {
        // Border conditions in k = 0 and k = O+1
        if (k == 0) x[IX(i, j, 0)] = (b == 3) ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
        if (k == O + 1) x[IX(i, j, O + 1)] = (b == 3) ? -x[IX(i, j, O)] : x[IX(i, j, O)];

        // Border conditions in i = 0 and i = M+1
        if (i == 0) x[IX(0, j, k)] = (b == 1) ? -x[IX(1, j, k)] : x[IX(1, j, k)];
        if (i == M + 1) x[IX(M + 1, j, k)] = (b == 1) ? -x[IX(M, j, k)] : x[IX(M, j, k)];

        // Border conditions in j = 0 and j = N+1
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
    int size = ((M + 2) * (N + 2) * (O + 2)) * sizeof(float);
    float* d_x;

    cudaMalloc(&d_x, size);
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_x);
    cudaDeviceSynchronize();

    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
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

__global__ void advect_kernel(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt, float dtX, float dtY, float dtZ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
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

        // Interpolation
        d[index] = s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
                         t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
                   s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
                         t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
    }
}

void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    int size = ((M + 2) * (N + 2) * (O + 2)) * sizeof(float);
    float *d_d, *d_d0, *d_u, *d_v, *d_w;

    checkCudaError(cudaMalloc((void**)&d_d, size), "advect: cudaMalloc d_d");
    checkCudaError(cudaMalloc((void**)&d_d0, size), "advect: cudaMalloc d_d0");
    checkCudaError(cudaMalloc((void**)&d_u, size), "advect: cudaMalloc d_u");
    checkCudaError(cudaMalloc((void**)&d_v, size), "advect: cudaMalloc d_v");
    checkCudaError(cudaMalloc((void**)&d_w, size), "advect: cudaMalloc d_w");

    // Copiar dados para a GPU
    checkCudaError(cudaMemcpy(d_d, d, size, cudaMemcpyHostToDevice), "advect: cudaMemcpy d_d");
    checkCudaError(cudaMemcpy(d_d0, d0, size, cudaMemcpyHostToDevice), "advect: cudaMemcpy d_d0");
    checkCudaError(cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice), "advect: cudaMemcpy d_u");
    checkCudaError(cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice), "advect: cudaMemcpy d_v");
    checkCudaError(cudaMemcpy(d_w, w, size, cudaMemcpyHostToDevice), "advect: cudaMemcpy d_w");

    // pre-compute dt * M, dt * N, dt * O
    float dtX = dt * M;
    float dtY = dt * N;
    float dtZ = dt * O;

    advect_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_d, d_d0, d_u, d_v, d_w, dt, dtX, dtY, dtZ);
    checkCudaError(cudaDeviceSynchronize(), "advect: cudaDeviceSynchronize after advect_kernel launch");
    checkCudaError(cudaGetLastError(), "advect: cudaGetLastError after advect_kernel launch");

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_d);
    checkCudaError(cudaDeviceSynchronize(), "advect: cudaDeviceSynchronize after set_bnd_kernel launch");
    checkCudaError(cudaGetLastError(), "advect: cudaGetLastError after set_bnd_kernel launch");

    // Copiar resultados de volta para a CPU
    checkCudaError(cudaMemcpy(d, d_d, size, cudaMemcpyDeviceToHost), "advect: cudaMemcpy d");

    // Libertar memória da GPU
    checkCudaError(cudaFree(d_d), "advect: cudaFree d_d");
    checkCudaError(cudaFree(d_d0), "advect: cudaFree d_d0");
    checkCudaError(cudaFree(d_u), "advect: cudaFree d_u");
    checkCudaError(cudaFree(d_v), "advect: cudaFree d_v");
    checkCudaError(cudaFree(d_w), "advect: cudaFree d_w");
}

__global__ void compute_div_and_init_p(int M, int N, int O, float *u, float *v, float *w, float *p, float *div, float inverso_MNO) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        int idx = IX(i, j, k);
        int idx_im1 = IX(i - 1, j, k);
        int idx_ip1 = IX(i + 1, j, k);
        int idx_jm1 = IX(i, j - 1, k);
        int idx_jp1 = IX(i, j + 1, k);
        int idx_km1 = IX(i, j, k - 1);
        int idx_kp1 = IX(i, j, k + 1);

        if (idx >= 0 && idx < (M + 2) * (N + 2) * (O + 2) &&
            idx_im1 >= 0 && idx_im1 < (M + 2) * (N + 2) * (O + 2) &&
            idx_ip1 >= 0 && idx_ip1 < (M + 2) * (N + 2) * (O + 2) &&
            idx_jm1 >= 0 && idx_jm1 < (M + 2) * (N + 2) * (O + 2) &&
            idx_jp1 >= 0 && idx_jp1 < (M + 2) * (N + 2) * (O + 2) &&
            idx_km1 >= 0 && idx_km1 < (M + 2) * (N + 2) * (O + 2) &&
            idx_kp1 >= 0 && idx_kp1 < (M + 2) * (N + 2) * (O + 2)) {
            div[idx] =
                -0.5f *
                (u[idx_ip1] - u[idx_im1] +
                 v[idx_jp1] - v[idx_jm1] +
                 w[idx_kp1] - w[idx_km1]) *
                 inverso_MNO;
            p[idx] = 0;
        }
    }
}

__global__ void update_velocities(int M, int N, int O, float *u, float *v, float *w, float *p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        int idx = IX(i, j, k);
        int idx_im1 = IX(i - 1, j, k);
        int idx_ip1 = IX(i + 1, j, k);
        int idx_jm1 = IX(i, j - 1, k);
        int idx_jp1 = IX(i, j + 1, k);
        int idx_km1 = IX(i, j, k - 1);
        int idx_kp1 = IX(i, j, k + 1);

        if (idx >= 0 && idx < (M + 2) * (N + 2) * (O + 2) &&
            idx_im1 >= 0 && idx_im1 < (M + 2) * (N + 2) * (O + 2) &&
            idx_ip1 >= 0 && idx_ip1 < (M + 2) * (N + 2) * (O + 2) &&
            idx_jm1 >= 0 && idx_jm1 < (M + 2) * (N + 2) * (O + 2) &&
            idx_jp1 >= 0 && idx_jp1 < (M + 2) * (N + 2) * (O + 2) &&
            idx_km1 >= 0 && idx_km1 < (M + 2) * (N + 2) * (O + 2) &&
            idx_kp1 >= 0 && idx_kp1 < (M + 2) * (N + 2) * (O + 2)) {
            u[idx] -= 0.5f * (p[idx_ip1] - p[idx_im1]);
            v[idx] -= 0.5f * (p[idx_jp1] - p[idx_jm1]);
            w[idx] -= 0.5f * (p[idx_kp1] - p[idx_km1]);
        }
    }
}

void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
    float inverso_MNO = 1.0f / (MAX(M, MAX(N, O)));
    int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);
    float *d_u, *d_v, *d_w, *d_p, *d_div;

    checkCudaError(cudaMalloc((void**)&d_u, size), "project: cudaMalloc d_u");
    checkCudaError(cudaMalloc((void**)&d_v, size), "project: cudaMalloc d_v");
    checkCudaError(cudaMalloc((void**)&d_w, size), "project: cudaMalloc d_w");
    checkCudaError(cudaMalloc((void**)&d_p, size), "project: cudaMalloc d_p");
    checkCudaError(cudaMalloc((void**)&d_div, size), "project: cudaMalloc d_div");

    // Copiar dados para a GPU
    checkCudaError(cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice), "project: cudaMemcpy d_u");
    checkCudaError(cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice), "project: cudaMemcpy d_v");
    checkCudaError(cudaMemcpy(d_w, w, size, cudaMemcpyHostToDevice), "project: cudaMemcpy d_w");
    checkCudaError(cudaMemcpy(d_p, p, size, cudaMemcpyHostToDevice), "project: cudaMemcpy d_p");
    checkCudaError(cudaMemcpy(d_div, div, size, cudaMemcpyHostToDevice), "project: cudaMemcpy d_div");

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    compute_div_and_init_p<<<numBlocks, threadsPerBlock>>>(M, N, O, d_u, d_v, d_w, d_p, d_div, inverso_MNO);
    checkCudaError(cudaDeviceSynchronize(), "project: cudaDeviceSynchronize after compute_div_and_init_p kernel launch");
    checkCudaError(cudaGetLastError(), "project: cudaGetLastError after compute_div_and_init_p kernel launch");

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 0, d_div);
    checkCudaError(cudaDeviceSynchronize(), "project: cudaDeviceSynchronize after set_bnd_kernel launch");
    checkCudaError(cudaGetLastError(), "project: cudaGetLastError after set_bnd_kernel launch");

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 0, d_p);
    checkCudaError(cudaDeviceSynchronize(), "project: cudaDeviceSynchronize after set_bnd_kernel launch");
    checkCudaError(cudaGetLastError(), "project: cudaGetLastError after set_bnd_kernel launch");

    update_velocities<<<numBlocks, threadsPerBlock>>>(M, N, O, d_u, d_v, d_w, d_p);
    checkCudaError(cudaDeviceSynchronize(), "project: cudaDeviceSynchronize after update_velocities kernel launch");
    checkCudaError(cudaGetLastError(), "project: cudaGetLastError after update_velocities kernel launch");

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 1, d_u);
    checkCudaError(cudaDeviceSynchronize(), "project: cudaDeviceSynchronize after set_bnd_kernel launch");
    checkCudaError(cudaGetLastError(), "project: cudaGetLastError after set_bnd_kernel launch");

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 2, d_v);
    checkCudaError(cudaDeviceSynchronize(), "project: cudaDeviceSynchronize after set_bnd_kernel launch");
    checkCudaError(cudaGetLastError(), "project: cudaGetLastError after set_bnd_kernel launch");

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 3, d_w);
    checkCudaError(cudaDeviceSynchronize(), "project: cudaDeviceSynchronize after set_bnd_kernel launch");
    checkCudaError(cudaGetLastError(), "project: cudaGetLastError after set_bnd_kernel launch");

    // Copiar resultados de volta para a CPU
    checkCudaError(cudaMemcpy(u, d_u, size, cudaMemcpyDeviceToHost), "project: cudaMemcpy u");
    checkCudaError(cudaMemcpy(v, d_v, size, cudaMemcpyDeviceToHost), "project: cudaMemcpy v");
    checkCudaError(cudaMemcpy(w, d_w, size, cudaMemcpyDeviceToHost), "project: cudaMemcpy w");
    checkCudaError(cudaMemcpy(p, d_p, size, cudaMemcpyDeviceToHost), "project: cudaMemcpy p");
    checkCudaError(cudaMemcpy(div, d_div, size, cudaMemcpyDeviceToHost), "project: cudaMemcpy div");

    // Libertar memória da GPU
    checkCudaError(cudaFree(d_u), "project: cudaFree d_u");
    checkCudaError(cudaFree(d_v), "project: cudaFree d_v");
    checkCudaError(cudaFree(d_w), "project: cudaFree d_w");
    checkCudaError(cudaFree(d_p), "project: cudaFree d_p");
    checkCudaError(cudaFree(d_div), "project: cudaFree d_div");
    checkCudaError(cudaFree(d_max_change), "project: cudaFree d_max_change");
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
