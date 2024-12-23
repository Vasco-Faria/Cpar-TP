#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cuda.h>

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x) { float *tmp = x0; x0 = x; x = tmp; }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CUDA_CALL(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error in file '%s' in line %i: %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Function to check CUDA version
void check_cuda_version() {
    int driverVersion = 0;
    int runtimeVersion = 0;

    // Check CUDA driver version
    CUDA_CALL(cudaDriverGetVersion(&driverVersion));

    // Check CUDA runtime version
    CUDA_CALL(cudaRuntimeGetVersion(&runtimeVersion));

    if (runtimeVersion > driverVersion) {
        printf("CUDA runtime version (%d) is higher than the installed CUDA driver version (%d).\n", runtimeVersion, driverVersion);
        printf("Please update your CUDA driver.\n");
        exit(EXIT_FAILURE);
    }
}

// Device function to compute atomic max
__device__ float atomicMaxFloat(float *addr, float value) {
    float old = *addr;
    if (old >= value) return old;
    unsigned int *p = (unsigned int *)addr;
    unsigned int old_bits, new_bits;
    do {
        old_bits = __float_as_uint(old);
        new_bits = __float_as_uint(value);
        old_bits = atomicCAS(p, old_bits, new_bits);
        old = __uint_as_float(old_bits);
    } while (old < value);
    return old;
}

__global__ void add_source_kernel(int M, int N, int O, float *x, float *s, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = (M + 2) * (N + 2) * (O + 2);
    if (idx < size) {
        x[idx] += dt * s[idx];
    }
}

void add_source(int M, int N, int O, float *x, float *s, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2);
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    add_source_kernel<<<numBlocks, blockSize>>>(M, N, O, x, s, dt);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
}

__global__ void set_bnd_kernel(int M, int N, int O, int b, float *x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = (M + 2) * (N + 2) * (O + 2);
    int i, j, k;

    if (idx < size) {
        k = idx / ((M + 2) * (N + 2));
        j = (idx % ((M + 2) * (N + 2))) / (M + 2);
        i = idx % (M + 2);

        if (i == 0 || i == M + 1 || j == 0 || j == N + 1 || k == 0 || k == O + 1) {
            if (i > 0 && i <= M && j > 0 && j <= N) {
                if (k == 0 || k == O + 1) x[IX(i, j, k)] = b == 3 ? -x[IX(i, j, k == 0 ? 1 : O)] : x[IX(i, j, k == 0 ? 1 : O)];
            }
            if (i > 0 && i <= M && k > 0 && k <= O) {
                if (j == 0 || j == N + 1) x[IX(i, j, k)] = b == 2 ? -x[IX(i, j == 0 ? 1 : N, k)] : x[IX(i, j == 0 ? 1 : N, k)];
            }
            if (j > 0 && j <= N && k > 0 && k <= O) {
                if (i == 0 || i == M + 1) x[IX(i, j, k)] = b == 1 ? -x[IX(i == 0 ? 1 : M, j, k)] : x[IX(i == 0 ? 1 : M, j, k)];
            }
        }
    }
}

void set_bnd(int M, int N, int O, int b, float *x) {
    int size = (M + 2) * (N + 2) * (O + 2);
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    set_bnd_kernel<<<numBlocks, blockSize>>>(M, N, O, b, x);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
    x[IX(M + 1, 0, 0)] = 0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
    x[IX(0, N + 1, 0)] = 0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
    x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] + x[IX(M + 1, N + 1, 1)]);
}

__global__ void lin_solve_kernel(int M, int N, int O, int b, float *x, float *x0, float a, float c, float *max_c, bool red_black) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= 1 && i <= M && j >= 1 && j <= N && k >= 1 && k <= O) {
        int idx = IX(i, j, k);
        int sum = i + j + k;

        if ((sum % 2 == 1) == red_black) {
            float old_x = x[idx];
            x[idx] = (x0[idx] + a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] + x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] + x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) / c;
            float change = fabs(x[idx] - old_x);
            atomicMaxFloat(max_c, change);
        }
    }
}

void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    float tol = 1e-7, max_c;
    int l = 0;

    float *d_max_c;
    CUDA_CALL(cudaMalloc((void**)&d_max_c, sizeof(float)));

    dim3 blockSize(8, 8, 8);
    dim3 numBlocks((M + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y, (O + blockSize.z - 1) / blockSize.z);

    do {
        max_c = 0.0f;
        CUDA_CALL(cudaMemcpy(d_max_c, &max_c, sizeof(float), cudaMemcpyHostToDevice));

        lin_solve_kernel<<<numBlocks, blockSize>>>(M, N, O, b, x, x0, a, c, d_max_c, true);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        lin_solve_kernel<<<numBlocks, blockSize>>>(M, N, O, b, x, x0, a, c, d_max_c, false);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaMemcpy(&max_c, d_max_c, sizeof(float), cudaMemcpyDeviceToHost));
        set_bnd(M, N, O, b, x);

    } while (max_c > tol && ++l < 20);

    CUDA_CALL(cudaFree(d_max_c));
}

void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
    int max = MAX(M, MAX(N, O));
    float a = dt * diff * max * max;
    lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

__global__ void advect_kernel(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= 1 && i <= M && j >= 1 && j <= N && k >= 1 && k <= O) {
        int idx = IX(i, j, k);
        float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

        float x = i - dtX * u[idx];
        float y = j - dtY * v[idx];
        float z = k - dtZ * w[idx];

        x = fmaxf(0.5f, fminf(x, M + 0.5f));
        y = fmaxf(0.5f, fminf(y, N + 0.5f));
        z = fmaxf(0.5f, fminf(z, O + 0.5f));

        int i0 = (int)x, i1 = i0 + 1;
        int j0 = (int)y, j1 = j0 + 1;
        int k0 = (int)z, k1 = k0 + 1;

        float s1 = x - i0, s0 = 1 - s1;
        float t1 = y - j0, t0 = 1 - t1;
        float u1 = z - k0, u0 = 1 - u1;

        d[idx] = s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
                       t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
                 s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
                       t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
    }
}

void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    dim3 blockSize(8, 8, 8);
    dim3 numBlocks((M + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y, (O + blockSize.z - 1) / blockSize.z);

    advect_kernel<<<numBlocks, blockSize>>>(M, N, O, b, d, d0, u, v, w, dt);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    set_bnd(M, N, O, b, d);
}

__global__ void project_step1_kernel(int M, int N, int O, float *u, float *v, float *w, float *p, float *div, float inverso_MNO) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= 1 && i <= M && j >= 1 && j <= N && k >= 1 && k <= O) {
        int idx = IX(i, j, k);
        div[idx] = -0.5f * (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] +
                            v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)] +
                            w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) * inverso_MNO;
        p[idx] = 0;
    }
}

__global__ void project_step2_kernel(int M, int N, int O, float *u, float *v, float *w, float *p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= 1 && i <= M && j >= 1 && j <= N && k >= 1 && k <= O) {
        int idx = IX(i, j, k);
        u[idx] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
        v[idx] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
        w[idx] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
    }
}

void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
    float inverso_MNO = 1.0f / MAX(M, MAX(N, O));

    dim3 blockSize(8, 8, 8);
    dim3 numBlocks((M + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y, (O + blockSize.z - 1) / blockSize.z);

    project_step1_kernel<<<numBlocks, blockSize>>>(M, N, O, u, v, w, p, div, inverso_MNO);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    set_bnd(M, N, O, 0, div);
    set_bnd(M, N, O, 0, p);

    lin_solve(M, N, O, 0, p, div, 1, 6);

    project_step2_kernel<<<numBlocks, blockSize>>>(M, N, O, u, v, w, p);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
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
    check_cuda_version();
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
