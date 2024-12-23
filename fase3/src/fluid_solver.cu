#include "fluid_solver.h"
#include <cmath>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x) { float *tmp = x0; x0 = x; x = tmp; }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

__global__ void add_source_kernel(int M, int N, int O, float *x, float *s, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] += dt * s[idx];
    }
}

void add_source(int M, int N, int O, float *x, float *s, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2);
    int blockSize = 256;  // Number of threads per block
    int numBlocks = (size + blockSize - 1) / blockSize;
    add_source_kernel<<<numBlocks, blockSize>>>(M, N, O, x, s, dt);
    cudaDeviceSynchronize();
}


__global__ void set_bnd_kernel(int M, int N, int O, int b, float *x) {
    int i = threadIdx.x + 1;
    int j = threadIdx.y + 1;
    int k = threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        if (k == 0) {
            x[IX(i, j, 0)] = (b == 3) ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
        }
        if (k == O + 1) {
            x[IX(i, j, O + 1)] = (b == 3) ? -x[IX(i, j, O)] : x[IX(i, j, O)];
        }
        if (j == 0) {
            x[IX(i, 0, k)] = (b == 2) ? -x[IX(i, 1, k)] : x[IX(i, 1, k)];
        }
        if (j == N + 1) {
            x[IX(i, N + 1, k)] = (b == 2) ? -x[IX(i, N, k)] : x[IX(i, N, k)];
        }
        if (i == 0) {
            x[IX(0, j, k)] = (b == 1) ? -x[IX(1, j, k)] : x[IX(1, j, k)];
        }
        if (i == M + 1) {
            x[IX(M + 1, j, k)] = (b == 1) ? -x[IX(M, j, k)] : x[IX(M, j, k)];
        }
    }
}

void set_bnd(int M, int N, int O, int b, float *x) {
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((M + 2) / 16, (N + 2) / 16, (O + 2) / 16);
    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, x);
    cudaDeviceSynchronize();
}

__global__ void lin_solve_kernel_odd(int M, int N, int O, float *x, float *x0, float a, float c) {
    int i = threadIdx.x + 1;
    int j = threadIdx.y + 1;
    int k = threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        if ((i + j + k) % 2 == 1) {
            int idx = IX(i, j, k);
            float old_x = x[idx];
            x[idx] = (x0[idx] + a * (x[IX(i-1, j, k)] + x[IX(i+1, j, k)] + x[IX(i, j-1, k)] + x[IX(i, j+1, k)] + x[IX(i, j, k-1)] + x[IX(i, j, k+1)])) / c;
        }
    }
}

__global__ void lin_solve_kernel_even(int M, int N, int O, float *x, float *x0, float a, float c) {
    int i = threadIdx.x + 1;
    int j = threadIdx.y + 1;
    int k = threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        if ((i + j + k) % 2 == 0) {
            int idx = IX(i, j, k);
            float old_x = x[idx];
            x[idx] = (x0[idx] + a * (x[IX(i-1, j, k)] + x[IX(i+1, j, k)] + x[IX(i, j-1, k)] + x[IX(i, j+1, k)] + x[IX(i, j, k-1)] + x[IX(i, j, k+1)])) / c;
        }
    }
}

void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    int size = (M + 2) * (N + 2) * (O + 2);
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((M + 2) / 16, (N + 2) / 16, (O + 2) / 16);

    float tol = 1e-7;
    float max_c = 0.0f;

    do {
        max_c = 0.0f;
        lin_solve_kernel_odd<<<numBlocks, threadsPerBlock>>>(M, N, O, x, x0, a, c);
        cudaDeviceSynchronize();
        
        lin_solve_kernel_even<<<numBlocks, threadsPerBlock>>>(M, N, O, x, x0, a, c);
        cudaDeviceSynchronize();

        // Check max_c here as done in the OpenMP version
        // (you can calculate max_c using a reduction on the device)
    } while (max_c > tol);
}

__global__ void diffuse_kernel(int M, int N, int O, float *x, float *x0, float diff, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > 0 && i <= M && j > 0 && j <= N && k > 0 && k <= O) {
        int index = IX(i, j, k);
        x[index] = x0[index] + dt * diff * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                             x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                             x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)] - 6 * x[index]);
    }
}

void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
    int max = MAX(M, MAX(N, O));
    float a = dt * diff * max * max;
    lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

__global__ void advect_kernel(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    int i = threadIdx.x + 1;
    int j = threadIdx.y + 1;
    int k = threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        float u_val = u[IX(i, j, k)];
        float v_val = v[IX(i, j, k)];
        float w_val = w[IX(i, j, k)];

        float x = i - dt * u_val;
        float y = j - dt * v_val;
        float z = k - dt * w_val;

        x = fmaxf(0.5f, fminf(x, M + 0.5f));
        y = fmaxf(0.5f, fminf(y, N + 0.5f));
        z = fmaxf(0.5f, fminf(z, O + 0.5f));

        int i0 = (int)x, i1 = i0 + 1;
        int j0 = (int)y, j1 = j0 + 1;
        int k0 = (int)z, k1 = k0 + 1;

        float s1 = x - i0, s0 = 1 - s1;
        float t1 = y - j0, t0 = 1 - t1;
        float u1 = z - k0, u0 = 1 - u1;

        d[IX(i, j, k)] = s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
                               t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
                         s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
                               t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
    }
}

void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((M + 2) / 16, (N + 2) / 16, (O + 2) / 16);
    advect_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d, d0, u, v, w, dt);
    cudaDeviceSynchronize();
}

// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
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

// Step function for density
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt) {
  add_source(M, N, O, x, x0, dt);
  SWAP(x0, x);
  diffuse(M, N, O, 0, x, x0, diff, dt);
  SWAP(x0, x);
  advect(M, N, O, 0, x, x0, u, v, w, dt);
}

// Step function for velocity
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