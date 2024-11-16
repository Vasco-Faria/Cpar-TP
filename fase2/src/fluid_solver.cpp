#include "fluid_solver.h"
#include <cmath>
#include <omp.h>
#include <algorithm>

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x)                                                            \
  {                                                                            \
    float *tmp = x0;                                                           \
    x0 = x;                                                                    \
    x = tmp;                                                                   \
  }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// Add sources (density or velocity)
void add_source(int M, int N, int O, float *x, float *s, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2);
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        x[i] += dt * s[i];
    }
}

// Set boundary conditions
void set_bnd(int M, int N, int O, int b, float *x) {
  int i, j;

  #pragma omp parallel for schedule(static)
  for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= M; i++) {
          x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
          x[IX(i, j, O + 1)] = b == 3 ? -x[IX(i, j, O)] : x[IX(i, j, O)];
      }
  }

  #pragma omp parallel for schedule(static)
  for (int j = 1; j <= O; j++) {
      for (int i = 1; i <= N; i++) {
          x[IX(0, i, j)] = b == 1 ? -x[IX(1, i, j)] : x[IX(1, i, j)];
          x[IX(M + 1, i, j)] = b == 1 ? -x[IX(M, i, j)] : x[IX(M, i, j)];
      }
  }

  #pragma omp parallel for schedule(static)
  for (int j = 1; j <= O; j++) {
      for (int i = 1; i <= M; i++) {
          x[IX(i, 0, j)] = b == 2 ? -x[IX(i, 1, j)] : x[IX(i, 1, j)];
          x[IX(i, N + 1, j)] = b == 2 ? -x[IX(i, N, j)] : x[IX(i, N, j)];
      }
  }

  // Set corners
  x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
  x[IX(M + 1, 0, 0)] =
      0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
  x[IX(0, N + 1, 0)] =
      0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
  x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] +
                                    x[IX(M + 1, N + 1, 1)]);
}

// Red-black solver with convergence check optimized with OpenMP
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    float tol = 1e-7, max_c, old_x, change;
    int l = 0;
    float inverso_c = 1.0f / c;

    do {
        max_c = 0.0f;

        #pragma omp parallel for collapse(2) schedule(static) reduction(max:max_c) private(old_x, change)
        for (int k = 1; k <= O; k++) {
            for (int j = 1; j <= N; j++) {
                for (int i = 1; i <= M; i++) {
                    if ((i + j + k) % 2 == 1) {
                        old_x = x[IX(i, j, k)];
                        x[IX(i, j, k)] = (x0[IX(i, j, k)] +
                                          a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                               x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                               x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) * inverso_c;
                        change = fabs(x[IX(i, j, k)] - old_x);
                        if (change > max_c) max_c = change;
                    }
                }
            }
        }

        #pragma omp parallel for collapse(2) schedule(static) reduction(max:max_c) private(old_x, change)
        for (int k = 1; k <= O; k++) {
            for (int j = 1; j <= N; j++) {
                for (int i = 1; i <= M; i++) {
                    if ((i + j + k) % 2 == 0) {
                        old_x = x[IX(i, j, k)];
                        x[IX(i, j, k)] = (x0[IX(i, j, k)] +
                                          a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                               x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                               x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) * inverso_c;
                        change = fabs(x[IX(i, j, k)] - old_x);
                        if (change > max_c) max_c = change;
                    }
                }
            }
        }

        set_bnd(M, N, O, b, x);
    } 

    while (max_c > tol && ++l < 20);
}

// Diffusion step (uses implicit method)
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
  int max = MAX(M, MAX(N, O));
  float a = dt * diff * max * max;
  lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

    // Parallelize the outermost loop using OpenMP
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 1; k <= O; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= M; i++) {
                int index = IX(i, j, k); // Precompute index for d and d0
                float u_val = u[index]; // Fetch u only once
                float v_val = v[index]; // Fetch v only once
                float w_val = w[index]; // Fetch w only once

                float x = i - dtX * u_val;
                float y = j - dtY * v_val;
                float z = k - dtZ * w_val;

                // Clamp to grid boundaries
                x = fmaxf(0.5f, fminf(x, M + 0.5f));
                y = fmaxf(0.5f, fminf(y, N + 0.5f));
                z = fmaxf(0.5f, fminf(z, O + 0.5f));

                int i0 = (int)x, i1 = i0 + 1;
                int j0 = (int)y, j1 = j0 + 1;
                int k0 = (int)z, k1 = k0 + 1;

                float s1 = x - i0, s0 = 1 - s1;
                float t1 = y - j0, t0 = 1 - t1;
                float u1 = z - k0, u0 = 1 - u1;

                // Use precomputed indices to minimize memory access
                d[index] =
                    s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
                          t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
                    s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
                          t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
            }
        }
    }
    set_bnd(M, N, O, b, d);
}

// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
  // Calculate divergence and initialize pressure field
  float inverso_MNO = 1.0f / (MAX(M, MAX(N, O)));
  #pragma omp parallel for collapse(2) schedule(static)
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
  #pragma omp parallel for collapse(2) schedule(static)
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