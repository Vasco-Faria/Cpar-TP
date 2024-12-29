#ifndef FLUID_SOLVER_H
#define FLUID_SOLVER_H

extern float* v_u;
extern float* v_v;
extern float* v_w;

extern float* d_x;

void init_cuda_mallocs_vel(int M, int N, int O, float* u, float* v, float* w, float* u0, float* v0, float* w0);
void free_cuda_mallocs_vel();
void init_cuda_mallocs_dens(float* x, float* x0);
void free_cuda_mallocs_dens();

void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt);
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0, float *v0, float *w0, float visc, float dt);

#endif // FLUID_SOLVER_H
