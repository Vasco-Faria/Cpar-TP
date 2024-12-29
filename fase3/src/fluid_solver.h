#ifndef FLUID_SOLVER_H
#define FLUID_SOLVER_H

void cpy_host_to_device(float* u, float* v, float* w, float* dens);
void cpy_device_to_host(float* u, float* v, float* w, float* dens);
void init_cuda_mallocs_vel(int M, int N, int O, float* u, float* v, float* w, float* u0, float* v0, float* w0);
void free_cuda_mallocs_vel(float* u, float* v, float* w, float* u0, float* v0, float* w0);
void init_cuda_mallocs_dens(float* x, float* x0);
void free_cuda_mallocs_dens(float* x, float* x0, float* u, float* v, float* w);

void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt);
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0, float *v0, float *w0, float visc, float dt);

#endif // FLUID_SOLVER_H
