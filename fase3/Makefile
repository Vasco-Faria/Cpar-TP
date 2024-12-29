# Compilers and flags
CPP = g++
CUDA = nvcc
CFLAGS = -Wall -Wextra -lm -march=native -funroll-loops -ffast-math -fopt-info -Ofast -ftree-vectorize -mavx
CUDAFLAGS = -Xptxas -O3 -g -std=c++11 -arch=sm_35 -Wno-deprecated-gpu-targets -use_fast_math -restrict

# Source files and object files
SRCS = src/main.cpp src/EventManager.cpp
CUDA_SRCS = src/fluid_solver.cu
OBJS_SEQ = $(SRCS:.cpp=_seq.o)
OBJS_PAR = $(SRCS:.cpp=_par.o)
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

# Executable names
EXEC_SEQ = fluid_sim_seq
EXEC_PAR = fluid_sim_par
EXEC_CUDA = fluid_sim_cuda

all: cuda

cuda:
	$(CUDA) $(CUDAFLAGS) $(SRCS) $(CUDA_SRCS) -o $(EXEC_CUDA)

# Run the program
run:
	sbatch run_SBATCH.sh

clean:
	@echo Cleaning up...
	@rm -f $(EXEC_SEQ) $(EXEC_PAR) $(EXEC_CUDA) gmon.out callgraph.png callgraph_seq.png fluid_sim_output.txt
	@echo Done.
