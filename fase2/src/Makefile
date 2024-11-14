# Compiler and flags
CPP = g++
CFLAGS = -Wall -Wextra -O3 -fopenmp -march=native -funroll-loops -ffast-math -fopt-info -Ofast -fprefetch-loop-arrays -ftree-vectorize
LDFLAGS = -fopenmp -flto

# Set OpenMP threads
OMP_NUM_THREADS ?= 4

# Source files and object files
SRCS = main.cpp fluid_solver.cpp EventManager.cpp
OBJS = $(SRCS:.cpp=.o)

# Executable name
EXEC = fluid_sim

# Default target
all: $(EXEC)

# Link object files to create the executable
$(EXEC): $(OBJS)
	module load gcc/11.2.0 && \
	$(CPP) $(OBJS) $(LDFLAGS) -o $@

# Compile .cpp files into .o files
%.o: %.cpp
	module load gcc/11.2.0 && \
	$(CPP) $(CFLAGS) -c $< -o $@

# Clean target to remove generated files
clean:
	@echo Cleaning up...
	@rm -f $(OBJS) $(EXEC)
	@echo Done.

# Run target with performance measurement
run: $(EXEC)
	export OMP_NUM_THREADS=$(OMP_NUM_THREADS) && \
	srun --partition=cpar --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./$(EXEC)

# Dependency rules to ensure proper recompilation
fluid_solver.o: fluid_solver.h
main.o: fluid_solver.h EventManager.h
EventManager.o: EventManager.h

# Phony targets
.PHONY: all clean run
