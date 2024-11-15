# Compiler and flags
CPP = g++
CFLAGS = -Wall -Wextra -g -fno-omit-frame-pointer -march=native -funroll-loops -ffast-math -fopt-info -Ofast -fprefetch-loop-arrays -ftree-vectorize -fno-omit-frame-pointer
OMPFLAGS = -fopenmp

# Source files and object files
SRCS = main.cpp fluid_solver.cpp EventManager.cpp
OBJS_SEQ = $(SRCS:.cpp=_seq.o)
OBJS_PAR = $(SRCS:.cpp=_par.o)

# Executable names
EXEC_SEQ = fluid_sim_seq
EXEC_PAR = fluid_sim_par

# Default target
all: $(EXEC_SEQ) $(EXEC_PAR)

# Link object files to create the sequential executable
$(EXEC_SEQ): $(OBJS_SEQ)

	$(CPP) $(OMPFLAGS) $(OBJS_SEQ) $(LDFLAGS) -o $@

# Link object files to create the parallel executable
$(EXEC_PAR): $(OBJS_PAR)

	$(CPP) $(OMPFLAGS) $(OBJS_PAR) $(LDFLAGS) -o $@

# Compile .cpp files into .o files for the sequential version
%_seq.o: %.cpp

	$(CPP) $(CFLAGS) -c $< -o $@

# Compile .cpp files into .o files for the parallel version
%_par.o: %.cpp

	$(CPP) $(CFLAGS) $(OMPFLAGS) -c $< -o $@

# Clean target to remove generated files
clean:
	@echo Cleaning up...
	@rm -f $(OBJS_SEQ) $(OBJS_PAR) $(EXEC_SEQ) $(EXEC_PAR)
	@echo Done.

# Run target for the parallel version with performance measurement
runpar:
	export OMP_NUM_THREADS=20; srun --partition=cpar --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./$(EXEC_PAR)
	
perf:
	export OMP_NUM_THREADS=20; perf stat -e cache-misses,cache-references,instructions,cycles ./fluid_sim_par

# Run target for the sequential version with performance measurement
runseq:
	srun --partition=cpar --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./$(EXEC_SEQ)

# Dependency rules to ensure proper recompilation
fluid_solver_seq.o: fluid_solver.h
fluid_solver_par.o: fluid_solver.h

main_seq.o: fluid_solver.h EventManager.h
main_par.o: fluid_solver.h EventManager.h

EventManager_seq.o: EventManager.h
EventManager_par.o: EventManager.h

# Phony targets
.PHONY: all clean runpar runseq

