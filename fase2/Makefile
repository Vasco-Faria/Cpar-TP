# Compiler and flags
CPP = g++
CFLAGS = -Wall -Wextra -lm -march=native -funroll-loops -ffast-math -fopt-info -Ofast -ftree-vectorize -mavx
OMPFLAGS = -fopenmp -msse4.1

# Source files and object files
SRCS = src/main.cpp src/fluid_solver.cpp src/EventManager.cpp
OBJS_SEQ = $(SRCS:.cpp=_seq.o)
OBJS_PAR = $(SRCS:.cpp=_par.o)

# Executable names
EXEC_SEQ = fluid_sim_seq
EXEC_PAR = fluid_sim_par

all: par seq

par:
	$(CPP) $(CFLAGS) $(OMPFLAGS) $(SRCS) -o $(EXEC_PAR)

seq:
	$(CPP) $(CFLAGS) $(SRCS) -o $(EXEC_SEQ)

# Run the program
runpar: $(EXEC_PAR)
	OMP_NUM_THREADS=20 ./fluid_sim_par

runseq: $(EXEC_SEQ)
	./fluid_sim_seq

clean:
	@echo Cleaning up...
	@rm -f $(EXEC_SEQ) $(EXEC_PAR) gmon.out callgraph.png callgraph_seq.png fluid_sim_output.txt
	@echo Done.
