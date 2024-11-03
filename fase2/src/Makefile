# Compiler and flags
CPP = g++
CFLAGS = -Wall -O3 -fopenmp -ffast-math

# Source files and object files
SRCS = main.cpp fluid_solver.cpp EventManager.cpp
OBJS = $(SRCS:.cpp=.o)

# Executable name
EXEC = fluid_sim

# Default target
all: $(EXEC)

# Link object files to create the executable
$(EXEC): $(OBJS)
	$(CPP) $(CFLAGS) -o $@ $^

# Compile .cpp files into .o files
%.o: %.cpp
	$(CPP) $(CFLAGS) -c $< -o $@

# Clean target to remove generated files
clean:
	@echo Cleaning up...
	@rm -f $(OBJS) $(EXEC)
	@echo Done.

# Dependency rules to ensure proper recompilation
fluid_solver.o: fluid_solver.h
main.o: fluid_solver.h EventManager.h
EventManager.o: EventManager.h

# Phony targets
.PHONY: all clean
