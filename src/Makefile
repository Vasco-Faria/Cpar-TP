CPP = g++ -pg -Wall 
SRCS = main.cpp fluid_solver.cpp EventManager.cpp
OUTPUT = fluid_sim

all:
	$(CPP) $(SRCS) -o $(OUTPUT)

perf:
	perf stat -e cache-misses,cache-references,instructions,cycles ./fluid_sim

gprof:
	gprof fluid_sim gmon.out > gprof_report.txt

graph:
	gprof fluid_sim gmon.out | gprof2dot | dot -Tpng -o graph.png

clean:
	@rm fluid
	@echo Done.
