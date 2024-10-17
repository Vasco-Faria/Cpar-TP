CPP = g++ -pg -Wall -Wextra -lm
SRCS = main.cpp fluid_solver.cpp EventManager.cpp
OUTPUT = fluid_sim

CFLAG_DEFAULT = -O2 -ffast-math
CFLAG_VEC = -ftree-vectorize -march=native
CFLAG_LOOP = -funroll-loops
CFLAG_O3 = -O3 -ffast-math
CFLAG_OFAST = -Ofast -ffast-math
CFLAG_ALIGN = -falign-functions=16 -falign-loops=16
CFLAG_LTO = -flto

all: default

default:
        $(CPP) $(CFLAG_DEFAULT) $(SRCS) -o $(OUTPUT)

vec:
        $(CPP) $(CFLAG_DEFAULT) $(CFLAG_VEC) $(SRCS) -o $(OUTPUT)

loop:
        $(CPP) $(CFLAG_DEFAULT) $(CFLAG_LOOP) $(SRCS) -o $(OUTPUT)

o3:
        $(CPP) $(CFLAG_O3) $(SRCS) -o $(OUTPUT)

ofast:
        $(CPP) $(CFLAG_OFAST) $(SRCS) -o $(OUTPUT)

align:
        $(CPP) $(CFLAG_DEFAULT) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)

lto:
        $(CPP) $(CFLAG_DEFAULT) $(CFLAG_LTO) $(SRCS) -o $(OUTPUT)

perf:
        perf stat -e cache-misses,cache-references,instructions,cycles ./fluid_sim

gprof:
        gprof fluid_sim gmon.out > gprof_report.txt

graph:
        gprof fluid_sim gmon.out | gprof2dot | dot -Tpng -o graph.png

clean:
        @rm fluid_sim
        @echo Done.