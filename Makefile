# Compiler
CXX := mpic++
NVCC := nvcc

# Source files
CPP_FILES := $(wildcard src/*.cpp)
CU_FILES := $(wildcard src/*.cu)
C_FILES := $(wildcard src/*.c)

# Object files
CPP_OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))
CU_OBJ_FILES := $(addprefix obj/,$(notdir $(CU_FILES:.cu=.o)))
C_OBJ_FILES := $(addprefix obj/,$(notdir $(C_FILES:.cu=.o)))
OBJ_FILES := $(CPP_OBJ_FILES) $(CU_OBJ_FILES) $(C_OBJ_FILES)

# Flags
CUDA_HOME := /usr/local/cuda
LD_FLAGS := -lGL -lGLU -lGLEW -lglfw3 -lX11 -lXxf86vm -lXrandr -lpthread -lXi -L$(CUDA_HOME)/lib64/ -lcudart
CC_FLAGS := -c -O3 -Wall -std=c++0x
CU_FLAGS := -m64 -arch=sm_35 -c

# Exe
EXECUTABLE := space

# Rules
all: $(EXECUTABLE)

recompile:
	make clean
	make -j

run:
	make all
	mpirun -n 1 ./$(EXECUTABLE)

$(EXECUTABLE): $(OBJ_FILES)
	$(CXX) $^ -o $@ $(LD_FLAGS)

obj/%.o: src/%.cpp
	$(CXX) $(CC_FLAGS) -o $@ $<

obj/%.o: src/%.cu
	$(NVCC) $(CU_FLAGS) -o $@ $<
	
clean:
	rm obj/*.o
	rm $(EXECUTABLE)
