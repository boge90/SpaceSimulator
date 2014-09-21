# Compiler
CXX := g++
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
COMMON_FLAGS := -arch=sm_35
LD_FLAGS := -lGL -lGLU -lGLEW -lglfw3 -lX11 -lXxf86vm -lXrandr -lpthread -lXi -lAntTweakBar $(COMMON_FLAGS)
CC_FLAGS := -c -O3 -Wall -std=c++0x
CU_FLAGS := -m64 -c $(COMMON_FLAGS)

# Exe
EXECUTABLE := space

# Rules
all: $(EXECUTABLE)

run:
	make all
	./$(EXECUTABLE)

$(EXECUTABLE): $(OBJ_FILES)
	$(NVCC) $^ -o $@ $(LD_FLAGS)

obj/%.o: src/%.cpp
	$(CXX) $(CC_FLAGS) -o $@ $<

obj/%.o: src/%.cu
	$(NVCC) $(CU_FLAGS) -o $@ $<
	
clean:
	rm obj/*.o
	rm $(EXECUTABLE)
