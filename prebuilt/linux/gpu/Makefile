###############################################################################
# Simple Makefile for examples_gpu_call (with GPU support)
###############################################################################

# Compiler and flags
CXX = g++
CXXFLAGS = -O2 -fopenmp -I./include

# GPU flag
STILES_GPU = 1

# If STILES_GPU is enabled, add -DSTILES_GPU to CXXFLAGS
ifeq ($(STILES_GPU),1)
  CXXFLAGS += -DSTILES_GPU
endif

# Linker flags
LDFLAGS = -L./lib
LIBS = -lstiles -fopenmp -lhwloc

# Target
TARGET = examples_gpu_call

# Sources and Objects
SRC = examples_gpu_call.cpp
OBJ = examples_gpu_call.o

###############################################################################
# Rules
###############################################################################

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS) $(LIBS)

$(OBJ): $(SRC)
	$(CXX) $(CXXFLAGS) -c $(SRC) -o $(OBJ)

clean:
	rm -f $(TARGET) $(OBJ)

