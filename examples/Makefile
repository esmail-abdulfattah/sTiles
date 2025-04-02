###############################################################################
# Environment Setup
###############################################################################

STILES_PATH = ./sTiles/prebuilt/linux
INCLUDE_PATH = -I$(STILES_PATH)/include
export LD_LIBRARY_PATH=$(STILES_PATH)/lib:$LD_LIBRARY_PATH
MKL_LAPACKE_ROOT = #

# Parallel make
MAKE = make -j 8

# Compilers
CC      = gcc 
CXX     = g++ 
FC      = gfortran

###############################################################################
# Include Paths
###############################################################################

# Check if MKL_LAPACKE_ROOT is empty
ifeq ($(MKL_LAPACKE_ROOT),)
    # Generic LAPACKE settings
    LIBLAPACKE = -llapack -lmkl_gf_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
    CXXFLAGS += -DUSE_GENERIC_LAPACKE
    CFLAGS += -DUSE_GENERIC_LAPACKE
else
    # MKL LAPACKE settings
    LIBLAPACKE = -lmkl_gf_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
    CXXFLAGS += -DUSE_MKL_LAPACKE
    CFLAGS += -DUSE_MKL_LAPACKE
endif

###############################################################################
# Compiler Flags
###############################################################################
CFLAGS      += -O2 -DADD_ -fopenmp
FFLAGS      = -O2 -fPIC
CXXFLAGS    += -O2 -DADD_ -fopenmp

###############################################################################
# LAPACK/BLAS Libraries
###############################################################################
LIBLAPACK   = $(LIBLAPACKE)

###############################################################################
# Check for NUMA and HWLOC libraries
###############################################################################
NUMA_AVAILABLE := $(shell ldconfig -p | grep -q libnuma && echo 1 || echo 0)
HWLOC_CFLAGS   := $(shell pkg-config --cflags hwloc 2>/dev/null)
HWLOC_LIBS     := $(shell pkg-config --libs hwloc 2>/dev/null)

ifeq ($(NUMA_AVAILABLE), 1)
    LIBEXT  += -lnuma
    CFLAGS  += -DNUMA_ENABLED
    CXXFLAGS += -DNUMA_ENABLED
endif

ifneq ($(HWLOC_CFLAGS),)
    CFLAGS   += -DSTILES_HWLOC $(HWLOC_CFLAGS)
    CXXFLAGS += -DSTILES_HWLOC $(HWLOC_CFLAGS)
    LIBEXT   += $(HWLOC_LIBS)
endif

CFLAGS      += $(INCLUDE_PATH)
CXXFLAGS    += $(INCLUDE_PATH)
###############################################################################
# Library Paths
###############################################################################
LIBPATHS = -L$(STILES_PATH)/lib

# Libraries (Dynamically Adjusted)
LIBS = -lstiles $(LIBEXT)
LIBS += $(LIBLAPACK)

###############################################################################
# Source, Object, and Executable
###############################################################################
SRC = $(wildcard *.cpp)  # Get all .cpp files in the directory
OBJ = $(SRC:.cpp=.o)     # Convert .cpp filenames to .o
EXE = $(SRC:.cpp=)       # Remove .cpp to get binary names

###############################################################################
# Phony Targets
###############################################################################
all: $(EXE)

debug: CXXFLAGS += -g -DDEBUG
debug: CFLAGS += -g -DDEBUG
debug: all

# Build the executables
$(EXE): % : %.o
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(STILES_PATH)/lib $(LIBS) -lhwloc

# Clean only .o files
clean:
	rm -f $(OBJ)

# Clean all generated files, including executables
cleanall:
	rm -f $(OBJ) $(EXE)

# Compilation rules
%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

examples_%: examples_%.o
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(STILES_PATH)/lib $(LIBS)

examples_%: examples_%.o
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(STILES_PATH)/lib $(LIBS)

# Phony targets
.PHONY: all clean cleanall debug

.SECONDARY: $(OBJ)




