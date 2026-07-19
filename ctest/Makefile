# ---------------------------------------------------------------------------
# Build + run the libstiles smoke test against a prebuilt CI artifact.
#
# Point LIB at the unzipped artifact folder (the one holding include/stiles.h
# and lib/libstiles.{so,dylib}):
#
#   Linux x86_64   : make run LIB=../binaries/libstiles-linux-x86_64
#   Linux arm64    : make run LIB=../binaries/libstiles-linux-arm64
#   macOS Apple    : make run LIB=../binaries/libstiles-macos-apple-arm64
#
# `make run` builds then runs; the rpath is baked in so no LD_LIBRARY_PATH /
# DYLD_LIBRARY_PATH is needed. Exit code is 0 iff every check passes.
# ---------------------------------------------------------------------------
LIB ?= ../binaries/libstiles-linux-x86_64

INC := $(LIB)/include
LD  := $(LIB)/lib

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CXX ?= clang++
else
    CXX ?= g++
endif

CXXFLAGS := -std=c++17 -O2 -Wall
# Bake the artifact's lib dir into the runtime search path (rpath) so the test
# finds libstiles.{so,dylib} without any env vars.
LDFLAGS  := -L$(LD) -lstiles -Wl,-rpath,$(abspath $(LD))

test_stiles: test_stiles.cpp $(INC)/stiles.h
	$(CXX) $(CXXFLAGS) -I$(INC) $< $(LDFLAGS) -o $@

scale_stiles: scale_stiles.cpp $(INC)/stiles.h
	$(CXX) $(CXXFLAGS) -I$(INC) $< $(LDFLAGS) -o $@

run_mtx: run_mtx.cpp $(INC)/stiles.h
	$(CXX) $(CXXFLAGS) -I$(INC) $< $(LDFLAGS) -o $@

.PHONY: run scale mtx clean
run: test_stiles
	./test_stiles

# Scaling probe: factor a heavier matrix at 1,2,4,8… cores and print speedup.
scale: scale_stiles
	./scale_stiles

# Factor a REAL Matrix Market file: make mtx MTX=inla_graph_ferris.mtx
mtx: run_mtx
	./run_mtx $(MTX)

clean:
	rm -f test_stiles scale_stiles run_mtx
