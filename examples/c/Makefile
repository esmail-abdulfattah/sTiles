# Build the C examples against an unpacked sTiles release archive.
#
#   make STILES=/path/to/libstiles-linux-x86_64     # every example
#   make STILES=/path/to/... 01_quickstart          # just one
#
# STILES is the folder that holds include/ and lib/. The rpath lets the
# programs find libstiles at run time without setting LD_LIBRARY_PATH.
# Windows: see the comment at the end.

STILES ?= $(error set STILES to the unpacked archive, e.g. make STILES=$$HOME/libstiles-linux-x86_64)

CC       ?= cc
CFLAGS   ?= -O2 -Wall
CPPFLAGS += -I$(STILES)/include
LDFLAGS  += -L$(STILES)/lib -Wl,-rpath,$(STILES)/lib
LDLIBS   += -lstiles -lm

EXAMPLES = 01_quickstart 02_solve 03_selected_inverse 04_reuse_analysis \
           05_sampling 06_matrix_market

all: $(EXAMPLES)

%: %.c common.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

run: all
	@for e in $(EXAMPLES); do echo "== $$e"; ./$$e || exit 1; done

clean:
	rm -f $(EXAMPLES) $(EXAMPLES:=.exe)

.PHONY: all run clean

# Windows, MSYS2 UCRT64 shell with MinGW-w64 gcc (no rpath; the DLLs are
# found through PATH):
#   gcc -O2 01_quickstart.c -I$STILES/include -L$STILES/lib -lstiles -o 01_quickstart.exe
#   PATH="$STILES/lib:$PATH" ./01_quickstart.exe
