# Config
LLVM_VER ?= 21
TARGET   ?= aarch64-linux-gnu
CPU      ?= neoverse-v2

CC  := clang-$(LLVM_VER)
CXX := clang++-$(LLVM_VER)
MCA := llvm-mca-$(LLVM_VER)

CFLAGS   ?= -std=c17 -O3 --target=$(TARGET) -mcpu=$(CPU) -I. -DNDEBUG \
            -fno-verbose-asm -fuse-ld=lld
CXXFLAGS ?= -std=c++23 -O3 --target=$(TARGET) -mcpu=$(CPU) -I. -DNDEBUG \
            -fno-verbose-asm -fno-rtti -fuse-ld=lld
LDFLAGS  ?= -fuse-ld=lld -Wl,--build-id=none

OUTDIR   := out
ASMDIR   := $(OUTDIR)/asm
MCADIR   := $(OUTDIR)/mca
SPLITTER := ../.common/split_cfi.awk
OUTFILES := "delta_naive_W32 prefix_fastpfor_W32 prefix_naive_W32 prefix_unrolled_W32 prefix_pipelined_W32 delta_transpose_W32 prefix_transpose_W32 deltaOfDelta_naive_W32 prefixOfPrefix_naive_W32 prefixOfPrefix_pipelined_W32 deltaOfDelta_transpose_W32 prefixOfPrefix_transpose_W32 xor_naive_W32 xorInv_naive_W32 xorInv_pipelined_W32 xor_transpose_W32 xorInv_transpose_W32"

SRC_MAIN := delta.c
SRC_EVAL := delta_eval.cpp
EVALBIN  := $(OUTDIR)/delta_eval

.PHONY: all asm mca eval clean

all: clean $(EVALBIN) asm mca

$(OUTDIR):
	mkdir -p "$@"

# Compile to ASM and split file by function
asm: | $(OUTDIR)
	mkdir -p "$(ASMDIR)"
	$(CC) $(CFLAGS) -S -o - $(SRC_MAIN) \
	  | sed -e 's:[[:space:]]*//.*$$::' -e '/^[[:space:]]*$$/d' \
	  | awk -v outdir="$(ASMDIR)" -v outfiles=$(OUTFILES) -f "$(SPLITTER)" -

# Run llvm-mca on each split .s file
mca:
	mkdir -p "$(MCADIR)"
	for f in $(ASMDIR)/*.s; do \
	  [ -f "$$f" ] || continue; \
	  b=$$(basename $$f .s); \
	  $(MCA) -mtriple=$(TARGET) -mcpu=$(CPU) -iterations=100 -timeline -resource-pressure \
	    < "$$f" > "$(MCADIR)/$$b.mca"; \
	done

# Build the evaluation binary
$(EVALBIN): $(SRC_EVAL) $(SRC_MAIN) | $(OUTDIR)
	$(CC) $(CFLAGS) -c $(SRC_MAIN) -o $(OUTDIR)/delta.o
	$(CXX) $(CXXFLAGS) $(SRC_EVAL) $(OUTDIR)/delta.o -o $@ $(LDFLAGS)

eval: $(EVALBIN)

clean:
	rm -rf "$(OUTDIR)"
