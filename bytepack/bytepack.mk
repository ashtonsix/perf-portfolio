# Config
LLVM_VER ?= 21
TARGET   ?= aarch64-linux-gnu
CPU      ?= neoverse-v2

CXX := clang++-$(LLVM_VER)
MCA := llvm-mca-$(LLVM_VER)

CXXFLAGS ?= -std=c++23 -O3 --target=$(TARGET) -mcpu=$(CPU) -I. -DNDEBUG \
            -fno-verbose-asm -fno-rtti -fuse-ld=lld
LDFLAGS  ?= -fuse-ld=lld -Wl,--build-id=none

OUTDIR   := out
ASMDIR   := $(OUTDIR)/asm
MCADIR   := $(OUTDIR)/mca
SPLITTER := ../.common/split_cfi.awk

OUTFILES ?= "pack1 unpack1 pack2 unpack2 pack3 unpack3 pack4 unpack4 pack5 unpack5 pack6 unpack6 pack7 unpack7 pack8 unpack8"

SRC_NEON     := bytepack.cpp
SRC_BASELINE := bytepack_baseline.cpp
SRC_EVAL     := bytepack_eval.cpp
EVALBIN      := $(OUTDIR)/bytepack_eval

.PHONY: all asm mca eval clean

all: clean $(EVALBIN) mca

$(OUTDIR):
	mkdir -p "$@"

# Compile to ASM and split file by function
asm: | $(OUTDIR)
	mkdir -p "$(ASMDIR)"
	$(CXX) $(CXXFLAGS) -S -o - $(SRC_NEON) \
	  | sed -e 's:[[:space:]]*//.*$$::' -e '/^[[:space:]]*$$/d' \
	  | awk -v outdir="$(ASMDIR)" -v outfiles=$(OUTFILES) -f "$(SPLITTER)" -

# Run llvm-mca on each .s file
mca: asm
	mkdir -p "$(MCADIR)"
	for f in $(ASMDIR)/*.s; do \
	  b=$$(basename $$f .s); \
	  $(MCA) -mtriple=$(TARGET) -mcpu=$(CPU) -iterations=100 -timeline -resource-pressure \
	    < "$$f" > "$(MCADIR)/$$b.mca"; \
	done

# Build the evaluation binary
$(EVALBIN): $(SRC_EVAL) $(SRC_NEON) $(SRC_BASELINE) | $(OUTDIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

eval: $(EVALBIN)

clean:
	rm -rf "$(OUTDIR)"
