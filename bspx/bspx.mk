CXX := g++
CXXFLAGS := -std=c++20 -O3 -march=native

BUILDDIR := out
BIN := $(BUILDDIR)/bspx
SRC := bspx_main.cpp

.PHONY: all clean run

all: clean $(BIN)

$(BIN): $(SRC) bspx_policy.hpp bspx_runtime.hpp
	mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN)

clean:
	rm -f $(BIN)
