CMAKE          ?= cmake
BUILD_DIR      ?= build
CXX            ?= clang++

.PHONY: all configure clean distclean run

all: configure
	$(CMAKE) --build $(BUILD_DIR)

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) \
	  -DCMAKE_CXX_COMPILER="$(CXX)" \
	  -DCMAKE_CXX_FLAGS="-O3" \
	  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

run: all
	./$(BUILD_DIR)/p4_synth

clean:
	$(CMAKE) --build $(BUILD_DIR) --target clean || true

distclean:
	rm -rf "$(BUILD_DIR)"
