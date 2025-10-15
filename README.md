# Ashton Six: Performance Portfolio

10+ years experience in software engineering, now specialising in HPC.

## Projects

1. **[NEON Bytepack](./bytepack/README.md)** — bit pack/unpack routines; ~1.9× GB/s vs SOTA plane-transpose baseline.
1. **[NEON Delta Coding](./delta/README.md)** — delta, delta-of-delta and xor-with-previous decoding; ~1.5–2.2x GB/s vs baseline.
1. More releasing soon.

## Contact

Available for hire: https://ashtonsix.com

Follow me on X [@ashtonsix](https://x.com/ashtonsix) and LinkedIn [in/ashtonsix](https://linkedin.com/in/ashtonsix).

## Development Environment

Launch a fresh `m8g.large` instance on AWS (Neoverse V2, Graviton4) with Ubuntu LTS, connect via VSCode, and run this setup:

```sh
# Basics
sudo apt update
sudo apt install -y curl gnupg lsb-release make gawk build-essential

# LLVM toolchain (v21 pinned)
codename="$(lsb_release -cs)"
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/llvm.gpg
echo "deb [signed-by=/etc/apt/keyrings/llvm.gpg] http://apt.llvm.org/$codename/ llvm-toolchain-$codename-21 main" | \
  sudo tee /etc/apt/sources.list.d/llvm-21.list >/dev/null
sudo apt update
sudo apt install -y clang-21 lld-21 llvm-21-tools
sudo ln -sf /usr/bin/ld.lld-21 /usr/bin/ld.lld

# SIMDe (SIMD Everywhere)
git clone --depth 1 https://github.com/simd-everywhere/simde /tmp/simde
sudo rm -rf /usr/local/include/simde
sudo mkdir -p /usr/local/include
sudo cp -R /tmp/simde/simde /usr/local/include/

# Source
git clone --depth 1 https://github.com/ashtonsix/perf-portfolio.git
```

## License

Apache 2.0
