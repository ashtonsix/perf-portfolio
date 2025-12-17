#!/usr/bin/env bash

set -euo pipefail

# Install EPFL logic synthesis libs into ./external
# - mockturtle (logic network library)
# - kitty      (truth table library)
# - percy      (exact synthesis library)
#
# All are header-only from the user's POV. percy carries SAT solvers
# (abc, glucose, etc.) as submodules in case you later want tests.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXTERNAL_DIR="$ROOT_DIR/external"

MOCK_URL="https://github.com/lsils/mockturtle.git"
KITTY_URL="https://github.com/lsils/kitty.git"
PERCY_URL="https://github.com/whaaswijk/percy.git"

mkdir -p "$EXTERNAL_DIR"

clone_or_update() {
  local url="$1"
  local dir="$2"
  local name="$3"

  if [[ -d "$dir/.git" ]]; then
    echo "==> Updating $name in $dir"
    git -C "$dir" fetch --all --tags
    git -C "$dir" pull --ff-only
    git -C "$dir" submodule update --init --recursive
  else
    echo "==> Cloning $name into $dir"
    git clone --recurse-submodules "$url" "$dir"
  fi
  echo
}

echo "==> Installing EPFL libs into $EXTERNAL_DIR"

clone_or_update "$MOCK_URL"  "$EXTERNAL_DIR/mockturtle" "mockturtle"
clone_or_update "$KITTY_URL" "$EXTERNAL_DIR/kitty"      "kitty"
clone_or_update "$PERCY_URL" "$EXTERNAL_DIR/percy"      "percy"

echo "==> Configuring nauty (required for percy)"
if [[ -f "$EXTERNAL_DIR/percy/nauty/configure" ]]; then
  (cd "$EXTERNAL_DIR/percy/nauty" && ./configure > /dev/null 2>&1)
  echo "   nauty configured"
else
  echo "   WARNING: nauty configure script not found"
fi

echo "==> Done."
echo "   mockturtle : $EXTERNAL_DIR/mockturtle"
echo "   kitty      : $EXTERNAL_DIR/kitty"
echo "   percy      : $EXTERNAL_DIR/percy"

