#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
native_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers \
    "$native_dir/point_plane_ab.cpp" -lvulkan -o "${1:-/tmp/sintergpu-point-plane-ab}"
