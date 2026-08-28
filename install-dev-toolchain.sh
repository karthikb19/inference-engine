#!/usr/bin/env bash
# Install GitHub CLI, OpenAI Codex CLI, CUDA Toolkit 13.1, GCC 15, and nlohmann/json.
# Usage: ./install-dev-toolchain.sh [project-directory]
#
# CUDA support is limited to x86_64 Ubuntu because NVIDIA publishes
# distribution-specific repositories. This script installs the toolkit only;
# install a compatible NVIDIA driver separately if one is not already present.

set -euo pipefail

readonly CODEX_INSTALL_URL="https://chatgpt.com/codex/install.sh"
readonly CUDA_KEYRING_VERSION="1.1-1"
readonly CUDA_TOOLKIT_PACKAGE="cuda-toolkit-13-1"
readonly JSON_VERSION="3.12.0"
readonly JSON_SHA256="aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63"
readonly JSON_URL="https://github.com/nlohmann/json/releases/download/v${JSON_VERSION}/json.hpp"

PROJECT_DIR="${1:-$PWD}"

info() { printf '\n==> %s\n' "$*"; }
die() { printf 'Error: %s\n' "$*" >&2; exit 1; }

need_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

run_as_root() {
  if [ "${EUID}" -eq 0 ]; then
    "$@"
  else
    need_command sudo
    sudo "$@"
  fi
}

install_gh_macos() {
  if ! command -v brew >/dev/null 2>&1; then
    die "Homebrew is required on macOS. Install it from https://brew.sh/, then run this script again."
  fi
  brew install gh
}

install_gh_linux() {
  if command -v apt-get >/dev/null 2>&1; then
    info "Installing GitHub CLI with apt"
    run_as_root apt-get update
    run_as_root apt-get install -y curl ca-certificates
    need_command curl
    local keyring=/usr/share/keyrings/githubcli-archive-keyring.gpg
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \
      run_as_root tee "$keyring" >/dev/null
    run_as_root chmod go+r "$keyring"
    printf '%s\n' \
      "deb [arch=$(dpkg --print-architecture) signed-by=$keyring] https://cli.github.com/packages stable main" | \
      run_as_root tee /etc/apt/sources.list.d/github-cli.list >/dev/null
    run_as_root apt-get update
    run_as_root apt-get install -y gh
  elif command -v dnf >/dev/null 2>&1; then
    info "Installing GitHub CLI with dnf"
    run_as_root dnf install -y dnf-plugins-core
    run_as_root dnf config-manager --add-repo https://cli.github.com/packages/rpm/gh-cli.repo
    run_as_root dnf install -y gh
  elif command -v zypper >/dev/null 2>&1; then
    info "Installing GitHub CLI with zypper"
    run_as_root zypper --non-interactive install gh
  elif command -v pacman >/dev/null 2>&1; then
    info "Installing GitHub CLI with pacman"
    run_as_root pacman -S --needed --noconfirm github-cli
  else
    die "Unsupported Linux distribution: no apt-get, dnf, zypper, or pacman found."
  fi
}

install_gh() {
  if command -v gh >/dev/null 2>&1; then
    info "GitHub CLI is already installed: $(gh --version | head -n 1)"
    return
  fi

  case "$(uname -s)" in
    Darwin) install_gh_macos ;;
    Linux) install_gh_linux ;;
    *) die "This script supports macOS and Linux only." ;;
  esac
}

install_codex() {
  need_command curl
  info "Installing or updating Codex"
  curl -fsSL "$CODEX_INSTALL_URL" | sh
}

install_cuda_and_gcc() {
  [ "$(uname -s)" = "Linux" ] || die "CUDA Toolkit installation is supported by this script on Linux only."
  [ "$(uname -m)" = "x86_64" ] || die "CUDA Toolkit installation is limited to x86_64."
  [ -r /etc/os-release ] || die "Cannot identify this Linux distribution."
  # shellcheck disable=SC1091
  . /etc/os-release

  case "${ID}:${VERSION_ID}" in
    ubuntu:22.04|ubuntu:24.04)
      ;;
    *)
      die "CUDA 13.1 is supported here only on Ubuntu 22.04/24.04. Detected ${PRETTY_NAME}."
      ;;
  esac

  command -v apt-get >/dev/null 2>&1 || die "apt-get is required for this CUDA installation."
  local distro="${ID}${VERSION_ID//./}"
  local keyring_deb="/tmp/cuda-keyring_${CUDA_KEYRING_VERSION}_all.deb"
  local keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_${CUDA_KEYRING_VERSION}_all.deb"

  info "Installing GCC 15 and C++ build tools"
  run_as_root apt-get update
  run_as_root apt-get install -y build-essential curl ca-certificates software-properties-common
  if ! apt-cache show gcc-15 >/dev/null 2>&1; then
    info "Adding Ubuntu Toolchain Test PPA for GCC 15"
    run_as_root add-apt-repository -y ppa:ubuntu-toolchain-r/test
    run_as_root apt-get update
  fi
  run_as_root apt-get install -y gcc-15 g++-15

  info "Adding the NVIDIA CUDA repository for ${distro}"
  curl -fsSL "$keyring_url" -o "$keyring_deb"
  run_as_root dpkg -i "$keyring_deb"
  rm -f "$keyring_deb"
  # cuda-keyring creates the signed repository entry. Remove only the legacy
  # duplicate entry if it refers to this exact repository without a keyring.
  local legacy_cuda_list="/etc/apt/sources.list.d/cuda.list"
  if [ -f "$legacy_cuda_list" ] && \
    grep -Fqx "deb https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64 /" "$legacy_cuda_list"; then
    run_as_root rm -f "$legacy_cuda_list"
  fi
  run_as_root apt-get update
  info "Installing CUDA Toolkit 13.1 (without the NVIDIA driver)"
  run_as_root apt-get install -y "$CUDA_TOOLKIT_PACKAGE"
}

validate_platform() {
  [ "$(uname -s)" = "Linux" ] || die "This complete toolchain installer supports Linux only because it installs CUDA."
  [ "$(uname -m)" = "x86_64" ] || die "This complete toolchain installer supports x86_64 only."
}

install_json_header() {
  [ -d "$PROJECT_DIR" ] || die "Project directory does not exist: $PROJECT_DIR"
  need_command curl
  need_command sha256sum

  local include_dir="$PROJECT_DIR/include"
  local json_header="$include_dir/json.hpp"
  mkdir -p "$include_dir"
  info "Downloading nlohmann/json ${JSON_VERSION} to $json_header"
  curl -fsSL "$JSON_URL" -o "$json_header"
  printf '%s  %s\n' "$JSON_SHA256" "$json_header" | sha256sum --check --status || {
    rm -f "$json_header"
    die "nlohmann/json checksum verification failed."
  }
}

verify_toolchain() {
  info "Verifying installed tools"
  gh --version | head -n 1
  codex --version
  gcc-15 --version | head -n 1
  g++-15 --version | head -n 1
  nvcc --version | tail -n 4
  printf '\nUse C++17 with:\n  g++-15 -std=c++17 -I%s/include ...\n' "$PROJECT_DIR"
  printf 'For CUDA builds:\n  nvcc -ccbin g++-15 -std=c++17 -I%s/include ...\n' "$PROJECT_DIR"
}

validate_platform
install_cuda_and_gcc
install_gh
install_codex
install_json_header

info "Installation complete"
verify_toolchain
printf '\nNext steps:\n  gh auth login\n  codex\n'
