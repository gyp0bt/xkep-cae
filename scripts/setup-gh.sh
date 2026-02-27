#!/usr/bin/env bash
# GitHub CLI (gh) セットアップスクリプト
# Usage: bash scripts/setup-gh.sh
#
# このスクリプトは gh CLI がインストールされていなければ
# 最新安定版をダウンロードしてインストールする。

set -euo pipefail

GH_VERSION="${GH_VERSION:-2.65.0}"
INSTALL_DIR="/usr/local/bin"

if command -v gh &>/dev/null; then
    echo "gh is already installed: $(gh --version | head -1)"
    exit 0
fi

echo "Installing gh v${GH_VERSION} ..."

ARCH="$(uname -m)"
case "$ARCH" in
    x86_64)  ARCH_LABEL="amd64" ;;
    aarch64) ARCH_LABEL="arm64" ;;
    *)       echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

URL="https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_${ARCH_LABEL}.tar.gz"
echo "Downloading from ${URL} ..."
curl -sL "$URL" -o "${TMP_DIR}/gh.tar.gz"
tar -xzf "${TMP_DIR}/gh.tar.gz" -C "$TMP_DIR"
cp "${TMP_DIR}/gh_${GH_VERSION}_linux_${ARCH_LABEL}/bin/gh" "${INSTALL_DIR}/gh"
chmod +x "${INSTALL_DIR}/gh"

echo "gh installed successfully: $(gh --version | head -1)"
