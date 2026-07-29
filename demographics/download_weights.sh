#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="Ali-Nigzu/backenddev"
RELEASE_TAG="demographic-model-v1"
ASSET_NAME="demographicweights.pth"
EXPECTED_SHA256="cc279b6914b3ee8be6a58139c06ecb24ca95751233cf6c07804b93184614eb17"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET_PATH="${SCRIPT_DIR}/${ASSET_NAME}"
DOWNLOAD_URL="https://github.com/${REPOSITORY}/releases/download/${RELEASE_TAG}/${ASSET_NAME}"

mkdir -p "${SCRIPT_DIR}"

if command -v gh >/dev/null 2>&1; then
    gh release download "${RELEASE_TAG}" \
        --repo "${REPOSITORY}" \
        --pattern "${ASSET_NAME}" \
        --dir "${SCRIPT_DIR}" \
        --clobber
elif command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 \
        "${DOWNLOAD_URL}" \
        --output "${TARGET_PATH}"
else
    echo "Error: either GitHub CLI (gh) or curl is required." >&2
    exit 1
fi

echo "${EXPECTED_SHA256}  ${TARGET_PATH}" | sha256sum --check --status || {
    echo "Error: downloaded demographic model failed SHA-256 verification." >&2
    rm -f "${TARGET_PATH}"
    exit 1
}

echo "Demographic model ready: ${TARGET_PATH}"
