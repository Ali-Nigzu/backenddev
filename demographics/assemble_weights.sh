#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARTS_DIR="${SCRIPT_DIR}/model_parts"
TARGET="${SCRIPT_DIR}/demographicweights.pth"
TEMP="${TARGET}.tmp"

EXPECTED_SHA256="cc279b6914b3ee8be6a58139c06ecb24ca95751233cf6c07804b93184614eb17"

parts=(
    "${PARTS_DIR}/demographicweights.pth.part-00"
    "${PARTS_DIR}/demographicweights.pth.part-01"
)

for part in "${parts[@]}"; do
    if [[ ! -f "${part}" ]]; then
        echo "Missing model part: ${part}" >&2
        exit 1
    fi
done

cat "${parts[@]}" > "${TEMP}"

actual_sha256="$(sha256sum "${TEMP}" | awk '{print $1}')"

if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
    echo "Model checksum verification failed." >&2
    echo "Expected: ${EXPECTED_SHA256}" >&2
    echo "Actual:   ${actual_sha256}" >&2
    rm -f "${TEMP}"
    exit 1
fi

mv "${TEMP}" "${TARGET}"

echo "MiVOLO weights assembled successfully:"
echo "${TARGET}"
