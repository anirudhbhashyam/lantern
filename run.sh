#!/bin/sh
set -eu
cd "$(dirname "$0")"
GOOSE_BIN=${GOOSE_BIN:-./goose-v2026.4-macos-arm64/goose}
PYTHON=${PYTHON:-python3}
export GOOSE_HOME="${GOOSE_HOME:-$PWD/.build/goose}"
mkdir -p .build artifacts

data() {
    mkdir -p data
    for pair in \
        'train-images-idx3-ubyte 440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609' \
        'train-labels-idx1-ubyte 3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c'
    do
        name=${pair%% *}
        hash=${pair#* }
        path="data/$name"
        if [ ! -f "$path.gz" ]; then
            curl -fsSL --retry 2 "https://storage.googleapis.com/cvdf-datasets/mnist/$name.gz" -o "$path.gz.part"
            mv "$path.gz.part" "$path.gz"
        fi
        printf '%s  %s\n' "$hash" "$path.gz" | shasum -a 256 -c
        gzip -dc "$path.gz" > "$path.part"
        mv "$path.part" "$path"
    done
}

build() {
    "$GOOSE_BIN" build diffusion.goose -o .build/diffusion
}

sample() {
    .build/diffusion sample "$1"
    "$PYTHON" scripts/render.py
}

case "${1:-all}" in
    data) data ;;
    build) build ;;
    train) build; .build/diffusion train "${2:-24000}" ;;
    sample) build; sample "${2:-47}" ;;
    test)
        "$GOOSE_BIN" build tensor_test.goose -o .build/tensor-test
        .build/tensor-test ;;
    all) data; build; .build/diffusion train "${2:-24000}"; sample "${3:-47}" ;;
    *) echo 'usage: ./run.sh [all [UPDATES [SEED]] | data | build | train [UPDATES] | sample [SEED] | test]' >&2; exit 2 ;;
esac
