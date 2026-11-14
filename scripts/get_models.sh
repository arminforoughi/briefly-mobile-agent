#!/usr/bin/env bash
# Download the Kokoro-82M voice model (int8 ONNX, ~120 MB total) into models/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
base=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
for f in kokoro-v1.0.int8.onnx voices-v1.0.bin; do
  [ -f "models/$f" ] || curl -fL --progress-bar -o "models/$f" "$base/$f"
done
echo "models ready in ./models"
