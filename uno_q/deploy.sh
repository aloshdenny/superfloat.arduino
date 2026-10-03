#!/usr/bin/env bash
# Assemble the Wildfire Sentinel App Lab app on an UNO Q.
#
# Run on the board (Debian, aarch64) from a clone of this repository:
#
#   ./uno_q/deploy.sh runs/smoke-sf8/smokenet-w1-sf8.sfm
#
# It builds libsfrt natively, vendors the sfedge package next to main.py
# (App Lab only mounts the app folder into its container), installs the model
# and keeps any existing config.json. Then open the app in App Lab and Run.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC="$ROOT/uno_q/wildfire-sentinel"
DEST=${DEST:-$HOME/ArduinoApps/wildfire-sentinel}
MODEL=${1:?usage: deploy.sh path/to/smokenet.sfm}

if ! command -v cc >/dev/null; then
  echo "no C compiler on this board: sudo apt install build-essential" >&2
  exit 1
fi

python3 "$ROOT/tools/sync_sketch.py" --check
make -s -C "$ROOT/runtime" clean
make -s -C "$ROOT/runtime"
"$ROOT/runtime/build/sfrt_run" "$MODEL" --random 20

mkdir -p "$DEST"
for item in app.yaml python sketch assets config.example.json; do
  rm -rf "${DEST:?}/$item"
  cp -R "$SRC/$item" "$DEST/"
done
mkdir -p "$DEST/python/vendor" "$DEST/models" "$DEST/data"
cp -R "$ROOT/sfedge" "$DEST/python/vendor/"
rm -rf "$DEST/python/vendor/sfedge/__pycache__"
# libsfrt.so on the board; .dylib when rehearsing the deploy on a Mac
cp "$ROOT"/runtime/build/libsfrt.* "$DEST/python/vendor/"
cp "$MODEL" "$DEST/models/smokenet.sfm"
[ -f "$DEST/config.json" ] || cp "$SRC/config.example.json" "$DEST/config.json"

echo "deployed to $DEST"
echo "edit $DEST/config.json (node_id, lat/lon, webhook_url), then Run it from App Lab"
