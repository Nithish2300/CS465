#!/bin/bash
set -e

# Build release binary if not present or if source is newer
BINARY="target/release/q17"
if [ ! -f "$BINARY" ] || [ "src/main.rs" -nt "$BINARY" ] || [ "Cargo.toml" -nt "$BINARY" ]; then
    echo "Building release binary..." >&2
    RUSTFLAGS="-C target-cpu=native" cargo build --release 2>&1 | grep -v "^$" >&2
fi

# Pass all arguments through to the binary
exec "$BINARY" "$@"
