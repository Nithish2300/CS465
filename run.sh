#!/bin/bash
set -e

# Build release binary if not present or if source is newer
BINARY="target/release/q17"
NEEDS_BUILD=false
if [ ! -f "$BINARY" ] || [ "Cargo.toml" -nt "$BINARY" ]; then
    NEEDS_BUILD=true
elif [ -n "$(find src -name '*.rs' -newer "$BINARY" 2>/dev/null)" ]; then
    NEEDS_BUILD=true
fi
if [ "$NEEDS_BUILD" = true ]; then
    echo "Building release binary..." >&2
    RUSTFLAGS="-C target-cpu=native" cargo build --release 2>&1 | grep -v "^$" >&2
fi

# Pass all arguments through to the binary
exec "$BINARY" "$@"
