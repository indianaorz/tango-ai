#!/bin/bash
set -euo pipefail

# Cleanup function
function cleanup {
    rm -rf tango_linux_workdir
}
trap cleanup EXIT
cleanup

# 1. Fix: Create a temporary workspace Cargo.toml if it's missing.
if [ ! -f Cargo.toml ]; then
    echo "⚠️  Root Cargo.toml missing. Generating a workspace definition..."
    cat <<EOF > Cargo.toml
[workspace]
members = ["tango", "tango-filesync"]
resolver = "2"

[workspace.lints.rust]
# Defaults to satisfy inheritance

[workspace.lints.clippy]
# Defaults to satisfy inheritance
EOF
fi

# 2. [NEW FIX] Patch tango-filesync for "never type fallback" error
# This replaces .collect::<Result<_, _>>() with .collect::<Result<(), _>>()
if [ -f tango-filesync/src/lib.rs ]; then
    echo "🔧 Patching tango-filesync/src/lib.rs for Rust compiler compatibility..."
    sed -i 's/\.collect::<Result<_, _>>()/\.collect::<Result<(), _>>()/' tango-filesync/src/lib.rs
else
    echo "⚠️  Warning: tango-filesync/src/lib.rs not found. Skipping patch."
fi

# 3. Download AppImageTool (if not present)
if [ ! -f appimagetool-x86_64.AppImage ]; then
    wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod a+x appimagetool-x86_64.AppImage
fi

# 4. Build Linux binaries
target_arch="x86_64"

# Pin dependencies for Rust 1.87 compatibility
echo "🔧 Pinning dependencies for compatibility..."
cargo update --manifest-path tango/Cargo.toml -p home --precise 0.5.11 || true

# Build
cargo build --bin tango --target="${target_arch}-unknown-linux-gnu" --no-default-features --features=sdl2-audio,wgpu,cpal --release

# 5. Assemble AppImage
mkdir -p "tango_linux_workdir/${target_arch}/bin"
cp tango/src/icon.png tango_linux_workdir/tango.png
cp linux/AppRun tango_linux_workdir/AppRun
cp linux/tango.desktop tango_linux_workdir/tango.desktop

# Handle variable binary location
if [ -f "target/${target_arch}-unknown-linux-gnu/release/tango" ]; then
    cp "target/${target_arch}-unknown-linux-gnu/release/tango" "tango_linux_workdir/${target_arch}/bin/tango"
else
    cp "tango/target/${target_arch}-unknown-linux-gnu/release/tango" "tango_linux_workdir/${target_arch}/bin/tango"
fi

# 6. Bundle ffmpeg
ffmpeg_version="6.0"
wget "https://github.com/eugeneware/ffmpeg-static/releases/download/b${ffmpeg_version}/ffmpeg-linux-x64" -O "tango_linux_workdir/${target_arch}/bin/ffmpeg"
chmod a+x "tango_linux_workdir/${target_arch}/bin/ffmpeg"

# 7. Build final AppImage
mkdir -p dist
./appimagetool-x86_64.AppImage tango_linux_workdir "dist/tango-${target_arch}-linux.AppImage"

echo "✅ Build Complete: dist/tango-${target_arch}-linux.AppImage"