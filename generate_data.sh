#!/bin/bash
set +e # Continue on error

# Config
TANGO_EXE="./dist/tango-x86_64-linux.AppImage"
RAW_DIR="data/raw"
FINAL_DIR="data/dataset"

# [FIX] Define paths to BOTH versions of the game
ROM_GREGAR="roms/bn6_gregar.gba"
ROM_FALZAR="roms/bn6_falzer.gba"

mkdir -p "$RAW_DIR"
mkdir -p "$FINAL_DIR"

# Validate ROMs exist
if [ ! -f "$ROM_GREGAR" ] || [ ! -f "$ROM_FALZAR" ]; then
    echo "❌ Error: One or both ROM files not found."
    echo "Checked: $ROM_GREGAR and $ROM_FALZAR"
    exit 1
fi

# Loop through all replays
for replay in replays/*.tangoreplay; do
    [ -e "$replay" ] || continue
    
    filename=$(basename "$replay" .tangoreplay)
    
    # Filter for BN6 only
    if [[ "$filename" != *"bn6"* ]]; then
        continue
    fi

    echo "========================================"
    echo "Processing: $filename"

    # [FIX] Try Gregar first
    echo "   Attempting with Gregar..."
    $TANGO_EXE export "$replay" \
        --output-path "$RAW_DIR/${filename}.mp4" \
        --rom-path "$ROM_GREGAR" > /dev/null 2>&1
    
    # Check success
    if [ $? -eq 0 ]; then
        echo "   ✅ Success (Gregar)"
    else
        # [FIX] If Gregar failed, try Falzar
        echo "   ⚠️ Gregar failed. Retrying with Falzar..."
        $TANGO_EXE export "$replay" \
            --output-path "$RAW_DIR/${filename}.mp4" \
            --rom-path "$ROM_FALZAR" > /dev/null 2>&1
        
        if [ $? -eq 0 ]; then
            echo "   ✅ Success (Falzar)"
        else
            echo "   ❌ Failed with both versions. Skipping."
            continue
        fi
    fi

    # 2. Prepare Output
    mkdir -p "$FINAL_DIR/$filename"
    mv "$RAW_DIR/${filename}.mp4" "$FINAL_DIR/$filename/video.mp4"
    
    # 3. Convert Inputs
    python3 scripts/convert_dataset.py \
        --input "$RAW_DIR/${filename}.jsonl" \
        --output "$FINAL_DIR/$filename/actions.jsonl"

    echo "   🎉 Dataset ready: $FINAL_DIR/$filename"
done