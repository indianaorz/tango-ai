#!/bin/bash
set +e 

# --- CONFIGURATION ---
# Ensure this points to your new build
TANGO_EXE="./dist/tango-x86_64-linux.AppImage"
RAW_DIR="data/raw"
FINAL_DIR="data/dataset"
ROM_GREGAR="roms/bn6_gregar.gba"
ROM_FALZAR="roms/bn6_falzer.gba"

# Ensure directories exist
mkdir -p "$RAW_DIR"
mkdir -p "$FINAL_DIR"

if [ ! -f "$TANGO_EXE" ]; then
    echo "❌ Error: AppImage not found at $TANGO_EXE"
    exit 1
fi

# Loop through all replays in the folder
for replay in replays/*.tangoreplay; do
    [ -e "$replay" ] || continue
    filename=$(basename "$replay" .tangoreplay)
    
    if [[ "$filename" != *"bn6"* ]]; then continue; fi

    echo "========================================"
    echo "Processing: $filename"

    # Clean up previous run artifact
    rm -f static_data.json

    # --- STEP 1: EXPORT VIDEO & RAW JSONL ---
    # Silenced output again for clean logs
    $TANGO_EXE export "$replay" --output-path "$RAW_DIR/${filename}.mp4" --rom-path "$ROM_GREGAR" > /dev/null 2>&1
    
    if [ $? -ne 0 ]; then
        echo "   ⚠️ Gregar failed. Retrying with Falzar..."
        $TANGO_EXE export "$replay" --output-path "$RAW_DIR/${filename}.mp4" --rom-path "$ROM_FALZAR" > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "   ❌ Failed to export. Skipping."
            continue
        fi
    else
        echo "   ✅ Exported (Gregar)"
    fi

    # --- STEP 2: DETECT PERSPECTIVE ---
    SWAP_FLAG=""
    if [ -f "$RAW_DIR/${filename}.jsonl" ]; then
        python3 scripts/detect_swap.py "$RAW_DIR/${filename}.jsonl"
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 1 ]; then
            echo "   🔄 Detector says: SWAP (I am Entity B)"
            SWAP_FLAG="--swap-players"
        else
            echo "   ➡️ Detector says: KEEP (I am Entity A)"
        fi
    else 
        echo "   ❌ Error: No JSONL generated."
        continue
    fi

    # --- STEP 3: CONVERT & CLEAN ---
    mkdir -p "$FINAL_DIR/$filename"
    
    python3 scripts/convert_dataset.py \
        --input "$RAW_DIR/${filename}.jsonl" \
        --output "$FINAL_DIR/$filename/actions.jsonl" \
        $SWAP_FLAG

    # Move Video
    if [ -f "$RAW_DIR/${filename}.mp4" ]; then
        mv "$RAW_DIR/${filename}.mp4" "$FINAL_DIR/$filename/video.mp4"
    fi

    # [FIX] Move Static Data
    if [ -f "static_data.json" ]; then
        mv "static_data.json" "$FINAL_DIR/$filename/static_data.json"
        echo "   💾 Static Data Saved"
    else
        echo "   ⚠️ Warning: No static_data.json found!"
    fi
    
    echo "   ✅ Done: $FINAL_DIR/$filename"
done