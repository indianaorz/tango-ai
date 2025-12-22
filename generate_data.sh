#!/bin/bash
set +e 

# --- CONFIGURATION ---
TANGO_EXE="./dist/tango-x86_64-linux.AppImage"
RAW_DIR="data/raw"
FINAL_DIR="data/dataset"
ROM_GREGAR="roms/bn6_gregar.gba"
ROM_FALZAR="roms/bn6_falzer.gba"

# Ensure directories exist
mkdir -p "$RAW_DIR"
mkdir -p "$FINAL_DIR"

# Loop through all replays in the folder
for replay in replays/*.tangoreplay; do
    [ -e "$replay" ] || continue
    filename=$(basename "$replay" .tangoreplay)
    
    # Filter: Only process BN6 replays
    if [[ "$filename" != *"bn6"* ]]; then continue; fi

    echo "========================================"
    echo "Processing: $filename"

    # --- STEP 1: EXPORT VIDEO & RAW JSONL ---
    # Only export if the raw JSONL doesn't already exist (speed optimization)
    if [ ! -f "$RAW_DIR/${filename}.jsonl" ]; then
        # Try Gregar ROM first
        $TANGO_EXE export "$replay" --output-path "$RAW_DIR/${filename}.mp4" --rom-path "$ROM_GREGAR" > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "   ⚠️ Gregar failed. Retrying with Falzar..."
            # Fallback to Falzar ROM
            $TANGO_EXE export "$replay" --output-path "$RAW_DIR/${filename}.mp4" --rom-path "$ROM_FALZAR" > /dev/null 2>&1
            if [ $? -ne 0 ]; then
                echo "   ❌ Failed to export with both ROMs. Skipping."
                continue
            fi
        else
            echo "   ✅ Exported (Gregar)"
        fi
    else
        echo "   ⏩ Skipping Export (Raw file exists)"
    fi

    # --- STEP 2: DETECT PERSPECTIVE (The "Magic" Check) ---
    # Use the python detector to correlate inputs with movement.
    # Exit Code 0 = Keep (I am Player A/Left). 
    # Exit Code 1 = Swap (I am Player B/Right).
    
    SWAP_FLAG=""
    
    python3 scripts/detect_swap.py "$RAW_DIR/${filename}.jsonl"
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 1 ]; then
        echo "   🔄 Detector says: SWAP (I am Entity B)"
        SWAP_FLAG="--swap-players"
    else
        echo "   ➡️ Detector says: KEEP (I am Entity A)"
    fi

    # --- STEP 3: CONVERT & CLEAN ---
    # Create the final dataset folder for this replay
    mkdir -p "$FINAL_DIR/$filename"
    
    # Run conversion with the determined Swap Flag
    python3 scripts/convert_dataset.py \
        --input "$RAW_DIR/${filename}.jsonl" \
        --output "$FINAL_DIR/$filename/actions.jsonl" \
        $SWAP_FLAG

    # Move the MP4 video to the final folder
    if [ -f "$RAW_DIR/${filename}.mp4" ]; then
        mv "$RAW_DIR/${filename}.mp4" "$FINAL_DIR/$filename/video.mp4"
    fi
    
    echo "   ✅ Done: $FINAL_DIR/$filename"
done