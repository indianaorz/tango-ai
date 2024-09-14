#!/bin/bash

# Path to the Tango AppImage
APP_PATH="./dist/tango-x86_64-linux.AppImage"

# Set the environment variables
export INIT_LINK_CODE="your_link_code"
export AI_MODEL_PATH="ai_model"

# export ROM_PATH="bn6,1"  # Replace with the actual path to your ROM file
# export SAVE_PATH="/home/lee/Documents/Tango/saves/BN6 Falzar 1.sav"  # Replace with the actual path to your save file


export ROM_PATH="bn6,0"  # Replace with the actual path to your ROM file
export SAVE_PATH="/home/lee/Documents/Tango/saves/BN6 Gregar.sav"  # Replace with the actual path to your save file

export MATCHMAKING_ID="your_matchmaking_id"  # Replace with the actual matchmaking ID

# Print the environment variables being used
echo "Running the application with the environment variables:"
echo "INIT_LINK_CODE: $INIT_LINK_CODE"
echo "AI_MODEL_PATH: $AI_MODEL_PATH"
echo "ROM_PATH: $ROM_PATH"
echo "SAVE_PATH: $SAVE_PATH"
echo "MATCHMAKING_ID: $MATCHMAKING_ID"

# Execute the command
"$APP_PATH"
