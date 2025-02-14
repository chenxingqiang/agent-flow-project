#!/bin/bash

# Check if OPENAI_API_KEY is provided as argument
if [ -z "$1" ]; then
    echo "Please provide your OpenAI API key as an argument"
    echo "Usage: source setup_env.sh YOUR_API_KEY"
    exit 1
fi

# Export the API key
export OPENAI_API_KEY="$1"

# Verify the key is set
echo "OPENAI_API_KEY has been set"

# Optional: Add any other environment variables needed for the project here 