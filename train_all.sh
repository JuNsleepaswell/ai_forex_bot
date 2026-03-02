#!/bin/bash

# Set the GPU memory growth environment variable for this script
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Define the list of tickers to train
TICKERS="EURUSD GBPUSD AUDUSD NZDUSD USDCAD USDCHF USDJPY"

# Loop through each ticker and run the training script
for TICKER in $TICKERS
do
  echo "----------------------------------------------------"
  echo "Starting training for $TICKER"
  echo "----------------------------------------------------"
  python src/05_lstm_model_training.py --ticker $TICKER
  # Check if the last command was successful before continuing
  if [ $? -ne 0 ]; then
    echo "ERROR: Training for $TICKER failed. Aborting script."
    exit 1
  fi
done

echo "----------------------------------------------------"
echo "All training scripts completed successfully."
echo "----------------------------------------------------"
