import subprocess
import time
import sys
import os
import torch # Add this to clear cache between runs

# Batches for 50% Load: [EURUSD, GBPUSD], then [XAUUSD, USDCAD], etc.
TICKERS = ["EURUSD", "GBPUSD", "XAUUSD", "USDCAD", "USDJPY", "AUDUSD", "NZDUSD"]
BATCH_SIZE = 2  # Set to 2 to keep load very low, or 3 for ~60% load


def start_fleet():
    fleet_version = time.strftime("%Y%m%d_%H%M")
    BATCH_SIZE = 2  # Keep this at 2 to avoid jams

    for i in range(0, len(TICKERS), BATCH_SIZE):
        batch = TICKERS[i:i + BATCH_SIZE]
        processes = []

        print(f"\n📦 Starting Batch: {batch}")
        for ticker in batch:
            # FIX 1: Use 'stdout=None' so the script prints directly to console
            # and doesn't jam the internal Python buffer.
            p = subprocess.Popen([
                sys.executable, "src/drl_training_pro.py",
                "--ticker", ticker, "--version", fleet_version
            ], stdout=None, stderr=None)
            processes.append(p)
            time.sleep(15)

        # FIX 2: Wait with a timeout and force-close
        for p in processes:
            try:
                p.wait()  # Wait for normal finish
            except Exception as e:
                print(f"⚠️ Process jammed, forcing close: {e}")
                p.kill()

        # FIX 3: THE MOST IMPORTANT PART
        # Force the GPU to empty its trash can before starting the next batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("🧹 GPU Cache Purged. Ready for next batch.")

    print(f"✅ Fleet {fleet_version} complete!")


if __name__ == "__main__":
    start_fleet()