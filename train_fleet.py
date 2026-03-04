import subprocess
import time
import sys # <--- 1. Import sys

TICKERS = ["EURUSD", "GBPUSD", "XAUUSD", "USDCAD", "USDJPY", "AUDUSD", "NZDUSD"]

def start_fleet():
    processes = []
    print(f"🚀 Launching {len(TICKERS)} agents in parallel using {sys.executable}...")

    for ticker in TICKERS:
        # 2. Use sys.executable instead of "python"
        p = subprocess.Popen([sys.executable, "src/drl_training_pro.py", "--ticker", ticker])
        processes.append(p)
        print(f"  Started {ticker}...")
        time.sleep(15)

    for p in processes:
        p.wait()

    print("✅ All agents have completed training!")

if __name__ == "__main__":
    start_fleet()