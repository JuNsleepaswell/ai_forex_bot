import subprocess
import time
import sys  # <--- Add this line

# The list of all the pairs you want to train tonight
TICKERS = [
    "EURUSD",
    "GBPUSD",
    "XAUUSD",
    "USDCAD",
    "USDJPY",
    "AUDUSD",
    "NZDUSD"
]


def launch_fleet():
    print("==================================================")
    print("🚀 INITIATING OVERNIGHT PRO FLEET TRAINING 🚀")
    print("==================================================")

    start_time = time.time()

    for ticker in TICKERS:
        print(f"\n[{time.strftime('%H:%M:%S')}] Dispatching agent for {ticker}...")

        # This runs the training script completely isolated, protecting your VRAM
        try:
            subprocess.run(
                [sys.executable, "src/drl_training_pro.py", "--ticker", ticker],
                check=True
            )
            print(f"✅ {ticker} Agent successfully trained and saved.")
        except subprocess.CalledProcessError:
            print(f"❌ ERROR: {ticker} Agent failed. Moving to next pair.")

        # Give the GPU 5 seconds to cool down and clear memory buffers
        time.sleep(5)

    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\n==================================================")
    print(f"🏆 FLEET TRAINING COMPLETE! Total Time: {int(hours)}h {int(minutes)}m")
    print("==================================================")


if __name__ == "__main__":
    launch_fleet()