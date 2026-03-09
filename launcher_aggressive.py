import subprocess
import time
import sys

# Group your tickers by how they performed
WINNERS = ["XAUUSD"] # These use standard settings
UNDERPERFORMERS = ["EURUSD", "GBPUSD"] # These need a "push", "USDCAD", "NZDUSD"

def start_targeted_fleet():
    fleet_version = time.strftime("%Y%m%d_%H%M_Targeted")
    processes = []

    # 1. Launch the Winners with "Steady" settings
    for ticker in WINNERS:
        p = subprocess.Popen([
            sys.executable, "src/drl_training_pro.py",
            "--ticker", ticker,
            "--version", fleet_version,
            "--ent_coef", "0.05", # Focused on profit
            "--lr", "1e-4"        # Slow, stable learning
        ])
        processes.append(p)
        time.sleep(15)

    # 2. Launch the Underperformers with "Aggressive" settings
    for ticker in UNDERPERFORMERS:
        p = subprocess.Popen([
            sys.executable, "src/drl_training_pro.py",
            "--ticker", ticker,
            "--version", fleet_version,
            "--ent_coef", "0.12", # High entropy forces them to try new things
            "--lr", "3e-4"        # Faster learning to escape the 0% return trap
        ])
        processes.append(p)
        time.sleep(15)

    for p in processes:
        p.wait()

if __name__ == "__main__":
    start_targeted_fleet()