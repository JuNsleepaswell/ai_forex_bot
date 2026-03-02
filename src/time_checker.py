import MetaTrader5 as mt5
from datetime import datetime

# Initialize
if not mt5.initialize():
    print("Failed to initialize")
    quit()

# --- FIX: Use the symbol with the suffix ---
symbol = "EURUSD"
# -------------------------------------------

tick = mt5.symbol_info_tick(symbol)

if tick:
    # Convert the timestamp to a readable format
    server_time = datetime.fromtimestamp(tick.time)
    local_time = datetime.now()

    print(f"--- TIME CHECK ---")
    print(f"Your Local Time:    {local_time.strftime('%H:%M:%S')}")
    print(f"Broker Server Time: {server_time.strftime('%H:%M:%S')}")

    # Calculate offset
    offset = server_time.hour - local_time.hour
    print(f"Hour Difference:    {offset} hours")
else:
    print(f"Symbol '{symbol}' not found. Make sure it is in Market Watch.")

mt5.shutdown()