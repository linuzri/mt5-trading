
# XAUUSD on Pepperstone:
# contract_size = 100, tick_value = 1.0, tick_size = 0.01
# P/L = lots * (price_change / tick_size) * tick_value
# P/L = lots * price_change * 100

print("=== XAUUSD Lot Size Investigation ===\n")

# Reverse calculate lot sizes from catastrophic trades
cases = [
    ("Feb 13 15:35", 199.90, 16.22),
    ("Feb 13 21:15", 149.20, 6.34),
    ("Feb 9 17:05", 120.60, 5.88),
    ("Feb 6 11:50", 71.45, 19.56),
    ("Feb 9 16:05", 65.90, 9.00),
]

print("--- Reverse-Engineering Lot Sizes ---")
for date, loss, move in cases:
    lot = loss / (move * 100)
    print(f"  {date}: Loss={loss:.2f}, Move={move:.2f}, Implied lot={lot:.4f}")

print("\n--- Position Sizer Calculation ---")
balance = 49430
risk_pct = 0.5
risk_amt = balance * risk_pct / 100
sl_pips = 40
calc_lot = risk_amt / sl_pips
print(f"  Balance: {balance}")
print(f"  Risk: {risk_pct}% = {risk_amt:.2f}")
print(f"  SL: {sl_pips}")
print(f"  Formula: risk_amt / sl_pips = {risk_amt:.2f} / {sl_pips} = {calc_lot:.4f} lots")
print(f"  Clamped to max 0.05")

print("\n--- The BUG ---")
print(f"  Position sizer: risk_amt / sl_pips = {calc_lot:.2f} lots (INSANE)")
print(f"  Gets clamped to 0.05 max, but...")
print(f"  0.05 lots * 40 SL * 100 contract = {0.05 * 40 * 100:.2f} actual risk")
print(f"  Intended risk was only {risk_amt:.2f}!")
print(f"")
print(f"  The formula IGNORES the contract multiplier (100)!")
print(f"  Correct formula: risk_amt / (sl_pips * contract_size)")
print(f"  Correct lot: {risk_amt:.2f} / ({sl_pips} * 100) = {risk_amt / (sl_pips * 100):.4f} lots")
print(f"  That's {risk_amt / (sl_pips * 100):.4f} lots for {risk_amt:.2f} risk")
print(f"  Actual loss at 0.05 lots with 40 SL = {0.05 * 40 * 100:.2f}")

print("\n--- But wait, the implied lots are 0.12-0.24! ---")
print(f"  Max lot is 0.05, so why are losses showing 0.12+ lots?")
print(f"  Possible: multiple positions open simultaneously, or partial profit")
print(f"  leaving bigger remaining position exposed")
