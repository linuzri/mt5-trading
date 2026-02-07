# MT5 Trading Bot

Automated trading bots for MetaTrader 5 with machine learning signal prediction.

## Supported Pairs

| Bot | Symbol | ML Model | Status |
|-----|--------|----------|--------|
| BTCUSD | Bitcoin/USD | Random Forest | ✅ Live |
| XAUUSD | Gold/USD | XGBoost | ✅ Live |
| EURUSD | Euro/USD | XGBoost | ✅ Live |

## Project Structure

```
mt5-trading/
├── btcusd/              # Bitcoin bot + ML pipeline
│   ├── trading.py       # Main bot logic
│   ├── ml/              # ML modules (features, training, prediction)
│   ├── models/          # Trained models + scalers
│   ├── config.json      # Bot configuration
│   └── train_ml_model.py
├── xauusd/              # Gold bot (same structure)
├── eurusd/              # Euro bot (same structure)
├── dashboard/           # Local web dashboard (Flask)
│   ├── server.py        # Dashboard backend
│   └── trading-dashboard.html
├── vercel-dashboard/    # Cloud dashboard (Vercel + Supabase)
│   └── index.html
├── ecosystem.config.js  # PM2 process manager config
├── daily_digest.py      # End-of-day performance summary
└── sync_to_supabase.py  # Historical data sync to cloud
```

## Features

- **ML-Powered Signals:** Each bot uses a trained model (Random Forest / XGBoost) with 28 features including technical indicators, crash detection, and market regime
- **Trend Momentum Filter:** Tracks last 3 trades — continues profitable streaks, blocks losing directions
- **Session-Aware Trading:** Adjusts confidence thresholds for Asian/EU/US sessions
- **Dynamic Position Sizing:** Risk-based lot calculation (0.5% per trade)
- **Spread Filter:** Skips trades when spread exceeds threshold
- **Market Hours Check:** Prevents orders during maintenance breaks
- **Supabase Sync:** Real-time trade data pushed to cloud for dashboard

## Quick Start

### Prerequisites
- Python 3.10+
- MetaTrader 5 terminal running
- MT5 Python package (`pip install MetaTrader5`)
- PM2 (`npm install -g pm2`)

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure MT5 credentials
# Create mt5_auth.json in each bot folder with your credentials

# Start all bots
pm2 start ecosystem.config.js
pm2 save

# Check status
pm2 status
```

### Train ML Models
```bash
cd btcusd
python train_ml_model.py --refresh

cd ../xauusd
python train_ml_model.py --refresh
```

## Dashboards

- **Local:** `cd dashboard && python server.py` → http://localhost:5000
- **Cloud:** https://trade-bot-hq.vercel.app

## Configuration

Each bot has its own `config.json`:

| Setting | Default | Description |
|---------|---------|-------------|
| `risk_percent` | 0.5% | Risk per trade |
| `max_lot` | 0.05 | Maximum lot size |
| `confidence_threshold` | 0.6 | ML prediction confidence minimum |
| `cooldown_minutes` | 5 | Wait time between trades |
| `off_hours` | true | Trade outside main sessions |

## PM2 Commands

```bash
pm2 status                # Check all bots
pm2 logs bot-btcusd       # View BTCUSD logs
pm2 restart bot-xauusd    # Restart Gold bot
pm2 stop all              # Stop everything
```

## License

MIT
