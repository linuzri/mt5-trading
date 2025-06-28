# mt5-trading
MT5 trading bot

## Strategy options for config.json

Set the `strategy` field in your `config.json` to select which strategy to use. 

```
"strategy": "ma_crossover"   # Moving Average Crossover (implemented)
"strategy": "rsi"            # RSI-based strategy (implemented)
"strategy": "macd"           # MACD-based strategy (implemented)
"strategy": "bollinger"      # Bollinger Bands breakout (implemented)
"strategy": "custom"         # Custom strategy (implement your own logic)
```

To use a strategy, set the `strategy` field in `config.json` to one of the above values.
