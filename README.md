# Trading Data Analytics

Python-based system for processing high-frequency futures data, detecting trading signals, and analyzing performance across multiple timeframes.

## Overview

This project processes 8-tick bar data from E-mini S&P 500 futures to identify and validate trading signals. The system combines market microstructure analysis with statistical validation to evaluate signal effectiveness.

## Data Pipeline

**Input**: Sierra Chart 8-tick bar exports with 103 columns including:
- Price data (OHLC, volume, bid/ask volume)
- Market structure indicators (POC, VWAP, value areas)
- Custom signals (Inversion/Exhaustion patterns)
- Order flow metrics (delta, volume outliers)

**Processing**:
1. Session-based data segmentation (8:30 AM start)
2. Signal classification and strength weighting
3. Support/resistance level proximity analysis
4. Forward return calculation (5-100 bars)
5. Stop-loss and MFE tracking

**Output**: Processed dataset with forward returns and performance metrics

## Signal Detection

### Inversion Signals
Market reversal patterns based on order flow imbalances:
- Buy/Sell classification with Strong/Weak variants
- Triggers only when price is within 12 ticks of significant levels
- Hierarchical logic prevents signal conflicts

### Exhaustion Signals  
Momentum exhaustion identification:
- 2x Exhaustion: Standard momentum breaks
- Accelerated Exhaustion: Extreme conditions
- Direction-specific buy/sell variants

## Analysis Framework

### Statistical Validation
- Regression analysis using statsmodels OLS
- Interaction effect testing between signal combinations
- P-value significance filtering (p < 0.05)
- Optimal threshold identification via percentile analysis

### Performance Metrics
- Forward returns across 8 timeframes (5-100 bars)
- Win rate calculation with configurable MFE targets
- PnL estimation using E-mini S&P 500 contract specifications
- Stop-loss implementation (1 tick beyond entry bar range)

## Files

```
analysis/
├── Process_Footprint_Data_Filters_-_Inversion_and_Close_BA_Proximity.py
└── Footprint_Analysis.py

data/
├── ES_8tick_250D.csv          # Raw Sierra Chart export
└── 8_tick_inv_and_ex.csv      # Processed data with signals
```

## Technical Details

**Dependencies**: pandas, numpy, statsmodels, seaborn, matplotlib, multiprocessing

**Performance**: Parallel session processing for handling 250 days of tick data

**Risk Management**: 
- Stop-loss: Low - 0.25 (buy signals), High + 0.25 (sell signals)
- Position sizing: Fixed contract with $50/tick value
- MFE tracking for exit optimization

**Data Validation**:
- Minimum sample size filtering (30+ observations)
- Missing data handling and chronological ordering
- Session boundary management across market days