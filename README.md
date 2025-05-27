# Trading Data Analytics

Advanced market microstructure analysis system for E-mini S&P 500 futures, combining footprint analysis, Auction Market Theory (AMT), and statistical validation to identify high-probability trading opportunities.

- [Overview](#-overview)
- [Market Microstructure Concepts](#-market-microstructure-concepts)
  - [Signal Detection Framework](#signal-detection-framework)
  - [Key Metrics & Calculations](#key-metrics--calculations)
- [Level-Based Trading System](#-level-based-trading-system)
  - [AMT Integration](#amt-integration)
  - [Level Proximity Filtering](#level-proximity-filtering)
- [Data Pipeline Architecture](#️-data-pipeline-architecture)
  - [Input Processing](#input-processing)
  - [Statistical Analysis](#statistical-analysis)
- [Technical Implementation](#-technical-implementation)
- [Files Structure](#-files-structure)

---

## Overview

This system processes **8-tick bar footprint data** to detect market reversal signals based on order flow imbalances and trapped trader scenarios. The analysis focuses on identifying **"Inversion"** and **"Exhaustion"** patterns that occur near significant support/resistance levels derived from Time-Price Opportunity (TPO) profile analysis.

---

## Market Microstructure Concepts

### Signal Detection Framework

#### Inversion Signals
*Detect trapped traders when footprint bars close opposite to their delta direction*

• **Standard Inversion**: Bar closes against its delta % direction, indicating absorption  
• **Strong Inversion**: Net delta closes within 75% of the bar's high/low wick (extreme absorption)  
• **Market Context**: Identifies scenarios where aggressive buyers/sellers become trapped  

#### Exhaustion Signals
*Identify momentum fatigue through volume tapering patterns*

• **2x Exhaustion**: Standard 50% volume reduction between consecutive price levels  
• **Accelerated Exhaustion**: Severe dropoff (50% to 20%) indicating extreme momentum loss  
• **Direction-Specific**: Separate classification for buy-side vs sell-side exhaustion  

### Key Metrics & Calculations

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Delta %** | `Delta ÷ Volume` | Measures directional aggression intensity |
| **Trapped Delta %** | `Cumulative Delta ÷ Minimum Delta` | Quantifies trapped trader exposure |
| **Vol\*** | `Current Volume ÷ 50-bar Rolling Avg` | Volume multiple vs historical average |
| **TD%** | `Close ÷ Low` (Buy) / `Close ÷ High` (Sell) | Entry timing relative to bar structure |

#### Example: Trapped Delta Calculation
```
Cumulative Delta: -200
Minimum Delta: -250
Trapped %: -200 ÷ -250 = 80%
→ High probability of bid pressure
```

---

## Level-Based Trading System

### AMT Integration
*Auction Market Theory-based level identification*

#### Composite Value Areas (CVA)
Multi-session merged profiles creating significant support/resistance:

• **Balance Areas**: Periods where market finds fair value (D-shaped distributions)  
• **Imbalance Moves**: Directional moves away from established value areas  
• **Level Significance**: Stronger levels come from longer timeframe merges  

#### Trade Scenarios

| Scenario | Description | Risk/Reward |
|----------|-------------|-------------|
| **SC1** - Return Pullback | Long pullbacks after acceptance into prior value | Standard |
| **SC2** - Rejection | Fade rejections off value area extremes | **Preferred** ⭐ |
| **SC3** - Balance Break | Trade breakouts from mini-auctions over levels | Advanced |

### Level Proximity Filtering

**12-Tick Rule**: Signals only trigger when price is within 12 ticks of significant levels

**Benefits:**
- Prevents low-probability trades in "no-man's land"
- Ensures confluence between microstructure signals and macro support/resistance
- Level significance determined by merge duration and market structure

---

## ⚙️ Data Pipeline Architecture

### Input Processing
*`process_footprint_data.py`*

#### Market Data Integration
• **Source**: Sierra Chart 8-tick bar exports (103 columns)  
• **Content**: OHLC, volume, bid/ask volume, delta metrics  
• **Signals**: Footprint signal classifications (Inversion/Exhaustion variants)  
• **Structure**: Market indicators (POC, VWAP, value areas)  

#### Signal Classification Engine
```
Raw Signals → Hierarchical Processing → Binary Classification
     ↓              ↓                        ↓
Multiple types   Strong overrides Weak   Buy/Sell × Inv/Exh
```

#### Forward Return Calculation
• **Timeframes**: 5, 10, 20, 30, 40, 50, 75, 100 bars  
• **Stop-Loss**: Entry bar Low/High ± 1 tick  
• **MFE Tracking**: Maximum Favorable Excursion for optimal exits  

### Statistical Analysis
*`footprint_analysis.py`*

#### Regression Framework
• **Interaction Testing**: Signal combination effectiveness  
• **Significance Filter**: P-value < 0.05  
• **Optimization**: Percentile-based threshold identification  

#### Performance Validation
• **Win Criteria**: MFE targets (25+ ticks)  
• **PnL Calculation**: ES contract specs ($50/tick)  
• **Sample Filter**: Minimum 30 observations  

---

## Technical Implementation

### Processing Pipeline
```
 Raw Data →   Signals →   Levels  →  Returns  → Validation
     ↓           ↓          ↓          ↓          ↓
 OHLC/Volume  Inversion   CVA       8 periods  Regression
 Delta data   Exhaustion  12-tick   Stop/MFE   Interactions
 Footprints   Strength    filter    tracking   Performance
```

### Key Features
• **Session Management**: 8:30 AM boundaries with proper handling  
• **Parallel Processing**: Multi-core optimization for 250+ trading days  
• **Memory Efficient**: Optimized handling of high-frequency datasets  
• **Signal Validation**: Multi-layer filtering ensuring quality  

### Performance Characteristics
| Metric | Value |
|--------|-------|
| **Data Coverage** | 250 trading days |
| **Signal Universe** | Inversion × Exhaustion × Volume × Levels |
| **Statistical Rigor** | P-value validation + interaction effects |
| **Risk Framework** | Dynamic stops + MFE optimization |

---

## Directory Structure

```
trading-data-analytics/
├── README.md
├── analysis/
│   ├── process_footprint_data.py        # Market data processing & signals
│   └── footprint_signal_analysis.py     # Statistical analysis & optimization
├── data/
│   ├── ES_8tick_250D.csv                # Raw Sierra Chart export
│   └── 8_tick_inv_and_ex.csv            # Processed signals + returns
└── requirements.txt                     # Python dependencies
```
