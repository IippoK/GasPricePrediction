# Gas Price Prediction

A machine learning model that predicts Finnish E95 pump prices one week ahead
using Brent crude oil prices, USD/EUR exchange rates, and historical fuel prices.

## Approach
- E95 price data for Finland collected via a custom web scraper
- Brent crude price data and USD/EUR rates downloaded manually and cleaned with a script
- Features include lag variables, rate-of-change, and cyclical seasonal encoding (sin/cos)
- Ridge regression on polynomial features to prevent overfitting
- Chronological train/test split to respect time series structure

## Results
| | MAE | R² |
|---|---|---|
| Ridge model | 0.043 €/liter | 0.881 |
| Naïve baseline | 0.038 €/liter | 0.900 |

The naïve baseline (last week's price = next week's price) outperforms the model,
reflecting the high autocorrelation of fuel prices.

## Built with
Python, scikit-learn, pandas, numpy

## Data sources
Collected weekly via custom scraper: Finnish E95 pump prices, Brent crude oil, USD/EUR rate

## Anomaly Detection
Implemented rolling z-score analysis to automatically flag 
statistically significant price surges and crashes. 
Identified 7 distinct market events (2005–2024), including 
the 2008 financial crisis, COVID-19 demand collapse, 
and the 2022 energy crisis following Russia's invasion of Ukraine.

## Future work
Weekly-resolution data limits forecast granularity. 
Daily pump-price data could enable a consumer-facing 
price prediction tool, pending further model validation 
against the naïve baseline.
