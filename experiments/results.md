## Results

### Dataset
- Jena Climate (scaled using train-only statistics)
- Window: 72, Horizon: 1, Target col: 0

### Test Metrics
| Model | MAE | RMSE |
|---|---:|---:|
| Naive last-value | 0.0084 | 0.0122 |
| LSTM | 0.1644 | 0.2094 |

Notes:
- Naive is a strong baseline for short-horizon forecasting.
- LSTM should beat naive if it learns temporal patterns beyond last value.
