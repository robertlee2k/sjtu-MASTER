# Readme 
This is the official code and supplementary materials for our AAAI-2024 paper: **MASTER: Market-Guided Stock Transformer for Stock Price Forecasting**. MASTER is a stock transformer for stock price forecasting, which models the momentary and cross-time stock correlation and guide feature selection with market information.

![MASTER framework](framework.png)

Our original experiments was conducted in a complex business codebase developed based on Qlib. The original code is confidential and exhaustive. In order to enable anyone to quickly use MASTER and reproduce the paper's results, here we publish our well-processed data and core code. 

## Usage

---
1. Install dependencies.
- pandas == 1.5.3
- torch == 1.11.0

2. Install [Qlib](github.com/microsoft/qlib). We have minimized the reliance on Qlib, and you can simply install it by
- <code>pip install pyqlib </code>
- pyqlib == 0.9.1.99

3. Unpack data into <code> data/ </code>.

4. Run main.py.

5. We provide two trained models: <code> model/csi300master_0.pkl, model/csi800master_0.pkl</code>

## Dataset

---
### Form
Grouped by prediction dates, the published data is of shape N, T, F, where:
- N - number of stocks
- T - length of lookback_window, T=8.
- F - 158 factors + 63 market information + 1 label        

For reference, the market information is generated by the following pseudocode. Note m is shared by all stocks.

```python
m = []
for S in csi300, csi500, csi800:
  m += [market_index(S,-1)]
  for d in [5, 10, 20, 30, 60]:
    m += [historical_market_index_mean(S, d), historical_market_index_std(S, d)]
    m += [historical_amount_mean(S, d), historical_amount_std(S, d)]
```

### Preprocessing
The published data went through the following necessary preprocessing.
1. Drop NA features, and perform robust daily Z-score normarlization on each feature dimension.  
2. Drop NA label and 5% of the most extreme labels, and perform **daily Z-score normalization** on labels. 
Daily Z-score normalization is a common practice in Qlib to standardize the labels for stock price forecasting.
To mitigate the difference between a normal distribution and groundtruth distribution, we filtered out 5\% most extreme labels in training.
Note that the reported RankIC compares the output ranking with the groundtruth, whose value is not affected by the label normalization.

## Cite

---
If you use the data or the code, please cite our work! :smile:


