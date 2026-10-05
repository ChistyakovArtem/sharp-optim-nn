# Sharpe-Based DL for MFT (S&P 500)

This repository contains my implementation of the ideas presented in the article [*Deep Learning for Portfolio Optimization*](https://arxiv.org/pdf/2005.13665), which has over 200 citations, along with my adaptations, use cases, and the results of applying this technique to mid-frequency trading.

# Results

- **Reproduced and extended** the method of [Zhang, Zohren & Roberts](https://arxiv.org/abs/2005.13665) for training neural networks directly on the Sharpe ratio net of transaction costs, moving it from daily data to a multi-asset **mid-frequency trading (MFT)** setting: two years of minute-level data for 488 S&P 500 stocks.
- **Resolved optimization collapse** in Sharpe-based training with a staged loss schedule (*Sharpe → Sharpe-PnL → PnL*), which improved backtest PnL by roughly **3×** on the chronological holdout without degrading Sharpe.
- **Built the training pipeline** for minute-level multi-asset data, including a **~200× speed-up** in data loading through offset-indexed CSV chunk reading and chunked loss evaluation.
- **Evaluated on a chronological holdout** (train July 2023 – December 2024, test January – July 2025) under a simplified proportional fee of 1.53 bp of traded volume. Bid–ask spreads, market impact and borrow costs are **not** modeled, and the best models concentrate in a small number of assets, so the backtest curves below are an exploratory check of the training procedure, not an estimate of achievable live performance.
- **Documented next steps**: position and turnover constraints, stacking of strategies, and MSE pretraining before Sharpe fine-tuning.

# Original Article

[*Deep Learning for Portfolio Optimization*] by Zihao Zhang, Stefan Zohren, and Stephen Roberts is a compelling article demonstrating the power of neural networks for complex portfolio optimization problems.  
Traditionally, machine learning has been used to predict price movements, volatility, or autocorrelations. These predictions were then fed into stochastic models or heuristics to generate trading strategies.  
This paper, however, trains a neural network **directly** to maximize the **Sharpe Ratio** of a portfolio — accounting for transaction costs.  
Using 14 years of daily candle data and a fee level of `1e-4`, the authors developed a functioning portfolio management strategy that outperformed the market.
Their strategy achieved a Sharpe ratio of **1.96**, compared to the market Sharpe of **1.52** over the same period.

I found the idea very interesting, and since no public code implementation was available, I decided to build it myself. I also explored how this approach would work in a different setting — **mid-frequency trading** (MFT).  
I was curious to see what kinds of strategies the neural network would converge to, and under what transaction cost regimes. This was my summer research practice at HSE University (July–August 2025).

# My Case & Adaptations

I used two years of **minute-level candle data** (from July 10, 2023, to July 3, 2025) for 488 out of the 500 S&P 500 stocks.  
The stock universe was fixed as of **June 2023** to avoid lookahead bias (e.g., PLTR was excluded), and 12 stocks were missing due to delisting, mergers, or data unavailability.  
All of this data can be freely retrieved using the [Polygon.io](https://polygon.io) API.

# Hypotheses and Experiments

## I tried three different sets of fees

- **Market-estimated fees**: \$0.003 per share (like in real NYSE/NASDAQ) + spread estimation by price and market cap.  
  My strategy learned to abuse this by trading stocks with the highest absolute price (almost entirely avoiding taker fees):

  - BKNG (5,457.86)
  - FICO (1,343.12)
  - GWW (935.63)
  - AZO (4,011.25)
  - NVR (7,906.91)  
  ...

  These stocks had the highest average allocation weights.  
  I discarded this setting, since I believe that such high prices likely imply wide spreads and the effect was likely a trivial inefficiency.

> This is not the first time my model came up with a surprising idea I hadn't thought of.

- **Volume-based fee**: `1.53e-4` of USD volume (based on the best Binance futures fees).  
  This is similar to the `1e-4` fee used in the original article. All assets are treated equally.  
  All experiments were performed under this fee setup unless stated otherwise.

- **Scaled-up fees**: `1.53e-4 * k`  
  What happens if fees are so high that directional (alpha-style) trading becomes impossible?  
  Will the model learn to do actual **portfolio management** (beta-style allocation)?

## Models

### Simple Allocator

```python
class SimplePortfolioAllocator(nn.Module):
    def __init__(self, cmf_dim=50, num_assets=500):
        super().__init__()
        self.num_assets = num_assets
        self.tmp_simple_linear = nn.Linear(cmf_dim, num_assets + 1)

    def forward(self, cmf, asset_features):
        """
        cmf: [batch_size, cmf_dim]
        asset_features: [batch_size, num_assets, asset_dim]
        """
        weights = self.tmp_simple_linear(cmf)
        weights = torch.softmax(weights, dim=-1)
        return weights
```

### Deep Allocator

```python
class DeepPortfolioAllocator_1(nn.Module):
    def __init__(self, cmf_dim=50, num_assets=500, asset_dim=8):
        super().__init__()
        self.num_assets = num_assets
        self.asset_dim = asset_dim

        self.cmf_net = nn.Sequential(
            nn.Linear(cmf_dim, 32),
            nn.ELU(),
            nn.Linear(32, num_assets + 1),
            nn.ELU()
        )

        self.shared_asset_net = nn.Sequential(
            nn.Linear(asset_dim + 1, 4),
            nn.ELU(),
            nn.Linear(4, 1)
        )

    def forward(self, cmf, asset_features):
        """
        cmf: [batch_size, cmf_dim]
        asset_features: [batch_size, num_assets, asset_dim]
        """
        batch_size = cmf.shape[0]
        cmf_out = self.cmf_net(cmf)  # [batch_size, num_assets + 1]
        asset_bias = cmf_out[:, :-1].unsqueeze(-1)  # [batch_size, num_assets, 1]
        cash_bias = cmf_out[:, -1:]  # [batch_size, 1]

        concat = torch.cat([asset_features, asset_bias], dim=-1)  # [batch_size, num_assets, asset_dim + 1]
        asset_scores = self.shared_asset_net(concat).squeeze(-1)  # [batch_size, num_assets]
        all_scores = torch.cat([asset_scores, cash_bias], dim=-1)  # [batch_size, num_assets + 1]

        weights = torch.softmax(all_scores, dim=-1)
        return weights
```

Deep Allocator has big problems with convergence. This is very common among these types of articles to use one layer.  
I believe this is because Sharpe Ratio with fees is a very bad loss to optimize into (compared to things like MSE or CrossEntropy).

## About the losses

Trying to fit on other losses like **PnL** or **Sharpe-PnL** (about it later) often results in **“Mode Collapsing”**, where the strategy converges to a local optimum (e.g. buy-and-hold of one asset), even though a smarter strategy outperforms it even on train.  
This problem can be mitigated by fitting on different losses **step-by-step**:  
**first into Sharpe**, then into **Sharpe-PnL**, then into **just PnL**.

> **Why Sharpe-PnL?**  
> With a normal Sharpe ratio, the model can put 10× less money into assets than possible and still have the same Sharpe:  
> `Sharpe(returns) = Sharpe(returns / 10)`.  
>  
> **Sharpe-PnL** fixes this by scaling Sharpe:  
> `Sharpe-PnL(returns) = Sharpe(returns) * |mean(returns)|`.

---

### Code of 3 losses (Sharpe, PnL, and Sharpe-PnL):

```python
def compute_loss(self, r_net: torch.Tensor, is_train: bool = True):
    # During training, optimize the loss (e.g., Sharpe or PnL),
    # but during validation also track true Sharpe ratio and plot PnL for monitoring.
    mean = r_net.mean()
    std = r_net.std(unbiased=False) + self.eps

    if self.loss_type == 'sharpe' or not is_train:
        scaler = 1
    elif self.loss_type == 'sharpe-pnl':
        # Scale Sharpe loss to match PnL scale for stable learning rate across loss types
        scaler = mean.abs().mean() * 10000  # scalers for losses to have similar scale and hence the same learning rates
    elif self.loss_type == 'pnl':
        # Scale Sharpe loss to match PnL scale for stable learning rate across loss types
        scaler = std * 10000

    sharpe = mean * scaler / std * np.sqrt(self.intervals_per_year)
    return -sharpe
```
# Pipelines & Tech Core

In order to fit the neural network on this large amount of data with a non-trivial loss function, I had to perform data loading, time-series feature construction, and loss evaluation on smaller chunks.

- The batch size was set to **1000** (minute candles). Increasing this number did not improve performance but slowed down training.  
  A model trading on minute frequency does not need to see the market state over two full days.

- I implemented **advanced CSV mapping with offsets** to retrieve a sample of size `k=1000` from `O(k)` time, rather than `O(n)` where `n ≈ 200,000` (the total number of rows).  
  The Sharpe ratio loss was also computed in chunks to avoid storing large gradients and overflowing RAM.
```python
import os
import io
import pandas as pd

class FastCSVChunkReader:
    def __init__(self, path, padding=0, batch_size=1000, has_header=True):
        self.path = path
        self.padding = padding
        self.batch_size = batch_size
        self.offsets = []
        self.has_header = has_header
        self._build_offsets()
        self.current_idx = 0
        self.asset_names = []
        self.end_idx = len(self.offsets) - 1

    def _build_offsets(self):
        with open(self.path, 'rb') as f:
            pos = f.tell()
            if self.has_header:
                f.readline()  # skip header
                pos = f.tell()
            self.offsets.append(pos)
            while line := f.readline():
                self.offsets.append(f.tell())

        self.total_rows = len(self.offsets)

    def __iter__(self):
        return self

    def set_split(self, start_row: int, end_row: int):
        self.current_idx = start_row
        self.end_idx = end_row

    def __next__(self):

        if self.current_idx >= self.end_idx:
            raise StopIteration

        start = self.current_idx
        end = min(self.current_idx + self.batch_size + self.padding, self.end_idx)

        with open(self.path, 'rb') as f:
            f.seek(self.offsets[start])
            lines = f.read(self.offsets[end] - self.offsets[start]).decode('utf-8')
            if self.has_header:
                with open(self.path, 'r') as h:
                    header = h.readline()
                result = header + lines
            else:
                result = lines

        self.current_idx += self.batch_size
        tmp = pd.read_csv(io.StringIO(result)).set_index('Unnamed: 0').rename_axis(None)
        tmp.index = pd.to_datetime(tmp.index)
        self.asset_names = tmp.columns.tolist()
        return tmp

    def reset(self):
        self.current_idx = 0
```

- Additionally, **padding** was added to enable time-series feature computation based on previous observations, ensuring **stationarity** of features across different positions in the batch.

# Features

## Common Market Features

I used **market-cap-weighted returns** across 18 different subsets of assets. Below is the code used to construct common market features:

```python
market_caps = self.min_prices.values * self.info['Shares'].values
market_shares = market_caps / market_caps.sum(axis=1, keepdims=True)
market_shares = pd.DataFrame(market_shares, index=self.min_prices.index, columns=self.min_prices.columns)

returns = self.min_prices.pct_change().fillna(0)
returns[self.min_prices.index.to_series().dt.date.shift(1) != self.min_prices.index.to_series().dt.date] = 0

# Info for loss
self.future_returns = self.min_prices.shift(-1).pct_change().fillna(0)
self.future_returns[self.min_prices.index.to_series().shift(-1).dt.date != self.min_prices.index.to_series().dt.date] = 0
self.minute_prices = self.min_prices
self.market_caps = pd.DataFrame(market_caps, index=self.min_prices.index, columns=self.min_prices.columns)
```

Then I defined masks over different industries, sectors and other subsets.
The classification of sectors and industries follows the [GICS](https://www.msci.com/our-solutions/indexes/gics) (Global Industry Classification Standard), used by MSCI and S&P.


```python
bool_masks = {
    'Ind:Semiconductors': self.info['Industry'] == 'Semiconductors',
    'Ind:Semiconductors without NVDA': (self.info['Industry'] == 'Semiconductors') & (self.info.index != 'NVDA'),
    'Ind:Integrated Oil & Gas': self.info['Industry'] == 'Integrated Oil & Gas',
    'Ind:Diversified Banks': self.info['Industry'] == 'Diversified Banks',
    'Ind:Investment Banking & Brokerage': self.info['Industry'] == 'Investment Banking & Brokerage',
    'Ind:Aerospace & Defense': self.info['Industry'] == 'Aerospace & Defense',
    'Ind:Financial Exchanges & Data': self.info['Industry'] == 'Financial Exchanges & Data',
    'Ind:Pharmaceuticals': self.info['Industry'] == 'Pharmaceuticals',

    'Sec:Energy': self.info['Sector'] == 'Energy',
    'Sec:Industrials': self.info['Sector'] == 'Industrials',
    'Sec:Consumer Discretionary': self.info['Sector'] == 'Consumer Discretionary',
    'Sec:Health Care': self.info['Sector'] == 'Health Care',
    'Sec:Financials': self.info['Sector'] == 'Financials',
    'Sec:Information Technology': self.info['Sector'] == 'Information Technology',
    'Sec:Communication Services': self.info['Sector'] == 'Communication Services',
    'Sec:Real Estate': self.info['Sector'] == 'Real Estate',

    'Add:All': pd.Series(True, index=self.info.index),
    'Add:Mag7': self.info.index.isin(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']),
}

bool_masks = pd.DataFrame(bool_masks, index=self.info.index)
weighted_returns = (
    (returns.values * market_shares.values) @ bool_masks.values
) / (
    market_shares.values @ bool_masks.values
)
```

I also added rolling averages and standard deviations of these weighted returns:

```python
for horizon in [5, 10, 30]:
    weighted_returns_tmp_mean = weighted_returns_orig.rolling(window=horizon).mean().add_suffix(f'_mean_{horizon}')
    weighted_returns_tmp_std = weighted_returns_orig.rolling(window=horizon).std().add_suffix(f'_std_{horizon}')
```
## Asset-Specific Features

Some architectures use **asset-specific features** to predict the future performance of each asset — and therefore to adjust the weight allocation on a per-asset basis.

These features are calculated from historical price movements of each asset individually:

```python
def get_asset_specific_features(self):
    features_list = []

    pct_change = self.min_prices.pct_change().fillna(0)
    pct_change[self.min_prices.index.to_series().shift(1) != self.min_prices.index.to_series()] = 0

    features_list.append(pct_change)                           # Instant returns
    features_list.append(pct_change.rolling(window=10).std())  # 10-minute volatility
    features_list.append(pct_change.rolling(window=20).std())  # 20-minute volatility
    features_list.append(pct_change.rolling(window=10).mean()) # 10-minute trend
    features_list.append(pct_change.rolling(window=20).mean()) # 20-minute trend

    return np.stack([f.values for f in features_list], axis=2)  # shape: (n_time, n_assets, n_features)
```

These features are used as the `asset_features` input in architectures like `DeepPortfolioAllocator`, where each asset is processed with its own features and combined with global context.

# PnL on Test Set (2025)

> **Caveat.** All curves use the simplified volume-proportional fee only. Spreads, market impact and borrow costs are not modeled, and several runs concentrate in one or two assets, so these are diagnostics of the optimization behaviour rather than performance claims.

Due to notebook saving issues, the PnL plots are currently messy (e.g., X-axis shows numeric indices instead of dates). I plan to rerun the experiments and replace the current images with cleaner versions in the future.

Train period: **July 2023 – December 2024 (inclusive)**  
Test period: **All available 2025 (until July)**

---

## 1. "Market Fees" — Simple Allocator
[Notebook](https://github.com/ChistyakovArtem/share-optim-nn/blob/exps/bs%3D1000/me_fees_lr%3D1e-1/start.ipynb)

![PnL - Simple Allocator](images_for_readme/test_1.png)

---

## 2. "Market Fees" — Deep Allocator  
[Notebook](https://github.com/ChistyakovArtem/share-optim-nn/tree/exps/af_incl/best_result_so_far)

![PnL - Deep Allocator](images_for_readme/test_2.png)

---

## 3. Default Fees — Sharpe Ratio Loss  
[Notebook](https://github.com/ChistyakovArtem/share-optim-nn/blob/exps/second_try_2_models/start.ipynb)

![PnL - Sharpe Loss](images_for_readme/test_3.png)  
![Average Weights](images_for_readme/avg_weights_3.png)

---

## 4. Default Fees — Sharpe-PnL Loss  
[Notebook](https://github.com/ChistyakovArtem/share-optim-nn/blob/exps/08-07/sharpe-pnl_and_pnl_losses/start.ipynb)

![PnL - Sharpe-PnL Loss](images_for_readme/test_4.png)

---

## 5. Default Fees — PnL Loss  
[Notebook](https://github.com/ChistyakovArtem/share-optim-nn/blob/exps/08-07/sharpe-pnl_and_pnl_losses/start.ipynb)

![PnL - PnL Loss](images_for_readme/test_5.png)

---

## 6. 3x Default Fees — Portfolio Manager Strategy
[Notebook](https://github.com/ChistyakovArtem/share-optim-nn/blob/exps/08-06/3x_fees_example/start.ipynb)

![PnL - PM Strategy](images_for_readme/test_6.png)  
![Average Weights](images_for_readme/avg_weights_6.png)

With fees too big for alpha trading, model downgraded to Portfolio Management.

# Future Ideas to work with

- **Change of constraints**  
  As we see in results, the best model just found one asset and abused it. Since we trade MFT (and not a billion-dollar PM strategy), we should move to assigning each instrument the max position and max delta position (to account for liquidity).  
  These constraints can be written as:

  ```
  Sum(abs(w))       <= C1  
  Max(abs(w))       <= C2  
  Sum(abs(delta w)) <= C3  
  Max(abs(delta w)) <= C4
  ```

- **Stacking of Strategies**  
  If we try to optimize the Sharpe Ratio, our strategy tends to stay in position in a small subset of assets.  
  This way, by building a directional strategy, we remove the possibility for a PM strategy alongside it.  
  This can be fixed by trying to make a new strategy with new constraints such that **together** they satisfy the constraints above.  
  This can be achieved by:
  - Iteratively adding strategies
  - Or by fitting weights for multiple strategies together

  I tried the first option, but due to a cash restriction (I forced it to close a position if the first strategy wanted to open it), the model downgraded to holding cash.

- **Convergence and model size**  
  Right now, with ~100 common market features and 488 assets, the simplest model has `100 × 488` parameters.  
  We can try to reduce complexity by transforming the 100 features into a market embedding and stacking it alongside asset-specific features.  
  However, all these architectures fail to converge.  
  It may be fixed by first training the model on MSE for future returns and then fine-tuning it with Sharpe Ratio loss — but this is a task for upcoming months.

- **Using CatBoost predictors as features**  
  Since on Kaggle, CatBoost and similar GBMs are often superior to Neural Networks, they can be used for the **forecasting** part,  
  with a Neural Network doing the **allocation** part (instead of traditionally used heuristics).

- **Encoding asset structure**  
  Right now, the model selects a small portion of assets with no visible grouping (besides the moment where, with "market fees", the model selected those with the highest absolute price).  
  Probably, the inefficiencies in these assets do not depend on any conventional market features (like grouping by capitalization, sector, etc.).  
  This limits architectures — I have to assume that the model must fit different weights for different assets at some point, with probably no way to fix it just by adding features.
