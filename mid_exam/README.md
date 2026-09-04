# Rwanda Food Price Prediction & Market Clustering

**Course:** MSDA 9213 (Data Mining)
**Program:** MSc Big Data Analytics, Adventist University of Central Africa
**Academic Year:** 2025–2026, Semester 3

## 1. Project Overview

This project analyses retail food commodity prices across Rwanda (2020–2026), enriched with climate and macroeconomic data, to address the two supervised tasks and the clustering task required by the exam brief:

- **Classification**: will a commodity's price rise month-over-month?
- **Regression**: what is the retail price level?
- **Clustering**: which market–commodity pairs share a price/climate profile?

The data was deliberately chosen to be **real, freely accessible, and regionally relevant** (East Africa), avoiding sources requiring lengthy access approval (e.g. DHS microdata), so that data collection would not put the project timeline at risk.

## 2. Objectives

| Task | Target | Type |
|---|---|---|
| Classification | `price_direction` (1 = price rose vs. previous month) | Binary classification |
| Regression | `log1p(price_rwf)`, back-transformed with `expm1()` | Regression |
| Clustering | Market–commodity profile (price level, volatility, temperature, precipitation, soil moisture) | Unsupervised segmentation |

## 3. Data Sources

| Source | Content | Access |
|---|---|---|
| [WFP Food Prices](https://data.humdata.org/dataset/wfp-food-prices) (HDX) | Monthly retail food prices by market | Free, no registration |
| [NASA POWER](https://power.larc.nasa.gov/) | Daily temperature (T2M), precipitation (PRECTOTCORR), soil moisture (GWETPROF) | Free, no registration |
| [Yahoo Finance](https://finance.yahoo.com/) (`RWF=X`) | Monthly USD/RWF exchange rate | Free, no registration |
| [NISR](https://www.statistics.gov.rw/) | Rwanda CPI (general and food) | Free Excel download |

**Scope:** 5 provinces, 33 markets, 14 commodity groups. Exchange rate and CPI are national-level series with no provincial breakdown.

**Coverage limitation:** WFP market reporting is opportunistic, not a designed survey. Eastern and Southern Provinces are heavily over-represented; Kigali City and Northern Province are sparse. Findings for the under-represented provinces are flagged as lower-confidence throughout rather than given equal weight.

## 4. Repository Structure

```text
├── data/
│   ├── raw/                              # WFP global CSVs, NASA POWER pull, FX, NISR CPI
│   └── cleaned/                          # Merged, feature-engineered dataset
├── figures/                              # Plots referenced by report/main.tex
├── models/                               # Serialized tuned models (joblib)
├── report/
│   ├── main.tex                          # Scientific paper
│   ├── references.bib
│   └── tables/                           # AUTO-GENERATED .tex tables (do not edit)
├── 0_data_extraction.ipynb               # Pulls climate, FX, and CPI data
├── 1_data_cleaning.ipynb                 # Filters, merges, engineers features
├── 2_exploratory_data_analysis.ipynb     # EDA
├── 3a_supervised_learning.ipynb          # Classification + regression
├── 3b_unsupervised_clustering.ipynb      # Clustering
├── requirements.txt
└── README.md
```

Notebooks must be run in numeric order; each reads the previous stage's output.

**The paper's numbers are generated, not hand-copied.** Notebooks `3a` and `3b` write LaTeX table fragments and `\newcommand` definitions into `report/tables/`, which `main.tex` pulls in via `\input{}`. Re-running the notebooks and recompiling the paper refreshes every result automatically, so the write-up cannot silently drift out of sync with the code.

## 5. Methodology

1. **Data collection** (`0_data_extraction.ipynb`) Rwanda prices extracted from the global WFP file via DuckDB streaming query; climate variables from NASA POWER per province centroid; FX from Yahoo Finance; CPI parsed from the NISR Excel release.
2. **Data cleaning** (`1_data_cleaning.ipynb`) filtered to KG-denominated, Retail, `actual`-flagged records; merged climate on `(date, province)` and macro series on `date`; engineered anomalies, lags, and rolling averages on an isolated per-province timeline.
3. **EDA** (`2_exploratory_data_analysis.ipynb`) verified data quality empirically, characterised both targets, and directly tested whether the newly added features actually separate rising from falling prices.
4. **Objectives** set from what the EDA showed, not decided in advance.
5. **Supervised learning** (`3a`) four algorithms per task including an MLP as the deep learning model, all inside `Pipeline` objects.
6. **Evaluation** precision/recall/F1/ROC-AUC for classification (accuracy alone is misleading under imbalance); RMSE/MAE/R² for regression.
7. **Improvement** `RandomizedSearchCV` under `TimeSeriesSplit`.
8. **Before/after comparison** tuned vs. untuned, reported as obtained.
9. **Clustering** (`3b`) K-Means, Agglomerative (Ward), and Gaussian Mixture compared by silhouette score and Adjusted Rand Index.

### Data leakage: three defects found and corrected

This is a running theme of the project, documented in the paper's Methods.

1. **Backward-filled lags.** `ffill().bfill()` on lagged features copied *future* values backward into early rows. Fixed: `ffill` only, then drop rows genuinely lacking history.
2. **Label/feature inconsistency.** `price_direction` was computed before `price_prev_month` was final, using a different grouping key than the one used to fill it. Fixed: one consistent key throughout, target computed last, with an assertion verifying zero mismatches.
3. **CPI publication lag.** Official CPI is published weeks into the *following* month, so same-month merging supplies a figure that had not yet been released. Fixed: each CPI observation's availability date is shifted forward one month. **The one-month lag is a disclosed assumption, not a verified fact about NISR's release calendar.**

Additionally, `1_data_cleaning.ipynb` prints the row cost of each macroeconomic column before dropping anything, so the trade-off between feature coverage and training-set size is made explicitly. Requiring year-over-year CPI would delete roughly the first year of data (~15% of rows) for a feature the models do not use, so that *column* is dropped and the *rows* are kept controlled by the `KEEP_CPI_YOY` flag.

### A tuning defect worth noting

An earlier run showed hyperparameter tuning making the regressor *worse*. The cause was the search space, not the data: the baseline's own configuration (`max_depth=12, min_samples_leaf=1`) was not reachable within the grid, so the search could not recover its own starting point. The grids now contain each baseline configuration, and tuning reports baseline vs. tuned under the *same* cross-validation protocol so the comparison is like-for-like.

## 6. Headline Results

All figures below are from a verified end-to-end run on the real data (13,325 observations; chronological split at 2025-04-15, 10,719 train /2,606 test).

- **Regression is easy; classification is hard.** Random forest reaches   R² = 0.94 and RMSE 466 RWF (vs. 1,747 RWF for mean prediction); after tuning, R² = 0.95 and RMSE 457 RWF. The best classifier reaches ROC-AUC 0.641. Price *level* is anchored to last month's price and commodity identity; price *direction* depends on month-to-month changes this data does not observe.
- **Tuning improved both tasks**, and improved them on cross-validation before the test set was consulted (CV F1 +0.055; CV RMSE +0.020 log units). Classification test ROC-AUC 0.608 → 0.641; regression test RMSE 466 → 457.
- **The `cpi_yoy_pct` diagnostic paid off.** The row-cost table showed six of seven macro columns cost zero rows, while year-over-year CPI alone would have deleted 2,025 rows (15.2%) the earliest year, which a chronological split needs most. Dropping the column instead of the rows recovered ~18% more data.
- **The enrichment helped, but not for the hypothesised reason.** Because the temperature-only and enriched pipelines ran over *identical* rows and splits, this is a controlled ablation. ROC-AUC rose for every model family (RF 0.556 → 0.608; GB 0.572 → 0.610; MLP 0.591 → 0.610; tuned RF 0.641). But precipitation separates rising from falling prices by ~0.33 mm/day, while month-over-month FX and CPI differ by ≤0.007. The gain came from measuring *climate* better, not from adding macro context plausibly because the macro series are constant across all markets in a month and so cannot explain why one market's maize rose while another's fell.
  - Note honestly: best F1 fell (0.552 → 0.479). The earlier figure came from a high-recall logistic regression at the default threshold. ROC-AUC is threshold-independent and rose consistently, so it is the more reliable evidence; a practitioner wanting recall would tune the threshold.
- **Clustering corroborates rather than discovers.** Two clusters (silhouette 0.531) isolate 26 Northern Province market–commodity pairs from 323 others. K-Means and Ward nearly agree (ARI = 0.951) while the Gaussian mixture diverges (ARI = 0.329) evidence the structure is compact and spherical. Northern Province is not only coolest (17.6°C vs 20.8°C) but by a wide margin wettest (4.72 vs 3.34 mm/day), so the added climate variables carried real information. Caveat: those 26 pairs average 15.8 observations each vs. 39.4 for the larger cluster, so their profile is noisier.

## 7. Reproducing This Project

```bash
pip install -r requirements.txt

jupyter nbconvert --to notebook --execute --inplace 0_data_extraction.ipynb
jupyter nbconvert --to notebook --execute --inplace 1_data_cleaning.ipynb
jupyter nbconvert --to notebook --execute --inplace 2_exploratory_data_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace 3a_supervised_learning.ipynb
jupyter nbconvert --to notebook --execute --inplace 3b_unsupervised_clustering.ipynb

cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Or just download the jupyter notebooks, create the virtual environment, install the required packages, and run the notebooks.

Requires `data/raw/CPI_time_series_July 2026.xls` to be present before running `0_data_extraction.ipynb`; the NASA POWER and Yahoo Finance calls are live and depend on external service availability.

## 8. Author

Thierry Donambi

## 9. References

- World Food Programme. *Food Prices* [Dataset]. Humanitarian Data Exchange. https://data.humdata.org/dataset/wfp-food-prices
- NASA Langley Research Center. *POWER Project* [API]. https://power.larc.nasa.gov/
- Yahoo Finance. *USD/RWF Exchange Rate* [Data]. https://finance.yahoo.com/
- National Institute of Statistics of Rwanda. *Consumer Price Index Time Series* [Dataset]. https://www.statistics.gov.rw/
- James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*. Springer.
