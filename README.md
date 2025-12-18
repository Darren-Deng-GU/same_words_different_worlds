# Same Words, Different Worlds
## Measuring Partisan Semantic Divergence in Congressional AI Discourse

**Author:** Darren Deng  
**Course:** PPOL 6801 - Text as Data 
**Institution:** Georgetown University  
**Date:** December 2025

---

## Project Overview

This project investigates whether Democrats and Republicans assign different *meanings* to the same words when discussing Artificial Intelligence. Using 3,201 AI-related tweets from members of Congress (2018-2024) and a fine-tuned RoBERTa language model, I measure semantic distance between partisan usage of contested concepts like "safety," "rights," and "regulation."

### Key Findings
- Contested concepts show **1.78x higher semantic distance** than control words (p = 0.024, Cohen's d = 0.99)
- 73% of analyzed words became **less polarized after ChatGPT's release**
- The Senate is **4.3x more polarized** than the House in AI discourse
- "Rights" means "civil rights" to Democrats vs. "individual/constitutional rights" to Republicans

---

## Repository Structure

```
same_words_different_worlds/
│
├── README.md                          # This file
│
├── data/
│   ├── raw/
│   │   └── congress_tweets_full_2018_2024.csv    # Raw tweet data (too big, now replaced by dropbox download link)
│   └── processed/
│       ├── 01_ai_tweets_clean.csv                # Filtered AI tweets
│       └── 02_tweets_with_embeddings.pkl         # Tweets with embeddings
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_fine_tuning.ipynb
│   ├── 04_embedding_extraction.ipynb
│   ├── 05_semantic_analysis.ipynb
│   ├── 06_statistical_validation.ipynb
│   ├── 07_temporal_analysis.ipynb
│   ├── 08_deeper_semantic_analysis.ipynb
│   ├── 09_subgroup_analysis.ipynb
│   └── 10_final_visualization.ipynb
│
├── models/
│   └── fine_tuned_roberta/            # Fine-tuned RoBERTa model (503 MB)(too big, now replaced by dropbox download link)
│
├── figures/
│   ├── 01_volume_by_year.png
│   ├── 02_log_odds_words.png
│   ├── 03_semantic_space.png
│   ├── 05_semantic_polarization.png
│   ├── 06_collocation_rights.png
│   ├── 07_semantic_distance_with_ci.png
│   ├── 08_temporal_polarization.png
│   ├── 09_pre_post_chatgpt.png
│   ├── 10_semaxis_rights.png
│   ├── 10_semaxis_grid.png
│   ├── 11_collocation_grid.png
│   ├── 12_chamber_comparison.png
│   ├── 13_top_speakers.png
│   └── 14_master_figure.png
│
├── outputs/
│   ├── semantic_distances.csv
│   ├── semantic_distances_validated.csv
│   ├── temporal_analysis.csv
│   ├── chamber_polarization.csv
│   ├── speaker_polarization.csv
│   ├── example_tweets.csv
│   ├── final_summary_statistics.csv
│   └── key_findings_report.txt
│
└── report/
    └── Same_Words_Different_Worlds_Report.docx   # Final 10-page report
```

---

## Notebooks: Detailed Documentation

### Notebook 01: Data Preparation
**File:** `notebooks/01_data_preparation.ipynb`

| Component | Description |
|-----------|-------------|
| **Inputs** | `data/raw/congress_tweets_full_2018_2024.csv` (1,893,564 tweets) |
| **Process** | - Converts Unix timestamps to datetime<br>- Filters tweets using AI-related keyword regex<br>- Removes Independent party members<br>- Cleans text (removes URLs, RT prefixes, normalizes whitespace)<br>- Preserves case/punctuation for RoBERTa<br>- Drops tweets < 10 characters |
| **Outputs** | `data/processed/01_ai_tweets_clean.csv` (3,201 tweets, 1.6 MB) |

---

### Notebook 02: Exploratory Data Analysis
**File:** `notebooks/02_eda.ipynb`

| Component | Description |
|-----------|-------------|
| **Inputs** | `data/processed/01_ai_tweets_clean.csv` |
| **Process** | - Analyzes tweet volume by year and party<br>- Computes log-odds ratios for partisan word usage<br>- Identifies shared vocabulary (451 words used ≥10 times by both parties)<br>- Selects contested concepts and control words for analysis |
| **Outputs** | - `figures/01_volume_by_year.png`<br>- `figures/02_log_odds_words.png`<br>- `data/outputs/shared_vocabulary.csv`<br>- `data/outputs/log_odds_results.csv` |

---

### Notebook 03: Model Fine-Tuning
**File:** `notebooks/03_fine_tuning.ipynb`

| Component | Description |
|-----------|-------------|
| **Inputs** | `data/processed/01_ai_tweets_clean.csv` |
| **Process** | - Loads pre-trained RoBERTa-base (124M parameters)<br>- Fine-tunes using Masked Language Modeling (MLM)<br>- Training: 3 epochs, batch_size=16, lr=2e-5, fp16<br>- Verifies domain adaptation with masked prediction tests |
| **Outputs** | `models/fine_tuned_roberta/` (503.6 MB) |
| **Requirements** | GPU recommended (Tesla T4 used, ~2.5 min training time) |

---

### Notebook 04: Embedding Extraction
**File:** `notebooks/04_embedding_extraction.ipynb`

| Component | Description |
|-----------|-------------|
| **Inputs** | - `data/processed/01_ai_tweets_clean.csv`<br>- `models/fine_tuned_roberta/` |
| **Process** | - Loads fine-tuned RoBERTa model<br>- Extracts 768-dimensional embeddings per tweet<br>- Uses mean pooling weighted by attention mask<br>- Validates embeddings with semantic similarity tests |
| **Outputs** | `data/processed/02_tweets_with_embeddings.pkl` (11.6 MB) |
| **Requirements** | GPU recommended (~30 seconds for 3,201 tweets) |

---

### Notebook 05: Semantic Divergence Analysis
**File:** `notebooks/05_semantic_analysis.ipynb`

| Component | Description |
|-----------|-------------|
| **Inputs** | `data/processed/02_tweets_with_embeddings.pkl` |
| **Process** | - Visualizes semantic space (PCA, UMAP)<br>- Trains classifier probe (logistic regression)<br>- Calculates semantic distance for contested vs. control words<br>- Performs permutation test (10,000 iterations)<br>- Computes bootstrap confidence intervals<br>- Conducts collocation analysis |
| **Outputs** | - `figures/03_semantic_space.png`<br>- `figures/05_semantic_polarization.png`<br>- `figures/06_collocation_rights.png`<br>- `data/outputs/semantic_distances.csv`<br>- `data/outputs/collocation_analysis.json` |

---

### Notebook 06: Statistical Validation
**File:** `notebooks/06_statistical_validation.ipynb`

| Component | Description |
|-----------|-------------|
| **Inputs** | - `data/processed/02_tweets_with_embeddings.pkl`<br>- `data/outputs/semantic_distances.csv` |
| **Process** | - Calculates bootstrap CIs for each word (1,000 iterations)<br>- Performs per-word permutation tests (5,000 iterations)<br>- Tests robustness with multiple distance metrics (Cosine, Euclidean, Manhattan)<br>- Computes rank correlations across metrics |
| **Outputs** | - `figures/07_semantic_distance_with_ci.png`<br>- `data/outputs/semantic_distances_validated.csv` |

---

### Notebook 07: Temporal Analysis
**File:** `notebooks/07_temporal_analysis.ipynb`

| Component | Description |
|-----------|-------------|
| **Inputs** | `data/processed/02_tweets_with_embeddings.pkl` |
| **Process** | - Calculates yearly semantic distance (2019-2024)<br>- Compares pre-ChatGPT (2019-2022) vs. post-ChatGPT (2023-2024) periods<br>- Performs statistical tests (sign test, Wilcoxon, paired t-test)<br>- Computes effect sizes |
| **Outputs** | - `figures/08_temporal_polarization.png`<br>- `figures/09_pre_post_chatgpt.png`<br>- `data/outputs/temporal_analysis.csv`<br>- `data/outputs/yearly_corpus_polarization.csv` |

---

### Notebook 08: Deeper Semantic Analysis
**File:** `notebooks/08_deeper_semantic_analysis.ipynb`

| Component | Description |
|-----------|-------------|
| **Inputs** | `data/processed/02_tweets_with_embeddings.pkl` |
| **Process** | - Trains classifier to define partisan axis<br>- Projects tweets onto partisan axis (SemAxis)<br>- Creates distribution visualizations per word<br>- Generates collocation charts for multiple words<br>- Extracts example tweets showing partisan framing |
| **Outputs** | - `figures/10_semaxis_rights.png`<br>- `figures/10_semaxis_grid.png`<br>- `figures/11_collocation_grid.png`<br>- `data/outputs/example_tweets.csv`<br>- `data/outputs/semaxis_results.csv` |

---

### Notebook 09: Subgroup Analysis
**File:** `notebooks/09_subgroup_analysis.ipynb`

| Component | Description |
|-----------|-------------|
| **Inputs** | `data/processed/02_tweets_with_embeddings.pkl` |
| **Process** | - Compares House vs. Senate polarization<br>- Calculates word-level distances by chamber<br>- Identifies top partisan speakers<br>- Analyzes individual member discourse patterns |
| **Outputs** | - `figures/12_chamber_comparison.png`<br>- `figures/13_top_speakers.png`<br>- `data/outputs/chamber_polarization.csv`<br>- `data/outputs/chamber_word_polarization.csv`<br>- `data/outputs/speaker_polarization.csv` |

---

### Notebook 10: Final Visualization & Export
**File:** `notebooks/10_final_visualization.ipynb`

| Component | Description |
|-----------|-------------|
| **Inputs** | All previous outputs from `data/outputs/` |
| **Process** | - Compiles master summary statistics table<br>- Creates combined master figure<br>- Generates key findings report |
| **Outputs** | - `figures/14_master_figure.png`<br>- `data/outputs/final_summary_statistics.csv`<br>- `data/outputs/key_findings_report.txt` |

---

## Data Files

### Raw Data
| File | Description | Size |
|------|-------------|------|
| `congress_tweets_full_2018_2024.csv` | Full congressional tweet dataset | ~500 MB |

**Note:** If the raw data exceeds GitHub's file size limits, it is hosted at https://www.dropbox.com/scl/fi/eqtx6ddyquwugymvqeslx/congress_tweets_full_2018_2024.csv?rlkey=afwanoeo7fhncpv1dov2dm0c9&st=4m4j8o0u&dl=0.

### Processed Data
| File | Description | Size |
|------|-------------|------|
| `01_ai_tweets_clean.csv` | Filtered AI-related tweets | 1.6 MB |
| `02_tweets_with_embeddings.pkl` | Tweets with 768-dim embeddings | 11.6 MB |

---

## Key Output Files

| File | Description |
|------|-------------|
| `semantic_distances_validated.csv` | Word-level distances with bootstrap CIs and p-values |
| `temporal_analysis.csv` | Pre/post ChatGPT comparison by word |
| `chamber_polarization.csv` | House vs. Senate polarization metrics |
| `speaker_polarization.csv` | Individual member partisan scores |
| `example_tweets.csv` | Representative tweets for qualitative analysis |
| `final_summary_statistics.csv` | All key statistics for reporting |

---

## Requirements

### Python Dependencies
```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.12.0
scikit-learn>=1.0.0
transformers>=4.25.0
torch>=1.12.0
tqdm>=4.64.0
umap-learn>=0.5.0
scipy>=1.9.0
```

### Hardware
- GPU recommended for Notebooks 03 and 04 (fine-tuning and embedding extraction)
- Tested on Google Colab with Tesla T4 GPU
- CPU-only execution possible but significantly slower

### Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn transformers torch tqdm umap-learn scipy
```

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Darren-Deng-GU/same_words_different_worlds.git
   cd same_words_different_worlds
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run notebooks in order:**
   - Start with `01_data_preparation.ipynb`
   - Proceed sequentially through `10_final_visualization.ipynb`
   - Each notebook saves outputs used by subsequent notebooks

4. **For Google Colab:**
   - Upload notebooks to Colab
   - Mount Google Drive
   - Update `BASE_PATH` variable to your Drive location

---

## Results Summary

| Metric | Value |
|--------|-------|
| Total AI Tweets | 3,201 |
| Classifier Accuracy | 72.1% |
| Contested/Control Distance Ratio | 1.78x |
| Permutation Test p-value | 0.024 |
| Effect Size (Cohen's d) | 0.99 |
| Words Less Polarized Post-ChatGPT | 8/11 (73%) |
| Senate/House Polarization Ratio | 4.3x |

---

## Citation

If you use this code or methodology, please cite:

```
[Darren Deng]. (2025). Same Words, Different Worlds: Measuring Partisan Semantic 
Divergence in Congressional AI Discourse. Georgetown University, PPOL 5205.
```

---

## License

This project is submitted for academic purposes as part of PPOL 5205 at Georgetown University.

---

## Contact

Darren Deng  
sd1511@georgetown.edu
Georgetown University
