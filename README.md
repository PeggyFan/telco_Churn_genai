# Telco Churn — Gen AI-assisted experiments

This repository contains a single analysis / experimentation script, `churn_genai.py`, that demonstrates a full workflow for predicting telco customer churn and exploring ways to augment the dataset using generative models and NLP pipelines.

It started as a Colab notebook and was converted to a script. The code trains baseline classifiers, generates synthetic customer feedback via LLM prompts (example code commented out), and explores several ML/ML+NLP pipelines including:

- baseline Random Forest and Logistic Regression churn models
- incorporating LLM-generated customer feedback as a feature (sentiment)
- batch sentiment analysis and zero-shot classification using Hugging Face pipelines
- student-model distillation (DistilBERT) to scale sentiment labeling
- embeddings + UMAP dimensionality reduction with lightweight classifiers
- business-impact / ROI thresholding utilities to pick operating points

The goal is to show how generative AI (for synthetic feedback) and small NLP models can be incorporated into a churn prediction pipeline and how to reason about operational thresholds from a business ROI perspective.

## Files

- `churn_genai.py` — main script (converted from a Colab notebook). Contains data prep, model training/evaluation, LLM/embedding experimentation, and ROI utilities.

## Quick start

Prerequisites: Python 3.8+ and the packages used in the script. The script expects the Telco Customer Churn dataset (Kaggle) placed in Google Drive when run inside Colab. Paths in the script reference `/content/gdrive/My Drive/` and will need changes to run locally.

Suggested minimal dependencies (install with pip):

```bash
pip install pandas numpy scikit-learn joblib torch transformers datasets sentence-transformers umap-learn ipywidgets
```

Notes:
- The script was originally written to run in Google Colab. Several file paths reference `/content/gdrive/My Drive/` (e.g. `telco.csv`, `telco_feedback_all.csv`) — update `file_path` and `root_path` at the top of the script if you run it locally.
- Parts of the code that call external LLMs (OpenAI) are commented out to avoid accidental API calls. To enable them, supply an API key via environment variable and uncomment the relevant sections.

## How the script is organized

1. Utility functions: saving models/data.
2. Data preparation: `prep_data()` handles encoding categorical columns, normalizing numeric features, and cleaning the dataset.
3. Splitting: `split_data()` returns train/val/test splits with stratification on churn.
4. Modeling: `benchmark_classifiers()` trains RandomForest and LogisticRegression models and returns evaluation metrics.
5. LLM / NLP experiments: commented examples show how to generate synthetic feedback with an LLM, then create sentiment and category features using Hugging Face pipelines and distilled student models.
6. Embedding pipeline: use `sentence-transformers` to create embeddings, reduce dimensionality with UMAP, and train lightweight classifiers on embeddings.
7. ROI and threshold utilities: `optimal_threshold_for_roi*` functions compute threshold choices and expected net savings for retention campaigns.
8. Interactive widgets: there is a small IPython widgets-based ROI calculator at the end of the notebook/script meant for interactive exploration inside a notebook.

## Running locally (recommended adjustments)

1. Place the Telco dataset `telco.csv` (Kaggle dataset) somewhere local, e.g. `data/telco.csv`.
2. Edit the top-level `file_path` and `root_path` variables inside `churn_genai.py` to point to your local data folder.
3. If you don't have a GPU, change Hugging Face pipeline `device` arguments (e.g. `device= -1` or remove `device`) to run on CPU.
4. To run the script (non-interactively):

```bash
python churn_genai.py
```

Warning: the script contains long-running sections (embedding generation, model fine-tuning, batch LLM calls). For quicker runs, either sample the input data or comment out the heavy blocks.

## Reproducibility & testing

- The script uses fixed random seeds in train_test_split (random_state=42) where applicable to aid reproducibility.
- I recommend iterating on smaller subsets before scaling to the full dataset. Several `pd.read_csv` calls expect intermediate CSVs (for example `telco_feedback_all.csv`) which are created in the commented-out LLM generation steps.

## Next steps / Improvements

- Parameterize paths and hyperparameters with a small CLI (argparse) or environment variables.
- Split the Colab notebook content into modular scripts: data preprocessing, model training, LLM-generation, embedding pipeline, ROI calculator.
- Add unit tests for core preprocessing and ROI functions.
- Add a lightweight requirements file (`requirements.txt`) with pinned versions.

If you'd like, I can:

- add a `requirements.txt` with versions that match the script, or
- refactor the script into smaller modules and add a simple CLI, or
- create a minimal unit test for `prep_data()` and `optimal_threshold_for_roi_pr()`.

