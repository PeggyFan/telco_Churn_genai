"""Customer Churn Prediction with LLM-Generated Features
Original file is located at
    https://colab.research.google.com/drive/16ZcVb-WWM2zGhlC1YJSt9t7im1lHxzfw

Telco Churn Dataset:
    https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data
"""

import os
import pandas as pd
import numpy as np
import warnings
import joblib
import time
import torch
import umap
import ipywidgets as widgets
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, roc_curve
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, pipeline, DataCollatorWithPadding, DistilBertForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
import random
from openai import OpenAI
from IPython.display import display, clear_output
pd.set_option('display.float_format', '{:,.2f}'.format)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# ===============================
# Utility Functions for Saving
# ===============================

def save_models(models: list, names: list, root_path: str) -> None:
    """
    Save models to disk with given names and root path.
    Args:
        models (list): List of trained model objects.
        names (list): List of filenames (without extension).
        root_path (str): Directory to save models.
    """
    for model, name in zip(models, names):
        joblib.dump(model, f"{root_path}/{name}.pkl")


def save_dataframe(df: pd.DataFrame, filename: str, root_path: str) -> None:
    """
    Save DataFrame to disk at root_path/filename.csv.
    Args:
        df (pd.DataFrame): DataFrame to save.
        filename (str): Filename (without extension).
        root_path (str): Directory to save file.
    """
    df.to_csv(f"{root_path}/{filename}.csv", index=False)


# ===============================
# Data Preparation Functions
# ===============================

def prep_data(category_cols: list, data: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns, scale numeric columns, and clean data.
    Args:
        category_cols (list): List of categorical column names.
        data (pd.DataFrame): Raw input data.
    Returns:
        pd.DataFrame: Preprocessed data.
    """
    data_encoded = pd.get_dummies(data , columns = category_cols )

    data_encoded['TotalCharges'] = data_encoded['TotalCharges'].replace(' ', np.nan)
    data_encoded['TotalCharges'] = data_encoded['TotalCharges'].astype(float)
    drop_cols = ['Unnamed: 0', 'customerID']
    data_encoded.drop(drop_cols, axis=1, inplace=True)
    data_encoded = data_encoded.dropna()

    mms = MinMaxScaler()
    data_encoded[['tenure','MonthlyCharges','TotalCharges']] = mms.fit_transform(data_encoded[['tenure','MonthlyCharges','TotalCharges']])

    return data_encoded


def split_data(df: pd.DataFrame) -> tuple:
    """
    Split data into train, validation, and test sets.
    Args:
        df (pd.DataFrame): Preprocessed data.
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    y = df['Churn_encoded'].values
    X = df.drop(columns=['Churn', 'Churn_encoded'], axis=1)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ===============================
# Modeling Functions
# ===============================

def benchmark_classifiers(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray) -> tuple:
    """
    Train and evaluate classifiers, returning metrics and trained models.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (np.ndarray): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (np.ndarray): Validation labels.
    Returns:
        tuple: (metrics DataFrame, list of trained models)
    """
    model_list = [RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs = -1),
                  LogisticRegression()]
    model_names = ["RandomForestClassifier", "LogisticRegression"]
    model_data = zip(model_list, model_names)

    acc_list = []
    model_trained = []
    recall_list = []
    precision_list = []
    roc_auc_scores = []
    pr_auc_scores = []

    for model, name in model_data:
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        model_trained.append(model)
        prec, rec, thresh = precision_recall_curve(y_val, y_pred_proba)
        balance_idx = np.argmin(np.abs(prec - rec))
        best_thresh = thresh[balance_idx]
        y_pred_class = (y_pred_proba >= best_thresh).astype(int)
        acc = accuracy_score(y_val, y_pred_class)
        recall = recall_score(y_val, y_pred_class)
        recall_list.append(recall)
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        roc_auc_scores.append(roc_auc)
        precision_at_default = precision_score(y_val, y_pred_class)
        precision_list.append(precision_at_default)

        # ===============================
        # Evaluation & Utility Functions
        # ===============================
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_val, y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)
        pr_auc_scores.append(pr_auc)

        acc_list.append(acc)

    acc_df = pd.DataFrame(columns = ['accuracy'])
    acc_df['accuracy'] = acc_list
    acc_df['classifier'] = model_names
    acc_df['recall'] = recall_list
    acc_df['precision'] = precision_list
    acc_df['roc_auc'] = roc_auc_scores
    acc_df['pr_auc'] = pr_auc_scores
    acc_df = acc_df[['classifier', 'accuracy', 'recall', 'precision', 'roc_auc', 'pr_auc']]

    return acc_df, model_trained


# ===============================
# Begin Data Loading and Preprocessing
# ===============================

file_path = '/content/gdrive/My Drive/telco.csv'
data = pd.read_csv(file_path)
data['Churn_encoded'] = data['Churn'].map({"Stayed": 0, "Churned": 1})

category_cols = ['gender',
 'SeniorCitizen',
 'Partner',
 'Dependents',
 'PhoneService',
 'MultipleLines',
 'InternetService',
 'OnlineSecurity',
 'OnlineBackup',
 'DeviceProtection',
 'TechSupport',
 'StreamingTV',
 'StreamingMovies',
 'Contract',
 'PaperlessBilling',
 'PaymentMethod']

root_path = '/content/gdrive/My Drive'

"""## Base Model"""

data2 = prep_data(category_cols, data)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(data2)
y_test_b = y_test.copy()
save_dataframe(pd.DataFrame(y_test_b), "y_test_b", root_path)
acc_df, model_trained = benchmark_classifiers(X_train, y_train, X_val, y_val)
save_models(model_trained, ["random_forest_model_v0", "logistic_model_v0"], root_path)
acc_df['data'] = 'v0'
save_dataframe(acc_df, "acc_df_v0", root_path)

"""## GENERATE LLM DATA"""
#### How long did this part take???

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
prompt_template = """
You are generating synthetic customer feedback for a telco churn dataset.

Customer details:
- Gender: {gender}
- Senior Citizen: {SeniorCitizen}
- Partner: {partner}
- Tenure: {tenure} months
- Contract type: {contract}
- Monthly charges: {charges}
- Churned: {churn}

Task:
Generate a short customer feedback statement (1-3 sentences) that reflects
their likelihood of churn. Make it realistic and vary tone across examples.
"""

def generate_feedback(row):
    prompt = prompt_template.format(
        gender=row["gender"],
        SeniorCitizen=row["SeniorCitizen"],
        partner=row["Partner"],
        tenure=row["tenure"],
        contract=row["Contract"],
        charges=row["MonthlyCharges"],
        churn=row["Churn"]
    )

    # Call the LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # cost-efficient, or use gpt-4o
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0.8,
    )

    return response.choices[0].message.content.strip()

# Generate feedback for a small subset (to test)
df_sample = data.sample(3, random_state=42)
df_sample["customer_feedback"] = df_sample.apply(generate_feedback, axis=1)

# Generate feedback for the full dataset in chunks to avoid timeouts
df = []
chunk = 500

for i in range(0, len(data2), chunk):
  current_chunk = i
  remainder = len(data2) - i

  if remainder < 0:
    break
  elif remainder < chunk:
    data_chunk = data2.iloc[i:i+remainder]
  else:
    data_chunk = data2.iloc[i:i+chunk]

  data_chunk["customer_feedback"] = data_chunk.apply(generate_feedback, axis=1)

  # file_path = f"/content/drive/MyDrive/Colab Notebooks/Outputs/telco_feedback_{chunk}.csv"
  # data_chunk.to_csv(file_path, index=False)

  df.append(data_chunk)

data_feedback = pd.concat(df, axis=0)
data_feedback['Churn_encoded'] = data_feedback['Churn'].map({"Stayed": 0, "Churned": 1})
data_feedback.to_csv("/content/drive/MyDrive/telco_feedback_7500.csv", index=False)

"""## 1. Batch / Bulk Inference

Instead of running one input at a time, process multiple feedback items in batches.

Transformers libraries like Hugging Face support DataLoader + GPU batching, which can increase throughput 5–10x.

If you’re CPU-bound, even batching 10–50 texts per forward pass reduces overhead.
"""

all_data = pd.read_csv(f"{root_path}/telco_feedback_7500.csv")
customer_feedback = list(all_data['customer_feedback'].values)

# Sentiment model
sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model_name, device=0)  # device=0 for GPU

# Zero-shot classification
zero_shot_model_name = "facebook/bart-large-mnli"
zero_shot_pipeline = pipeline("zero-shot-classification", model=zero_shot_model_name, device=0)

def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

batch_size = 40  # adjust depending on GPU/CPU memory
start_time = time.perf_counter()
all_sentiments = []
with torch.no_grad():  # disable gradient calculations
    for batch in batchify(customer_feedback, batch_size):
        results = sentiment_pipeline(batch)
        for i in range(len(results)):
            label = results[i]['label']
            all_sentiments.append(label)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

### Nice to have but no time

# start_time = time.perf_counter()
# candidate_labels = ["billing", "customer service", "product quality", "technical issues", "other"]

# all_classifications = []

# with torch.no_grad():  # disable gradient calculations
#     for batch in batchify(customer_feedback, batch_size):
#       results = zero_shot_pipeline(batch, candidate_labels)
#       category = results[0]['labels']
#       all_classifications.extend(category)

category_cols.append('sentiment')

all_data['sentiment'] = all_sentiments
data_senti = prep_data(category_cols, all_data)
data_senti.drop('customer_feedback', axis=1, inplace=True)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(data_senti)
pd.DataFrame(X_test).to_csv('f'{root_path}/X_test_v1.csv', index=False)
X_test.to_csv('X_test_v1.csv', index=False)

acc_df, model_trained = benchmark_classifiers(X_train, y_train, X_val, y_val)
clf = model_trained[0]
joblib.dump(clf, f"{root_path}/logistic_model_v1.pkl")
clf = model_trained[1]
joblib.dump(clf, f"{root_path}/random_forest_model_v1.pkl")

acc_df['data'] = 'v1'
acc_df.to_csv('f'{root_path}/acc_df_v1.csv')

"""## 2. Semi-Supervised / Distillation Approach
Since your zero-shot models already generate labels:
Sample a representative subset of your customer data.
Run your zero-shot models once to label them.
Train a smaller student model on this pseudo-labeled dataset:
DistilBERT, TinyBERT, or even a non-transformer classifier.
Deploy the student model for the full dataset.
Benefit: You retain the “LLM intelligence” but cut inference cost drastically.

### How many samples are enough for fine-tuning a student model?

### Option A: DistilBERT / TinyBERT Fine-Tuning
"""
def encode_labels(example):
    example["label"] = label2id[example["old_labels"]]
    return example

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True)

os.environ["WANDB_DISABLED"] = "true"

## Should be using the first 4000 rows as training data

texts = customer_feedback[:4000]
labels = all_sentiments[:4000]
dataset = Dataset.from_dict({'text': texts, 'label': labels})
dataset = dataset.rename_column("label", "old_labels")
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

dataset = dataset.map(encode_labels)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

start_time = time.perf_counter()
training_args = TrainingArguments(output_dir="./test_trainer", per_device_train_batch_size=8, report_to="none")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
end_time = time.perf_counter()
elapsed_time = end_time - start_time

texts = all_data.iloc[4000:]['customer_feedback'].values
dataset_test = Dataset.from_dict({'text': texts})

"""### This is to generate sentiment column using the student model. Then use result (predicted sentiment) as a feature in Binary Classification Model

"""

start_time = time.perf_counter()
encoded_dataset = dataset_test.map(tokenize, batched=True)
encoded_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
predictions = trainer.predict(encoded_dataset)
logits = predictions.predictions
predicted_classes = np.argmax(logits, axis=1)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
predicted_labels = [id2label[i] for i in predicted_classes]

# Chain first 4000 labels and next 4000 (predicted) labels for sentiment column on the dataset
cut_off = 4000
sentiment_list = all_sentiments[:cut_off] + predicted_labels
all_data['sentiment'] = sentiment_list
data_senti2 = prep_data(category_cols, all_data)
data_senti2.drop('customer_feedback', axis=1, inplace=True)

X_train, X_val, X_test, y_train, y_val, y_test = split_data(data_senti2)
pd.DataFrame(X_test).to_csv('f'{root_path}/X_test_v2.csv', index=False)
X_test.to_csv('X_test_v2.csv', index=False)

acc_df, model_trained = benchmark_classifiers(X_train, y_train, X_val, y_val)
clf = model_trained[0]
joblib.dump(clf, f"{root_path}/logistic_model_v2.pkl")
clf = model_trained[1]
joblib.dump(clf, f"{root_path}/random_forest_model_v2.pkl")
acc_df['data'] = 'v2'
acc_df.to_csv(f'{root_path}/acc_df_v2.csv')


"""### Option B: Non-Transformer Lightweight Model (DID NOT DO BUT CAN WRITE ABOUT IT)
"""

"""## 3. Embedding + Lightweight Classifier Pipeline
You can combine embeddings with small classifiers:
Generate sentence embeddings (e.g., all-MiniLM-L6-v2) for all feedback.
Cluster / reduce dimensions (UMAP, PCA) if needed.
Train simple classifiers (Logistic Regression, XGBoost, LightGBM) on these embeddings to predict sentiment / category.
Why it’s cheaper:
One forward pass through a small embedding model is faster than repeated zero-shot LLM calls.
Embeddings are reusable; you can compute them once and store them.
Classifier inference is near-instant on CPU.
""" 

all_data = pd.read_csv(f"{root_path}/telco_feedback_all.csv")
model = SentenceTransformer('all-MiniLM-L6-v2')
all_data["embedding"] = all_data["customer_feedback"].apply(lambda x: model.encode(x).tolist())
embedding_series = all_data['embedding']
embeddings_array = np.stack(embedding_series.values)
reducer = umap.UMAP(n_components=5, random_state=42)
reduced_embeddings = reducer.fit_transform(embeddings_array)
reduced_df = pd.DataFrame(reduced_embeddings, columns=[f'umap_{i}' for i in range(reduced_embeddings.shape[1])])

all_data = all_data.reset_index(drop=True)
all_data_1 = pd.concat([all_data, reduced_df], axis=1)

category_cols.remove('sentiment')
all_data_1_2 = prep_data(category_cols, all_data_1)
all_data_1_2.drop(['customer_feedback', 'embedding'], axis=1, inplace=True)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(all_data_1_2)

acc_df, model_trained = benchmark_classifiers(X_train, y_train, X_val, y_val)
clf = model_trained[0]
joblib.dump(clf, f"{root_path}/logistic_model_v3.pkl")
clf = model_trained[1]
joblib.dump(clf, f"{root_path}/random_forest_model_v3.pkl")
acc_df['data'] = 'v3'
acc_df.to_csv('f'{root_path}/acc_df_v3.csv')


"""## Sanity check feedback generated by LLM"""
# Option 1 — Use Embedding Similarity
# Compute embedding similarity between generated text and archetypal “churn” or “loyalty” texts.

model = SentenceTransformer("all-MiniLM-L6-v2")
churn_text = "I am unhappy with the service and plan to cancel."
stay_text = "I am satisfied and plan to continue my subscription."

churn_emb = model.encode(churn_text, convert_to_tensor=True)
stay_emb = model.encode(stay_text, convert_to_tensor=True)

df = pd.read_csv(f"{root_path}/telco_feedback_all.csv")
df['Churn_encoded'] = df['Churn'].map({"Stayed": 0, "Churned": 1})
df["embedding"] = df["customer_feedback"].apply(lambda x: model.encode(x, convert_to_tensor=True))
df["sim_churn"] = df["embedding"].apply(lambda x: float(util.cos_sim(x, churn_emb)))
df["sim_stay"] = df["embedding"].apply(lambda x: float(util.cos_sim(x, stay_emb)))

df["churn_alignment"] = df["sim_churn"] - df["sim_stay"]
alignment_corr = df[["churn_alignment", "Churn_encoded"]].corr().iloc[0,1]
# Alignment correlation: 0.632

# Option 2 — Use Sentiment Polarity

# If the LLM-generated text is supposed to reflect emotions (complaints, praise, etc.):
df = pd.read_csv(f"{root_path}/telco_feedback_all.csv")
df['Churn_encoded'] = df['Churn'].map({"Stayed": 0, "Churned": 1})
sentiment_analyzer = pipeline("sentiment-analysis", 
                              model="distilbert-base-uncased-finetuned-sst-2-english")
df["sentiment"] = df["customer_feedback"].apply(lambda x: sentiment_analyzer(x)[0]["label"])
# df["sentiment_score"] = df["customer_feedback"].apply(lambda x: sentiment_analyzer(x)[0]["score"])
df["sentiment_num"] = df["sentiment"].map({"POSITIVE": 0, "NEGATIVE": 1})
correlation = df[["sentiment_num", "Churn_encoded"]].corr().iloc[0,1]
# Correlation between sentiment and churn: 0.877

"""## PICK One or Two Models to get business impact calculations
"""

acc_df_v0 = pd.read_csv(f"{root_path}/acc_df_v0.csv")
acc_df_v1 = pd.read_csv(f"{root_path}/acc_df_v1.csv")
acc_df_v2 = pd.read_csv(f"{root_path}/acc_df_v2.csv")
acc_df_v3 = pd.read_csv(f"{root_path}/acc_df_v3.csv")

acc_df = pd.concat([acc_df_v0, acc_df_v1, acc_df_v2, acc_df_v3], axis=0)
acc_df.drop('Unnamed: 0', axis=1, inplace=True)
acc_df.sort_values('roc_auc', ascending=False, inplace=True)

X_test_v1 = pd.read_csv('f'{root_path}/X_test_v1.csv')
X_test_v2 = pd.read_csv('f'{root_path}/X_test_v2.csv')
log_model_v1 = joblib.load(f"{root_path}/logistic_model_v1.pkl")
log_model_v2 = joblib.load(f"{root_path}/logistic_model_v2.pkl")

y_scores_v1 = log_model_v1.predict_proba(X_test_v1)[:, 1]
y_scores_v2 = log_model_v2.predict_proba(X_test_v2)[:, 1]


"""## Business Impact Calculation"""

file_path = '/content/gdrive/My Drive/telco.csv'
data = pd.read_csv(file_path)
data['Churn_encoded'] = data['Churn'].map({"Stayed": 0, "Churned": 1})
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

churn_rate = data['Churn_encoded'].mean()
customer_ltv = data['TotalCharges'].mean()
retention_cost = data['MonthlyCharges'].mean()*0.2
y_test_b = pd.read_csv('f'{root_path}/y_test_b.csv')['0'].values
total_customers = len(y_test_b)


def optimal_threshold_for_roi(y_true, y_scores, total_customers, churn_rate, customer_ltv, retention_cost, model_version):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    best_result = None

    for thresh, recall, fp_rate in zip(thresholds, tpr, fpr):
        actual_churners = int(total_customers * churn_rate)
        actual_non_churners = total_customers - actual_churners

        tp = int(recall * actual_churners)
        fn = actual_churners - tp
        fp = int(fp_rate * actual_non_churners)
        flagged_customers = tp + fp

        precision = tp / (tp + fp) if flagged_customers > 0 else 0

        campaign_cost = flagged_customers * retention_cost
        revenue_protected = tp * customer_ltv
        revenue_lost = fn * customer_ltv
        net_savings = revenue_protected - campaign_cost

        best_savings = -np.inf
        best_thresh = None
        best_stats = None

        if net_savings > best_savings:
                # ...existing code...
            best_savings = net_savings
            best_thresh = thresh
            best_stats  = {
                "model_version": model_version,
                "threshold": thresh,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "flagged_customers": flagged_customers,
                "campaign_cost": campaign_cost,
                "revenue_protected": revenue_protected,
                "revenue_lost": revenue_lost,
                "net_savings": net_savings,
                "precision": precision,
                "recall": recall,
                "evaluation": auc,
            }

    return pd.DataFrame(best_stats, index=[0])


def optimal_threshold_for_roi_pr(y_true, y_scores, total_customers, churn_rate, customer_ltv, retention_cost,
                              model_version, min_precision=0.8):
    """
    Find the threshold that maximizes net savings using precision-recall curve.

    Args:
        y_true: ground truth labels (0/1)
        y_scores: predicted probabilities for positive class (churn=1)
        total_customers: total number of customers
        churn_rate: fraction of customers that churn
        customer_ltv: revenue per churned customer
        retention_cost: cost to attempt retention per flagged customer
        min_precision: optional minimum precision constraint to reduce false positives

    Returns:
        dict: best threshold, corresponding net savings, and stats
    """
    prec, rec, thresholds = precision_recall_curve(y_true, y_scores)

    # precision_recall_curve returns thresholds of length n-1, append 1 for alignment
    thresholds = np.append(thresholds, 1.0)

    best_savings = -np.inf
    best_thresh = None
    best_stats = None

    for p, r, t in zip(prec, rec, thresholds):
        if p < min_precision:
            continue  # skip thresholds that do not satisfy precision constraint

        # Use your ROI function
        actual_churners = int(total_customers * churn_rate)
        actual_non_churners = total_customers - actual_churners

        true_positives = int(r * actual_churners)
        false_negatives = actual_churners - true_positives
        false_positives = int(true_positives * (1 / p - 1))
        flagged_customers = true_positives + false_positives

        campaign_cost = flagged_customers * retention_cost
        revenue_protected = true_positives * customer_ltv
        revenue_lost = false_negatives * customer_ltv
        net_savings = revenue_protected - campaign_cost

        if net_savings > best_savings:
            best_savings = net_savings
            best_thresh = t
            best_stats = {
                "model_version": model_version,
                "threshold": t,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "flagged_customers": flagged_customers,
                "campaign_cost": campaign_cost,
                "revenue_protected": revenue_protected,
                "revenue_lost": revenue_lost,
                "net_savings": net_savings,
                "precision": p,
                "recall": r,
                "evaluation": prc,
            }
        best_result_df = pd.DataFrame(best_stats, index=[0])

    return best_result_df


model_version = 'Logistic V1'
result_v1_roc = optimal_threshold_for_roi(y_test_b, y_scores_v1, total_customers,
                                            churn_rate, customer_ltv, retention_cost,
                                            model_version)
result_v1_pr = optimal_threshold_for_roi_pr(y_test_b, y_scores_v1, total_customers,
                                            churn_rate, customer_ltv, retention_cost,
                                            model_version, min_precision=0.8)
results_v1 = pd.concat([result_v1_roc, result_v1_pr], axis=0)

model_version = 'Logistic V2'
result_v2_roc = optimal_threshold_for_roi(y_test_b, y_scores_v2, total_customers,
                                            churn_rate, customer_ltv, retention_cost,
                                            model_version)

result_v2_pr = optimal_threshold_for_roi_pr(y_test_b, y_scores_v2, total_customers,
                                            churn_rate, customer_ltv, retention_cost,
                                            model_version, min_precision=0.8)
results_v2 = pd.concat([result_v2_roc, result_v2_pr], axis=0)


# --- helper for ROI computation ---
def churn_roi_summary(customer_ltv, retention_cost, churn_rate, total_customers,
                      precision, recall):
    actual_churners = int(total_customers * churn_rate)
    actual_nonchurners = total_customers - actual_churners
    tp = int(recall * actual_churners)
    fn = actual_churners - tp
    fp = int(tp * (1 / precision - 1))
    flagged = tp + fp

    campaign_cost = flagged * retention_cost
    revenue_protected = tp * customer_ltv
    revenue_lost = fn * customer_ltv
    net_savings = revenue_protected - campaign_cost

    return {
        "Flagged Customers": flagged,
        "Revenue Protected ($)": revenue_protected,
        "Campaign Cost ($)": campaign_cost,
        "Revenue Lost ($)": revenue_lost,
        "Net Savings ($)": net_savings,
        "Precision": round(precision,3),
        "Recall": round(recall,3)
    }

# --- interactive UI ---
ltv_in = widgets.FloatText(value=500,  description="Customer LTV $")
cost_in = widgets.FloatText(value=20,   description="Retention Cost $")
churn_in = widgets.FloatSlider(value=0.25, min=0.01, max=0.6, step=0.01,
                                    description="Churn Rate")
cust_in = widgets.IntText(value=10000, description="Total Customers")
prec_in = widgets.FloatSlider(value=0.9, min=0.1, max=1.0, step=0.01,
                                    description="Precision")
rec_in = widgets.FloatSlider(value=0.6, min=0.1, max=1.0, step=0.01,
                                    description="Recall")

out = widgets.Output()

def update_table(_=None):
    with out:
        clear_output()
        res = churn_roi_summary(
            ltv_in.value, cost_in.value, churn_in.value, cust_in.value,
            prec_in.value, rec_in.value
        )
        df = pd.DataFrame([res])
        display(df.style.format({
            "Revenue Protected ($)": "{:,.0f}",
            "Campaign Cost ($)": "{:,.0f}",
            "Revenue Lost ($)": "{:,.0f}",
            "Net Savings ($)": "{:,.0f}"
        }).background_gradient(subset=["Net Savings ($)"], cmap="Greens"))

for w in [ltv_in, cost_in, churn_in, cust_in, prec_in, rec_in]:
    w.observe(update_table, names="value")

display(widgets.VBox([ltv_in, cost_in, churn_in, cust_in, prec_in, rec_in, out]))
update_table()


