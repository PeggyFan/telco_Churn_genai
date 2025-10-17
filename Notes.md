# Customer Churn Prediction Leverage LLM Generated Data


## Premise:
Customer churn is one of the most common...
Telco Dataset: Description can be found here: https://www.kaggle.com/datasets/blastchar/telco-customer-churn


This exercise explores:

#### 1. What does it take to use LLMs to generate synthetic data 
#### 2. How to evaluate the qualtiy of synthetic data generated
#### 3. How to feature engineer synthetic data into an useful feature for model training
#### 4. How to tie churn model outputs to business ROI


## 1. Generating synthetic data using LLM
- Customer feedback. Passing several customer attributes to the OpenAI API
- Tokens used were ~ 1.5-1.8M
- Cost ~$0.97
- Time taken: ~2 hours


## 2. Evaluating LLM generated feedback
### 1. Use Embedding Similarity: Provide examples of a typical feedback for churned customer and customer who stayed.
Use transformer model to calculate the similarity between all the feedback and those two canonical archetypes. 

`churn_text = "I am unhappy with the service and plan to cancel."`

`stay_text = "I am satisfied and plan to continue my subscription."`

Then calculate "churn alignment": ``` churn alignment = similarity score to churn - similarity score to stayed ```

``` 
if churn alignment < 0:
    -> the feedback is leaning towards stayed

if churn alignment < 1:
    -> the feedback is leaning towards churned

if churn_alignment = 0:
    -> the feedback is neutral
```

Then we calculate the correlation between the churn alignment score and the churn outcome (1 for churned, 0 for stayed). This The correlation is 0.632

### 2. Use the pre-trained Bert model to analyze the sentiment of the feedback, giving a label (and a score). 
Given that for sentiment `negative = 1` and `positive = 0`, calculate the correlation between those sentiment score and churn outcome. This correlation is 0.877 

#### Those two approaches show that the LLM generated feedback are decent and directionally correct. 
Based on sentiment, which is a very broad measure, the correlation is stronger with actual churn outcome. Whether this result is satisfying is up to the stakeholders. 

Checking similarity against archetypical texts give lower correlations due to how the texts are written. 

In the synthetic data generate phase, I could have used a few-shots approach to hone in on the type of feedback customers of a particular company or industry would give. Then the evaluation step would be a great check against how the LLM ingested the specific requirements of feedback of the customers. 

All in all, one should not skip an evaluation step against the synthetic data generated. 
If zero or few shots approach cannot deliver high quality data, one might consider fine-tuning, which is more involved.

## 3. Feature engineering option based on the LLM generated feedback:

In the case of customer churn, there are two ways I can leverage the feedback data.

### 1. Sentiment based on feedback
I used the transformer model (Bert) trained on sentiment analysis to generate sentiment label (as a new feature) based on the `feedback` column.

I applied a batch processing method to avoid memory/speed issue. Because the telco dataset is only ~7500 rows, this did not take long. This I called dataset V1.

But imagine if you have 7.5M rows. What if you cannot afford using LLM to make zero-shot or few-shot predictions for a very large dataset? If your dataset is 1M plus that uses a large number of tokens, they can cost hundreds if not thousands of dollars each run.

So I tried the option of only processing a subset of your data using the models to generate the labels, which produces a pseudo gold dataset. Then train my own student (smaller) model using that pseudo dataset, and make predictions for the rest of the original data. 
I used again the Bert model as the student model, which trained on half of the data. It then generated sentiment labels for the rest of the dataset. But one can easily choose a simpler classifier for the student model as well.
This I called dataset V2.

### 2. Defined categories based on feedback 
I came up with five general themes that a feedback can touch upon: "billing", "customer service", "product quality", "technical issues", "other."

I tried to replicate the same approach to generate defined categories as a new feature, but it was taking much longer. I decided to stop.

### 3. Unsupervised clustering/topics based on feedback
Using embeddings created from the `feedback` column, I used `umap` to create 5 clusters. I chose five to align with the defined categories I tried above. Those 5 clusters then are the features I added to the data. This is dataset V3.

So this project does not focus on model selection; I ran only two models- Logistic and Random Forest. I was more interested in the data augmentation aspect of the problem given that we can add synthetic data such as feedback to the original dataset. 

The best performing models were Random Forest models (no surprise) on dataset V1 and V2.
This is very interesting, because V1 is the dataset where all the sentiment labels are generated by the sentiment model, and V2 is the student model approach. While I don't have a clear answer on what percentage of the data needed to be alloted for a student model to train and be affect, or what types of model make a good student model, it is promising to see its performance can be on par with the feature generated directly by the LLM.

## 4. Busines ROI
What is key is how to translate the model predictions into business context that stakeholders can understand. I created a business ROI calculator.

[![Launch Voila](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/PeggyFan/telco_Churn_genai/HEAD?urlpath=voila/render/churn_roi_widget.ipynb)
