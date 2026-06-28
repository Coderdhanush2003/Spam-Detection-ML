# SMS Spam Detection (ML)

This repository contains an end-to-end Machine Learning pipeline designed to analyze, preprocess, and classify SMS text messages as either **Ham** (legitimate) or **Spam**.

The project leverages text vectorization strategies (Bag-of-Words) paired with a probabilistic classifier to evaluate how effectively historical message patterns can predict and filter out fraudulent communication.

## 📊 Dataset Overview

**Dataset Source:** "https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset" (SMS Spam Collection)

The dataset contains 5,572 individual message entries featuring:

* **Categorical Features (Target):** `v1` (Raw labels: `ham` or `spam`), which are mapped into a binary numerical format (`0` for Ham, `1` for Spam).
* **Text Feature (Predictor):** `v2` (The raw textual content of the SMS messages).

## 🛠️ Workflow & Code Structure

1. **Exploratory Data Analysis (EDA) & Cleaning:** Checking dataset dimensions using `df.shape`, identifying null values via `df.isnull().sum()`, and dropping redundant unmapped structural columns (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`).
2. **Data Preprocessing:** Mapping target strings to clean machine-readable integers and structuring text inputs into independent feature series.
3. **Feature Extraction (Text Vectorization):** Implementing `CountVectorizer` to transform raw text sentences into a sparse numerical token-count matrix (Bag-of-Words model).
4. **Data Splitting:** Partitioning data into an 80% Training set and a 20% Testing set using `train_test_split` to guarantee an unbiased generalization check.
5. **Model Selection & Training:** Training a `MultinomialNB` (Multinomial Naive Bayes) classifier—an algorithm highly optimal for discrete word count frequencies.
6. **Evaluation:** Metric benchmarking using overall classification accuracy alongside a breakdown of precision, recall, and F1-scores to monitor false-positive vs. false-negative trade-offs.

## 📈 Model Performance Report

| Model | Class | Precision | Recall | F1-Score | Overall Accuracy |
| --- | --- | --- | --- | --- | --- |
| **Multinomial Naive Bayes** | **0 (Ham)** | 0.99 | 0.99 | 0.99 | **97.9%** |
|  | **1 (Spam)** | 0.91 | 0.94 | 0.92 |  |

### Key Takeaway:

The Naive Bayes architecture proves highly effective for this text classification task, maintaining an exceptional **99% precision and recall for legitimate messages**. While spam detection pulls a slightly lower precision (91%) due to a few borderline promotional phrases crossing boundaries, the model captures **94% of incoming spam (recall)**, establishing it as a reliable baseline filter.

## 🚀 Technologies Used

* **Python 3**
* **Pandas** (Data cleaning, column mapping, and structural adjustments)
* **Scikit-Learn** (Text vectorization, train/test splitting, Naive Bayes modeling, and classification metrics)
* **Google Colab** (Interactive development environment)
