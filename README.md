# BD-CricketSent: A Dataset for Sentiment Analysis of Bangladeshi Cricket Fans' Comments

This repository contains the official dataset and codebase for the paper **BD-CricketSent: A Dataset for Sentiment Analysis of Bangladeshi Cricket Fans' Comments**. It provides a comprehensive framework for evaluating sentiment analysis models on noisy, low-resource, and highly informal Bangla social media text.

## Overview
Sentiment analysis in Bangla cricket commentary is challenging due to non-standard spellings, code-mixing, colloquial slang, and sarcasm. **BD-CricketSent** addresses this by providing a high-quality, balanced benchmark dataset of 25,780 social media comments. 

This repository includes scripts to reproduce the paper's experiments across three distinct modeling paradigms:
1. **Classical Machine Learning** (TF-IDF features)
2. **Deep Learning** (Word2Vec and FastText static embeddings)
3. **Pre-trained Transformers** (Contextual subword representations)

## Dataset
The BD-CricketSent dataset comprises **25,780 annotated comments** carefully curated and filtered from three sources:
* **Newly Scraped Facebook Data:** 5,149 comments
* **Bangla Sports Sentiment Corpus:** 17,652 filtered cricket comments
* **ABSA Cricket Corpus:** 2,979 comments

### Class Distribution
| Sentiment | Sample Count | Example Translation |
| :--- | :--- | :--- |
| **Positive** | 9,111 | "T Sports live coverage and scoreboard were amazing Best wishes" |
| **Negative** | 9,089 | "Rishad Hossain seeing your batting in yesterdays match hurt me deeply" |
| **Neutral** | 7,580 | "When and where will the Bangladesh vs Sri Lanka match take place" |

*Note: The raw Bengali text undergoes regex-based cleaning to strip Latin characters, digits, and emojis, retaining only valid Bengali Unicode characters.*

## Methodology
The codebase evaluates thirteen pipelines using consistent data partitions and hyperparameter tuning:

* **Classical ML:** Random Forest (RF), Logistic Regression (LR), Support Vector Machine (SVM), and Naive Bayes (NB) utilizing 10,000 TF-IDF unigram and bigram features.
* **Deep Neural Networks (DNN):** 1D-CNN, LSTM, and GRU architectures paired with 200-dimensional Word2Vec and FastText embeddings.
* **Transformers:** Fine-tuning scripts for **BanglaBERT**, **BanglaBERT-base**, and **XLM-RoBERTa** using the HuggingFace `transformers` library.

## Key Results
The results demonstrate that contextual subword representations (Transformers) significantly outperform traditional and static-embedding methods when handling the morphological richness and noise of informal Bangla text. 

| Model Family | Best Model | Embedding/Feature | Accuracy (%) | Macro $F_1$ (%) | Micro-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Classical ML** | Random Forest | TF-IDF | 82.93 | 83.77 | 0.958 |
| **Deep Learning** | LSTM | Word2Vec | 83.86 | 82.37 | 0.950 |
| **Deep Learning** | 1D-CNN | FastText | 83.45 | 83.18 | 0.960 |
| **Transformer** | **BanglaBERT** | **Contextual Subword** | **84.98** | **85.78** | **0.970** |

**Key Insights:**
* **Neutral is easiest:** All models performed best on the Neutral class ($F_1 \ge 0.97$) due to direct, factual vocabulary.
* **Embedding Trade-offs:** FastText benefits CNN architectures by capturing local subword features, while Word2Vec slightly favors LSTM networks for capturing sequential dependency.
* **Vocabulary Dynamics:** Positive sentiment relies heavily on honorifics and epithets (e.g., *Best*, *Legend*), whereas Negative sentiment is often driven by off-field controversies (e.g., *Gambling*).

## Repository Structure

```text
├── data/
│   ├── raw/                 # Original combined dataset
│   └── processed/           # Cleaned and tokenized splits (Train/Val/Test)
├── embeddings/              # Directory for pre-trained Word2Vec/FastText weights
├── src/
│   ├── preprocess.py        # Regex cleaning, filtering, and TF-IDF extraction
│   ├── train_ml.py          # Script for LR, NB, SVM, RF models
│   ├── train_dl.py          # Script for 1D-CNN, LSTM, GRU (PyTorch/TensorFlow)
│   └── train_transformer.py # Script for BanglaBERT and XLM-RoBERTa fine-tuning
├── requirements.txt         # Python dependencies
└── README.md
