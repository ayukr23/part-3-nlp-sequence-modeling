# Part 3: NLP and Sequence Modeling

## Project Overview
Customer support ticket sentiment classification using traditional ML and sequence modeling approaches.

**Dataset:** 1,500 customer support messages  
**Task:** 3-class sentiment classification (positive / negative / neutral)

## Results Summary

| Model | Vectorization | Accuracy |
|-------|--------------|----------|
| Logistic Regression | TF-IDF (bigrams) | 100.00% |
| Naïve Bayes | Bag of Words | 100.00% |

> High accuracy reflects clearly separable sentiment vocabulary in this dataset.

## Repository Structure
```
part-3-nlp-sequence-modeling/
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── model_evaluation.png
    └── sample_predictions.txt
```

## Task Completion

- **Task 1: Dataset Understanding** — 1,500 records, 3 balanced classes, avg 12.7 words/message
- **Task 2: Preprocessing** — lowercase, strip special chars, tokenize, remove stopwords
- **Task 3: Vectorization** — Bag of Words (CountVectorizer) + TF-IDF with bigrams
- **Task 4: Baseline Models** — Logistic Regression & Naïve Bayes with classification reports
- **Task 5: LSTM Architecture** — Embedding → LSTM(64) → Dense(32) → Softmax(3)
- **Task 6: Transformer Reflection** — RNN limitations, LSTM gating, attention, transformers

