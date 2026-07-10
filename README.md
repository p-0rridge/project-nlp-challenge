
**News and Fake News Detection: Model Evaluation and Structural Bias Analysis**

## About
This project implements and evaluates a progressive series of NLP models for text classification. While the models achieve exceptionally high accuracy, the project includes a critical structural analysis showing how data leakage and writing-style artifacts can artificially inflate baseline performance.

## Short description
A comparative Machine Learning project evaluating four different NLP approaches—ranging from traditional Naive Bayes to dense Word Embeddings—to detect fake news, combined with a deep dive into dataset-specific formatting biases.


## Problem statement
* **Dataset used:** News and Fake News Dataset
* **Task:** Binary Text Classification (True vs. Fake)
* **Goal:** To compare different vectorization and modeling techniques, evaluate their generalization capabilities, and identify potential data leakage.

## Dataset
* **Source:** Public News and Fake News Dataset
* **Size:** *Hier ungefähre Zeilenanzahl eintragen, z.B. ~45,000 articles*
* **Classes:** 0 (Fake), 1 (True)
* **Link:** *Hier Link einfügen oder Zeile löschen*
* **License:** Public Dataset / Academic Use

## Model architecture
The project progressively evaluates four distinct modeling pipelines to track performance gains:
1. **Baseline:** Naive Bayes with `CountVectorizer`
2. **Iteration 1:** Naive Bayes with `TF-IDF Vectorizer`
3. **Iteration 2:** Logistic Regression with `CountVectorizer`
4. **Iteration 3:** Logistic Regression with Word Embeddings (`Word2Vec` average pooling)

## Results
* **Baseline (NB + Count):** *95.20%* Accuracy
* **Iteration 1 (NB + TFIDF):** *93.99%* Accuracy
* **Iteration 2 (LR + Count):** *98.44%* Accuracy
* **Iteration 3 (LR + Word2Vec):** *96.84%* Accuracy

### Key Finding (The "Prettified Trap")
Despite the high accuracy across all models, the critical finding of this project lies in the data structure itself. The "True News" class consists heavily of automated news ticker snippets. The models are not necessarily learning deep "fake vs. real" semantics; instead, they easily optimize for structural artifacts unique to real news formatting (e.g., timestamps, weekdays, and journalistic placeholders like `said` or `told`) that are mostly absent in the fake news class.

