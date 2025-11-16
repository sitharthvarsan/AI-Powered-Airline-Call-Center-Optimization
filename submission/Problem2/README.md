Problem 2— Airline Review Sentiment Classification System

1. Installation

Install all required dependencies:
pip install -r sentiment_model/requirements.txt
Ensure the dataset file is present:
sentiment_model/2026_dataset.xlsx

2. Running the Code

Execute the main script:
python sentiment_model/main.py

This will automatically:
•	Load and clean the dataset
•	Auto-label each review using the RoBERTa sentiment model
•	Train a TF-IDF + Logistic Regression classifier
•	Print evaluation metrics
•	Output sentiment predictions for example reviews
Results will be printed as plain text:
positive
negative

3. Approach

RoBERTa-Based Auto-Labeling
A pretrained model (cardiffnlp/twitter-roberta-base-sentiment) is used to generate sentiment labels for each review.
This replaces unreliable manual rule-based thresholds and ensures high-quality ground-truth labels.
TF-IDF + Logistic Regression Classifier
Once reviews are labeled by RoBERTa, a lightweight ML classifier is trained to reproduce the same sentiment predictions.
This allows fast, low-resource inference without needing to run RoBERTa during deployment.
Pipeline Summary
Review Text → RoBERTa Label → TF-IDF Features → Logistic Regression → Final Sentiment Output