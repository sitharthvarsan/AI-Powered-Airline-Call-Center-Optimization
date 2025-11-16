# ✈️ AI Projects — Airline Automation & Sentiment Intelligence

### 🚀 *Problem 1: Two-Agent Airline Information System*

### 🤖 *Problem 2: Airline Review Sentiment Classification System*

---

## 🌟 Overview

This repository contains **two advanced AI systems** built around the airline domain:

### **🔹 Problem 1 — Two-Agent Airline Information System**

A multi-agent LLM pipeline that answers user queries about flight details using **LangChain**, **tool calling**, and **structured JSON responses**.

### **🔹 Problem 2 — Airline Review Sentiment Classification System**

A hybrid NLP model combining **RoBERTa transformer auto-labeling** with a lightweight **TF-IDF + Logistic Regression classifier** for fast inference.

Both projects demonstrate:

✔ Multi-agent reasoning
✔ Tool integration / function calling
✔ Text classification
✔ Transformer-based NLP
✔ Clean modular code
✔ Production-like architecture

---

# 🧩 Problem 1 — Two-Agent Airline Information System

## 📝 Summary

This project uses a **QA Agent** + **Info Agent** architecture to answer user questions about flight details.

* The **QA Agent** interprets the query and extracts the flight number.
* The **Info Agent** is implemented as a LangChain `@tool` that fetches flight details from a CSV dataset.
* The final answer must always follow a strict JSON schema:

```json
{ "answer": "..." }
```

---

## 🔑 Key Features

### ✨ Multi-Agent Pipeline

QA Agent orchestrates Info Agent calls to fetch relevant data.

### ✨ Tool Calling

Info Agent is implemented as a LangChain tool with structured outputs.

### ✨ Strict JSON Responses

Enforced using Pydantic models to ensure format consistency.

### ✨ Intelligent Prompting

QA Agent is instructed to extract flight numbers and use tools when needed.

---

## 📂 Project Structure

```
submission/
└── problem1/
    ├── main.py
    ├── api_keys.env
    ├── indian_flights_dataset_2000_nozeros.csv
    ├── requirements.txt
    └── README.md
```

---

## ⚙️ Installation

```bash
pip install -r problem1/requirements.txt
```

Add your OpenAI key inside:

```
problem1/api_keys.env
```

Format:

```
OPENAI_API_KEY="your-key-here"
```

---

## ▶️ Running the Program

```bash
python problem1/main.py
```

---

## 🧪 Test Behavior

### ✔ Valid Flight Query

Extracts the flight number → calls Info Agent → returns flight details as JSON.

### ✔ Invalid Flight Query

Returns a JSON error message:

```json
{ "answer": "Flight not found." }
```

### 📌 Example Output

```json
{
  "answer": "Flight AI123 departs at 08:00 AM to Delhi. Current status: On Time."
}
```

---

# 💬 Problem 2 — Airline Review Sentiment Classification System

## 📝 Summary

This project performs high-quality sentiment classification on airline reviews using a **two-step hybrid pipeline**:

### **Step 1 — Auto-Labeling Using RoBERTa**

* Pretrained RoBERTa model:
  `cardiffnlp/twitter-roberta-base-sentiment`
* Generates sentiment labels (*positive* or *negative*)
* Avoids noisy thresholding based on review scores

### **Step 2 — Train Lightweight ML Classifier**

* TF-IDF vectorizer
* Logistic Regression
* Extremely fast inference
* Suitable for deployment

All final outputs are plain-text sentiment:

```
positive
negative
```

---

## 🔑 Key Features

### 🔹 RoBERTa Auto-Labeling

Removes manual rules, improves label quality.

### 🔹 TF-IDF + Logistic Regression

Efficient classifier trained on transformer-quality labels.

### 🔹 Text Preprocessing

Handles:

* Lowercase
* Punctuation cleanup
* ASCII normalization
* Negation handling (`not good → not_good`)

### 🔹 Model Evaluation

Outputs:

* Accuracy
* Confusion matrix
* Classification report

---

## 📂 Project Structure

```
submission/
└── sentiment_model/
    ├── main.py
    ├── requirements.txt
    ├── 2026_dataset.xlsx
    └── README.md
```

---

## ⚙️ Installation

```bash
pip install -r sentiment_model/requirements.txt
```

Place your dataset in:

```
sentiment_model/2026_dataset.xlsx
```

Required columns:

| Column       | Description                       |
| ------------ | --------------------------------- |
| Title        | Airline review text               |
| OverallScore | Numeric score (not used directly) |

---

## ▶️ Running the Program

```bash
python sentiment_model/main.py
```

---

## 🧪 What Happens When You Run It

### ✔ 1. Data Loading

Reads Excel → drops missing rows.

### ✔ 2. RoBERTa Auto-Labeling

Each review is assigned:

```
positive
negative
```

### ✔ 3. Model Training

TF-IDF + Logistic Regression classifier is trained.

### ✔ 4. Evaluation Printed

Accuracy + metrics displayed.

### ✔ 5. Demo Predictions

Outputs sentiment for predefined examples:

```
positive
negative
negative
```

---

# 🔧 Core Dependencies (Both Problems)

* **LangChain** • Multi-agent system
* **Transformers** • RoBERTa tokenizer & model
* **OpenAI / LangChain-OpenAI**
* **Torch** • Model backend
* **Pandas** • Dataset operations
* **Scikit-Learn** • TF-IDF + Logistic Regression
* **Python-dotenv** • API key management

---

# 🏁 Final Notes

This repository showcases:

✨ Agentic AI
✨ Transformer-based NLP
✨ Applied Machine Learning
✨ Real airline domain use cases
