# ✈️ AI Projects — Airline Automation & Sentiment Intelligence

### 🚀 *Problem 1: Two-Agent Airline Information System*

### 🤖 *Problem 2: Airline Review Sentiment Classification System*

---

## 🌟 Overview

This repository contains **two advanced AI systems** focused on airline automation:

### **🔹 Problem 1 — Two-Agent Airline Information System**

A multi-agent LangChain-based application that interprets flight-related queries, extracts flight numbers, and retrieves structured flight information using intelligent tool calling.

### **🔹 Problem 2 — Airline Review Sentiment Classification System**

A hybrid NLP pipeline that auto-labels airline reviews using a RoBERTa Transformer model and trains a lightweight TF-IDF + Logistic Regression classifier for fast sentiment prediction.

Together, these projects demonstrate:

✔ Multi-agent reasoning
✔ Tool/function calling
✔ Transformer-based NLP
✔ End-to-end ML pipeline design
✔ Clean, production-ready architecture

---

# 🧩 Problem 1 — Two-Agent Airline Information System

## 📝 Overview

This system uses a **two-agent architecture** to answer airline-related queries.

* **QA Agent**: Understands user questions, extracts the flight number.
* **Info Agent**: Implemented as a LangChain `@tool` that fetches flight info from a dataset.

All responses follow a strict JSON schema:

```json
{ "answer": "..." }
```

---

## 🔑 Key Features

### ✨ Multi-Agent Collaboration

The QA Agent coordinates with the Info Agent through automated tool calls.

### ✨ LangChain Tool Integration

Info Agent fetches flight data from the CSV dataset.

### ✨ Enforced JSON Formatting

Pydantic schema ensures consistent output format.

### ✨ Prompt Engineering

The QA Agent is instructed to always return valid JSON.

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

Add your OpenAI key in:

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

## 🧪 Expected Behavior

### ✔ Valid Flight Query

Extracts flight number → calls Info Agent → returns JSON answer.

### ✔ Invalid Flight Query

Returns error JSON:

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

## 📝 Overview

This project classifies airline reviews as **positive** or **negative** using a **hybrid two-step NLP approach**:

### **1. Auto-Labeling with RoBERTa**

Pretrained model:
`cardiffnlp/twitter-roberta-base-sentiment`

This model analyzes each review and assigns high-quality sentiment labels.

### **2. Classifier Training (TF-IDF + Logistic Regression)**

A lightweight classifier learns from RoBERTa-generated labels and provides extremely fast inference.

This approach combines the **accuracy of transformers** with the **speed of classical ML**.

Final output values:

```
positive
negative
```

---

## 🔑 Key Features

### 🔹 RoBERTa Auto-Labeling

Removes manual score-based heuristics and improves label quality.

### 🔹 TF-IDF + Logistic Regression

Efficient, explainable, and ideal for deployment.

### 🔹 Text Preprocessing

Handles lowercase conversion, punctuation cleanup, ASCII normalization, and negation handling (`not good → not_good`).

### 🔹 Detailed Model Evaluation

Outputs accuracy, confusion matrix, and classification report.

---

## 📊 **Model Performance**

Based on the dataset provided:

* **Training Accuracy:** **97%**
* **Testing Accuracy:** **90%**

These results are expected for transformer-quality labels combined with TF-IDF features.

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

Place your dataset into:

```
sentiment_model/2026_dataset.xlsx
```

Required columns:

| Column       | Description                |
| ------------ | -------------------------- |
| Title        | Review text                |
| OverallScore | Rating (not used directly) |

---

## ▶️ Running the Program

```bash
python sentiment_model/main.py
```

---

## 🧪 What Happens During Execution

### **1. Load and Clean Dataset**

Missing rows removed.

### **2. Auto-Label with RoBERTa**

Each review gets `"positive"` or `"negative"`.

### **3. Train the Classifier**

TF-IDF + Logistic Regression pipeline is trained on the generated labels.

### **4. Evaluate the Model**

The script prints detailed metrics.

### **5. Predict Sample Reviews**

Example output:

```
positive
negative
negative
```

---

# 🧰 Core Dependencies (Both Problems)

* **LangChain** — Agentic architecture
* **LangChain-OpenAI / OpenAI API**
* **Transformers** — RoBERTa model
* **Torch** — Inference backend
* **Pandas** — Data loading
* **NumPy** — Numerical utilities
* **Scikit-Learn** — TF-IDF, Logistic Regression
* **Python-dotenv** — API key handling

---

# 🏁 Final Notes

Both the systems in this repository showcase:

✨ Real-world airline domain automation
✨ Intelligent multi-agent LLM systems
✨ Transformer-driven text classification
✨ Practical and efficient ML deployment patterns

Just ask — happy to help!
