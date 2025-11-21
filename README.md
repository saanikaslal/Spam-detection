# 🧠 Spam Detection using LSTM + Profit Data Visualization

## 📋 Project Overview

This project demonstrates two key **Machine Learning** and **Data Analysis** concepts:

1. **Spam Detection using LSTM (Deep Learning)**  
   A text classification model that classifies emails or messages as **spam** or **ham** (not spam).

2. **Data Visualization**  
   Analyzing and visualizing **product-based profit data** using **Matplotlib** and **Pandas**.

---

## 📁 Dataset

The dataset used for spam detection has the following columns:

| Column       | Description                                  |
|---------------|----------------------------------------------|
| `Unnamed: 0`  | Row index (can be ignored)                  |
| `label`       | Text label (`ham` or `spam`)                |
| `text`        | Email/message content                        |
| `label_num`   | Numeric label (`0` = ham, `1` = spam)        |

Example:

| Unnamed: 0 | label | text                                   | label_num |
|-------------|--------|----------------------------------------|------------|
| 605         | ham    | Subject: enron methanol ...            | 0          |
| 4685        | spam   | Subject: photoshop, windows, office... | 1          |

---

## 🚀 Model Overview: LSTM for Spam Detection

### 🔹 Steps Involved

1. **Data Preprocessing**
   - Tokenize text data using `Tokenizer`
   - Convert tokens into padded sequences
   - Split dataset into training and testing sets

2. **Model Architecture**
   ```python
   model = Sequential([
       Embedding(input_dim=vocab_size, output_dim=64, input_length=maxlen),
       LSTM(64, return_sequences=False),
       Dropout(0.5),
       Dense(1, activation='sigmoid')
   ])
   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
