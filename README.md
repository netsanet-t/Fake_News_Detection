# 📰 Fake News Detection with NLP and Deep Learning

This project demonstrates how to build a fake news detection system using Natural Language Processing (NLP) techniques and a deep learning model implemented with TensorFlow/Keras. It involves data cleaning, text preprocessing, exploratory data analysis (EDA), and training/testing of a binary classifier to detect misleading information.

## 📁 Dataset

* `train.csv`: Training dataset with labeled news articles.
* `test.csv`: Unlabeled news data for prediction/evaluation.

## 📌 Key Features

* Data cleaning (null handling, duplicates removal)
* Exploratory Data Analysis (EDA)
* Advanced NLP preprocessing (stemming, lemmatization, tokenization, etc.)
* Feature extraction using `TF-IDF`
* Deep learning model using LSTM layers
* Model evaluation with accuracy, confusion matrix, and classification report

## 🧪 Steps in the Notebook

### 1. Load and Clean Data

* Check for missing values
* Drop duplicates
* Prepare train/test sets

### 2. Exploratory Data Analysis (EDA)

* Distribution of fake vs. real news
* Text statistics (word count, sentence length)
* Word clouds and histograms

### 3. Text Preprocessing

* Lowercasing
* Removing punctuation and special characters
* Stopwords removal
* Tokenization
* Stemming & Lemmatization (using `NLTK` and `spaCy`)

### 4. Feature Extraction

* TF-IDF vectorization of the cleaned text data

### 5. Model Building

* Sequential model with LSTM layer
* Dropout layers to prevent overfitting
* Binary classification (Fake or Real)

### 6. Evaluation

* Accuracy
* Confusion matrix
* Classification report

## 🛠️ Technologies Used

* Python 🐍
* Pandas & NumPy
* NLTK & spaCy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib & Seaborn

## 🧠 Model Architecture

```
Input → Embedding → LSTM → Dropout → Dense → Output
```

## 📊 Results

Achieved promising performance with a well-generalized model. Metrics such as accuracy and F1-score indicate the model is reliable for detecting fake news articles.

## 🚀 How to Run

1. Clone this repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:

   ```bash
   jupyter notebook Fake_News_Detection.ipynb
   ```

## 📌 Note

Ensure you have `train.csv` and `test.csv` files in your working directory before running the notebook.
