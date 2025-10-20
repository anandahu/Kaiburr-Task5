# 🤖 Consumer Financial Complaints Classification System

> Multi-class text classification using Machine Learning and Deep Learning to categorize consumer financial complaints

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5+-150458.svg?logo=pandas)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter)](https://jupyter.org/)

---

## 📋 Project Overview

This project implements a comprehensive **text classification pipeline** to automatically categorize consumer financial complaints into four distinct categories. Using data from the Consumer Financial Protection Bureau (CFPB), we build and compare multiple machine learning models to achieve high-accuracy classification.

### Business Problem

Financial institutions receive thousands of consumer complaints daily. Manual categorization is:
- ⏰ **Time-consuming**: Hours of manual review
- 💰 **Expensive**: Requires skilled staff
- 🎯 **Inconsistent**: Human error and bias
- 📈 **Unscalable**: Can't handle volume spikes

### Our Solution

An automated ML system that:
- ✅ Classifies complaints with **97.25% accuracy**
- ⚡ Processes complaints in milliseconds
- 📊 Handles imbalanced datasets effectively
- 🔄 Scales to millions of complaints

---

## 🎯 Classification Categories

| Label | Category | Example Issues |
|-------|----------|----------------|
| **0** | **Credit Reporting** | Incorrect information, identity theft, credit repair |
| **1** | **Debt Collection** | Illegal contact, false threats, harassment |
| **2** | **Consumer Loans** | Vehicle loans, personal loans, student loans |
| **3** | **Mortgage** | Loan modifications, foreclosure, servicing issues |

---

## ✨ Key Features

### Data Processing
- 📥 **Large Dataset Handling**: 100,000+ complaint records
- 🔄 **Intelligent Resampling**: Balances imbalanced classes
- 🧹 **Text Preprocessing**: Comprehensive cleaning pipeline
- 📊 **Feature Engineering**: TF-IDF, n-grams, and statistical features

### Model Development
- 🤖 **Multiple Algorithms**: Logistic Regression, Naive Bayes, SVM, Neural Networks
- 📈 **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- 🎨 **Rich Visualizations**: Confusion matrices, word clouds, distribution plots
- 🏆 **Model Comparison**: Side-by-side performance analysis

### Advanced Analytics
- 📝 **Text Analytics**: Word frequency, n-gram analysis, length distributions
- ☁️ **Word Clouds**: Visual representation of common terms per category
- 📊 **Statistical Analysis**: Comprehensive exploratory data analysis (EDA)

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Data Acquisition                        │
│         (CFPB Consumer Complaints Database)              │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Data Preprocessing                          │
│  • Category Mapping                                      │
│  • Text Cleaning                                         │
│  • Class Balancing                                       │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Exploratory Data Analysis (EDA)                  │
│  • Distribution Analysis                                 │
│  • Word Clouds                                           │
│  • N-gram Analysis                                       │
│  • Statistical Features                                  │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           Feature Engineering                            │
│  • TF-IDF Vectorization                                  │
│  • Feature Selection (Chi-squared)                       │
│  • Dimensionality Reduction                              │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Model Training                              │
│  • Logistic Regression                                   │
│  • Multinomial Naive Bayes                               │
│  • Linear SVM                                            │
│  • Feed-Forward Neural Network (PyTorch)                 │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│        Model Evaluation & Comparison                     │
│  • Performance Metrics                                   │
│  • Confusion Matrices                                    │
│  • Model Selection                                       │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and tools

### Deep Learning
- **PyTorch**: Neural network framework
- **torch.nn**: Neural network layers
- **torch.optim**: Optimization algorithms

### NLP & Text Processing
- **NLTK**: Natural language processing toolkit
- **WordCloud**: Text visualization
- **CountVectorizer & TfidfVectorizer**: Text feature extraction

### Visualization
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualizations
- **wordcloud**: Word cloud generation

### Additional Tools
- **imbalanced-learn**: Handling imbalanced datasets
- **joblib**: Model serialization

---

## 📁 Project Structure

```
consumer-complaints-classification/
│
├── data/
│   ├── complaints.csv                      # Raw dataset
│   └── complaints.csv.zip                  # Compressed dataset
│
├── notebooks/
│   └── classification_pipeline.ipynb       # Main Jupyter notebook
│
├── models/                                  # Saved models
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── linear_svm.pkl
│   └── ffnn_model.pth
│
├── visualizations/                          # Generated plots
│   ├── distribution_plots/
│   ├── word_clouds/
│   ├── confusion_matrices/
│   └── model_comparison.png
│
├── src/                                     # Source code (optional modular structure)
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── evaluation.py
│
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

---

## 🚀 Getting Started

### Prerequisites

- ✅ **Python 3.8+** installed
- ✅ **Jupyter Notebook** or **JupyterLab**
- ✅ **8GB+ RAM** recommended for large dataset processing
- ✅ **GPU** (optional, for faster neural network training)

### Installation

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd consumer-complaints-classification
```

#### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -q scikit-learn xgboost imbalanced-learn matplotlib seaborn nltk wordcloud transformers datasets torch pandas numpy jupyter
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
```

#### 4. Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

---

## 📊 Dataset Information

### Source
**Consumer Financial Protection Bureau (CFPB)**
- URL: https://files.consumerfinance.gov/ccdb/complaints.csv.zip
- Size: ~1.5 GB compressed
- Records: 100,000+ complaints (subset used)

### Dataset Features

| Column | Description | Example |
|--------|-------------|---------|
| **Date received** | Date complaint was received | 2025-10-14 |
| **Product** | Financial product category | Credit reporting |
| **Sub-product** | Specific product type | Credit reporting |
| **Issue** | Primary issue | Incorrect information |
| **Sub-issue** | Detailed issue | Identity theft |
| **Consumer complaint narrative** | Free-text complaint description | "I found errors..." |
| **Company** | Company name | EQUIFAX, INC. |
| **State** | Consumer's state | TX |
| **ZIP code** | Consumer's ZIP | 75062 |
| **Company response** | Response status | In progress |

### Data Statistics (After Processing)

```
Total Records: 97,452
Class Distribution (After Balancing):
  - Credit Reporting: 66.7%
  - Debt Collection: 23.8%
  - Consumer Loans: 4.9%
  - Mortgage: 4.7%

Text Statistics:
  - Average Word Count: 25.6 words
  - Average Character Count: 152.6 characters
  - Max Word Count: 2,018 words
```

---

## 🔬 Methodology

### 1. Data Acquisition & Loading

```python
# Download dataset
!wget -O complaints.csv.zip "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
!unzip complaints.csv.zip

# Load data (first 100,000 rows for efficiency)
df = pd.read_csv('complaints.csv', nrows=100000, low_memory=False)
```

### 2. Category Mapping

Intelligent product categorization:

```python
def map_category(prod):
    p = str(prod).lower()
    if "credit" in p:
        return 0  # Credit reporting
    elif "debt" in p:
        return 1  # Debt collection
    elif "mortgage" in p:
        return 3  # Mortgage
    elif "loan" in p and "mortgage" not in p:
        return 2  # Consumer Loan
    else:
        return None
```

### 3. Text Preprocessing Pipeline

Comprehensive text cleaning:

```python
1. Lowercase conversion
2. Remove special characters (keep only letters)
3. Tokenization (word_tokenize)
4. Stopword removal (NLTK stopwords)
5. Stemming (PorterStemmer)
6. Lemmatization (WordNetLemmatizer)
7. Short word removal (< 3 characters)
8. Join tokens back to text
```

### 4. Handling Class Imbalance

**Problem**: 89,811 Credit Reporting vs 1,070 Mortgage complaints

**Solution**: Strategic downsampling

```python
from sklearn.utils import resample

# Downsample majority class (Credit Reporting)
df_majority_down = resample(df_majority,
                            replace=False,
                            n_samples=len(df_minority)*2,
                            random_state=42)

# Combine with minority classes
df_balanced = pd.concat([df_majority_down, df_minority])
```

**Result**: Balanced distribution (66.7% / 23.8% / 4.9% / 4.7%)

### 5. Feature Engineering

#### TF-IDF Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1,2)  # Unigrams + Bigrams
)
X_tfidf = tfidf.fit_transform(corpus)
```

#### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, chi2

# Select top 1000 features using Chi-squared test
selector = SelectKBest(chi2, k=1000)
X_selected = selector.fit_transform(X_tfidf, y)
```

### 6. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintain class distribution
)

# Train: 18,338 samples | Test: 4,585 samples
```

---

## 🤖 Models Implemented

### 1. Logistic Regression

**Configuration**:
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    multi_class='ovr',      # One-vs-Rest
    max_iter=1000,
    random_state=42
)
```

**Performance**:
- ✅ Accuracy: **97.10%**
- ✅ Fast training and prediction
- ✅ Interpretable coefficients
- ✅ Works well with high-dimensional sparse data

---

### 2. Multinomial Naive Bayes

**Configuration**:
```python
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
```

**Performance**:
- ✅ Accuracy: **95.66%**
- ✅ Extremely fast training
- ✅ Works well with text data
- ⚠️ Assumes feature independence

---

### 3. Linear Support Vector Machine (SVM)

**Configuration**:
```python
from sklearn.svm import LinearSVC

svc = LinearSVC(
    max_iter=5000,
    random_state=42
)
```

**Performance**:
- ✅ Accuracy: **97.12%**
- ✅ Effective in high-dimensional spaces
- ✅ Robust to overfitting
- ⚠️ Slower training than Naive Bayes

---

### 4. Feed-Forward Neural Network (PyTorch)

**Architecture**:
```python
class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Configuration
input_dim = 1000      # TF-IDF features
hidden_dim = 128      # Hidden layer neurons
num_classes = 4       # Categories
```

**Training Configuration**:
```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
batch_size = 64
epochs = 5
```

**Performance**:
- ✅ Accuracy: **97.25%** (Best performance!)
- ✅ Can learn complex patterns
- ✅ Flexible architecture
- ⚠️ Requires more computational resources

---

## 📈 Results & Performance

### Model Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | 97.10% | 97.09% | 97.10% | 97.02% |
| **Multinomial NB** | 95.66% | 95.70% | 95.66% | 95.67% |
| **Linear SVM** | 97.12% | 97.10% | 97.12% | 97.05% |
| **FFNN (PyTorch)** | **97.25%** | **97.23%** | **97.25%** | **97.20%** |

### Per-Class Performance (FFNN - Best Model)

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| **Credit Reporting** | 0.97 | 0.99 | 0.98 | 3,057 |
| **Debt Collection** | 0.97 | 0.96 | 0.96 | 1,090 |
| **Consumer Loans** | 0.95 | 0.79 | 0.86 | 224 |
| **Mortgage** | 0.99 | 0.95 | 0.97 | 214 |

### Key Insights

1. **🏆 Best Overall**: Feed-Forward Neural Network (97.25%)
2. **⚡ Fastest**: Multinomial Naive Bayes (95.66%)
3. **⚖️ Best Balance**: Linear SVM (97.12%)
4. **📊 Consistent**: All models > 95% accuracy

---

## 📊 Visualizations

### 1. Word Cloud Analysis

**Purpose**: Identify most frequent terms per category

```python
from wordcloud import WordCloud

text_all = ' '.join(df_balanced['text'].astype(str))
wordcloud = WordCloud(
    width=1000,
    height=600,
    background_color='white'
).generate(text_all)
```

**Key Findings**:
- **Credit Reporting**: "incorrect information", "report", "investigation"
- **Debt Collection**: "collect debt", "attempts", "written notification"
- **Consumer Loans**: "loan lease", "lender servicer", "managing loan"
- **Mortgage**: "loan modification", "payment process", "trouble payment"

---

### 2. N-gram Analysis

**Top Bigrams/Trigrams**:

| N-gram | Frequency | Category |
|--------|-----------|----------|
| "xxxx xxxx" | 16,412 | All (redacted info) |
| "incorrect information" | 8,348 | Credit Reporting |
| "collect debt" | 1,838 | Debt Collection |
| "loan modification" | 264 | Mortgage |

---

### 3. Distribution Analysis

**Word Count Distribution**:
- Mean: 25.6 words per complaint
- Median: 5 words
- Max: 2,018 words
- Credit Reporting complaints tend to be longer

---

### 4. Confusion Matrices

Visual representation showing:
- ✅ High diagonal values (correct predictions)
- ⚠️ Few off-diagonal values (misclassifications)
- Consumer Loans has slightly more confusion

---

## 🔍 Exploratory Data Analysis Highlights

### Statistical Features

```python
# Engineered features
df_balanced['word_count'] = df_balanced['text'].apply(lambda x: len(str(x).split()))
df_balanced['char_count'] = df_balanced['text'].apply(lambda x: len(str(x)))
df_balanced['avg_word_length'] = df_balanced['char_count'] / (df_balanced['word_count'] + 1)
```

### Distribution Insights

1. **Skewed Distribution**: Most complaints are short (5-9 words)
2. **Outliers**: Some extremely long narratives (1000+ words)
3. **Category Differences**: Credit Reporting has longest complaints on average

### Top Companies by Complaints

Most frequent companies in dataset:
1. EQUIFAX, INC.
2. TRANSUNION INTERMEDIATE HOLDINGS, INC.
3. Experian Information Solutions Inc.
4. CAPITAL ONE FINANCIAL CORPORATION

---

## 💡 Key Takeaways

### What Worked Well

1. **✅ TF-IDF + Chi-squared**: Excellent feature representation
2. **✅ Class Balancing**: Improved minority class performance
3. **✅ Deep Learning**: Slight edge over classical ML
4. **✅ Comprehensive Preprocessing**: Clean data = better models

### Challenges Addressed

1. **⚠️ Imbalanced Classes**: Downsampling solved this
2. **⚠️ High Dimensionality**: Feature selection reduced to 1000 features
3. **⚠️ Large Dataset**: Processed 100K rows efficiently
4. **⚠️ Sparse Text**: TF-IDF handles this naturally

### Areas for Improvement

1. **🔄 Advanced Models**: Try BERT, RoBERTa, or other transformers
2. **📊 More Data**: Full dataset might improve performance
3. **🎯 Ensemble Methods**: Combine predictions from multiple models
4. **⚙️ Hyperparameter Tuning**: Grid/random search for optimization
5. **📝 More Features**: Add metadata (state, company, date features)

---

## 🚀 Usage

### Quick Start

```python
# 1. Load and preprocess data
import pandas as pd
df = pd.read_csv('complaints.csv', nrows=100000)

# 2. Apply category mapping
df['label'] = df['Product'].apply(map_category)
df['text'] = df['Consumer complaint narrative'].fillna(df['Issue'])

# 3. Balance dataset
from sklearn.utils import resample
# ... (balancing code)

# 4. Feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(corpus)

# 5. Train model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Predict
predictions = model.predict(X_test)
```

### Making Predictions on New Complaints

```python
# Load trained model
import joblib
model = joblib.load('models/logistic_regression.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

# New complaint
new_complaint = "I found incorrect information on my credit report"

# Transform and predict
X_new = tfidf.transform([new_complaint])
prediction = model.predict(X_new)

categories = {0: 'Credit Reporting', 1: 'Debt Collection', 
              2: 'Consumer Loans', 3: 'Mortgage'}
print(f"Category: {categories[prediction[0]]}")
```




## 🐛 Troubleshooting

### Common Issues

#### Issue: Out of Memory

**Symptom**: Kernel crashes during training

**Solution**:
```python
# Reduce nrows when loading
df = pd.read_csv('complaints.csv', nrows=50000)

# Or reduce max_features
tfidf = TfidfVectorizer(max_features=1000)  # Instead of 5000
```

#### Issue: NLTK Data Not Found

**Symptom**: LookupError: Resource stopwords not found

**Solution**:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

#### Issue: PyTorch CUDA Errors

**Symptom**: CUDA out of memory

**Solution**:
```python
# Force CPU execution
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Or reduce batch size
train_loader = DataLoader(train_dataset, batch_size=32)  # Instead of 64
```

---



## 📚 References & Resources

### Datasets
- [CFPB Consumer Complaints Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)

### Documentation
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [NLTK Documentation](https://www.nltk.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### Research Papers
- Joulin et al. (2016) - "Bag of Tricks for Efficient Text Classification"
- Devlin et al. (2018) - "BERT: Pre-training of Deep Bidirectional Transformers"
- Kim (2014) - "Convolutional Neural Networks for Sentence Classification"



---<img width="1920" height="1080" alt="Screenshot (221)" src="https://github.com/user-attachments/assets/3066c372-5fe6-47ef-a0ac-de80b57849c1" />
<img width="1920" height="1080" alt="Screenshot (210)" src="https://github.com/user-attachments/assets/7e4ea2f5-2a85-43b4-96cf-dbee52d6dc4e" />
<img width="1920" height="1080" alt="Screenshot (211)" src="https://github.com/user-attachments/assets/19046255-4045-414b-a73d-7050b3dcf211" />
<img width="1920" height="1080" alt="Screenshot (214)" src="https://github.com/user-attachments/assets/a6f4366b-a0e7-41d8-ab6e-9bc5ed0ddcc5" />
<img width="1920" height="1080" alt="Screenshot (215)" src="https://github.com/user-attachments/assets/c593eaf1-b263-47e4-95b5-bbb47e19c8ee" />
<img width="1920" height="1080" alt="Screenshot (216)" src="https://github.com/user-attachments/assets/728b26d6-5515-4a20-ba97-474dd6aeb8d7" />
<img width="1920" height="1080" alt="Screenshot (217)" src="https://github.com/user-attachments/assets/85c6c849-8e7d-4b37-b9d5-3c492ebd3f36" />
<img width="1920" height="1080" alt="Screenshot (218)" src="https://github.com/user-attachments/assets/7cefdb2b-3bdc-493b-9dd8-9a8e47f7f239" />
<img width="1920" height="1080" alt="Screenshot (219)" src="https://github.com/user-attachments/assets/bd92b02b-efac-42db-a5dd-d66190b4b6cc" />
<img width="1920" height="1080" alt="Screenshot (220)" src="https://github.com/user-attachments/assets/ac521363-1d7a-4b40-8972-471d9fd4c1b1" />


