Overview

This project performs Text Classification on the Consumer Complaint Database published by the U.S. Consumer Financial Protection Bureau (CFPB).
It categorizes consumer complaints into four main product-related categories using Machine Learning and Deep Learning models.

🎯 Objective

To automatically classify customer complaints into the following categories based on their textual description:

Label	Category
0	Credit reporting, repair, or other
1	Debt collection
2	Consumer loan
3	Mortgage
🧰 Technologies Used

Language: Python 3

Environment: Jupyter Notebook / Google Colab

Libraries:

Data Handling: pandas, numpy

Visualization: matplotlib, seaborn, wordcloud

Text Processing: nltk, re, string

Feature Extraction: CountVectorizer, TfidfVectorizer

Machine Learning: scikit-learn

Deep Learning: PyTorch

Balancing: imbalanced-learn

Model Persistence: joblib

Pretrained Transformers (optional): transformers, datasets

⚙️ Setup & Prerequisites
1. Clone the Repository
git clone https://github.com/your-username/consumer-complaint-text-classification.git
cd consumer-complaint-text-classification

2. Install Dependencies
pip install -q scikit-learn xgboost imbalanced-learn matplotlib seaborn nltk wordcloud transformers datasets torch

3. Download the Dataset

The official Consumer Complaint dataset is available at:

🔗 Consumer Complaint Database

Alternatively, you can download it directly in your Colab/Notebook:

!wget -O complaints.csv.zip "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
!unzip complaints.csv.zip

4. Import Libraries and Load Data
import pandas as pd
df = pd.read_csv('complaints.csv', nrows=100000, low_memory=False)

🧩 Steps & Methodology
1. Data Loading & Inspection

Load 100,000 rows for manageable analysis.

Inspect key columns like Product, Consumer complaint narrative, Issue, and Company.

2. Data Cleaning

Select relevant columns.

Handle missing values (fillna).

Remove rows with no meaningful text.

3. Category Mapping

The Product column is mapped to the four target labels using keyword matching:

def map_category(prod):
    p = str(prod).lower()
    if "credit" in p:
        return 0
    elif "debt" in p:
        return 1
    elif "mortgage" in p:
        return 3
    elif "loan" in p and "mortgage" not in p:
        return 2

4. Balancing the Dataset

The dataset is highly imbalanced (most are credit-related).
To fix this, downsampling is applied:

from sklearn.utils import resample
df_majority_down = resample(df_majority, replace=False, n_samples=len(df_minority)*2, random_state=42)

5. Exploratory Data Analysis (EDA)

Visualize class distribution using seaborn.

Analyze word counts, average word lengths, and top n-grams.

Generate Word Clouds for each category.

6. Text Preprocessing

Each complaint text is cleaned as follows:

text = text.lower()
text = re.sub(r'[^a-zA-Z]', ' ', text)
tokens = word_tokenize(text)
tokens = [word for word in tokens if word not in stopwords.words('english')]
tokens = [PorterStemmer().stem(word) for word in tokens]
tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if len(word) > 2]
text = ' '.join(tokens)

7. Feature Engineering

Multiple text representations are tested:

Bag of Words (BoW)

TF-IDF (Word-Level, N-Gram, Character-Level)

Feature Selection using Chi-Square (SelectKBest)

Final TF-IDF feature matrix:

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(df_balanced['text'])

8. Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, stratify=y, random_state=42)

🧠 Models Implemented
1. Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

2. Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)

3. Linear SVM
from sklearn.svm import LinearSVC
svc = LinearSVC(max_iter=5000)
svc.fit(X_train, y_train)

4. Feedforward Neural Network (FFNN)

A simple PyTorch-based network:

class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

model = FFNN(input_dim=1000, hidden_dim=128, num_classes=4)


Trained for 5 epochs using CrossEntropyLoss and Adam optimizer.

📊 Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Confusion Matrix (plotted for all models)

Example:

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test, y_pred_lr))

🧾 Results Summary
Model	Accuracy	Precision	Recall	F1-score
Logistic Regression	0.9709	0.9709	0.9709	0.9702
Multinomial NB	0.9566	0.9569	0.9566	0.9566
Linear SVM	0.9712	0.9709	0.9712	0.9705
Feedforward Neural Network (FFNN)	0.9725	0.9723	0.9725	0.9719

🟢 Best Performing Model: FFNN (Feedforward Neural Network)

📈 Visualizations

Complaint distribution by product

Word count distributions

Word clouds for each category

Top 20 bi/trigrams

Model-wise confusion matrices

Performance comparison bar chart

🧪 Key Learnings

TF-IDF features + simple classifiers can achieve high accuracy.

FFNN slightly outperforms traditional ML models.

Proper preprocessing and balancing improve model robustness.

Consumer complaint data has heavy textual redundancy and sensitive info (e.g., masked names like “XXXX”).

🚀 Future Improvements

Use transformer-based models (BERT, DistilBERT) for contextual understanding.

Implement SMOTE or advanced resampling for balanced training.

Deploy the model as an API or web app using FastAPI/Streamlit.

Add topic modeling for unsupervised complaint discovery.

├── complaints.csv.zip               # Raw dataset (downloaded)
├── complaints.csv                   # Extracted data file
├── text_classification.ipynb        # Main notebook
├── requirements.txt                 # Dependencies
├── README.md                        # Project documentation
└── models/
    ├── logistic_regression.pkl
    ├── naive_bayes.pkl
    ├── svm.pkl
    └── ffnn_model.pt


🧑‍💻 Author

Anandakrishnan K V
📧 [your-email@example.com
]
💼 Data Science & AI Enthusiast
