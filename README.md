# Ecommerce-Product-Categorization-Hackathon
This repository contains the code, notebooks, and documentation for my submission to the KnowledgeHut "Ecommerce Product Categorization" hackathon.
Data

The dataset used for this project includes information about various products, including their descriptions and corresponding product categories.

Methodology

Data Exploration and Preprocessing:

Exploratory Data Analysis (EDA): Conducted thorough EDA to understand data characteristics, identify missing values, and analyze the distribution of product categories.
Data Cleaning: Handled missing values, removed duplicates, and corrected inconsistencies in the data.
Text Preprocessing:
Converted text to lowercase.
Removed punctuation, special characters, and stop words.
Handled contractions (e.g., "don't" to "do not").
Performed stemming or lemmatization to reduce words to their base forms.
Feature Engineering:

TF-IDF Vectorization: Transformed text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) to capture the importance of words in each document.
Word Embeddings: (Optional) Explored word embeddings (e.g., Word2Vec, GloVe) to capture semantic relationships between words.
Model Development:

Model Selection: Trained and evaluated various machine learning models, including:
Naive Bayes: (MultinomialNB, BernoulliNB)
Support Vector Machines (SVM)
Random Forest
Logistic Regression
Deep Learning Models: (Optional) Trained and evaluated deep learning models like Recurrent Neural Networks (RNNs) or Convolutional Neural Networks (CNNs) for more complex text representations.
Class Imbalance Handling:

Addressed class imbalance using techniques like:
Oversampling: SMOTE (Synthetic Minority Over-sampling Technique)
Undersampling: Random UnderSampling
Class Weights: Assigning higher weights to minority classes.
Model Evaluation and Selection:

Evaluated models using metrics such as:
Accuracy
Precision
Recall
F1-score
Confusion matrix
Performed hyperparameter tuning using techniques like grid search or random search to optimize model performance.
Selected the best-performing model based on the evaluation metrics.


Results

Accuracy on Test Data: 0.760852407261247
                             precision    recall  f1-score   support

                Automotive        0.70      0.92      0.80        75
                 Baby Care        0.15      0.16      0.16       259
     Bags, Wallets & Belts        0.98      0.81      0.89       107
                  Clothing        0.95      0.73      0.83       882
                 Computers        0.56      0.32      0.41        47
                  Footwear        0.92      0.99      0.95       144
Home Decor & Festive Needs        0.46      0.90      0.61       215
                 Jewellery        0.97      0.95      0.96       313
          Kitchen & Dining        0.66      0.78      0.72        37
     Mobiles & Accessories        0.99      0.93      0.96       331
         Pens & Stationery        0.75      0.86      0.80        49
          Tools & Hardware        0.33      0.08      0.13        12
    Toys & School Supplies        0.83      0.85      0.84        59
                   Watches        0.60      0.75      0.67         4

                   accuracy                           0.76      2534
                  macro avg       0.70      0.72      0.69      2534
               weighted avg       0.81      0.76      0.77      2534

