# SMS-Spam-Prediction

## 1. Overview

This project implements a machine learning model to classify email messages as either "ham" (not spam) or "spam". It utilizes Natural Language Processing (NLP) techniques to process the text data and a Logistic Regression model for classification. The entire workflow, from data loading and preprocessing to model training and evaluation, is contained within the `Spam_Mail_Prediction.ipynb` Jupyter Notebook.

## 2. Dataset

The project uses the `mail_data.csv` file, which contains two columns:

*   `Category`: Label indicating if the message is 'ham' or 'spam'.
*   `Message`: The raw text content of the email/SMS message.

This dataset appears to be a standard SMS spam collection dataset.

## 3. Workflow & Features

The notebook follows these key steps:

1.  **Import Libraries:** Necessary libraries like `numpy`, `pandas`, and `scikit-learn` are imported.
2.  **Data Loading:** The `mail_data.csv` is loaded into a pandas DataFrame.
3.  **Data Preprocessing:**
    *   Handling null values (replacing with empty strings).
    *   Label Encoding: Converting categorical labels ('spam', 'ham') into numerical format (spam: 0, ham: 1).
    *   Splitting Data: Dividing the dataset into training (80%) and testing (20%) sets.
4.  **Feature Extraction (TF-IDF):**
    *   The text data (`Message` column) is converted into numerical feature vectors using the Term Frequency-Inverse Document Frequency (TF-IDF) technique via `TfidfVectorizer`.
    *   This process ignores common English stop words and converts text to lowercase.
5.  **Model Training:**
    *   A Logistic Regression model is trained using the TF-IDF features from the training data (`X_train_features`) and the corresponding labels (`Y_train`).
6.  **Model Evaluation:**
    *   The trained model's performance is evaluated on both the training data and the unseen test data.
    *   Accuracy is used as the evaluation metric (`accuracy_score`).
7.  **Predictive System:**
    *   A simple example demonstrates how to use the trained model and the fitted TF-IDF vectorizer to predict the category of a new, unseen input email message.

## 4. Technology Stack

*   **Language:** Python 3.x
*   **Core Libraries:**
    *   Pandas: For data manipulation and loading CSV.
    *   NumPy: For numerical operations (often used implicitly by pandas/sklearn).
    *   Scikit-learn:
        *   `model_selection.train_test_split`: For splitting data.
        *   `feature_extraction.text.TfidfVectorizer`: For TF-IDF feature extraction.
        *   `linear_model.LogisticRegression`: For the classification model.
        *   `metrics.accuracy_score`: For evaluating model performance.
*   **Environment:** Jupyter Notebook (`.ipynb`)

## 6. How to Run

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
2.  **Ensure Prerequisites:** Make sure Python and pip are installed, and install the required libraries as mentioned above.
3.  **Place Dataset:** Ensure the `mail_data.csv` file is in the same directory as the `Spam_Mail_Prediction.ipynb` notebook, or update the path in the notebook accordingly.
4.  **Launch Jupyter:**
    ```bash
    jupyter notebook
    # OR
    # jupyter lab
    ```
5.  **Open and Run:** Navigate to the `Spam_Mail_Prediction.ipynb` file in the Jupyter interface and run the cells sequentially.

## 7. Results

The Logistic Regression model achieved high accuracy on this dataset:

*   **Accuracy on Training Data:** ~96.77%
*   **Accuracy on Test Data:** ~96.68%

