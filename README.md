
# 📧 SMS Spam Detection using Naive Bayes

This project is a simple yet powerful machine learning model that classifies SMS messages as **Spam** or **Ham (Not Spam)** using the **Multinomial Naive Bayes** algorithm.

## 📌 Project Overview

- ✅ Dataset: SMS Spam Collection Dataset  
- ✅ Model: Multinomial Naive Bayes  
- ✅ Features: TF-IDF Vectorizer with bigrams  
- ✅ Accuracy: ~96%  
- ✅ Libraries: pandas, scikit-learn

## 🗂️ Dataset

- Source: [Kaggle - SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- File: `spam.csv`
- Columns Used:
  - `v1`: Label (`ham` or `spam`)
  - `v2`: SMS message content


## ⚙️ How to Run

python spam_classifier.py


## 🔍 Features

* Preprocessing: Lowercasing, punctuation removal
* TF-IDF with:

  * `stop_words='english'`
  * `ngram_range=(1, 2)`
  * `min_df=5`, `max_df=0.95`
* Output:

  * Accuracy and classification report
  * Test message prediction with spam probability

## 📊 Example Output

```
====== Naive Bayes Classifier ======
Accuracy: 0.9623

Classification Report:
               precision    recall  f1-score   support
           0       0.96      1.00      0.98       965
           1       1.00      0.72      0.84       150

Spam probability: 0.78
Prediction for message: Win a free iPhone! Click here
Predicted as: Spam
```


## 📦 Dependencies

* `pandas`
* `scikit-learn`


## 📁 Folder Structure

```
sms-spam-classifier/
├── spam_classifier.py
├── spam.csv
├── requirements.txt
└── README.md
```

---

## 💡 Future Improvements

* Add Streamlit UI for real-time classification
* Try deep learning models (LSTM/BERT)
* Evaluate on multilingual SMS datasets

---

## 👨‍💻 Author

**Your Name**
[GitHub](https://github.com/your-username)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

```

---

Would you like a Streamlit version of this project to make it interactive on the web?
```
