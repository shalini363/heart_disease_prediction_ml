# **Heart Disease Prediction Using Machine Learning**

This project explores **heart disease prediction** using various **machine learning models**, including **Support Vector Machine (SVM), Random Forest (RF), K-Nearest Neighbors (KNN), and Long Short-Term Memory (LSTM) networks**. The dataset consists of **medical records**, including factors like **age, cholesterol, blood pressure, chest pain type, and exercise-induced angina**, to classify individuals as having or not having heart disease.

---

## **Features**

### **1. Data Preprocessing**
- **Loaded and cleaned the dataset** from a CSV file.
- **Handled missing values** and performed exploratory data analysis (EDA).
- **One-hot encoding** applied to categorical features like **chest pain type, sex, and ECG results**.
- **Feature scaling** applied using standardization to normalize numerical features.

### **2. Machine Learning Models**
- **Support Vector Machine (SVM):** Tuned with hyperparameter optimization (Grid Search).
- **Random Forest (RF):** Used ensemble learning with multiple decision trees.
- **K-Nearest Neighbors (KNN):** Compared with other models for nearest-neighbor classification.
- **LSTM (Long Short-Term Memory):** Applied deep learning for sequential data analysis.

### **3. Performance Evaluation**
- **Cross-validation:** Used **10-fold Stratified Cross-Validation** for model evaluation.
- **Metrics Used:**
  - **Accuracy** (overall correctness)
  - **Precision, Recall, F1-score** (for classification performance)
  - **ROC-AUC Score** (to evaluate model discrimination capability)
  - **Confusion Matrix** (to analyze true/false positives and negatives)

---

## **Getting Started**

### **Prerequisites**
- **Python 3.8 or higher**
- **Required Libraries:** Install dependencies using: pip install -r requirements.txt

### **Dataset**
- The dataset contains **918 medical records** with **12 features** related to heart disease.
- The data includes:
  - **Age, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, and Maximum Heart Rate**
  - **Categorical Features:** Sex, Chest Pain Type, ECG results, Exercise Angina, and ST-Slope.
  - **Target Variable:** **HeartDisease** (1: Disease Present, 0: No Disease)

---

## **How to Run**

1. **Clone this repository:** git clone https://github.com/shalini363/heart_disease_prediction_ml.git
   
2. **Navigate to the project directory:** cd heart-disease-prediction
   

3. **Launch Jupyter Notebook:** jupyter notebook heart_disease_prediction.ipynb

4. **Run the notebook cells to:**
   - Preprocess the data.
   - Train multiple machine learning models.
   - Evaluate model performance using accuracy, precision, and recall.

---

## **Results**

| Model       | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------------|----------|-----------|--------|----------|---------|
| **SVM**    | 95%      | 91%       | 96%    | 94%      | 0.97    |
| **Random Forest** | 94% | 93% | 95% | 94% | 0.98 |
| **KNN**    | 89%      | 88%       | 90%    | 89%      | 0.88    |
| **LSTM**   | 93%      | 91%       | 94%    | 93%      | 0.96    |

### **Observations**
- **Random Forest performed the best in terms of overall prediction accuracy and stability.**
- **SVM also showed strong performance with high recall, making it suitable for medical applications.**
- **LSTM demonstrated potential but requires more tuning and a larger dataset.**
- **KNN performed slightly worse compared to other models due to its sensitivity to dataset size.**

---

## **Key Insights**
- **Medical data preprocessing is crucial for improving model performance.**
- **Random Forest and SVM models were highly effective in heart disease classification.**
- **Deep learning (LSTM) performed well but required careful hyperparameter tuning.**
- **The dataset was relatively small, which may impact deep learning model performance.**

---

## **Future Work**
- **Test the models on a larger dataset** to validate performance.
- **Implement Explainable AI (XAI) techniques** to understand model decisions.
- **Apply additional feature engineering techniques** for better prediction accuracy.

---

## **Acknowledgments**
- **Dataset Source:** Publicly available medical datasets.
- **Reference Libraries:** Scikit-learn, TensorFlow, Pandas, Matplotlib.

---
