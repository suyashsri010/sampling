# Sampling Techniques for Imbalanced Credit Card Fraud Dataset

## Objective
The objective of this assignment is to understand the importance of sampling techniques in handling highly imbalanced datasets and to analyze how different sampling strategies affect the performance of various machine learning models.  
This study focuses on a real-world credit card fraud dataset, where fraudulent transactions are extremely rare compared to normal transactions.

---

## Dataset Description
The dataset used in this assignment is a credit card transaction dataset containing **772 records** and **31 features**.  
The target variable is **Class**, where:
- `Class = 0` represents a normal (non-fraud) transaction
- `Class = 1` represents a fraudulent transaction

### Class Distribution (Original Dataset)
- Normal transactions (Class 0): 763
- Fraudulent transactions (Class 1): 9

This severe imbalance makes it difficult for machine learning models to correctly learn patterns related to fraud detection, thereby motivating the need for sampling techniques.

Dataset Source:  
https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv

---

## Methodology

The overall workflow followed in this assignment is outlined below:

### 1. Data Preprocessing
- The dataset was loaded using Pandas.
- The target column (`Class`) was separated from the feature set.
- Feature scaling was applied using **StandardScaler** to normalize all numerical features.

### 2. Handling Class Imbalance
To balance the dataset, five different sampling techniques were applied. Each technique generated a new balanced dataset.

### 3. Sampling Techniques Used
The following five sampling techniques were implemented:

1. **Random Undersampling**  
   Reduces the number of majority class samples to match the minority class.

2. **Random Oversampling**  
   Randomly duplicates minority class samples to balance the dataset.

3. **SMOTE (Synthetic Minority Over-sampling Technique)**  
   Generates synthetic samples for the minority class based on nearest neighbors.

4. **NearMiss Undersampling**  
   Selects majority class samples that are closest to minority class samples.

5. **SMOTE + Tomek Links**  
   A hybrid approach combining oversampling and cleaning overlapping samples.

---

## Machine Learning Models Used

Each balanced dataset was trained and evaluated using the following five machine learning models:

- **M1:** Logistic Regression  
- **M2:** Decision Tree Classifier  
- **M3:** Random Forest Classifier  
- **M4:** K-Nearest Neighbors (KNN)  
- **M5:** Support Vector Machine (SVM)

---

## Experimental Setup
- Each sampled dataset was split into training and testing sets using a 70:30 ratio.
- The same models and parameters were used across all sampling techniques to ensure fair comparison.
- Model performance was evaluated using **accuracy**.

---

## Results

The final accuracy results obtained for each combination of sampling technique and machine learning model are shown below.


| Model | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 (SMOTETomek) |
|------|-----------|-----------|-----------|-----------|-------------------------|
| M1_LogisticRegression | 33.33 | 91.70 | 91.70 | 100.00 | 91.70 |
| M2_DecisionTree | 66.67 | 99.13 | 97.38 | 83.33 | 97.82 |
| M3_RandomForest | 33.33 | 99.78 | 99.34 | 66.67 | 99.34 |
| M4_KNN | 33.33 | 96.51 | 94.54 | 83.33 | 94.54 |
| M5_SVM | 16.67 | 96.51 | 96.94 | 16.67 | 96.94 |



---

## Discussion

- Oversampling techniques such as **SMOTE** and **SMOTE + Tomek Links** generally improved performance for distance-based models like KNN and SVM.
- **Random Undersampling** sometimes resulted in high accuracy for tree-based models but may cause information loss.
- **NearMiss** showed comparatively lower performance due to aggressive removal of majority class samples.
- Tree-based models like **Decision Trees** and **Random Forests** were more robust to imbalance compared to linear models.
- The choice of sampling technique significantly impacts model performance, and no single method performs best for all models.

---

## Conclusion

This assignment demonstrates that handling class imbalance is critical for building effective machine learning models.  
Different sampling techniques influence different models in unique ways, and selecting an appropriate combination of sampling strategy and model is essential for achieving optimal performance in real-world fraud detection systems.

---

## Files Included
- `Creditcard_data.csv` – Original dataset
- `sampling_models.py` – Python implementation
- `sampling_model_accuracy_results.csv` – Final results table
- `README.md` – Project documentation

---

## Author
**Aishita Kumar**  
**Roll Number:** 102303744
