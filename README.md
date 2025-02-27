# Text-Pair-Classification

## **1. Project Overview**
This project focuses on **Natural Language Inference (NLI)**, a crucial task in Natural Language Processing (NLP), where models classify the relationship between sentence pairs into three categories:  
- **Contradiction:** The hypothesis contradicts the premise.  
- **Neutral:** The hypothesis is unrelated to the premise.  
- **Entailment:** The hypothesis is logically inferred from the premise.  

We compare multiple machine learning and deep learning models to evaluate their effectiveness in solving this problem.  

---

## **2. Dataset Description**
- **File:** `train.csv`  
- **Total Rows:** 12,120  
- **Columns:**  
  - `id`: Unique identifier for each instance  
  - `premise`: First sentence  
  - `hypothesis`: Second sentence related to the premise  
  - `lang_abv`: Language abbreviation  
  - `language`: Full language name  
  - `label`: Target class (**0 = Contradiction, 1 = Neutral, 2 = Entailment**)  

The dataset is **multilingual**, meaning that it contains data from different languages, which makes the task more complex.  

---

## **3. Model Implementation Details**
### **3.1 Preprocessing Steps**
- Handled missing values (if any).  
- Converted text into numerical features using **TF-IDF**.  
- Split dataset into **training and testing sets**.  
- Applied **label encoding** to transform text labels into numerical values.  

### **3.2 Models Implemented**
| Model               | Description |
|--------------------|--------------------------------------|
| **Random Forest**   | Ensemble learning method using multiple decision trees. |
| **Decision Tree**   | A single-tree classifier with max depth tuning. |
| **XGBoost**         | Gradient boosting-based decision tree model. |
| **Artificial Neural Network (ANN)** | Multi-layer perceptron (MLP) for deep learning classification. |
| **LSTM Model**      | A recurrent neural network (RNN) specialized for sequential data. |
| **BERT Fine-Tuning (Skipped)** | Transformer-based model (Commented out due to execution time constraints). |

**Note:** BERT fine-tuning was **not executed** due to high resource consumption but is included in the code as comments.

---

## **4. Steps to Run the Code**
### **4.1 Prerequisites**
Ensure you have the following libraries installed before running the code:  
```bash
pip install numpy pandas matplotlib seaborn sklearn xgboost tensorflow transformers




---

### **Explanation of Each Section**
1. **Project Overview** → Briefly explains what the project is about.  
2. **Dataset Description** → Provides information on dataset structure and target labels.  
3. **Model Implementation Details** → Lists preprocessing steps and models used.  
4. **Steps to Run the Code** → Guides the user on how to install dependencies and execute the notebook.  
5. **Model Evaluation Results** → Summarizes accuracy, precision, recall, and F1 scores.  
6. **Confusion Matrix** → Mentions how the confusion matrix helps in model analysis.  
7. **Future Improvements** → Suggests enhancements to boost performance.  
8. **Additional Notes** → Extra details about execution, BERT skipping, and visualization.  
9. **Authors & Contributors** → Your name and role in the project.  

---

### **Final Steps**
✅ **Copy-paste this README.md into your GitHub repository.**  
✅ **Fill in missing accuracy, precision, recall, and F1-score values from your notebook results.**  
✅ **Ensure all instructions match your execution environment.**  

Let me know if you need any modifications!