# Text-Pair-Classification
# Multilingual Natural Language Inference (NLI) Model

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