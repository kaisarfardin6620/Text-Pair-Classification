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




4.2 Running the Notebook

1. Open Google Colab or Jupyter Notebook.


2. Upload the dataset (train.csv).


3. Load and execute the Untitled3.ipynb notebook step by step.


4. Train and evaluate different models on the dataset.


5. Visualize results using confusion matrices and performance metrics.




---

### **5. Model Evaluation Results

The models were evaluated using the following metrics:

Accuracy: Measures overall correctness.

Precision: Measures how many predicted positive labels were actually correct.

Recall: Measures how many actual positive labels were correctly identified.

F1 Score: Harmonic mean of precision and recall, balancing both metrics.




---

### **6. Confusion Matrix

A confusion matrix helps visualize model performance by showing the number of correct and incorrect predictions for each class.

A heatmap of the confusion matrix is included in the notebook for a better understanding of misclassified instances.

The ideal model should have high values along the diagonal, indicating correct classifications.

Models with significant misclassification rates may require further optimization.



---

### **7. Future Improvements

To further enhance model performance, consider the following improvements:

Hyperparameter Tuning: Use GridSearchCV or Bayesian Optimization to find optimal model parameters.

BERT Fine-Tuning: Re-enable and optimize BERT for improved language understanding.

Data Augmentation: Use paraphrasing techniques or back-translation to increase training data.

Feature Engineering: Experiment with Word2Vec, FastText, and Sentence-BERT embeddings.

Hybrid Models: Combine traditional ML models with deep learning architectures to improve performance.



---

### **8. Additional Notes

The code was developed and tested in Google Colab.

BERT-based training was commented out due to high computational requirements.

The project includes visualizations (confusion matrices, accuracy plots, and feature importance charts) for better interpretability.

If using local execution, ensure you have sufficient GPU resources for deep learning models.



---

### **9. Authors & Contributors

Abdullah Kaisar Fardin

Contributions: Data Preprocessing, Model Training, Evaluation, Report Writing




