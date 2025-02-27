# Text pair classification 

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

### 4.2 Running the Notebook

Follow these steps to execute the notebook:

1. Open **Google Colab** or **Jupyter Notebook**.
2. Upload the dataset file (**train.csv**).
3. Load and execute the **Untitled3.ipynb** notebook step by step.
4. Train and evaluate different models on the dataset.
5. Visualize results using confusion matrices and performance metrics.

---

## 5. Model Evaluation Results

The models were evaluated using the following metrics:

- **Accuracy**: Measures overall correctness.
- **Precision**: Measures how many predicted positive labels were actually positive.
- **Recall**: Measures how many actual positive labels were correctly identified.
- **F1 Score**: Harmonic mean of precision and recall, balancing both measures.

---

## 6. Confusion Matrix

A **confusion matrix** helps visualize model performance by showing true and false predictions.

- A heatmap of the confusion matrix is included in the notebook for better understanding.
- The **ideal model** should have high values along the diagonal, indicating correct predictions.
- **Models with significant misclassification rates** may require further improvements.

---

## 7. Future Improvements

Several enhancements can be made to improve the model:

- **Hyperparameter tuning**: Optimize model parameters for better performance.
- **Data augmentation**: Increase training data variety to improve generalization.
- **Advanced architectures**: Experiment with transformer-based models like **BERT**.
- **Fine-tuning**: Use pre-trained models for better contextual understanding.

---

## 8. Challenges Faced

During model training and evaluation, some key challenges were encountered:

- **Computational limitations**: Training deep learning models required significant resources.
- **Execution time**: Transformer-based models were skipped due to high execution time.
- **Multilingual complexity**: Handling diverse languages made feature extraction challenging.

---

## 9. Conclusion

This project explored **Text-Pair Classification** using various machine learning and deep learning models. Key takeaways:

- **Classical ML models (Random Forest, XGBoost)** performed well with TF-IDF features.
- **Deep learning models (ANN, LSTM)** provided promising results but required extensive tuning.
- **BERT fine-tuning was skipped** due to resource constraints but remains a potential future improvement.

The results highlight the effectiveness of machine learning techniques in **Natural Language Inference (NLI)**, with room for further enhancements.