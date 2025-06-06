# Cancer Diagnosis Classification with PySpark

This project focuses on classifying cancer diagnoses using two ensemble machine learning algorithms: **Random Forest (RF)** and **Gradient Boosted Trees (GBT)**. The implementation is done in **PySpark** to handle large datasets efficiently in a distributed environment.

---

## 📁 Dataset

- Input: `project3_data.csv`
- Each row represents a patient's cancer diagnostic metrics.
- The `diagnosis` column is the target (label) with binary classification (e.g., malignant or benign).
- The `id` column, if present, is dropped.

---

## 📊 Chosen Algorithms

- **Random Forest (RF):**

  - An ensemble method using bagging with multiple decision trees.
  - Reduces overfitting and works well with high-dimensional data.

- **Gradient Boosted Trees (GBT):**
  - A boosting technique where each tree corrects errors of the previous ones.
  - Often achieves better accuracy due to its sequential learning approach.

---

## ⚙️ Training Process

- A Spark session is created with optimized settings for memory and performance.
- The dataset is read from a CSV file.
- The id column is dropped (since it's not useful for modeling).
- All feature columns are cast to numeric types.
- Columns with all missing values are removed.
- Missing values in the remaining columns are imputed using the mean
- The diagnosis column (categorical) is converted into numeric labels.
- All feature columns are combined into a single feature vector.
- The dataset is split into 80% training and 20% testing data.
- Two classifiers are trained:
  - Random Forest with 50 trees.
  - Gradient Boosted Trees (GBT) with 50 iterations and depth of 3.
  - 3-fold cross-validation is used to evaluate each model during training.
- Both models are evaluated on the test set using multiple metrics:
  - Accuracy
  - F1 Score
  - Precision
  - Recall
- Evaluation scores are printed for comparison.
- The Spark session is stopped to release resources.

## 📈 Evaluation Metrics

Models were evaluated using the following metrics:

- **Accuracy**: Overall correctness of the model.
- **F1 Score**: Harmonic mean of precision and recall.
- **Precision**: Correctly predicted positives out of all predicted positives.
- **Recall**: Correctly predicted positives out of all actual positives.

---

## ✅ Testing Results

| Metric        | Random Forest | Gradient Boosting |
| ------------- | ------------- | ----------------- |
| **Accuracy**  | 0.9535        | **0.9767**        |
| **F1 Score**  | 0.9650        | **0.9766**        |
| **Precision** | 0.9652        | **0.9776**        |
| **Recall**    | 0.9651        | **0.9767**        |

**Insight**: Gradient Boosting outperformed Random Forest across all metrics, likely due to its ability to iteratively correct prior errors.

---

## ⚠️ Limitations

- Limited hyperparameter tuning due to time and performance constraints.
- Used default settings for many parameters to ensure quicker training time.
- Spark cluster resources were modest (`2g` executor memory and 8 partitions), which could impact scalability.

---

## 🚀 Suggestions for Future Improvements

- Perform grid search over more hyperparameters (e.g., `maxDepth`, `minInstancesPerNode`).
- Use more folds in cross-validation for better generalization.
- Deploy the model as a real-time prediction service.

---

## 📌 Requirements

- Apache Spark
- PySpark

Install via pip:

```bash
pip install pyspark
```
