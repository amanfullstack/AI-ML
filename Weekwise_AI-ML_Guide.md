# Weekwise AI/ML Guide — Lectures & Labs (Beginner Friendly)

This guide summarizes the weekwise lecture topics and corresponding lab notebooks in the workspace. Each section explains core concepts in plain language, shows minimal example code, and links to the provided notebooks for hands-on practice.

---

## How to use this guide

- Read a week's short explanation to get the idea.
- Open the linked notebook to run and explore the examples.
- Re-run code cells, tweak values, and observe outcomes — that's the fastest way to learn.

---

## Week 1 — Train/Test split, Cross-validation

Topics covered:
- Train/Test Split
- K-Fold Cross-Validation
- Leave-One-Out CV

Why these matter:
- We split data to evaluate how well models generalize to new data.
- Cross-validation uses multiple splits to get a more reliable estimate.

Key concept: Train/Test Split

Simple idea: keep part of your data to train the model and a separate part to test it.

Minimal code (scikit-learn):

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

What to try in the notebook:
- Open the lab: [Programs/Week 1 (end_time - 17_May_6PM)/May 09/Lab/U1W1_01_Train_Test_Split.ipynb](Programs/Week%201%20%28end_time%20-%2017_May_6PM%29/May%2009/Lab/U1W1_01_Train_Test_Split.ipynb)
- See the demo lecture: [Programs/Week 1 (end_time - 17_May_6PM)/May 09/Lecture/Demo_Train_test_split_Penguins.ipynb](Programs/Week%201%20%28end_time%20-%2017_May_6PM%29/May%2009/Lecture/Demo_Train_test_split_Penguins.ipynb)

K-Fold Cross-Validation

- Splits the data into K parts; trains on K-1 parts and validates on the remaining part, repeating K times.
- Gives a more stable performance estimate than a single train/test split.

Try the lab: [Programs/Week 1 (end_time - 17_May_6PM)/May 10/Lab/U1W1_02_KFold_and_Leaveoneout.ipynb](Programs/Week%201%20%28end_time%20-%2017_May_6PM%29/May%2010/Lab/U1W1_02_KFold_and_Leaveoneout.ipynb)
Demo: [Programs/Week 1 (end_time - 17_May_6PM)/May 10/Lecture/Demo_KFold.ipynb](Programs/Week%201%20%28end_time%20-%2017_May_6PM%29/May%2010/Lecture/Demo_KFold.ipynb)

Quick code:

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

kf = KFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=kf)
print('CV scores:', scores)
```

Leave-One-Out is K-Fold with K = number of samples (useful for tiny datasets).

---

## Week 2 — Preprocessing, NumPy, Pandas, Linear Regression, KNN

Topics covered:
- Data cleaning & preprocessing (scaling, encoding, missing values)
- NumPy basics for arrays and vectorized operations
- Pandas for tabular data
- Linear regression
- K-Nearest Neighbors (KNN)

Why these matter:
- Good preprocessing produces better models.
- NumPy and Pandas are the foundational libraries for data manipulation.

Key preprocessing steps (simple explanations):
- Scaling: bring numeric features to similar ranges (e.g., StandardScaler) so some algorithms behave correctly.
- Encoding: convert categorical values to numbers (one-hot, label encoding).
- Missing values: remove or impute (fill with mean/median or a model).

Open the Week 2 notebooks:
- Titanic preprocessing: [Programs/Week 2 (end_time - 24_May_6PM)/May 16/Lab/U1W2_03_Titanic_Preprocessing_sklearn.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2016/Lab/U1W2_03_Titanic_Preprocessing_sklearn.ipynb)
- NumPy exercises: [Programs/Week 2 (end_time - 24_May_6PM)/May 16/Lab/U1W2_04_Numpy_Exercise.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2016/Lab/U1W2_04_Numpy_Exercise.ipynb)
- Pandas intro: [Programs/Week 2 (end_time - 24_May_6PM)/May 16/Lab/U1W2_05_Introduction_to_Pandas.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2016/Lab/U1W2_05_Introduction_to_Pandas.ipynb)

Linear regression — plain language:
- Fits a straight line (or hyperplane) that predicts a numeric target from features.
- The model learns coefficients that multiply features; prediction is a weighted sum.

Minimal code example:

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

KNN — plain language:
- Predicts a sample's label by looking at the labels of its nearest neighbors in feature space.
- Simple, non-parametric; performance depends on distance scaling and k choice.

See Week 2 labs and demos:
- Linear regression labs: [Programs/Week 2 (end_time - 24_May_6PM)/May 17/Lab/U1W2_06_Linear_Regression_A.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2017/Lab/U1W2_06_Linear_Regression_A.ipynb)
  (also B & C variations in same folder)
- KNN labs: [Programs/Week 2 (end_time - 24_May_6PM)/May 17/Lab/U1W2_07_KNN_Iris_data_A.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2017/Lab/U1W2_07_KNN_Iris_data_A.ipynb)

Useful tips:
- Always split data before preprocessing that uses the target (impute/scale on training and apply same transform to test).
- Visualize relationships (scatter plots) to spot outliers and trends.

---

## Week 3 — Logistic Regression, Decision Trees, KNN/DT on different datasets, Visualization

Topics covered:
- Logistic regression (classification)
- Decision Trees and entropy/gini
- KNN, Decision Tree visualization, overfitting

Logistic Regression — plain language:
- Used for binary classification (yes/no). Instead of predicting a number directly, it predicts probabilities using the logistic (sigmoid) function.
- Model learns weights like linear regression but maps outputs through a sigmoid to produce values between 0 and 1.

Minimal code:

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)[:, 1]
preds = clf.predict(X_test)
```

Decision Trees — plain language:
- Splits data using feature thresholds to create a tree of decisions.
- Uses measures like entropy or Gini impurity to decide the best splits.
- Trees are easy to visualize and interpret but can overfit if grown too deep.

See Week 3 notebooks:
- Logistic labs: [Programs/Week 3 (end_time - 31_Dec_6PM)/May 23/Lab/U1W3_08_Logistic_Regression_A.ipynb](Programs/Week%203%20%28end_time%20-%2031_Dec_6PM%29/May%2023/Lab/U1W3_08_Logistic_Regression_A.ipynb)
- Decision tree labs and visualization: [Programs/Week 3 (end_time - 31_Dec_6PM)/May 24/Lab/U1W3_11_DT_Iris_data_Visualization_A.ipynb](Programs/Week%203%20%28end_time%20-%2031_Dec_6PM%29/May%2024/Lab/U1W3_11_DT_Iris_data_Visualization_A.ipynb)
- KNN/DT/Zoo and fruits demos: [Programs/Week 3 (end_time - 31_Dec_6PM)/May 23/Lab/U1W3_09_LC_Fruits_data_A.ipynb](Programs/Week%203%20%28end_time%20-%2031_Dec_6PM%29/May%2023/Lab/U1W3_09_LC_Fruits_data_A.ipynb)

Visualization tips:
- Plot decision boundaries for 2D features to see how classifiers separate classes.
- Use confusion matrices to inspect errors (which class is confused with which).

---

## Practical next steps for learners

1. Open each linked notebook and run cells top-to-bottom.
2. For each lab, change small parts of code (e.g., `test_size`, `n_neighbors`, `max_depth`) and observe the effect.
3. Try to explain results in your own words — teaching is the best test of understanding.

---

## Quick reference to notebooks (workspace links)

- Week 1 train/test split lab: [Programs/Week 1 (end_time - 17_May_6PM)/May 09/Lab/U1W1_01_Train_Test_Split.ipynb](Programs/Week%201%20%28end_time%20-%2017_May_6PM%29/May%2009/Lab/U1W1_01_Train_Test_Split.ipynb)
- Week 1 KFold lab: [Programs/Week 1 (end_time - 17_May_6PM)/May 10/Lab/U1W1_02_KFold_and_Leaveoneout.ipynb](Programs/Week%201%20%28end_time%20-%2017_May_6PM%29/May%2010/Lab/U1W1_02_KFold_and_Leaveoneout.ipynb)
- Week 2 Titanic preprocessing: [Programs/Week 2 (end_time - 24_May_6PM)/May 16/Lab/U1W2_03_Titanic_Preprocessing_sklearn.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2016/Lab/U1W2_03_Titanic_Preprocessing_sklearn.ipynb)
- Week 2 NumPy: [Programs/Week 2 (end_time - 24_May_6PM)/May 16/Lab/U1W2_04_Numpy_Exercise.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2016/Lab/U1W2_04_Numpy_Exercise.ipynb)
- Week 2 Pandas: [Programs/Week 2 (end_time - 24_May_6PM)/May 16/Lab/U1W2_05_Introduction_to_Pandas.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2016/Lab/U1W2_05_Introduction_to_Pandas.ipynb)
- Week 2 Linear Regression & KNN labs: [Programs/Week 2 (end_time - 24_May_6PM)/May 17/Lab/](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2017/Lab/)
- Week 3 Logistic & DT labs: [Programs/Week 3 (end_time - 31_Dec_6PM)/May 23/Lab/](Programs/Week%203%20%28end_time%20-%2031_Dec_6PM%29/May%2023/Lab/)

---

If you'd like, I can:
- expand each section into a step-by-step walkthrough of the notebooks,
- extract runnable Python scripts from the notebooks, or
- add short quizzes/exercises per topic.

Tell me which follow-up you prefer.
