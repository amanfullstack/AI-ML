# Weekwise AI/ML Notes with Program Examples

This file explains the course week by week in very easy language. The focus is on the notebook examples in the `Programs` folder, and when a `C` version exists, that is used as the main reference because it usually contains the full worked solution.

---

## How to read this file

- First read the simple explanation.
- Then open the matching notebook and run it.
- Finally, change one small value in the code and see what happens.

This is the fastest way to learn AI/ML.

---

## Week 1: Train/Test Split, K-Fold, Leave-One-Out

### What this week is about

Before making any model, we need to answer one question: how do we know the model is really good and not just memorizing the data?

That is why we split data into training and testing parts.

- Training data is used to teach the model.
- Testing data is used to check how well the model works on new data.

### 1) Train/Test Split

Simple idea:
- Imagine giving 80% of your notes to a student to study and keeping 20% hidden for the exam.
- The student studies the 80% and then is tested on the hidden 20%.

Why this is important:
- If the model does well on training data but poorly on test data, it is overfitting.
- Overfitting means the model learned the answers too well instead of learning the pattern.

Program example:
- Lab: [U1W1_01_Train_Test_Split.ipynb](Programs/Week%201%20%28end_time%20-%2017_May_6PM%29/May%2009/Lab/U1W1_01_Train_Test_Split.ipynb)
- Lecture demo: [Demo_Train_test_split_Penguins.ipynb](Programs/Week%201%20%28end_time%20-%2017_May_6PM%29/May%2009/Lecture/Demo_Train_test_split_Penguins.ipynb)

Easy code idea:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

What each part means:
- `X` means input features.
- `y` means the answer or target.
- `test_size=0.2` means 20% test data.
- `random_state=42` makes the split repeatable.

### 2) K-Fold Cross Validation

Simple idea:
- Instead of using only one split, divide the data into K parts.
- Train on K-1 parts and test on the remaining part.
- Repeat this K times.

Why this is useful:
- It gives a more reliable score than a single split.
- It helps when data is small.

Program example:
- Lab: [U1W1_02_KFold_and_Leaveoneout.ipynb](Programs/Week%201%20%28end_time%20-%2017_May_6PM%29/May%2010/Lab/U1W1_02_KFold_and_Leaveoneout.ipynb)
- Lecture demo: [Demo_KFold.ipynb](Programs/Week%201%20%28end_time%20-%2017_May_6PM%29/May%2010/Lecture/Demo_KFold.ipynb)

Easy code idea:

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

kf = KFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=kf)
print(scores)
```

What this means:
- The model is trained and tested 5 times.
- Each time a different fold is used as the test part.

### 3) Leave-One-Out Cross Validation

Simple idea:
- This is like K-Fold where K equals the number of rows.
- Each time, one row is tested and all the others are used for training.

When to use:
- Very small datasets.
- When you want to use almost all the data for training.

### What to remember from Week 1

- A model should be tested on data it has not seen before.
- Train/test split is the first basic method.
- K-Fold is better when you want a more stable result.
- Leave-One-Out is useful for very small datasets.

---

## Week 2: NumPy, Pandas, Preprocessing, Linear Regression, KNN

### What this week is about

This week builds the base needed for machine learning.

- NumPy helps with fast number handling.
- Pandas helps with table data.
- Preprocessing cleans and prepares data.
- Linear Regression predicts numbers.
- KNN predicts by looking at nearby points.

### 1) NumPy

Why it matters:
- Machine learning uses numbers everywhere.
- NumPy stores numbers in arrays and lets us work fast.

Easy idea:

```python
import numpy as np

a = np.array([1, 2, 3])
b = a * 2
```

Meaning:
- `np.array` makes a numeric array.
- `a * 2` multiplies every value.

Program example:
- Lab: [U1W2_04_Numpy_Exercise.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2016/Lab/U1W2_04_Numpy_Exercise.ipynb)

### 2) Pandas

Why it matters:
- Real data usually comes in tables.
- Pandas helps read, inspect, filter, and clean tables.

Easy idea:

```python
import pandas as pd

df = pd.read_csv("data.csv")
print(df.head())
```

Meaning:
- `read_csv` loads a table from a file.
- `head()` shows the first few rows.

Program example:
- Lab: [U1W2_05_Introduction_to_Pandas.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2016/Lab/U1W2_05_Introduction_to_Pandas.ipynb)

### 3) Preprocessing

This is one of the most important ideas in the course.

Preprocessing means preparing raw data so the model can learn better.

Common preprocessing steps:
- Missing values: fill or remove empty cells.
- Encoding: convert text values into numbers.
- Scaling: put values on a similar scale.

Why scaling is useful:
- If one feature is very large and another is very small, some models behave badly.
- KNN especially needs scaling because it uses distance.

Program example:
- Lab: [U1W2_03_Titanic_Preprocessing_sklearn.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2016/Lab/U1W2_03_Titanic_Preprocessing_sklearn.ipynb)
- Lecture demo: [Demo_Preprocessing.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2016/Lecture/Demo_Preprocessing.ipynb)

Easy code idea:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Meaning:
- `fit_transform` learns the scaling from training data and applies it.
- `transform` applies the same scaling to test data.
- Always learn from training data only.

### 4) Linear Regression

What it does:
- Predicts a number.
- Example: house price, salary, temperature, sales.

Simple idea:
- The model tries to draw the best straight line through the data.
- It learns the line that gives the smallest error.

Program example:
- Lab C: [U1W2_06_Linear_Regression_C.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2017/Lab/U1W2_06_Linear_Regression_C.ipynb)
- Lecture demos: [Demo_Linear_Regression_Diabetes.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2017/Lecture/Demo_Linear_Regression_Diabetes.ipynb), [Demo_Linear_Regression_Pendulum.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2017/Lecture/Demo_Linear_Regression_Pendulum.ipynb)

Easy code idea:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

What to understand:
- `fit` means learn from data.
- `predict` means give output for new input.

### 5) K-Nearest Neighbors

What it does:
- Predicts based on nearby examples.
- For classification, it looks at the most common label among neighbors.

Simple idea:
- If a new point is close to many red points, it is likely red.

Program example:
- Lab C: [U1W2_07_KNN_Iris_data_C.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2017/Lab/U1W2_07_KNN_Iris_data_C.ipynb)
- Lecture demos: [Demo_KNN_Advertising_data.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2017/Lecture/Demo_KNN_Advertising_data.ipynb), [Demo_KNN_Scaling.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2017/Lecture/Demo_KNN_Scaling.ipynb)

Easy code idea:

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
```

What to remember:
- Small `n_neighbors` can make the model noisy.
- Large `n_neighbors` can make it too smooth.
- KNN works better when features are scaled.

### What to remember from Week 2

- NumPy is for fast numeric arrays.
- Pandas is for table data.
- Preprocessing is essential before modeling.
- Linear Regression predicts numbers.
- KNN predicts using nearby points.

---

## Week 3: Logistic Regression, Decision Tree, Fruits, Zoo, Visualization

### What this week is about

This week moves from number prediction to classification.

- Logistic Regression predicts class labels.
- Decision Trees split data step by step.
- Visualization helps you see how the model is working.

### 1) Logistic Regression

What it does:
- Used for classification, often yes/no problems.
- It gives probability and then chooses a class.

Simple idea:
- Instead of drawing a straight line for numbers, it draws a boundary between classes.

Program example:
- Lab C: [U1W3_08_Logistic_Regression_C.ipynb](Programs/Week%203%20%28end_time%20-%2031_Dec_6PM%29/May%2023/Lab/U1W3_08_Logistic_Regression_C.ipynb)
- Lecture demo: [Demo_Logistic_Regression_Diabetes.ipynb](Programs/Week%203%20%28end_time%20-%2031_Dec_6PM%29/May%2023/Lecture/Demo_Logistic_Regression_Diabetes.ipynb)

Easy code idea:

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

Meaning:
- `predict` gives class labels.
- `predict_proba` gives probabilities.

### 2) Fruits data and class boundaries

This notebook helps you see how classification works with a simple example.

Program example:
- Lab C: [U1W3_09_LC_Fruits_data_C.ipynb](Programs/Week%203%20%28end_time%20-%2031_Dec_6PM%29/May%2023/Lab/U1W3_09_LC_Fruits_data_C.ipynb)

What this teaches:
- How features separate different classes.
- Why a model can draw boundaries between groups.

### 3) KNN and Decision Tree on Zoo data

Zoo data is a good practice dataset because it has clear classes and many features.

Program example:
- Lab C: [U1W3_10_KNN_DT_LC_Zoo_data_C.ipynb](Programs/Week%203%20%28end_time%20-%2031_Dec_6PM%29/May%2024/Lab/U1W3_10_KNN_DT_LC_Zoo_data_C.ipynb)

What this teaches:
- KNN can classify based on nearby animals with similar properties.
- Decision Trees can split animals using yes/no questions.

Decision Tree idea:
- First ask one question.
- Then ask another question.
- Keep splitting until the class is clear.

### 4) Decision Tree visualization with Iris

This is one of the best notebooks for understanding how a tree works.

Program example:
- Lab C: [U1W3_11_DT_Iris_data_Visualization_C.ipynb](Programs/Week%203%20%28end_time%20-%2031_Dec_6PM%29/May%2024/Lab/U1W3_11_DT_Iris_data_Visualization_C.ipynb)
- Lecture demo: [Demo_DT_Overfitting.ipynb](Programs/Week%203%20%28end_time%20-%2031_Dec_6PM%29/May%2024/Lecture/Demo_DT_Overfitting.ipynb)
- Lecture demo: [Demo_DT_Fruits_data.ipynb](Programs/Week%203%20%28end_time%20-%2031_Dec_6PM%29/May%2024/Lecture/Demo_DT_Fruits_data.ipynb)

Easy code idea:

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)
```

Why `max_depth` matters:
- Small depth: simpler tree, less overfitting.
- Large depth: more complex tree, may memorize training data.

### 5) Entropy and overfitting

Entropy means uncertainty.

Simple idea:
- If a group has only one class, entropy is low.
- If a group has many mixed classes, entropy is high.

Decision Trees try to make splits that reduce entropy.

Overfitting in trees:
- If the tree keeps splitting too much, it learns the training data too exactly.
- Then it may fail on new data.

### What to remember from Week 3

- Logistic Regression is for classification.
- Decision Trees use simple questions to split data.
- Visualization makes the model easier to understand.
- Overfitting is a major risk for deep trees.

---

## Short concept map

- Train/Test Split: separate data for learning and checking.
- K-Fold: repeat the test several times.
- NumPy: fast numeric arrays.
- Pandas: table handling.
- Preprocessing: clean and prepare data.
- Linear Regression: predict numbers.
- KNN: predict by nearest neighbors.
- Logistic Regression: predict classes.
- Decision Tree: split data with questions.

---

## Best way to study these notebooks

1. Run the C notebook for each topic first.
2. Read the markdown cells before the code.
3. Change one value at a time.
4. Write down what changed in the output.
5. Try to explain the result in simple words.

---

## Notebook links used in this file

- [U1W1_01_Train_Test_Split.ipynb](Programs/Week%201%20%28end_time%20-%2017_May_6PM%29/May%2009/Lab/U1W1_01_Train_Test_Split.ipynb)
- [U1W1_02_KFold_and_Leaveoneout.ipynb](Programs/Week%201%20%28end_time%20-%2017_May_6PM%29/May%2010/Lab/U1W1_02_KFold_and_Leaveoneout.ipynb)
- [U1W2_03_Titanic_Preprocessing_sklearn.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2016/Lab/U1W2_03_Titanic_Preprocessing_sklearn.ipynb)
- [U1W2_04_Numpy_Exercise.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2016/Lab/U1W2_04_Numpy_Exercise.ipynb)
- [U1W2_05_Introduction_to_Pandas.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2016/Lab/U1W2_05_Introduction_to_Pandas.ipynb)
- [U1W2_06_Linear_Regression_C.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2017/Lab/U1W2_06_Linear_Regression_C.ipynb)
- [U1W2_07_KNN_Iris_data_C.ipynb](Programs/Week%202%20%28end_time%20-%2024_May_6PM%29/May%2017/Lab/U1W2_07_KNN_Iris_data_C.ipynb)
- [U1W3_08_Logistic_Regression_C.ipynb](Programs/Week%203%20%28end_time%20-%2031_Dec_6PM%29/May%2023/Lab/U1W3_08_Logistic_Regression_C.ipynb)
- [U1W3_09_LC_Fruits_data_C.ipynb](Programs/Week%203%20%28end_time%20-%2031_Dec_6PM%29/May%2023/Lab/U1W3_09_LC_Fruits_data_C.ipynb)
- [U1W3_10_KNN_DT_LC_Zoo_data_C.ipynb](Programs/Week%203%20%28end_time%20-%2031_Dec_6PM%29/May%2024/Lab/U1W3_10_KNN_DT_LC_Zoo_data_C.ipynb)
- [U1W3_11_DT_Iris_data_Visualization_C.ipynb](Programs/Week%203%20%28end_time%20-%2031_Dec_6PM%29/May%2024/Lab/U1W3_11_DT_Iris_data_Visualization_C.ipynb)

If you want, I can next make this into:
- separate weekwise files,
- a more visual guide with tables and flow charts, or
- a notebook-style study sheet with questions and answers.
