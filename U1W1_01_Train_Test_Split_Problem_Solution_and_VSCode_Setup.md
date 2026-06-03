# U1W1_01 Train Test Split: Problem, Solution, and VS Code Setup

This file is written from the notebook [U1W1_01_Train_Test_Split.ipynb](Programs/Week%201%20%28end_time%20-%2017_May_6PM%29/May%2009/Lab/U1W1_01_Train_Test_Split.ipynb) and keeps only the main problem and solution idea.

---

## Problem

The goal is to understand how to split a dataset into two parts:

- Train set: used to teach the model.
- Test set: used to check how well the model works on new data.

In the notebook, the example uses the diabetes dataset. The main question is:

How do we divide the data so that the model learns from one part and is evaluated on another part it has never seen before?

Why this matters:

- If you test the model on the same data it learned from, the result is not trustworthy.
- A separate test set gives a more honest estimate of performance.
- If the dataset is very small, a single split can be weak, so k-fold cross validation may be better later.

---

## Solution

The notebook shows two ways to do the split.

### 1) Manual split after shuffling

First, the data is loaded into a Pandas dataframe.

Then the rows are shuffled so the train and test sets do not follow the original order.

Simple idea:

```python
shuffle_df = dataframe.sample(frac=1)
train_size = int(0.7 * len(dataframe))
train_set = shuffle_df[:train_size]
test_set = shuffle_df[train_size:]
```

What this means:

- `sample(frac=1)` shuffles all rows.
- `train_size` is 70% of the dataset.
- The first 70% goes to training.
- The remaining 30% goes to testing.

The notebook then checks the size of both sets and also looks at the class distribution using `groupby(['Outcome']).count()`.

### 2) Using scikit-learn

This is the standard and safer way.

```python
from sklearn.model_selection import train_test_split

features = dataframe.drop('Outcome', axis=1)
target = dataframe['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.3
)
```

What this does:

- `features` contains the input columns.
- `target` contains the answer column.
- `test_size=0.3` means 30% test data and 70% train data.
- `x_train`, `y_train` are used to train the model.
- `x_test`, `y_test` are used to evaluate the model.

### Simple takeaway

The main idea is:

1. Load the data.
2. Separate input and target.
3. Split the data into train and test parts.
4. Train on the train set.
5. Check performance on the test set.

---

## Very Easy Explanation

Think of it like this:

- Train set = practice questions.
- Test set = exam questions.

If a student only practices and never takes a real exam, you do not know whether they truly learned.
In machine learning, the test set is the real exam.

---

## VS Code Setup for AI/ML

VS Code is a very good choice for AI/ML work. It is lightweight, flexible, and widely used in industry.

It is especially good when you combine it with:

- Python support
- Jupyter notebook support
- Pylance for smart code completion and type checking
- Git integration
- Terminal access inside the editor

### Recommended extensions

Install these in VS Code:

- Python
- Jupyter
- Pylance
- GitLens
- Data Wrangler
- Black Formatter
- isort
- Ruff
- Markdown All in One
- GitHub Copilot, if available

### Recommended environment setup

For a proper AI/ML setup, use:

- Python 3.10 or 3.11
- A virtual environment such as `venv`
- `pip` for package installation
- Jupyter notebooks for experiments
- Git for version control

### Common packages for AI/ML

These are the basics most learners need:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- jupyter
- ipykernel

Optional later:

- xgboost
- lightgbm
- scipy
- statsmodels
- tensorflow or pytorch, depending on your path

### Minimal project structure

```text
ai-ml-project/
  data/
  notebooks/
  src/
  models/
  requirements.txt
  README.md
```

### Good workflow in VS Code

1. Create and activate a virtual environment.
2. Install the required packages.
3. Open the notebook in VS Code.
4. Select the correct Python kernel.
5. Run cells step by step.
6. Save your final code in a clean script if needed.

---

## Short Answer

Yes, VS Code is a strong industry-standard choice for AI/ML learning and development, especially for Python, notebooks, and model building.

If you set it up with the right extensions and a clean Python environment, it works very well for:

- data cleaning
- model training
- notebook experiments
- script development
- debugging
- Git-based project work

---

## What this notebook teaches

- Why train/test split is needed
- How to split data manually
- How to split data using scikit-learn
- Why shuffling is useful before splitting
- How to think about model evaluation in simple terms

---

## Next step

If you want, I can create one more file with:

- the same format for Week 1 KFold,
- a complete VS Code AI/ML setup checklist, or
- a full `requirements.txt` for your AI/ML environment.
