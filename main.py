# Decision Tree Project

#This project involves predicting whether a customer will purchase based on features such as gender, age, and annual salary using a Decision Tree Classifier.

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Load the dataset
df = pd.read_csv("E:\\car_data.csv")
df.head()

# Preprocessing
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Age'] = le.fit_transform(df['Age'])
df['AnnualSalary'] = le.fit_transform(df['AnnualSalary'])

# Prepare target and features
x = df.drop(['Purchased', 'User ID'], axis=1)
y = df['Purchased']

# Split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Train the model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Test accuracy
accuracy = model.score(x_test, y_test)
print(f"Accuracy: {accuracy}")

# Plot the Decision Tree
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=3)
clf.fit(x_train, y_train)
tree.plot_tree(clf, filled=True)
