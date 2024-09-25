# Purchase-Classifier-with-Decision-Tree-Algorithm
This project involves predicting whether a customer will purchase a product based on features such as gender, age, and annual salary using a Decision Tree Classifier. The dataset contains 1000 entries with features like Gender, Age, and AnnualSalary, and the target variable Purchased.

Project Overview
The goal of this project is to classify customers as potential buyers or non-buyers using a Decision Tree algorithm. The project involves:

Data preprocessing
Training a Decision Tree Classifier
Evaluating the model’s accuracy
Visualizing the decision tree
Dataset
The dataset contains the following columns:

User ID: Unique ID for each customer
Gender: Male or Female (encoded to numerical values)
Age: Customer's age
AnnualSalary: Customer's annual salary
Purchased: Target variable (0 for not purchased, 1 for purchased)
Project Steps
Data Preprocessing: Convert categorical features into numerical ones using LabelEncoder.
Splitting Data: Split the dataset into training and testing sets.
Model Training: Train a Decision Tree Classifier.
Model Evaluation: Check accuracy on the test data.
Visualization: Visualize the Decision Tree structure.
Libraries Used
pandas: For data manipulation
numpy: For numerical operations
matplotlib & seaborn: For visualization
sklearn: For machine learning algorithms and utilities
