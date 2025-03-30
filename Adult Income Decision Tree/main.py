# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# import graphviz
# from sklearn import tree

# # Load the dataset
# df = pd.read_csv(r"D:\5th sem\Machiene learning\Adult-Income-Decision-Tree-main\Adult-Income-Decision-Tree-main\AdultIncome (1).csv")

# # Check for missing values
# print(df.isnull().sum())

# # Apply one-hot encoding
# df = pd.get_dummies(df, drop_first=True)

# # Separate features (X) and target (y)
# x = df.iloc[:, :-1]
# y = df.iloc[:, -1]

# # Check class distribution to ensure it's not imbalanced
# print(y.value_counts())

# # Split the dataset into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# # Train the decision tree classifier with a max depth to prevent overfitting
# dtc = DecisionTreeClassifier(random_state=0, max_depth=5)
# dtc.fit(x_train, y_train)

# # Make predictions
# y_pred = dtc.predict(x_test)

# # Evaluate the model
# cm = confusion_matrix(y_test, y_pred)
# score = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# print("Confusion Matrix:\n", cm)
# print("Accuracy Score:", score)
# print("Classification Report:\n", report)

# # Cross-validation scores
# cross_val_scores = cross_val_score(dtc, x, y, cv=5)
# print("Cross-Validation Scores:", cross_val_scores)

# # Print the column names and group by workclass-related columns
# print(df.columns)
# print(df.filter(like='wc_').sum())  # Sum of the one-hot encoded workclass columns

# # Create and visualize the decision tree
# full = list(df.columns)
# feat = full.pop()
# features = full
# target = feat
# print(target)

# dot_data = tree.export_graphviz(dtc, out_file=None, 
#                                 feature_names=features,  
#                                 class_names=[target],  # Since target is a single variable, pass as a list
#                                 filled=True)

# graph = graphviz.Source(dot_data, format="jpg")
# graph.render("decision_tree_graphviz")


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import graphviz
from sklearn import tree

# Load the dataset
df = pd.read_csv(r"D:\5th sem\Machiene learning\Adult-Income-Decision-Tree-main\Adult-Income-Decision-Tree-main\AdultIncome (1).csv")

# Check for missing values
print(df.isnull().sum())

# Apply one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Separate features (X) and target (y)
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Check class distribution to ensure it's not imbalanced
print(y.value_counts())

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train the decision tree classifier with a max depth to prevent overfitting
dtc = DecisionTreeClassifier(random_state=0, max_depth=5)
dtc.fit(x_train, y_train)

# Make predictions
y_pred = dtc.predict(x_test)

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Accuracy Score:", score)
print("Classification Report:\n", report)

# Cross-validation scores
cross_val_scores = cross_val_score(dtc, x, y, cv=5)
print("Cross-Validation Scores:", cross_val_scores)

# Print the column names and group by workclass-related columns
print(df.columns)
print(df.filter(like='wc_').sum())  # Sum of the one-hot encoded workclass columns

# Create and visualize the decision tree
full = list(df.columns)
feat = full.pop()
features = full
target = feat
print(target)

dot_data = tree.export_graphviz(dtc, out_file=None, 
                                feature_names=features,  
                                class_names=['<=50K', '>50K'],  # Ensure this matches the actual class names
                                filled=True)

graph = graphviz.Source(dot_data, format="jpg")
graph.render("decision_tree_graphviz")
