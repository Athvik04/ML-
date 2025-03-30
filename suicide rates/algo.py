# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r'D:\5th sem\Machiene learning\disha\Crude suicide rates.csv'
df = pd.read_csv(file_path)

# Strip any extra whitespace from column names
df.columns = df.columns.str.strip()

# Check column names to confirm '20to29' is present
print("Columns in the dataset:", df.columns)

# Define a binary target variable (e.g., high suicide rate in 20to29 group)
# Here, we classify as 1 if the suicide rate is above 10 in the 20to29 group, else 0.
if '20to29' in df.columns:
    df['High_Suicide_20to29'] = (df['20to29'] > 10).astype(int)
else:
    raise KeyError("Column '20to29' not found in the dataset. Please check the column names.")

# Select features and target variable
X = df[['80_above', '70to79', '60to69', '50to59', '40to49', '30to39', '20to29', '10to19']]
y = df['High_Suicide_20to29']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# Display results
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
print("AUC-ROC Score:", roc_auc)

# Confusion matrix heatmap
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label="AUC = %.2f" % roc_auc)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
