# # Import necessary libraries
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
# from sklearn.utils import resample
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the dataset
# file_path = r'D:\5th sem\Machiene learning\income\AdultIncome (1).csv'
# df = pd.read_csv(file_path)

# # Data preprocessing
# # Encode 'IncomeClass' as a binary variable
# df['IncomeClass'] = df['IncomeClass'].apply(lambda x: 1 if x == '>50K' else 0)

# # Check for class imbalance
# print("Class distribution before balancing:\n", df['IncomeClass'].value_counts())

# # Balance the dataset by oversampling the minority class
# df_majority = df[df['IncomeClass'] == 0]
# df_minority = df[df['IncomeClass'] == 1]
# df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
# df_balanced = pd.concat([df_majority, df_minority_upsampled])

# print("Class distribution after balancing:\n", df_balanced['IncomeClass'].value_counts())

# # Select features and encode categorical variables
# # For simplicity, encoding 'gender', 'wc', and 'education' with one-hot encoding
# df_encoded = pd.get_dummies(df_balanced, columns=['wc', 'education', 'gender'], drop_first=True)

# # Define features (X) and target (y)
# X = df_encoded[['age', 'hours per week'] + [col for col in df_encoded.columns if col.startswith('wc_') or col.startswith('education_') or col.startswith('gender_')]]
# y = df_encoded['IncomeClass']

# # Split data into training and testing sets with stratification to ensure both classes are present in each set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Initialize and train the logistic regression model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)
# y_proba = model.predict_proba(X_test)[:, 1]

# # Calculate evaluation metrics
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, y_proba)

# # Display results
# print("Accuracy:", accuracy)
# print("\nConfusion Matrix:\n", conf_matrix)
# print("\nClassification Report:\n", class_report)
# print("AUC-ROC Score:", roc_auc)

# # Confusion matrix heatmap
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# # ROC Curve
# fpr, tpr, _ = roc_curve(y_test, y_proba)
# plt.plot(fpr, tpr, label="AUC = %.2f" % roc_auc)
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.show()

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r'D:\5th sem\Machiene learning\income\AdultIncome (1).csv'
df = pd.read_csv(file_path)

# Display unique values in 'IncomeClass' to verify classes
print("Unique values in IncomeClass column before processing:\n", df['IncomeClass'].unique())

# Encode 'IncomeClass' as a binary variable (1 for >50K, 0 for <=50K)
df['IncomeClass'] = df['IncomeClass'].apply(lambda x: 1 if x == '>50K' else 0)

# Check for class distribution after conversion
print("Class distribution after conversion:\n", df['IncomeClass'].value_counts())

# Proceed only if both classes are present
if len(df['IncomeClass'].value_counts()) > 1:
    # Balance the dataset by oversampling the minority class
    df_majority = df[df['IncomeClass'] == 0]
    df_minority = df[df['IncomeClass'] == 1]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_upsampled])

    # Continue with model training and evaluation
    # (The rest of your logistic regression code would go here)
else:
    print("Error: Only one class present in IncomeClass after conversion. Cannot proceed with model training.")

