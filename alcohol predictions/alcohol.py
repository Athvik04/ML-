import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the data
file_path = r"D:\5th sem\Machiene learning\alcohol\student-mat.csv"
data = pd.read_csv(file_path)

# Define the passing threshold
passing_threshold = 10

# Create a binary target variable based on the passing threshold
data['pass'] = data['G3'] >= passing_threshold
data['pass'] = data['pass'].astype(int)  # 1 for pass, 0 for fail

# Drop the original grade columns (G1, G2, G3) as we are focusing on 'pass'
data = data.drop(columns=['G1', 'G2', 'G3'])

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split the data into features (X) and target (y)
X = data.drop(columns=['pass'])
y = data['pass']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)

# Visualize the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=['Fail', 'Pass'], filled=True, rounded=True)
plt.show()

