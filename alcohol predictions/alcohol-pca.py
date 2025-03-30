import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the data
file_path = r"D:\5th sem\Machiene learning\alcohol\student-mat.csv"
data = pd.read_csv(file_path)

# Define the target variable (pass/fail based on a threshold of 10)
passing_threshold = 10
data['pass'] = (data['G3'] >= passing_threshold).astype(int)

# Drop the original grade columns
data = data.drop(columns=['G1', 'G2', 'G3'])

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Separate features and target variable
X = data.drop(columns=['pass'])
y = data['pass']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce to 2 components for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the PCA results
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.7)
plt.colorbar(scatter, label='Pass (1) / Fail (0)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Student Performance Dataset')
plt.show()
