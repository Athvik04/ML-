# import numpy as np
# import pandas as pd
# from sklearn.datasets import load_breast_cancer

# cancer = load_breast_cancer()

# print(cancer.DESCR) # Print the data set description
# cancer.keys()
# def load_df():
#     df = pd.DataFrame(cancer.data, columns=[cancer.feature_names])
#     df['target'] = pd.Series(data=cancer.target, index=df.index)
#     return df
# def split_data():
#     cancerdf = load_df()
#     X = cancerdf[cancerdf.columns[:-1]]
#     y = cancerdf[cancerdf.columns[-1]]
#     return X,y
# from sklearn.model_selection import train_test_split

# def train_test_split_data():
#     X, y = split_data()
#     X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
#     return X_train, X_test, y_train, y_test
# from sklearn.neighbors import KNeighborsClassifier

# def fit_model():
#     X_train, X_test, y_train, y_test = train_test_split_data()
#     knn = KNeighborsClassifier(n_neighbors = 1)
#     model = knn.fit(X_train,y_train)
#     return model
# def predict_for_means():
#     cancerdf = load_df()
#     means = cancerdf.mean()[:-1].values.reshape(1, -1)
#     model = fit_model()
#     result = model.predict(means)
#     return result
# predict_for_means()
# def predict_labels_for_test():
#     X_train, X_test, y_train, y_test = train_test_split_data()
#     knn = fit_model()
#     result = knn.predict(X_test)
#     return result
# predict_labels_for_test()
# def score_model():
#     X_train, X_test, y_train, y_test = train_test_split_data()
#     knn = fit_model()
#     score = knn.score(X_test,y_test)
#     return score
# score_model()
# def accuracy_plot():
#     import matplotlib.pyplot as plt

#     # %matplotlib notebook

#     X_train, X_test, y_train, y_test = answer_four()

#     # Find the training and testing accuracies by target value (i.e. malignant, benign)
#     mal_train_X = X_train[y_train==0]
#     mal_train_y = y_train[y_train==0]
#     ben_train_X = X_train[y_train==1]
#     ben_train_y = y_train[y_train==1]

#     mal_test_X = X_test[y_test==0]
#     mal_test_y = y_test[y_test==0]
#     ben_test_X = X_test[y_test==1]
#     ben_test_y = y_test[y_test==1]

#     knn = answer_five()

#     scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
#               knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


#     plt.figure()

#     # Plot the scores as a bar chart
#     bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

#     # directly label the score onto the bars
#     for bar in bars:
#         height = bar.get_height()
#         plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
#                      ha='center', color='w', fontsize=11)

#     # remove all the ticks (both axes), and tick labels on the Y axis
#     plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

#     # remove the frame of the chart
#     for spine in plt.gca().spines.values():
#         spine.set_visible(False)

#     plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
#     plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
# accuracy_plot() 


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load dataset
cancer = load_breast_cancer()

# Print dataset description
print(cancer.DESCR)

# Load dataset into a DataFrame
def load_df():
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = pd.Series(data=cancer.target, index=df.index)
    return df

# Split data into features and target
def split_data():
    cancerdf = load_df()
    X = cancerdf[cancerdf.columns[:-1]]
    y = cancerdf[cancerdf.columns[-1]]
    return X, y

# Train-test split function
def train_test_split_data():
    X, y = split_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test

# Model fitting function
def fit_model():
    X_train, X_test, y_train, y_test = train_test_split_data()
    knn = KNeighborsClassifier(n_neighbors=1)
    model = knn.fit(X_train, y_train)
    return model

# Predict for means function
def predict_for_means():
    cancerdf = load_df()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    model = fit_model()
    result = model.predict(means)
    return result

# Predict labels for test set
def predict_labels_for_test():
    X_train, X_test, y_train, y_test = train_test_split_data()
    model = fit_model()
    result = model.predict(X_test)
    return result

# Model scoring function
def score_model():
    X_train, X_test, y_train, y_test = train_test_split_data()
    model = fit_model()
    score = model.score(X_test, y_test)
    return score

# Accuracy plot function
def accuracy_plot():
    X_train, X_test, y_train, y_test = train_test_split_data()

    # Separate malignant and benign cases in training and testing sets
    mal_train_X = X_train[y_train == 0]
    mal_train_y = y_train[y_train == 0]
    ben_train_X = X_train[y_train == 1]
    ben_train_y = y_train[y_train == 1]

    mal_test_X = X_test[y_test == 0]
    mal_test_y = y_test[y_test == 0]
    ben_test_X = X_test[y_test == 1]
    ben_test_y = y_test[y_test == 1]

    # Fit the model
    knn = fit_model()

    # Calculate scores
    scores = [
        knn.score(mal_train_X, mal_train_y), 
        knn.score(ben_train_X, ben_train_y), 
        knn.score(mal_test_X, mal_test_y), 
        knn.score(ben_test_X, ben_test_y)
    ]

    # Plot the scores as a bar chart
    plt.figure()
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0', '#4c72b0', '#55a868', '#55a868'])

    # Directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width() / 2, height * .90, f'{height:.2f}', 
                       ha='center', color='w', fontsize=11)

    # Customize plot appearance
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=True)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0, 1, 2, 3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8)
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
    plt.show()

# Call the functions to run the analysis and display results
print("Prediction for mean values:", predict_for_means())
print("Test set predictions:", predict_labels_for_test())
print("Model accuracy on test set:", score_model())
accuracy_plot()
