import scipy.io.arff as arff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Define the file path for the ARFF dataset file
arff_file = "C:\\Users\\ASHAMARTS\\Documents\\DDSAI\\Data_set\\mnist_784.arff"

# Load the ARFF dataset
dataset, meta = arff.loadarff(arff_file)

# Convert the dataset to a NumPy array
mnist_data = np.array(dataset.tolist(), dtype=np.uint8)

# Split the dataset into features and labels
X = mnist_data[:, :-1]  # Features
y = mnist_data[:, -1]   # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the KNN classifier
k = 3  # You can change the value of k
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of KNN with k={k}: {accuracy*100:.2f}%")

# Plot accuracy
plt.figure(figsize=(8, 6))
plt.plot(range(1, 21), accuracy_history, marker='o', linestyle='--', color='b')
plt.title("KNN Accuracy vs. K Value")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

