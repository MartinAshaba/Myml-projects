import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Define the file paths for the dataset files
image_data_file = "C:\Users\ASHAMARTS\Documents\DDSAI\Data_set\mnist_784.arff"
label_data_file = "C:\Users\ASHAMARTS\Documents\DDSAI\Data_set\mnist_784.arff"

# Load the image data
#mnist_data = np.genfromtxt(image_data_file, delimiter=',')
# Load the image data
mnist_data = np.fromfile(image_data_file, dtype=np.uint8)

# Load the label data
mnist_labels = np.fromfile(label_data_file, dtype=np.uint8)
# Load the label data
#mnist_labels = np.genfromtxt(label_data_file, delimiter=',')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mnist_data, mnist_labels, test_size=0.2, random_state=42)

# Initialize and train the KNN classifier
k = 3  # You can change the value of k
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of KNN with k={k}: {accuracy*100:.2f}%")
