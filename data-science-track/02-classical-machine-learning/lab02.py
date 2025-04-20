from sklearn.datasets import load_iris # step 1
import pandas as pd
from sklearn.model_selection import train_test_split # step 2
from sklearn.neighbors import KNeighborsClassifier # step 3
from sklearn.metrics import accuracy_score # step 4
from sklearn.metrics import confusion_matrix, classification_report # step 5
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score # step 6

# Step 1 load the iris dataset and inspect it
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

print(X.head())
print(y.head())

# Step 2 split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3 train the classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Step 4 validate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Step 5 evaluate the performance
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Predicted vs Actual")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('predicated-vs-actual.png')
plt.close()

# Classification Report
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 6 cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-Validation Accuracy: {scores.mean():.2f}")
