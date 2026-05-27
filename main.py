import pandas as pd
import numpy as np

data = pd.read_csv("HR-Employee-Attrition.csv")

data['Attrition'] = data['Attrition'].map({
    'Yes': 1,
    'No': 0
})

features = [
    'Education',
    'EnvironmentSatisfaction',
    'JobInvolvement',
    'JobSatisfaction',
    'PerformanceRating',
    'RelationshipSatisfaction',
    'WorkLifeBalance'
]

X = data[features].values
y = data['Attrition'].values.reshape(-1, 1)

X = (X - X.mean(axis=0)) / X.std(axis=0)

split = int(0.8 * len(X))

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

class LogisticRegression:

    def __init__(self, lr=0.01, epochs=1000, batch_size=32):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize(self, n_features):
        self.W = np.zeros((n_features, 1))
        self.b = 0

    def fit(self, X, y):

        m, n = X.shape
        self.initialize(n)

        for epoch in range(self.epochs):

            for i in range(0, m, self.batch_size):

                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]

                z = np.dot(X_batch, self.W) + self.b
                y_pred = self.sigmoid(z)

                dz = y_pred - y_batch
                dW = np.dot(X_batch.T, dz) / len(X_batch)
                db = np.sum(dz) / len(X_batch)

                self.W -= self.lr * dW
                self.b -= self.lr * db

    def predict(self, X):

        z = np.dot(X, self.W) + self.b
        y_pred = self.sigmoid(z)

        return (y_pred > 0.5).astype(int)

model = LogisticRegression(
    lr=0.01,
    epochs=1000,
    batch_size=32
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

def evaluate(y_true, y_pred):

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / len(y_true)

    precision = TP / (TP + FP + 1e-10)

    recall = TP / (TP + FN + 1e-10)

    f1_score = (
        2 * precision * recall
    ) / (precision + recall + 1e-10)

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1_score)

evaluate(y_test, y_pred)
