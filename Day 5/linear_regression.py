import numpy as np

class LinearRegression:
    def _init_(self):
        self.slope = None
        self.intercept = None
    
    def fit(self, X, y):
        # Calculate mean of X and y
        mean_x = np.mean(X)
        mean_y = np.mean(y)
        
        # Calculate the slope (m)
        numerator = np.sum((X - mean_x) * (y - mean_y))
        denominator = np.sum((X - mean_x) ** 2)
        self.slope = numerator / denominator
        
        # Calculate the intercept (b)
        self.intercept = mean_y - self.slope * mean_x
    
    def predict(self, X):
        if self.slope is None or self.intercept is None:
            raise Exception("Model has not been trained yet. Please fit the model first.")
        
        return self.slope * X + self.intercept

# Example usage:
X = np.array([10, 15, 20, 25, 30])
y = np.array([2, 3, 4, 5, 6])

model = LinearRegression()
model.fit(X, y)

print("Slope (m):", model.slope)
print("Intercept (b):", model.intercept)

new_X = 6
predicted_y = model.predict(new_X)
print("Predicted value for X =", new_X, ":",predicted_y)