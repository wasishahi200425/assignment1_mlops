import pandas as pd

# Load the dataset
df = pd.read_csv('C:\Users\wasin\Downloads\PremierLeague.csv')

# Display the first few rows of the dataframe
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Assuming 'Outcome' is the target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# If you have categorical features, you would encode them here
# For demonstration, let's assume 'Team' is a categorical feature that needs encoding
encoder = OneHotEncoder(sparse=False)
X_categorical = encoder.fit_transform(X[['Team']])
X = X.drop('Team', axis=1)
X = pd.concat([X, pd.DataFrame(X_categorical)], axis=1)

# Split dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
import joblib

# Train a model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'premier_league_model.joblib')
