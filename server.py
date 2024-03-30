from flask import Flask, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)

@app.route('/train-model', methods=['GET'])
def train_model():
    # Load the dataset
    df = pd.read_csv(r'C:\Users\wasin\Downloads\PremierLeague.csv')

    # Preprocess the data
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    encoder = OneHotEncoder(sparse=False)
    X_categorical = encoder.fit_transform(X[['Team']])
    X = X.drop('Team', axis=1)
    X = pd.concat([X, pd.DataFrame(X_categorical, index=X.index)], axis=1)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Save the trained model
    model_path = 'premier_league_model.joblib'
    joblib.dump(model, model_path)

    return jsonify({'message': 'Model trained and saved successfully!', 'model_path': model_path})

if __name__ == '__main__':
    app.run(debug=True)
