import pandas as pd
import pytest
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Fixture for creating a small sample dataset
@pytest.fixture
def create_sample_data():
    data = {
        'Team': ['Team1', 'Team2', 'Team1', 'Team2'],
        'Outcome': [1, 0, 1, 0],
        'Feature1': [23, 45, 65, 32],
        'Feature2': [1.2, 2.3, 3.4, 4.5]
    }
    df = pd.DataFrame(data)
    return df

# Fixture for logistic regression model
@pytest.fixture
def trained_model():
    model = LogisticRegression(max_iter=200)
    return model


def test_data_preprocessing(create_sample_data):
    df = create_sample_data
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    encoder = OneHotEncoder(sparse=False)
    X_categorical = encoder.fit_transform(X[['Team']])
    X = X.drop('Team', axis=1)
    X = pd.concat([X, pd.DataFrame(X_categorical)], axis=1)

    # Assert 'Team' is removed and 'Outcome' is target
    assert 'Team' not in X.columns
    assert len(y) == len(df)
    # Assert One-Hot Encoding increased column count
    assert X.shape[1] == len(df.columns) - 1 + len(df['Team'].unique()) - 1



def test_model_training_and_saving(create_sample_data, trained_model, tmpdir):
    df = create_sample_data
    X = df[['Feature1', 'Feature2']]  # Simplified for test
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Assume model is trained here
    model = trained_model
    model.fit(X_train, y_train)

    # Saving model
    save_path = tmpdir.join("model.joblib")
    joblib.dump(model, save_path)
    
    # Assert model file exists
    assert save_path.isfile()
    # Load model and make a prediction to ensure it's saved correctly
    loaded_model = joblib.load(save_path)
    prediction = loaded_model.predict(X_test)
    assert prediction is not None
