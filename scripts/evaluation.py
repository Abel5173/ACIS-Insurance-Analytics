import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_regression_model(model, X, y, model_name='Model', result_path='docs/task-4_results.txt', test_size=0.2, random_state=42):
    """
    Train, test, evaluate, and log a regression model.
    
    Parameters:
        model: sklearn-like regression model instance
        X (DataFrame or array): Features
        y (Series or array): Target
        model_name (str): Name to print and log
        result_path (str): Where to save results
        test_size (float): Fraction of test data
        random_state (int): Seed for reproducibility
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    result = f"{model_name} - RMSE: {rmse:.2f}, R²: {r2:.2f}"
    print(result)

    # Save to file
    with open(result_path, 'a') as f:
        f.write(result + '\n')

    return {'model': model, 'rmse': rmse, 'r2': r2, 'predictions': y_pred}
