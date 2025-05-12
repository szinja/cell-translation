from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import json

def train_lasso_model(X_train, y_train, X_test=None, y_test=None):
    """
    Trains a Lasso regression model using GridSearchCV for hyperparameter tuning.
    This function performs a grid search over several values of alpha, fit_intercept,
    and max_iter. Optionally, it evaluates the best model on a test set.

    Args:
        X_train (pd.DataFrame): Training data features.
        y_train (pd.Series): Training data target.
        X_test (pd.DataFrame, optional): Testing data features.
        y_test (pd.Series, optional): Testing data target.

    Returns:
        sklearn.linear_model.Lasso: Trained Lasso regression model with best hyperparameters.
    """
    # Define the parameter grid
    param_grid = {
        "alpha": [0.01, 0.1, 1, 10, 30, 50, 70, 100],  # Regularization strength
        #"fit_intercept": [True, False],  # Whether to include an intercept
        "precompute": [False],  # Precompute Gram matrix (not used often)
        #"max_iter": [500, 1000, 2000, 3000],  # Number of iterations
        "copy_X": [True],  # Whether to copy X or overwrite it
       # "tol": [1e-4, 1e-3, 1e-2],  # Tolerance for stopping
        #"warm_start": [False, True],  # Reuse previous solution as initialization
        #"positive": [False, True],  # Force coefficients to be positive
        #"random_state": [None, 42],  # Random seed
        "selection": ["random"],  # Feature selection strategy
    }

# grid for alpha should follow a structured pattern, 0.1,0.3,1,3,10, 30
#
    lasso = Lasso()
    grid_search = GridSearchCV(lasso, param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_

    best_params["model"] = "lasso"
        # Save best parameters to a JSON file
    with open("best_params.json", "a") as f:
        json.dump(best_params, f, indent=4)
        f.write("\n")

    print("Best Lasso parameters:", best_params)
    best_model = grid_search.best_estimator_
    

    
    return best_model