from sklearn.linear_model import Ridge
from evaluation import *
from sklearn.model_selection import GridSearchCV
import json

def train_ridge_model(X_train, y_train):

    # Define parameter grid
    param_grid = {
        'alpha': [0.01, 0.1, 1, 10, 20, 30, 50, 70 , 100],  # Regularization strength
        'fit_intercept': [True, False],    # Whether to fit intercept
         "copy_X": [True],  # Whether to copy X or overwrite it
        'max_iter': [500, 1000, 5000, 10000],   # Maximum iterations
        'tol': [1e-4, 1e-3, 1e-2],         # Convergence tolerance
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'lbfgs'],  # Different solvers
        'positive': [False, True]          # Enforce positive coefficients
    }

    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    print("Best Ridge parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    
    best_params["model"] = "ridge"
    # Save best parameters to a JSON file
    with open("best_params.json", "a") as f:
        json.dump(best_params, f, indent=4)
        f.write("\n")
    
    best_model.fit(X_train, y_train)
    return best_model
