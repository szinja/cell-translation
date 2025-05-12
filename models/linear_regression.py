from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import json

def train_linear_model(X_train, y_train, X_test, y_test):
    """
    Trains a linear regression model using GridSearchCV for hyperparameter tuning
    (specifically, tuning the 'fit_intercept' parameter).

    Args:
        X_train (pd.DataFrame): Training data features.
        y_train (pd.Series): Training data target.
        X_test (pd.DataFrame): Testing data features.
        y_test (pd.Series): Testing data target.

    Returns:
        sklearn.linear_model: Trained linear regression model.
    """

    # Define the parameter grid
    param_grid = {
        "alpha": [0.01,0.1, 1, 10, 100],  # Regularization strength
        "l1_ratio": np.linspace(0, 1, 5),  # Mix between L1 and L2 (0 = Ridge, 1 = Lasso)
        "fit_intercept": [True, False],  # Whether to include an intercept
        "precompute": [False],  # Precompute Gram matrix (not used often)
        "max_iter": [500, 1000, 2000],  # Number of iterations
        "copy_X": [True],  # Whether to copy X or overwrite it
        "tol": [1e-4, 1e-3, 1e-2],  # Tolerance for stopping
        "warm_start": [False, True],  # Reuse previous solution as initialization
        "positive": [False, True],  # Force coefficients to be positive
        "random_state": [None, 42],  # Random seed
        "selection": ["cyclic", "random"],  # Feature selection strategy
    }

    #to do, learning curve
    # l1 ratio - 0.3, 1, 3,
    # Fit intercept, warm_start - remove
    # ElasticNet CV - to try
    # fix random seed might be a good idea
    # Aplha 0.01, 0.03, 0.1, 0.3, 1

    # # Define parameter grid for LinearRegression
    # param_grid = {'fit_intercept': [True, False], # Purpose: Determines whether the model should calculate an intercept or assume the data is already centered around zero.
    #               'copy_X': [True, False], # Purpose: Controls whether X (input data) is copied before fitting.
    #               'n_jobs': [None, 2, 4,6,8]} # Purpose: Specifies the number of CPU cores to use for computation (kind of useless in our case as only 800 rows of data, haha! :))

    # Initialize Linear Regression model
    # linear_regression = LinearRegression()
    linear_regression = ElasticNet()

    # Perform GridSearchCV
    grid_search = GridSearchCV(linear_regression, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=True)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Best Linear Regression Elastic Net parameters:", grid_search.best_params_)

    best_params_file = "best_params.json"

    best_params["model"] = "elastic net"
       # Save best parameters to a JSON file
    with open(best_params_file, "a") as f:
        json.dump(best_params, f, indent=4)
        f.write("\n")

    print(f"Parameters saved in {best_params_file}")
    # Evaluate the best model
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)


    return best_model