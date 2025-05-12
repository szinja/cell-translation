from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from xgboost import plot_importance
from evaluation import *
import json

# Define the XGBRegressor model
def train_xgboost_model(X_train, y_train):
    xgb_model = XGBRegressor(objective='reg:squarederror', )

# Define the parameters for grid search
    param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7],            # Depth of each tree
    'learning_rate': [0.001, 0.01, 0.1, 0.2],    # Step size shrinkage
    'n_estimators': [10, 20, 50, 100],        # Number of boosting rounds
    'subsample': [0.8, 1],                # Fraction of samples used per tree
    'colsample_bytree': [0.8, 1],         # Fraction of features used per tree
    'gamma': [0, 0.1, 0.2],               # Minimum loss reduction required
    'reg_alpha': [0, 0.1, 1],             # L1 regularization
    'reg_lambda': [1, 1.5, 2],
    'device' : ['cuda'],
    'tree_method': ['hist']            # L2 regularization
}

# Set up the GridSearchCV
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=5, verbose=1) # GridSearchCV maximizes the scoring function, so mean_squared_error is negated. This way, minimizing the MSE becomes equivalent to maximizing the negative MSE.

# Fit grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    best_params["model"] = "nn"
        # Save best parameters to a JSON file
    with open("best_params.json", "a") as f:
        json.dump(best_params, f, indent=4)
        f.write("\n")
    print("Best parameters:", best_params)

# Get the best estimator and evaluate on test set
    best_model = grid_search.best_estimator_
    return best_model


