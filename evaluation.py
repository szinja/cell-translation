# evaluation/evaluation.py
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import torch
import pandas as pd


def evaluate_model(model,X_train, y_train, X_test, y_test,model_name):
    
    if model_name=="nn":
        X_train_nn = torch.FloatTensor(X_train.values)
    
        X_test_nn = torch.FloatTensor(X_test.values)

        # Ensure the model is in evaluation mode if necessary
        model.eval()

        # Get predictions from the model on the training data
        with torch.no_grad():  # Ensuring no gradients are being computed
            # y_train_pred = model(X_train)
            y_train_pred = model(X_train_nn).cpu().detach().numpy()
            y_pred = model(X_test_nn).cpu().detach().numpy()

    else:
        # Make predictions on the testing data
        y_pred = model.predict(X_test)


        #prediction on training data
        y_train_pred = model.predict(X_train)



    train_results= train_evaluate(y_train, y_train_pred)
    test_results = test_evaluate(y_test, y_pred)

    return train_results, test_results

def train_evaluate(y_train, y_train_pred):
        
        LR_train_mse_1 = mean_squared_error(y_train.values[:,0], y_train_pred[:,0])
        LR_train_mse_2 = mean_squared_error(y_train.values[:,1], y_train_pred[:,1])

    # Calculating the mean absolute error of the predictions
        LR_train_mae_1 = mean_absolute_error(y_train.values[:,0], y_train_pred[:,0])
        LR_train_mae_2 = mean_absolute_error(y_train.values[:,1], y_train_pred[:,1])

    # Calculating the root mean squared error of the predictions
        LR_train_rmse_1 = np.sqrt(LR_train_mse_1)
        LR_train_rmse_2 = np.sqrt(LR_train_mse_2)

        LR_train_Rsquared_1 = r2_score(y_train.values[:,0], y_train_pred[:,0])
        LR_train_Rsquared_2 = r2_score(y_train.values[:,1], y_train_pred[:,1])

        return [LR_train_mse_1,LR_train_mse_2,LR_train_mae_1,LR_train_mae_2,LR_train_rmse_1,LR_train_rmse_2,LR_train_Rsquared_1, LR_train_Rsquared_2 ]


# Calculating the mean squared error of the predictions
def test_evaluate(y_test,y_pred):
    LR_test_mse_1 = mean_squared_error(y_test.values[:,0], y_pred[:,0])
    LR_test_mse_2 = mean_squared_error(y_test.values[:,1], y_pred[:,1])

# Calculating the mean absolute error of the predictions
    LR_test_mae_1 = mean_absolute_error(y_test.values[:,0], y_pred[:,0])
    LR_test_mae_2 = mean_absolute_error(y_test.values[:,1], y_pred[:,1])

# Calculating the root mean squared error of the predictions
    LR_test_rmse_1 = np.sqrt(LR_test_mse_1)
    LR_test_rmse_2 = np.sqrt(LR_test_mse_2)


    # print(r2_score(y_test.values[:,0], y_pred[:,0]))
    LR_test_Rsquared_1 = r2_score(y_test.values[:,0], y_pred[:,0])
    LR_test_Rsquared_2 = r2_score(y_test.values[:,1], y_pred[:,1])
    return [LR_test_mse_1,LR_test_mse_2,LR_test_mae_1,LR_test_mae_2,LR_test_rmse_1,LR_test_rmse_2, LR_test_Rsquared_1,LR_test_Rsquared_2]
