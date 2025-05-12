# scripts/train_model.py
from preprocessing import *
from models.linear_regression import *
from evaluation import *
import pyreadr
import pandas as pd
from models.lasso_regression import train_lasso_model
from models.ridge_regression import train_ridge_model
from models.neural_network import train_nn_model
from models.xgboost import train_xgboost_model
from datetime import datetime
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress ConvergenceWarning from sklearn
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Your code
only_tree_features = False
clinicaldata_with_tree_features = True

def data_checks():

    clinical_df = pyreadr.read_r("data/20221109_TRACERx421_all_patient_df.rds")
    df = pd.DataFrame(clinical_df[None])
    len(df)

    new_df = pd.read_csv("data/tree_features.csv")
    new_df = new_df.rename(columns = {'Unnamed: 0':'cruk_id'})
    # print(len(new_df))
    new_df.columns


    if clinicaldata_with_tree_features:
        df = pd.merge(df,new_df, on = 'cruk_id')
        print('Clinical Data and Tree features')

    elif only_tree_features:
        df = pd.merge(df[['cruk_id','os_time','dfs_time']],new_df, on = 'cruk_id')
        print('Only Tree features')
    else:
        print('Only Clinical Data')

    return df

#in process
# def train_models(X_train, X_test, y_train, y_test):
#     model = [train_linear_model]
#     for i in range (0,4):
# # Train and evaluate model
#         model = train_linear_model(X_train, y_train)
#         mse, y_pred = evaluate_linear_model(model, X_train, y_train, X_test, y_test)


def main():
    # Load and preprocess data
    df = data_checks()
    df_clean = clean_data(df, only_tree_features)
    df_dropped = drop_columns(df_clean, only_tree_features)
    df_encoded = encode_data(df_dropped, only_tree_features)

    X_train, X_test, y_train, y_test = split_data(df_encoded, ['os_time','dfs_time'])
    
    # Dictionary to store models and their names
    models = {
        'Linear Regression_EN': train_linear_model(X_train, y_train, X_test, y_test),
        'Lasso Regression': train_lasso_model(X_train, y_train),
        'Ridge Regression': train_ridge_model(X_train, y_train),
        'Neural Network': train_nn_model(X_train, y_train),
        'XGBoost': train_xgboost_model(X_train, y_train)
    }
    
    # Dictionary to store results
    results = {}
    
    # Evaluate each model
    for model_name, model in models.items():
        model_type = "nn" if model_name == "Neural Network" else "all"
        train_metrics, test_metrics = evaluate_model(model, X_train, y_train, X_test, y_test, model_type)
        
        results[model_name] = {
            'Training Metrics': {
                'OS Time': {
                    'MSE': train_metrics[0],
                    'RMSE': train_metrics[4],
                    'MAE': train_metrics[2],
                    'R2': train_metrics[6]  # Add R-squared for OS Time
                },
                'DFS Time': {
                    'MSE': train_metrics[1],
                    'RMSE': train_metrics[5],
                    'MAE': train_metrics[3],
                    'R2': train_metrics[7]  # Add R-squared for DFS Time
                }
            },
            'Testing Metrics': {
                'OS Time': {
                    'MSE': test_metrics[0],
                    'RMSE': test_metrics[4],
                    'MAE': test_metrics[2],
                    'R2': test_metrics[6]  # Add R-squared for OS Time
                },
                'DFS Time': {
                    'MSE': test_metrics[1],
                    'RMSE': test_metrics[5],
                    'MAE': test_metrics[3],
                    'R2': test_metrics[7]  # Add R-squared for DFS Time
                }
            }
        }

    # Print results in a structured format
    for model_name, metrics in results.items():
        print('\n' + '='*50)
        print(f'\nResults for {model_name}:')
        print('\nTraining Metrics:')
        print('-----------------')
        for time_type, measures in metrics['Training Metrics'].items():
            print(f'\n{time_type}:')
            for metric_name, value in measures.items():
                print(f'{metric_name}: {value:.4f}')
        
        print('\nTesting Metrics:')
        print('----------------')
        for time_type, measures in metrics['Testing Metrics'].items():
            print(f'\n{time_type}:')
            for metric_name, value in measures.items():
                print(f'{metric_name}: {value:.4f}')
    
    # Optionally, save results to a CSV file
    save_results_to_csv(results)

def save_results_to_csv(results):
    """Save the results to a CSV file"""
    # Create lists to store the data
    rows = []
    for model_name, metrics in results.items():
        for split in ['Training Metrics', 'Testing Metrics']:
            for time_type in ['OS Time', 'DFS Time']:
                row = {
                    'Model': model_name,
                    'Split': split,
                    'Time Type': time_type,
                    'MSE': metrics[split][time_type]['MSE'],
                    'MAE': metrics[split][time_type]['MAE'],
                    'RMSE': metrics[split][time_type]['RMSE'],
                    'R2': metrics[split][time_type]['R2']  # Add R2 to CSV output
                }
                rows.append(row)
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(rows)
    # results_df.to_csv('model_evaluation_results.csv', index=False)
    # print('\nResults saved to model_evaluation_results.csv')

    # Create OS Time DataFrame
    os_data = []
    for model_name, metrics in results.items():
        os_row = {
            'Model': model_name,
            'Train_MSE': metrics['Training Metrics']['OS Time']['MSE'],
            'Test_MSE': metrics['Testing Metrics']['OS Time']['MSE'],
            'Train_MAE': metrics['Training Metrics']['OS Time']['MAE'],
            'Test_MAE': metrics['Testing Metrics']['OS Time']['MAE'],
            'Train_RMSE': metrics['Training Metrics']['OS Time']['RMSE'],
            'Test_RMSE': metrics['Testing Metrics']['OS Time']['RMSE'],
            'Train_R2': metrics['Training Metrics']['OS Time']['R2'],
            'Test_R2': metrics['Testing Metrics']['OS Time']['R2']
        }
        os_data.append(os_row)
    
    # Create DFS Time DataFrame
    dfs_data = []
    for model_name, metrics in results.items():
        dfs_row = {
            'Model': model_name,
            'Train_MSE': metrics['Training Metrics']['DFS Time']['MSE'],
            'Test_MSE': metrics['Testing Metrics']['DFS Time']['MSE'],
            'Train_MAE': metrics['Training Metrics']['DFS Time']['MAE'],
            'Test_MAE': metrics['Testing Metrics']['DFS Time']['MAE'],
            'Train_RMSE': metrics['Training Metrics']['DFS Time']['RMSE'],
            'Test_RMSE': metrics['Testing Metrics']['DFS Time']['RMSE'],
            'Train_R2': metrics['Training Metrics']['DFS Time']['R2'],
            'Test_R2': metrics['Testing Metrics']['DFS Time']['R2']
        }
        dfs_data.append(dfs_row)
    
    os_df = pd.DataFrame(os_data)
    dfs_df = pd.DataFrame(dfs_data)
    
    # Round values
    numeric_cols = os_df.columns.drop('Model')
    os_df[numeric_cols] = os_df[numeric_cols].round(4)
    dfs_df[numeric_cols] = dfs_df[numeric_cols].round(4)
    
    # Save both tables to single file with separator
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'model_results_{timestamp}.csv'
    
    with open(filename, 'w') as f:
        f.write("OS Time Results\n")
        os_df.to_csv(f, index=False)
        f.write("\n\nDFS Time Results\n")
        dfs_df.to_csv(f, index=False)
    
    print(f"\nResults saved to {filename}")
    
if __name__ == "__main__":
    start_time=datetime.now()
    main()
    end_time=datetime.now()

    execution_time = end_time - start_time
    print(f"Execution Time : {execution_time.total_seconds():.3f} seconds")


