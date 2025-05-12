# models/nn.py
import torch
import torch.nn as nn
import torch.optim as optim
from evaluation import *
from models.lasso_regression import *
from models.ridge_regression import *
from models.neural_network import *
from models.xgboost import *

class FeedForwardNN(nn.Module):
    def __init__(self, input_size,hidden_dims1=64,hidden_dims2=32, dropout_rate=0.4, num_layers=3):
        super(FeedForwardNN, self).__init__()

        layers =  []
        # Input Layers
        layers.append(nn.Linear(input_size,hidden_dims1))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        # Number of Hidden layers (-1 as we have already added an input layer)
        current_dim = hidden_dims1
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim,hidden_dims2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim=hidden_dims2

        # Output Layer
        layers.append(nn.Linear(current_dim,2))
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        return self.model(x)
            
    #     self.fc1 = nn.Linear(input_size, hidden_dims1)
    #     self.dropout1 = nn.Dropout(dropout_rate) # Dropout layer after first hidden layer
    #     self.fc2 = nn.Linear(hidden_dims1, hidden_dims2)
    #     self.dropout2 = nn.Dropout(dropout_rate) # Dropout layer after second hidden layer
    #     self.fc3 = nn.Linear(hidden_dims2, 2) # As we have 2 output nodes
    #     self.relu = nn.ReLU()

    # def forward(self, x):
    #     x = self.relu(self.fc1(x))
    #     x = self.dropout1(x) # Applying dropout
    #     x = self.relu(self.fc2(x))
    #     x = self.dropout2(x) # Applying dropout
    #     x = self.fc3(x)
    #     return x


def nn_model(X_train, y_train, num_epochs,learning_rate,hidden_dims1=64,hidden_dims2=32, dropout_rate=0.4, num_layers=3):
    model = FeedForwardNN(X_train.shape[1],hidden_dims1,hidden_dims2, dropout_rate, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return model


def train_nn_model(X_train, y_train):
    # Converting to tensors
    X_train = torch.FloatTensor(X_train.values)
    y_train = torch.FloatTensor(y_train.values)


    learning_rates = [0.001, 0.01, 0.1]
    num_epochs_list = [50, 100, 500, 1000]
    # weight_decays = [0, 0.0001]
    dropout_rates = [0.2, 0.4, 0.5, 0.7]
    hidden_dims1 = [64, 128, 256]  # Larger dimensions for first hidden layer
    hidden_dims2 = [32, 64, 128]   # Smaller or equal dimensions for subsequent layers
    num_layers_list = [3, 4]    # Minimum 2 layers (input + output)
    
    #h_dim , 8,16,32,64
    # one hidden layer only
    # l1 regularisation check here
    # Split into Train valid and Test (hold out set)


    best_loss = float('inf')
    best_params = None
    best_model = None

    for lr in learning_rates:
        for epochs in num_epochs_list:
            # for wd in weight_decays:
            for dr in dropout_rates:
                for hd1 in hidden_dims1:
                   for hd2 in hidden_dims2:
                        for num_layers in num_layers_list:
                            print(f"Training with learning_rate={lr}, num_epochs={epochs}, dropout_rate={dr}, hidden_dim1={hd1}, hidden_dim2={hd2}, num_layers={num_layers}")
                            model = nn_model(X_train, y_train,epochs, lr, hidden_dims1=hd1,hidden_dims2=hd2,dropout_rate=dr, num_layers=num_layers)
                            # Evaluate model on training set
                            # X_tensor = torch.FloatTensor(X_train.values)
                            # y_tensor = torch.FloatTensor(y_train.values)
                            model.eval()
                            with torch.no_grad():
                                outputs = model(X_train)
                                loss = nn.MSELoss()(outputs, y_train).item()
                            print(f"Loss: {loss}")
                            if loss < best_loss:
                                best_loss = loss
                                best_params = {'learning_rate': lr, 'num_epochs': epochs, 'dropout_rate': dr, 'hidden_dim1': hd1, 'hidden_dim2': hd2, 'num_layers':num_layers}
                                best_model = model
    

    best_params["model"] = "nn"
        # Save best parameters to a JSON file
    with open("best_params.json", "a") as f:
        json.dump(best_params, f, indent=4)
        f.write("\n")

    print("Best hyperparameters:", best_params, "with loss:", best_loss)
    return best_model





