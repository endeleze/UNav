import torch.nn as nn

class TruncatedMagneticFieldModel_1(nn.Module):
    def __init__(self, n_inputs): # Note: n_outputs is no longer needed here
        super(TruncatedMagneticFieldModel_1, self).__init__()
        self.device = "cuda" if content["cuda"] else "cpu"
        self.fc1 = nn.Linear(n_inputs, 20)
        self.fc2 = nn.Linear(20, 5) # This will now be the last layer
        
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        # No fc3 layer, so no initialization for it

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # No fc3 layer, so return x directly after fc2
        return x

class MagneticFieldModel_1(TruncatedMagneticFieldModel_1):
    def __init__(self, n_inputs, n_outputs):
        super(MagneticFieldModel_1, self).__init__(n_inputs)
        # Output layer (5 neurons) to n_outputs
        self.fc3 = nn.Linear(5, n_outputs)

        nn.init.kaiming_uniform_(self.fc3.weight) # For the output layer, often no specific non-linearity is implied

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def load_truncated_state_dict(self, full_state_dict):
        new_state_dict = {}
        for k, v in full_state_dict.items():
            if not k.startswith('fc3.'): # Exclude keys that start with 'fc3.'
                new_state_dict[k] = v
        # Load the filtered state_dict into the truncated model
        # The strict=False argument allows for missing or unexpected keys,
        # but it's better to explicitly filter to avoid silent issues.
        try:
            self.super.load_state_dict(new_state_dict)
            self.super.eval().to(self.device)
        except RuntimeError as e:
            print(f"\nError loading state_dict (expected if strict=True and keys mismatch): {e}")
            print("This usually means there's a mismatch. Ensure you filtered correctly.")
            # If you get a RuntimeError here, it means 'strict=True' caught an issue.
            # Often, you'd use strict=False for minor mismatches, but explicit filtering is cleaner.
            # Let's try with strict=False to show it works after filtering.
            truncated_model.load_state_dict(new_state_dict, strict=False)
            print("State_dict loaded with strict=False (to demonstrate successful loading after filtering).")
    
    def fit(X_train, X_test, y_train, y_test, dataset_name, loss_func='mse', num_features=3, num_targets=2, epochs=10, validation_split=0.2):
        # get model
        # Define the loss function
        if loss_func == 'mae':
            criterion = nn.L1Loss()  # Mean Absolute Error
        elif loss_func == 'mse':
            criterion = nn.MSELoss() # Mean Squared Error (common default)
        else:
            raise ValueError(f"Unsupported loss function: {loss_func}. Choose 'mae' or 'mse'.")
        
        # Define the optimizer
        optimizer = optim.Adam(model.parameters())

        # fit the model on all data
        history = dict()
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        print("Starting PyTorch Model Training...")
        # 5. The Training Loop!
        for epoch in range(epochs):
            # 5a. Zero the gradients
            # Clear the gradients of all optimized torch.Tensor s.
            optimizer.zero_grad() 

            # 5b. Forward pass: compute predicted y by passing x to the model
            outputs = self(X_train_tensor)

            # 5c. Compute loss
            loss = criterion(outputs, y_train_tensor)
            history['loss'] = loss.detach().numpy()

            # 5d. Backward pass: compute gradient of the loss with respect to model parameters
            # This populates the .grad attribute for all parameters that have requires_grad=True
            loss.backward()

            # 5e. Update model parameters
            # Performs a single optimization step (parameter update)
            optimizer.step()
        
            # Print progress (optional)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        print("\nTraining Finished! ✅")

        self.eval()
        yhat = model(X_test_tensor)
        yhat_np = yhat.detach().numpy()
        mse = mean_squared_error(yhat_np, y_test)
        rmse = root_mean_squared_error(yhat_np, y_test) # Make sure yhat and y_test should be in original scale
        error_report(mse, y_test, dataset_name, metric='MSE')
        error_report(rmse, y_test, dataset_name, metric='RMSE')

        return history, yhat_np, rmse, self