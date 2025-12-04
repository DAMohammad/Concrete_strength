# Concrete Compressive Strength Prediction with GUI

## Project Description
The goal of this project is to predict the compressive strength of concrete using the UCI Concrete dataset and provide a graphical user interface for easy prediction.

## Project Steps

1. Reading the dataset
- I imported the Concrete dataset into the program and specified the columns with shorter headers to make it easier to work with.

2. Data preprocessing
- Removing outliers
- Scaling features for uniformity of scales

3. Modeling
- Using Linear Regression to predict concrete strength
- Training the model on prepared data

4. Model evaluation
- Calculating MSE: 40.179
- Calculating RMSE: 6.338 → This means that the model has an average error of about 6.33 MPa

5. Saving the model
- I saved the trained model using joblib in the concrete_model.pkl file so that it can be loaded into the program

6. User interface (GUI)
- Creating a user interface with Tkinter
- The user can enter the values ​​of the material and age of the concrete and click on Predict to see the predicted strength of the concrete