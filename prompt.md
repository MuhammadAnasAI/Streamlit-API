# Prompt


Hey Chat GPT, act as a application developer expert, in python using streamlit, and buit a machine learning application using scikit learn with following workflow:

1. Greet the user with welcome massage and breif about the description of application.
2. Ask the user to upload the dataset or use the example dataset
3. If the user select to upload the data, show the uploader section on the sidebar, upload the dataset in csv, xlsx, tsv or any possible data format.
4. If the user do not want to upload the data then provide the default dataset in the selection dataset on the sidebar. This box should download the dataset automatic from sns.load_dataset() function. The dataset include titanic, iris and tips.
5. Print the basic dataset information such as data head, data shape, data info, data description and column names
6. Ask the user to select the columns as feature and columns as target.
7. Identify the problem if the target column is continous or numeric print the message that the problem is Regression or print the message that the problem is Classification.
8. Preprocess the dataset, if any missing value in the dataset , dealing with the missing values using iterative imputer function of scikit learn, if the features are not in same scale , then scale the dataset using Standard Scaler function of scikit learn, if the features are categorical then encode the variables using Label Encoder function of scikit learn. Please keep in mind to keep encoder seperate for each column as we need to inverse transform the data at the end.
9. Ask the user to provide the train test split size via slider or user input function.
10. Ask the user to select the model from the sidebar, the model should include linear regression, Decision Tree, Random Forest and support vector machines and same classes of models for Classification problem.
11. Train the Model on the training data and Evaluate the model on test dataset.
12. If the problem is Regression problem then use the mean squared error, RMSE, MAE, AURO and r2 score for model evaluation, if the problem is classification then use the accuracy score, percision, recall ,f1 score and draw a confusion Matrix for evaluation
13. Print the Evaluation matrix for each models
14. Highlight the best model based on the evaluation matrix
15. Ask the user if he want to download the model , if yes then download the model in the pickle format.
16. Ask the user if he want to make the prediction, if yes then as the use to provide the input data using slider or upload file and make the predictions using the best model
17. Show the predictions to the user
