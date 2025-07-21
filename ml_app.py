import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import numpy as np
import pickle

# Greet the user
st.title("Welcome to the Machine Learning Application")
st.write("This application allows you to upload a dataset or use an example dataset to train and evaluate machine learning models using Scikit-Learn.")

# Sidebar for dataset selection
st.sidebar.title("Dataset Selection")
data_source = st.sidebar.radio("Choose the data source:", ("Upload your own dataset", "Use an example dataset"))

# Upload dataset section
df = None
if data_source == "Upload your own dataset":
    uploaded_file = st.sidebar.file_uploader("Upload a dataset", type=["csv", "xlsx", "tsv"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.tsv'):
            df = pd.read_csv(uploaded_file, sep='\t')
else:
    example_data = st.sidebar.selectbox("Choose an example dataset:", ("titanic", "iris", "tips"))
    df = sns.load_dataset(example_data)

# Check if the dataset is loaded
if df is not None:
    st.write("## Dataset Information")
    st.write("### First 5 rows of the dataset")
    st.write(df.head())
    st.write("### Dataset Shape")
    st.write(df.shape)
    st.write("### Column Names")
    st.write(df.columns.tolist())
    st.write("### Describe the Dataset")
    st.write(df.describe())

    # Ask the user to select feature columns and target column
    st.sidebar.write("### Feature and Target Selection")
    all_columns = df.columns.tolist()
    feature_columns = st.sidebar.multiselect("Select feature columns:", all_columns)
    target_column = st.sidebar.selectbox("Select the target column:", all_columns)

    # Ask the user to specify the problem type
    problem_type = st.sidebar.selectbox("Select problem type:", ("Regression", "Classification"))

    # Button to run the analysis
    run_analysis = st.sidebar.button("Run Analysis")

    if run_analysis and feature_columns and target_column:
        X = df[feature_columns]
        y = df[target_column]

        # Preprocessing the dataset
        st.write("## Data Preprocessing")

        # Handling missing values
        imputer = IterativeImputer()
        X = pd.DataFrame(imputer.fit_transform(X), columns=feature_columns)
        st.write("Missing values handled using Iterative Imputer.")

        # Identify categorical and numerical columns
        categorical_columns = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Set up ColumnTransformer for encoding categorical data and scaling numerical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_columns),
                ('cat', OneHotEncoder(), categorical_columns)
            ]
        )

        # Select appropriate models based on the problem type
        if problem_type == "Regression":
            model_choice = st.sidebar.selectbox("Select a regression model:", 
                                                ("Linear Regression", "Decision Tree", "Random Forest", "Support Vector Machine"))
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Support Vector Machine": SVR()
            }
        else:
            model_choice = st.sidebar.selectbox("Select a classification model:", 
                                                ("Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine"))
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Support Vector Machine": SVC()
            }
        
        model = models[model_choice]

        # Create a pipeline that preprocesses data and then applies the model
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

        # Train-test split
        train_size = st.sidebar.slider("Train-test split ratio:", 0.1, 0.9, 0.8)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
        st.write(f"Train-test split: {train_size*100}% train, {(1-train_size)*100}% test.")

        # Train the model
        clf.fit(X_train, y_train)
        st.write(f"Model trained using **{model_choice}**.")

        # Evaluate the model
        y_pred = clf.predict(X_test)
        st.write("## Model Evaluation")

        if problem_type == "Regression":
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse}")
            st.write(f"Root Mean Squared Error: {rmse}")
            st.write(f"Mean Absolute Error: {mae}")
            st.write(f"R2 Score: {r2}")
        else:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            st.write(f"Accuracy: {accuracy}")
            st.write(f"Precision: {precision}")
            st.write(f"Recall: {recall}")
            st.write(f"F1 Score: {f1}")
            st.write("Confusion Matrix:")
            st.write(cm)

        # Model download option
        download_model = st.sidebar.checkbox("Download model?")
        if download_model:
            with open("best_model.pkl", "wb") as f:
                pickle.dump(clf, f)
            st.sidebar.write("Model saved as `best_model.pkl`.")

        # Prediction option
        st.write("## Make Predictions")
        make_prediction = st.sidebar.checkbox("Make a prediction?")
        if make_prediction:
            st.write("Provide input data for prediction:")
            input_data = {}
            for col in feature_columns:
                if col in categorical_columns:
                    input_data[col] = st.sidebar.selectbox(f"Select {col}", options=df[col].unique())
                else:
                    input_data[col] = st.sidebar.number_input(f"Enter {col}", value=0.0)
            input_df = pd.DataFrame([input_data])
            prediction = clf.predict(input_df)
            st.write(f"The predicted value is: {prediction[0]}")
