#Import the libraries:
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#Title of app:
st.title("Exploratory Data Analysis App")
st.subheader("A simple EDA App created by Muhammad Anas Ahmadani")

#Load the dataset:
dataset_options = ['cancer.csv', 'pride_index_tags.csv', 'Student_performance_data_.csv', 'titanic.csv']
selected_dataset = st.sidebar.selectbox("Select a dataset", dataset_options)

#Load the Selected datasets:
if selected_dataset == 'cancer.csv':
    df = pd.read_csv('cancer.csv')
elif selected_dataset == 'pride_index_tags.csv':
    df = pd.read_csv('pride_index_tags.csv')
elif selected_dataset == 'Student_performance_data_.csv':
    df = pd.read_csv('Student_performance_data_.csv')
elif selected_dataset == 'titanic.csv':
    df = pd.read_csv('titanic.csv')        

#Button to load the Custom dataset:
uploaded_file = st.sidebar.file_uploader('Uploaded Custom dataset', type=['csv', 'xlxs' ])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

#Display the dataset:
st.write(df)    

#Number of rows and columns in the selected dataset:
st.write(f"Number of rows in the dataset: {df.shape[0]}")
st.write(f"Number of columns in the dataset: {df.shape[1]}")

#Write the name of columns and its data type:
st.write("Name of columns and its data type:", df.dtypes)

#Print the null if those are greater > 0:
if df.isnull().sum().sum() > 0:
    st.write("Number of missing values in the dataset: ", df.isnull().sum().sort_values(ascending=False))
else:
    st.write("There are no missing values in the dataset")

#Summary of Statistical Dataset:
st.write("Summary Statistics: ",df.describe())

#Create a pairplot of datasets:
st.subheader('Pairplot of Datasets')
hue_column = st.selectbox("Select the column as hue:", df.columns)
st.pyplot(sns.pairplot(df, hue=hue_column))

#Create a heatmap of Dataset:
st.subheader('Heatmap of Datasets')
#Only set the numeric columns of dataset for heatamp plot:
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = df[numeric_columns].corr()
from plotly import graph_objects as go
heatmap_fig = go.Figure(data=go.Heatmap(
    z = corr_matrix.values,
    x = corr_matrix.columns,
    y = corr_matrix.columns,
    colorscale = 'Viridis',
    showscale = True
))
st.plotly_chart(heatmap_fig)





