import streamlit as st

#Import the address file
st.title("Anas Khan Ahmadani AI Course")

#Add a new text:
st.write("Welcome to my AI course!")

# User input:
number = st.slider('Pick a number', 0, 100,  10)

#Print the text number:
st.write(f"You picked {number}")

#Adding a button:
if st.button("Click me"):
    st.write("Hi, Hello dear!")
else:
    st.write("Good Bye")  
# Add Radio Button with Options:
genre = st.radio(
    "What is your favorite movie genre?",
    ("Drama", "Comdy", "Documentary"))
#Print the genre with text:
st.write(f"You like {genre} movies")

#Add a drop down list with options:
#option = st.selectbox('How would like to contact with us?', ['Phone', 'Email',
#                                           'Website'])

#Add the drop down list with left side bar:
option = st.sidebar.selectbox('How would like to contact with us?', ['Phone', 'Email',
                                                            'Home'])

#Add whatts app number:
st.sidebar.text_input("Enter your whattapp number!")

#Upload the files:
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "txt", "csv", "xls"])

#Create a line plot with data ploting with random dataset:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
data = pd.DataFrame({
    'first column' : list(range(1, 11)),
    'second column' : np.arange(number, number + 10)
})
st.line_chart(data)


      