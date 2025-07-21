# Speech to Text App creation using Streamlt:
#Import the libraries:
import streamlit as st
from openai import OpenAI
import tempfile
import os
# Create a Streamlit app
#Create a sidebar for open_ai_key:
st.sidebar.title("Setting")
api_key = st.sidebar.text_input("Open AI Key", type="password")
#Create Main app title:
st.title("Speech-to-Text Whisper Model")
#Upload the Audio File:
audio_file = st.file_uploader("Upload a Audio file", type=['mp3', 'wav'])
#Inatalize the openai key for the client:
client = OpenAI(api_key=api_key)
if audio_file is not None and api_key:
    #Create a temporary file for the uploaded audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.' + audio_file.name.split('.')[1]) as tmp_file:

    #Save the uploaded audio file to the temporary file
        tmp_file.write(audio_file.getvalue())
        tmp_file_path = tmp_file.name
    try:
        with open(tmp_file_path, "rb") as audio_file:
            transcription_reponse = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
            )
            #Assign the Transcriptions files directly:
            transcription_text = transcription_reponse.result.file
            #Display the transcription file in the app
            st.write("Transcriptions: ",transcription_text)
    except Exception as e:
        st.write("Error: ", e)
    finally:
        #Remove the temporary file
        os.remove(tmp_file_path)



 