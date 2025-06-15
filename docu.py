import streamlit as st
import pandas as pd
import openai
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# This will parse a .env file and then load all the variables found as environment variables.
load_dotenv()

# Reading azure openai api key, openai endpoint and openai api version values from .env file to local variables.
# These values will be used to connect to azure openai.
ai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
ai_api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
ai_api_version = os.getenv("OPENAI_API_VERSION")

# Constructing a new synchronous azure openai client instance. This takes parameters api key, api version and api endpoint read from the environment file.
client = AzureOpenAI(
    api_key = ai_api_key,  
    api_version = ai_api_version,
    azure_endpoint = ai_api_base
)

# Configures the default settings of the page. Sets page title and layout.
st.set_page_config(page_title="CSV Document Analyzer", layout="centered")
st.title("📊 CSV Document Analyzer with Azure OpenAI")

# This code display a file uploader widget and uploaded files are limited to 200MB.
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Check if a file is uploaded
if uploaded_file is not None:
    # This code reads the CSV file content and is assigned to variable df.
    df = pd.read_csv(uploaded_file)
    st.success("CSV successfully uploaded and previewed below.")
    # Display a dataframe as an interactive table. This table will contain data from CSV file.
    st.dataframe(df.head(df.shape[0]))

    # Display a single-line text input widget. In this text input user can type in question related to data from CSV file.
    question = st.text_input("Ask a question about the data:")

    # Check if a question is provided
    if question:
        # This code will take dataframe data and convert to csv format and assign to variable df_sample.
        df_sample = df.head(df.shape[0]).to_csv(index=False)

        # Prompt construction. This prompt tells openai to be a data analyst assistant and it needs to answer user questions based on the CSV file data. Prompt will have csv file data and user question.
        prompt = f"""
You are a data analyst assistant. Analyze the following CSV data and answer the user's question.

CSV Data:
{df_sample}

User Question: {question}
"""

        # Call Azure OpenAI
        try:
            # Make the API call to Azure OpenAI. This code creates a model response for the given chat conversation. Here azure openai object is used to call the chat completion create method.
            # To this method pass model to be used, message which contains the prompt, temperature, max tokens, top and frequency penalty values.
            response = client.chat.completions.create(
                model="gpt-4o-mini", #Allowed values for ApiUser: gpt-4o,gpt-4o-mini,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for data analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=256,
                top_p= 0.6,
                frequency_penalty= 0.7
            )
            
            # Display text in subheader formatting.
            st.subheader("🧠 Answer")
            # Display the response from Azure OpenAI based on the user query. Response object has choices array. In this message content has the answer to user query which is read and written to UI using the write method of streamlit object.
            st.write(response.choices[0].message.content)

        # Handle errors in API call. In case of any error, it is handeled here.
        except Exception as e:
            # Error is capture in the exception object e and is written to UI using the error method of streamlit object.
            st.error(f"Error communicating with Azure OpenAI: {e}")
