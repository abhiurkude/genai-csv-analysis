import streamlit as st
import pandas as pd
import openai
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

ai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
ai_api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
ai_api_version = os.getenv("OPENAI_API_VERSION")

client = AzureOpenAI(
    api_key = ai_api_key,  
    api_version = ai_api_version,
    azure_endpoint = ai_api_base
)

# App title
st.set_page_config(page_title="CSV Document Analyzer", layout="centered")
st.title("📊 CSV Document Analyzer with Azure OpenAI")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV successfully uploaded and previewed below.")
    st.dataframe(df.head(df.shape[0]))

    # Ask a question
    question = st.text_input("Ask a question about the data:")

    if question:
        # Convert DataFrame to string (sample or full)
        df_sample = df.head(df.shape[0]).to_csv(index=False)

        # Prompt construction
        prompt = f"""
You are a data analyst assistant. Analyze the following CSV data and answer the user's question.

CSV Data:
{df_sample}

User Question: {question}
"""

        # Call Azure OpenAI
        try:
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
            #answer = response['choices'][0]['message']['content']
            st.subheader("🧠 Answer")
            st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"Error communicating with Azure OpenAI: {e}")
