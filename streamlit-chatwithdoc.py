"""
Required installations:
pip install streamlit
pip install ibm-watsonx-ai
pip install chromadb
pip install requests

Optional for PDF handling (if implementing later):
pip install pypdf
"""

import streamlit as st
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
import sqlite3
from langchain_core.prompts import PromptTemplate
import requests
import json

# Set page config for a cleaner look
st.set_page_config(page_title="IBM WatsonX Chat", layout="wide")

# Database setup
conn = sqlite3.connect('apikeys.db')
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS credentials (api_key TEXT, project_id TEXT)')

def save_credentials(api_key, project_id):
    try:
        c.execute('DELETE FROM credentials')  # Clear existing credentials
        c.execute('INSERT INTO credentials (api_key, project_id) VALUES (?, ?)', (api_key, project_id))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error saving credentials: {str(e)}")
        return False

# Add this after the imports
WATSON_REGIONS = {
    "Dallas (us-south)": "https://us-south.ml.cloud.ibm.com",
    "Frankfurt (eu-de)": "https://eu-de.ml.cloud.ibm.com",
    "London (eu-gb)": "https://eu-gb.ml.cloud.ibm.com",
    "Tokyo (jp-tok)": "https://jp-tok.ml.cloud.ibm.com",
    "Sydney (au-syd)": "https://au-syd.ml.cloud.ibm.com"
}

def get_iam_token(api_key):
    """Get IAM token using API key"""
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key
    }

    try:
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            st.error(f"Failed to get IAM token. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error getting IAM token: {str(e)}")
        return None

def verify_credentials(api_key, project_id, region_url):
    try:
        # First get IAM token
        iam_token = get_iam_token(api_key)
        if not iam_token:
            return False

        url = f"{region_url}/ml/v1/text/generation?version=2023-05-29"

        # Simple test prompt
        body = {
            "input": "<|start_of_role|>system<|end_of_role|>You are Granite, an AI language model.<|end_of_text|>",
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 10,
                "repetition_penalty": 1
            },
            "model_id": "ibm/granite-3-8b-instruct",
            "project_id": project_id
        }

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {iam_token}"  # Use IAM token instead of API key
        }

        response = requests.post(
            url,
            headers=headers,
            json=body
        )

        if response.status_code == 200:
            st.success(f"Successfully connected to IBM WatsonX! Model response received.")
            return True
        else:
            st.error(f"Failed to connect. Status code: {response.status_code}. Error: {response.text}")
            return False
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return False

def fetch_supported_models(api_key, project_id, region_url):
    """Fetch available foundation models from WatsonX"""
    try:
        # First get IAM token
        iam_token = get_iam_token(api_key)
        if not iam_token:
            return ["ibm/granite-3-8b-instruct"]  # Return default model if token fetch fails

        url = f"{region_url}/ml/v1/foundation_model_specs?version=2024-01-01"

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {iam_token}"
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            models_data = response.json()
            # Extract model IDs from the response
            models = [
                model['model_id'] for model in models_data.get('resources', [])
                if model.get('model_id')  # Only include if model_id exists
            ]
            return models if models else ["ibm/granite-13b-chat-v2"]
        else:
            st.error(f"Failed to fetch models. Status code: {response.status_code}. Error: {response.text}")
            return ["ibm/granite-13b-chat-v2"]

    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return ["ibm/granite-13b-chat-v2"]

def load_credentials():
    c.execute('SELECT api_key, project_id FROM credentials ORDER BY rowid DESC LIMIT 1')
    return c.fetchone()

# Sidebar for credentials management
with st.sidebar:
    st.title("📝 Credentials Management")

    # Region selection
    selected_region = st.selectbox(
        "Select Region",
        list(WATSON_REGIONS.keys()),
        index=0
    )
    region_url = WATSON_REGIONS[selected_region]

    # Load existing credentials
    existing_credentials = load_credentials()
    if existing_credentials:
        existing_api_key, existing_project_id = existing_credentials
    else:
        existing_api_key, existing_project_id = "", ""
        st.warning("No credentials found. Please enter your API Key and Project ID.")

    # Credentials input
    api_key = st.text_input("API Key", value=existing_api_key, type="password")
    project_id = st.text_input("Project ID", value=existing_project_id, type="password")

    show_credentials = st.checkbox("Show Credentials")
    if show_credentials:
        st.text_input("API Key (visible)", value=api_key, disabled=True)
        st.text_input("Project ID (visible)", value=project_id, disabled=True)

    # Separate columns for Save and Verify buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Save Credentials"):
            if api_key and project_id:
                if save_credentials(api_key, project_id):
                    st.success("Credentials saved successfully!")
            else:
                st.error("Please enter both API Key and Project ID")

    with col2:
        if st.button("Verify Connection"):
            if api_key and project_id:
                verify_credentials(api_key, project_id, region_url)
            else:
                st.error("Please enter both API Key and Project ID")

# Main chat interface
st.title("🤖 IBM WatsonX Chat")

if existing_credentials:
    # Model selection and parameters
    col1, col2 = st.columns(2)
    with col1:
        supported_models = fetch_supported_models(existing_api_key, existing_project_id, region_url)
        selected_model = st.selectbox(
            "Select Model",
            supported_models,
            index=supported_models.index("ibm/granite-3-8b-instruct") if "ibm/granite-3-8b-instruct" in supported_models else 0
        )

    with col2:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

    # Advanced parameters in expander
    with st.expander("Advanced Parameters"):
        col3, col4 = st.columns(2)
        with col3:
            max_tokens = st.number_input("Max Tokens", 1, 2048, 150)
            frequency_penalty = st.slider("Frequency Penalty", 0.0, 2.0, 0.0, 0.1)
        with col4:
            top_p = st.slider("Top P", 0.0, 1.0, 1.0, 0.1)
            presence_penalty = st.slider("Presence Penalty", 0.0, 2.0, 0.0, 0.1)

    # Chat interface
    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Initialize chat with parameters
        with st.chat_message("assistant"):
            try:
                # Initialize credentials
                credentials = Credentials(
                    api_key=existing_api_key,
                    url=region_url
                )

                # Initialize WatsonX model with credentials object
                model = Model(
                    model_id=selected_model,
                    credentials=credentials,
                    project_id=existing_project_id,
                    params={
                        "temperature": temperature,
                        "max_new_tokens": max_tokens,
                        "top_p": top_p,
                        "frequency_penalty": frequency_penalty,
                        "presence_penalty": presence_penalty
                    }
                )

                # Format the prompt
                formatted_prompt = f"""System: You are a helpful assistant.
User: {prompt}
Assistant:"""

                # Generate response
                response = model.generate_text(formatted_prompt)

                # Display response
                st.markdown(response)

                # Save to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                error_msg = str(e)
                if "Request failed with:" in error_msg:
                    try:
                        # Extract the JSON part from the error message
                        json_str = error_msg.split("Request failed with: ")[1].split(" (400)")[0]
                        error_data = json.loads(json_str)

                        # Get the actual error message
                        if "errors" in error_data and len(error_data["errors"]) > 0:
                            error_msg = error_data["errors"][0]["message"]
                    except:
                        # If parsing fails, keep original error message
                        pass

                # Display user-friendly error
                st.error(f"⚠️ {error_msg}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"_Sorry, I encountered an error: {error_msg}_"
                })

else:
    st.info("Please enter your credentials in the sidebar to start chatting.")



