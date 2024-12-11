import sqlite3
import streamlit as st
import requests

class CredentialsManager:
    WATSON_REGIONS = {
        "Dallas (us-south)": "https://us-south.ml.cloud.ibm.com",
        "Frankfurt (eu-de)": "https://eu-de.ml.cloud.ibm.com",
        "London (eu-gb)": "https://eu-gb.ml.cloud.ibm.com",
        "Tokyo (jp-tok)": "https://jp-tok.ml.cloud.ibm.com",
        "Sydney (au-syd)": "https://au-syd.ml.cloud.ibm.com"
    }

    def __init__(self):
        self.conn = sqlite3.connect('apikeys.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('CREATE TABLE IF NOT EXISTS credentials (api_key TEXT, project_id TEXT)')

    def save_credentials(self, api_key, project_id):
        try:
            self.cursor.execute('DELETE FROM credentials')
            self.cursor.execute('INSERT INTO credentials (api_key, project_id) VALUES (?, ?)',
                              (api_key, project_id))
            self.conn.commit()
            return True
        except Exception as e:
            st.error(f"Error saving credentials: {str(e)}")
            return False

    def load_credentials(self):
        self.cursor.execute('SELECT api_key, project_id FROM credentials ORDER BY rowid DESC LIMIT 1')
        return self.cursor.fetchone()

    @staticmethod
    def get_iam_token(api_key):
        url = "https://iam.cloud.ibm.com/identity/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
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

    def verify_credentials(self, api_key, project_id, region_url):
        try:
            iam_token = self.get_iam_token(api_key)
            if not iam_token:
                return False

            url = f"{region_url}/ml/v1/text/generation?version=2023-05-29"
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
                "Authorization": f"Bearer {iam_token}"
            }

            response = requests.post(url, headers=headers, json=body)

            if response.status_code == 200:
                st.success("Successfully connected to IBM WatsonX! Model response received.")
                return True
            else:
                st.error(f"Failed to connect. Status code: {response.status_code}. Error: {response.text}")
                return False
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
            return False

    def fetch_supported_models(self, api_key, project_id, region_url):
        try:
            iam_token = self.get_iam_token(api_key)
            if not iam_token:
                return ["ibm/granite-13b-chat-v2"]

            url = f"{region_url}/ml/v1/foundation_model_specs?version=2024-01-01"
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {iam_token}"
            }

            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                models_data = response.json()
                models = [
                    model['model_id'] for model in models_data.get('resources', [])
                    if model.get('model_id')
                ]
                return models if models else ["ibm/granite-13b-chat-v2"]
            else:
                st.error(f"Failed to fetch models. Status code: {response.status_code}. Error: {response.text}")
                return ["ibm/granite-13b-chat-v2"]

        except Exception as e:
            st.error(f"Error fetching models: {str(e)}")
            return ["ibm/granite-13b-chat-v2"]

    def __del__(self):
        self.conn.close()