from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
import streamlit as st

class WatsonXManager:
    def __init__(self, api_key, project_id, region_url):
        self.credentials = Credentials(
            api_key=api_key,
            url=region_url
        )
        self.project_id = project_id

    def generate_response(self, prompt, model_id, params):
        try:
            # Initialize model
            model = ModelInference(
                model_id=model_id,
                credentials=self.credentials,
                project_id=self.project_id
            )

            # Debug log
            st.sidebar.write(f"Debug - Max Tokens Setting: {params.get('max_tokens', 300)}")

            # Set generation parameters according to IBM documentation
            generation_params = {
                GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
                GenParams.MAX_NEW_TOKENS: int(params.get('max_tokens', 300)),
                GenParams.MIN_NEW_TOKENS: 1,
                GenParams.TEMPERATURE: float(params.get('temperature', 0.7)),
                GenParams.TOP_P: float(params.get('top_p', 1.0)),
                GenParams.TOP_K: int(params.get('top_k', 50)),
                GenParams.REPETITION_PENALTY: float(params.get('repetition_penalty', 1.0)),
            }

            # Set parameters and log them
            st.sidebar.write("Debug - Model Parameters:", generation_params)

            # Generate response with proper prompt formatting
            formatted_prompt = prompt.strip()
            response = model.generate(prompt=formatted_prompt, params=generation_params)

            # Debug log the raw response
            st.sidebar.write("Debug - Raw Response:", response)

            # Process response
            if isinstance(response, dict) and 'results' in response:
                generated_text = response['results'][0]['generated_text'].strip()

                # Debug log the generated text
                st.sidebar.write(f"Debug - Generated Text Length: {len(generated_text)}")

                return {
                    'text': generated_text,
                    'generated_tokens': response['results'][0].get('generated_token_count', 0),
                    'input_tokens': response['results'][0].get('input_token_count', 0),
                    'warnings': response.get('system', {}).get('warnings', [])
                }
            else:
                st.warning("Unexpected response format from the model")
                return {
                    'text': str(response),
                    'generated_tokens': 0,
                    'input_tokens': 0,
                    'warnings': []
                }

        except Exception as e:
            st.error(f"Full error: {str(e)}")
            st.error(f"Error type: {type(e)}")
            raise Exception(f"Error generating response: {str(e)}")