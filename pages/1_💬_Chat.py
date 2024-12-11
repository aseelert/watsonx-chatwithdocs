import streamlit as st
from utils.credentials_helpers import CredentialsManager
from utils.watsonx_helpers import WatsonXManager
from utils.chroma_helpers import ChromaManager

st.set_page_config(page_title="Chat", layout="wide")
st.title("🤖 watsonx chatter")

# Initialize managers
credentials_manager = CredentialsManager()
chroma_manager = ChromaManager()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "use_rag" not in st.session_state:
    st.session_state.use_rag = False

# Load credentials
credentials = credentials_manager.load_credentials()
if not credentials:
    st.error("Please set up your credentials in the Credentials page first.")
    st.stop()

existing_api_key, existing_project_id = credentials

# Sidebar settings
with st.sidebar:
    st.title("⚙️ Chat Settings")

    # Region selection
    selected_region = st.selectbox(
        "Select Region",
        list(CredentialsManager.WATSON_REGIONS.keys()),
        index=0
    )
    region_url = CredentialsManager.WATSON_REGIONS[selected_region]

    # Model selection
    supported_models = credentials_manager.fetch_supported_models(
        existing_api_key,
        existing_project_id,
        region_url
    )
    selected_model = st.selectbox(
        "Select Model",
        supported_models,
        index=supported_models.index("ibm/granite-3-8b-instruct")
        if "ibm/granite-3-8b-instruct" in supported_models else 0
    )

    # Model parameters
    st.subheader("Model Parameters")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

    with st.expander("Advanced Parameters"):
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4096,
            value=400,
            help="Maximum number of tokens to generate in the response"
        )
        top_p = st.radio(
            "Top P",
            options=[0.0, 1.0],
            index=1,
            horizontal=True,
            help="0 = Disabled, 1 = Enabled"
        )
        top_k = st.slider(
            "Top K",
            min_value=0,
            max_value=100,
            value=50,
            step=10,
            help="Number of tokens to consider for sampling (0-100)"
        )
        repetition_penalty = st.radio(
            "Repetition Penalty",
            options=[1.0, 2.0],
            index=0,
            horizontal=True,
            help="1 = Normal, 2 = More penalty for repetition"
        )

    # RAG settings
    st.divider()
    st.subheader("RAG Settings")
    use_rag = st.toggle(
        "Use Document Knowledge",
        value=st.session_state.use_rag,
        help="Enable to use knowledge from uploaded documents"
    )
    st.session_state.use_rag = use_rag

# Initialize WatsonX manager
watsonx_manager = WatsonXManager(existing_api_key, existing_project_id, region_url)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?", key="chat_input"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        try:
            # Prepare prompt with RAG if enabled
            if use_rag and chroma_manager.vectorstore:
                relevant_docs = chroma_manager.vectorstore.similarity_search(prompt, k=3)
                context = "\n".join([doc.page_content for doc in relevant_docs])
                formatted_prompt = f"""System: You are a helpful assistant. Use the following context to answer the question.
Context: {context}

User: {prompt}
Assistant:"""
            else:
                formatted_prompt = f"""System: You are a helpful assistant.
User: {prompt}
Assistant:"""

            # Generate response with explicit parameters
            response = watsonx_manager.generate_response(
                prompt=formatted_prompt,
                model_id=selected_model,
                params={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty
                }
            )

            # Display response
            st.markdown(response['text'])

            # Display token counts
            st.caption(
                f"📊 Tokens - Generated: {response['generated_tokens']}, "
                f"Input: {response['input_tokens']}"
            )

            # Save to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"{response['text']}\n\n_📊 Tokens - "
                f"Generated: {response['generated_tokens']}, "
                f"Input: {response['input_tokens']}_"
            })

            # Show relevant warnings
            if response['warnings']:
                relevant_warnings = [
                    w for w in response['warnings']
                    if 'max_new_tokens' not in w.get('message', '')
                ]
                if relevant_warnings:
                    with st.expander("System Warnings"):
                        for warning in relevant_warnings:
                            st.warning(warning['message'])

        except Exception as e:
            error_msg = str(e)
            st.error(f"⚠️ {error_msg}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"_Sorry, I encountered an error: {error_msg}_"
            })