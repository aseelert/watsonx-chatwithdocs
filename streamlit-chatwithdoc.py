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
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
import sqlite3
from langchain_core.prompts import PromptTemplate
import requests
import json
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredFileLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import tempfile
import os
import time  # Add this to the imports section

# Move these functions to the top, after imports but before any usage
def initialize_chroma():
    """Initialize ChromaDB with HuggingFace embeddings"""
    # Create directory if it doesn't exist
    persist_directory = "./chroma_db"
    os.makedirs(persist_directory, exist_ok=True)

    # Initialize embeddings with explicit cache directory
    cache_dir = "./models_cache"
    os.makedirs(cache_dir, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        cache_folder=cache_dir,
        model_kwargs={'device': 'cpu'}
    )

    try:
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        vectorstore.persist()  # Make sure to persist after initialization
        return vectorstore
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {str(e)}")
        return None

def get_document_loader(file_path, file_type):
    """Return appropriate loader based on file type"""
    if file_type == "pdf":
        return PyPDFLoader(file_path)
    elif file_type == "csv":
        return CSVLoader(file_path)
    elif file_type == "txt":
        return TextLoader(file_path)
    elif file_type == "docx":
        return Docx2txtLoader(file_path)
    else:
        return UnstructuredFileLoader(file_path)

def calculate_optimal_chunk_size(file_size_bytes, content_length):
    """Calculate optimal chunk size based on file size and content length"""
    # Convert bytes to MB
    file_size_mb = file_size_bytes / (1024 * 1024)

    # Base chunk size for small files (< 1MB)
    if file_size_mb < 1:
        base_chunk_size = 1000
    # Medium files (1MB - 5MB)
    elif file_size_mb < 5:
        base_chunk_size = 1500
    # Large files (5MB - 10MB)
    elif file_size_mb < 10:
        base_chunk_size = 2000
    # Very large files (> 10MB)
    else:
        base_chunk_size = 2500

    # Adjust based on content length
    content_factor = content_length / 1000  # per 1000 characters
    adjusted_chunk_size = int(base_chunk_size * (1 + (content_factor / 100)))

    # Set reasonable limits
    min_chunk_size = 500
    max_chunk_size = 3000

    return max(min_chunk_size, min(adjusted_chunk_size, max_chunk_size))

def process_document(uploaded_file):
    """Process uploaded document and return chunks with metadata"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        file_name = uploaded_file.name
        file_size = len(uploaded_file.getvalue())

        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file.flush()
            file_path = tmp_file.name

        # Load document
        loader = get_document_loader(file_path, file_extension)
        documents = loader.load()

        # Calculate total content length
        total_content_length = sum(len(doc.page_content) for doc in documents)

        # Calculate optimal chunk size
        chunk_size = calculate_optimal_chunk_size(file_size, total_content_length)
        overlap_size = int(chunk_size * 0.1)  # 10% overlap

        # Show chunking info in the UI
        st.info(f"""Document Statistics:
        - File Size: {file_size / (1024 * 1024):.2f} MB
        - Content Length: {total_content_length:,} characters
        - Calculated Chunk Size: {chunk_size:,} characters
        - Chunk Overlap: {overlap_size:,} characters""")

        # Split text with calculated chunk size
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)

        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "source": file_name,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "chunk_size": chunk_size,
                "file_size_mb": file_size / (1024 * 1024),
                "content_length": total_content_length
            })

        return chunks
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None
    finally:
        if 'file_path' in locals():
            try:
                os.unlink(file_path)
            except Exception as e:
                st.warning(f"Could not delete temporary file: {str(e)}")

def get_document_summary(vectorstore):
    """Get summary of documents stored in ChromaDB"""
    try:
        all_docs = vectorstore.get()
        docs_summary = {}

        # Group by source document
        for doc_id, metadata in zip(all_docs.get('ids', []), all_docs.get('metadatas', [])):
            source = metadata.get('source', 'Unknown')
            if source not in docs_summary:
                docs_summary[source] = {
                    'chunk_count': 0,
                    'ids': []
                }
            docs_summary[source]['chunk_count'] += 1
            docs_summary[source]['ids'].append(doc_id)

        return docs_summary
    except Exception as e:
        st.error(f"Error getting document summary: {str(e)}")
        return {}

# Set page config for a cleaner look
st.set_page_config(page_title="IBM WatsonX Chat", layout="wide")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "use_rag" not in st.session_state:
    st.session_state.use_rag = False

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

# Create tabs for Chat and Document Management
tab1, tab2 = st.tabs(["Chat", "Document Management"])

with tab2:
    st.title("📚 Document Management")

    # Initialize ChromaDB
    vectorstore = initialize_chroma()

    # Show existing documents
    if vectorstore is not None:
        try:
            docs_summary = get_document_summary(vectorstore)

            if docs_summary:
                st.subheader(f"📑 Existing Documents ({len(docs_summary)} documents)")

                # Select all checkbox
                select_all = st.checkbox("Select All Documents")

                # Document list with individual checkboxes
                selected_docs = {}
                for doc_name, info in docs_summary.items():
                    col1, col2 = st.columns([0.1, 0.9])
                    with col1:
                        is_selected = st.checkbox(
                            label=f"Select {doc_name}",
                            key=f"select_{doc_name}",
                            value=select_all,
                            label_visibility="collapsed"
                        )
                        if is_selected:
                            selected_docs[doc_name] = info['ids']

                    with col2:
                        with st.expander(f"📄 {doc_name}"):
                            st.text(f"Number of chunks: {info['chunk_count']}")

                # Delete buttons
                col1, col2 = st.columns(2)
                with col1:
                    if selected_docs and st.button("🗑️ Delete Selected", type="primary"):
                        try:
                            for doc_name, ids in selected_docs.items():
                                vectorstore.delete(ids=ids)
                                st.success(f"Successfully deleted {doc_name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting documents: {str(e)}")

                with col2:
                    if st.button("🗑️ Delete All Documents", type="secondary"):
                        try:
                            all_ids = [id for info in docs_summary.values() for id in info['ids']]
                            vectorstore.delete(ids=all_ids)
                            st.success("Successfully deleted all documents")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting all documents: {str(e)}")
            else:
                st.info("No documents stored in the database.")

        except Exception as e:
            st.error(f"Error accessing documents: {str(e)}")

    # File uploader
    st.divider()
    st.subheader("📤 Upload New Documents")
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "csv", "txt", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Process files first to show statistics
        files_to_process = []
        total_chunks = 0

        # Get existing documents
        existing_docs = get_document_summary(vectorstore) if vectorstore else {}

        st.subheader("📊 Document Analysis")
        for uploaded_file in uploaded_files:
            with st.expander(f"Analyzing {uploaded_file.name}"):
                try:
                    # Check if document already exists
                    if uploaded_file.name in existing_docs:
                        st.warning(f"Document '{uploaded_file.name}' already exists and will be replaced.")

                    # Process document to get chunk information
                    chunks = process_document(uploaded_file)
                    if chunks:
                        files_to_process.append({
                            'file': uploaded_file,
                            'chunks': chunks,
                            'chunk_count': len(chunks),
                            'replace_ids': existing_docs.get(uploaded_file.name, {}).get('ids', [])
                        })
                        total_chunks += len(chunks)
                except Exception as e:
                    st.error(f"Error analyzing {uploaded_file.name}: {str(e)}")

        if files_to_process:
            st.info(f"""
            📝 Summary:
            - Total files to process: {len(files_to_process)}
            - Total chunks to create: {total_chunks}
            - Files to be replaced: {sum(1 for f in files_to_process if f['replace_ids'])}
            """)

            # Add confirmation button
            if st.button("✅ Confirm and Process Documents", type="primary"):
                all_files_processed = True

                progress_bar = st.progress(0)
                for idx, file_info in enumerate(files_to_process):
                    with st.expander(f"Processing {file_info['file'].name}"):
                        try:
                            # Delete existing document if it exists
                            if file_info['replace_ids']:
                                vectorstore.delete(ids=file_info['replace_ids'])
                                st.info(f"Deleted existing version of {file_info['file'].name}")

                            # Store new version in ChromaDB
                            vectorstore.add_documents(file_info['chunks'])
                            vectorstore.persist()
                            st.success(f"Successfully processed and stored {file_info['file'].name}")
                        except Exception as e:
                            st.error(f"Error processing {file_info['file'].name}: {str(e)}")
                            all_files_processed = False

                    # Update progress bar
                    progress_bar.progress((idx + 1) / len(files_to_process))

                if all_files_processed:
                    st.success("All documents processed successfully. Refreshing page...")
                    time.sleep(1)
                    st.rerun()

with tab1:
    # Add RAG toggle
    st.sidebar.divider()
    st.sidebar.subheader("RAG Settings")
    use_rag = st.sidebar.toggle(
        "Use RAG (Query Documents)",
        value=st.session_state.use_rag,
        key="rag_toggle"
    )
    st.session_state.use_rag = use_rag

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Single chat input for both RAG and non-RAG modes
    if prompt := st.chat_input("What would you like to know?", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                # Initialize credentials and model
                credentials = Credentials(
                    api_key=existing_api_key,
                    url=region_url
                )

                # If RAG is enabled, get relevant documents
                if use_rag:
                    vectorstore = initialize_chroma()
                    relevant_docs = vectorstore.similarity_search(prompt, k=3)
                    context = "\n".join([doc.page_content for doc in relevant_docs])

                    # Format prompt with context
                    formatted_prompt = f"""System: You are a helpful assistant. Use the following context to answer the question.
Context: {context}

User: {prompt}
Assistant:"""
                else:
                    # Regular prompt without context
                    formatted_prompt = f"""System: You are a helpful assistant.
User: {prompt}
Assistant:"""

                # Initialize model and generate response
                model = ModelInference(
                    model_id=selected_model,
                    credentials=credentials,
                    project_id=existing_project_id
                )

                # Set parameters after initialization
                model.params = {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "top_p": top_p,
                    "decoding_method": DecodingMethods.GREEDY,
                    "repetition_penalty": 1.0,
                    "stop_sequences": ["User:", "System:", "Assistant:"]
                }

                response = model.generate(formatted_prompt)

                # Extract the generated text and token counts from the response
                if isinstance(response, dict) and 'results' in response:
                    response_text = response['results'][0]['generated_text']
                    generated_tokens = response['results'][0].get('generated_token_count', 0)
                    input_tokens = response['results'][0].get('input_token_count', 0)
                else:
                    response_text = str(response)
                    generated_tokens = 0
                    input_tokens = 0

                # Display response
                st.markdown(response_text)

                # Display token counts
                st.caption(f"📊 Tokens - Generated: {generated_tokens}, Input: {input_tokens}")

                # Save to chat history (including token counts)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"{response_text}\n\n_📊 Tokens - Generated: {generated_tokens}, Input: {input_tokens}_"
                })

                # Optionally display warnings if they exist
                if isinstance(response, dict) and 'system' in response and 'warnings' in response['system']:
                    relevant_warnings = [
                        warning for warning in response['system']['warnings']
                        if 'max_new_tokens' not in warning.get('message', '')
                    ]
                    if relevant_warnings:  # Only show expander if there are relevant warnings
                        with st.expander("System Warnings"):
                            for warning in relevant_warnings:
                                st.warning(warning['message'])

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



