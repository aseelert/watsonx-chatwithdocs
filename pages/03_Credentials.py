import streamlit as st
from utils.credentials_helpers import CredentialsManager

st.set_page_config(page_title="Credentials Management", layout="wide")
st.title("🔑 Credentials Management")

# Initialize credentials manager
credentials_manager = CredentialsManager()

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # Region selection
    selected_region = st.selectbox(
        "Select Region",
        list(CredentialsManager.WATSON_REGIONS.keys()),
        index=0
    )
    region_url = CredentialsManager.WATSON_REGIONS[selected_region]

    # Load existing credentials
    existing_credentials = credentials_manager.load_credentials()
    if existing_credentials:
        existing_api_key, existing_project_id = existing_credentials
    else:
        existing_api_key, existing_project_id = "", ""
        st.warning("No credentials found. Please enter your API Key and Project ID.")

    # Credentials input
    api_key = st.text_input("API Key", value=existing_api_key, type="password")
    project_id = st.text_input("Project ID", value=existing_project_id, type="password")

    # Show/Hide credentials
    show_credentials = st.checkbox("Show Credentials")
    if show_credentials:
        st.text_input("API Key (visible)", value=api_key, disabled=True)
        st.text_input("Project ID (visible)", value=project_id, disabled=True)

    # Action buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Save Credentials", type="primary"):
            if api_key and project_id:
                if credentials_manager.save_credentials(api_key, project_id):
                    st.success("Credentials saved successfully!")
            else:
                st.error("Please enter both API Key and Project ID")

    with col_btn2:
        if st.button("Verify Connection"):
            if api_key and project_id:
                credentials_manager.verify_credentials(api_key, project_id, region_url)
            else:
                st.error("Please enter both API Key and Project ID")

with col2:
    st.info("""
    ### How to Get Credentials

    1. Log in to [IBM Cloud](https://cloud.ibm.com)
    2. Navigate to watsonx.ai service
    3. Create or select a project
    4. Get your API Key and Project ID

    ### Important Notes
    - Keep your credentials secure
    - Don't share your API Key
    - Choose the closest region
    - Verify connection after saving
    """)