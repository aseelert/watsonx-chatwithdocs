import streamlit as st

st.set_page_config(page_title="IBM watsonx Chat", layout="wide")

st.title("🤖 Welcome to IBM watsonx Chat")

st.markdown("""
### 👋 Welcome to the IBM watsonx Chat Application!

This application allows you to:
- 💬 Chat with IBM watsonx AI models
- 📚 Manage and query your documents
- 🔑 Configure your IBM watsonx credentials

### Getting Started:
1. First, go to the **Credentials** page to set up your IBM watsonx API credentials
2. Visit the **Document Management** page to upload and manage your documents
3. Use the **Chat** page to interact with the AI and your documents

### Navigation:
- Use the sidebar to navigate between different pages
- Each page has its own specific functionality
- Your settings and documents are preserved between sessions

### Need Help?
- Check IBM watsonx documentation for API details
- Use the Document Management page to see your uploaded files
- Enable RAG in the Chat page to query your documents
""")

# Add some useful links
st.divider()
st.subheader("Useful Links")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📖 Documentation")
    st.markdown("- [IBM watsonx Documentation](https://www.ibm.com/docs/en/watsonx-as-a-service)")
    st.markdown("- [API Reference](https://cloud.ibm.com/apidocs/watsonx-ai)")

with col2:
    st.markdown("#### 🛠️ Tools")
    st.markdown("- [IBM Cloud Console](https://cloud.ibm.com)")
    st.markdown("- [watsonx.ai Hub](https://www.ibm.com/products/watsonx-ai)")

with col3:
    st.markdown("#### 💡 Resources")
    st.markdown("- [Model Catalog](https://www.ibm.com/products/watsonx-ai/foundation-models)")
    st.markdown("- [Best Practices](https://www.ibm.com/docs/en/watsonx-as-a-service?topic=models-best-practices)")