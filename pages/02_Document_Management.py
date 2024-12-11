import streamlit as st
from utils.chroma_helpers import ChromaManager
import time

st.set_page_config(page_title="Document Management", layout="wide")
st.title("📚 Document Management")

# Initialize ChromaDB manager
chroma_manager = ChromaManager()

# Show existing documents
if chroma_manager.vectorstore is not None:
    try:
        docs_summary = chroma_manager.get_document_summary()

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
                            chroma_manager.vectorstore.delete(ids=ids)
                            st.success(f"Successfully deleted {doc_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting documents: {str(e)}")

            with col2:
                if st.button("🗑️ Delete All Documents", type="secondary"):
                    try:
                        all_ids = [id for info in docs_summary.values() for id in info['ids']]
                        chroma_manager.vectorstore.delete(ids=all_ids)
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
    existing_docs = chroma_manager.get_document_summary()

    st.subheader("📊 Document Analysis")
    for uploaded_file in uploaded_files:
        with st.expander(f"Analyzing {uploaded_file.name}"):
            try:
                # Check if document already exists
                if uploaded_file.name in existing_docs:
                    st.warning(f"Document '{uploaded_file.name}' already exists and will be replaced.")

                # Process document to get chunk information
                chunks = chroma_manager.process_document(uploaded_file)
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
                            chroma_manager.vectorstore.delete(ids=file_info['replace_ids'])
                            st.info(f"Deleted existing version of {file_info['file'].name}")

                        # Store new version in ChromaDB
                        chroma_manager.vectorstore.add_documents(file_info['chunks'])
                        chroma_manager.vectorstore.persist()
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