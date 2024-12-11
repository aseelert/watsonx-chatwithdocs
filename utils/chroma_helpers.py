import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredFileLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

class ChromaManager:
    def __init__(self):
        self.vectorstore = self.initialize_chroma()

    def initialize_chroma(self):
        """Initialize ChromaDB with HuggingFace embeddings"""
        persist_directory = "./chroma_db"
        os.makedirs(persist_directory, exist_ok=True)

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
            return vectorstore
        except Exception as e:
            st.error(f"Error initializing ChromaDB: {str(e)}")
            return None

    @staticmethod
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

    @staticmethod
    def calculate_optimal_chunk_size(file_size_bytes, content_length):
        """Calculate optimal chunk size based on file size and content length"""
        file_size_mb = file_size_bytes / (1024 * 1024)

        if file_size_mb < 1:
            base_chunk_size = 1000
        elif file_size_mb < 5:
            base_chunk_size = 1500
        elif file_size_mb < 10:
            base_chunk_size = 2000
        else:
            base_chunk_size = 2500

        content_factor = content_length / 1000
        adjusted_chunk_size = int(base_chunk_size * (1 + (content_factor / 100)))

        return max(500, min(adjusted_chunk_size, 3000))

    def process_document(self, uploaded_file):
        """Process uploaded document and return chunks with metadata"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            file_name = uploaded_file.name
            file_size = len(uploaded_file.getvalue())

            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file.flush()
                file_path = tmp_file.name

            loader = self.get_document_loader(file_path, file_extension)
            documents = loader.load()

            total_content_length = sum(len(doc.page_content) for doc in documents)
            chunk_size = self.calculate_optimal_chunk_size(file_size, total_content_length)
            overlap_size = int(chunk_size * 0.1)

            st.info(f"""Document Statistics:
            - File Size: {file_size / (1024 * 1024):.2f} MB
            - Content Length: {total_content_length:,} characters
            - Calculated Chunk Size: {chunk_size:,} characters
            - Chunk Overlap: {overlap_size:,} characters""")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap_size,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)

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

    def get_document_summary(self):
        """Get summary of documents stored in ChromaDB"""
        try:
            all_docs = self.vectorstore.get()
            docs_summary = {}

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