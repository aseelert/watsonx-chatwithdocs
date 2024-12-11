# watsonx.ai Chat Application

A Streamlit-based chat application that interfaces with IBM's watsonx.ai foundation models, featuring document-based knowledge retrieval (RAG) capabilities.

## Features

- 💬 Interactive chat interface with watsonx.ai models
- 📚 Document management with RAG (Retrieval-Augmented Generation)
- 🔧 Adjustable model parameters (temperature, tokens, etc.)
- 📊 Token usage tracking
- 📁 Support for multiple document formats (PDF, CSV, TXT, DOCX)
- 🔐 Secure credential management

## Prerequisites

1. IBM Cloud Account with watsonx.ai access
2. API Key and Project ID from IBM Cloud
3. Python 3.8 or higher
4. pip package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aseelert/watsonx-chatwithdocs
cd watsonx-chatwithdocs
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv312
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Configuration

1. Launch the application:
```bash
streamlit run Home.py
```

2. Navigate to the "Credentials" page
3. Enter your watsonx.ai credentials:
   - API Key
   - Project ID
4. Select your closest region
5. Verify the connection

## Usage

**Chat with watsonx.ai**
![alt text](pictures/image.png)

**Simple chat interface with RAG enabled.**
![alt text](pictures/image-1.png)

**use Document Knowledge**
![alt text](pictures/image-2.png)

### Chat Interface
- Navigate to the "Chat" page
- Adjust model parameters in the sidebar:
  - Temperature (0.0 - 1.0)
  - Max Tokens (100 - 4096)
  - Top P (0 or 1)
  - Top K (0 - 100)
  - Repetition Penalty (1 or 2)

### Document Management
- Go to the "Document Management" page
- Upload documents (PDF, CSV, TXT, DOCX)
- Enable "Use Document Knowledge" in Chat to use RAG

## File Structure

```
watsonx-chatwithdocs/
├── Home.py                 # Main application entry
├── pages/
│   ├── 01_Chat.py         # Chat interface
│   ├── 02_Documents.py    # Document handling
│   └── 03_Credentials.py  # Credential management
├── utils/
│   ├── watsonx_helpers.py  # watsonx.ai integration
│   ├── chroma_helpers.py   # Vector database management
│   └── credentials_helpers.py # Credential handling
└── requirements.txt        # Python dependencies
```

## Dependencies

- streamlit
- langchain
- chromadb
- ibm-watsonx-ai
- python-dotenv
- sentence-transformers
- unstructured
- pypdf
- python-docx

## Security Notes

- Credentials are stored locally in a SQLite database
- API keys are stored securely and not exposed in the UI
- Choose the closest region for optimal performance

## Troubleshooting

1. If ChromaDB fails to initialize:
   - Ensure write permissions in the application directory
   - Check if required models are downloaded

2. If watsonx.ai connection fails:
   - Verify API key and Project ID
   - Check region selection
   - Ensure IBM Cloud account has necessary permissions

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License