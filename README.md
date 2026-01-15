# DentalAI

## Dental AI Assistant

A sophisticated medical chatbot designed to assist with dental-related queries using advanced Retrieval-Augmented Generation (RAG) techniques. This project implements two distinct RAG scenarios to provide accurate, context-aware responses for dental professionals and patients.

## Features

- **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with generative AI for precise answers.
- **Two RAG Scenarios**:
  - **RAG with Chat History**: Maintains conversation context across interactions for more coherent, personalized responses.
  - **Corrective RAG**: Includes document relevance grading and web search supplementation to improve answer quality and accuracy.
- **Vector Store Integration**: Uses ChromaDB for efficient document storage and retrieval.
- **Multiple LLM Support**: Compatible with local models (Ollama) and cloud models (OpenAI).
- **Web Search Integration**: Leverages Tavily API for additional information when needed.
- **Containerized Deployment**: Includes Docker support for easy deployment.

## Scenarios

### 1. RAG with Chat History
This scenario maintains conversation history to provide context-aware responses. It uses LangChain's conversation memory to track previous interactions, ensuring the chatbot remembers past queries and responses within a session.

- **File**: `rag_with_hist.py`
- **Notebook**: `notebooks/RAG_Chain_with_Chat_History.ipynb`
- **Use Case**: Ideal for multi-turn conversations where context from previous messages is crucial.

### 2. Corrective RAG
This advanced scenario implements a self-correcting RAG pipeline that:
- Retrieves relevant documents from the vector store.
- Grades document relevance to filter out irrelevant information.
- Performs web search if retrieved documents are insufficient.
- Transforms queries for better retrieval when needed.

- **File**: `corrective_rag.py`
- **Notebook**: `notebooks/Corrective_RAG.ipynb`
- **Use Case**: Best for high-stakes medical queries where accuracy is paramount and external verification may be required.

## Installation

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized deployment)
- API keys for external services (OpenAI, Tavily)

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd DentalAI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up API keys in `api_keys.txt` or environment variables:
   - `OPENAI_API_KEY`
   - `TAVILY_API_KEY`

4. Prepare the vector store:
   ```bash
   python indexing.py  # Index documents into ChromaDB
   ```

### Docker Deployment
```bash
docker-compose up --build
```

## Usage

### Running the Chatbot
- **Main Application**: `python main.py`
- **RAG with History**: `python rag_with_hist.py`
- **Corrective RAG**: `python corrective_rag.py`

### Notebooks
Explore the interactive notebooks in the `notebooks/` directory:
- `RAG.ipynb`: Basic RAG implementation
- `RAG_Chain_with_Chat_History.ipynb`: RAG with conversation history
- `Corrective_RAG.ipynb`: Self-correcting RAG pipeline
- `Self_RAG.ipynb`: Additional RAG variations

### Configuration
- Modify `run_local` variable to switch between local and cloud LLMs.
- Adjust vector store parameters in the respective files.
- Update prompts and chains as needed for customization.

## Project Structure

```
DentalAI/
├── api_keys.txt              # API keys (not committed)
├── compose.yaml              # Docker Compose configuration
├── corrective_rag.py         # Corrective RAG implementation
├── corrective_rag_with_hist.py  # Corrective RAG with history
├── Dockerfile                # Docker image definition
├── indexing.py               # Document indexing script
├── loading.py                # Data loading utilities
├── main.py                   # Main application entry point
├── rag_with_hist.py          # RAG with chat history
├── RAG.ipynb                 # Basic RAG notebook
├── README.Docker.md          # Docker-specific documentation
├── requirements.txt          # Python dependencies
├── view.py                   # Utility for viewing data
├── __pycache__/              # Python cache
├── data/                     # Raw data files
├── notebooks/                # Jupyter notebooks
│   ├── Corrective_RAG.ipynb
│   ├── RAG_Chain_with_Chat_History.ipynb
│   └── Self_RAG.ipynb
└── vectorstore/              # ChromaDB vector store
    ├── chroma.sqlite3
    └── [collection files]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add license information here]

## Disclaimer

This is a research/demo project. Not intended for production medical use without proper validation and regulatory approval.
