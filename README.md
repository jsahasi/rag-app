# RAG Application

A powerful Retrieval-Augmented Generation (RAG) application that lets you chat with your documents using AI. Supports multiple LLM providers (Anthropic Claude and OpenAI GPT) and flexible embedding options.

## Features

- **Multi-LLM Support**: Switch between Anthropic Claude and OpenAI GPT
- **Dual Embedding Options**: Use OpenAI embeddings or free local embeddings (sentence-transformers)
- **Wide File Format Support**: txt, md, code files, PDF, DOCX
- **Custom Personas**: Add an `instructions.txt` file to customize AI behavior
- **Streaming Responses**: Real-time response streaming in chat mode
- **Persistent Index**: Document embeddings stored locally for fast queries

## Installation

```bash
# Clone the repository
git clone https://github.com/jsahasi/rag-app.git
cd rag-app

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Create a `.env` file with your API keys:

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-your-key-here
DEFAULT_LLM=anthropic
DEFAULT_EMBEDDING=local
```

## Usage

### Index Documents

```bash
# Index a folder of documents
python main.py index ./your_docs

# Force rebuild the index
python main.py index ./your_docs --rebuild

# Use OpenAI embeddings instead of local
python main.py index ./your_docs --embedding openai
```

### Interactive Chat

```bash
# Start chat session (uses default LLM)
python main.py chat ./your_docs

# Use specific LLM provider
python main.py chat ./your_docs --llm openai
python main.py chat ./your_docs --llm anthropic
```

**Chat Commands:**
- Type your question and press Enter
- `switch` - Toggle between Anthropic and OpenAI
- `quit` or `exit` - End the session

### Single Query

```bash
python main.py query ./your_docs "What is this project about?"
```

### Check Status

```bash
python main.py status ./your_docs
```

## Custom Persona

Add an `instructions.txt` file to your document folder to customize the AI's behavior:

```text
You are a helpful technical documentation assistant.
Always be concise and provide code examples when relevant.
If you're unsure about something, say so clearly.
```

## Supported File Types

| Category | Extensions |
|----------|------------|
| Text | `.txt`, `.md`, `.markdown` |
| Code | `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.json`, `.yaml`, `.yml`, `.html`, `.css`, `.sql`, `.sh`, `.java`, `.c`, `.cpp`, `.go`, `.rs`, `.rb`, `.php`, `.swift`, `.kt` |
| Documents | `.pdf`, `.docx` |

## Architecture

```
rag_app/
├── main.py              # CLI entry point
├── config.py            # Configuration management
├── document_loader.py   # Multi-format document loading
├── embeddings.py        # OpenAI & local embedding services
├── vector_store.py      # ChromaDB vector storage
├── llm_service.py       # LLM provider abstraction
├── rag_engine.py        # Core RAG pipeline
└── requirements.txt     # Dependencies
```

## Dependencies

- **anthropic** - Anthropic Claude API
- **openai** - OpenAI API
- **chromadb** - Vector database
- **sentence-transformers** - Local embeddings
- **pypdf** - PDF parsing
- **python-docx** - DOCX parsing
- **rich** - Beautiful CLI output
- **click** - CLI framework

## License

MIT License

## Credits

Created by:
- **Jayesh Sahasi**
- **Veer Sahasi**
- **Ruhan Sahasi**
- **Arushi Sahasi**

---

Built with Claude Code
