## RAG based Document Question Answering System
This project implements a RAG system that allows users to upload a PDF document and ask questions about its contents through a Streamlit interface. When a document is uploaded the system extracts the text, makes smaller chunks and converts them into vector embeddings which are then stored in the Chroma vector DB. This allows semantic search in the document.

When the user asks a question, teh system retrieves relevant chunks from the vector DB and passes it to the LLM as context along with the question asked. The model generates answers strictly based on the context provided and thus has good accuracy. The application also displays the source passages used to generate the answer making the responses easy to verify.


## Tech Stack

* **Python**
* **Streamlit** – user interface for uploading PDFs and asking questions
* **LangChain** – orchestration of the RAG pipeline
* **Sentence Transformers** – generating embeddings for document chunks
* **ChromaDB** – vector database for semantic search
* **Ollama (phi3)** – local large language model for answer generation
* **PyPDF** – PDF text extraction
