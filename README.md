# 🤖 AI-Powered RAG Chatbot  

[![Chatbot](https://img.shields.io/badge/Try%20Chatbot-Click%20Here-blue?style=for-the-badge)](https://llm-chatbot-9.onrender.com/)  

An intelligent **Retrieval-Augmented Generation (RAG) chatbot** that leverages **vector stores** and **LangChain** to deliver precise and context-aware responses. It extracts information from PDFs and enables natural, human-like interactions.  

---

## 🚀 Features  
✅ **RAG-based Conversational AI** – Extracts relevant info from PDF documents  
✅ **Vector Store Search** – Retrieves answers efficiently using embeddings  
✅ **LangChain-powered Chat** – Maintains context across multiple queries  
✅ **Flask + WebSocket** – Enables real-time responses  
✅ **Google Gemini AI** – Provides intelligent, structured replies  

---

## 🔗 Try the Chatbot  
👉 **[Live Demo](https://llm-chatbot-9.onrender.com/)**  

---

## 🛠 Tech Stack  
- **Backend:** Flask, Flask-SocketIO  
- **LLM:** Google Gemini AI (via LangChain)  
- **RAG Framework:** LangChain  
- **Vector Store:** FAISS  
- **PDF Processing:** PyPDFLoader  
- **Frontend:** HTML, JavaScript, WebSockets  

---

## 📊 Chatbot Flowchart  

```mermaid
graph TD
    A[User Input] -->|Question| B{Check Session}
    B -->|New Session| C[Create Conversation Chain]
    B -->|Existing Session| D[Retrieve Memory]
    C --> E[Process Query]
    D --> E
    E --> F[Retrieve Relevant Docs (FAISS)]
    F --> G[Generate Response (Gemini AI)]
    G --> H[Return Answer to User]



