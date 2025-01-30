import eventlet
eventlet.monkey_patch()
import os
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate




app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


PDF_PATHS = ["Mobily AR_2022_English (1).pdf", "Operation-and-Maintenance-Manual_SEBU8407-06 (1).pdf"]
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    raise ValueError("No GEMINI_API_KEY found in environment variables")



embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)


SYSTEM_PROMPT = """
You are a customer support assistant for Mobily and Caterpillar Operation & Maintenance.  
Your primary goal is to provide clear, friendly, and professional assistance based on available information.  
Maintain a natural conversation flow while ensuring responses are concise and relevant.  

### **Response Prioritization:**  
1. **High Priority:**  
   - Answer the current question based on the most relevant information from the knowledge base.  
   - Retrieve and summarize relevant details without referencing the data source.  

2. **Context Awareness:**  
   - Use past conversation history  only to maintain coherence.  
   - Do NOT let previous responses alter new answers unless explicitly requested.  
   - Do NOT repeat the same response unless the user asks for clarification or repetition.  

3. **Conversational & Engaging:**  
   - Keep responses natural and user-friendly.  
   - Use contractions like "you're," "don't," and "it's" for a human-like tone.  
   - Avoid robotic or overly formal language.  

4. **Handling Unclear Queries:**  
   - If a question is ambiguous, ask for clarification instead of making assumptions.  
   - If you lack an answer, say: *"I'm not sure about that, but I can check for you!"*  

5. **What to Avoid:**  
   - Do **not** mention that you're an AI or LLM.  
   - Do **not** say "Based on the document..." or refer to any data source.  
   - Do **not** offer any services apart from the knowledge base provided.  
   - Do **not** provide financial or legal advice.  

---
### **Conversation Context (for maintaining flow, NOT altering responses):**  
{context}  

### **Chat History:**  
{chat_history}  
  
"""



prompt_template = ChatPromptTemplate.from_messages([
    ("human", SYSTEM_PROMPT),  # Changed from "system" to "human"
    HumanMessagePromptTemplate.from_template("{question}")
])


def create_vector_store():
    docs = []
    for path in PDF_PATHS:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return FAISS.from_documents(text_splitter.split_documents(docs), embeddings)

vector_store = create_vector_store()



conversation_chains = {}

def get_conversation_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=GEMINI_API_KEY
    )
    
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=5,
        return_messages=True,
        output_key="answer"
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

@app.route('/')
def index():
    return render_template("index.html")

@socketio.on('user_message')
def handle_message(data):
    session_id = request.sid
    user_query = data.get("query", "").strip()
    
    try:
        if session_id not in conversation_chains:
            conversation_chains[session_id] = get_conversation_chain()
        
        chain = conversation_chains[session_id]
        response = chain({"question": user_query})
        answer = response["answer"]
        emit('bot_response', {"response": answer}, room=session_id)
    
    except Exception as e:
        
        error_message = f"Error processing your request: {str(e)}"
        emit('bot_response', {"response": error_message}, room=session_id)
        print(f"Error in session {session_id}: {str(e)}")

application = app


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}")  # Add logging
    socketio.run(
        app,
        host='0.0.0.0',
        port=port,
        debug=False,
        allow_unsafe_werkzeug=True
    )

