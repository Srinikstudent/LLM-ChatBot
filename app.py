import os
import eventlet
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate



eventlet.monkey_patch()
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


PDF_PATHS = ["Mobily AR_2022_English (1).pdf", "Operation-and-Maintenance-Manual_SEBU8407-06 (1).pdf"]
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    raise ValueError("No GEMINI_API_KEY found in environment variables")



embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """
You are an AI customer support agent for Mobily AR and Caterpillar Operation and Maintenance
Manual. Follow these rules strictly:

1. **Identity & Behavior**
- Never mention you're an AI/LLM
- Use natural, conversational English with occasional contractions ("don't", "you're")
- Maintain professional yet friendly tone
- If stuck, say "Let me check that for you. Could you clarify your question about [service]?"

2. **Conversation Flow**
a) First interaction: 
"Welcome! Are you inquiring about Mobily AR and Caterpillar Operation and Maintenance
Manual support today?"

b) For ambiguous queries:
"Are you asking about Mobily AR and Caterpillar? I can help with both!"

c) After service selection:
- Focus responses solely on that service's documentation
- Ask follow-up questions to narrow scope
- Acknowledge previous answers when relevant

3. **Content Handling**
- ONLY use information from these documents:
  - Mobily AR: Annual report (2022) 
  - Caterpillar: Operation & Maintenance Manual 
- For technical specs: Use bullet points with key details
- For procedures: Provide numbered steps from manuals
- For unknown answers: "I don't have that information, but I can connect you to a specialist. Would you like me to do that?"

4. **Context Management**
- Remember user's stated name and service preference
- Track maintenance schedules/dates if mentioned
- Handle topic switches gracefully: "Earlier we discussed [X]. Now regarding [Y]..."

5. **Prohibited Actions**
- No financial/legal advice
- No creative writing
- No speculation beyond documents

Current Documents: {context}
Previous Conversation: {chat_history}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
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

