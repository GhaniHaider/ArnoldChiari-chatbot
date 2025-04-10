# app.py - Streamlit Neurosurgeon Chatbot for Chiari Malformation

import streamlit as st
import os
import tempfile
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Page configuration
st.set_page_config(
    page_title="Neurosurgeon Chiari Consultant",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS with improved contrast
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .st-emotion-cache-16txtl3 h1 {
        color: #2c3e50;
    }
    .neurosurgeon {
        background-color: #e6f3ff;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #4e8cff;
        color: #000000 !important; /* Ensuring dark text color */
        font-weight: 400;
    }
    .user {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #a0a0a0;
        color: #000000 !important; /* Ensuring dark text color */
        font-weight: 400;
    }
    /* Additional styling to ensure text is visible */
    .neurosurgeon p, .user p {
        color: #000000 !important;
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

# Domain keywords for checking relevance
DOMAIN_KEYWORDS = [
    "brain", "chiari", "spine", "nerve", "type", "types", "malformation",
    "symptom", "treatment", "surgery", "diagnosis", "mri", "headache",
    "pain", "syrinx", "cerebrospinal", "fluid", "cerebellum", "tonsils",
    "decompression", "syringomyelia", "herniation", "foramen magnum"
]

# Neurosurgeon persona prompt template
NEUROSURGEON_TEMPLATE = """
You are a board-certified neurosurgeon with extensive specialized experience in treating Chiari malformations and related neurological conditions. You're known for your empathetic approach to patient care, technical expertise, and ability to explain complex medical concepts in understandable terms.

As a medical professional, you always:
- Provide accurate, evidence-based information
- Speak with compassion and understanding
- Acknowledge the challenges patients face
- Explain concepts clearly without overwhelming medical jargon
- Maintain a calm, reassuring tone
- Balance honesty about conditions with hope and support

Please consider the conversation history and context information to answer the current question about Chiari malformation with a professional, caring, and trustworthy approach.

Chat History:
{chat_history}

Question: {question}

Context information from medical literature: 
{context}

Neurosurgeon's response:
"""

PROMPT = PromptTemplate(
    template=NEUROSURGEON_TEMPLATE,
    input_variables=["chat_history", "context", "question"]
)

def check_if_domain_relevant(query: str) -> bool:
    """Check if the query contains any domain-specific keywords"""
    query = query.lower()
    for keyword in DOMAIN_KEYWORDS:
        if keyword in query:
            return True
    return False

def process_pdf(pdf_file):
    """Process uploaded PDF file and create a vector database"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_file.getvalue())
        temp_file_path = temp_file.name
    
    # Load and process the PDF
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    
    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Remove temp file
    os.unlink(temp_file_path)
    
    return vectorstore

def create_conversation_chain(vectorstore):
    """Create a conversational chain with the processed PDF"""
    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
    
    # Setup Conversational Retrieval Chain
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=False,
        return_generated_question=False,
    )
    
    return conversation_chain

def main():
    st.title("🧠 Neurosurgeon Chiari Malformation Consultant")
    
    # Sidebar for API key and file upload
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        if not st.session_state.gemini_api_key:
            api_key = st.text_input("Enter Gemini API Key:", type="password")
            if api_key:
                try:
                    # Configure Gemini with the provided API key
                    os.environ["GOOGLE_API_KEY"] = api_key
                    genai.configure(api_key=api_key)
                    st.session_state.gemini_api_key = api_key
                    st.success("API key configured successfully!")
                except Exception as e:
                    st.error(f"Error configuring API key: {e}")
        else:
            st.success("API key is configured.")
            if st.button("Reset API Key"):
                st.session_state.gemini_api_key = ""
                st.rerun()  # Updated from experimental_rerun
        
        # PDF upload
        if st.session_state.gemini_api_key and not st.session_state.pdf_processed:
            st.subheader("Upload Chiari PDF File")
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            
            if uploaded_file is not None:
                with st.spinner("Processing PDF... This may take a moment."):
                    try:
                        # Process the PDF and create a conversation chain
                        vectorstore = process_pdf(uploaded_file)
                        st.session_state.conversation_chain = create_conversation_chain(vectorstore)
                        st.session_state.pdf_processed = True
                        st.success("PDF processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")
        
        # Information section
        st.subheader("About")
        st.markdown("""
        This chatbot is powered by a neurosurgeon AI specializing in Chiari malformation.
        
        Ask questions about:
        - Chiari malformation types and causes
        - Symptoms and diagnosis
        - Treatment options and surgery
        - Recovery and prognosis
        """)

    # Main chat interface
    if st.session_state.conversation_chain:
        # Introduction message when first loading
        if not st.session_state.chat_history:
            st.markdown(
                """<div class="neurosurgeon">
                <p>Hello, I'm a neurosurgeon specializing in Chiari malformations. I'm here to answer your questions about this condition and provide the guidance you need. How can I help you today?</p>
                </div>""", 
                unsafe_allow_html=True
            )
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""<div class="user"><p>{message["content"]}</p></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="neurosurgeon"><p>{message["content"]}</p></div>""", unsafe_allow_html=True)
        
        # Input for new questions
        user_question = st.chat_input("Ask your question about Chiari malformation...")
        
        if user_question:
            # Add user message to chat history for display
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Display user message
            st.markdown(f"""<div class="user"><p>{user_question}</p></div>""", unsafe_allow_html=True)
            
            with st.spinner("Thinking..."):
                if check_if_domain_relevant(user_question):
                    # Get response from the conversation chain
                    response = st.session_state.conversation_chain({"question": user_question})
                    answer = response['answer']
                else:
                    # Off-topic response
                    answer = "I'm sorry, but I specialize specifically in Chiari malformation and related neurological conditions. To provide you with the most helpful information, could you please ask me about topics related to Chiari, brain structure, spinal issues, or neurological symptoms? I want to ensure I give you accurate and relevant guidance."
                    
                    # We should also update the memory for continuity
                    if hasattr(st.session_state.conversation_chain, 'memory'):
                        st.session_state.conversation_chain.memory.save_context(
                            {"question": user_question}, 
                            {"answer": answer}
                        )
            
            # Display assistant response
            st.markdown(f"""<div class="neurosurgeon"><p>{answer}</p></div>""", unsafe_allow_html=True)
            
            # Add assistant message to chat history for display
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            
            # Removed the experimental_rerun() call that was causing issues
    else:
        # Instructions when not yet configured
        if not st.session_state.gemini_api_key:
            st.info("👈 Please enter your Gemini API key in the sidebar to get started.")
        elif not st.session_state.pdf_processed:
            st.info("👈 Please upload the Chiari PDF file in the sidebar to continue.")

if __name__ == "__main__":
    main()
