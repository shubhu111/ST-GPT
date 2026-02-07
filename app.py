import streamlit as st
import os  # Required for file cleanup
import time # <--- Added for Rate Limiting
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
import tempfile 
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# --- DEPLOYMENT FIX: BRIDGE TO STREAMLIT SECRETS ---
# We use a try-except block to prevent a crash when running locally without a secrets.toml file
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except FileNotFoundError:
    pass # Running locally, secrets.toml not found. We rely on load_dotenv() above.
except Exception:
    pass # Any other secrets error, just ignore and fall back to .env

# Page Configuration
st.set_page_config(
    page_title="ST-GPT",
    page_icon="🤖",
    layout="wide"
)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🔧 Tools")
option = st.sidebar.radio(
    "Choose a Task:",
    ["🤖 AI Tutor (ChatBot)", "📹 Chat with YouTube", "📄 Chat with PDF"]
)

# --- MEMORY RESET LOGIC ---
if "current_tool" not in st.session_state:
    st.session_state.current_tool = option

# Clear memory if tool changes
if st.session_state.current_tool != option:
    if "chat_history" in st.session_state:
        del st.session_state["chat_history"]
    if "vector_store" in st.session_state:
        del st.session_state["vector_store"]
    if "youtube_chat_history" in st.session_state:
        del st.session_state["youtube_chat_history"]
    if "pdf_chat_history" in st.session_state:
        del st.session_state["pdf_chat_history"]
    
    st.session_state.current_tool = option
    st.toast(f"Switched to {option} - Memory Cleared!", icon="🧹")

# --- MAIN APP LOGIC ---

# 1. AI FRIEND & HELPER (Updated Logic)
if option == "🤖 AI Tutor (ChatBot)":
    st.title("🤖 My AI Friend")
    st.write("Hey! I'm here to help you solve problems, learn new things, or just chat. What's on your mind?")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display previous messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    user_input = st.chat_input("Talk to your friend...")
    if user_input:
        # 1. Display User Message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. Prepare the "Brain" (Context + System Prompt)
        history_for_llm = []
        
        # A. The Persona (System Prompt)
        system_instruction = SystemMessage(content="""
        You are ST-GPT, a smart, friendly, and helpful AI assistant created by Shubham Tade.

        ### YOUR BEHAVIOR:
        - Be friendly and supportive, like a smart best friend.
        - If the user asks for code or technical help, be precise and professional.
        
        ### LANGUAGE RULES (STRICT):
        1. IF the user speaks in ENGLISH -> Respond in pure ENGLISH. (you can use emojis)
        2. IF the user speaks in HINDI or HINGLISH -> Respond in HINGLISH (Hindi + English mix).
        3. Do NOT mix languages if the user is speaking clearly in English.
        
        If asked "Who made you?", answer: "I was created by Shubham Tade, an AI Engineer." (Only provide his LinkedIn if asked 'https://www.linkedin.com/in/shubham-tade123/').
        
        """)
        history_for_llm.append(system_instruction)
        
        # B. Add Chat History (Memory)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                history_for_llm.append(HumanMessage(content=msg["content"]))
            else:
                history_for_llm.append(AIMessage(content=msg["content"]))

        # 3. Generate Response
        with st.chat_message("assistant"):
            
            # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.6)
            llm = ChatGroq(model='llama-3.1-8b-instant',temperature=0.3)
            
            # Pass the WHOLE history to enabling memory
            response_stream = llm.stream(history_for_llm)
            response = st.write_stream(response_stream)
            
            # Rate Limit Protection (Wait 2s to be safe)
            time.sleep(2)
        
        # 4. Save AI Message
        st.session_state.chat_history.append({"role": "assistant", "content": response})


# 2. CHAT WITH YOUTUBE (RAG)
elif option == "📹 Chat with YouTube":
    st.title("📹 Chat with YouTube Videos")
    
    # Input Section
    video_id = st.text_input("Enter YouTube Video ID (e.g., 'dQw4w9WgXcQ'):")
    
    # Process Video Button
    if st.button("Analyze Video"):
        if video_id:
            with st.spinner("Getting Transcript & Creating Embeddings..."):
                try:
                    # A. Fetch Transcript
                    transcript_data = YouTubeTranscriptApi().fetch(video_id, languages=['en', 'hi'])
                    transcript = " ".join(item.text for item in transcript_data)

                    # B. Split Text
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    chunks = splitter.create_documents([transcript])

                    # C. Create Vector Store (FAISS)
                    embed_model = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')
                    vector_store = FAISS.from_documents(documents=chunks, embedding=embed_model)
                    
                    # D. Save to Session State
                    st.session_state.vector_store = vector_store
                    st.success("Video Processed! You can now ask questions below.")

                except TranscriptsDisabled:
                    st.error("Transcripts are disabled for this video.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # Chat Interface (Only shows if video is processed)
    if "vector_store" in st.session_state:
        st.divider()
        st.subheader("💬 Chat with Video")

        if "youtube_chat_history" not in st.session_state:
            st.session_state.youtube_chat_history = []

        # Display Chat History
        for message in st.session_state.youtube_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        query = st.chat_input("Ask something about the video...")
        if query:
            # 1. Display User Message
            st.session_state.youtube_chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # 2. Generate Answer using RAG
            with st.chat_message("assistant"):
                retriever = st.session_state.vector_store.as_retriever(kwargs={'k': 5})
                llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite', temperature=0.2)
                # llm = ChatGroq(model='llama-3.1-8b-instant',temperature=0.2)
                
                template = """
                You are a helpful assistant. Answer the question based ONLY on the provided video transcript context.
                
                CRITICAL LANGUAGE RULE: 
                - You must ALWAYS answer in ENGLISH, even if the video transcript is in Hindi or another language.
                - ONLY answer in a different language if the user explicitly asks for it (e.g., "Answer in Hindi").
                
                If the answer is not in the context, say "I don't know based on this video."
                
                Context: {context}
                Question: {question}
                """
                prompt = PromptTemplate.from_template(template)
                
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                Parallel_chain = RunnableParallel({
                    'context' : retriever | RunnableLambda(format_docs),
                    'question': RunnablePassthrough()
                })
                parser = StrOutputParser()

                chain = Parallel_chain | prompt | llm | parser
                
                response = chain.invoke(query)
                st.markdown(response)
                
            # 3. Save Assistant Message
            st.session_state.youtube_chat_history.append({"role": "assistant", "content": response})


# 3. CHAT WITH PDF (Complete Implementation)
elif option == "📄 Chat with PDF":
    st.title("📄 Chat with PDF Documents")
    
    uploaded_file = st.file_uploader("Upload your PDF", type=['pdf'])
    
    if uploaded_file:
        if st.button("Process PDF"):
            with st.spinner("Reading PDF & Creating Embeddings..."):
                try:
                    # 1. Save uploaded file to a temporary file
                    # We need a real file path for PyPDFLoader
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name

                    # 2. Load and Split PDF
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    
                    # Clean up: Delete the temp file from disk
                    os.remove(tmp_file_path)

                    # 3. Split Text
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    chunks = splitter.split_documents(docs)

                    # 4. Create Vector Store (FAISS)
                    embed_model = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')
                    vector_store = FAISS.from_documents(documents=chunks, embedding=embed_model)
                    
                    # 5. Save to Session State
                    st.session_state.vector_store = vector_store
                    st.success("PDF Processed! You can now ask questions below.")

                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

    # Chat Interface (Matches YouTube Logic)
    if "vector_store" in st.session_state:
        st.divider()
        st.subheader("💬 Chat with Document")

        if "pdf_chat_history" not in st.session_state:
            st.session_state.pdf_chat_history = []

        # Display Chat History
        for message in st.session_state.pdf_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        query = st.chat_input("Ask something about the PDF...")
        if query:
            # 1. User Message
            st.session_state.pdf_chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # 2. Assistant Response
            with st.chat_message("assistant"):
                retriever = st.session_state.vector_store.as_retriever(kwargs={'k': 5})
                llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite', temperature=0.2)
                # llm = ChatGroq(model='llama-3.1-8b-instant',temperature=0.2)
                
                template = """
                You are a helpful assistant. Answer the question based ONLY on the provided document context.
                
                CRITICAL LANGUAGE RULE: 
                - You must ALWAYS answer in ENGLISH, even if the document is in another language.
                - ONLY answer in a different language if the user explicitly asks for it.
                
                If the answer is not in the context, say "I don't know based on this document."
                
                Context: {context}
                Question: {question}
                """
                prompt = PromptTemplate.from_template(template)
                
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                Parallel_chain = RunnableParallel({
                    'context': retriever | RunnableLambda(format_docs),
                    'question': RunnablePassthrough()
                })
                parser = StrOutputParser()
                chain = Parallel_chain | prompt | llm | parser
                
                response = chain.invoke(query)
                st.markdown(response)
            
            # 3. Save Assistant Message
            st.session_state.pdf_chat_history.append({"role": "assistant", "content": response})

# --- SIDEBAR FOOTER (Connect with Me) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Connect with Me")

st.sidebar.markdown(
    """
    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
        <a href="https://www.linkedin.com/in/shubham-tade123/" target="_blank">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" width="100" />
        </a>
        <a href="https://github.com/shubhu111" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" width="100" />
        </a>
        <a href="mailto:Shubhamgtade123@gmail.com">
            <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" width="80" />
        </a>
        <a href="https://www.instagram.com/shubhamtade2068/" target="_blank">
            <img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white" width="100" />
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Shubham Tade | AI Engineer")