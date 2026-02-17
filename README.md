# ST-GPT

# 🤖 ST-GPT: GenAI Super App

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=googlebard&logoColor=white)

**ST-GPT** is an all-in-one Generative AI application built with **Streamlit** and **Google Gemini**. It serves as a personal AI assistant, a video analyst, and a document researcher, all in a single interface.

## 🚀 Features

### 1. 🤖 AI Friend & Helper
* A conversational chatbot acting as a supportive friend and tutor.
* **Memory Enabled:** Remembers your conversation context.
* **Persona:** Friendly, empathetic, and capable of teaching coding/math concepts.
* **Powered by:** `Llama-3.3-70B-Instruct` for fast responses.

### 2. 📹 Chat with YouTube (RAG)
* **Input:** Paste any YouTube video URL.
* **Process:** Extracts transcripts, creates embeddings, and stores them in a vector database.
* **Output:** Ask specific questions about the video content and get accurate answers with timestamps context.
* **Tech:** `YouTubeTranscriptApi`, `FAISS`, `LangChain RAG`.

### 3. 📄 Chat with PDF (RAG)
* **Input:** Upload a PDF document (Resumes, Reports, Books).
* **Process:** Splits text into chunks and indexes them for retrieval.
* **Output:** Interact with your document to summarize or extract specific details.
* **Tech:** `PyPDFLoader`, `RecursiveCharacterTextSplitter`.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** Llama 3.3-70B-Instruct (Groq), Google Gemini Flash
* **Framework:** LangChain
* **Vector Database:** FAISS (In-Memory)
* **Embedding Model:** `gemini-embedding-001`

---

## 💻 Installation & Setup

Follow these steps to run the project locally:

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/shubhu111/ST-GPT.git](https://github.com/shubhu111/ST-GPT.git)
    cd ST-GPT
    ```

2.  **Create a Virtual Environment** (Optional but Recommended)
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up API Keys**
    * Create a `.env` file in the root directory.
    * Add your Google API Key:
    ```text
    GOOGLE_API_KEY=your_actual_api_key_here
    ```

5.  **Run the App**
    ```bash
    streamlit run app.py
    ```

---

## 📂 Project Structure

```text
ST-GPT/
├── app.py                # Main application file (UI + Logic)
├── requirements.txt      # List of dependencies
├── .env                  # API Keys (Not uploaded to GitHub)
└── README.md             # Project Documentation


📬 Connect with Me
- Developer: Shubham Tade
- Role: AI Engineer
- LinkedIn: Shubham Tade
- GitHub: shubhu111
- Email: Shubhamgtade123@gmail.com
