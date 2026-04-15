import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from ollama import chat, ChatResponse


def get_response(prompt: str) -> str:
    """Send the prompt to Ollama (llama3:8b) and return the response as plain text."""
    response: ChatResponse = chat(
        model='llama3:8b',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response.message.content


class RAGAssistant:
    """
    Simple RAG assistant that answers questions based on a WW1 PDF document.
    """

    def __init__(self, pdf_path: str, verbose: bool = True):
        """
        Initialize the RAG assistant.
            pdf_path: Path to the World War 1 PDF file
            verbose: Set to True to see detailed logs during processing
        """
        self.pdf_path = pdf_path
        self.verbose = verbose
        self.history = []           # Keeps track of recent conversation turns
        self.k = 3                  

        # Load the PDF once when the class starts
        self._log("Loading PDF", f"Reading file: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        self.documents = loader.load()

        self._log("PDF Loaded", f"Successfully loaded {len(self.documents)} pages.")

        self.embeddings = None
        self.faiss_index = None

    def _log(self, title: str, content: str):
        """Print clean, bordered logs when verbose mode is on."""
        if self.verbose:
            print(f"\n{'=' * 65}")
            print(f"   {title.upper()}")
            print('=' * 65)
            print(content.strip())
            print('-' * 65)

    def splitter(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """Split the PDF into smaller chunks so the model can retrieve relevant parts better."""
        self._log("Splitting Text", f"Creating chunks (size={chunk_size}, overlap={chunk_overlap})")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        return text_splitter.split_documents(self.documents)

    def create_vector_store(self, embed_model: str = "intfloat/multilingual-e5-large-instruct"):
        """
        Create or load a FAISS vector store.
        I force it to run on CPU to avoid any GPU memory headaches.
        """
        # Don't recreate everything if we already have the index
        if self.faiss_index is not None:
            return self.faiss_index

        index_path = "./faiss_index"

        # Set up embeddings (running on CPU)
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embed_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

        # Load existing index if available, otherwise build a new one
        if os.path.exists(index_path) and os.path.isdir(index_path):
            self._log("Vector Store", "Found existing FAISS index → Loading it...")
            vector_store = FAISS.load_local(
                index_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self._log("Vector Store", "No saved index found → Building a new one from the PDF...")
            texts = self.splitter()
            vector_store = FAISS.from_documents(texts, embedding=self.embeddings)
            vector_store.save_local(index_path)
            self._log("Vector Store", f"New FAISS index saved to: {index_path}")

        self.faiss_index = vector_store
        return vector_store

    def retrieve(self, question, k= 4):
        """Find the most relevant chunks from the PDF for the user's question."""
        self._log("Retrieving", f"Searching for info about: '{question}'")

        faiss_index = self.create_vector_store()

        # Add instruction prefix (works better with the e5 embedding model)
        prefixed_question = (
            "Instruct: Given a question, retrieve relevant passages that best answer the query.\n"
            f"Query: {question}"
        )

        docs = faiss_index.similarity_search_with_score(prefixed_question, k=k)

        context = ""
        log_chunks = ""

        for i, (doc, score) in enumerate(docs):
            context += doc.page_content + "\n\n"
            
            preview = doc.page_content[:320] + "..." if len(doc.page_content) > 320 else doc.page_content
            log_chunks += f"\n[Chunk {i+1} | Score: {score:.4f} | Page: {doc.metadata.get('page', 'N/A')}]\n{preview}\n"

        self._log("Retrieved Chunks", log_chunks or "No relevant chunks found.")

        if not context.strip():
            context = "No relevant information was found in the document."

        return context

    def build_prompt(self, context, history, question) -> str:
        """Build the final prompt for the LLM with context, history, and friendly guidelines."""
        template = """You are a friendly and supportive Research Partner helping users with history topics, especially World War 1 (WW1) articles and OER materials.

### GUIDELINES:
- Be warm, natural, and a little friendly in your tone.
- When the user greets you (e.g., says "hi", "hello", or similar), greet them back naturally (you can use their name "Mohamed" occasionally if it feels right, but not in every single message).
- For all other messages, respond directly to what the user asked without adding automatic greetings or openers like "Hi Mohamed!" every time.
- Do NOT act like official OER Project staff. Do not ask generic questions such as "What brings you to The OER Project today?"
- For questions about the document/PDF: Answer **only based on the RETRIEVED CONTEXT**. If the context doesn't have the information, say so clearly.
- If the question is casual or general (not about the document), respond naturally and briefly using the conversation history.
- Keep responses helpful, focused, and conversational — add a touch of friendliness without unnecessary fluff.

### CONVERSATION HISTORY:
{history}

### RETRIEVED CONTEXT (from the WW1 PDF or OER materials):
{context}

### USER QUESTION / MESSAGE:
{question}

### YOUR RESPONSE:
"""

        return template.format(
            history=history or "(No previous messages yet)",
            context=context,
            question=question
        )

    def llm_response(self, question):
        """Main method: Take a question, retrieve context, and get an answer from the LLM."""
        self._log("User Question", question)

        # Step 1: Get relevant chunks from the PDF
        context = self.retrieve(question)

        # Step 2: Prepare conversation history as a string
        history_str = ""
        for user_msg, ai_msg in self.history:
            history_str += f"User: {user_msg}\nAI: {ai_msg}\n\n"

        self._log("Conversation History", history_str or "(First message - no history yet)")

        # Step 3: Build the prompt and call Ollama
        prompt = self.build_prompt(context, history_str, question)

        self._log("Sending to LLM", "Waiting for llama3:8b response...")
        response = get_response(prompt)

        self._log("LLM Response", response)

        # Save this turn to history (keep only the last few)
        self.history.append((question, response))
        if len(self.history) > self.k:
            self.history.pop(0)

        return response