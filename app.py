import gradio as gr
from rag import RAGAssistant   # Make sure the file is named rag_assistant.py

# ========================= CONFIG =========================
PDF_PATH = "WHP 7721 Read  The Course of WWI  1040L.pdf"   # Change if needed
PORT = 8001                                             # Change this if you still get port error (e.g. 7880, 8000)

# Initialize the RAG pipeline once when the app starts
print("Loading PDF and creating vector store... This may take a minute the first time.")
pipeline = RAGAssistant(pdf_path=PDF_PATH, verbose=False)   # verbose=False for cleaner terminal
print("✅ PDF loaded successfully! Starting Gradio app...\n")

def response(message, history):
    """Function called by Gradio every time user sends a message."""
    if not message or not message.strip():
        return "Please ask a question about the document."
    
    try:
        # Get answer from your RAG system
        bot_message = pipeline.llm_response(message)
        return bot_message
    except Exception as e:
        return f"❌ Sorry, an error occurred: {str(e)}"


# ====================== Gradio Chat Interface ======================
demo = gr.ChatInterface(
    fn=response,
    title="📄 WWI History Research Assistant",
    description="Ask any questions about the uploaded PDF document: **'The Course of WWI'**",
    examples=[
        "What is the main topic of this document?",
        "Summarize the causes of World War I.",
        "What were the major events during the war?",
        "Who were the key figures mentioned?",
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",   # Localhost (safe on Windows)
        server_port=PORT,          # You can change PORT at the top
        share=False,               # Set to True for a temporary public link
        debug=True                 # Shows more helpful error messages
    )

