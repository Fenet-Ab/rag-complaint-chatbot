import gradio as gr
from src.rag_pipeline import rag_answer

def chat(question):
    answer, sources = rag_answer(question)
    source_text = "\n\n---\n\n".join(sources[:2])
    return answer, source_text

interface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="Ask a question"),
    outputs=[
        gr.Textbox(label="AI Answer"),
        gr.Textbox(label="Sources Used")
    ],
    title="CrediTrust Complaint Analysis Chatbot",
    description="Ask questions about customer complaints across financial products."
)

interface.launch()
