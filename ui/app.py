"""
Clean Chat UI - No user-facing settings.
Just a simple chat interface with file upload.
All intelligence is handled by the backend.
"""
import gradio as gr
from typing import List, Tuple, Optional
from pathlib import Path
import asyncio

from core import Orchestrator
from config import settings


# Global orchestrator
orchestrator = Orchestrator()


def create_app() -> gr.Blocks:
    """Create the Gradio application - clean, simple, no settings exposed"""
    
    with gr.Blocks(title="AI Assistant") as app:
        
        # Clean header
        gr.Markdown(
            """
            # 🤖 AI Assistant
            
            *Ask anything • Upload files • Get intelligent answers*
            """,
            elem_classes=["title"]
        )
        
        # Main chat area
        chatbot = gr.Chatbot(
            label="",
            height=500,
            elem_classes=["chatbot"],
            show_label=False,
            avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=assistant")
        )
        
        # Input area
        with gr.Row():
            with gr.Column(scale=1):
                file_upload = gr.File(
                    label="📎 Attach files",
                    file_count="multiple",
                    file_types=[
                        ".docx", ".xlsx", ".pptx", ".pdf",
                        ".csv", ".json", ".txt", ".md",
                        ".db", ".sqlite", ".qvd",
                        ".png", ".jpg", ".jpeg", ".gif", ".webp"
                    ],
                    height=80
                )
            
            with gr.Column(scale=4):
                msg_input = gr.Textbox(
                    placeholder="Ask me anything... I can search the web, read documents, and analyze images.",
                    label="",
                    show_label=False,
                    lines=2,
                    max_lines=5
                )
        
        with gr.Row():
            send_btn = gr.Button("Send", variant="primary", scale=3)
            clear_btn = gr.Button("🗑️ Clear", scale=1)
        
        # Status indicator (minimal)
        status = gr.Markdown("", elem_classes=["subtitle"])
        
        # Event handlers
        def chat_response(message: str, history: List, files: Optional[List]):
            """Process user message and generate response"""
            if not message.strip() and not files:
                return history, "", None, ""
            
            # Parse files
            doc_files = []
            image_files = []
            
            if files:
                for f in files:
                    file_path = f.name if hasattr(f, 'name') else f
                    ext = Path(file_path).suffix.lower()
                    
                    with open(file_path, 'rb') as fp:
                        file_bytes = fp.read()
                    
                    filename = Path(file_path).name
                    
                    if ext in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}:
                        image_files.append((filename, file_bytes))
                    else:
                        doc_files.append((filename, file_bytes))
            
            # Build display message
            display_msg = message
            if doc_files or image_files:
                attachments = [f[0] for f in doc_files + image_files]
                display_msg = f"{message}\n\n📎 *Attached: {', '.join(attachments)}*"
            
            # Add user message to history
            history.append({"role": "user", "content": display_msg})
            history.append({"role": "assistant", "content": ""})
            
            # Process and stream response
            response = ""
            
            try:
                # Direct sync call - orchestrator is now fully synchronous
                for chunk in orchestrator.process(
                    query=message,
                    files=doc_files if doc_files else None,
                    images=image_files if image_files else None
                ):
                    response += chunk
                    history[-1] = {"role": "assistant", "content": response}
                    yield history, "", None, ""
                    
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                history[-1] = {"role": "assistant", "content": error_msg}
                yield history, "", None, ""
        
        def clear_chat():
            """Clear chat history"""
            orchestrator.clear_history()
            return [], "", None, ""
        
        # Wire up events
        msg_input.submit(
            fn=chat_response,
            inputs=[msg_input, chatbot, file_upload],
            outputs=[chatbot, msg_input, file_upload, status]
        )
        
        send_btn.click(
            fn=chat_response,
            inputs=[msg_input, chatbot, file_upload],
            outputs=[chatbot, msg_input, file_upload, status]
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, msg_input, file_upload, status]
        )
        
        # Startup check
        def check_status():
            model_status = orchestrator.check_models()
            missing = [t.value for t, available in model_status.items() if not available]
            
            if missing:
                return f"⚠️ Some models not installed: {', '.join(missing)}"
            return ""
        
        app.load(fn=check_status, outputs=[status])
    
    return app


def launch_ui(share: bool = False, server_port: int = None):
    """Launch the application"""
    port = server_port or settings.ui_port
    app = create_app()
    app.launch(
        share=share,
        server_port=port,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    launch_ui()
