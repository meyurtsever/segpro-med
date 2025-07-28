"""
Login UI component for crowdsourcing authentication
"""

import gradio as gr
import logging
from auth.auth_manager import AuthManager

logger = logging.getLogger(__name__)

def create_login_interface(auth_callback):
    """Create the login interface"""
    
    auth_manager = AuthManager()
    
    def handle_login(username, password):
        """Handle login attempt"""
        if not username or not password:
            return "Please enter both username and password", None
        
        user_data = auth_manager.authenticate(username, password)
        if user_data:
            logger.info(f"Successful login for user: {username} (role: {user_data['role']})")
            auth_callback(user_data)
            return f"Login successful! Welcome {username}", user_data
        else:
            logger.warning(f"Failed login attempt for username: {username}")
            return "Invalid username or password", None
    
    with gr.Blocks(title="SegMed-Pro Login") as login_interface:
        gr.Markdown("# SegMed-Pro: Medical Imaging Annotation Tool")
        gr.Markdown("## Please login to continue")
        
        with gr.Row():
            with gr.Column(scale=1):
                pass  # Empty column for centering
            
            with gr.Column(scale=2):
                username_input = gr.Textbox(
                    label="Username",
                    placeholder="Enter your username",
                    interactive=True
                )
                
                password_input = gr.Textbox(
                    label="Password",
                    placeholder="Enter your password",
                    type="password",
                    interactive=True
                )
                
                login_button = gr.Button("Login", variant="primary")
                
                login_status = gr.Markdown("", visible=False)
                
                # Demo credentials info
                gr.Markdown("""
                ### Demo Credentials:
                **Admin:** username: `admin1`, password: `adminpass`  
                **Expert:** username: `john_doe`, password: `pass123`  
                **Expert:** username: `jane_smith`, password: `expert456`
                """)
            
            with gr.Column(scale=1):
                pass  # Empty column for centering
        
        # Event handlers
        login_button.click(
            fn=handle_login,
            inputs=[username_input, password_input],
            outputs=[login_status, gr.State()]
        ).then(
            fn=lambda msg, user_data: gr.update(value=msg, visible=True),
            inputs=[login_status, gr.State()],
            outputs=[login_status]
        )
        
        # Allow Enter key to trigger login
        password_input.submit(
            fn=handle_login,
            inputs=[username_input, password_input],
            outputs=[login_status, gr.State()]
        ).then(
            fn=lambda msg, user_data: gr.update(value=msg, visible=True),
            inputs=[login_status, gr.State()],
            outputs=[login_status]
        )
    
    return login_interface
