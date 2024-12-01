import streamlit as st # type: ignore
from repo_handler import RepoHandler
from code_analyzer import CodeAnalyzer
from pathlib import Path
import os
from typing import Optional

class CodeExplainerBot:
    def __init__(self):
        """Initialize the code explainer bot."""
        self.repo_handler = RepoHandler()
        self.code_analyzer = CodeAnalyzer()
        self.current_repo_url: Optional[str] = None

    def load_repository(self, repo_url: str) -> str:
        """Load a repository for analysis."""
        try:
            # Clean up previous repository if exists
            self.repo_handler.cleanup()
            self.code_analyzer.reset()
            
            # Clone new repository
            repo_path = self.repo_handler.clone_repo(repo_url)
            
            # Get all code files
            code_files = self.repo_handler.get_all_files()
            
            # Process code files
            self.code_analyzer.process_code_files(code_files)
            
            self.current_repo_url = repo_url
            return f"✅ Repository loaded successfully! You can now ask questions about the code."
            
        except Exception as e:
            return f"❌ Error loading repository: {str(e)}"

    def get_response(self, user_input: str) -> str:
        """Get response for user query."""
        if not self.current_repo_url:
            return "Please load a repository first by providing a GitHub URL."
        
        try:
            return self.code_analyzer.get_code_explanation(user_input)
        except Exception as e:
            return f"Error generating response: {str(e)}"

def main():
    st.title("💻 Code Explainer Bot")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "bot" not in st.session_state:
        st.session_state.bot = CodeExplainerBot()
    
    # Repository URL input
    repo_url = st.text_input("Enter GitHub Repository URL:")
    if repo_url:
        if repo_url != st.session_state.get("current_repo"):
            with st.spinner("Loading repository..."):
                response = st.session_state.bot.load_repository(repo_url)
                st.session_state.current_repo = repo_url
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about the code (e.g., 'explain the main function in app.py')"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.spinner("Analyzing..."):
            response = st.session_state.bot.get_response(prompt)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main() 