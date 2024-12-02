from typing import Dict, List, Optional
from pathlib import Path
import os
from dotenv import load_dotenv # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain_community.vectorstores import Chroma # type: ignore  
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_openai import ChatOpenAI # type: ignore
from langchain.chains import ConversationalRetrievalChain # type: ignore
from langchain.memory import ConversationBufferMemory # type: ignore
from bs4 import BeautifulSoup # type: ignore
import markdown # type: ignore
import time
from tenacity import ( # type: ignore
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import requests.exceptions
import re

# Load environment variables
load_dotenv()

class CodeAnalyzer:
    def __init__(self):
        """Initialize the code analyzer with necessary components."""
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
            
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize LLM with GPT-4
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.2,
            request_timeout=90,  # Increased timeout for GPT-4
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize text splitter with larger chunks for GPT-4's context window
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,   # Increased chunk size for GPT-4
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.vector_store = None
        self.conversation_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question"
        )
        
        # Keep track of processed files and their contents
        self.file_map = {}
        self.file_contents = {}

    def process_code_files(self, files: List[Path]) -> None:
        """Process code files and create a vector store."""
        if not files:
            print("No files found to analyze")
            return
        
        documents = []
        
        for file_path in files:
            try:
                # Read file content
                content = file_path.read_text(encoding='utf-8')
                
                # Skip empty files
                if not content.strip():
                    continue
                
                # Store the original content
                self.file_contents[str(file_path)] = content
                
                # Add file metadata as context
                file_context = f"""
                File: {file_path.name}
                Path: {file_path}
                Type: {file_path.suffix[1:]} file
                """
                
                # Create chunks for the content
                chunks = self.text_splitter.split_text(content)
                
                # Store file path for each chunk
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():  # Skip empty chunks
                        continue
                    doc_id = f"{file_path}_{i}"
                    self.file_map[doc_id] = str(file_path)
                    documents.append({
                        "text": chunk,
                        "metadata": {
                            "file": str(file_path),
                            "chunk_id": i,
                            "file_type": file_path.suffix[1:],
                            "file_name": file_path.name
                        }
                    })
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue

        if documents:
            try:
                # Create vector store
                self.vector_store = Chroma.from_texts(
                    texts=[doc["text"] for doc in documents],
                    embedding=self.embeddings,
                    collection_name="code_chunks",
                    metadatas=[doc["metadata"] for doc in documents],
                    persist_directory="./chroma_db"
                )
                
                # Create conversation chain with improved system prompt
                self.conversation_chain = ConversationalRetrievalChain.from_llm(
                    self.llm,
                    retriever=self.vector_store.as_retriever(
                        search_kwargs={
                            "k": 6  # Increased relevant documents for better context
                        }
                    ),
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=True
                )
            except Exception as e:
                print(f"Error creating vector store: {str(e)}")
                raise

    def _find_file_path(self, query: str) -> Optional[str]:
        """Find the full file path from a query, handling nested folders."""
        query_lower = query.lower()
        
        # First try exact matches
        for file_path in self.file_map.values():
            if file_path.lower() in query_lower:
                return file_path
        
        # Then try filename matches
        for file_path in self.file_map.values():
            file_name = Path(file_path).name
            if file_name.lower() in query_lower:
                return file_path
        
        # Try partial path matches (e.g., "components/Button.tsx")
        for file_path in self.file_map.values():
            path_parts = Path(file_path).parts
            for i in range(len(path_parts)):
                partial_path = str(Path(*path_parts[-(i+1):]))
                if partial_path.lower() in query_lower:
                    return file_path
        
        return None

    def get_code_explanation(self, query: str) -> str:
        """Get explanation for code-related queries with formatted snippets."""
        if not self.conversation_chain:
            return "❌ Please load a repository first."
        
        try:
            # Detect query type
            is_logic_question = any(keyword in query.lower() for keyword in [
                'can i', 'how to', 'is it possible', 'what if', 'should i',
                'better to', 'difference between', 'compare', 'versus', 'vs'
            ])
            
            # Extract file name from query if present
            file_match = re.search(r'(?:explain|show|analyze).*?(?:code|file).*?[\'"`]?([\w\-./]+\.\w+)', query.lower())
            show_code = 'show' in query.lower() or 'display' in query.lower()
            
            if file_match:
                requested_file = file_match.group(1)
                
                # Find the closest matching file
                matching_files = [f for f in self.file_map.values() if requested_file.lower() in str(f).lower()]
                if not matching_files:
                    return f"""❌ File '{requested_file}' not found. Available files:
                    
```
{chr(10).join(['- ' + str(f) for f in sorted(set(self.file_map.values()))])}
```"""
                
                requested_file = str(matching_files[0])
                file_content = self.file_contents.get(requested_file)
                
                if not file_content:
                    return f"❌ Couldn't read contents of '{requested_file}'"
                
                # Format the response with code snippet and explanation
                file_extension = Path(requested_file).suffix[1:]
                enhanced_query = f"""
                Analyze this code file and provide a detailed explanation:
                
                File: {Path(requested_file).name}
                Path: {requested_file}
                
                ```{file_extension}
                {file_content}
                ```
                
                Please provide:
                1. A brief overview of the file's purpose
                2. Key components and their functionality
                3. Important methods/functions and their roles
                4. Any notable patterns or design choices
                5. Dependencies and their usage
                
                Format the response with clear sections and include relevant code snippets when explaining specific parts.
                """
                
                response = self.conversation_chain({"question": enhanced_query})
                
                # Format the response
                formatted_response = f"""## 📄 {Path(requested_file).name}

{response['answer']}

**File Location:** `{requested_file}`
"""
                
                # Add source information if available
                if hasattr(response, 'source_documents') and response.source_documents:
                    sources = set(doc.metadata['file'] for doc in response.source_documents)
                    formatted_response += "\n\n**Related Files:**\n" + "\n".join(f"- `{src}`" for src in sources)
                
                return formatted_response
                
            else:
                # General code query
                return f"""Please specify which file you'd like me to explain. Available files:

```
{chr(10).join(['- ' + str(f) for f in sorted(set(self.file_map.values()))])}
```

Example queries:
- "Explain the code in app.py"
- "Show me the contents of utils.js"
- "Analyze the code in src/components/Header.tsx"
"""
                
        except Exception as e:
            print(f"Error: {str(e)}")
            return f"❌ Error analyzing code: {str(e)}"

    def reset(self):
        """Reset the analyzer state."""
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
            except:
                pass
            self.vector_store = None
        if self.conversation_chain:
            self.conversation_chain = None
        self.memory.clear()
        self.file_map.clear()
        self.file_contents.clear()  # Clear stored file contents