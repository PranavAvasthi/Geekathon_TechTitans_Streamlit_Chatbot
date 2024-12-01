import os
import shutil
from git import Repo # type: ignore
from pathlib import Path
import tempfile
from typing import Optional, List

class RepoHandler:
    def __init__(self):
        """Initialize the repository handler with a temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path: Optional[Path] = None
        self.current_repo = None

    def clone_repo(self, repo_url: str) -> Path:
        """Clone a repository and return its path."""
        try:
            # Extract repo name from URL
            repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
            self.repo_path = Path(self.temp_dir) / repo_name
            
            # Clone the repository
            self.current_repo = Repo.clone_from(repo_url, self.repo_path)
            return self.repo_path
        except Exception as e:
            raise Exception(f"Failed to clone repository: {str(e)}")

    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get the content of a specific file."""
        try:
            if self.repo_path:
                full_path = self.repo_path / file_path
                if full_path.exists() and full_path.is_file():
                    return full_path.read_text(encoding='utf-8')
            return None
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return None

    def get_all_files(self, extensions: Optional[List[str]] = None) -> List[Path]:
        """Get all files in the repository with specified extensions."""
        if not extensions:
            extensions = ['.ts', '.tsx', '.js', '.jsx', '.css', '.json']  # Common React Native extensions
        
        files = []
        if self.repo_path:
            for ext in extensions:
                # Skip common directories to avoid processing unnecessary files
                for file_path in self.repo_path.rglob(f"*{ext}"):
                    if not any(part.startswith('.') or part in {
                        'node_modules', 'venv', '__pycache__', 'build', 'dist',
                        '.expo', '.expo-shared', 'android', 'ios'  # Skip React Native specific directories
                    } for part in file_path.parts):
                        files.append(file_path)
        return files

    def cleanup(self):
        """Clean up temporary files."""
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error cleaning up: {str(e)}") 