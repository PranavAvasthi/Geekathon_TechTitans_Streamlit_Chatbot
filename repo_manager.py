import os
import shutil
import tempfile
from git import Repo # type: ignore
from typing import Optional
from pathlib import Path

class RepoManager:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.current_repo: Optional[Repo] = None
        self.repo_path: Optional[Path] = None

    def clone_repository(self, repo_url: str) -> Path:
        """Clone a repository and return its path."""
        try:
            # Create a unique directory name from the repo URL
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            self.repo_path = Path(self.temp_dir) / repo_name
            
            # Clone the repository
            self.current_repo = Repo.clone_from(repo_url, self.repo_path)
            return self.repo_path
        except Exception as e:
            raise Exception(f"Failed to clone repository: {str(e)}")

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def get_file_content(self, file_path: str) -> str:
        """Get content of a file from the repository."""
        full_path = self.repo_path / file_path
        if full_path.exists():
            return full_path.read_text(encoding='utf-8')
        raise FileNotFoundError(f"File {file_path} not found in repository")

    def get_all_files(self, extensions=None):
        """Get all files in the repository with specified extensions."""
        if not extensions:
            extensions = ['.py', '.js', '.java', '.cpp', '.hpp', '.h', '.c', '.cs']
        
        files = []
        for ext in extensions:
            files.extend(self.repo_path.rglob(f"*{ext}"))
        return files 