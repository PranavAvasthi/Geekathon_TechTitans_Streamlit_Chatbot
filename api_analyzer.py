import requests
from typing import Dict, Any, Optional
import json

class GitAPIAnalyzer:
    def __init__(self, base_url: str | None = None):
        self.base_url = base_url.rstrip('/') if base_url else None
        
    async def analyze_repository(self, repo_url: str) -> Dict[str, Any]:
        """Comprehensive repository analysis using the API."""
        try:
            if not self.base_url:
                return {}  # Skip API analysis if no base URL is provided
            
            results = {}
            
            # Get commit analysis
            commit_response = await self._make_request(
                f"{self.base_url}/api/v1/commits",
                {"url": repo_url}
            )
            if commit_response:
                results["commit_analysis"] = commit_response
            
            # Get pull requests analysis
            pr_response = await self._make_request(
                f"{self.base_url}/api/v1/pull-requests",
                {"url": repo_url}
            )
            if pr_response:
                results["pull_requests"] = pr_response
            
            return results
        except Exception as e:
            return {"error": f"API analysis failed: {str(e)}"}
    
    async def get_user_activity(self, repo_url: str, username: str) -> Dict[str, Any]:
        """Get detailed user activity analysis."""
        try:
            response = await self._make_request(
                f"{self.base_url}/api/v1/commits/{username}",
                {"url": repo_url}
            )
            return response or {}
        except Exception as e:
            return {"error": f"Failed to get user activity: {str(e)}"}
    
    async def _make_request(self, url: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API request with error handling."""
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {str(e)}")
            return None 