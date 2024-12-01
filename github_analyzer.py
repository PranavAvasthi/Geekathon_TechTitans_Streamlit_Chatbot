import github3 # type: ignore
from datetime import datetime
import re
from streamlit_code_analysis.repo_manager import RepoManager
from code_analyzer import CodeAnalyzer
from typing import Dict, Any
import os
from pydriller import Repository # type: ignore
from api_analyzer import GitAPIAnalyzer
import asyncio

class GitHubAnalyzer:
    def __init__(self, github_token):
        if not github_token:
            raise ValueError("GitHub token is required. Please set GITHUB_TOKEN in your .env file")
        
        self.github = github3.login(token=github_token)
        if not self.github:
            raise ValueError("Failed to authenticate with GitHub. Please check your token")
        
        self.repo_manager = RepoManager()
        self.code_analyzer = None
        self.api_analyzer = GitAPIAnalyzer(os.getenv("GIT_API_BASE_URL"))

    def _extract_repo_info(self, repo_url):
        """Extract owner and repo name from various GitHub URL formats."""
        pattern = r"github\.com[/:]([^/]+)/([^/]+?)(?:\.git)?/?$"
        match = re.search(pattern, repo_url)
        
        if not match:
            raise ValueError("Invalid GitHub repository URL format")
        
        return match.group(1), match.group(2)

    def analyze_repository(self, repo_url: str) -> Dict[str, Any]:
        """Perform comprehensive repository analysis."""
        try:
            if not repo_url:
                return {"error": "Please provide a GitHub repository URL"}

            # Validate GitHub token
            if not self.github:
                return {
                    "error": "GitHub authentication failed. Please check your GITHUB_TOKEN",
                    "help": "Make sure you have set a valid GitHub token in your .env file"
                }

            # Extract repo information
            try:
                owner, repo_name = self._extract_repo_info(repo_url)
            except ValueError as e:
                return {"error": f"Invalid repository URL: {str(e)}"}
            
            # Get GitHub repository
            try:
                gh_repo = self.github.repository(owner, repo_name)
                if not gh_repo:
                    return {"error": f"Repository {owner}/{repo_name} not found"}
            except github3.exceptions.NotFoundError:
                return {
                    "error": f"Repository {owner}/{repo_name} not found or private",
                    "help": "Make sure the repository exists and your token has access to it"
                }
            except github3.exceptions.AuthenticationFailed:
                return {
                    "error": "GitHub authentication failed",
                    "help": "Please check your GitHub token permissions"
                }
            except Exception as e:
                return {"error": f"Failed to access repository: {str(e)}"}

            # Clone repository
            try:
                repo_path = self.repo_manager.clone_repository(repo_url)
                self.code_analyzer = CodeAnalyzer(str(repo_path))
            except Exception as e:
                return {"error": f"Failed to clone repository: {str(e)}"}

            # Get API analysis
            api_metrics = asyncio.run(self.api_analyzer.analyze_repository(repo_url))
            
            # Collect all metrics
            metrics = {}
            
            try:
                metrics["basic_info"] = self._get_basic_info(gh_repo)
            except Exception as e:
                metrics["basic_info"] = {"error": f"Failed to get basic info: {str(e)}"}

            try:
                metrics["code_analysis"] = self._analyze_codebase(repo_path)
                if "files" in metrics["code_analysis"]:
                    del metrics["code_analysis"]["files"]
            except Exception as e:
                metrics["code_analysis"] = {"error": f"Failed to analyze code: {str(e)}"}

            # Merge API metrics with local analysis
            if "commit_analysis" in api_metrics:
                metrics["git_history"] = api_metrics["commit_analysis"]
            else:
                try:
                    metrics["git_history"] = self.code_analyzer.analyze_git_history()
                    if "commit_messages" in metrics["git_history"]:
                        del metrics["git_history"]["commit_messages"]
                except Exception as e:
                    metrics["git_history"] = {"error": f"Failed to analyze git history: {str(e)}"}

            if "pull_requests" in api_metrics:
                metrics["pull_requests"] = api_metrics["pull_requests"]

            try:
                metrics["team_metrics"] = self._analyze_team_metrics(gh_repo)
            except Exception as e:
                metrics["team_metrics"] = {"error": f"Failed to analyze team metrics: {str(e)}"}

            return metrics

        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}
        finally:
            try:
                self.repo_manager.cleanup()
            except Exception:
                pass

    def _get_basic_info(self, repo) -> Dict[str, Any]:
        """Get basic repository information."""
        return {
            "name": repo.name,
            "description": repo.description,
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "open_issues": repo.open_issues_count,
        }

    def _analyze_codebase(self, repo_path) -> Dict[str, Any]:
        """Analyze entire codebase."""
        analysis = {
            "files": {},
            "overall_metrics": {
                "total_files": 0,
                "total_lines": 0,
                "total_complexity": 0,
                "average_maintainability": 0
            }
        }

        files = self.repo_manager.get_all_files()
        for file_path in files:
            if file_path.is_file():
                file_analysis = self.code_analyzer.analyze_code_quality(str(file_path))
                relative_path = str(file_path.relative_to(repo_path))
                analysis["files"][relative_path] = file_analysis
                
                # Update overall metrics
                analysis["overall_metrics"]["total_files"] += 1
                if "raw_metrics" in file_analysis:
                    analysis["overall_metrics"]["total_lines"] += \
                        file_analysis["raw_metrics"].get("loc", 0)
                if "complexity" in file_analysis:
                    analysis["overall_metrics"]["total_complexity"] += \
                        file_analysis["complexity"].get("average_complexity", 0)

        if analysis["overall_metrics"]["total_files"] > 0:
            analysis["overall_metrics"]["average_maintainability"] = \
                analysis["overall_metrics"]["total_complexity"] / \
                analysis["overall_metrics"]["total_files"]

        return analysis

    def _analyze_team_metrics(self, repo) -> Dict[str, Any]:
        """Analyze team-related metrics."""
        team_metrics = {
            "contributors": [],
            "pull_requests": {
                "open": 0,
                "closed": 0,
                "merged": 0
            },
            "issues": {
                "open": 0,
                "closed": 0
            }
        }

        try:
            # Analyze contributors
            for contributor in repo.contributors():
                team_metrics["contributors"].append({
                    "login": contributor.login,
                    "contributions": contributor.contributions
                })
        except Exception as e:
            team_metrics["contributors"] = [{"error": f"Failed to fetch contributors: {str(e)}"}]

        try:
            # Analyze pull requests
            for pr in repo.pull_requests(state='all'):
                if pr.state == 'open':
                    team_metrics["pull_requests"]["open"] += 1
                elif getattr(pr, 'merged_at', None) is not None:
                    team_metrics["pull_requests"]["merged"] += 1
                else:
                    team_metrics["pull_requests"]["closed"] += 1
        except Exception as e:
            team_metrics["pull_requests"]["error"] = f"Failed to analyze PRs: {str(e)}"

        try:
            # Analyze issues
            for issue in repo.issues(state='all'):
                if issue.state == 'open':
                    team_metrics["issues"]["open"] += 1
                else:
                    team_metrics["issues"]["closed"] += 1
        except Exception as e:
            team_metrics["issues"]["error"] = f"Failed to analyze issues: {str(e)}"

        return team_metrics