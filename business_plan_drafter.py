#!/usr/bin/env python3
"""
Business Plan Drafter - A CLI tool to generate business plan drafts
using Mural board embeddings and OpenAI's GPT.
"""

import os
import sys
import json
import time
import webbrowser
import http.server
import socketserver
import urllib.parse
import base64
import secrets
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from tabulate import tabulate
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
import re

# Load environment variables
load_dotenv()

# Initialize console for rich text output
console = Console()

# API Authentication
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MURAL_CLIENT_ID = os.getenv("MURAL_CLIENT_ID")
MURAL_CLIENT_SECRET = os.getenv("MURAL_CLIENT_SECRET")

# File paths
CONFIG_DIR = Path.home() / ".business_plan_drafter"
CONFIG_FILE = CONFIG_DIR / "config.json"
TOKEN_FILE = CONFIG_DIR / "tokens.json"

# Create config directory if it doesn't exist
CONFIG_DIR.mkdir(exist_ok=True)

# OAuth callback server for local authentication
AUTH_SERVER_PORT = 8085
AUTH_REDIRECT_PATH = "/oauth/callback"
OAUTH_CALLBACK_URL = f"http://localhost:{AUTH_SERVER_PORT}{AUTH_REDIRECT_PATH}"

# Mural OAuth URLs
MURAL_AUTH_URL = "https://app.mural.co/api/public/v1/authorization/oauth2/"
MURAL_TOKEN_URL = "https://app.mural.co/api/public/v1/authorization/oauth2/token"
MURAL_API_BASE_URL = "https://app.mural.co/api/public/v1"

# OAuth scope (read access to murals and workspaces)
MURAL_SCOPES = "murals:read workspaces:read"


class OAuthCallbackHandler(http.server.SimpleHTTPRequestHandler):
    """Handler for OAuth callback."""

    auth_code = None
    state = None

    def do_GET(self):
        """Handle GET requests to the callback URL."""
        if self.path.startswith(AUTH_REDIRECT_PATH):
            # Parse the query parameters
            query = urllib.parse.urlparse(self.path).query
            query_params = urllib.parse.parse_qs(query)

            if "code" in query_params and "state" in query_params:
                OAuthCallbackHandler.auth_code = query_params["code"][0]
                OAuthCallbackHandler.state = query_params["state"][0]

                # Send a success response to the browser
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()

                html_response = """
                <html>
                <head><title>Authorization Successful</title></head>
                <body>
                <h1>Authorization Successful!</h1>
                <p>You've successfully authorized Business Plan Drafter to access your Mural account.</p>
                <p>You can now close this window and return to the application.</p>
                </body>
                </html>
                """
                self.wfile.write(html_response.encode())
            else:
                # Handle error
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()

                html_response = """
                <html>
                <head><title>Authorization Failed</title></head>
                <body>
                <h1>Authorization Failed</h1>
                <p>There was an error during authorization. Please try again.</p>
                </body>
                </html>
                """
                self.wfile.write(html_response.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress HTTP server logs."""
        return


class MuralOAuth:
    """Class to handle Mural OAuth authentication."""

    def __init__(self, client_id: str, client_secret: str):
        """Initialize Mural OAuth handler."""
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_file = TOKEN_FILE
        self.tokens = self._load_tokens()

    def _load_tokens(self) -> Dict[str, Any]:
        """Load OAuth tokens from file."""
        if self.token_file.exists():
            try:
                with open(self.token_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_tokens(self, tokens: Dict[str, Any]):
        """Save OAuth tokens to file."""
        with open(self.token_file, "w") as f:
            json.dump(tokens, f)
        self.tokens = tokens

    def get_access_token(self) -> Optional[str]:
        """Get a valid access token, refreshing if necessary."""
        if not self.tokens:
            # No tokens available, need to authenticate
            return None

        # Check if the access token is expired
        expires_at = self.tokens.get("expires_at", 0)

        if time.time() > expires_at:
            # Token is expired, refresh it
            return self.refresh_token()

        return self.tokens.get("access_token")

    def refresh_token(self) -> Optional[str]:
        """Refresh the access token using the refresh token."""
        if not self.tokens or "refresh_token" not in self.tokens:
            # No refresh token available
            return None

        refresh_token = self.tokens["refresh_token"]

        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }

        try:
            response = requests.post(MURAL_TOKEN_URL, data=data)
            response.raise_for_status()

            token_data = response.json()
            # Update tokens with new access token
            self.tokens.update(
                {
                    "access_token": token_data["access_token"],
                    "expires_at": time.time()
                    + token_data["expires_in"]
                    - 60,  # 60 seconds buffer
                    "refresh_token": token_data.get(
                        "refresh_token", self.tokens["refresh_token"]
                    ),
                }
            )

            self._save_tokens(self.tokens)
            return self.tokens["access_token"]

        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]Error refreshing token: {str(e)}[/bold red]")
            # Token refresh failed, need to re-authenticate
            return None

    def authenticate(self) -> bool:
        """Perform OAuth authentication flow."""
        # Generate a random state value for security
        state = secrets.token_urlsafe(16)

        # Build the authorization URL
        auth_url = f"{MURAL_AUTH_URL}?client_id={self.client_id}&redirect_uri={OAUTH_CALLBACK_URL}&scope={MURAL_SCOPES}&state={state}&response_type=code"

        # Open the browser for user authentication
        console.print("\n[cyan]Opening browser for Mural authentication...[/cyan]")
        webbrowser.open(auth_url)

        # Start a local server to handle the callback
        server = socketserver.TCPServer(("", AUTH_SERVER_PORT), OAuthCallbackHandler)

        # Run the server in a separate thread
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        # Wait for the callback (with timeout)
        timeout = 300  # 5 minutes
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Waiting for authentication...", total=None)

            while OAuthCallbackHandler.auth_code is None:
                if time.time() - start_time > timeout:
                    server.shutdown()
                    console.print(
                        "[bold red]Authentication timeout. Please try again.[/bold red]"
                    )
                    return False
                time.sleep(0.5)

            # Verify the state parameter
            if OAuthCallbackHandler.state != state:
                server.shutdown()
                console.print(
                    "[bold red]Authentication failed: State mismatch. Please try again.[/bold red]"
                )
                return False

            # Exchange the code for tokens
            auth_code = OAuthCallbackHandler.auth_code

            # Reset the handler variables for future use
            OAuthCallbackHandler.auth_code = None
            OAuthCallbackHandler.state = None

            # Stop the server
            server.shutdown()
            server.server_close()

            # Exchange code for tokens
            data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": auth_code,
                "redirect_uri": OAUTH_CALLBACK_URL,
                "grant_type": "authorization_code",
            }

            progress.update(task, description="[cyan]Exchanging code for tokens...")

            try:
                response = requests.post(MURAL_TOKEN_URL, data=data)
                response.raise_for_status()

                token_data = response.json()

                # Save the tokens
                tokens = {
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data["refresh_token"],
                    "expires_at": time.time()
                    + token_data["expires_in"]
                    - 60,  # 60 seconds buffer
                }

                self._save_tokens(tokens)
                progress.update(task, description="[green]Authentication successful!")
                return True

            except requests.exceptions.RequestException as e:
                console.print(
                    f"[bold red]Error exchanging code for tokens: {str(e)}[/bold red]"
                )
                return False


class MuralAPI:
    """Class to handle all interactions with the Mural API."""

    def __init__(self, oauth: MuralOAuth):
        self.oauth = oauth
        self.base_url = MURAL_API_BASE_URL

    def _get_headers(self) -> Dict[str, str]:
        """Get the headers needed for API requests."""
        access_token = self.oauth.get_access_token()

        if not access_token:
            # Need to authenticate
            if not self.oauth.authenticate():
                console.print("[bold red]Authentication failed. Exiting.[/bold red]")
                sys.exit(1)
            access_token = self.oauth.get_access_token()

        return {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def get_workspaces(self) -> List[Dict[str, Any]]:
        """Fetch all workspaces accessible to the user."""
        url = f"{self.base_url}/workspaces"
        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json().get("value", [])

    def get_murals(self, workspace_id: str) -> List[Dict[str, Any]]:
        """Fetch all murals in a workspace."""
        url = f"{self.base_url}/workspaces/{workspace_id}/murals"
        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json().get("value", [])

    def get_mural_content(self, mural_id: str) -> Dict[str, Any]:
        """Fetch content of a specific mural using the official Mural API format."""
        try:
            # First check if we can access the mural itself
            mural_url = f"{self.base_url}/murals/{mural_id}"
            console.print(f"[cyan]Attempting to access mural: {mural_url}[/cyan]")

            mural_response = requests.get(mural_url, headers=self._get_headers())

            # If we can't access the mural directly, try with just the numeric ID part
            if mural_response.status_code != 200 and "." in mural_id:
                short_id = mural_id.split(".")[1]  # Get the part after the dot
                mural_url = f"{self.base_url}/murals/{short_id}"
                console.print(f"[cyan]Trying with numeric ID only: {mural_url}[/cyan]")
                mural_response = requests.get(mural_url, headers=self._get_headers())

                if mural_response.status_code == 200:
                    # If the short ID worked, use it for future calls
                    mural_id = short_id
                    console.print(
                        f"[cyan]Using shortened mural ID for future calls: {mural_id}[/cyan]"
                    )

            # If we still can't access the mural, raise an exception
            if mural_response.status_code != 200:
                console.print(
                    f"[yellow]Could not access mural with ID {mural_id}. Status code: {mural_response.status_code}[/yellow]"
                )
                mural_response.raise_for_status()

            # We successfully accessed the mural, now get its widgets
            console.print(f"[green]Successfully accessed mural![/green]")

            # Now get the widgets for this mural, using the ORIGINAL ID that worked for accessing the mural
            widgets_url = f"{self.base_url}/murals/{mural_id}/widgets"
            console.print(f"[cyan]Fetching widgets from: {widgets_url}[/cyan]")

            widgets_response = requests.get(widgets_url, headers=self._get_headers())
            if widgets_response.status_code != 200:
                console.print(
                    f"[yellow]Failed to get widgets. Status code: {widgets_response.status_code}[/yellow]"
                )

                # Try one more approach - use the returned ID if available
                if "id" in mural_response.json():
                    api_mural_id = mural_response.json()["id"]
                    if api_mural_id != mural_id:
                        console.print(
                            f"[cyan]Trying with API-provided mural ID: {api_mural_id}[/cyan]"
                        )
                        alt_widgets_url = (
                            f"{self.base_url}/murals/{api_mural_id}/widgets"
                        )
                        widgets_response = requests.get(
                            alt_widgets_url, headers=self._get_headers()
                        )

                        if widgets_response.status_code == 200:
                            console.print(
                                f"[green]Successfully fetched mural widgets with API ID![/green]"
                            )
                            return widgets_response.json()

                # If all attempts failed, raise the exception
                widgets_response.raise_for_status()

            console.print(f"[green]Successfully fetched mural widgets![/green]")
            return widgets_response.json()

        except requests.exceptions.RequestException as e:
            console.print(
                f"[bold red]Error fetching mural content: {str(e)}[/bold red]"
            )
            console.print(
                "[yellow]Failed to access mural content. Continuing with empty content...[/yellow]"
            )
            return {}

    def extract_mural_text(self, mural_content: Dict[str, Any]) -> List[str]:
        """Extract text content from mural widgets."""
        text_items = []

        # Based on the official Mural API, we're working with widgets data
        widgets = []

        # The API appears to return widgets in a 'value' field
        if "value" in mural_content and isinstance(mural_content["value"], list):
            widgets = mural_content["value"]
            console.print(
                f"[cyan]Found widgets in 'value' field: {len(widgets)} widgets[/cyan]"
            )
        # Fallback to looking for a 'widgets' field
        elif "widgets" in mural_content and isinstance(mural_content["widgets"], list):
            widgets = mural_content["widgets"]
            console.print(
                f"[cyan]Found widgets in 'widgets' field: {len(widgets)} widgets[/cyan]"
            )
        # Check if the response is a list directly
        elif isinstance(mural_content, list):
            widgets = mural_content
            console.print(
                f"[cyan]Found widgets in direct list: {len(widgets)} widgets[/cyan]"
            )
        # As a fallback, check if there's a 'data' field
        elif "data" in mural_content and isinstance(mural_content["data"], list):
            widgets = mural_content["data"]
            console.print(
                f"[cyan]Found widgets in 'data' field: {len(widgets)} widgets[/cyan]"
            )

        console.print(f"[cyan]Found {len(widgets)} widgets in mural[/cyan]")

        # If we have a "next" field, it means there are more pages of widgets
        if "next" in mural_content and mural_content["next"]:
            console.print(
                f"[yellow]Note: There are more widgets available on additional pages[/yellow]"
            )

        # Track how many widgets had text content
        text_count = 0

        # Process widgets (sticky notes, text boxes, etc.)
        for widget in widgets:
            text = None
            widget_type = widget.get("type", "").lower()

            # Debug the widget structure
            if text_count == 0 and len(widgets) > 0:
                console.print(f"[cyan]First widget type: {widget_type}[/cyan]")
                console.print(
                    f"[cyan]First widget keys: {sorted(widget.keys())}[/cyan]"
                )

            # Check for text in htmlText field first (common in sticky notes)
            if "htmlText" in widget and widget.get("htmlText"):
                # Extract plain text from HTML
                html_text = widget.get("htmlText", "")
                if html_text and isinstance(html_text, str):
                    # Simple HTML tag stripping (a more robust solution would use BeautifulSoup)
                    cleaned_text = re.sub(r"<[^>]*>", " ", html_text)
                    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
                    if cleaned_text:
                        text = cleaned_text

            # If no text from HTML, try regular text fields
            if not text:
                # Extract based on widget type
                if widget_type == "sticky" or widget_type == "sticky note":
                    text = widget.get("text", "")
                elif widget_type == "text" or widget_type == "textbox":
                    text = widget.get("text", "")
                elif widget_type == "shape" and "text" in widget:
                    text = widget.get("text", "")
                elif widget_type == "connector" and "text" in widget:
                    text = widget.get("text", "")
                elif "text" in widget and widget.get("text"):
                    # Fallback for any widget with a text field
                    text = widget.get("text", "")
                elif "title" in widget and widget.get("title"):
                    # Some widgets might use title instead of text
                    text = widget.get("title", "")
                elif "content" in widget and widget.get("content"):
                    # Some widgets might use content
                    text = widget.get("content", "")

            # If we found text, add it to our list
            if text and isinstance(text, str) and text.strip():
                text_items.append(text)
                text_count += 1

        console.print(f"[cyan]Extracted text from {text_count} widgets[/cyan]")

        # If the API has a format that includes comments separately, get those too
        comments = mural_content.get("comments", [])
        for comment in comments:
            text = comment.get("text", "")
            if text and isinstance(text, str) and text.strip():
                text_items.append(text)

        # Filter out empty strings and log the result
        result = [item for item in text_items if item and item.strip()]
        console.print(f"[cyan]Total text items extracted: {len(result)}[/cyan]")

        return result


class OpenAIAPI:
    """Class to handle all interactions with the OpenAI API."""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of text items."""
        if not texts:
            return []

        response = self.client.embeddings.create(
            model="text-embedding-3-small", input=texts
        )

        return [data.embedding for data in response.data]

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_business_plan_section(
        self, section_title: str, context: str, sources: List[str] = None
    ) -> str:
        """Generate a business plan section using GPT."""
        sources_text = ""
        if sources:
            sources_text = "\n\nSOURCES USED:\n" + "\n".join(
                [f"- {src}" for src in sources]
            )

        prompt = f"""
        You are an expert business consultant and writer. Use the following context information from a Mural board 
        to draft a coherent and professional {section_title} section for a business plan.
        
        CONTEXT FROM MURAL BOARD:
        {context}
        
        Please write a comprehensive and well-structured {section_title} section that incorporates the relevant 
        information from the context. The section should be detailed, professional, and ready to include in a 
        formal business plan document.
        
        At the end of your response, please include a list of all the sources used from the Mural board, formatted as follows:
        
        ## Sources Used
        {sources_text}
        """

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",  # Using GPT-4 Turbo as a substitute for GPT-4.5
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert business consultant that creates professional business plan sections.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1500,
        )

        return response.choices[0].message.content


class SemanticSearch:
    """Class to handle semantic search functionality."""

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_vec1 = sum(a * a for a in vec1) ** 0.5
        norm_vec2 = sum(b * b for b in vec2) ** 0.5

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0

        return dot_product / (norm_vec1 * norm_vec2)

    @staticmethod
    def search(
        query_embedding: List[float],
        item_embeddings: List[List[float]],
        items: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float, int]]:
        """Search for the most similar items to the query."""
        similarities = [
            (
                item,
                SemanticSearch.cosine_similarity(query_embedding, item_embedding),
                idx,
            )
            for idx, (item, item_embedding) in enumerate(zip(items, item_embeddings))
        ]

        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k results
        return similarities[:top_k]


class BusinessPlanDrafter:
    """Main class for the Business Plan Drafter CLI tool."""

    def __init__(self):
        """Initialize the Business Plan Drafter."""
        self.console = Console()
        self.check_api_keys()
        self.mural_oauth = MuralOAuth(MURAL_CLIENT_ID, MURAL_CLIENT_SECRET)
        self.mural_api = MuralAPI(self.mural_oauth)
        self.openai_api = OpenAIAPI(OPENAI_API_KEY)
        self.selected_project = None
        self.mural_texts = []
        self.mural_embeddings = []

        # Load any saved configuration
        self.load_config()

    def check_api_keys(self):
        """Check if API keys are available."""
        if not OPENAI_API_KEY:
            console.print(
                "[bold red]Error:[/bold red] OpenAI API key not found. "
                "Please add it to your .env file as OPENAI_API_KEY."
            )
            sys.exit(1)

        if not MURAL_CLIENT_ID or not MURAL_CLIENT_SECRET:
            console.print(
                "[bold red]Error:[/bold red] Mural OAuth credentials not found. "
                "Please add them to your .env file as MURAL_CLIENT_ID and MURAL_CLIENT_SECRET."
            )
            sys.exit(1)

    def load_config(self):
        """Load configuration from the config file."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)
                    self.selected_project = config.get("selected_project")
            except json.JSONDecodeError:
                self.selected_project = None

    def save_config(self):
        """Save configuration to the config file."""
        config = {"selected_project": self.selected_project}

        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f)

    def display_projects(self) -> Dict[str, Any]:
        """Display available Mural projects and let user select one."""
        all_murals = []

        # First phase: Fetch all projects with progress spinner
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("[cyan]Fetching Mural projects...", total=None)

            try:
                # Get workspaces
                workspaces = self.mural_api.get_workspaces()

                if not workspaces:
                    self.console.print(
                        "[yellow]No workspaces found. Please check your Mural account.[/yellow]"
                    )
                    sys.exit(1)

                # Debug workspace information
                self.console.print(f"[cyan]Found {len(workspaces)} workspaces[/cyan]")

                # Get murals from each workspace
                for workspace in workspaces:
                    # Add safe access to workspace properties with defaults
                    workspace_id = workspace.get("id", "unknown")
                    workspace_name = workspace.get("name", f"Workspace {workspace_id}")

                    self.console.print(
                        f"[cyan]Fetching murals from workspace: {workspace_name} (ID: {workspace_id})[/cyan]"
                    )

                    try:
                        murals = self.mural_api.get_murals(workspace_id)

                        for mural in murals:
                            # Add safe access to mural properties with defaults
                            # Use 'title' field for mural name (instead of 'name' which doesn't exist)
                            mural_id = mural.get("id", "unknown")
                            mural_title = mural.get("title", f"Mural {mural_id}")

                            all_murals.append(
                                {
                                    "id": mural_id,
                                    "name": mural_title,  # Store title as 'name' for simplicity in later code
                                    "workspace_id": workspace_id,
                                    "workspace_name": workspace_name,
                                }
                            )

                    except Exception as e:
                        self.console.print(
                            f"[yellow]Warning: Could not fetch murals from workspace {workspace_name}: {str(e)}[/yellow]"
                        )

                progress.update(task, completed=100)

                if not all_murals:
                    self.console.print(
                        "[yellow]No murals found. Please check your Mural account.[/yellow]"
                    )
                    sys.exit(1)

            except Exception as e:
                self.console.print(
                    f"[bold red]Error fetching projects: {str(e)}[/bold red]"
                )
                self.console.print(
                    "[yellow]For detailed debugging, run 'python mural_api_test.py'[/yellow]"
                )
                sys.exit(1)

        # Second phase: Display projects and handle user selection (outside of progress spinner)
        # Display available murals
        self.console.print("\n[bold cyan]Available Projects:[/bold cyan]")

        # Prepare table data
        table_data = []
        for i, mural in enumerate(all_murals, 1):
            table_data.append([i, mural["name"], mural["workspace_name"]])

        # Display table
        headers = ["#", "Mural Name", "Workspace"]
        table = tabulate(table_data, headers=headers, tablefmt="simple")
        self.console.print(table)

        # Let user select a mural
        while True:
            try:
                selection = int(input(f"\nSelect a project (1-{len(all_murals)}): "))
                if 1 <= selection <= len(all_murals):
                    selected_mural = all_murals[selection - 1]
                    self.console.print(
                        f"\n[green]Project \"{selected_mural['name']}\" selected![/green]"
                    )
                    return selected_mural
                else:
                    self.console.print(
                        f"[red]Please enter a number between 1 and {len(all_murals)}.[/red]"
                    )
            except ValueError:
                self.console.print("[red]Please enter a valid number.[/red]")

    def fetch_mural_content(self, mural_id: str, workspace_id: str):
        """Fetch content from the selected Mural board."""
        success = False
        error_message = None
        mural_content = {}

        # First phase: Fetch content with progress spinner
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("[cyan]Fetching Mural content...", total=None)

            try:
                # Get mural content
                mural_content = self.mural_api.get_mural_content(mural_id)

                # Extract text from mural content
                self.mural_texts = self.mural_api.extract_mural_text(mural_content)

                if self.mural_texts:
                    success = True

                progress.update(task, completed=100)

            except Exception as e:
                error_message = str(e)
                progress.update(task, completed=100)

        # Second phase: Handle results or errors (outside of progress spinner)
        if success:
            # Get embeddings for the mural text content
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("[cyan]Generating embeddings...", total=None)
                self.mural_embeddings = self.openai_api.get_embeddings(self.mural_texts)
                progress.update(task, completed=100)

            return

        # Handle empty content or errors
        if error_message:
            self.console.print(
                f"[bold red]Error fetching Mural content: {error_message}[/bold red]"
            )
        else:
            self.console.print(
                "[yellow]No text content found in the selected Mural board.[/yellow]"
            )

        self.console.print("[yellow]Options:[/yellow]")
        self.console.print(
            "[yellow]1. Continue anyway (the AI will generate content without Mural context)[/yellow]"
        )
        self.console.print("[yellow]2. Go back and select a different mural[/yellow]")
        self.console.print("[yellow]3. Exit and try again later[/yellow]")

        choice = input("\nEnter your choice (1-3): ")

        if choice == "1":
            self.console.print("[cyan]Continuing without Mural content...[/cyan]")
            # Create a single dummy text item so embeddings will work
            self.mural_texts = [
                "No Mural content available. Generating content based on section description only."
            ]

            # Generate embeddings for the dummy text
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("[cyan]Generating embeddings...", total=None)
                self.mural_embeddings = self.openai_api.get_embeddings(self.mural_texts)
                progress.update(task, completed=100)

        elif choice == "2":
            self.selected_project = self.display_projects()
            self.save_config()
            self.fetch_mural_content(
                self.selected_project["id"], self.selected_project["workspace_id"]
            )
            return
        else:
            self.console.print("[cyan]Exiting. Please try again later.[/cyan]")
            sys.exit(0)

    def generate_business_plan_section(self, section_title: str) -> str:
        """Generate a business plan section based on the provided title and Mural content."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                # Get embedding for the section title
                task = progress.add_task(
                    "[cyan]Searching Mural board for relevant content...", total=None
                )
                section_embedding = self.openai_api.get_embeddings([section_title])[0]

                # Search for relevant content
                search_results = SemanticSearch.search(
                    section_embedding, self.mural_embeddings, self.mural_texts, top_k=10
                )

                # Extract the relevant text and source information
                relevant_text = "\n\n".join([result[0] for result in search_results])

                # Prepare source information - include original text snippet and similarity score
                sources = [
                    f"Source {idx+1} (Similarity: {result[1]:.4f}): {result[0][:100]}..."
                    for idx, result in enumerate(search_results)
                ]

                progress.update(task, completed=100)

                # Generate the business plan section
                task = progress.add_task(
                    f"[cyan]Generating {section_title}...", total=None
                )
                generated_section = self.openai_api.generate_business_plan_section(
                    section_title, relevant_text, sources
                )

                progress.update(task, completed=100)

                return generated_section

        except Exception as e:
            self.console.print(
                f"[bold red]Error generating business plan section: {str(e)}[/bold red]"
            )
            return ""

    def export_to_file(self, section_title: str, content: str, format: str = "md"):
        """Export the generated content to a file."""
        try:
            # Normalize the section title for the filename
            filename = section_title.lower().replace(" ", "_")
            file_path = Path(f"{filename}.{format}")

            with open(file_path, "w") as f:
                f.write(content)

            self.console.print(f"\n[green]Section exported to {file_path}[/green]")

        except Exception as e:
            self.console.print(
                f"[bold red]Error exporting to file: {str(e)}[/bold red]"
            )

    def run(self):
        """Main execution flow of the Business Plan Drafter."""
        self.console.print(
            Panel.fit(
                "[bold cyan]✨ Business Plan Drafter ✨[/bold cyan]\n\n"
                "Generate business plan drafts using Mural board content and AI.",
                title="Welcome",
                border_style="cyan",
            )
        )

        # Select or use previously selected project
        if not self.selected_project:
            self.selected_project = self.display_projects()
            self.save_config()
        else:
            # Check if user wants to use the previously selected project
            self.console.print(
                f"\n[cyan]Previously selected project: {self.selected_project['name']}[/cyan]"
            )
            use_previous = input("Use this project? (y/n): ").lower()

            if use_previous != "y":
                self.selected_project = self.display_projects()
                self.save_config()

        # Fetch Mural content and generate embeddings
        self.fetch_mural_content(
            self.selected_project["id"], self.selected_project["workspace_id"]
        )

        # Business plan section drafting loop
        while True:
            self.console.print(
                "\n[bold cyan]Business Plan Section Generator[/bold cyan]"
            )
            section_title = input(
                "\nEnter a business plan section title (or 'q' to quit): "
            )

            if section_title.lower() == "q":
                break

            # Generate the business plan section
            generated_section = self.generate_business_plan_section(section_title)

            if generated_section:
                # Display the generated section
                self.console.print("\n" + "-" * 40)
                self.console.print(generated_section)
                self.console.print("-" * 40)

                # Ask if user wants to export to file
                export = input("\nExport this section to a file? (y/n): ").lower()
                if export == "y":
                    format_choice = input(
                        "Export format (md/txt) [default: md]: "
                    ).lower()
                    format = format_choice if format_choice in ["md", "txt"] else "md"
                    self.export_to_file(section_title, generated_section, format)

        self.console.print(
            "\n[bold green]Thank you for using Business Plan Drafter![/bold green]"
        )


if __name__ == "__main__":
    try:
        drafter = BusinessPlanDrafter()
        drafter.run()
    except KeyboardInterrupt:
        console.print("\n\n[bold yellow]Program interrupted. Exiting...[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred: {str(e)}[/bold red]")
        sys.exit(1)
