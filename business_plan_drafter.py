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
from urllib.parse import urlparse, parse_qs

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

            # Initialize a result object to store all widgets
            all_widgets = {"value": []}

            # Now get the widgets for this mural, using the ORIGINAL ID that worked for accessing the mural
            base_widgets_url = f"{self.base_url}/murals/{mural_id}/widgets"
            next_url = base_widgets_url
            has_more_pages = True
            page_count = 0
            next_token = None

            while has_more_pages:
                page_count += 1

                # Construct the URL with the next token as a query parameter if it exists
                current_url = next_url
                if next_token:
                    current_url = f"{base_widgets_url}?next={next_token}"

                console.print(
                    f"[cyan]Fetching widgets page {page_count} from: {current_url}[/cyan]"
                )

                widgets_response = requests.get(
                    current_url, headers=self._get_headers()
                )

                if widgets_response.status_code != 200:
                    console.print(
                        f"[yellow]Failed to get widgets. Status code: {widgets_response.status_code}[/yellow]"
                    )

                    # Try one more approach if this is the first page - use the returned ID if available
                    if page_count == 1 and "id" in mural_response.json():
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
                                next_url = alt_widgets_url
                            else:
                                # If all attempts failed, raise the exception
                                widgets_response.raise_for_status()
                        else:
                            # If all attempts failed, raise the exception
                            widgets_response.raise_for_status()
                    else:
                        # If not the first page or no ID available, raise the exception
                        widgets_response.raise_for_status()

                response_data = widgets_response.json()

                # Extract widgets from this page
                if "value" in response_data and isinstance(
                    response_data["value"], list
                ):
                    all_widgets["value"].extend(response_data["value"])
                    console.print(
                        f"[green]Added {len(response_data['value'])} widgets from page {page_count}[/green]"
                    )
                elif "widgets" in response_data and isinstance(
                    response_data["widgets"], list
                ):
                    all_widgets["value"].extend(response_data["widgets"])
                    console.print(
                        f"[green]Added {len(response_data['widgets'])} widgets from page {page_count}[/green]"
                    )
                elif isinstance(response_data, list):
                    all_widgets["value"].extend(response_data)
                    console.print(
                        f"[green]Added {len(response_data)} widgets from page {page_count}[/green]"
                    )
                elif "data" in response_data and isinstance(
                    response_data["data"], list
                ):
                    all_widgets["value"].extend(response_data["data"])
                    console.print(
                        f"[green]Added {len(response_data['data'])} widgets from page {page_count}[/green]"
                    )
                else:
                    console.print(
                        f"[yellow]No widgets found in page {page_count}[/yellow]"
                    )

                # Check if there are more pages
                if "next" in response_data and response_data["next"]:
                    next_token = response_data["next"]
                    # If the next token is a full URL, extract just the token part
                    if next_token.startswith("http"):
                        # Try to extract just the token from the URL
                        url_parts = urlparse(next_token)
                        query_params = parse_qs(url_parts.query)
                        if "next" in query_params:
                            next_token = query_params["next"][0]
                else:
                    has_more_pages = False

            console.print(
                f"[green]Successfully fetched all mural widgets across {page_count} pages![/green]"
            )
            console.print(
                f"[cyan]Total widgets collected: {len(all_widgets['value'])}[/cyan]"
            )

            return all_widgets

        except requests.exceptions.RequestException as e:
            console.print(
                f"[bold red]Error fetching mural content: {str(e)}[/bold red]"
            )
            console.print(
                "[yellow]Failed to access mural content. Continuing with empty content...[/yellow]"
            )
            return {}

    def extract_mural_text(self, mural_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract text and image content from mural widgets."""
        content_items = []

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
        # This check is now redundant since we fetch all pages in get_mural_content
        # but we'll keep it for backward compatibility
        if "next" in mural_content and mural_content["next"]:
            console.print(
                f"[yellow]Warning: Found a 'next' pagination link that wasn't processed. This should not happen as we now fetch all pages.[/yellow]"
            )

        # Track how many widgets had text or image content
        text_count = 0
        image_count = 0

        # Debug the first widget structure to see available fields
        if widgets and len(widgets) > 0:
            console.print("[cyan]First widget structure sample:[/cyan]")
            # Print some fields that are likely to exist in widget data
            sample_widget = widgets[0]
            position_fields = {
                key: sample_widget.get(key)
                for key in [
                    "x",
                    "y",
                    "width",
                    "height",
                    "position",
                    "left",
                    "top",
                    "scale",
                    "rotation",
                    "transform",
                    "bounds",
                ]
                if key in sample_widget
            }
            console.print(f"[cyan]Position-related fields: {position_fields}[/cyan]")
            console.print(
                f"[cyan]All available keys: {sorted(sample_widget.keys())}[/cyan]"
            )

        # Process widgets (sticky notes, text boxes, images, etc.)
        for idx, widget in enumerate(widgets):
            content = None
            widget_type = widget.get("type", "").lower()
            content_type = "text"  # Default content type
            widget_id = widget.get("id", f"widget-{idx}")

            # Extract position information
            position_data = {}
            for pos_field in [
                "x",
                "y",
                "width",
                "height",
                "left",
                "top",
                "scale",
                "rotation",
            ]:
                if pos_field in widget:
                    position_data[pos_field] = widget.get(pos_field)

            # Check for nested position data structures
            if "position" in widget and isinstance(widget.get("position"), dict):
                position_data.update(widget.get("position"))
            if "bounds" in widget and isinstance(widget.get("bounds"), dict):
                position_data.update(widget.get("bounds"))
            if "transform" in widget and isinstance(widget.get("transform"), dict):
                position_data.update(widget.get("transform"))

            # Debug the widget structure of first items
            if (text_count == 0 and image_count == 0) and len(widgets) > 0:
                console.print(f"[cyan]First widget type: {widget_type}[/cyan]")
                console.print(
                    f"[cyan]First widget keys: {sorted(widget.keys())}[/cyan]"
                )

            # Check for image widgets
            if widget_type == "image" or "imageUrl" in widget or "image" in widget:
                image_url = None
                # Try different possible image URL fields
                if "imageUrl" in widget and widget.get("imageUrl"):
                    image_url = widget.get("imageUrl")
                elif "src" in widget and widget.get("src"):
                    image_url = widget.get("src")
                elif "url" in widget and widget.get("url"):
                    image_url = widget.get("url")
                elif (
                    "image" in widget
                    and isinstance(widget.get("image"), dict)
                    and "url" in widget.get("image")
                ):
                    image_url = widget.get("image").get("url")

                if image_url:
                    content = image_url
                    content_type = "image"
                    image_count += 1

            # If not an image or no image URL found, check for text
            if not content:
                # Check for text in htmlText field first (common in sticky notes)
                if "htmlText" in widget and widget.get("htmlText"):
                    # Extract plain text from HTML
                    html_text = widget.get("htmlText", "")
                    if html_text and isinstance(html_text, str):
                        # Simple HTML tag stripping (a more robust solution would use BeautifulSoup)
                        cleaned_text = re.sub(r"<[^>]*>", " ", html_text)
                        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
                        if cleaned_text:
                            content = cleaned_text
                            content_type = "text"

                # If no text from HTML, try regular text fields
                if not content:
                    # Extract based on widget type
                    if widget_type == "sticky" or widget_type == "sticky note":
                        content = widget.get("text", "")
                    elif widget_type == "text" or widget_type == "textbox":
                        content = widget.get("text", "")
                    elif widget_type == "shape" and "text" in widget:
                        content = widget.get("text", "")
                    elif widget_type == "connector" and "text" in widget:
                        content = widget.get("text", "")
                    elif "text" in widget and widget.get("text"):
                        # Fallback for any widget with a text field
                        content = widget.get("text", "")
                    elif "title" in widget and widget.get("title"):
                        # Some widgets might use title instead of text
                        content = widget.get("title", "")
                    elif "content" in widget and widget.get("content"):
                        # Some widgets might use content
                        content = widget.get("content", "")

                    if content and isinstance(content, str) and content.strip():
                        text_count += 1

            # If we found content, add it to our list with type and id information
            if content:
                content_items.append(
                    {
                        "content": content,
                        "type": content_type,
                        "id": widget_id,
                        "widget_type": widget_type,
                        "position": position_data,  # Include position data
                    }
                )

        console.print(
            f"[cyan]Extracted text from {text_count} widgets and found {image_count} images[/cyan]"
        )

        # If the API has a format that includes comments separately, get those too
        comments = mural_content.get("comments", [])
        for idx, comment in enumerate(comments):
            text = comment.get("text", "")
            if text and isinstance(text, str) and text.strip():
                content_items.append(
                    {
                        "content": text,
                        "type": "text",
                        "id": comment.get("id", f"comment-{idx}"),
                        "widget_type": "comment",
                        "position": {},  # Comments might not have position data
                    }
                )

        # Filter out empty content
        result = [item for item in content_items if item["content"]]
        console.print(f"[cyan]Total content items extracted: {len(result)}[/cyan]")

        return result

    def cluster_widgets_by_proximity(
        self, content_items: List[Dict[str, Any]], distance_threshold: float = 300
    ) -> List[Dict[str, Any]]:
        """
        Cluster widgets based on their spatial proximity on the mural board.

        Args:
            content_items: List of content items with position data
            distance_threshold: Maximum distance between widgets to be considered in the same cluster

        Returns:
            List of clustered content items
        """
        console.print("[cyan]Clustering widgets by proximity...[/cyan]")

        # Filter content items to only include those with valid position data
        valid_items = []
        for item in content_items:
            position = item.get("position", {})
            # Check if we have at least x and y coordinates
            if (
                position
                and ("x" in position or "left" in position)
                and ("y" in position or "top" in position)
            ):
                # Normalize coordinate names (prefer x,y but use left,top if needed)
                if "x" not in position and "left" in position:
                    position["x"] = position["left"]
                if "y" not in position and "top" in position:
                    position["y"] = position["top"]
                valid_items.append(item)

        if not valid_items:
            console.print(
                "[yellow]No items with valid position data found for clustering[/yellow]"
            )
            return content_items

        console.print(
            f"[cyan]Found {len(valid_items)} items with valid position data[/cyan]"
        )

        # Initialize clusters
        clusters = []
        assigned = set()

        # Function to calculate Euclidean distance between two widgets
        def distance(item1, item2):
            x1 = item1["position"]["x"]
            y1 = item1["position"]["y"]
            x2 = item2["position"]["x"]
            y2 = item2["position"]["y"]
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

        # Cluster widgets
        for i, item in enumerate(valid_items):
            if i in assigned:
                continue

            # Start a new cluster
            cluster = [item]
            assigned.add(i)

            # Find all items close to this cluster
            j = 0
            while j < len(cluster):
                for k, candidate in enumerate(valid_items):
                    if (
                        k not in assigned
                        and distance(cluster[j], candidate) <= distance_threshold
                    ):
                        cluster.append(candidate)
                        assigned.add(k)
                j += 1

            clusters.append(cluster)

        console.print(
            f"[green]Created {len(clusters)} proximity-based clusters[/green]"
        )

        # Create new content items from clusters
        clustered_items = []

        for i, cluster in enumerate(clusters):
            # Sort cluster items by position (top to bottom, left to right)
            cluster.sort(
                key=lambda item: (item["position"]["y"], item["position"]["x"])
            )

            # Combine text from all items in the cluster
            combined_content = []
            widget_ids = []
            widget_types = set()

            for item in cluster:
                content = item["content"]
                if content:
                    combined_content.append(content)
                widget_ids.append(item["id"])
                widget_types.add(item["widget_type"])

            # Only create a cluster item if it has content
            if combined_content:
                # Calculate the centroid of the cluster
                centroid_x = sum(item["position"]["x"] for item in cluster) / len(
                    cluster
                )
                centroid_y = sum(item["position"]["y"] for item in cluster) / len(
                    cluster
                )

                cluster_item = {
                    "content": "\n\n".join(combined_content),
                    "type": "text",
                    "id": f"cluster-{i}",
                    "widget_type": "proximity_cluster",
                    "widget_ids": widget_ids,
                    "widget_types": list(widget_types),
                    "cluster_size": len(cluster),
                    "position": {"x": centroid_x, "y": centroid_y},
                }
                clustered_items.append(cluster_item)

        console.print(
            f"[green]Generated {len(clustered_items)} clustered content items[/green]"
        )

        return clustered_items


class OpenAIAPI:
    """Class to handle all interactions with the OpenAI API."""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.image_cache_file = Path.home() / ".mural_api" / "image_analysis_cache.json"
        self.image_cache = self._load_image_cache()
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "saved_time_seconds": 0,  # Estimate 3 seconds per API call
            "saved_cost": 0.0,  # Estimate $0.01 per image analysis
        }

    def _load_image_cache(self) -> Dict[str, str]:
        """Load image analysis cache from disk."""
        try:
            cache_dir = self.image_cache_file.parent
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True, exist_ok=True)

            if self.image_cache_file.exists():
                with open(self.image_cache_file, "r") as f:
                    cache_data = json.load(f)

                    # Handle the case where the cache is in the old format (just a dict of URLs to analyses)
                    if cache_data and isinstance(
                        next(iter(cache_data.values()), {}), str
                    ):
                        # Convert old format to new format
                        console.print(
                            "[yellow]Converting image cache to new format...[/yellow]"
                        )
                        new_cache = {}
                        for url, analysis in cache_data.items():
                            new_cache[url] = {
                                "analysis": analysis,
                                "timestamp": time.time(),
                                "model": "gpt-4o",
                            }
                        return new_cache
                    return cache_data
            else:
                return {}
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not load image cache: {str(e)}[/yellow]"
            )
            return {}

    def _save_image_cache(self):
        """Save image analysis cache to disk."""
        try:
            cache_dir = self.image_cache_file.parent
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True, exist_ok=True)

            with open(self.image_cache_file, "w") as f:
                json.dump(self.image_cache, f, indent=2)
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not save image cache: {str(e)}[/yellow]"
            )

    def clear_image_cache(self):
        """Clear the image analysis cache."""
        try:
            self.image_cache = {}
            self._save_image_cache()
            console.print("[green]Image analysis cache cleared successfully[/green]")
            return True
        except Exception as e:
            console.print(f"[bold red]Error clearing image cache: {str(e)}[/bold red]")
            return False

    def get_cache_stats(self):
        """Get statistics about the image cache."""
        cached_images = len(self.image_cache)
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (
            self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        )

        stats = {
            "cached_images": cached_images,
            "cache_hits": self.cache_stats["hits"],
            "cache_misses": self.cache_stats["misses"],
            "hit_rate": f"{hit_rate:.2%}",
            "estimated_time_saved": f"{self.cache_stats['saved_time_seconds']:.1f} seconds",
            "estimated_cost_saved": f"${self.cache_stats['saved_cost']:.2f}",
        }

        return stats

    def print_cache_stats(self):
        """Print statistics about the image cache."""
        stats = self.get_cache_stats()

        console.print(
            Panel(
                "\n".join(
                    [
                        f"[cyan]Cached Images:[/cyan] {stats['cached_images']}",
                        f"[cyan]Cache Hits:[/cyan] {stats['cache_hits']}",
                        f"[cyan]Cache Misses:[/cyan] {stats['cache_misses']}",
                        f"[cyan]Hit Rate:[/cyan] {stats['hit_rate']}",
                        f"[cyan]Estimated Time Saved:[/cyan] {stats['estimated_time_saved']}",
                        f"[cyan]Estimated Cost Saved:[/cyan] {stats['estimated_cost_saved']}",
                    ]
                ),
                title="[bold green]Image Cache Statistics[/bold green]",
            )
        )

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
    def analyze_image(self, image_url: str, force_reanalysis: bool = False) -> str:
        """Analyze an image using OpenAI's vision capabilities to extract text and context.

        Args:
            image_url: URL of the image to analyze
            force_reanalysis: If True, reanalyze the image even if it's in the cache
        """
        # Check if this image URL is in the cache
        if not force_reanalysis and image_url in self.image_cache:
            console.print(
                f"[green]Using cached analysis for image (cached on {time.strftime('%Y-%m-%d', time.localtime(self.image_cache[image_url]['timestamp']))})[/green]"
            )
            self.cache_stats["hits"] += 1
            self.cache_stats[
                "saved_time_seconds"
            ] += 3  # Estimate 3 seconds per API call
            self.cache_stats["saved_cost"] += 0.01  # Estimate $0.01 per image analysis
            return self.image_cache[image_url]["analysis"]

        self.cache_stats["misses"] += 1
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing business-related images. Extract all visible text and describe the key elements and concepts shown in the image in detail.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image from a business mural board. Extract all text content and describe what you see.",
                            },
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                max_tokens=500,
            )
            analysis_time = time.time() - start_time
            analysis = response.choices[0].message.content

            # Cache the result with metadata
            self.image_cache[image_url] = {
                "analysis": analysis,
                "timestamp": time.time(),
                "model": "gpt-4o",
                "processing_time": analysis_time,
            }
            self._save_image_cache()

            return analysis
        except Exception as e:
            console.print(f"[bold red]Error analyzing image: {str(e)}[/bold red]")
            return f"[Failed to analyze image: {str(e)}]"

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_business_plan_section(
        self, section_title: str, context: str, sources: List[str] = None
    ) -> str:
        """Generate a business plan section using GPT."""
        sources_text = ""
        if sources:
            sources_text = "\n".join([f"- {src}" for src in sources])

        prompt = f"""
        You are an expert business consultant and writer. Use the following content extracted from a Mural board 
        to draft a coherent and professional {section_title} section for a business plan.
        
        CONTENT FROM MURAL BOARD:
        {context}
        
        SOURCES (Reference these by their number in your response):
        {sources_text}
        
        Please write a comprehensive and well-structured {section_title} section that incorporates ONLY the 
        information from the provided content. The section should be detailed, professional, and ready to include in a 
        formal business plan document.
        
        IMPORTANT: DO NOT make up any information. Only use what is provided in the content above.
        Use DIRECT QUOTES and SPECIFIC INFORMATION from the sources whenever possible.
        
        At the end of your response, include a list of all the sources used from the Mural board, formatted as follows:
        
        ## Sources Used
        {sources_text}
        """

        response = self.client.chat.completions.create(
            model="gpt-4.5-preview",  # change to 4.5 later
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert business consultant that creates professional business plan sections. You MUST use ONLY the information from the provided context and sources - do not invent or make up ANY information. Reference source numbers explicitly throughout your response.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,  # Lower temperature for more deterministic outputs
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
        # Initialize the config with default values
        self.config = {"last_mural": None}

        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    loaded_config = json.load(f)
                    # Copy the loaded config to the instance config
                    self.config = loaded_config
                    # For backward compatibility
                    self.selected_project = loaded_config.get("selected_project")
            except json.JSONDecodeError:
                self.selected_project = None
                self.console.print(
                    "[yellow]Warning: Could not parse config file. Using defaults.[/yellow]"
                )
        else:
            self.selected_project = None
            self.console.print(
                "[yellow]No configuration file found. Starting with defaults.[/yellow]"
            )

    def save_config(self):
        """Save configuration to the config file."""
        # Make sure selected_project is saved in the config
        self.config["selected_project"] = self.selected_project

        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=2)

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
        """Fetch content from a mural and prepare for analysis."""
        try:
            self.console.print(
                Panel(
                    f"Fetching content from Mural: {mural_id}",
                    title="[bold cyan]Fetching Mural Content[/bold cyan]",
                )
            )

            # Get the mural content
            mural_content = self.mural_api.get_mural_content(mural_id)

            if not mural_content:
                self.console.print(
                    "[bold red]Failed to retrieve mural content. Please check the mural ID and try again.[/bold red]"
                )
                return

            self.console.print("[green]Successfully retrieved mural content![/green]")

            # Extract text and image content
            content_items = self.mural_api.extract_mural_text(mural_content)

            if not content_items:
                self.console.print(
                    "[bold yellow]No text or image content found in the mural.[/bold yellow]"
                )
                return

            # Ask if the user wants to use proximity-based clustering
            use_clustering = False
            clustering_threshold = 500  # Default distance threshold

            clustering_choice = (
                self.console.input(
                    "[yellow]Do you want to cluster widgets based on proximity? (y/N): [/yellow]"
                )
                .strip()
                .lower()
            )
            use_clustering = clustering_choice in ("y", "yes")

            if use_clustering:
                self.console.print(
                    "[cyan]Will cluster widgets based on proximity.[/cyan]"
                )

                # Ask for distance threshold
                try:
                    threshold_input = self.console.input(
                        "[yellow]Enter distance threshold for clustering (default: 300): [/yellow]"
                    ).strip()

                    if threshold_input:
                        clustering_threshold = float(threshold_input)

                    self.console.print(
                        f"[cyan]Using distance threshold: {clustering_threshold}[/cyan]"
                    )

                    # Apply clustering
                    clustered_items = self.mural_api.cluster_widgets_by_proximity(
                        content_items, distance_threshold=clustering_threshold
                    )

                    if clustered_items:
                        content_items = clustered_items
                        self.console.print(
                            "[green]Successfully clustered widgets![/green]"
                        )
                    else:
                        self.console.print(
                            "[yellow]Clustering did not produce any results. Using original items.[/yellow]"
                        )

                except ValueError:
                    self.console.print(
                        "[yellow]Invalid threshold value. Using default threshold of 300.[/yellow]"
                    )
                    # Apply clustering with default threshold
                    clustered_items = self.mural_api.cluster_widgets_by_proximity(
                        content_items
                    )
                    if clustered_items:
                        content_items = clustered_items
                        self.console.print(
                            "[green]Successfully clustered widgets![/green]"
                        )
            else:
                self.console.print(
                    "[cyan]Using individual widgets without clustering.[/cyan]"
                )

            # Separate text and image items
            text_items = []
            image_items = []

            for item in content_items:
                if item["type"] == "text":
                    text_items.append(item)
                elif item["type"] == "image":
                    image_items.append(item)

            self.console.print(
                f"[cyan]Found {len(text_items)} text items and {len(image_items)} image items.[/cyan]"
            )

            # Ask if the user wants to analyze images
            analyze_images = False
            if image_items:
                analyze_choice = (
                    self.console.input(
                        "[yellow]Do you want to analyze images in this mural? (y/N): [/yellow]"
                    )
                    .strip()
                    .lower()
                )
                analyze_images = analyze_choice in ("y", "yes")

                if not analyze_images:
                    self.console.print(
                        "[cyan]Skipping image analysis as requested.[/cyan]"
                    )

                    # Add placeholders for images that are not being analyzed
                    for item in image_items:
                        text_items.append(
                            {
                                "content": "[Image not analyzed by user request]",
                                "type": "text",
                                "id": f"{item['id']}-skipped-analysis",
                                "widget_type": "image_skipped",
                                "source_image": item["id"],
                                "image_url": item.get("content", ""),
                            }
                        )
                else:
                    self.console.print("[cyan]Will analyze images as requested.[/cyan]")

            # Process images with OpenAI Vision API
            if image_items and analyze_images:
                # Ask if the user wants to reanalyze cached images
                force_reanalysis = False
                if len(self.openai_api.image_cache) > 0:
                    # Display cache statistics before asking
                    self.openai_api.print_cache_stats()

                    choice = (
                        self.console.input(
                            "[yellow]Do you want to force reanalysis of all images, even if they are in the cache? (y/N): [/yellow]"
                        )
                        .strip()
                        .lower()
                    )
                    force_reanalysis = choice in ("y", "yes")

                    if force_reanalysis:
                        self.console.print(
                            "[cyan]Will reanalyze all images, ignoring cache.[/cyan]"
                        )
                    else:
                        self.console.print(
                            "[cyan]Will use cached analysis when available.[/cyan]"
                        )

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                ) as progress:
                    image_analysis_task = progress.add_task(
                        "[cyan]Analyzing images with OpenAI...", total=len(image_items)
                    )

                    for item in image_items:
                        image_url = item["content"]
                        # Check if it's a valid URL before trying to analyze
                        if not image_url.startswith(("http://", "https://")):
                            self.console.print(
                                f"[yellow]Skipping invalid image URL: {image_url[:50]}...[/yellow]"
                            )
                            progress.update(image_analysis_task, advance=1)
                            continue

                        # Analyze the image
                        self.console.print(
                            f"[cyan]Processing image: {item['id']}[/cyan]"
                        )

                        try:
                            analysis = self.openai_api.analyze_image(
                                image_url, force_reanalysis=force_reanalysis
                            )

                            # Create a new text item with the image analysis
                            if analysis and not analysis.startswith(
                                "[Failed to analyze"
                            ):
                                text_items.append(
                                    {
                                        "content": analysis,
                                        "type": "text",
                                        "id": f"{item['id']}-analysis",
                                        "widget_type": "image_analysis",
                                        "source_image": item["id"],
                                        "image_url": image_url,
                                    }
                                )
                            else:
                                self.console.print(
                                    f"[yellow]Could not analyze image: {item['id']}[/yellow]"
                                )
                        except Exception as e:
                            self.console.print(
                                f"[yellow]Error analyzing image {item['id']}: {str(e)}[/yellow]"
                            )

                        progress.update(image_analysis_task, advance=1)

                # Display updated cache statistics after processing
                self.openai_api.print_cache_stats()

            # Extract content as simple string list for embedding
            content_for_embedding = [item["content"] for item in text_items]

            # Get embeddings for the content
            self.console.print(
                "[cyan]Generating embeddings for mural content...[/cyan]"
            )
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                ) as progress:
                    embedding_task = progress.add_task(
                        "[cyan]Generating embeddings...", total=None
                    )
                    self.mural_embeddings = self.openai_api.get_embeddings(
                        content_for_embedding
                    )
                    progress.update(embedding_task, completed=100)

                self.console.print("[green]Successfully generated embeddings![/green]")

                # Store the processed content items for reference
                self.mural_texts = content_for_embedding
                self.mural_content_items = text_items

                # Store the mural in the config
                self.config["last_mural"] = {
                    "id": mural_id,
                    "workspace_id": workspace_id,
                    "item_count": len(content_for_embedding),
                    "image_count": len(image_items),
                }
                self.save_config()

                return True

            except Exception as e:
                self.console.print(
                    f"[bold red]Error generating embeddings: {str(e)}[/bold red]"
                )
                return False

        except Exception as e:
            self.console.print(
                f"[bold red]Error fetching mural content: {str(e)}[/bold red]"
            )
            return False

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

                # Extract the relevant text
                relevant_text = "\n\n".join([result[0] for result in search_results])

                # Prepare source information with detailed context
                sources = []
                for idx, result in enumerate(search_results):
                    content_idx = result[2]  # Get the original index
                    content_item = self.mural_content_items[content_idx]

                    # Basic source info with full content
                    source_info = f"Source {idx+1} (Similarity: {result[1]:.4f}): "

                    # Add widget type (sticky note, shape, text, etc.)
                    widget_type = content_item.get("widget_type", "unknown")
                    source_info += f"{widget_type} - {result[0]}"

                    # If this is an image analysis, include the image URL
                    if content_item.get("widget_type") == "image_analysis":
                        source_info += f"\nImage URL: {content_item.get('image_url', 'Not available')}"

                    sources.append(source_info)

                progress.update(task, completed=100)

                # Generate the business plan section
                task = progress.add_task(
                    f"[cyan]Generating {section_title}...", total=None
                )

                # Debug information
                self.console.print(
                    f"\n[dim]Found {len(search_results)} relevant items in Mural board.[/dim]"
                )
                self.console.print(
                    f"[dim]Passing {len(relevant_text)} characters of content to GPT.[/dim]"
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

    def manage_image_cache(self):
        """Manage the image analysis cache."""
        while True:
            self.console.print(
                Panel.fit(
                    "[bold cyan]Image Analysis Cache Management[/bold cyan]\n\n"
                    "Choose an option:",
                    title="Cache Management",
                    border_style="cyan",
                )
            )

            # Display current cache statistics
            self.openai_api.print_cache_stats()

            # Show options
            self.console.print("\n[bold cyan]Options:[/bold cyan]")
            self.console.print("1. View cached images")
            self.console.print("2. Clear entire cache")
            self.console.print("3. Return to main menu")

            choice = input("\nEnter your choice (1-3): ").strip()

            if choice == "1":
                # View cached images
                cache_size = len(self.openai_api.image_cache)
                if cache_size == 0:
                    self.console.print("[yellow]No images in cache.[/yellow]")
                    continue

                self.console.print(f"\n[cyan]Cached Images ({cache_size}):[/cyan]")

                for i, (url, data) in enumerate(self.openai_api.image_cache.items(), 1):
                    cache_date = time.strftime(
                        "%Y-%m-%d", time.localtime(data["timestamp"])
                    )
                    model = data.get("model", "unknown")
                    url_short = url[:50] + "..." if len(url) > 50 else url
                    self.console.print(
                        f"{i}. [green]{url_short}[/green] (Cached on: {cache_date}, Model: {model})"
                    )

                input("\nPress Enter to return to menu...")

            elif choice == "2":
                # Clear cache
                confirm = input(
                    "[yellow]Are you sure you want to clear the entire image cache? (y/n): [/yellow]"
                ).lower()
                if confirm == "y":
                    self.openai_api.clear_image_cache()
                else:
                    self.console.print("[cyan]Cache clearing cancelled.[/cyan]")

            elif choice == "3":
                # Return to main menu
                break

            else:
                self.console.print(
                    "[yellow]Invalid choice. Please enter a number between 1 and 3.[/yellow]"
                )

    def debug_embeddings(self):
        """Debug function to check the state of embeddings and Mural content."""
        self.console.print(
            "\n[bold cyan]Debugging Embeddings and Mural Content[/bold cyan]"
        )

        # Check if embeddings exist
        if not hasattr(self, "mural_embeddings") or not self.mural_embeddings:
            self.console.print(
                "[bold red]No embeddings found! Have you fetched Mural content?[/bold red]"
            )
            return False

        # Display embedding statistics
        self.console.print(
            f"[green]Found {len(self.mural_embeddings)} embeddings[/green]"
        )
        self.console.print(f"[green]Found {len(self.mural_texts)} text items[/green]")

        # Display sample content
        if self.mural_texts:
            self.console.print(
                "\n[cyan]Sample content from Mural (first 5 items):[/cyan]"
            )
            for i, text in enumerate(self.mural_texts[:5]):
                self.console.print(f"[dim]Item {i+1}:[/dim] {text[:200]}...")

        # Check if content items exist
        if hasattr(self, "mural_content_items") and self.mural_content_items:
            self.console.print(
                f"\n[green]Found {len(self.mural_content_items)} content items[/green]"
            )

            # Display widget types
            widget_types = {}
            for item in self.mural_content_items:
                widget_type = item.get("widget_type", "unknown")
                widget_types[widget_type] = widget_types.get(widget_type, 0) + 1

            self.console.print("\n[cyan]Widget types in Mural:[/cyan]")
            for widget_type, count in widget_types.items():
                self.console.print(f"- {widget_type}: {count}")

        return True

    def debug_clustering(self):
        """Debug function to test and visualize proximity-based clustering."""
        self.console.print(
            "\n[bold cyan]Debugging Proximity-Based Clustering[/bold cyan]"
        )

        # Check if mural content exists
        if not hasattr(self, "mural_content_items") or not self.mural_content_items:
            self.console.print(
                "[bold red]No mural content found! Have you fetched Mural content?[/bold red]"
            )
            return False

        # Count items with position data
        items_with_position = 0
        for item in self.mural_content_items:
            if "position" in item and item["position"]:
                items_with_position += 1

        self.console.print(
            f"[cyan]Found {items_with_position} out of {len(self.mural_content_items)} items with position data[/cyan]"
        )

        if items_with_position == 0:
            self.console.print(
                "[yellow]No items with position data found. Cannot perform clustering.[/yellow]"
            )
            return False

        # Ask for distance threshold
        try:
            threshold_input = self.console.input(
                "[yellow]Enter distance threshold for clustering (default: 300): [/yellow]"
            ).strip()

            clustering_threshold = 300  # Default
            if threshold_input:
                clustering_threshold = float(threshold_input)

            self.console.print(
                f"[cyan]Using distance threshold: {clustering_threshold}[/cyan]"
            )

            # Apply clustering
            clustered_items = self.mural_api.cluster_widgets_by_proximity(
                self.mural_content_items, distance_threshold=clustering_threshold
            )

            if not clustered_items:
                self.console.print(
                    "[yellow]Clustering did not produce any results.[/yellow]"
                )
                return False

            # Display cluster statistics
            self.console.print(
                f"[green]Created {len(clustered_items)} clusters[/green]"
            )

            # Calculate average cluster size
            avg_size = (
                sum(item["cluster_size"] for item in clustered_items)
                / len(clustered_items)
                if clustered_items
                else 0
            )
            self.console.print(
                f"[cyan]Average cluster size: {avg_size:.2f} widgets[/cyan]"
            )

            # Display cluster sizes distribution
            sizes = [item["cluster_size"] for item in clustered_items]
            size_counts = {}
            for size in sizes:
                size_counts[size] = size_counts.get(size, 0) + 1

            self.console.print("[cyan]Cluster size distribution:[/cyan]")
            for size, count in sorted(size_counts.items()):
                self.console.print(f"  - {size} widgets: {count} clusters")

            # Show sample clusters
            self.console.print("\n[cyan]Sample clusters (first 3):[/cyan]")
            for i, cluster in enumerate(clustered_items[:3]):
                self.console.print(
                    f"\n[bold]Cluster {i+1} (size: {cluster['cluster_size']}):[/bold]"
                )
                self.console.print(
                    f"Widget types: {', '.join(cluster['widget_types'])}"
                )
                self.console.print(
                    f"Content (first 200 chars): {cluster['content'][:200]}..."
                )

            # Ask if user wants to use these clusters for embeddings
            use_clusters = self.console.input(
                "[yellow]Do you want to use these clusters for embeddings? (y/N): [/yellow]"
            ).strip().lower() in ("y", "yes")

            if use_clusters:
                # Extract content as simple string list for embedding
                content_for_embedding = [item["content"] for item in clustered_items]

                # Get embeddings for the content
                self.console.print(
                    "[cyan]Generating embeddings for clustered content...[/cyan]"
                )

                try:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=self.console,
                    ) as progress:
                        embedding_task = progress.add_task(
                            "[cyan]Generating embeddings...", total=None
                        )
                        self.mural_embeddings = self.openai_api.get_embeddings(
                            content_for_embedding
                        )
                        progress.update(embedding_task, completed=100)

                    self.console.print(
                        "[green]Successfully generated cluster embeddings![/green]"
                    )

                    # Store the processed content items for reference
                    self.mural_texts = content_for_embedding
                    self.mural_content_items = clustered_items

                    # Update the config
                    if "last_mural" in self.config:
                        self.config["last_mural"]["clustered"] = True
                        self.config["last_mural"]["cluster_count"] = len(
                            clustered_items
                        )
                        self.config["last_mural"][
                            "clustering_threshold"
                        ] = clustering_threshold
                        self.save_config()

                    return True

                except Exception as e:
                    self.console.print(
                        f"[bold red]Error generating embeddings for clusters: {str(e)}[/bold red]"
                    )
                    return False
            else:
                self.console.print("[cyan]Keeping original embeddings.[/cyan]")
                return True

        except ValueError:
            self.console.print("[bold red]Invalid threshold value.[/bold red]")
            return False

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

        # Show image cache management options
        show_cache_menu = (
            input("Do you want to manage image analysis cache? (y/n): ").lower() == "y"
        )
        if show_cache_menu:
            self.manage_image_cache()

        # Show clustering options
        show_clustering_menu = (
            input("Do you want to test proximity-based clustering? (y/n): ").lower()
            == "y"
        )
        if show_clustering_menu:
            self.debug_clustering()

        # Fetch Mural content and generate embeddings
        fetch_result = self.fetch_mural_content(
            self.selected_project["id"], self.selected_project["workspace_id"]
        )

        # Debug embeddings if there's an issue
        if (
            not fetch_result
            or input("\nDebug embeddings and Mural content? (y/n): ").lower() == "y"
        ):
            self.debug_embeddings()

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
