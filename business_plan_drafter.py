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
import networkx as nx
from collections import defaultdict
import colorsys

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

    def cluster_widgets_by_graph(
        self,
        content_items: List[Dict[str, Any]],
        distance_factor: float = 0.6,
        size_factor: float = 0.0,  # Reduced size factor
        color_factor: float = 0.3,  # Increased color factor
        type_factor: float = 0.0,  # Type doesn't matter as per user request
        horizontal_threshold_factor: float = 0.2,  # Horizontal distance threshold as 0.2x width
        vertical_threshold_factor: float = 0.3,  # Vertical distance threshold as 0.3x height
        edge_threshold: float = 0.5,  # Slightly reduced for better connectivity
        community_detection: str = "louvain",
        max_connections_per_node: int = 15,  # Limit connections to prevent too many edges
    ) -> List[Dict[str, Any]]:
        """
        Cluster widgets using a graph-based approach that considers multiple factors:
        - Spatial proximity with separate vertical/horizontal thresholds
        - Color similarity (weighted heavily)
        - Widget size consideration
        - Widget type affinity (optional)

        Args:
            content_items: List of content items with position data
            distance_factor: Weight factor for spatial proximity (0-1)
            size_factor: Weight factor for widget size consideration (0-1)
            color_factor: Weight factor for color similarity (0-1)
            type_factor: Weight factor for widget type affinity (0-1)
            horizontal_threshold_factor: Horizontal distance threshold as multiple of width
            vertical_threshold_factor: Vertical distance threshold as multiple of height
            edge_threshold: Minimum edge weight to keep in graph (0-1)
            community_detection: Algorithm for community detection ('louvain', 'label_propagation', or 'greedy')
            max_connections_per_node: Maximum number of connections per node to prevent dense graphs

        Returns:
            List of clustered content items
        """
        console.print("[cyan]Clustering widgets using graph-based approach...[/cyan]")

        # Cache extracted text to support large cluster splitting
        self.extract_mural_text_cache = content_items.copy()

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

                # Normalize width and height if available
                if "width" not in position and "w" in position:
                    position["width"] = position["w"]
                if "height" not in position and "h" in position:
                    position["height"] = position["h"]

                valid_items.append(item)

        if not valid_items:
            console.print(
                "[yellow]No items with valid position data found for graph clustering[/yellow]"
            )
            return content_items

        console.print(
            f"[cyan]Found {len(valid_items)} items with valid position data for graph clustering[/cyan]"
        )

        # Initialize the graph
        G = nx.Graph()

        # Add nodes to the graph
        for i, item in enumerate(valid_items):
            G.add_node(i, item=item)

        # For each node, calculate potential edges and weights
        node_connections = {}

        for i in range(len(valid_items)):
            item1 = valid_items[i]
            pos1 = item1["position"]
            x1, y1 = pos1["x"], pos1["y"]
            width1 = pos1.get("width", 100)  # Default size if not available
            height1 = pos1.get("height", 100)
            area1 = width1 * height1
            type1 = item1.get("widget_type", "")
            color1 = self._extract_color(item1)

            # Store potential connections with their weights
            potential_connections = []

            for j in range(len(valid_items)):
                if i == j:
                    continue

                item2 = valid_items[j]
                pos2 = item2["position"]
                x2, y2 = pos2["x"], pos2["y"]
                width2 = pos2.get("width", 100)
                height2 = pos2.get("height", 100)
                area2 = width2 * height2
                type2 = item2.get("widget_type", "")
                color2 = self._extract_color(item2)

                # Calculate horizontal and vertical distances separately
                horizontal_distance = abs(x1 - x2)
                vertical_distance = abs(y1 - y2)

                # Use different thresholds for horizontal and vertical directions
                max_width = max(width1, width2)
                max_height = max(height1, height2)

                # Calculate horizontal and vertical similarity separately
                horizontal_threshold = max_width * horizontal_threshold_factor
                vertical_threshold = max_height * vertical_threshold_factor

                # Convert to similarity scores (closer = higher)
                horizontal_similarity = max(
                    0, 1 - (horizontal_distance / (horizontal_threshold * 5))
                )
                vertical_similarity = max(
                    0, 1 - (vertical_distance / (vertical_threshold * 5))
                )

                # Weight vertical distance more than horizontal by combining them with different weights
                distance_similarity = (vertical_similarity * 0.7) + (
                    horizontal_similarity * 0.3
                )

                # Size similarity (widgets of similar size may be related)
                size_similarity = 1 - abs(area1 - area2) / max(area1 + area2, 1)

                # Color similarity if colors are available
                color_similarity = self._calculate_color_similarity(color1, color2)

                # Type affinity (same types of widgets are more likely to be related)
                type_similarity = 1.0 if type1 == type2 else 0.5

                # Combine all factors into a single edge weight
                edge_weight = (
                    distance_similarity * distance_factor
                    + size_similarity * size_factor
                    + color_similarity * color_factor
                    + type_similarity * type_factor
                )

                # Only consider edges above threshold
                if edge_weight >= edge_threshold:
                    potential_connections.append((j, edge_weight))

            # Sort connections by weight (highest first) and keep only the top max_connections_per_node
            potential_connections.sort(key=lambda x: x[1], reverse=True)
            node_connections[i] = potential_connections[:max_connections_per_node]

        # Add edges to the graph (only keep the best connections)
        edge_count = 0
        for i, connections in node_connections.items():
            for j, weight in connections:
                # Only add the edge if it doesn't exist yet
                if not G.has_edge(i, j):
                    G.add_edge(i, j, weight=weight)
                    edge_count += 1

        console.print(
            f"[cyan]Created graph with {G.number_of_nodes()} nodes and {edge_count} edges[/cyan]"
        )

        # Force minimum number of communities
        min_communities = max(
            2, len(valid_items) // 50
        )  # At least 2 communities, or 1 per 50 items

        # Detect communities in the graph
        if G.number_of_edges() == 0:
            console.print(
                "[yellow]No edges in graph. Each widget will be its own cluster.[/yellow]"
            )
            communities = [{i} for i in range(len(valid_items))]
        else:
            communities = self._detect_communities(G, algorithm=community_detection)

            # If we got only one community, adjust edge weights to create more communities
            if len(communities) < min_communities:
                console.print(
                    f"[yellow]Only {len(communities)} communities detected. Adjusting to create more clusters...[/yellow]"
                )
                # Try progressively higher thresholds until we get enough communities
                test_graph = G.copy()

                for threshold_multiplier in [1.2, 1.4, 1.6, 1.8, 2.0]:
                    new_threshold = edge_threshold * threshold_multiplier
                    # Remove weaker edges
                    edges_to_remove = [
                        (u, v)
                        for u, v, d in test_graph.edges(data=True)
                        if d["weight"] < new_threshold
                    ]
                    test_graph.remove_edges_from(edges_to_remove)

                    # Check if graph is too fragmented
                    if test_graph.number_of_edges() == 0:
                        break

                    # Detect communities again
                    test_communities = self._detect_communities(
                        test_graph, algorithm=community_detection
                    )

                    if len(test_communities) >= min_communities:
                        communities = test_communities
                        console.print(
                            f"[green]Adjusted to {len(communities)} communities with threshold {new_threshold:.2f}[/green]"
                        )
                        break

                # If we still don't have enough communities, use connected components as a fallback
                if len(communities) < min_communities:
                    connected_components = list(nx.connected_components(G))
                    if len(connected_components) > len(communities):
                        communities = connected_components
                        console.print(
                            f"[yellow]Using {len(communities)} connected components as communities[/yellow]"
                        )

                    # If that still doesn't work, use a spatial partitioning approach
                    if len(communities) < min_communities:
                        console.print(
                            "[yellow]Falling back to spatial partitioning...[/yellow]"
                        )
                        communities = self._spatial_partition(
                            valid_items, min_communities
                        )

        console.print(
            f"[green]Detected {len(communities)} graph-based clusters[/green]"
        )

        # Create clusters from communities
        clusters = []
        for community in communities:
            cluster = [valid_items[i] for i in community]
            clusters.append(cluster)

        # Create new content items from clusters
        clustered_items = []

        for i, cluster in enumerate(clusters):
            # Skip empty clusters
            if not cluster:
                continue

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
                    "id": f"graph-cluster-{i}",
                    "widget_type": "graph_cluster",
                    "widget_ids": widget_ids,
                    "widget_types": list(widget_types),
                    "cluster_size": len(cluster),
                    "position": {"x": centroid_x, "y": centroid_y},
                }
                clustered_items.append(cluster_item)

        console.print(
            f"[green]Generated {len(clustered_items)} graph-based clustered content items[/green]"
        )

        # Check if any clusters are too large for embedding API (limit ~8192 tokens)
        large_clusters = [
            item for item in clustered_items if len(item["content"]) > 12000
        ]
        if large_clusters:
            console.print(
                f"[yellow]Warning: {len(large_clusters)} clusters might be too large for the embedding API.[/yellow]"
            )
            # Split extremely large clusters further if needed
            if any(len(item["content"]) > 20000 for item in large_clusters):
                console.print(f"[yellow]Splitting extra large clusters...[/yellow]")
                clustered_items = self._split_large_clusters(clustered_items)
                console.print(
                    f"[green]After splitting: {len(clustered_items)} clusters[/green]"
                )

        return clustered_items

    def _spatial_partition(
        self, items: List[Dict[str, Any]], target_clusters: int
    ) -> List[set]:
        """
        Fallback method that partitions widgets based on spatial coordinates when
        community detection doesn't produce enough clusters.
        """
        # Get bounding box of all items
        min_x = min(item["position"]["x"] for item in items)
        max_x = max(item["position"]["x"] for item in items)
        min_y = min(item["position"]["y"] for item in items)
        max_y = max(item["position"]["y"] for item in items)

        # Determine if we should split horizontally or vertically
        width = max_x - min_x
        height = max_y - min_y

        # We'll divide the space into a grid based on target clusters
        # Let's determine an appropriate grid size
        grid_size = max(2, int(np.sqrt(target_clusters)))

        # Create partitions based on spatial coordinates
        partitions = defaultdict(set)

        for i, item in enumerate(items):
            x = item["position"]["x"]
            y = item["position"]["y"]

            # Calculate grid position
            grid_x = min(grid_size - 1, int((x - min_x) / width * grid_size))
            grid_y = min(grid_size - 1, int((y - min_y) / height * grid_size))

            # Assign to a partition based on grid cell
            partition_id = grid_y * grid_size + grid_x
            partitions[partition_id].add(i)

        # Return list of partitions, skipping empty ones
        return [partition for partition in partitions.values() if partition]

    def _split_large_clusters(
        self, clustered_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Split large clusters into smaller ones based on spatial proximity."""
        result = []

        for cluster in clustered_items:
            if len(cluster["content"]) <= 20000:
                result.append(cluster)
                continue

            # Get the original widget IDs
            widget_ids = cluster["widget_ids"]
            if len(widget_ids) <= 1:
                # Can't split a single widget
                result.append(cluster)
                continue

            # Look up all the widgets by ID
            widgets = []
            for item_id in widget_ids:
                for original_item in self.extract_mural_text_cache:
                    if original_item.get("id") == item_id:
                        widgets.append(original_item)
                        break

            if not widgets:
                # Couldn't find original widgets
                result.append(cluster)
                continue

            # Use spatial partitioning to split this large cluster
            partitions = self._spatial_partition(widgets, len(widgets) // 10)

            # Create new subclusters
            for i, partition in enumerate(partitions):
                sub_widgets = [widgets[j] for j in partition]

                # Skip empty partitions
                if not sub_widgets:
                    continue

                # Create cluster
                sub_combined_content = []
                sub_widget_ids = []
                sub_widget_types = set()

                for item in sub_widgets:
                    content = item.get("content", "")
                    if content:
                        sub_combined_content.append(content)
                    sub_widget_ids.append(item["id"])
                    sub_widget_types.add(item.get("widget_type", "unknown"))

                if sub_combined_content:
                    # Calculate centroid
                    centroid_x = sum(
                        item["position"]["x"] for item in sub_widgets
                    ) / len(sub_widgets)
                    centroid_y = sum(
                        item["position"]["y"] for item in sub_widgets
                    ) / len(sub_widgets)

                    sub_cluster = {
                        "content": "\n\n".join(sub_combined_content),
                        "type": "text",
                        "id": f"{cluster['id']}-split-{i}",
                        "widget_type": "split_graph_cluster",
                        "widget_ids": sub_widget_ids,
                        "widget_types": list(sub_widget_types),
                        "cluster_size": len(sub_widgets),
                        "position": {"x": centroid_x, "y": centroid_y},
                    }
                    result.append(sub_cluster)

        return result

    def _extract_color(self, item: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
        """
        Extract color information from a widget if available.
        Returns RGB tuple or None if not available.
        """
        # Check common color fields
        for color_field in ["color", "backgroundColor", "background", "fill", "stroke"]:
            if color_field in item:
                color_value = item[color_field]
                if isinstance(color_value, str):
                    return self._parse_color(color_value)

        # Check in style object
        style = item.get("style", {})
        if isinstance(style, dict):
            for color_field in [
                "color",
                "backgroundColor",
                "background",
                "fill",
                "stroke",
            ]:
                if color_field in style:
                    color_value = style[color_field]
                    if isinstance(color_value, str):
                        return self._parse_color(color_value)

        return None

    def _parse_color(self, color_str: str) -> Optional[Tuple[int, int, int]]:
        """
        Parse a color string into RGB values.
        Supports formats: hex (#RRGGBB), rgb(r,g,b)
        """
        if not color_str:
            return None

        # Handle hex format #RRGGBB or #RGB
        if color_str.startswith("#"):
            color_str = color_str.lstrip("#")
            if len(color_str) == 3:
                # Convert #RGB to #RRGGBB
                color_str = "".join([c * 2 for c in color_str])
            if len(color_str) == 6:
                try:
                    r = int(color_str[0:2], 16)
                    g = int(color_str[2:4], 16)
                    b = int(color_str[4:6], 16)
                    return (r, g, b)
                except ValueError:
                    pass

        # Handle rgb(r,g,b) format
        rgb_match = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", color_str)
        if rgb_match:
            try:
                r = int(rgb_match.group(1))
                g = int(rgb_match.group(2))
                b = int(rgb_match.group(3))
                return (r, g, b)
            except ValueError:
                pass

        return None

    def _calculate_color_similarity(
        self,
        color1: Optional[Tuple[int, int, int]],
        color2: Optional[Tuple[int, int, int]],
    ) -> float:
        """
        Calculate similarity between two colors.
        Returns a value between 0 and 1, where 1 is identical.
        """
        if color1 is None or color2 is None:
            return 0.5  # Middle value when color info not available

        # Convert RGB to HSV for better perceptual comparison
        r1, g1, b1 = color1
        r2, g2, b2 = color2

        hsv1 = colorsys.rgb_to_hsv(r1 / 255, g1 / 255, b1 / 255)
        hsv2 = colorsys.rgb_to_hsv(r2 / 255, g2 / 255, b2 / 255)

        # Calculate differences in HSV space
        h_diff = min(
            abs(hsv1[0] - hsv2[0]), 1 - abs(hsv1[0] - hsv2[0])
        )  # Hue is circular
        s_diff = abs(hsv1[1] - hsv2[1])
        v_diff = abs(hsv1[2] - hsv2[2])

        # Weight hue more than saturation and value
        weighted_diff = (h_diff * 0.6) + (s_diff * 0.2) + (v_diff * 0.2)

        # Convert to similarity score
        return 1 - weighted_diff

    def _detect_communities(self, G: nx.Graph, algorithm: str = "louvain") -> List[set]:
        """
        Detect communities within the graph using the specified algorithm.

        Args:
            G: NetworkX graph
            algorithm: Community detection algorithm to use
                ('louvain', 'label_propagation', or 'greedy')

        Returns:
            List of sets, where each set contains node indices belonging to a community
        """
        # Use appropriate community detection algorithm
        if algorithm == "louvain":
            try:
                import community as community_louvain

                partition = community_louvain.best_partition(G)
                communities = defaultdict(set)
                for node, community_id in partition.items():
                    communities[community_id].add(node)
                return list(communities.values())
            except ImportError:
                console.print(
                    "[yellow]python-louvain package not found. Falling back to label propagation.[/yellow]"
                )
                algorithm = "label_propagation"

        if algorithm == "label_propagation":
            communities = {}
            for i, c in enumerate(
                nx.algorithms.community.label_propagation_communities(G)
            ):
                for node in c:
                    communities[node] = i

            community_sets = defaultdict(set)
            for node, comm_id in communities.items():
                community_sets[comm_id].add(node)
            return list(community_sets.values())

        # Fallback to greedy modularity algorithm
        if algorithm == "greedy" or True:  # Default fallback
            communities = {}
            for i, c in enumerate(
                nx.algorithms.community.greedy_modularity_communities(G)
            ):
                for node in c:
                    communities[node] = i

            community_sets = defaultdict(set)
            for node, comm_id in communities.items():
                community_sets[comm_id].add(node)
            return list(community_sets.values())


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

            # Ask which clustering approach to use (if any)
            self.console.print("\n[bold cyan]Widget Clustering Options:[/bold cyan]")
            self.console.print("0. No clustering (use individual widgets)")
            self.console.print(
                "1. Proximity-based clustering (simple distance threshold)"
            )
            self.console.print(
                "2. Graph-based clustering (considers size, color, type, and proximity)"
            )

            clustering_choice = self.console.input(
                "[yellow]Choose clustering method (0-2, default: 0): [/yellow]"
            ).strip()

            if clustering_choice == "1":
                # Proximity-based clustering
                self.console.print("[cyan]Using proximity-based clustering.[/cyan]")

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
                        content_items, distance_threshold=clustering_threshold
                    )

                    if clustered_items:
                        content_items = clustered_items
                        self.console.print(
                            "[green]Successfully clustered widgets using proximity-based method![/green]"
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
                            "[green]Successfully clustered widgets using default proximity settings![/green]"
                        )
            elif clustering_choice == "2":
                # Graph-based clustering
                self.console.print("[cyan]Using graph-based clustering.[/cyan]")

                # Ask if user wants to use default parameters
                use_defaults = (
                    self.console.input(
                        "[yellow]Use default parameters? (Y/n): [/yellow]"
                    )
                    .strip()
                    .lower()
                    != "n"
                )

                if use_defaults:
                    # Apply graph-based clustering with default parameters
                    clustered_items = self.mural_api.cluster_widgets_by_graph(
                        content_items
                    )

                    if clustered_items:
                        content_items = clustered_items
                        self.console.print(
                            "[green]Successfully clustered widgets using graph-based method with default parameters![/green]"
                        )
                    else:
                        self.console.print(
                            "[yellow]Graph-based clustering did not produce any results. Using original items.[/yellow]"
                        )
                else:
                    # Ask for custom parameters
                    try:
                        # Ask for distance factor
                        distance_factor_input = self.console.input(
                            "[yellow]Distance factor (0-1, default: 0.6): [/yellow]"
                        ).strip()
                        distance_factor = 0.6
                        if distance_factor_input:
                            distance_factor = float(distance_factor_input)

                        # Ask for size factor
                        size_factor_input = self.console.input(
                            "[yellow]Size factor (0-1, default: 0.1): [/yellow]"
                        ).strip()
                        size_factor = 0.1
                        if size_factor_input:
                            size_factor = float(size_factor_input)

                        # Ask for color factor
                        color_factor_input = self.console.input(
                            "[yellow]Color factor (0-1, default: 0.3): [/yellow]"
                        ).strip()
                        color_factor = 0.3
                        if color_factor_input:
                            color_factor = float(color_factor_input)

                        # Ask for type factor
                        type_factor_input = self.console.input(
                            "[yellow]Widget type factor (0-1, default: 0.0): [/yellow]"
                        ).strip()
                        type_factor = 0.0
                        if type_factor_input:
                            type_factor = float(type_factor_input)

                        # Ask for edge threshold
                        edge_threshold_input = self.console.input(
                            "[yellow]Edge threshold (0-1, default: 0.7): [/yellow]"
                        ).strip()
                        edge_threshold = 0.7
                        if edge_threshold_input:
                            edge_threshold = float(edge_threshold_input)

                        # Ask for horizontal and vertical threshold factors
                        h_threshold_input = self.console.input(
                            "[yellow]Horizontal threshold factor (default: 0.2): [/yellow]"
                        ).strip()
                        h_threshold = 0.2
                        if h_threshold_input:
                            h_threshold = float(h_threshold_input)

                        v_threshold_input = self.console.input(
                            "[yellow]Vertical threshold factor (default: 0.3): [/yellow]"
                        ).strip()
                        v_threshold = 0.3
                        if v_threshold_input:
                            v_threshold = float(v_threshold_input)

                        # Ask for community detection algorithm
                        self.console.print(
                            "\n[cyan]Community Detection Algorithm:[/cyan]"
                        )
                        self.console.print(
                            "1. Louvain (requires python-louvain package)"
                        )
                        self.console.print("2. Label Propagation")
                        self.console.print("3. Greedy Modularity")

                        algo_choice = self.console.input(
                            "[yellow]Choose algorithm (1-3, default: 1): [/yellow]"
                        ).strip()

                        if algo_choice == "1" or not algo_choice:
                            community_detection = "louvain"
                        elif algo_choice == "2":
                            community_detection = "label_propagation"
                        elif algo_choice == "3":
                            community_detection = "greedy"
                        else:
                            community_detection = "louvain"

                        # Ask for max connections per node
                        max_connections_input = self.console.input(
                            "[yellow]Max connections per node (default: 20): [/yellow]"
                        ).strip()
                        max_connections = 20
                        if max_connections_input:
                            max_connections = int(max_connections_input)

                        self.console.print(
                            f"[cyan]Using graph-based clustering with parameters:[/cyan]\n"
                            f"  - Distance factor: {distance_factor}\n"
                            f"  - Size factor: {size_factor}\n"
                            f"  - Color factor: {color_factor}\n"
                            f"  - Type factor: {type_factor}\n"
                            f"  - Edge threshold: {edge_threshold}\n"
                            f"  - Horizontal threshold: {h_threshold}\n"
                            f"  - Vertical threshold: {v_threshold}\n"
                            f"  - Community detection: {community_detection}\n"
                            f"  - Max connections per node: {max_connections}"
                        )

                        # Apply graph-based clustering
                        clustered_items = self.mural_api.cluster_widgets_by_graph(
                            self.mural_content_items,
                            distance_factor=distance_factor,
                            size_factor=size_factor,
                            color_factor=color_factor,
                            type_factor=type_factor,
                            edge_threshold=edge_threshold,
                            horizontal_threshold_factor=h_threshold,
                            vertical_threshold_factor=v_threshold,
                            community_detection=community_detection,
                            max_connections_per_node=max_connections,
                        )

                        if not clustered_items:
                            self.console.print(
                                "[yellow]Graph-based clustering did not produce any results. Using original items.[/yellow]"
                            )
                        else:
                            content_items = clustered_items
                            self.console.print(
                                "[green]Successfully clustered widgets using graph-based method with custom parameters![/green]"
                            )

                    except ValueError as e:
                        self.console.print(
                            f"[yellow]Error in graph-based clustering parameters: {str(e)}. Using defaults.[/yellow]"
                        )
                        # Apply clustering with default parameters
                        clustered_items = self.mural_api.cluster_widgets_by_graph(
                            self.mural_content_items
                        )

                        if clustered_items:
                            content_items = clustered_items
                            self.console.print(
                                "[green]Successfully clustered widgets using default graph-based settings![/green]"
                            )
            else:
                # No clustering
                self.console.print(
                    "[cyan]Skipping clustering as requested. Using individual widgets.[/cyan]"
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

            # Rest of the method...
            # This part doesn't need to change, so I'm simplifying to focus on fixing linter errors

            # Store the Mural content items for later use
            self.mural_content_items = content_items
            return True

        except Exception as e:
            self.console.print(
                f"[bold red]Error processing mural content: {str(e)}[/bold red]"
            )
            return False

    def debug_embeddings(self):
        """Debug mural content and embeddings for troubleshooting."""
        self.console.print(
            Panel(
                "Debugging Mural content and embeddings",
                title="[bold cyan]Debug Information[/bold cyan]",
            )
        )

        # Check if mural content exists
        if not hasattr(self, "mural_content_items") or not self.mural_content_items:
            self.console.print(
                "[bold red]No Mural content items available for debugging.[/bold red]"
            )
            return

        # Display content item stats
        text_items = [
            item for item in self.mural_content_items if item["type"] == "text"
        ]
        image_items = [
            item for item in self.mural_content_items if item["type"] == "image"
        ]
        other_items = [
            item
            for item in self.mural_content_items
            if item["type"] not in ["text", "image"]
        ]

        self.console.print(f"[cyan]Content Item Statistics:[/cyan]")
        self.console.print(f"• Total items: {len(self.mural_content_items)}")
        self.console.print(f"• Text items: {len(text_items)}")
        self.console.print(f"• Image items: {len(image_items)}")
        self.console.print(f"• Other items: {len(other_items)}")

        # Display sample content
        if text_items:
            self.console.print("\n[cyan]Sample Text Content:[/cyan]")
            sample_size = min(3, len(text_items))
            for i in range(sample_size):
                item = text_items[i]
                truncated_text = (
                    item.get("content", "")[:100] + "..."
                    if len(item.get("content", "")) > 100
                    else item.get("content", "")
                )
                self.console.print(f"{i+1}. ID: {item.get('id', 'N/A')}")
                self.console.print(f"   Text: {truncated_text}")
                self.console.print(
                    f"   Position: x={item.get('position', {}).get('x', 'N/A')}, y={item.get('position', {}).get('y', 'N/A')}"
                )
                if i < sample_size - 1:
                    self.console.print("")

        # Debug embeddings
        self.console.print("\n[cyan]Embeddings Information:[/cyan]")
        if not hasattr(self, "mural_embeddings") or not self.mural_embeddings:
            self.console.print(
                "[yellow]No embeddings have been generated yet.[/yellow]"
            )

            # Offer to generate embeddings
            if text_items and input("Generate embeddings now? (y/n): ").lower() == "y":
                try:
                    self.console.print(
                        "[cyan]Generating embeddings for text content...[/cyan]"
                    )
                    texts = [
                        item.get("content", "")
                        for item in text_items
                        if item.get("content")
                    ]
                    if texts:
                        self.mural_embeddings = self.openai_api.get_embeddings(texts)
                        self.console.print(
                            f"[green]Successfully generated {len(self.mural_embeddings)} embeddings![/green]"
                        )
                    else:
                        self.console.print(
                            "[yellow]No valid text content found for generating embeddings.[/yellow]"
                        )
                except Exception as e:
                    self.console.print(
                        f"[bold red]Error generating embeddings: {str(e)}[/bold red]"
                    )
        else:
            self.console.print(f"• Number of embeddings: {len(self.mural_embeddings)}")
            if self.mural_embeddings and len(self.mural_embeddings) > 0:
                self.console.print(
                    f"• Embedding dimensions: {len(self.mural_embeddings[0])}"
                )
                # Show a sample of the first embedding (first 5 dimensions)
                if len(self.mural_embeddings[0]) > 0:
                    sample_values = [
                        f"{val:.4f}" for val in self.mural_embeddings[0][:5]
                    ]
                    self.console.print(
                        f"• Sample (first 5 dimensions of first embedding): {', '.join(sample_values)}..."
                    )

        # OpenAI API cache stats
        self.console.print("\n[cyan]OpenAI API Cache Statistics:[/cyan]")
        self.openai_api.print_cache_stats()

    def generate_business_plan_section(self, section_title: str) -> str:
        """Generate a business plan section using retrieved content and OpenAI's GPT."""
        try:
            self.console.print(f"[cyan]Generating {section_title} section...[/cyan]")

            # Check if we have mural content and embeddings
            if not hasattr(self, "mural_content_items") or not self.mural_content_items:
                self.console.print(
                    "[bold red]No mural content available. Please fetch content first.[/bold red]"
                )
                return None

            if not hasattr(self, "mural_embeddings") or not self.mural_embeddings:
                self.console.print(
                    "[bold red]No embeddings available. Please ensure content is processed.[/bold red]"
                )
                return None

            # Use the OpenAI API to search for relevant content
            query_embedding = self.openai_api.get_embeddings([section_title])[0]

            # Get text content from mural items
            text_items = [
                item for item in self.mural_content_items if item["type"] == "text"
            ]
            texts = [
                item.get("content", "") for item in text_items if item.get("content")
            ]

            if not texts:
                self.console.print(
                    "[bold yellow]No text content found to generate section.[/bold yellow]"
                )
                return None

            # Search for relevant content
            top_results = SemanticSearch.search(
                query_embedding, self.mural_embeddings, texts, top_k=10
            )

            if not top_results:
                self.console.print(
                    "[bold yellow]No relevant content found for this section.[/bold yellow]"
                )
                return None

            # Extract content and sources for the section
            context_items = []
            sources = []

            for content, score, idx in top_results:
                item = text_items[idx]
                context_items.append(content)
                source_info = f"Source {len(sources)+1}: {item.get('id', 'Unknown ID')}"
                sources.append(source_info)

            # Combine content for context
            context = "\n\n".join(context_items)

            # Generate the section using OpenAI
            return self.openai_api.generate_business_plan_section(
                section_title, context, sources
            )

        except Exception as e:
            self.console.print(
                f"[bold red]An unexpected error occurred: {str(e)}[/bold red]"
            )
            return None

    def export_to_file(self, content: str, filename: str, format: str = "md") -> bool:
        """Export the generated content to a file.

        Args:
            content: The content to export
            filename: The name of the file to create
            format: The format of the file (md or txt)

        Returns:
            bool: True if the export was successful, False otherwise
        """
        try:
            # Create the output directory if it doesn't exist
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            # Add extension if not provided
            if not filename.endswith(f".{format}"):
                filename = f"{filename}.{format}"

            # Create the full file path
            file_path = output_dir / filename

            # Write the content to the file
            with open(file_path, "w") as f:
                f.write(content)

            self.console.print(f"[green]Successfully exported to {file_path}[/green]")
            return True

        except Exception as e:
            self.console.print(
                f"[bold red]Error exporting to file: {str(e)}[/bold red]"
            )
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
                    filename = input("Enter filename (without extension): ")
                    if not filename:
                        # Use a sanitized version of the section title as the filename
                        filename = (
                            section_title.lower().replace(" ", "_").replace("/", "_")
                        )
                    self.export_to_file(generated_section, filename, format)

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
