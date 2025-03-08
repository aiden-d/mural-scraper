#!/usr/bin/env python3
"""
Mural API Test Script - Diagnose issues with Mural OAuth and API connectivity
"""

import os
import sys
import json
import time
import webbrowser
import http.server
import socketserver
import urllib.parse
import secrets
import threading
import requests
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables
load_dotenv()

# Initialize console for rich text output
console = Console()

# API Authentication
MURAL_CLIENT_ID = os.getenv("MURAL_CLIENT_ID")
MURAL_CLIENT_SECRET = os.getenv("MURAL_CLIENT_SECRET")

# File paths
CONFIG_DIR = Path.home() / ".business_plan_drafter"
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
        console.print(
            f"[cyan]>>> Received callback request to path: {self.path}[/cyan]"
        )

        if self.path.startswith(AUTH_REDIRECT_PATH):
            # Parse the query parameters
            query = urllib.parse.urlparse(self.path).query
            query_params = urllib.parse.parse_qs(query)

            console.print(f"[cyan]>>> Query parameters: {query_params}[/cyan]")

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
                <p>You've successfully authorized the application to access your Mural account.</p>
                <p>You can now close this window and return to the application.</p>
                </body>
                </html>
                """
                self.wfile.write(html_response.encode())
                console.print(
                    "[green]>>> Authorization code received successfully[/green]"
                )
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
                console.print(
                    "[bold red]>>> Error in authorization callback - missing code or state[/bold red]"
                )
        else:
            self.send_response(404)
            self.end_headers()
            console.print(
                f"[yellow]>>> Received request to unknown path: {self.path}[/yellow]"
            )

    def log_message(self, format, *args):
        """Suppress HTTP server logs."""
        return


def authenticate():
    """Perform OAuth authentication flow and return tokens."""
    # Generate a random state value for security
    state = secrets.token_urlsafe(16)

    # Build the authorization URL
    auth_url = f"{MURAL_AUTH_URL}?client_id={MURAL_CLIENT_ID}&redirect_uri={OAUTH_CALLBACK_URL}&scope={MURAL_SCOPES}&state={state}&response_type=code"

    console.print(f"[cyan]>>> Using client ID: {MURAL_CLIENT_ID}[/cyan]")
    console.print(f"[cyan]>>> Redirect URL: {OAUTH_CALLBACK_URL}[/cyan]")
    console.print(f"[cyan]>>> Scopes: {MURAL_SCOPES}[/cyan]")

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
                return None
            time.sleep(0.5)

        # Verify the state parameter
        if OAuthCallbackHandler.state != state:
            server.shutdown()
            console.print(
                "[bold red]Authentication failed: State mismatch. Please try again.[/bold red]"
            )
            return None

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
            "client_id": MURAL_CLIENT_ID,
            "client_secret": MURAL_CLIENT_SECRET,
            "code": auth_code,
            "redirect_uri": OAUTH_CALLBACK_URL,
            "grant_type": "authorization_code",
        }

        progress.update(task, description="[cyan]Exchanging code for tokens...")

        try:
            console.print(
                f"[cyan]>>> Sending token request to: {MURAL_TOKEN_URL}[/cyan]"
            )
            response = requests.post(MURAL_TOKEN_URL, data=data)

            console.print(
                f"[cyan]>>> Token response status code: {response.status_code}[/cyan]"
            )

            if response.status_code != 200:
                console.print(
                    f"[bold red]>>> Error response: {response.text}[/bold red]"
                )

            response.raise_for_status()

            token_data = response.json()
            console.print(
                f"[green]>>> Successfully retrieved tokens. Access token starts with: {token_data['access_token'][:10]}...[/green]"
            )

            # Save the tokens
            tokens = {
                "access_token": token_data["access_token"],
                "refresh_token": token_data["refresh_token"],
                "expires_at": time.time()
                + token_data["expires_in"]
                - 60,  # 60 seconds buffer
            }

            return tokens

        except requests.exceptions.RequestException as e:
            console.print(
                f"[bold red]Error exchanging code for tokens: {str(e)}[/bold red]"
            )
            return None


def save_tokens(tokens):
    """Save tokens to file."""
    try:
        with open(TOKEN_FILE, "w") as f:
            json.dump(tokens, f)
        console.print(f"[green]>>> Tokens saved to {TOKEN_FILE}[/green]")
        return True
    except Exception as e:
        console.print(f"[bold red]>>> Error saving tokens: {str(e)}[/bold red]")
        return False


def load_tokens():
    """Load tokens from file."""
    if TOKEN_FILE.exists():
        try:
            with open(TOKEN_FILE, "r") as f:
                tokens = json.load(f)
            console.print(f"[green]>>> Tokens loaded from {TOKEN_FILE}[/green]")
            return tokens
        except json.JSONDecodeError as e:
            console.print(
                f"[bold red]>>> Error loading tokens (invalid JSON): {str(e)}[/bold red]"
            )
            return None
    else:
        console.print(f"[yellow]>>> Token file not found: {TOKEN_FILE}[/yellow]")
        return None


def get_access_token():
    """Get a valid access token, refreshing if necessary."""
    tokens = load_tokens()

    if not tokens:
        console.print("[yellow]>>> No tokens found, initiating authentication[/yellow]")
        tokens = authenticate()
        if tokens:
            save_tokens(tokens)
        return tokens["access_token"] if tokens else None

    # Check if the access token is expired
    expires_at = tokens.get("expires_at", 0)

    if time.time() > expires_at:
        console.print("[yellow]>>> Token expired, refreshing...[/yellow]")
        # Token is expired, refresh it
        refresh_token = tokens["refresh_token"]

        data = {
            "client_id": MURAL_CLIENT_ID,
            "client_secret": MURAL_CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }

        try:
            response = requests.post(MURAL_TOKEN_URL, data=data)
            response.raise_for_status()

            token_data = response.json()
            console.print("[green]>>> Token refreshed successfully[/green]")

            # Update tokens with new access token
            tokens.update(
                {
                    "access_token": token_data["access_token"],
                    "expires_at": time.time()
                    + token_data["expires_in"]
                    - 60,  # 60 seconds buffer
                    "refresh_token": token_data.get(
                        "refresh_token", tokens["refresh_token"]
                    ),
                }
            )

            save_tokens(tokens)
            return tokens["access_token"]

        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]>>> Error refreshing token: {str(e)}[/bold red]")

            # Try a full re-authentication
            console.print("[yellow]>>> Trying re-authentication...[/yellow]")
            tokens = authenticate()
            if tokens:
                save_tokens(tokens)
                return tokens["access_token"]
            return None

    console.print("[green]>>> Using existing valid token[/green]")
    return tokens["access_token"]


def test_get_workspaces():
    """Test fetching workspaces from Mural API."""
    access_token = get_access_token()

    if not access_token:
        console.print("[bold red]>>> Failed to get access token[/bold red]")
        return

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    url = f"{MURAL_API_BASE_URL}/workspaces"
    console.print(f"[cyan]>>> Making request to: {url}[/cyan]")

    try:
        response = requests.get(url, headers=headers)
        console.print(f"[cyan]>>> Response status code: {response.status_code}[/cyan]")

        if response.status_code != 200:
            console.print(f"[bold red]>>> Error response: {response.text}[/bold red]")

        response.raise_for_status()

        data = response.json()

        # Analyze response structure
        console.print("[cyan]>>> API Response structure:[/cyan]")
        for key, value in data.items():
            if isinstance(value, list):
                console.print(f"[cyan]>>>   {key}: List with {len(value)} items[/cyan]")
                if value and len(value) > 0:
                    console.print(
                        "[cyan]>>>   First item keys: "
                        + ", ".join(value[0].keys())
                        + "[/cyan]"
                    )
            else:
                console.print(f"[cyan]>>>   {key}: {type(value).__name__}[/cyan]")

        # Try to find workspaces in either "value" or "data" fields
        workspaces = []
        if "value" in data and isinstance(data["value"], list):
            workspaces = data["value"]
            console.print("[green]>>> Found workspaces in 'value' field[/green]")
        elif "data" in data and isinstance(data["data"], list):
            workspaces = data["data"]
            console.print("[green]>>> Found workspaces in 'data' field[/green]")
        else:
            console.print(
                "[yellow]>>> Could not find workspaces in 'value' or 'data' fields[/yellow]"
            )
            # If we can't find workspaces in expected fields, examine lists in the response
            for key, value in data.items():
                if (
                    isinstance(value, list)
                    and len(value) > 0
                    and isinstance(value[0], dict)
                    and "name" in value[0]
                ):
                    workspaces = value
                    console.print(
                        f"[green]>>> Found potential workspaces in '{key}' field[/green]"
                    )
                    break

        console.print(f"[green]>>> Found {len(workspaces)} workspaces[/green]")

        if workspaces:
            # Check for required keys in workspace objects
            required_keys = ["id", "name"]
            missing_keys = set()

            for workspace in workspaces:
                for key in required_keys:
                    if key not in workspace:
                        missing_keys.add(key)

            if missing_keys:
                console.print(
                    f"[yellow]>>> Warning: Workspace objects missing required keys: {', '.join(missing_keys)}[/yellow]"
                )
                console.print(
                    f"[cyan]>>> Workspace object structure: {json.dumps(workspaces[0], indent=2)}[/cyan]"
                )

            # Display workspace info
            workspace_info = []
            for workspace in workspaces[:5]:  # Limit to first 5
                info = {}
                for key in ["id", "name"]:
                    info[key] = workspace.get(key, f"<missing {key}>")
                workspace_info.append(f"ID: {info['id']}, Name: {info['name']}")

            console.print(
                Panel.fit(
                    "\n".join(workspace_info)
                    + ("\n..." if len(workspaces) > 5 else ""),
                    title="Workspaces (First 5)",
                    border_style="green",
                )
            )
        else:
            console.print("[yellow]>>> No workspaces found in response[/yellow]")
            console.print(
                f"[cyan]>>> Full response: {json.dumps(data, indent=2)}[/cyan]"
            )

    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]>>> Error fetching workspaces: {str(e)}[/bold red]")
    except KeyError as e:
        console.print(
            f"[bold red]>>> KeyError while processing workspaces: {str(e)}[/bold red]"
        )
        console.print(f"[cyan]>>> Full response: {json.dumps(data, indent=2)}[/cyan]")


def test_get_murals():
    """Test fetching murals from a workspace and inspect their structure."""
    access_token = get_access_token()

    if not access_token:
        console.print("[bold red]>>> Failed to get access token[/bold red]")
        return

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    # First get workspaces
    url = f"{MURAL_API_BASE_URL}/workspaces"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()
        workspaces = []

        # Try to find workspaces in either "value" or "data" fields
        if "value" in data and isinstance(data["value"], list):
            workspaces = data["value"]
        elif "data" in data and isinstance(data["data"], list):
            workspaces = data["data"]

        if not workspaces:
            console.print("[yellow]>>> No workspaces found to test murals[/yellow]")
            return

        # Select the first workspace to test murals
        workspace = workspaces[0]
        workspace_id = workspace.get("id")
        workspace_name = workspace.get("name", f"Workspace {workspace_id}")

        console.print(
            f"[cyan]>>> Testing murals API with workspace: {workspace_name} (ID: {workspace_id})[/cyan]"
        )

        # Get murals from the workspace
        murals_url = f"{MURAL_API_BASE_URL}/workspaces/{workspace_id}/murals"
        console.print(f"[cyan]>>> Making request to: {murals_url}[/cyan]")

        murals_response = requests.get(murals_url, headers=headers)
        console.print(
            f"[cyan]>>> Response status code: {murals_response.status_code}[/cyan]"
        )

        if murals_response.status_code != 200:
            console.print(
                f"[bold red]>>> Error response: {murals_response.text}[/bold red]"
            )
            return

        murals_data = murals_response.json()

        # Analyze response structure
        console.print("[cyan]>>> Murals API Response structure:[/cyan]")
        for key, value in murals_data.items():
            if isinstance(value, list):
                console.print(f"[cyan]>>>   {key}: List with {len(value)} items[/cyan]")
                if value and len(value) > 0:
                    console.print(
                        "[cyan]>>>   First item keys: "
                        + ", ".join(sorted(value[0].keys()))
                        + "[/cyan]"
                    )
            else:
                console.print(f"[cyan]>>>   {key}: {type(value).__name__}[/cyan]")

        # Try to find murals in either "value" or "data" fields
        murals = []
        if "value" in murals_data and isinstance(murals_data["value"], list):
            murals = murals_data["value"]
            console.print("[green]>>> Found murals in 'value' field[/green]")
        elif "data" in murals_data and isinstance(murals_data["data"], list):
            murals = murals_data["data"]
            console.print("[green]>>> Found murals in 'data' field[/green]")

        if not murals:
            console.print("[yellow]>>> No murals found in response[/yellow]")
            console.print(
                f"[cyan]>>> Full response: {json.dumps(murals_data, indent=2)}[/cyan]"
            )
            return

        console.print(f"[green]>>> Found {len(murals)} murals[/green]")

        # Inspect first mural in detail
        if murals:
            first_mural = murals[0]
            console.print("[cyan]>>> First mural structure:[/cyan]")
            console.print(f"[cyan]>>> {json.dumps(first_mural, indent=2)}[/cyan]")

            # Look for potential name fields
            potential_name_fields = [
                "name",
                "title",
                "displayName",
                "display_name",
                "label",
            ]
            found_fields = []

            for field in potential_name_fields:
                if field in first_mural:
                    found_fields.append(f"{field}: {first_mural[field]}")

            if found_fields:
                console.print("[green]>>> Potential name fields found:[/green]")
                for field in found_fields:
                    console.print(f"[green]>>>   {field}[/green]")
            else:
                console.print(
                    "[yellow]>>> No standard name fields found. Here are all fields:[/yellow]"
                )
                for key, value in first_mural.items():
                    if isinstance(value, str):
                        console.print(f"[yellow]>>>   {key}: {value}[/yellow]")

    except Exception as e:
        console.print(f"[bold red]>>> Error testing murals API: {str(e)}[/bold red]")


def test_get_mural_widgets():
    """Test fetching widgets from a mural and inspect their structure."""
    access_token = get_access_token()

    if not access_token:
        console.print("[bold red]>>> Failed to get access token[/bold red]")
        return

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    # First get workspaces
    console.print("[cyan]>>> Getting workspaces...[/cyan]")
    url = f"{MURAL_API_BASE_URL}/workspaces"
    response = requests.get(url, headers=headers)
    workspaces = response.json().get("value", [])

    if not workspaces:
        console.print("[yellow]>>> No workspaces found[/yellow]")
        return

    # Select the first workspace
    workspace = workspaces[0]
    workspace_id = workspace.get("id")
    console.print(
        f"[cyan]>>> Using workspace: {workspace.get('name')} (ID: {workspace_id})[/cyan]"
    )

    # Get murals from the workspace
    console.print("[cyan]>>> Getting murals...[/cyan]")
    murals_url = f"{MURAL_API_BASE_URL}/workspaces/{workspace_id}/murals"
    murals_response = requests.get(murals_url, headers=headers)
    murals = murals_response.json().get("value", [])

    if not murals:
        console.print("[yellow]>>> No murals found[/yellow]")
        return

    # Select the first mural
    mural = murals[0]
    mural_id = mural.get("id")
    mural_title = mural.get("title")
    console.print(f"[cyan]>>> Using mural: {mural_title} (ID: {mural_id})[/cyan]")

    # Get the widgets from the mural
    console.print("[cyan]>>> Fetching mural content...[/cyan]")

    # First try to access the mural itself
    mural_url = f"{MURAL_API_BASE_URL}/murals/{mural_id}"
    console.print(f"[cyan]>>> Accessing mural at: {mural_url}[/cyan]")
    mural_response = requests.get(mural_url, headers=headers)
    console.print(f"[cyan]>>> Mural access status: {mural_response.status_code}[/cyan]")

    if mural_response.status_code != 200:
        console.print(
            "[yellow]>>> Could not access mural directly, trying alternative ID format...[/yellow]"
        )
        if "." in mural_id:
            short_id = mural_id.split(".")[1]  # Get the part after the dot
            mural_url = f"{MURAL_API_BASE_URL}/murals/{short_id}"
            console.print(f"[cyan]>>> Trying with numeric ID only: {mural_url}[/cyan]")
            mural_response = requests.get(mural_url, headers=headers)
            console.print(
                f"[cyan]>>> Alternative mural access status: {mural_response.status_code}[/cyan]"
            )

            if mural_response.status_code == 200:
                mural_id = short_id
                console.print(
                    f"[green]>>> Successfully accessed mural with alternative ID[/green]"
                )

    # Try getting widgets
    widgets_url = f"{MURAL_API_BASE_URL}/murals/{mural_id}/widgets"
    console.print(f"[cyan]>>> Fetching widgets from: {widgets_url}[/cyan]")
    widgets_response = requests.get(widgets_url, headers=headers)
    console.print(
        f"[cyan]>>> Widgets response status: {widgets_response.status_code}[/cyan]"
    )

    if widgets_response.status_code != 200:
        console.print(
            "[yellow]>>> Failed to get widgets directly. Trying alternative approaches...[/yellow]"
        )

        # Try with API-provided ID if available
        if mural_response.status_code == 200 and "id" in mural_response.json():
            api_mural_id = mural_response.json()["id"]
            if api_mural_id != mural_id:
                console.print(
                    f"[cyan]>>> Trying with API-provided mural ID: {api_mural_id}[/cyan]"
                )
                alt_widgets_url = f"{MURAL_API_BASE_URL}/murals/{api_mural_id}/widgets"
                widgets_response = requests.get(alt_widgets_url, headers=headers)
                console.print(
                    f"[cyan]>>> Alternative widgets response status: {widgets_response.status_code}[/cyan]"
                )

    # Print the response details
    if widgets_response.status_code == 200:
        widgets_data = widgets_response.json()
        console.print("[cyan]>>> Widgets API Response structure:[/cyan]")

        # Print the top-level structure
        for key, value in widgets_data.items():
            if isinstance(value, list):
                console.print(f"[cyan]>>>   {key}: List with {len(value)} items[/cyan]")
                if value and len(value) > 0:
                    console.print(
                        f"[cyan]>>>   First item keys: {sorted(value[0].keys())}[/cyan]"
                    )
            else:
                console.print(f"[cyan]>>>   {key}: {type(value).__name__}[/cyan]")

        # Try to find widgets in either "widgets", "value" or "data" fields
        widgets = []
        if "widgets" in widgets_data and isinstance(widgets_data["widgets"], list):
            widgets = widgets_data["widgets"]
            console.print("[green]>>> Found widgets in 'widgets' field[/green]")
        elif "value" in widgets_data and isinstance(widgets_data["value"], list):
            widgets = widgets_data["value"]
            console.print("[green]>>> Found widgets in 'value' field[/green]")
        elif "data" in widgets_data and isinstance(widgets_data["data"], list):
            widgets = widgets_data["data"]
            console.print("[green]>>> Found widgets in 'data' field[/green]")
        elif isinstance(widgets_data, list):
            widgets = widgets_data
            console.print("[green]>>> Response is directly a list of widgets[/green]")

        console.print(f"[green]>>> Found {len(widgets)} widgets[/green]")

        if widgets:
            # Show a sample of the first widget
            console.print("[cyan]>>> Sample widget structure:[/cyan]")
            console.print(json.dumps(widgets[0], indent=2))

            # Count widget types
            widget_types = {}
            for widget in widgets:
                widget_type = widget.get("type", "unknown")
                widget_types[widget_type] = widget_types.get(widget_type, 0) + 1

            console.print("[cyan]>>> Widget types found:[/cyan]")
            for widget_type, count in widget_types.items():
                console.print(f"[cyan]>>>   {widget_type}: {count}[/cyan]")
    else:
        console.print(
            f"[bold red]>>> Failed to get widgets. Status code: {widgets_response.status_code}[/bold red]"
        )
        console.print(
            f"[bold red]>>> Error response: {widgets_response.text}[/bold red]"
        )


def run_tests():
    """Run all diagnostic tests."""
    console.print(
        Panel.fit("Mural API Diagnostic Tests", title="Started", border_style="cyan")
    )

    # Check environment variables
    console.print("\n[bold cyan]1. Checking environment variables...[/bold cyan]")

    if not MURAL_CLIENT_ID:
        console.print(
            "[bold red]ERROR: MURAL_CLIENT_ID not found in environment variables[/bold red]"
        )
        return
    console.print("[green]✓ MURAL_CLIENT_ID found[/green]")

    if not MURAL_CLIENT_SECRET:
        console.print(
            "[bold red]ERROR: MURAL_CLIENT_SECRET not found in environment variables[/bold red]"
        )
        return
    console.print("[green]✓ MURAL_CLIENT_SECRET found[/green]")

    # Test token loading
    console.print("\n[bold cyan]2. Testing token loading...[/bold cyan]")
    tokens = load_tokens()
    if tokens:
        console.print("[green]✓ Existing tokens found[/green]")

        # Check token expiration
        expires_at = tokens.get("expires_at", 0)
        if time.time() > expires_at:
            console.print("[yellow]! Token is expired, will need to refresh[/yellow]")
        else:
            remaining = int(expires_at - time.time())
            console.print(f"[green]✓ Token valid for {remaining} more seconds[/green]")
    else:
        console.print(
            "[yellow]! No existing tokens found, will need to authenticate[/yellow]"
        )

    # Test workspaces API
    console.print("\n[bold cyan]3. Testing workspaces API...[/bold cyan]")
    test_get_workspaces()

    # Test murals API
    console.print("\n[bold cyan]4. Testing murals API...[/bold cyan]")
    test_get_murals()

    # Test mural widgets API
    console.print("\n[bold cyan]5. Testing mural widgets API...[/bold cyan]")
    test_get_mural_widgets()

    console.print("\n[bold green]Tests completed![/bold green]")


if __name__ == "__main__":
    try:
        # Check for command line flags
        if len(sys.argv) > 1 and sys.argv[1] == "--force-auth":
            console.print("[bold cyan]Forcing new authentication...[/bold cyan]")
            if TOKEN_FILE.exists():
                os.remove(TOKEN_FILE)
                console.print(
                    f"[yellow]Removed existing token file: {TOKEN_FILE}[/yellow]"
                )

        run_tests()
    except KeyboardInterrupt:
        console.print("\n\n[bold yellow]Tests interrupted. Exiting...[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred: {str(e)}[/bold red]")
        sys.exit(1)
