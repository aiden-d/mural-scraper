#!/usr/bin/env python3
"""
Mural OAuth authentication handler.
"""

import json
import time
import webbrowser
import secrets
import threading
import socketserver
from typing import Dict, List, Any, Optional
from pathlib import Path
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from mural_api.auth_handler import OAuthCallbackHandler

# Initialize console for rich text output
console = Console()

# File paths
CONFIG_DIR = Path.home() / ".business_plan_drafter"
TOKEN_FILE = CONFIG_DIR / "tokens.json"

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
