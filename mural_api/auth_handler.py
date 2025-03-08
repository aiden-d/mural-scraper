#!/usr/bin/env python3
"""
Handler for OAuth callback.
"""

import http.server
import urllib.parse

# Constants
AUTH_REDIRECT_PATH = "/oauth/callback"


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
