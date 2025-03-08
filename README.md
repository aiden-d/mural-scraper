# Business Plan Drafter

A simple CLI tool that generates business plan drafts using Mural board embeddings and OpenAI's GPT.

## Setup

1. Clone this repository:
```
git clone https://github.com/yourusername/business-plan-drafter.git
cd business-plan-drafter
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Register an OAuth Application with Mural:
   - Go to the [Mural Developer Portal](https://developers.mural.co/)
   - Register a new application
   - Set the redirect URI to: `http://localhost:8085/oauth/callback`
   - Copy your Client ID and Client Secret

4. Create a `.env` file in the root directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
MURAL_CLIENT_ID=your_mural_client_id
MURAL_CLIENT_SECRET=your_mural_client_secret
```

## Usage

Run the tool with:
```
python business_plan_drafter.py
```

### Authentication

The first time you run the tool, it will:
1. Open your web browser to authenticate with Mural
2. Prompt you to log in to your Mural account (if not already logged in)
3. Ask for permission to access your Mural data
4. Redirect back to the application

After you authenticate, the tool will save your tokens securely for future use. The tokens will be automatically refreshed when they expire.

### Workflow

1. The tool will display a list of your Mural projects
2. Select a project by entering its number
3. Paste a business plan section specification or title
4. The tool will search your Mural board for relevant content and generate a draft
5. The generated section will be displayed in the terminal
6. Optionally export the results to a file

## Example

```
$ python business_plan_drafter.py

✨ Business Plan Drafter ✨

Opening browser for Mural authentication...
Authentication successful!

Fetching Mural projects...

Available Projects:
1. Strategic Planning 2023
2. Product Launch Brainstorm
3. Market Research Board

Select a project (1-3): 1

Project "Strategic Planning 2023" selected!

Enter a business plan section title (or "q" to quit): Executive Summary

Searching Mural board for relevant content...
Generating Executive Summary...

----------------------------------------
# Executive Summary

Our company aims to revolutionize the industry by providing innovative solutions
that address key customer pain points. With a focused go-to-market strategy and
a talented team of industry experts, we are well-positioned to capture significant
market share within the next 24 months.

Our financial projections indicate potential revenue of $2.5M in year one,
growing to $8.7M by year three, with break-even expected in month 18.
----------------------------------------

Enter a business plan section title (or "q" to quit): q

Thank you for using Business Plan Drafter!
```

## Requirements

- Python 3.10 or higher
- OpenAI API key
- Mural OAuth credentials
- Port 8085 available for OAuth callback server

## Security Notes

- Your OAuth tokens are stored locally in `~/.business_plan_drafter/tokens.json`
- Your OpenAI API key and Mural OAuth credentials are stored in the `.env` file
- The tool only requests read access to your Mural boards, not write access 

## Diagnostic Script

You can use the `mural_api_test.py` script to help identify why you're getting "No workspaces found" despite having workspaces in your Mural account. This script provides detailed debugging information about each step of the OAuth and API process.

### How to Use the Test Script

1. First, ensure your `.env` file has your Mural OAuth credentials:
   ```
   MURAL_CLIENT_ID=your_mural_client_id_here
   MURAL_CLIENT_SECRET=your_mural_client_secret_here
   ```

2. Run the diagnostic script:
   ```
   python mural_api_test.py
   ```

3. If you want to force a new authentication (remove existing tokens), run:
   ```
   python mural_api_test.py --force-auth
   ```

### What the Test Script Does

The script performs three main tests:

1. **Environment Variables Check**: Verifies that your Mural credentials are available in the environment
2. **Token Test**: Checks if OAuth tokens exist and are valid
3. **Workspaces API Test**: Attempts to fetch your workspaces and displays detailed API responses

Throughout the process, it provides verbose logging so you can see:
- The exact URLs being accessed
- HTTP status codes
- Authentication flow details
- Any error responses
- The full JSON response structure when trying to fetch workspaces

### Common Issues the Script Can Help Identify

1. **Missing or Invalid Credentials**: Confirms your client ID and secret are accessible
2. **OAuth Scopes**: Shows the exact scopes being requested
3. **Authentication Flow Problems**: Logs each step of the browser authentication
4. **Token Refresh Issues**: Logs token refresh attempts and responses
5. **API Response Details**: Shows the full API response for troubleshooting

After running this script, look for:
1. Any red error messages
2. The HTTP status code when fetching workspaces (should be 200)
3. The full JSON response structure, which may contain error messages or clues

This will help pinpoint exactly where the issue is occurring and provide detailed information to resolve it. 