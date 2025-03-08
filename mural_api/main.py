#!/usr/bin/env python3
"""
Business Plan Drafter - A CLI tool to generate business plan drafts
using Mural board embeddings and OpenAI's GPT.
"""

import os
import sys
from rich.console import Console
from dotenv import load_dotenv

# Import all the classes from their respective files
from mural_api.oauth import MuralOAuth
from mural_api.api import MuralAPI
from mural_api.openai_client import OpenAIAPI
from mural_api.search import SemanticSearch
from mural_api.business_plan_drafter import BusinessPlanDrafter

# Load environment variables
load_dotenv()

# Initialize console for rich text output
console = Console()

# API Authentication
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MURAL_CLIENT_ID = os.getenv("MURAL_CLIENT_ID")
MURAL_CLIENT_SECRET = os.getenv("MURAL_CLIENT_SECRET")


def main():
    """Main entry point for the Business Plan Drafter."""
    try:
        drafter = BusinessPlanDrafter()
        drafter.run()
    except KeyboardInterrupt:
        console.print("\n\n[bold yellow]Program interrupted. Exiting...[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred: {str(e)}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
