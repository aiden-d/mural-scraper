#!/usr/bin/env python3
"""
Main Business Plan Drafter class that orchestrates the entire application.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from tabulate import tabulate

from mural_api.oauth import MuralOAuth
from mural_api.api import MuralAPI
from mural_api.openai_client import OpenAIAPI
from mural_api.search import SemanticSearch

# File paths
CONFIG_DIR = Path.home() / ".business_plan_drafter"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Create config directory if it doesn't exist
CONFIG_DIR.mkdir(exist_ok=True)


class BusinessPlanDrafter:
    """Main class for the Business Plan Drafter CLI tool."""

    def __init__(self):
        """Initialize the Business Plan Drafter."""
        self.console = Console()
        self.check_api_keys()

        # Get API keys from environment variables
        from mural_api.main import MURAL_CLIENT_ID, MURAL_CLIENT_SECRET, OPENAI_API_KEY

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
        from mural_api.main import (
            OPENAI_API_KEY,
            MURAL_CLIENT_ID,
            MURAL_CLIENT_SECRET,
            console,
        )

        if not OPENAI_API_KEY:
            console.print(
                "[bold red]Error:[/bold red] OpenAI API key not found. "
                "Please add it to your .env file as OPENAI_API_KEY."
            )
            sys.exit(1)

        if not MURAL_CLIENT_ID or not MURAL_CLIENT_SECRET:
            console.print(
                "[bold red]Error:[/bold red] Mural API credentials not found. "
                "Please add them to your .env file as MURAL_CLIENT_ID and MURAL_CLIENT_SECRET."
            )
            sys.exit(1)

    def load_config(self):
        """Load configuration from file."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)
                    self.selected_project = config.get("selected_project")
                    self.console.print(
                        "[green]Configuration loaded successfully[/green]"
                    )
            except json.JSONDecodeError:
                self.console.print(
                    "[yellow]Warning: Could not parse configuration file. Using defaults.[/yellow]"
                )
            except Exception as e:
                self.console.print(
                    f"[yellow]Warning: Error loading configuration: {str(e)}[/yellow]"
                )

    def save_config(self):
        """Save configuration to file."""
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
                            # Add workspace information to each mural
                            mural["workspace_name"] = workspace_name
                            mural["workspace_id"] = workspace_id
                            all_murals.append(mural)

                    except Exception as e:
                        self.console.print(
                            f"[yellow]Error fetching murals from workspace {workspace_name}: {str(e)}[/yellow]"
                        )

            except Exception as e:
                self.console.print(
                    f"[bold red]Error fetching projects: {str(e)}[/bold red]"
                )
                sys.exit(1)

        # Second phase: Display murals and let user select one
        if not all_murals:
            self.console.print(
                "[yellow]No murals found. Please check your Mural account.[/yellow]"
            )
            sys.exit(1)

        self.console.print(f"\n[green]Found {len(all_murals)} murals[/green]")

        # Prepare table data
        table_data = []
        for i, mural in enumerate(all_murals, 1):
            table_data.append(
                [
                    i,
                    mural.get("title", "Untitled"),
                    mural.get("workspace_name", "Unknown"),
                    mural.get("id", "Unknown ID"),
                ]
            )

        # Display table
        self.console.print("\n[bold cyan]Available Murals:[/bold cyan]")
        self.console.print(
            tabulate(
                table_data,
                headers=["#", "Mural Name", "Workspace", "Mural ID"],
                tablefmt="grid",
            )
        )

        # Let user select a mural
        while True:
            try:
                selection = input("\nEnter the number of the mural to use: ")
                index = int(selection) - 1
                if 0 <= index < len(all_murals):
                    selected_mural = all_murals[index]

                    # Create a project object with all necessary information
                    project = {
                        "name": selected_mural.get("title", "Untitled"),
                        "mural_id": selected_mural.get("id", ""),
                        "workspace_id": selected_mural.get("workspace_id", ""),
                        "workspace_name": selected_mural.get("workspace_name", ""),
                    }

                    self.console.print(
                        f"\n[green]Selected mural: {project['name']}[/green]"
                    )
                    return project
                else:
                    self.console.print(
                        "[yellow]Invalid selection. Please try again.[/yellow]"
                    )
            except ValueError:
                self.console.print("[yellow]Please enter a valid number.[/yellow]")
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Selection cancelled.[/yellow]")
                sys.exit(0)

    def export_to_file(self, content: str, filename: str, format: str = "md") -> bool:
        """Export content to a file.

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
                return False

            self.console.print("[green]Successfully retrieved mural content![/green]")

            # Extract text and image content
            content_items = self.mural_api.extract_mural_text(mural_content)

            if not content_items:
                self.console.print(
                    "[bold yellow]No text or image content found in the mural.[/bold yellow]"
                )
                return False

            # Store the content items
            self.mural_content_items = content_items

            # Create text versions for embedding
            self.mural_texts = []
            for item in content_items:
                if item["type"] == "text" and item.get("content"):
                    self.mural_texts.append(item["content"])
                elif item["type"] == "image" and item.get("analysis"):
                    self.mural_texts.append(item["analysis"])

            if not self.mural_texts:
                self.console.print(
                    "[bold yellow]No text content found for analysis.[/bold yellow]"
                )
                return False

            # Generate embeddings for the text content
            self.console.print(
                "[cyan]Generating embeddings for mural content...[/cyan]"
            )
            self.mural_embeddings = self.openai_api.get_embeddings(self.mural_texts)

            self.console.print(
                f"[green]Successfully processed {len(self.mural_texts)} content items![/green]"
            )
            return True

        except Exception as e:
            self.console.print(
                f"[bold red]Error fetching mural content: {str(e)}[/bold red]"
            )
            return False

    def generate_business_plan_section(self, section_title: str) -> Dict[str, Any]:
        """Generate a business plan section using retrieved content and OpenAI's GPT.

        Returns:
            Dict containing the generated section text and sources
        """
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

            # Early check before proceeding
            if not self.mural_texts:
                self.console.print(
                    "[bold yellow]No text content found for analysis.[/bold yellow]"
                )
                return None

            # Perform semantic search to find relevant content
            results = SemanticSearch.search(
                query_embedding, self.mural_embeddings, self.mural_texts, top_k=5
            )

            if not results:
                self.console.print(
                    "[bold yellow]No relevant content found for this section.[/bold yellow]"
                )
                return None

            # Create context from the search results
            context_items = []
            sources = []

            for result_text, score, idx in results:
                # For each result, find the corresponding original item
                for i, text in enumerate(self.mural_texts):
                    if text == result_text and i < len(text_items):
                        item = text_items[i]

                        # Add the content to context
                        context_items.append(result_text)

                        # Format source information
                        source_info = {
                            "text": (
                                result_text[:100] + "..."
                                if len(result_text) > 100
                                else result_text
                            ),
                            "score": f"{score:.2f}",
                            "id": item.get("widget_id", "Unknown ID"),
                            "type": item.get("widget_type", "Unknown type"),
                        }
                        sources.append(source_info)
                        break

            # Combine content for context
            context = "\n\n".join(context_items)

            # Format sources for OpenAI
            source_strings = [
                f"Source {i+1} ({s['score']}): {s['text']}"
                for i, s in enumerate(sources)
            ]

            # Generate the section using OpenAI
            generated_text = self.openai_api.generate_business_plan_section(
                section_title, context, source_strings
            )

            # Return both the generated text and sources
            return {
                "text": generated_text,
                "sources": sources,
            }

        except Exception as e:
            self.console.print(
                f"[bold red]Error generating section: {str(e)}[/bold red]"
            )
            return None

    def debug_clustering(self):
        """Debug clustering of mural content."""
        self.console.print(
            Panel(
                "Clustering mural content using graph-based approach",
                title="[bold cyan]Graph Clustering Debug[/bold cyan]",
            )
        )

        # Check if mural content exists
        if not hasattr(self, "mural_content_items") or not self.mural_content_items:
            # Fetch the mural content if not already loaded
            if not self.selected_project:
                self.console.print(
                    "[bold red]No project selected. Please select a project first.[/bold red]"
                )
                return

            fetch_result = self.fetch_mural_content(
                self.selected_project["mural_id"], self.selected_project["workspace_id"]
            )

            if not fetch_result:
                self.console.print(
                    "[bold red]Failed to fetch mural content. Cannot proceed with clustering.[/bold red]"
                )
                return

        # Allow user to customize clustering parameters
        self.console.print("\n[bold cyan]Graph-based Clustering Options[/bold cyan]")
        self.console.print(
            "Default values shown in brackets. Press Enter to use defaults."
        )

        try:
            # Get clustering parameters from user
            distance_factor_str = input("Distance factor [0.6]: ").strip()
            distance_factor = float(distance_factor_str) if distance_factor_str else 0.6

            color_factor_str = input("Color factor [0.3]: ").strip()
            color_factor = float(color_factor_str) if color_factor_str else 0.3

            edge_threshold_str = input("Edge threshold [0.5]: ").strip()
            edge_threshold = float(edge_threshold_str) if edge_threshold_str else 0.5

            algorithm_options = ["louvain", "label_propagation", "greedy"]
            algorithm = (
                input(f"Community detection algorithm {algorithm_options} [louvain]: ")
                .strip()
                .lower()
            )
            if not algorithm or algorithm not in algorithm_options:
                algorithm = "louvain"

            self.console.print(
                "\n[cyan]Running graph-based clustering with parameters:[/cyan]"
            )
            self.console.print(f"Distance factor: {distance_factor}")
            self.console.print(f"Color factor: {color_factor}")
            self.console.print(f"Edge threshold: {edge_threshold}")
            self.console.print(f"Algorithm: {algorithm}")

            # Run the clustering
            clustered_items = self.mural_api.cluster_widgets_by_graph(
                self.mural_content_items,
                distance_factor=distance_factor,
                color_factor=color_factor,
                edge_threshold=edge_threshold,
                community_detection=algorithm,
            )

            # Show results
            if clustered_items:
                self.console.print(
                    f"\n[green]Generated {len(clustered_items)} clusters[/green]"
                )

                # Display some statistics
                cluster_sizes = [
                    item.get("cluster_size", 0) for item in clustered_items
                ]

                if cluster_sizes:
                    avg_size = sum(cluster_sizes) / len(cluster_sizes)
                    min_size = min(cluster_sizes)
                    max_size = max(cluster_sizes)

                    self.console.print(
                        f"[cyan]Average cluster size: {avg_size:.1f} widgets[/cyan]"
                    )
                    self.console.print(
                        f"[cyan]Smallest cluster: {min_size} widgets[/cyan]"
                    )
                    self.console.print(
                        f"[cyan]Largest cluster: {max_size} widgets[/cyan]"
                    )

                # Show sample clusters
                self.console.print("\n[bold cyan]Sample Clusters:[/bold cyan]")
                for i, cluster in enumerate(
                    clustered_items[:3]
                ):  # Show first 3 clusters
                    content = cluster.get("content", "")
                    preview = content[:200] + "..." if len(content) > 200 else content
                    cluster_size = cluster.get("cluster_size", 0)

                    self.console.print(
                        f"\n[bold]Cluster {i+1} ({cluster_size} widgets):[/bold]"
                    )
                    self.console.print(f"[green]{preview}[/green]")

                # Ask if user wants to use these clustered items
                use_clusters = (
                    input("\nUse these clustered items for analysis? (y/n): ").lower()
                    == "y"
                )
                if use_clusters:
                    self.mural_content_items = clustered_items

                    # Update text items for embeddings
                    self.mural_texts = []
                    for item in clustered_items:
                        if item["type"] == "text" and item.get("content"):
                            self.mural_texts.append(item["content"])

                    # Generate new embeddings
                    if self.mural_texts:
                        self.console.print(
                            "[cyan]Generating embeddings for clustered content...[/cyan]"
                        )
                        self.mural_embeddings = self.openai_api.get_embeddings(
                            self.mural_texts
                        )
                        self.console.print(
                            f"[green]Successfully generated {len(self.mural_embeddings)} embeddings for clustered content![/green]"
                        )
            else:
                self.console.print("[yellow]No clusters were generated.[/yellow]")

        except ValueError as e:
            self.console.print(f"[bold red]Invalid input: {str(e)}[/bold red]")
        except Exception as e:
            self.console.print(
                f"[bold red]Error during clustering: {str(e)}[/bold red]"
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

        # Ask if the user wants to use graph-based clustering
        show_clustering_menu = (
            input("Do you want to test graph-based clustering? (y/n): ").lower() == "y"
        )
        if show_clustering_menu:
            self.debug_clustering()

        # Fetch Mural content and generate embeddings if not already done in clustering
        if not hasattr(self, "mural_content_items") or not self.mural_content_items:
            fetch_result = self.fetch_mural_content(
                self.selected_project["mural_id"], self.selected_project["workspace_id"]
            )

            if not fetch_result:
                self.console.print(
                    "[bold red]Failed to fetch or process mural content. Exiting.[/bold red]"
                )
                return

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
            result = self.generate_business_plan_section(section_title)

            if result:
                generated_text = result["text"]
                sources = result["sources"]

                # Display the generated section
                self.console.print("\n" + "-" * 40)
                self.console.print(generated_text)
                self.console.print("-" * 40)

                # Display sources
                if sources and input("\nShow sources? (y/n): ").lower() == "y":
                    self.console.print("\n[bold cyan]Sources:[/bold cyan]")
                    for i, source in enumerate(sources):
                        self.console.print(
                            f"[green]{i+1}. Score: {source['score']} | ID: {source['id']} | Type: {source['type']}[/green]"
                        )
                        self.console.print(f"   {source['text']}")
                        self.console.print("")

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
                    self.export_to_file(generated_text, filename, format)

                    # Export sources if requested
                    if sources and input("Export sources too? (y/n): ").lower() == "y":
                        sources_text = f"# Sources for {section_title}\n\n"
                        for i, source in enumerate(sources):
                            sources_text += f"## Source {i+1}\n"
                            sources_text += f"- Score: {source['score']}\n"
                            sources_text += f"- ID: {source['id']}\n"
                            sources_text += f"- Type: {source['type']}\n"
                            sources_text += f"- Content: {source['text']}\n\n"

                        sources_filename = f"{filename}_sources"
                        self.export_to_file(sources_text, sources_filename, format)

        self.console.print(
            "\n[bold green]Thank you for using Business Plan Drafter![/bold green]"
        )
