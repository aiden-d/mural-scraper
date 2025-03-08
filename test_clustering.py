#!/usr/bin/env python3

import os
import json
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from business_plan_drafter import MuralOAuth, MuralAPI, OpenAIAPI

# Initialize console
console = Console()


def test_proximity_clustering():
    """Test the proximity-based clustering functionality on Mural content."""
    console.print(
        Panel.fit(
            "[bold cyan]Mural Widget Proximity Clustering Test[/bold cyan]\n\n"
            "This test demonstrates clustering widgets based on their spatial proximity.",
            title="Proximity Clustering Test",
            border_style="cyan",
        )
    )

    # Load credentials
    config_path = os.path.expanduser("~/.business_plan_drafter/config.json")
    if not os.path.exists(config_path):
        console.print(
            "[bold red]Config file not found. Please run business_plan_drafter.py first.[/bold red]"
        )
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # Check for required credentials
    if not config.get("mural_client_id") or not config.get("mural_client_secret"):
        console.print("[bold red]Mural API credentials not found in config.[/bold red]")
        return

    console.print("[cyan]Initializing Mural API...[/cyan]")

    # Initialize Mural API
    oauth = MuralOAuth(config["mural_client_id"], config["mural_client_secret"])
    mural_api = MuralAPI(oauth)

    # Get access token
    if not oauth.get_access_token():
        console.print("[bold red]Failed to get Mural access token.[/bold red]")
        return

    # Get workspaces
    console.print("[cyan]Fetching workspaces...[/cyan]")
    workspaces = mural_api.get_workspaces()

    if not workspaces:
        console.print("[yellow]No workspaces found.[/yellow]")
        return

    # Display workspaces
    console.print("\n[bold cyan]Available Workspaces:[/bold cyan]")
    for workspace in workspaces:
        console.print(f"ID: {workspace.get('id')} - Name: {workspace.get('name')}")

    # Select workspace
    workspace_id = console.input("\n[cyan]Enter workspace ID: [/cyan]")
    if not workspace_id:
        console.print("[yellow]No workspace selected. Exiting.[/yellow]")
        return

    # Get murals from selected workspace
    console.print(f"[cyan]Fetching murals for workspace {workspace_id}...[/cyan]")
    murals = mural_api.get_murals(workspace_id)

    if not murals:
        console.print("[yellow]No murals found in this workspace.[/yellow]")
        return

    # Display murals
    console.print("\n[bold cyan]Available Murals:[/bold cyan]")
    for mural in murals:
        console.print(f"ID: {mural.get('id')} - Title: {mural.get('title')}")

    # Select mural
    mural_id = console.input("\n[cyan]Enter mural ID: [/cyan]")
    if not mural_id:
        console.print("[yellow]No mural selected. Exiting.[/yellow]")
        return

    # Fetch mural content
    console.print(f"[cyan]Fetching content from mural {mural_id}...[/cyan]")
    mural_content = mural_api.get_mural_content(mural_id)

    if not mural_content:
        console.print("[bold red]Failed to retrieve mural content.[/bold red]")
        return

    console.print("[green]Successfully retrieved mural content![/green]")

    # Extract content items
    console.print("[cyan]Extracting content items...[/cyan]")
    content_items = mural_api.extract_mural_text(mural_content)

    if not content_items:
        console.print("[yellow]No content items found in the mural.[/yellow]")
        return

    console.print(f"[green]Found {len(content_items)} content items![/green]")

    # Count items with position data
    items_with_position = 0
    for item in content_items:
        if "position" in item and item["position"]:
            items_with_position += 1

    console.print(f"[cyan]{items_with_position} items have position data.[/cyan]")

    if items_with_position == 0:
        console.print(
            "[yellow]No items with position data found. Cannot perform clustering.[/yellow]"
        )
        return

    # Test clustering with different thresholds
    thresholds = [100, 200, 300, 500, 1000]

    console.print("\n[bold cyan]Testing Different Clustering Thresholds:[/bold cyan]")

    for threshold in thresholds:
        console.print(f"\n[cyan]Clustering with threshold = {threshold}...[/cyan]")

        clustered_items = mural_api.cluster_widgets_by_proximity(
            content_items, distance_threshold=threshold
        )

        # Calculate statistics
        if clustered_items:
            avg_size = sum(item["cluster_size"] for item in clustered_items) / len(
                clustered_items
            )

            # Count clusters by size
            size_counts = {}
            for item in clustered_items:
                size = item["cluster_size"]
                size_counts[size] = size_counts.get(size, 0) + 1

            console.print(f"[green]Created {len(clustered_items)} clusters[/green]")
            console.print(f"[cyan]Average cluster size: {avg_size:.2f} widgets[/cyan]")

            # Display cluster size distribution
            console.print("[cyan]Cluster size distribution:[/cyan]")
            for size, count in sorted(size_counts.items()):
                console.print(f"  - {size} widgets: {count} clusters")

            # Show sample clusters
            if len(clustered_items) > 0:
                console.print("\n[cyan]Sample cluster content:[/cyan]")
                sample = clustered_items[0]
                console.print(f"[bold]Cluster size: {sample['cluster_size']}[/bold]")
                console.print(f"Widget types: {', '.join(sample['widget_types'])}")
                console.print(
                    f"Content: {sample['content'][:200]}..."
                    if len(sample["content"]) > 200
                    else f"Content: {sample['content']}"
                )
        else:
            console.print("[yellow]No clusters created with this threshold.[/yellow]")

    # Ask for preferred threshold
    threshold_input = console.input(
        "\n[cyan]Enter your preferred distance threshold: [/cyan]"
    )

    try:
        preferred_threshold = float(threshold_input)

        console.print(
            f"[cyan]Clustering with threshold = {preferred_threshold}...[/cyan]"
        )

        final_clusters = mural_api.cluster_widgets_by_proximity(
            content_items, distance_threshold=preferred_threshold
        )

        if final_clusters:
            console.print(f"[green]Created {len(final_clusters)} clusters![/green]")

            # Show all clusters
            console.print("\n[bold cyan]All Clusters:[/bold cyan]")

            for i, cluster in enumerate(final_clusters):
                console.print(
                    f"\n[bold]Cluster {i+1} (size: {cluster['cluster_size']}):[/bold]"
                )
                console.print(f"Widget types: {', '.join(cluster['widget_types'])}")
                console.print(
                    f"Content: {cluster['content'][:100]}..."
                    if len(cluster["content"]) > 100
                    else f"Content: {cluster['content']}"
                )

            # Save clusters to file
            save_choice = console.input(
                "\n[yellow]Save clusters to file? (y/n): [/yellow]"
            ).lower()

            if save_choice == "y":
                output_file = (
                    f"mural_{mural_id}_clusters_{int(preferred_threshold)}.json"
                )

                with open(output_file, "w") as f:
                    json.dump(final_clusters, f, indent=2)

                console.print(f"[green]Saved clusters to {output_file}![/green]")
        else:
            console.print("[yellow]No clusters created with this threshold.[/yellow]")

    except ValueError:
        console.print("[bold red]Invalid threshold value.[/bold red]")


if __name__ == "__main__":
    test_proximity_clustering()
