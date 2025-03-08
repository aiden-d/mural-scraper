#!/usr/bin/env python3
"""
Mural API client for interacting with the Mural API.
"""

import sys
import requests
import re
from typing import Dict, List, Any, Optional, Tuple
from rich.console import Console
import networkx as nx
from collections import defaultdict
import colorsys
import numpy as np

from mural_api.oauth import MuralOAuth, MURAL_API_BASE_URL

# Initialize console for rich text output
console = Console()


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

                # Process the response
                widgets_response.raise_for_status()
                page_data = widgets_response.json()

                # Add the widgets from this page
                if "value" in page_data and page_data["value"]:
                    all_widgets["value"].extend(page_data["value"])
                    console.print(
                        f"[green]Added {len(page_data['value'])} widgets from page {page_count}[/green]"
                    )

                # Check if there's another page
                if "next" in page_data and page_data["next"]:
                    next_token = page_data["next"]
                else:
                    has_more_pages = False

            return all_widgets

        except Exception as e:
            console.print(
                f"[bold red]Error fetching mural content: {str(e)}[/bold red]"
            )
            raise

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

        # Track how many widgets had text or image content
        text_count = 0
        image_count = 0

        # Process each widget
        for widget in widgets:
            try:
                # Check what type of widget this is
                widget_type = widget.get("type", "unknown")

                # Text widgets
                if widget_type in ["text", "sticky"]:
                    text = widget.get("text", "")
                    if not text and "content" in widget:
                        text = widget.get("content", "")

                    if text:
                        # Clean up text (remove HTML tags and trim whitespace)
                        clean_text = self._clean_text(text)
                        if clean_text:
                            content_items.append(
                                {
                                    "type": "text",
                                    "content": clean_text,
                                    "widget_id": widget.get("id", "unknown"),
                                    "widget_type": widget_type,
                                    "x": widget.get("x", 0),
                                    "y": widget.get("y", 0),
                                    "width": widget.get("width", 0),
                                    "height": widget.get("height", 0),
                                    "color": widget.get("style", {}).get(
                                        "backgroundColor", None
                                    ),
                                }
                            )
                            text_count += 1

                # Image widgets
                elif widget_type == "image":
                    image_url = widget.get("url", "")
                    if image_url:
                        content_items.append(
                            {
                                "type": "image",
                                "url": image_url,
                                "widget_id": widget.get("id", "unknown"),
                                "widget_type": widget_type,
                                "x": widget.get("x", 0),
                                "y": widget.get("y", 0),
                                "width": widget.get("width", 0),
                                "height": widget.get("height", 0),
                            }
                        )
                        image_count += 1

                # Shape widgets with text
                elif widget_type == "shape" and widget.get("text"):
                    clean_text = self._clean_text(widget.get("text", ""))
                    if clean_text:
                        content_items.append(
                            {
                                "type": "text",
                                "content": clean_text,
                                "widget_id": widget.get("id", "unknown"),
                                "widget_type": widget_type,
                                "x": widget.get("x", 0),
                                "y": widget.get("y", 0),
                                "width": widget.get("width", 0),
                                "height": widget.get("height", 0),
                                "color": widget.get("style", {}).get(
                                    "backgroundColor", None
                                ),
                            }
                        )
                        text_count += 1

            except Exception as e:
                console.print(f"[yellow]Error processing widget: {str(e)}[/yellow]")
                continue

        console.print(
            f"[green]Extracted {text_count} text items and {image_count} images[/green]"
        )
        return content_items

    def _clean_text(self, text: str) -> str:
        """Clean up text from Mural widgets."""
        # Simple HTML tag removal
        text = re.sub(r"<[^>]*>", " ", text)

        # Remove excess whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

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

        # Create a copy of content items for later reference
        content_items_cache = content_items.copy()

        # Filter content items to only include those with valid position data
        valid_items = []
        for item in content_items:
            if "x" in item and "y" in item:
                valid_items.append(item)
            elif "position" in item:
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

            # Get position data, handling different item structures
            if "x" in item1 and "y" in item1:
                x1, y1 = item1["x"], item1["y"]
                width1 = item1.get("width", 100)  # Default size if not available
                height1 = item1.get("height", 100)
            else:
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

                # Get position data, handling different item structures
                if "x" in item2 and "y" in item2:
                    x2, y2 = item2["x"], item2["y"]
                    width2 = item2.get("width", 100)
                    height2 = item2.get("height", 100)
                else:
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
            if "position" in cluster[0]:
                cluster.sort(
                    key=lambda item: (item["position"]["y"], item["position"]["x"])
                )
            else:
                cluster.sort(key=lambda item: (item["y"], item["x"]))

            # Combine text from all items in the cluster
            combined_content = []
            widget_ids = []
            widget_types = set()

            for item in cluster:
                content = item.get("content", "")
                if content:
                    combined_content.append(content)
                widget_ids.append(item.get("id", item.get("widget_id", "unknown")))
                widget_types.add(item.get("widget_type", "unknown"))

            # Only create a cluster item if it has content
            if combined_content:
                # Calculate the centroid of the cluster
                if "position" in cluster[0]:
                    centroid_x = sum(item["position"]["x"] for item in cluster) / len(
                        cluster
                    )
                    centroid_y = sum(item["position"]["y"] for item in cluster) / len(
                        cluster
                    )
                else:
                    centroid_x = sum(item["x"] for item in cluster) / len(cluster)
                    centroid_y = sum(item["y"] for item in cluster) / len(cluster)

                cluster_item = {
                    "content": "\n\n".join(combined_content),
                    "type": "text",
                    "id": f"graph-cluster-{i}",
                    "widget_id": f"graph-cluster-{i}",  # For compatibility with both id fields
                    "widget_type": "graph_cluster",
                    "widget_ids": widget_ids,
                    "widget_types": list(widget_types),
                    "cluster_size": len(cluster),
                    "x": centroid_x,
                    "y": centroid_y,
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
                clustered_items = self._split_large_clusters(
                    clustered_items, content_items_cache
                )
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
        if "position" in items[0]:
            min_x = min(item["position"]["x"] for item in items)
            max_x = max(item["position"]["x"] for item in items)
            min_y = min(item["position"]["y"] for item in items)
            max_y = max(item["position"]["y"] for item in items)
        else:
            min_x = min(item["x"] for item in items)
            max_x = max(item["x"] for item in items)
            min_y = min(item["y"] for item in items)
            max_y = max(item["y"] for item in items)

        # Determine if we should split horizontally or vertically
        width = max_x - min_x
        height = max_y - min_y

        # We'll divide the space into a grid based on target clusters
        # Let's determine an appropriate grid size
        grid_size = max(2, int(np.sqrt(target_clusters)))

        # Create partitions based on spatial coordinates
        partitions = defaultdict(set)

        for i, item in enumerate(items):
            if "position" in item:
                x = item["position"]["x"]
                y = item["position"]["y"]
            else:
                x = item["x"]
                y = item["y"]

            # Calculate grid position
            grid_x = min(grid_size - 1, int((x - min_x) / width * grid_size))
            grid_y = min(grid_size - 1, int((y - min_y) / height * grid_size))

            # Assign to a partition based on grid cell
            partition_id = grid_y * grid_size + grid_x
            partitions[partition_id].add(i)

        # Return list of partitions, skipping empty ones
        return [partition for partition in partitions.values() if partition]

    def _split_large_clusters(
        self,
        clustered_items: List[Dict[str, Any]],
        original_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Split large clusters into smaller ones based on spatial proximity."""
        result = []

        for cluster in clustered_items:
            if len(cluster["content"]) <= 20000:
                result.append(cluster)
                continue

            # Get the original widget IDs
            widget_ids = cluster.get("widget_ids", [])
            if len(widget_ids) <= 1:
                # Can't split a single widget
                result.append(cluster)
                continue

            # Look up all the widgets by ID
            widgets = []
            for item_id in widget_ids:
                for original_item in original_items:
                    if (
                        original_item.get("id", original_item.get("widget_id", ""))
                        == item_id
                    ):
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
                    sub_widget_ids.append(
                        item.get("id", item.get("widget_id", "unknown"))
                    )
                    sub_widget_types.add(item.get("widget_type", "unknown"))

                if sub_combined_content:
                    # Calculate centroid
                    if "position" in sub_widgets[0]:
                        centroid_x = sum(
                            item["position"]["x"] for item in sub_widgets
                        ) / len(sub_widgets)
                        centroid_y = sum(
                            item["position"]["y"] for item in sub_widgets
                        ) / len(sub_widgets)
                    else:
                        centroid_x = sum(item["x"] for item in sub_widgets) / len(
                            sub_widgets
                        )
                        centroid_y = sum(item["y"] for item in sub_widgets) / len(
                            sub_widgets
                        )

                    sub_cluster = {
                        "content": "\n\n".join(sub_combined_content),
                        "type": "text",
                        "id": f"{cluster['id']}-split-{i}",
                        "widget_id": f"{cluster['id']}-split-{i}",
                        "widget_type": "split_graph_cluster",
                        "widget_ids": sub_widget_ids,
                        "widget_types": list(sub_widget_types),
                        "cluster_size": len(sub_widgets),
                        "x": centroid_x,
                        "y": centroid_y,
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
