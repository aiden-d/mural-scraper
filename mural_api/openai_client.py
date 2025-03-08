#!/usr/bin/env python3
"""
OpenAI API client for interacting with OpenAI services.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any
import openai
from rich.console import Console
from rich.panel import Panel
from tenacity import retry, stop_after_attempt, wait_exponential

# Initialize console for rich text output
console = Console()


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

        # If not in cache or force reanalysis, call the API
        console.print(f"[cyan]Analyzing image: {image_url}[/cyan]")
        self.cache_stats["misses"] += 1

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that analyzes images from Mural boards. "
                        "Describe the content of the image in detail, focusing on any text "
                        "visible in the image, diagrams, charts, and visual elements. "
                        "If there is text, make sure to preserve it exactly as shown. "
                        "For diagrams or charts, describe their structure and meaning. "
                        "Be thorough but concise.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe the content of this image from a Mural board in detail.",
                            },
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                max_tokens=1000,
            )

            analysis = response.choices[0].message.content

            # Cache the results
            self.image_cache[image_url] = {
                "analysis": analysis,
                "timestamp": time.time(),
                "model": "gpt-4o",
            }
            self._save_image_cache()

            return analysis

        except Exception as e:
            console.print(f"[bold red]Error analyzing image: {str(e)}[/bold red]")
            return f"Error analyzing image: {str(e)}"

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_business_plan_section(
        self, section_title: str, context: str, sources: List[str] = None
    ) -> str:
        """Generate a business plan section based on context and sources.

        Args:
            section_title: The title of the section to generate
            context: The context for generation
            sources: List of sources to cite
        """
        if sources is None:
            sources = []

        sources_text = "\n".join([f"- {source}" for source in sources])

        prompt = f"""
        # Task: Generate a {section_title} section for a business plan

        ## Context Information:
        {context}

        ## Information Sources:
        {sources_text}

        ## Guidelines:
        - Write a comprehensive, well-structured {section_title} section for a business plan
        - Use a professional business tone
        - Be specific and detailed using the information provided
        - Follow standard business plan formatting for this section
        - Focus on actionable insights and clear explanations
        - Keep the length appropriate for a business plan section (300-600 words)
        - Do not include placeholder text or notes about missing information

        ## Output: 
        Please provide the complete {section_title} section text in markdown format.
        """

        response = self.client.chat.completions.create(
            model="gpt-4.5-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional business plan writer with expertise in creating clear, "
                    "comprehensive, and compelling business plans. Your writing is concise, "
                    "specific, and tailored to the business context.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
        )

        return response.choices[0].message.content
