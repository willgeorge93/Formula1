"""Data clients for external APIs and web scraping."""

from f1_predictor.data.clients.ergast import ErgastClient
from f1_predictor.data.clients.scrapers import WikipediaScraper, F1FansiteScraper

__all__ = ["ErgastClient", "WikipediaScraper", "F1FansiteScraper"]
