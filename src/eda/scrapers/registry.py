"""Scraper registry for managing available scrapers."""

from typing import Dict, Type, Optional
from .base import BaseScraper


class ScraperRegistry:
    """Registry for managing data scrapers."""
    
    def __init__(self):
        """Initialize empty registry."""
        self._scrapers: Dict[str, Type[BaseScraper]] = {}
        self._instances: Dict[str, BaseScraper] = {}
    
    def register(self, name: str, scraper_class: Type[BaseScraper]):
        """
        Register a scraper class.
        
        Parameters
        ----------
        name : str
            Scraper name
        scraper_class : Type[BaseScraper]
            Scraper class (not instance)
        """
        if not issubclass(scraper_class, BaseScraper):
            raise TypeError(f"{scraper_class} must inherit from BaseScraper")
        
        self._scrapers[name] = scraper_class
        print(f"[REGISTRY] Registered scraper: {name}")
    
    def get(self, name: str, **kwargs) -> BaseScraper:
        """
        Get scraper instance.
        
        Parameters
        ----------
        name : str
            Scraper name
        **kwargs
            Arguments for scraper initialization
        
        Returns
        -------
        BaseScraper
            Scraper instance
        """
        if name not in self._scrapers:
            available = ", ".join(self._scrapers.keys())
            raise ValueError(f"Scraper '{name}' not found. Available: {available}")
        
        # Create instance if not cached
        if name not in self._instances:
            self._instances[name] = self._scrapers[name](**kwargs)
        
        return self._instances[name]
    
    def list_scrapers(self) -> list:
        """Get list of registered scraper names."""
        return list(self._scrapers.keys())
    
    def clear_instances(self):
        """Clear all cached scraper instances."""
        self._instances.clear()
    
    def get_all_stats(self) -> dict:
        """Get statistics for all active scrapers."""
        return {
            name: scraper.get_stats()
            for name, scraper in self._instances.items()
        }

