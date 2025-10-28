"""Data scrapers module for automated data collection."""

from .registry import ScraperRegistry
from .base import BaseScraper
from .bcb_extended import BCBExtendedScraper
from .yfinance_robust import YFinanceRobustScraper
from .inmet_climate import INMETClimateScraper
from .nasa_power import NASAPowerScraper
from .fundamentals_suzano import SuzanoFundamentalsScraper
from .ibge_macro import IBGEMacroScraper

__all__ = [
    "ScraperRegistry",
    "BaseScraper",
    "BCBExtendedScraper",
    "YFinanceRobustScraper",
    "INMETClimateScraper",
    "NASAPowerScraper",
    "SuzanoFundamentalsScraper",
    "IBGEMacroScraper",
]

# Initialize global registry
registry = ScraperRegistry()

# Register all scrapers
registry.register("bcb_extended", BCBExtendedScraper)
registry.register("yfinance", YFinanceRobustScraper)
registry.register("inmet", INMETClimateScraper)
registry.register("nasa_power", NASAPowerScraper)
registry.register("fundamentals", SuzanoFundamentalsScraper)
registry.register("ibge_macro", IBGEMacroScraper)

