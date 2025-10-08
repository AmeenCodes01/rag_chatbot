"""
Simple demo script showing basic usage of RagScraper.
"""

import asyncio
from webscraper import RagScraper
import json
from pprint import pprint

async def main():
    url = "https://langchain-ai.github.io/langgraph/"
    
    # Initialize the scraper with custom settings
    scraper = RagScraper(
        base_url=url,
        output_dir="data/scraped_content",
        rate_limit=1.0,  # 1 second between requests
        max_retries=3,
        respect_robots=True,
        max_workers=5
    )
    
    # Example 1: Scrape a single page
    print("Example 1: Scraping a single page...")
    content = await scraper.scrape_page(url)
    if content:
        print("\nContent extracted successfully!")
        print(f"Title: {content.get('title', 'N/A')}")
        print(f"Content Length: {len(content.get('main_content', ''))}")
        print(f"Number of Links: {len(content.get('links', []))}")
    
    # Example 2: Scrape multiple pages (limited to 5 pages)
    print("\nExample 2: Scraping multiple pages (limited to 5)...")
    scraper = RagScraper(
        base_url=url,
        output_dir="data/scraped_content",
        rate_limit=1.0,
        max_retries=3,
        respect_robots=True
    )
    await scraper.scrape_website(max_pages=5)

if __name__ == "__main__":
    asyncio.run(main())
