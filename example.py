#!/usr/bin/env python3
"""
Example script demonstrating the usage of RagScraper.
Can be run from command line with various options.
"""

import asyncio
import argparse
import logging
from webscraper import RagScraper
from typing import Optional
import sys


def setup_logging():
    """Configure logging for the example script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='RagScraper: Web Scraping for RAG/LLM Applications'
    )
    parser.add_argument(
        'url',
        help='URL to scrape'
    )
    parser.add_argument(
        '--output-dir',
        default='data/scraped_content',
        help='Directory to save scraped content (default: data/scraped_content)'
    )
    parser.add_argument(
        '--single-page',
        action='store_true',
        help='Scrape only the provided page'
    )
    parser.add_argument(
        '--max-pages',
        type=int,
        help='Maximum number of pages to scrape'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=1.0,
        help='Minimum time between requests in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Maximum number of concurrent workers (default: 5)'
    )
    parser.add_argument(
        '--no-robots',
        action='store_true',
        help='Disable robots.txt compliance'
    )
    return parser.parse_args()


async def scrape_single_page(scraper: RagScraper, url: str) -> Optional[dict]:
    """Scrape a single page and display results."""
    logging.info(f"Scraping single page: {url}")
    content = await scraper.scrape_page(url)
    
    if content:
        logging.info("\nPage Details:")
        logging.info(f"Title: {content['title']}")
        logging.info(f"Content length: {len(content['main_content'])}")
        logging.info(f"RAG content length: {len(content['rag_content'])}")
        logging.info(f"Number of metadata items: {len(content['metadata'])}")
        logging.info(f"Number of internal links: {len([l for l in content['links'] if l['type'] == 'internal'])}")
        logging.info(f"Number of external links: {len([l for l in content['links'] if l['type'] == 'external'])}")
        return content
    else:
        logging.error("Failed to scrape page")
        return None


async def scrape_website(scraper: RagScraper, max_pages: Optional[int] = None):
    """Scrape multiple pages from the website."""
    logging.info(f"Scraping website (max pages: {max_pages or 'unlimited'})")
    await scraper.scrape_website(max_pages=max_pages)
    
    # Display statistics
    stats = scraper.get_statistics()
    logging.info("\nScraping Statistics:")
    logging.info(f"Total pages scraped: {stats['total_pages']}")
    logging.info(f"Failed URLs: {len(stats['failed_urls'])}")
    logging.info(f"Total content size: {stats['total_content_size'] / 1024:.2f} KB")


async def main():
    """Main entry point for the script."""
    args = parse_arguments()
    setup_logging()
    
    # Initialize scraper
    scraper = RagScraper(
        base_url=args.url,
        output_dir=args.output_dir,
        rate_limit=args.rate_limit,
        max_retries=3,
        respect_robots=not args.no_robots,
        max_workers=args.max_workers
    )
    
    try:
        if args.single_page:
            await scrape_single_page(scraper, args.url)
        else:
            await scrape_website(scraper, args.max_pages)
    except KeyboardInterrupt:
        logging.info("\nScraping interrupted by user")
    except Exception as e:
        logging.error(f"Error during scraping: {str(e)}")
    finally:
        # Ensure proper cleanup
        if scraper.session:
            await scraper.session.close()


if __name__ == "__main__":
    asyncio.run(main())
