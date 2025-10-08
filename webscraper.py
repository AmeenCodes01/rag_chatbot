import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import validators
import html2text
from tqdm import tqdm
import os
import json
from typing import Set, Dict, List, Optional, Union, Any
import logging
import time
import asyncio
import aiohttp
from fake_useragent import UserAgent
from tenacity import retry, stop_after_attempt, wait_exponential
from urllib import robotparser
from datetime import datetime
from pathlib import Path
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

class RagScraper:
    """
    RagScraper: A web scraper optimized for RAG (Retrieval-Augmented Generation) applications.
    
    Features:
    - Asynchronous processing
    - Rate limiting
    - Robots.txt compliance
    - Content cleaning for RAG/LLM use
    - Structured output with metadata
    - Error handling and retries
    """
    def __init__(self, base_url: str, output_dir: str = "data/scraped_content", 
                 rate_limit: float = 1.0, max_retries: int = 3,
                 respect_robots: bool = True, max_workers: int = 5):
        """
        Initialize the WebScraper with enhanced configuration.
        
        Args:
            base_url (str): The base URL to scrape
            output_dir (str): Directory to save scraped content (default: data/scraped_content)
            rate_limit (float): Minimum time between requests in seconds
            max_retries (int): Maximum number of retry attempts for failed requests
            respect_robots (bool): Whether to respect robots.txt rules
            max_workers (int): Maximum number of concurrent workers for async operations
        """
        if not validators.url(base_url):
            raise ValueError("Invalid URL provided")
            
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        
        # Create a domain-specific output directory
        domain_dir = self.domain.replace(":", "_").replace("/", "_")
        self.output_dir = os.path.join(output_dir, domain_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging directory
        self.log_dir = os.path.join("data", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.visited_urls: Set[str] = set()
        self.last_request_time = 0
        
        # Initialize HTML converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.body_width = 0  # No wrapping
        
        # Setup logging
        self._setup_logging()
        
        # Initialize robots.txt parser
        self.respect_robots = respect_robots
        if respect_robots:
            self._setup_robots_parser()
        
        # Initialize user agent rotator
        self.ua = UserAgent()
        
        # Initialize async session
        self.session = None
        
    def _setup_logging(self):
        """Configure logging with detailed formatting"""
        # Create a logger for this instance
        self.logger = logging.getLogger('webscraper')
        self.logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            # File handler
            log_file = os.path.join(self.log_dir, f'scraper_{int(time.time())}.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

    def _setup_robots_parser(self):
        """Initialize and fetch robots.txt rules"""
        self.rp = robotparser.RobotFileParser()
        robots_url = urljoin(self.base_url, '/robots.txt')
        try:
            self.rp.set_url(robots_url)
            self.rp.read()
            self.logger.info(f"Successfully parsed robots.txt from {robots_url}")
        except Exception as e:
            self.logger.warning(f"Could not fetch robots.txt: {str(e)}")
            self.rp = None

    def _can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt"""
        if not self.respect_robots or self.rp is None:
            return True
        return self.rp.can_fetch("*", url)

    def _rate_limit_delay(self):
        """Implement rate limiting between requests"""
        if self.rate_limit > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch page content with retry logic and rate limiting.
        """
        if not self._can_fetch(url):
            self.logger.warning(f"URL not allowed by robots.txt: {url}")
            return None

        self._rate_limit_delay()
        
        headers = {'User-Agent': self.ua.random}
        try:
            if self.session is None:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url, headers=headers, timeout=30) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    self.logger.error(f"Failed to fetch {url}: Status {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            raise

    def _get_soup(self, html: str) -> Optional[BeautifulSoup]:
        """Parse HTML content into BeautifulSoup object"""
        try:
            return BeautifulSoup(html, 'html5lib')
        except Exception as e:
            self.logger.error(f"Error parsing HTML: {str(e)}")
            return None

    def _extract_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Union[str, Dict]]:
        """
        Extract and clean content from the page with enhanced metadata.
        """
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
            element.decompose()

        content = {
            'url': url,
            'title': soup.title.string.strip() if soup.title else '',
            'main_content': '',
            'metadata': {},
            'timestamp': datetime.now().isoformat(),
            'links': [],
            'headers': []
        }

        # Extract meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name', meta.get('property', ''))
            if name:
                content['metadata'][name] = meta.get('content', '')

        # Extract headers for structure
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            content['headers'].append({
                'level': int(header.name[1]),
                'text': header.get_text(strip=True)
            })

        # Extract main content
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        if main_content:
            content['main_content'] = self.html_converter.handle(str(main_content))

        # Extract links
        content['links'] = self._get_links(soup, url)

        return content

    def _get_links(self, soup: BeautifulSoup, current_url: str) -> List[Dict[str, str]]:
        """Extract and classify links from the page"""
        links = []
        for link in soup.find_all('a', href=True):
            url = urljoin(current_url, link['href'])
            parsed_url = urlparse(url)
            if parsed_url.netloc == self.domain and '#' not in url:
                links.append({
                    'url': url,
                    'text': link.get_text(strip=True),
                    'title': link.get('title', ''),
                    'type': 'internal'
                })
            else:
                links.append({
                    'url': url,
                    'text': link.get_text(strip=True),
                    'title': link.get('title', ''),
                    'type': 'external'
                })
        return links

    def _generate_filename(self, url: str) -> str:
        """Generate a unique filename for a URL using SHA-256"""
        return hashlib.sha256(url.encode()).hexdigest()[:16] + '.json'

    def _clean_content_for_rag(self, content: str) -> str:
        """
        Clean content specifically for RAG/LLM training purposes.
        Removes unwanted elements and normalizes the text.
        """
        # Remove special characters and normalize whitespace
        cleaned = re.sub(r'\s+', ' ', content)
        
        # Remove common unwanted elements
        patterns_to_remove = [
            r'¶',  # Remove paragraph marks
            r'\[.*?\]',  # Remove square bracket content
            r'©.*?(?=\n|$)',  # Remove copyright notices
            r'Cookie.*?(?=\n|$)',  # Remove cookie notices
            r'Privacy.*?(?=\n|$)',  # Remove privacy notices
            r'\d+\s*min read',  # Remove read time estimates
            r'Last updated:.*?(?=\n|$)',  # Remove update timestamps
            r'Share on.*?(?=\n|$)',  # Remove share buttons text
            r'Follow us on.*?(?=\n|$)',  # Remove social media text
        ]
        
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned)
        
        # Remove multiple newlines and spaces
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        cleaned = re.sub(r' +', ' ', cleaned)
        
        # Remove leading/trailing whitespace from each line
        cleaned = '\n'.join(line.strip() for line in cleaned.split('\n'))
        
        return cleaned.strip()

    async def scrape_page(self, url: str) -> Dict[str, Any]:
        """
        Scrape a single page asynchronously.
        """
        self.logger.info(f"Scraping page: {url}")
        
        html = await self._fetch_page(url)
        if not html:
            return {}
        
        soup = self._get_soup(html)
        if not soup:
            return {}
        
        content = self._extract_content(soup, url)
        rag_content = self._clean_content_for_rag(content['main_content'])
        content['rag_content'] = rag_content
        self._save_content(url, content)
        return content

    async def scrape_website(self, max_pages: Optional[int] = None) -> None:
        """
        Scrape the entire website asynchronously with concurrent workers.
        """
        try:
            queue = [self.base_url]
            scraped_content = {}
            
            with tqdm(total=max_pages or float('inf'), desc="Scraping pages") as pbar:
                while queue and (max_pages is None or len(self.visited_urls) < max_pages):
                    # Process URLs in batches
                    batch = queue[:self.max_workers]
                    queue = queue[self.max_workers:]
                    
                    tasks = []
                    for url in batch:
                        if url not in self.visited_urls:
                            self.visited_urls.add(url)
                            tasks.append(self.scrape_page(url))
                    
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        for url, result in zip(batch, results):
                            if isinstance(result, Exception):
                                self.logger.error(f"Error processing {url}: {str(result)}")
                                continue
                            
                            if result:
                                scraped_content[url] = result
                                # Add new URLs to queue
                                new_urls = [link['url'] for link in result.get('links', [])
                                          if link['type'] == 'internal' and 
                                          link['url'] not in self.visited_urls]
                                queue.extend(new_urls)
                            
                            pbar.update(1)
            
            # Save index of all scraped pages
            self._save_index(scraped_content)
        
        finally:
            # Always close the session
            if self.session:
                await self.session.close()
                self.session = None

    def _save_content(self, url: str, content: Dict[str, str]) -> None:
        """Save scraped content to a file with error handling"""
        try:
            filename = self._generate_filename(url)
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving content for {url}: {str(e)}")

    def _save_index(self, content: Dict[str, Dict[str, str]]) -> None:
        """Save enhanced index of all scraped pages"""
        index = {
            'base_url': self.base_url,
            'total_pages': len(content),
            'scrape_timestamp': datetime.now().isoformat(),
            'pages': [{
                'url': url,
                'title': page_content.get('title', ''),
                'filename': self._generate_filename(url),
                'headers': page_content.get('headers', []),
                'metadata': page_content.get('metadata', {})
            } for url, page_content in content.items()]
        }
        
        with open(os.path.join(self.output_dir, 'index.json'), 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def get_statistics(self) -> Dict[str, Union[int, List[str]]]:
        """Get scraping statistics"""
        return {
            'total_pages': len(self.visited_urls),
            'failed_urls': [url for url in self.visited_urls 
                          if not os.path.exists(os.path.join(
                              self.output_dir, 
                              self._generate_filename(url)))],
            'total_content_size': sum(
                os.path.getsize(os.path.join(self.output_dir, f))
                for f in os.listdir(self.output_dir)
                if f.endswith('.json')
            )
        }
