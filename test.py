import os
import asyncio
import json
from pydantic import BaseModel, Field
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig, RateLimiter, DisplayMode, CrawlerMonitor,UndetectedAdapter

from crawl4ai import LLMExtractionStrategy
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
import re 

from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
class Product(BaseModel):
    name: str
    price: str

data = [
  {
    "index": 0,
    "tags": [
      "introduction"
    ],
    "content": [
      "Get Online Fast with PK Domains"
    ]
  },
  {
    "index": 1,
    "tags": [
      "navigation"
    ],
    "content": [
      "Domain",
      "Hosting",
      "Development",
      "Marketing",
      "Security",
      "My Account",
      "+92 322 5252352"
    ]
  },
  {
    "index": 2,
    "tags": [
      "pricing"
    ],
    "content": [
      "PKR Let’s Talk",
      ".com Rs2,899/yr",
      ".net Rs4,204/yr",
      ".pk Rs3,199/2yr",
      ".com.pk Rs3,199/2yr",
      ".ae Rs11,680/yr"
    ]
  },
  {
    "index": 3,
    "tags": [
      "web_hosting_promotion"
    ],
    "content": [
      "Fast Web Hosting",
      "Avail 47% OFF on Hosting with Free Domain!",
      "Get Started"
    ]
  },
  {
    "index": 4,
    "tags": [
      "web_development_promotion"
    ],
    "content": [
      "Stunning Websites, Tailored for You!",
      "Get Started"
    ]
  },
  {
    "index": 5,
    "tags": [
      "seo_services_promotion"
    ],
    "content": [
      "Skyrocket Rankings with Expert SEO!",
      "Explore Plan"
    ]
  },
  {
    "index": 6,
    "tags": [
      "services_overview"
    ],
    "content": [
      "Our Services"
    ]
  },
  {
    "index": 7,
    "tags": [
      "domain_registration"
    ],
    "content": [
      "Domain Registration",
      "Start Your Digital Journey - Secure Your .COM Domain for Just Rs2899/Year!",
      "Make Your Mark Online.",
      "Find more"
    ]
  },
  {
    "index": 8,
    "tags": [
      "web_hosting"
    ],
    "content": [
      "Web Hosting",
      "Host with Confidence - Get Reliable Web Hosting from Just Rs999/Year!",
      "Fast, Secure, and Always Online.",
      "Find more"
    ]
  },
  {
    "index": 9,
    "tags": [
      "vps_hosting"
    ],
    "content": [
      "VPS Hosting",
      "Power Your Project with VPS Hosting - Flexible, Scalable, and Secure Solutions for Every Need!",
      "Find more"
    ]
  },
  {
    "index": 10,
    "tags": [
      "ssl_certificates"
    ],
    "content": [
      "SSL Certificates",
      "Enhance Site Security with SSL Certificates - Safeguard Your Online Presence and Boost User Confidence!",
      "Find more"
    ]
  },
  {
    "index": 11,
    "tags": [
      "website_development"
    ],
    "content": [
      "Website Development",
      "Shape Your Online Presence - Discover Excellence in Website Development with Our Free Expert Consultation!",
      "Find more"
    ]
  },
  {
    "index": 12,
    "tags": [
      "mobile_apps"
    ],
    "content": [
      "Mobile Apps",
      "Your Gateway to Today's Digital World - Partner with Us for Bespoke Mobile Apps Solutions!",
      "Find more"
    ]
  },
  {
    "index": 13,
    "tags": [
      "seo_services"
    ],
    "content": [
      "SEO Services",
      "Rank High, Boost Sales - Elevate Your Business with Our Expert SEO Services for Real Results!",
      "Find more"
    ]
  },
  {
    "index": 14,
    "tags": [
      "google_ads"
    ],
    "content": [
      "Google Ads",
      "Boost Your Sales with Google Ads - Drive Targeted Traffic and Grow Your Business Effectively!",
      "Find more"
    ]
  },
  {
    "index": 15,
    "tags": [
      "digital_marketing"
    ],
    "content": [
      "360 Digital Marketing Solution",
      "Complete Digital Marketing for Your Brand - Get Seen Everywhere with Our All-in-One Solution!"
    ]
  },
  {
    "index": 0,
    "tags": [
      "services"
    ],
    "content": [
      "We are Pakistan’s leading Web Hosting Provider – trusting your site to us means a powerful online presence with an uptime that meets all your expectations, guaranteed."
    ]
  },
  {
    "index": 1,
    "tags": [
      "company_information"
    ],
    "content": [
      "Our Company",
      "* [About Us](https://websouls.com/about)",
      "* [Our Team](https://websouls.com/team)",
      "* [Why Us](https://websouls.com/whyus)",
      "* [Clients](https://websouls.com/clients)",
      "* [Portfolio](https://websouls.com/portfolio)"
    ]
  },
  {
    "index": 2,
    "tags": [
      "hosting_services"
    ],
    "content": [
      "Our Services",
      "* [Shared Hosting](https://websouls.com/shared-hosting)",
      "* [Business Hosting](https://websouls.com/web-hosting)",
      "* [Reseller Hosting](https://websouls.com/reseller-hosting)",
      "* [WordPress Hosting](https://websouls.com/wordpress-hosting-in-pakistan)",
      "* [Dedicated Server](https://websouls.com/dedicated-server)",
      "* [VPS Hosting](https://websouls.com/vps-hosting)",
      "* [Pak Based VPS](https://websouls.com/pk-vps)",
      "* [SSL Certificates](https://websouls.com/ssl-certificates)",
      "* [Domain Pricing](https://websouls.com/domain-registration)",
      "* [Domain Transfer](https://websouls.com/domain-transfer)",
      "* [PK Domains](https://websouls.com/buy-pk-domain)",
      "* [AE Domains](https://websouls.com/buy-ae-domains)"
    ]
  },
  {
    "index": 3,
    "tags": [
      "client_information"
    ],
    "content": [
      "Client Information",
      "* [Billing Area](https://billing.websouls.com/clientarea.php)",
      "* [Announcement](https://billing.websouls.com/announcements.php)",
      "* [Generate a Lead](https://billing.websouls.com/submitticket.php?step=2&deptid=2)",
      "* [Acceptable Use Policy](https://websouls.com/policy)",
      "* [Privacy Policy](https://websouls.com/privacy)",
      "* [Feedback](https://billing.websouls.com/submitticket.php?step=2&deptid=5)",
      "* [Sitemap](https://websouls.com/sitemap)"
    ]
  },
  {
    "index": 4,
    "tags": [
      "support"
    ],
    "content": [
      "Support Center",
      "* [Open A Ticket](https://billing.websouls.com//submitticket.php)",
      "* [Knowledgebase Articles](https://billing.websouls.com/knowledgebase.php)",
      "* [Network Status](https://billing.websouls.com/serverstatus.php)",
      "* [FAQ's](https://websouls.com/faq)",
      "* [Payment Method](https://websouls.com/payment-methods)",
      "* [Contact Us](https://websouls.com/contactus)"
    ]
  },
  {
    "index": 5,
    "tags": [
      "website_development"
    ],
    "content": [
      "Website Development",
      "* [Business Website](https://websouls.com/web-development)",
      "* [Ecommerce Solution](https://websouls.com/ecommerce-solution)",
      "* [SEO](https://websouls.com/seo-services)"
    ]
  },
  {
    "index": 0,
    "tags": [
      "customer_reviews"
    ],
    "content": [
      "Haq 5 out of 5 stars Excellent, Professional, And prompt. My recent problem was solved efficiently by support executive Mr. Osama at #Websouls Solutions. First response on webchat, and then helping me on anydesk to solve email problem. I am a proud customer of #Websouls Pakistan. :) - Rizwan Ashar",
      "5 out of 5 stars I wanted to take a moment to express my utmost satisfaction with the service I received from WEBSouls. As someone who recently purchased a domain, my experience has been nothing short of excellent. From the user-friendly interface to the prompt and efficient customer support, WEBSoul has truly exceeded my expectations...See More - Liaqat Ali",
      "5 out of 5 stars Will highly recommend it for your digital needs. I have been with them since last 20 years. Mr. Gulrez is very professional and responsive with his prompt services. - Umer Ghauri",
      "5 out of 5 stars Very reliable company. It's been more than a decade that I am doing business with them. I am very satisfied with their services. Their team members are cooperative and friendly, especially their team member Ms. Sajal is prompt in response and polite. I am thankful to her for keeping me reminding regarding my about to expire...See More - Naeem Shahid"
    ]
  },
  {
    "index": 1,
    "tags": [
      "services_overview"
    ],
    "content": [
      "Stay Ahead ## With Websouls At Websouls, we live with technology; we know it from the core. The journey starts from operating systems where everything is built on, then it comes to web hosting, website development, mobile app development and it ends on a brand building through digital marketing. We don’t leave our customers in the sky, if we have hosted you let us make you visible to the world with our unique approaches."
    ]
  },
  {
    "index": 2,
    "tags": [
      "domain_registration"
    ],
    "content": [
      "01. Domain Registration Every business starts with a domain registration. Think about the name of your business, book it now at Websouls in minutes. A short name which must be easy to remember, self-explanatory and reflects your vision is an ideal domain name that you should choose. Book it for multiple years, it gives a signal to search engines that how serious you are about your business."
    ]
  },
  {
    "index": 3,
    "tags": [
      "web_hosting"
    ],
    "content": [
      "02. Web Hosting Services A fast, secure, and reliable web hosting can make your business trustworthy to search engine crawlers and to your customers as well. Websouls offers SSD fast web hosting with proactive customer support. Years of experience have taught us what needs to be fixed, and we did it in advance for you, which means you are one step ahead of your competitors when you are hosted with us."
    ]
  },
  {
    "index": 4,
    "tags": [
      "website_development"
    ],
    "content": [
      "03. Website Development A unique and user-friendly digital presence of your business can help you build your own brand identity, which is an important factor in today’s digital space to establish your significant brand position in the market. Without this, you will be another shop in the street and will not be in the focus of your target audience. We do website development in Pakistan with this vision in mind."
    ]
  },
  {
    "index": 0,
    "tags": [
      "hosting_features"
    ],
    "content": [
      "250GB NVMe Storage * 200 Websites * Powerful AI Website Builder Included Powerful AI website builder allows you to create a professional website with an AI content generator. SEO friendly content will help you generate online business.",
      "* Free Domain (Rs2,799 value) One free .Com domain with purchase of a new 1 or more years hosting plan, will renew at the then-current renewal price",
      "* Local Telephonic Support We speak your language, we know you & your problem. Our experts will help you better",
      "* Unlimited Bandwidth",
      "* Unlimited Databases",
      "* Free Backup (Rs3,920/yr value)",
      "* Unlimited Email IDs You may create unlimited email IDs, each mail box comes with a 1GB space limit",
      "* Advanced & Professional Email Service Our advanced email solution ensures exceptional deliverability with features like Automatic DKIM/SPF, Feedback Loops, and Blacklist Monitoring. Enjoy robust security with TLS/SSL, Spamtrap De",
      "* Free SSL (Rs2,920/yr value) Label your website 'secure' with a Let's Encrypt SSL certificate",
      "* Malware Scanning (Rs3,470/yr value)",
      "* Free Website Migration Our tech experts will move your websites to our platform, free of cost with no worries",
      "* 7-Days Free Trial Available Money-back guarantee? You don't even pay us until you're satisfied. Try our 7-day free trial and enjoy!",
      "* Free Whois Privacy (Rs1,460/yr value) Hide your personal information from whois, free of cost",
      "* Free 1-Click Softwares Installation",
      "* Super Fast Resources See our knowledge-base articles for more details",
      "* cPanel with SSH Access",
      "* NodeJs Supported"
    ]
  },
  {
    "index": 1,
    "tags": [
      "company_background"
    ],
    "content": [
      "Why Choose Websouls?",
      "**web hosting industry**, we know what your needs are and that is how we were able to serve more than 80,000 websites.",
      "Serving Since 2002 A UK based web hosting company, we’ve been in the industry for more than two decades now and offer everything your website needs to run smoothly.",
      "cPanel Official Partner We’ve brought together technology and web hosting experts with years of experience, to provide a web hosting platform that’s complete with cPanel.",
      "1000+ Corporate Clients With our customer-centric approach, we pride in making a difference for over 1000 web hosting corporate and govt organizations whom we continue to serve.",
      "Bash Scripts The security of our customers is our top priority and we use many custom scripts to accomplish the challenging tasks. Our clients never have to fear cyber-attacks.",
      "Smooth Email Service Our email hosting service is more than smooth – it’s fast, reliable, and designed to protect businesses from viruses, helping them grow every day.",
      "99.9% Uptime Guarantee Our ongoing website support is aimed at helping businesses stay up and running 24/7, while we monitor performance around the clock. 4.7 out of 5 stars based on 1,395 Google reviews."
    ]
  }
]

urls = [
    "https://websouls.com/web-hosting",
    "https://websouls.com",
   "https://websouls.com/website-security-management",
     "https://websouls.com/online-store-management",
    "https://websouls.com/360-degree-digital-marketing",
###     "https://websouls.com/ui-ux-design",
    "https://websouls.com/custom-software-development",
    "https://websouls.com/laravel-custom-development",
    "https://websouls.com/react-custom-development",
    "https://websouls.com/shopify-development",
    "https://websouls.com/wordpress-development",
  "https://websouls.com/social-media-marketing",  
    "https://websouls.com/google-ads",
    "https://websouls.com/content-writing",
###    "https://websouls.com/mobile-app-development",
    "https://websouls.com/web-hosting-with-domain",
    "https://websouls.com/seo-services",
    "https://websouls.com/contactus",
    "https://websouls.com/about",
    "https://websouls.com/team",
    "https://websouls.com/shared-hosting",
    "https://websouls.com/domain-transfer",
    "https://websouls.com/ecommerce-solution",
    "https://websouls.com/policy",
    "https://websouls.com/buy-pk-domain",
    "https://websouls.com/ssl-certificates",
    "https://websouls.com/pk-vps",
    "https://websouls.com/vps-hosting",
    "https://websouls.com/wordpress-hosting-in-pakistan",
    "https://websouls.com/reseller-hosting",
    "https://websouls.com/buy-ae-domains",
    "https://websouls.com/whyus",
    "https://websouls.com/dedicated-server",
    "https://websouls.com/privacy",
    "https://websouls.com/web-development",
    "https://websouls.com/domain-registration",
    "https://websouls.com/payment-methods"
]


def json_to_markdown(data):
    """Convert extracted JSON (list of dicts) into markdown text for RAG."""
    md_sections = []

    for item in data:
        if not isinstance(item, dict):
            # skip invalid/stray items
            continue  

        tags = item.get("tags", [])
        tags_str = ", ".join(tags) if tags else "No tags"

        content = item.get("content", [])
        if isinstance(content, str):
            content = [content]

        md = f"### {tags_str}\n\n"
        for line in content:
            md += f"- {line}\n"
        md_sections.append(md.strip())

    return "\n\n".join(md_sections)

def sanitize_filename(url: str) -> str:
    # Replace all non-alphanumeric chars with "_"
    return re.sub(r'[^a-zA-Z0-9]', '_', url)


async def crawl_streaming():
    undetected_adapter = UndetectedAdapter()

    browser_config = BrowserConfig(headless=False, verbose=False)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=True,  # Enable streaming mode,
            exclude_all_images=True,
            scan_full_page=True,  # Tells the crawler to try scrolling the entire page
        scroll_delay=0.2,     # Delay (seconds) between scroll steps
                word_count_threshold=200,
                js_code=[
                    "await new Promise(r => setTimeout(r, 3000));",
                    """
    console.log('>>> Waiting for USD toggle…');
    const intervalUsd = setInterval(() => {
        const usdBtn = [...document.querySelectorAll('div,button,span')]
                        .find(el => el.innerText.trim() === 'USD');
        if (usdBtn) {
            console.log('Found USD button:', usdBtn);
            usdBtn.querySelector('div')?.click();
            clearInterval(intervalUsd);
        }
    }, 200);
    """,
    """
    console.log('>>> Waiting for PKR option…');
    const intervalPkr = setInterval(() => {
        const pkrBtn = [...document.querySelectorAll('li')]
                        .find(el => el.innerText.trim() === 'PKR');
        if (pkrBtn) {
            console.log('Found PKR option:', pkrBtn);
            pkrBtn.click();
            clearInterval(intervalPkr);
        }
    }, 200);
    """
                ],
magic=True, 
                delay_before_return_html=2.0

    )

    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=100.0,
        check_interval=1.0,
        max_session_permit=1,
        rate_limiter=RateLimiter(       # Optional rate limiting
        base_delay=(1.0, 2.0),
        max_delay=30.0,
        max_retries=2
    ),
    #     monitor=CrawlerMonitor(         # Optional monitoring
    #     max_visible_rows=15,
    #     display_mode=DisplayMode.DETAILED
    # )
    )

    crawler_strategy = AsyncPlaywrightCrawlerStrategy(
        browser_config=browser_config,
        browser_adapter=undetected_adapter
    )
    async with AsyncWebCrawler(config=browser_config,crawler_strategy=crawler_strategy) as crawler:
        # Process results as they become available
        async for result in await crawler.arun_many(
            urls=urls,
            config=run_config,
            dispatcher=dispatcher,
             magic=True
            
        ):
            if result.success:
                # Process each result immediately
                filename = sanitize_filename(result.url) + ".md"
                with open(filename, "a", encoding="utf-8") as f:
                  f.write(result.markdown + "\n\n")
               
            else:
                print(f"Failed to crawl {result.url}: {result.error_message}")

# async def main():
#    # print(json_to_markdown(data))
#   #  1. Define the LLM extraction strategy
#     llm_strategy = LLMExtractionStrategy(
#        llm_config = LLMConfig(provider="openai/gpt-4o-mini", api_token=os.getenv("OPENAI_API_KEY")),
# #    schema=Product.schema_json(), # Or use model_json_schema()
#         extraction_type="block",
#         instruction = """
# Extract **all customer-facing information** from this page about the company's offerings. 
# This includes **every service, plan, package, feature, price, benefit, promotion, discount, 
# guarantee, policy, contact detail, FAQ, and unique selling point**. Nothing should be skipped.

# Be exhaustive: include **every plan and tier**, including entry-level, promotional, or hidden options. 
# Capture renewal prices, limitations, special conditions, and all details that a customer or sales assistant 
# would need to understand, compare, or sell the services effectively.

# Exclude only irrelevant content like repeated menus, navigation bars, ads, or unrelated links.
# """
# ,
#         chunk_token_threshold=4000,
#         overlap_rate=0.0,
#         apply_chunking=False,
#         input_format="markdown",   # or "html", "fit_markdown"
#         extra_args={"temperature": 0.0, "max_tokens": 800}
#     )

#     # 2. Build the crawler config
#     run_config = CrawlerRunConfig(
#        #   extraction_strategy=llm_strategy,
#  js_code = [
#     """
#     console.log('>>> Waiting for USD toggle…');
#     const intervalUsd = setInterval(() => {
#         const usdBtn = [...document.querySelectorAll('div,button,span')]
#                         .find(el => el.innerText.trim() === 'USD');
#         if (usdBtn) {
#             console.log('Found USD button:', usdBtn);
#             usdBtn.querySelector('div')?.click();
#             clearInterval(intervalUsd);
#         }
#     }, 200);
#     """,
#     """
#     console.log('>>> Waiting for PKR option…');
#     const intervalPkr = setInterval(() => {
#         const pkrBtn = [...document.querySelectorAll('li')]
#                         .find(el => el.innerText.trim() === 'PKR');
#         if (pkrBtn) {
#             console.log('Found PKR option:', pkrBtn);
#             pkrBtn.click();
#             clearInterval(intervalPkr);
#         }
#     }, 200);
#     """
# ]
# ,
#     delay_before_return_html=2.0,
#     page_timeout=60000,
#       stream=False ,
#        cache_mode=CacheMode.BYPASS,
# )    
#     dispatcher = MemoryAdaptiveDispatcher(
#         memory_threshold_percent=70.0,
#         check_interval=1.0,
#         max_session_permit=10,
#         # monitor=CrawlerMonitor(
#         #     display_mode=DisplayMode.DETAILED
#         # )
#     )

#     # 3. Create a browser config if needed
#     browser_cfg = BrowserConfig(headless=False)

#     async with AsyncWebCrawler(config=browser_cfg) as crawler:
#         # 4. Let's say we want to crawl a single page
#         results = await crawler.arun_many(
#              urls=urls,
#             config=run_config,
#             dispatcher=dispatcher
#         )
        
#         for result in results:
#             if result.success:
#              if result.url != None and result.markdown != None:
                 
#                 await   print(result.url +  result.markdown)
#                 # filename = sanitize_filename(result.url) + ".md"
#                 # with open(filename, "a", encoding="utf-8") as f:
#                 #   f.write(result.markdown + "\n\n")
#             else:
#                 print(f"Failed to crawl {result.url}: {result.error_message}")


# import asyncio
# from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
# from crawl4ai.content_filter_strategy import PruningContentFilter
# from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

async def main():
    # Create the undetected adapter
    undetected_adapter = UndetectedAdapter()

    # Create browser config
    browser_config = BrowserConfig(
        headless=False,  # Headless mode can be detected easier
        verbose=True,
        enable_stealth=True
    )

    # Create the crawler strategy with undetected adapter
    crawler_strategy = AsyncPlaywrightCrawlerStrategy(
        browser_config=browser_config,
        browser_adapter=undetected_adapter
    )

    # Create the crawler with our custom strategy
    async with AsyncWebCrawler(
        crawler_strategy=crawler_strategy,
        config=browser_config
    ) as crawler:
        # Your crawling code here
        result = await crawler.arun(
            url="https://websouls.com/mobile-app-development",
            config=CrawlerRunConfig(
                js_code=[
                    "await new Promise(r => setTimeout(r, 10000));"
                ],
            )
        )
        print(result.markdown)

asyncio.run(crawl_streaming())
