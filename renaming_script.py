import os 
import base64
import re 

def url_to_filename(url: str) -> str:
    encoded = base64.urlsafe_b64encode(url.encode()).decode()
    return f"{encoded}.md"

def filename_to_url(filename: str) -> str:
    encoded = filename.replace(".md", "")
    return base64.urlsafe_b64decode(encoded.encode()).decode()


def url_to_filename_fix(url: str) -> str:
    url_no_protocol = re.sub(r'^https?://', '', url)
    base, ext = os.path.splitext(url_no_protocol)
    safe_base = re.sub(r'[^a-zA-Z0-9]', '_', base)
    return f"https___{safe_base}{ext}.md"

directory = "./websouls_scraped_md"
for filename in os.listdir(directory):
    old_path = os.path.join(directory, filename)
    print("Hello")
    new_path = os.path.join(directory, filename + ".md")
    print(new_path)
    os.rename(old_path, new_path)
