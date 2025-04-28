import urllib.request
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser
import sys
import time

if len(sys.argv) != 2:
    print("Usage: python flat_crawler.py https://example.com")
    sys.exit(1)

start_url = sys.argv[1]
domain = urlparse(start_url).netloc
depth_limit = 2
visited = set()
queue = [(start_url, depth_limit)]

# A super basic parser object
parser = HTMLParser()
parser.links = []

def handle_starttag(tag, attrs):
    if tag == 'a':
        for attr, val in attrs:
            if attr == 'href':
                parser.links.append(val)

parser.handle_starttag = handle_starttag

while queue:
    url, depth = queue.pop(0)
    if url in visited or depth == 0:
        continue
    visited.add(url)
    print(f"[Depth {depth}] Visiting: {url}")
    try:
        response = urllib.request.urlopen(url, timeout=5)
        content_type = response.headers.get('Content-Type', '')
        if "text/html" not in content_type:
            continue
        html = response.read().decode(errors='ignore')
        parser.links.clear()
        parser.feed(html)
        for link in parser.links:
            absolute = urljoin(url, link)
            parsed = urlparse(absolute)
            if parsed.netloc == domain:
                queue.append((absolute, depth - 1))
        time.sleep(0.5)
    except Exception as e:
        print(f"Error accessing {url}: {e}")
