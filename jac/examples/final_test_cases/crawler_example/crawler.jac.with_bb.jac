import:py from urllib.request, urlopen ; # BB: 0 Execution frequency: 1 Total execution time: 0.001 ms
import:py from urllib.parse, urljoin, urlparse ; # BB: 3 Execution frequency: 1 Total execution time: 0.000 ms
import:py from html.parser, HTMLParser ; # BB: 6 Execution frequency: 1 Total execution time: 0.000 ms
import:py sys ; # BB: 9 Execution frequency: 0 Total execution time: 0.000 ms
import:py time ; # BB: 12 Execution frequency: 0 Total execution time: 0.000 ms

with entry {
    start_url:str = "https://books.toscrape.com/"; # BB: 15 Execution frequency: 0 Total execution time: 0.000 ms

    domain: str = urlparse(start_url).netloc; # BB: 15
    depth_limit: int = 2; # BB: 15

    visited: set[str] = <>set(); # BB: 15
    queue: list[tuple[str, int]] = [(start_url, depth_limit)]; # BB: 15

    parser: HTMLParser = HTMLParser(); # BB: 15
    parser.links: list[str] = []; # BB: 15

    pages_crawled: int = 0; # BB: 15
}

can handle_starttag(tag: str, attrs: str) { # BB: 15
    if tag == 'a' { 
        for (attr, val) in attrs {
            if attr == 'href' {
                parser.links.append(val) ;
            }
        }
    }
}

with entry {
    parser.handle_starttag = handle_starttag; # BB: 15

    while queue { # BB: 16 Execution frequency: 0 Total execution time: 0.000 ms
        (url, depth) = queue.pop(0); # BB: 17 Execution frequency: 0 Total execution time: 0.000 ms

        if url in visited or depth == 0 { # BB: 17
            continue ; # BB: 19 Execution frequency: 0 Total execution time: 0.000 ms
        }

        visited.add(url) ; # BB: 20 Execution frequency: 0 Total execution time: 0.000 ms
        pages_crawled += 1; # BB: 20

        print(f"{'[Depth '}{depth}{'] Visiting: '}{url}") ; # BB: 20

        if pages_crawled >= 5 { # BB: 20
            print("Stopping after 10 pages."); # BB: 21 Execution frequency: 0 Total execution time: 0.000 ms
            break; # BB: 21
        }

        response: Any = urlopen(url, timeout=5); # BB: 22 Execution frequency: 0 Total execution time: 0.000 ms
        content_type: str = response.headers.get('Content-Type', ''); # BB: 22
        if 'text/html' not in content_type { # BB: 22
            continue ; # BB: 23 Execution frequency: 0 Total execution time: 0.000 ms
        }

        html: str = response.read().decode(errors='ignore'); # BB: 24 Execution frequency: 0 Total execution time: 0.000 ms
        parser.links.clear() ; # BB: 24
        parser.feed(html) ; # BB: 24

        for link in parser.links { # BB: 24
            absolute: str = urljoin(url, link); # BB: 25 Execution frequency: 0 Total execution time: 0.000 ms
            parsed:str = urlparse(absolute); # BB: 25

            if parsed.netloc == domain { # BB: 25
                queue.append((absolute, (depth - 1))) ; # BB: 27 Execution frequency: 0 Total execution time: 0.000 ms
            }
        }

        time.sleep(0.5) ; # BB: 28 Execution frequency: 0 Total execution time: 0.000 ms
    }
}