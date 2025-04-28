import:py from urllib.request, urlopen ;
import:py from urllib.parse, urljoin, urlparse ;
import:py from html.parser, HTMLParser ;
import:py sys ;
import:py time ;

with entry {
    start_url:str = "https://books.toscrape.com/";

    domain: str = urlparse(start_url).netloc;
    depth_limit: int = 2;

    visited: set[str] = <>set();
    queue: list[tuple[str, int]] = [(start_url, depth_limit)];

    parser: HTMLParser = HTMLParser();
    parser.links: list[str] = [];

    pages_crawled: int = 0;
}

can handle_starttag(tag: str, attrs: str) -> str{
    if tag == 'a' { 
        for (attr, val) in attrs {
            if attr == 'href' {
                parser.links.append(val) ;
            }
        }
    }
}

with entry {
    parser.handle_starttag:str = handle_starttag;

    while queue {
        (url, depth) = queue.pop(0);

        if url in visited or depth == 0 {
            continue ;
        }

        visited.add(url) ;
        pages_crawled += 1;

        print(f"{'[Depth '}{depth}{'] Visiting: '}{url}") ;

        if pages_crawled >= 5 {
            print("Stopping after 10 pages.");
            break;
        }

        response: Any = urlopen(url, timeout=5);
        content_type: str = response.headers.get('Content-Type', '');
        if 'text/html' not in content_type {
            continue ;
        }

        html: str = response.read().decode(errors='ignore');
        parser.links.clear() ;
        parser.feed(html) ;

        for link in parser.links {
            absolute: str = urljoin(url, link);
            parsed:str = urlparse(absolute);

            if parsed.netloc == domain {
                queue.append((absolute, (depth - 1))) ;
            }
        }

        time.sleep(0.5) ;
    }
}
