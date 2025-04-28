import:py from urllib.request, urlopen ;
import:py from urllib.parse, urljoin, urlparse ;
import:py from html.parser, HTMLParser ;
import:py sys ;
import:py time ;

with entry {
    start_url = "https://books.toscrape.com/";

    domain = urlparse(start_url).netloc;    
    depth_limit = 2;    

    visited = <>set();    
    queue = [(start_url, depth_limit)];    

    parser = HTMLParser();    
    parser.links = [];

    pages_crawled = 0;
}

can handle_starttag(tag: Any, attrs: Any) {
    if tag == 'a' { 
        for (attr, val) in attrs {
            if attr == 'href' {
                parser.links.append(val) ;
            }
        }
    }
}

with entry {
    parser.handle_starttag = handle_starttag;    

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

        try  {
            response = urlopen(url, timeout=5);
            content_type = response.headers.get('Content-Type', '');    
            if 'text/html' not in content_type {
                continue ;
            }

            html = response.read().decode(errors='ignore');    
            parser.links.clear() ;    
            parser.feed(html) ;    

            for link in parser.links {
                absolute = urljoin(url, link);    
                parsed = urlparse(absolute);    

                if parsed.netloc == domain {
                    queue.append((absolute, (depth - 1))) ;
                }
            }

            time.sleep(0.5) ;
        } except Exception as e {
            print(f"{'Error accessing '}{url}{': '}{e}") ;
        }
    }
}
