import:py from urllib.request, urlopen ;
#   Instructions: [SETUP_ANNOTATIONS(None)]
import:py from urllib.parse, urljoin, urlparse ;
#   Instructions: [LOAD_NAME(_jac_typ), LOAD_ATTR(TYPE_CHECKING) [no line], POP_JUMP_IF_FALSE(152) [no line]]
import:py from html.parser, HTMLParser ;
#   Instructions: [LOAD_NAME(_jac_typ), LOAD_ATTR(TYPE_CHECKING) [no line], POP_JUMP_IF_FALSE(230) [no line]]
import:py sys ;
#   Instructions: [LOAD_NAME(_jac_typ), LOAD_ATTR(TYPE_CHECKING) [no line], POP_JUMP_IF_FALSE(300) [no line]]
import:py time ;
#   Instructions: [LOAD_NAME(_jac_typ), LOAD_ATTR(TYPE_CHECKING) [no line], POP_JUMP_IF_FALSE(366) [no line]]

with entry {
    start_url:str = "https://books.toscrape.com/";
    #   Instructions: [LOAD_CONST(https://books.toscrape.com/), STORE_NAME(start_url) [no line], LOAD_CONST(str) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(start_url) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(urlparse) [no line], LOAD_NAME(start_url) [no line], CALL(1) [no line], LOAD_ATTR(netloc) [no line], STORE_NAME(domain) [no line], LOAD_CONST(str) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(domain) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(depth_limit) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(depth_limit) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(set) [no line], CALL(0) [no line], STORE_NAME(visited) [no line], LOAD_CONST(set[str]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(visited) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(depth_limit) [no line], BUILD_TUPLE(2) [no line], BUILD_LIST(1) [no line], STORE_NAME(queue) [no line], LOAD_CONST(list[tuple[str, int]]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(queue) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(HTMLParser) [no line], CALL(0) [no line], STORE_NAME(parser) [no line], LOAD_CONST(HTMLParser) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(parser) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(parser) [no line], STORE_ATTR(links) [no line], STORE_NAME(pages_crawled) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(pages_crawled) [no line], STORE_SUBSCR(None) [no line], LOAD_CONST(<code object handle_starttag at 0x7f374dc65cb0, file "/root/jaseci-ginS/jac/examples/final_test_cases/crawler_example/crawler.jac", line 22>) [no line], MAKE_FUNCTION(4) [no line], STORE_NAME(handle_starttag) [no line], LOAD_NAME(parser) [no line], STORE_ATTR(handle_starttag) [no line]]

    domain: str = urlparse(start_url).netloc;
    #   Instructions: [PUSH_NULL(None)]
    depth_limit: int = 2;
    #   Instructions: [LOAD_CONST(2)]

    visited: set[str] = <>set();
    #   Instructions: [PUSH_NULL(None)]
    queue: list[tuple[str, int]] = [(start_url, depth_limit)];
    #   Instructions: [LOAD_NAME(start_url)]

    parser: HTMLParser = HTMLParser();
    #   Instructions: [PUSH_NULL(None)]
    parser.links: list[str] = [];
    #   Instructions: [BUILD_LIST(0)]

    pages_crawled: int = 0;
    #   Instructions: [LOAD_CONST(0)]
}

can handle_starttag(tag: str, attrs: str) {
#   Instructions: [LOAD_CONST(('tag', 'str', 'attrs', 'str', 'return', 'None'))]
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
    #   Instructions: [LOAD_NAME(handle_starttag)]

    while queue {
    #   Instructions: [LOAD_NAME(queue), EXTENDED_ARG(1) [no line], POP_JUMP_IF_FALSE(1236) [no line], LOAD_NAME(queue)]
        (url, depth) = queue.pop(0);
        #   Instructions: [LOAD_NAME(queue), LOAD_ATTR(pop) [no line], LOAD_CONST(0) [no line], CALL(1) [no line], UNPACK_SEQUENCE(2) [no line], STORE_NAME(url) [no line], STORE_NAME(depth) [no line], LOAD_NAME(visited) [no line], CONTAINS_OP(0) [no line], POP_JUMP_IF_TRUE(654) [no line]]

        if url in visited or depth == 0 {
        #   Instructions: [LOAD_NAME(url)]
            continue ;
            #   Instructions: [JUMP_BACKWARD(590)]
        }

        visited.add(url) ;
        #   Instructions: [LOAD_NAME(visited), LOAD_ATTR(add) [no line], LOAD_NAME(url) [no line], CALL(1) [no line], POP_TOP(None) [no line], LOAD_CONST(1) [no line], BINARY_OP(13) [no line], STORE_NAME(pages_crawled) [no line], LOAD_NAME(print) [no line], LOAD_CONST([Depth ) [no line], FORMAT_VALUE((None, False)) [no line], LOAD_NAME(depth) [no line], FORMAT_VALUE((None, False)) [no line], LOAD_CONST(] Visiting: ) [no line], FORMAT_VALUE((None, False)) [no line], LOAD_NAME(url) [no line], FORMAT_VALUE((None, False)) [no line], BUILD_STRING(4) [no line], CALL(1) [no line], POP_TOP(None) [no line], LOAD_CONST(5) [no line], COMPARE_OP(>=) [no line], POP_JUMP_IF_FALSE(760) [no line]]
        pages_crawled += 1;
        #   Instructions: [LOAD_NAME(pages_crawled)]

        print(f"{'[Depth '}{depth}{'] Visiting: '}{url}") ;
        #   Instructions: [PUSH_NULL(None)]

        if pages_crawled >= 5 {
        #   Instructions: [LOAD_NAME(pages_crawled)]
            print("Stopping after 10 pages.");
            #   Instructions: [PUSH_NULL(None), LOAD_NAME(print) [no line], LOAD_CONST(Stopping after 10 pages.) [no line], CALL(1) [no line], POP_TOP(None) [no line]]
            break;
            #   Instructions: [RETURN_CONST(None)]
        }

        response: Any = urlopen(url, timeout=5);
        #   Instructions: [PUSH_NULL(None), LOAD_NAME(urlopen) [no line], LOAD_NAME(url) [no line], LOAD_CONST(5) [no line], KW_NAMES(('timeout',)) [no line], CALL(2) [no line], STORE_NAME(response) [no line], LOAD_CONST(Any) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(response) [no line], STORE_SUBSCR(None) [no line], LOAD_ATTR(headers) [no line], LOAD_ATTR(get) [no line], LOAD_CONST(Content-Type) [no line], LOAD_CONST() [no line], CALL(2) [no line], STORE_NAME(content_type) [no line], LOAD_CONST(str) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(content_type) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(content_type) [no line], CONTAINS_OP(1) [no line], POP_JUMP_IF_FALSE(866) [no line]]
        content_type: str = response.headers.get('Content-Type', '');
        #   Instructions: [LOAD_NAME(response)]
        if 'text/html' not in content_type {
        #   Instructions: [LOAD_CONST(text/html)]
            continue ;
            #   Instructions: [JUMP_BACKWARD(590)]
        }

        html: str = response.read().decode(errors='ignore');
        #   Instructions: [LOAD_NAME(response), LOAD_ATTR(read) [no line], CALL(0) [no line], LOAD_ATTR(decode) [no line], LOAD_CONST(ignore) [no line], KW_NAMES(('errors',)) [no line], CALL(1) [no line], STORE_NAME(html) [no line], LOAD_CONST(str) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(html) [no line], STORE_SUBSCR(None) [no line], LOAD_ATTR(links) [no line], LOAD_ATTR(clear) [no line], CALL(0) [no line], POP_TOP(None) [no line], LOAD_ATTR(feed) [no line], LOAD_NAME(html) [no line], CALL(1) [no line], POP_TOP(None) [no line], LOAD_ATTR(links) [no line], GET_ITER(None) [no line]]
        parser.links.clear() ;
        #   Instructions: [LOAD_NAME(parser)]
        parser.feed(html) ;
        #   Instructions: [LOAD_NAME(parser)]

        for link in parser.links {
        #   Instructions: [LOAD_NAME(parser), END_FOR(None), LOAD_NAME(time) [no line], LOAD_ATTR(sleep) [no line], LOAD_CONST(0.5) [no line], CALL(1) [no line], POP_TOP(None) [no line], POP_JUMP_IF_FALSE(1234) [no line]]
            absolute: str = urljoin(url, link);
            #   Instructions: [FOR_ITER(1188) [no line], STORE_NAME(link) [no line], PUSH_NULL(None), LOAD_NAME(urljoin) [no line], LOAD_NAME(url) [no line], LOAD_NAME(link) [no line], CALL(2) [no line], STORE_NAME(absolute) [no line], LOAD_CONST(str) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(absolute) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(urlparse) [no line], LOAD_NAME(absolute) [no line], CALL(1) [no line], STORE_NAME(parsed) [no line], LOAD_CONST(str) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(parsed) [no line], STORE_SUBSCR(None) [no line], LOAD_ATTR(netloc) [no line], LOAD_NAME(domain) [no line], COMPARE_OP(==) [no line], POP_JUMP_IF_TRUE(1142) [no line]]
            parsed:str = urlparse(absolute);
            #   Instructions: [PUSH_NULL(None)]

            if parsed.netloc == domain {
            #   Instructions: [LOAD_NAME(parsed)]
                queue.append((absolute, (depth - 1))) ;
                #   Instructions: [LOAD_NAME(queue), LOAD_ATTR(append) [no line], LOAD_NAME(absolute) [no line], LOAD_NAME(depth) [no line], LOAD_CONST(1) [no line], BINARY_OP(10) [no line], BUILD_TUPLE(2) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(1050) [no line]]
            }
        }

        time.sleep(0.5) ;
        #   Instructions: [PUSH_NULL(None)]
    }
}