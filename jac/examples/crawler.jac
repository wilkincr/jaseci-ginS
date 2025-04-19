import:py re ;
import:py sys ;
import:py time ;
import:py math ;
import:py urllib ;
import:py urllib.error ;
import:py urllib.request ;
import:py from urllib.parse, urlparse ;
import:py optparse ;
import:py hashlib ;
import:py from cgi, escape ;
import:py from queue, Queue, Empty as QueueEmpty ;
import:py from bs4, BeautifulSoup ;

with entry {
    if __name__ == '__main__' {

        __version__ = '0.2';    

        AGENT = ('WebCrawler/%s' % __version__);
        

        USAGE = '%prog [options] <url>';    

        VERSION = ('%prog v' + __version__);
        

        parser = optparse.OptionParser(usage=USAGE, version=VERSION);    

        parser.add_option(
            '-q',
            '--quiet',
            action='store_true',
            default=False,
            dest='quiet',
            help='Enable quiet mode'
        ) ;    

        parser.add_option(
            '-l',
            '--links',
            action='store_true',
            default=False,
            dest='links',
            help='Get links for specified url only'
        ) ;    

        parser.add_option(
            '-d',
            '--depth',
            action='store',
            type='int',
            default=30,
            dest='depth_limit',
            help='Maximum depth to traverse'
        ) ;    

        parser.add_option(
            '-c',
            '--confine',
            action='store',
            type='string',
            dest='confine',
            help='Confine crawl to specified prefix'
        ) ;    

        parser.add_option(
            '-x',
            '--exclude',
            action='append',
            type='string',
            dest='exclude',
            default=[],
            help='Exclude URLs by prefix'
        ) ;    

        parser.add_option(
            '-L',
            '--show-links',
            action='store_true',
            default=False,
            dest='out_links',
            help='Output links found'
        ) ;    

        parser.add_option(
            '-u',
            '--show-urls',
            action='store_true',
            default=False,
            dest='out_urls',
            help='Output URLs found'
        ) ;    

        parser.add_option(
            '-D',
            '--dot',
            action='store_true',
            default=False,
            dest='out_dot',
            help='Output Graphviz dot file'
        ) ;    

        (opts, args) = parser.parse_args();    
        if len(args) < 1 {

            parser.print_help(sys.stderr) ;    
            sys.exit(1) ;
        }
        
        if opts.out_links and opts.out_urls {

            parser.print_help(sys.stderr) ;    
            parser.error('options -L and -u are mutually exclusive') ;
        }
        

        url = args[0];    

        node_alias = {};
        
        if opts.links {

            request = urllib.request.Request(url);    
            request.add_header('User-Agent', AGENT) ;    
            handle = urllib.request.build_opener();    
            data = handle.open(request);    
            mime_type = data.info().get_content_type();    
            if mime_type == 'text/html' {

                content = data.read().decode('utf-8', errors='replace');    

                soup = BeautifulSoup(content, 'html.parser');    

                tags = soup('a');    

                out_urls = [];
                
                for tag in tags {

                    href = tag.get('href');    
                    if href is not None {

                        url_joined = urllib.parse.urljoin(url, escape(href));    
                        if url_joined not in out_urls {

                            out_urls.append(url_joined) ;
                        }
                    }
                }
                
                for (i, url) in enumerate(out_urls) {

                    print(('%d. %s' % (i, url))) ;
                }
            }

            
            sys.exit(0) ;
        }
        

        depth_limit = opts.depth_limit;    

        confine_prefix = opts.confine;    

        exclude = opts.exclude;    

        sTime = time.time();    

        print(
            sys.stderr,
            ('Crawling %s (Max Depth: %d)' % (url, depth_limit))
        ) ;
        

        host = urlparse(url)[1];    

        urls_seen = <>set();    

        urls_remembered = <>set();    

        visited_links = <>set();    

        links_remembered = <>set();    

        num_links = 0;    

        num_followed = 0;    

        q = Queue();    

        q.put((url, 0)) ;    
        class Link {
            can init(src: Any, dst: Any, link_type: Any) {

                self.src = src;    
                self.dst = dst;    
                self.link_type = link_type;
            }
            
            can __hash__() {

                return hash((self.src, self.dst, self.link_type)) ;
            }
            
            can __eq__(other: Any) {
                return self.src == other.src
                    and self.dst == other.dst
                    and self.link_type == other.link_type ;
            }
            
            can __str__() {
                return ((self.src + ' -> ') + self.dst) ;
            }
        }

        
        while not q.empty() {

            (this_url, depth) = q.get();    
            if depth > depth_limit {
                continue ;
            }

            
            prefix_ok = confine_prefix is None
                or this_url.startswith(confine_prefix);    
            exclude_ok = all([not this_url.startswith(p) for p in exclude]);    
            not_visited = this_url not in visited_links;    
            same_host = False;    
            host_from_url = urlparse(this_url)[1];    
            same_host = re.match(('.*%s' % host), host_from_url) is not None;    
            if prefix_ok
                and exclude_ok
                and not_visited
                and same_host or depth == 0 {

                visited_links.add(this_url) ;    
                num_followed += 1;    
                request = urllib.request.Request(this_url);    
                request.add_header('User-Agent', AGENT) ;    
                handle = None;    
                handle = urllib.request.build_opener();    
                if handle {

                    data = None;    
                    data = handle.open(request);    
                    if data {

                        mime_type = data.info().get_content_type();    
                        url = data.geturl();    
                        if mime_type == 'text/html' {

                            content = data.read().decode('utf-8', errors='replace');    
                            soup = BeautifulSoup(content, 'html.parser');    
                            tags = soup('a');    
                            for tag in tags {

                                href = tag.get('href');    
                                if href is not None {

                                    link_url = urllib.parse.urljoin(this_url, escape(href));    
                                    (base, frag) = urllib.parse.urldefrag(link_url);    
                                    link_url = base;    
                                    if link_url not in urls_seen {

                                        q.put((link_url, (depth + 1))) ;
                                        
                                        urls_seen.add(link_url) ;
                                    }
                                    
                                    out_prefix_ok = confine_prefix is None
                                        or link_url.startswith(confine_prefix);    
                                    out_same_host = False;    
                                    out_host = urlparse(link_url)[1];    
                                    out_same_host = re.match(('.*%s' % host), out_host) is not None;    
                                    if out_prefix_ok and out_same_host {

                                        num_links += 1;    
                                        urls_remembered.add(link_url) ;    
                                        link = Link(this_url, link_url, 'href');    
                                        if link not in links_remembered {

                                            links_remembered.add(link) ;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if opts.out_urls {

            print('\n'.join(urls_remembered)) ;
        }
        
        if opts.out_links {
            for link in links_remembered {

                print(((link.src + ' -> ') + link.dst)) ;
            }
        }

        
        if opts.out_dot {

            print('digraph Crawl {') ;    
            print('\t edge [K=0.2, len=0.1];') ;    
            for link in links_remembered {

                src_url = link.src;    

                dst_url = link.dst;    
                if src_url not in node_alias {

                    m = hashlib.md5();    

                    m.update(src_url.encode('utf-8')) ;    

                    name = ('N' + m.hexdigest());
                    

                    node_alias[src_url] = name;    

                    print(('\t%s [label="%s"];' % (name, src_url))) ;
                }

                
                if dst_url not in node_alias {

                    m = hashlib.md5();    

                    m.update(dst_url.encode('utf-8')) ;    

                    name = ('N' + m.hexdigest());
                    

                    node_alias[dst_url] = name;    

                    print(('\t%s [label="%s"];' % (name, dst_url))) ;
                }

                

                print(
                    (((('\t' + node_alias[src_url]) + ' -> ') + node_alias[dst_url]) + ';')
                ) ;
            }

            
            print('}') ;
        }
        

        eTime = time.time();    

        tTime = (eTime - sTime);
        

        print(sys.stderr, ('Found:    %d' % num_links)) ;
        

        print(sys.stderr, ('Followed: %d' % num_followed)) ;
        

        print(
            sys.stderr,
            ('Stats:    (%d/s after %0.2fs)' % (int(math.ceil((float(num_links) / tTime)))
            ,tTime))
        ) ;
    }
}

