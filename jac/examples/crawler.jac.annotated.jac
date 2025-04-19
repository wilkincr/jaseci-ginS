import:py re ;  # bb0 → bb2, bb1 [exec=1, time=0.0046s] [mem=57284 bytes] | bb1 → bb3 [exec=0, time=0.0000s] [mem=57284 bytes] | bb2 → bb3 [exec=1, time=0.0032s] [mem=57284 bytes]
        #   [LOAD_CONST 0 [jump_t=False] [arg=0,], LOAD_CONST ('annotations' [jump_t=False] [arg=1,], IMPORT_NAME __future__ [jump_t=False] [arg=0,], IMPORT_FROM annotations [jump_t=False] [arg=1,], STORE_NAME annotations [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,], LOAD_CONST 0 [jump_t=False] [arg=0,], LOAD_CONST ('jac_import' [jump_t=False] [arg=2,], IMPORT_NAME jaclang [jump_t=False] [arg=2,], IMPORT_FROM jac_import [jump_t=False] [arg=3,], STORE_NAME __jac_import__ [jump_t=False] [arg=4,], POP_TOP [jump_t=False] [arg=None,], LOAD_CONST 0 [jump_t=False] [arg=0,], LOAD_CONST None [jump_t=False] [arg=3,], IMPORT_NAME typing [jump_t=False] [arg=5,], STORE_NAME _jac_typ [jump_t=False] [arg=6,], LOAD_NAME _jac_typ [jump_t=False] [arg=6,], LOAD_ATTR TYPE_CHECKING [jump_t=False] [arg=14,], POP_JUMP_IF_FALSE to 68 [jump_t=False] [arg=5,], LOAD_CONST 0 [jump_t=False] [arg=0,], LOAD_CONST None [jump_t=False] [arg=3,], IMPORT_NAME re [jump_t=False] [arg=8,], STORE_NAME re [jump_t=False] [arg=8,], JUMP_FORWARD to 100 [jump_t=False] [arg=16,], PUSH_NULL [jump_t=True] [arg=None,], LOAD_NAME __jac_import__ [jump_t=False] [arg=4,], LOAD_CONST 're' [jump_t=False] [arg=4,], LOAD_NAME __file__ [jump_t=False] [arg=9,], LOAD_CONST 'py' [jump_t=False] [arg=5,], LOAD_CONST False [jump_t=False] [arg=6,], LOAD_CONST None [jump_t=False] [arg=3,], BUILD_MAP 0 [jump_t=False] [arg=0,], KW_NAMES ('target' [jump_t=False] [arg=7,], CALL 6 [jump_t=False] [arg=6,], UNPACK_SEQUENCE 1 [jump_t=False] [arg=1,], STORE_NAME re [jump_t=False] [arg=8,]]
import:py sys ;  # bb5 → bb6 [exec=1, time=0.0031s] [mem=57284 bytes] | bb4 → bb6 [exec=0, time=0.0000s] [mem=57284 bytes] | bb3 → bb5, bb4 [exec=1, time=0.0007s] [mem=57284 bytes]
        #   [LOAD_NAME _jac_typ [jump_t=True] [arg=6,], LOAD_ATTR TYPE_CHECKING [jump_t=False] [arg=14,], POP_JUMP_IF_FALSE to 134 [jump_t=False] [arg=5,], LOAD_CONST 0 [jump_t=False] [arg=0,], LOAD_CONST None [jump_t=False] [arg=3,], IMPORT_NAME sys [jump_t=False] [arg=10,], STORE_NAME sys [jump_t=False] [arg=10,], JUMP_FORWARD to 166 [jump_t=False] [arg=16,], PUSH_NULL [jump_t=True] [arg=None,], LOAD_NAME __jac_import__ [jump_t=False] [arg=4,], LOAD_CONST 'sys' [jump_t=False] [arg=8,], LOAD_NAME __file__ [jump_t=False] [arg=9,], LOAD_CONST 'py' [jump_t=False] [arg=5,], LOAD_CONST False [jump_t=False] [arg=6,], LOAD_CONST None [jump_t=False] [arg=3,], BUILD_MAP 0 [jump_t=False] [arg=0,], KW_NAMES ('target' [jump_t=False] [arg=7,], CALL 6 [jump_t=False] [arg=6,], UNPACK_SEQUENCE 1 [jump_t=False] [arg=1,], STORE_NAME sys [jump_t=False] [arg=10,]]
import:py time ;  # bb6 → bb8, bb7 [exec=1, time=0.0007s] [mem=57284 bytes] | bb7 → bb9 [exec=0, time=0.0000s] [mem=57284 bytes] | bb8 → bb9 [exec=1, time=0.0034s] [mem=57284 bytes]
        #   [LOAD_NAME _jac_typ [jump_t=True] [arg=6,], LOAD_ATTR TYPE_CHECKING [jump_t=False] [arg=14,], POP_JUMP_IF_FALSE to 200 [jump_t=False] [arg=5,], LOAD_CONST 0 [jump_t=False] [arg=0,], LOAD_CONST None [jump_t=False] [arg=3,], IMPORT_NAME time [jump_t=False] [arg=11,], STORE_NAME time [jump_t=False] [arg=11,], JUMP_FORWARD to 232 [jump_t=False] [arg=16,], PUSH_NULL [jump_t=True] [arg=None,], LOAD_NAME __jac_import__ [jump_t=False] [arg=4,], LOAD_CONST 'time' [jump_t=False] [arg=9,], LOAD_NAME __file__ [jump_t=False] [arg=9,], LOAD_CONST 'py' [jump_t=False] [arg=5,], LOAD_CONST False [jump_t=False] [arg=6,], LOAD_CONST None [jump_t=False] [arg=3,], BUILD_MAP 0 [jump_t=False] [arg=0,], KW_NAMES ('target' [jump_t=False] [arg=7,], CALL 6 [jump_t=False] [arg=6,], UNPACK_SEQUENCE 1 [jump_t=False] [arg=1,], STORE_NAME time [jump_t=False] [arg=11,]]
import:py math ;  # bb9 → bb11, bb10 [exec=1, time=0.0008s] [mem=57284 bytes] | bb10 → bb12 [exec=0, time=0.0000s] [mem=57284 bytes] | bb11 → bb12 [exec=1, time=0.0033s] [mem=57284 bytes]
        #   [LOAD_NAME _jac_typ [jump_t=True] [arg=6,], LOAD_ATTR TYPE_CHECKING [jump_t=False] [arg=14,], POP_JUMP_IF_FALSE to 266 [jump_t=False] [arg=5,], LOAD_CONST 0 [jump_t=False] [arg=0,], LOAD_CONST None [jump_t=False] [arg=3,], IMPORT_NAME math [jump_t=False] [arg=12,], STORE_NAME math [jump_t=False] [arg=12,], JUMP_FORWARD to 298 [jump_t=False] [arg=16,], PUSH_NULL [jump_t=True] [arg=None,], LOAD_NAME __jac_import__ [jump_t=False] [arg=4,], LOAD_CONST 'math' [jump_t=False] [arg=10,], LOAD_NAME __file__ [jump_t=False] [arg=9,], LOAD_CONST 'py' [jump_t=False] [arg=5,], LOAD_CONST False [jump_t=False] [arg=6,], LOAD_CONST None [jump_t=False] [arg=3,], BUILD_MAP 0 [jump_t=False] [arg=0,], KW_NAMES ('target' [jump_t=False] [arg=7,], CALL 6 [jump_t=False] [arg=6,], UNPACK_SEQUENCE 1 [jump_t=False] [arg=1,], STORE_NAME math [jump_t=False] [arg=12,]]
import:py urllib ;  # bb13 → bb15 [exec=0, time=0.0000s] [mem=57284 bytes] | bb14 → bb15 [exec=1, time=0.0032s] [mem=57284 bytes] | bb12 → bb14, bb13 [exec=1, time=0.0007s] [mem=57284 bytes]
        #   [LOAD_NAME _jac_typ [jump_t=True] [arg=6,], LOAD_ATTR TYPE_CHECKING [jump_t=False] [arg=14,], POP_JUMP_IF_FALSE to 332 [jump_t=False] [arg=5,], LOAD_CONST 0 [jump_t=False] [arg=0,], LOAD_CONST None [jump_t=False] [arg=3,], IMPORT_NAME urllib [jump_t=False] [arg=13,], STORE_NAME urllib [jump_t=False] [arg=13,], JUMP_FORWARD to 364 [jump_t=False] [arg=16,], PUSH_NULL [jump_t=True] [arg=None,], LOAD_NAME __jac_import__ [jump_t=False] [arg=4,], LOAD_CONST 'urllib' [jump_t=False] [arg=11,], LOAD_NAME __file__ [jump_t=False] [arg=9,], LOAD_CONST 'py' [jump_t=False] [arg=5,], LOAD_CONST False [jump_t=False] [arg=6,], LOAD_CONST None [jump_t=False] [arg=3,], BUILD_MAP 0 [jump_t=False] [arg=0,], KW_NAMES ('target' [jump_t=False] [arg=7,], CALL 6 [jump_t=False] [arg=6,], UNPACK_SEQUENCE 1 [jump_t=False] [arg=1,], STORE_NAME urllib [jump_t=False] [arg=13,]]
import:py urllib.error ;  # bb17 → bb18 [exec=1, time=0.0048s] [mem=57284 bytes] | bb15 → bb17, bb16 [exec=1, time=0.0007s] [mem=57284 bytes] | bb16 → bb18 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME _jac_typ [jump_t=True] [arg=6,], LOAD_ATTR TYPE_CHECKING [jump_t=False] [arg=14,], POP_JUMP_IF_FALSE to 398 [jump_t=False] [arg=5,], LOAD_CONST 0 [jump_t=False] [arg=0,], LOAD_CONST None [jump_t=False] [arg=3,], IMPORT_NAME urllib.error [jump_t=False] [arg=14,], STORE_NAME urllib [jump_t=False] [arg=13,], JUMP_FORWARD to 430 [jump_t=False] [arg=16,], PUSH_NULL [jump_t=True] [arg=None,], LOAD_NAME __jac_import__ [jump_t=False] [arg=4,], LOAD_CONST 'urllib.error' [jump_t=False] [arg=12,], LOAD_NAME __file__ [jump_t=False] [arg=9,], LOAD_CONST 'py' [jump_t=False] [arg=5,], LOAD_CONST False [jump_t=False] [arg=6,], LOAD_CONST None [jump_t=False] [arg=3,], BUILD_MAP 0 [jump_t=False] [arg=0,], KW_NAMES ('target' [jump_t=False] [arg=7,], CALL 6 [jump_t=False] [arg=6,], UNPACK_SEQUENCE 1 [jump_t=False] [arg=1,], STORE_NAME urllib [jump_t=False] [arg=13,]]
import:py urllib.request ;  # bb20 → bb21 [exec=1, time=0.0036s] [mem=57284 bytes] | bb19 → bb21 [exec=0, time=0.0000s] [mem=57284 bytes] | bb18 → bb20, bb19 [exec=1, time=0.0007s] [mem=57284 bytes]
        #   [LOAD_NAME _jac_typ [jump_t=True] [arg=6,], LOAD_ATTR TYPE_CHECKING [jump_t=False] [arg=14,], POP_JUMP_IF_FALSE to 464 [jump_t=False] [arg=5,], LOAD_CONST 0 [jump_t=False] [arg=0,], LOAD_CONST None [jump_t=False] [arg=3,], IMPORT_NAME urllib.request [jump_t=False] [arg=15,], STORE_NAME urllib [jump_t=False] [arg=13,], JUMP_FORWARD to 496 [jump_t=False] [arg=16,], PUSH_NULL [jump_t=True] [arg=None,], LOAD_NAME __jac_import__ [jump_t=False] [arg=4,], LOAD_CONST 'urllib.request' [jump_t=False] [arg=13,], LOAD_NAME __file__ [jump_t=False] [arg=9,], LOAD_CONST 'py' [jump_t=False] [arg=5,], LOAD_CONST False [jump_t=False] [arg=6,], LOAD_CONST None [jump_t=False] [arg=3,], BUILD_MAP 0 [jump_t=False] [arg=0,], KW_NAMES ('target' [jump_t=False] [arg=7,], CALL 6 [jump_t=False] [arg=6,], UNPACK_SEQUENCE 1 [jump_t=False] [arg=1,], STORE_NAME urllib [jump_t=False] [arg=13,]]
import:py from urllib.parse, urlparse ;  # bb21 → bb23, bb22 [exec=1, time=0.0008s] [mem=57284 bytes] | bb22 → bb24 [exec=0, time=0.0000s] [mem=57284 bytes] | bb23 → bb24 [exec=1, time=0.0036s] [mem=57284 bytes]
        #   [LOAD_NAME _jac_typ [jump_t=True] [arg=6,], LOAD_ATTR TYPE_CHECKING [jump_t=False] [arg=14,], POP_JUMP_IF_FALSE to 534 [jump_t=False] [arg=7,], LOAD_CONST 0 [jump_t=False] [arg=0,], LOAD_CONST ('urlparse' [jump_t=False] [arg=14,], IMPORT_NAME urllib.parse [jump_t=False] [arg=16,], IMPORT_FROM urlparse [jump_t=False] [arg=17,], STORE_NAME urlparse [jump_t=False] [arg=17,], POP_TOP [jump_t=False] [arg=None,], JUMP_FORWARD to 570 [jump_t=False] [arg=18,], PUSH_NULL [jump_t=True] [arg=None,], LOAD_NAME __jac_import__ [jump_t=False] [arg=4,], LOAD_CONST 'urllib.parse' [jump_t=False] [arg=15,], LOAD_NAME __file__ [jump_t=False] [arg=9,], LOAD_CONST 'py' [jump_t=False] [arg=5,], LOAD_CONST False [jump_t=False] [arg=6,], LOAD_CONST None [jump_t=False] [arg=3,], LOAD_CONST 'urlparse' [jump_t=False] [arg=16,], LOAD_CONST None [jump_t=False] [arg=3,], BUILD_MAP 1 [jump_t=False] [arg=1,], KW_NAMES ('target' [jump_t=False] [arg=7,], CALL 6 [jump_t=False] [arg=6,], UNPACK_SEQUENCE 1 [jump_t=False] [arg=1,], STORE_NAME urlparse [jump_t=False] [arg=17,]]
import:py optparse ;  # bb26 → bb27 [exec=1, time=0.0076s] [mem=57284 bytes] | bb24 → bb26, bb25 [exec=1, time=0.0007s] [mem=57284 bytes] | bb25 → bb27 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME _jac_typ [jump_t=True] [arg=6,], LOAD_ATTR TYPE_CHECKING [jump_t=False] [arg=14,], POP_JUMP_IF_FALSE to 604 [jump_t=False] [arg=5,], LOAD_CONST 0 [jump_t=False] [arg=0,], LOAD_CONST None [jump_t=False] [arg=3,], IMPORT_NAME optparse [jump_t=False] [arg=18,], STORE_NAME optparse [jump_t=False] [arg=18,], JUMP_FORWARD to 636 [jump_t=False] [arg=16,], PUSH_NULL [jump_t=True] [arg=None,], LOAD_NAME __jac_import__ [jump_t=False] [arg=4,], LOAD_CONST 'optparse' [jump_t=False] [arg=17,], LOAD_NAME __file__ [jump_t=False] [arg=9,], LOAD_CONST 'py' [jump_t=False] [arg=5,], LOAD_CONST False [jump_t=False] [arg=6,], LOAD_CONST None [jump_t=False] [arg=3,], BUILD_MAP 0 [jump_t=False] [arg=0,], KW_NAMES ('target' [jump_t=False] [arg=7,], CALL 6 [jump_t=False] [arg=6,], UNPACK_SEQUENCE 1 [jump_t=False] [arg=1,], STORE_NAME optparse [jump_t=False] [arg=18,]]
import:py hashlib ;  # bb28 → bb30 [exec=0, time=0.0000s] [mem=57284 bytes] | bb27 → bb29, bb28 [exec=1, time=0.0007s] [mem=57284 bytes] | bb29 → bb30 [exec=1, time=0.0037s] [mem=57284 bytes]
        #   [LOAD_NAME _jac_typ [jump_t=True] [arg=6,], LOAD_ATTR TYPE_CHECKING [jump_t=False] [arg=14,], POP_JUMP_IF_FALSE to 670 [jump_t=False] [arg=5,], LOAD_CONST 0 [jump_t=False] [arg=0,], LOAD_CONST None [jump_t=False] [arg=3,], IMPORT_NAME hashlib [jump_t=False] [arg=19,], STORE_NAME hashlib [jump_t=False] [arg=19,], JUMP_FORWARD to 702 [jump_t=False] [arg=16,], PUSH_NULL [jump_t=True] [arg=None,], LOAD_NAME __jac_import__ [jump_t=False] [arg=4,], LOAD_CONST 'hashlib' [jump_t=False] [arg=18,], LOAD_NAME __file__ [jump_t=False] [arg=9,], LOAD_CONST 'py' [jump_t=False] [arg=5,], LOAD_CONST False [jump_t=False] [arg=6,], LOAD_CONST None [jump_t=False] [arg=3,], BUILD_MAP 0 [jump_t=False] [arg=0,], KW_NAMES ('target' [jump_t=False] [arg=7,], CALL 6 [jump_t=False] [arg=6,], UNPACK_SEQUENCE 1 [jump_t=False] [arg=1,], STORE_NAME hashlib [jump_t=False] [arg=19,]]
import:py from cgi, escape ;  # bb31 → bb33 [exec=0, time=0.0000s] [mem=57284 bytes] | bb32 → bb33 [exec=1, time=0.0000s] [mem=57284 bytes] | bb30 → bb32, bb31 [exec=1, time=0.0007s] [mem=57284 bytes]
        #   [LOAD_NAME _jac_typ [jump_t=True] [arg=6,], LOAD_ATTR TYPE_CHECKING [jump_t=False] [arg=14,], POP_JUMP_IF_FALSE to 740 [jump_t=False] [arg=7,], LOAD_CONST 0 [jump_t=False] [arg=0,], LOAD_CONST ('escape' [jump_t=False] [arg=19,], IMPORT_NAME cgi [jump_t=False] [arg=20,], IMPORT_FROM escape [jump_t=False] [arg=21,], STORE_NAME escape [jump_t=False] [arg=21,], POP_TOP [jump_t=False] [arg=None,], JUMP_FORWARD to 776 [jump_t=False] [arg=18,], PUSH_NULL [jump_t=True] [arg=None,], LOAD_NAME __jac_import__ [jump_t=False] [arg=4,], LOAD_CONST 'cgi' [jump_t=False] [arg=20,], LOAD_NAME __file__ [jump_t=False] [arg=9,], LOAD_CONST 'py' [jump_t=False] [arg=5,], LOAD_CONST False [jump_t=False] [arg=6,], LOAD_CONST None [jump_t=False] [arg=3,], LOAD_CONST 'escape' [jump_t=False] [arg=21,], LOAD_CONST None [jump_t=False] [arg=3,], BUILD_MAP 1 [jump_t=False] [arg=1,], KW_NAMES ('target' [jump_t=False] [arg=7,], CALL 6 [jump_t=False] [arg=6,], UNPACK_SEQUENCE 1 [jump_t=False] [arg=1,], STORE_NAME escape [jump_t=False] [arg=21,]]
import:py from queue, Queue, Empty as QueueEmpty ;  # bb34 → bb36 [exec=0, time=0.0000s] [mem=57284 bytes] | bb33 → bb35, bb34 [exec=0, time=0.0000s] [mem=57284 bytes] | bb35 → bb36 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME _jac_typ [jump_t=True] [arg=6,], LOAD_ATTR TYPE_CHECKING [jump_t=False] [arg=14,], POP_JUMP_IF_FALSE to 818 [jump_t=False] [arg=9,], LOAD_CONST 0 [jump_t=False] [arg=0,], LOAD_CONST ('Queue' [jump_t=False] [arg=22,], IMPORT_NAME queue [jump_t=False] [arg=22,], IMPORT_FROM Queue [jump_t=False] [arg=23,], STORE_NAME Queue [jump_t=False] [arg=23,], IMPORT_FROM Empty [jump_t=False] [arg=24,], STORE_NAME QueueEmpty [jump_t=False] [arg=25,], POP_TOP [jump_t=False] [arg=None,], JUMP_FORWARD to 858 [jump_t=False] [arg=20,], PUSH_NULL [jump_t=True] [arg=None,], LOAD_NAME __jac_import__ [jump_t=False] [arg=4,], LOAD_CONST 'queue' [jump_t=False] [arg=23,], LOAD_NAME __file__ [jump_t=False] [arg=9,], LOAD_CONST 'py' [jump_t=False] [arg=5,], LOAD_CONST False [jump_t=False] [arg=6,], LOAD_CONST None [jump_t=False] [arg=3,], LOAD_CONST None [jump_t=False] [arg=3,], LOAD_CONST 'QueueEmpty' [jump_t=False] [arg=24,], LOAD_CONST ('Queue' [jump_t=False] [arg=22,], BUILD_CONST_KEY_MAP 2 [jump_t=False] [arg=2,], KW_NAMES ('target' [jump_t=False] [arg=7,], CALL 6 [jump_t=False] [arg=6,], UNPACK_SEQUENCE 2 [jump_t=False] [arg=2,], STORE_NAME Queue [jump_t=False] [arg=23,], STORE_NAME QueueEmpty [jump_t=False] [arg=25,]]
import:py from bs4, BeautifulSoup ;  # bb38 → bb39 [exec=0, time=0.0000s] [mem=57284 bytes] | bb36 → bb38, bb37 [exec=0, time=0.0000s] [mem=57284 bytes] | bb37 → bb39 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME _jac_typ [jump_t=True] [arg=6,], LOAD_ATTR TYPE_CHECKING [jump_t=False] [arg=14,], POP_JUMP_IF_FALSE to 896 [jump_t=False] [arg=7,], LOAD_CONST 0 [jump_t=False] [arg=0,], LOAD_CONST ('BeautifulSoup' [jump_t=False] [arg=25,], IMPORT_NAME bs4 [jump_t=False] [arg=26,], IMPORT_FROM BeautifulSoup [jump_t=False] [arg=27,], STORE_NAME BeautifulSoup [jump_t=False] [arg=27,], POP_TOP [jump_t=False] [arg=None,], JUMP_FORWARD to 932 [jump_t=False] [arg=18,], PUSH_NULL [jump_t=True] [arg=None,], LOAD_NAME __jac_import__ [jump_t=False] [arg=4,], LOAD_CONST 'bs4' [jump_t=False] [arg=26,], LOAD_NAME __file__ [jump_t=False] [arg=9,], LOAD_CONST 'py' [jump_t=False] [arg=5,], LOAD_CONST False [jump_t=False] [arg=6,], LOAD_CONST None [jump_t=False] [arg=3,], LOAD_CONST 'BeautifulSoup' [jump_t=False] [arg=27,], LOAD_CONST None [jump_t=False] [arg=3,], BUILD_MAP 1 [jump_t=False] [arg=1,], KW_NAMES ('target' [jump_t=False] [arg=7,], CALL 6 [jump_t=False] [arg=6,], UNPACK_SEQUENCE 1 [jump_t=False] [arg=1,], STORE_NAME BeautifulSoup [jump_t=False] [arg=27,]]

with entry {
    if __name__ == '__main__' {  # bb39 → bb103, bb40 [exec=0, time=0.0000s] [mem=57284 bytes] | bb103 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME __name__ [jump_t=True] [arg=28,], LOAD_CONST '__main__' [jump_t=False] [arg=28,], COMPARE_OP == [jump_t=False] [arg=40,], EXTENDED_ARG 7 [jump_t=False] [arg=7,], POP_JUMP_IF_FALSE to 4916 [jump_t=False] [arg=1986,], RETURN_CONST None [jump_t=True] [arg=3,], SWAP 2 [jump_t=False] [arg=2,], POP_TOP [jump_t=False] [arg=None,]]

        __version__ = '0.2';      # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '0.2' [jump_t=False] [arg=29,], STORE_NAME __version__ [jump_t=False] [arg=29,]]

        AGENT = ('WebCrawler/%s' % __version__);  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'WebCrawler/%s' [jump_t=False] [arg=30,], LOAD_NAME __version__ [jump_t=False] [arg=29,], BINARY_OP % [jump_t=False] [arg=6,], STORE_NAME AGENT [jump_t=False] [arg=30,]]
        

        USAGE = '%prog [options] <url>';      # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '%prog [options] <url>' [jump_t=False] [arg=31,], STORE_NAME USAGE [jump_t=False] [arg=31,]]

        VERSION = ('%prog v' + __version__);  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '%prog v' [jump_t=False] [arg=32,], LOAD_NAME __version__ [jump_t=False] [arg=29,], BINARY_OP + [jump_t=False] [arg=0,], STORE_NAME VERSION [jump_t=False] [arg=32,]]
        

        parser = optparse.OptionParser(usage=USAGE, version=VERSION);      # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME optparse [jump_t=False] [arg=18,], LOAD_ATTR OptionParser [jump_t=False] [arg=66,], LOAD_NAME USAGE [jump_t=False] [arg=31,], LOAD_NAME VERSION [jump_t=False] [arg=32,], KW_NAMES ('usage' [jump_t=False] [arg=33,], CALL 2 [jump_t=False] [arg=2,], STORE_NAME parser [jump_t=False] [arg=34,]]

        parser.add_option(  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME parser [jump_t=False] [arg=34,], LOAD_ATTR NULL|self + add_option [jump_t=False] [arg=71,], KW_NAMES ('action' [jump_t=False] [arg=39,], CALL 6 [jump_t=False] [arg=6,], POP_TOP [jump_t=False] [arg=None,]]
            '-q',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '-q' [jump_t=False] [arg=34,]]
            '--quiet',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '--quiet' [jump_t=False] [arg=35,]]
            action='store_true',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'store_true' [jump_t=False] [arg=36,]]
            default=False,  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST False [jump_t=False] [arg=6,]]
            dest='quiet',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'quiet' [jump_t=False] [arg=37,]]
            help='Enable quiet mode'  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'Enable quiet mode' [jump_t=False] [arg=38,]]
        ) ;    

        parser.add_option(  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME parser [jump_t=False] [arg=34,], LOAD_ATTR NULL|self + add_option [jump_t=False] [arg=71,], KW_NAMES ('action' [jump_t=False] [arg=39,], CALL 6 [jump_t=False] [arg=6,], POP_TOP [jump_t=False] [arg=None,]]
            '-l',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '-l' [jump_t=False] [arg=40,]]
            '--links',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '--links' [jump_t=False] [arg=41,]]
            action='store_true',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'store_true' [jump_t=False] [arg=36,]]
            default=False,  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST False [jump_t=False] [arg=6,]]
            dest='links',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'links' [jump_t=False] [arg=42,]]
            help='Get links for specified url only'  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'Get links for specified url only' [jump_t=False] [arg=43,]]
        ) ;    

        parser.add_option(  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME parser [jump_t=False] [arg=34,], LOAD_ATTR NULL|self + add_option [jump_t=False] [arg=71,], KW_NAMES ('action' [jump_t=False] [arg=51,], CALL 7 [jump_t=False] [arg=7,], POP_TOP [jump_t=False] [arg=None,]]
            '-d',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '-d' [jump_t=False] [arg=44,]]
            '--depth',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '--depth' [jump_t=False] [arg=45,]]
            action='store',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'store' [jump_t=False] [arg=46,]]
            type='int',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'int' [jump_t=False] [arg=47,]]
            default=30,  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 30 [jump_t=False] [arg=48,]]
            dest='depth_limit',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'depth_limit' [jump_t=False] [arg=49,]]
            help='Maximum depth to traverse'  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'Maximum depth to traverse' [jump_t=False] [arg=50,]]
        ) ;    

        parser.add_option(  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME parser [jump_t=False] [arg=34,], LOAD_ATTR NULL|self + add_option [jump_t=False] [arg=71,], KW_NAMES ('action' [jump_t=False] [arg=57,], CALL 6 [jump_t=False] [arg=6,], POP_TOP [jump_t=False] [arg=None,]]
            '-c',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '-c' [jump_t=False] [arg=52,]]
            '--confine',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '--confine' [jump_t=False] [arg=53,]]
            action='store',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'store' [jump_t=False] [arg=46,]]
            type='string',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'string' [jump_t=False] [arg=54,]]
            dest='confine',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'confine' [jump_t=False] [arg=55,]]
            help='Confine crawl to specified prefix'  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'Confine crawl to specified prefix' [jump_t=False] [arg=56,]]
        ) ;    

        parser.add_option(  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME parser [jump_t=False] [arg=34,], LOAD_ATTR NULL|self + add_option [jump_t=False] [arg=71,], KW_NAMES ('action' [jump_t=False] [arg=63,], CALL 7 [jump_t=False] [arg=7,], POP_TOP [jump_t=False] [arg=None,]]
            '-x',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '-x' [jump_t=False] [arg=58,]]
            '--exclude',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '--exclude' [jump_t=False] [arg=59,]]
            action='append',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'append' [jump_t=False] [arg=60,]]
            type='string',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'string' [jump_t=False] [arg=54,]]
            dest='exclude',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'exclude' [jump_t=False] [arg=61,]]
            default=[],  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [BUILD_LIST 0 [jump_t=False] [arg=0,]]
            help='Exclude URLs by prefix'  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'Exclude URLs by prefix' [jump_t=False] [arg=62,]]
        ) ;    

        parser.add_option(  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME parser [jump_t=False] [arg=34,], LOAD_ATTR NULL|self + add_option [jump_t=False] [arg=71,], KW_NAMES ('action' [jump_t=False] [arg=39,], CALL 6 [jump_t=False] [arg=6,], POP_TOP [jump_t=False] [arg=None,]]
            '-L',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '-L' [jump_t=False] [arg=64,]]
            '--show-links',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '--show-links' [jump_t=False] [arg=65,]]
            action='store_true',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'store_true' [jump_t=False] [arg=36,]]
            default=False,  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST False [jump_t=False] [arg=6,]]
            dest='out_links',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'out_links' [jump_t=False] [arg=66,]]
            help='Output links found'  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'Output links found' [jump_t=False] [arg=67,]]
        ) ;    

        parser.add_option(  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME parser [jump_t=False] [arg=34,], LOAD_ATTR NULL|self + add_option [jump_t=False] [arg=71,], KW_NAMES ('action' [jump_t=False] [arg=39,], CALL 6 [jump_t=False] [arg=6,], POP_TOP [jump_t=False] [arg=None,]]
            '-u',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '-u' [jump_t=False] [arg=68,]]
            '--show-urls',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '--show-urls' [jump_t=False] [arg=69,]]
            action='store_true',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'store_true' [jump_t=False] [arg=36,]]
            default=False,  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST False [jump_t=False] [arg=6,]]
            dest='out_urls',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'out_urls' [jump_t=False] [arg=70,]]
            help='Output URLs found'  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'Output URLs found' [jump_t=False] [arg=71,]]
        ) ;    

        parser.add_option(  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME parser [jump_t=False] [arg=34,], LOAD_ATTR NULL|self + add_option [jump_t=False] [arg=71,], KW_NAMES ('action' [jump_t=False] [arg=39,], CALL 6 [jump_t=False] [arg=6,], POP_TOP [jump_t=False] [arg=None,]]
            '-D',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '-D' [jump_t=False] [arg=72,]]
            '--dot',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '--dot' [jump_t=False] [arg=73,]]
            action='store_true',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'store_true' [jump_t=False] [arg=36,]]
            default=False,  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST False [jump_t=False] [arg=6,]]
            dest='out_dot',  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'out_dot' [jump_t=False] [arg=74,]]
            help='Output Graphviz dot file'  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'Output Graphviz dot file' [jump_t=False] [arg=75,]]
        ) ;    

        (opts, args) = parser.parse_args();      # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME parser [jump_t=False] [arg=34,], LOAD_ATTR NULL|self + parse_args [jump_t=False] [arg=73,], CALL 0 [jump_t=False] [arg=0,], UNPACK_SEQUENCE 2 [jump_t=False] [arg=2,], STORE_NAME opts [jump_t=False] [arg=37,], STORE_NAME args [jump_t=False] [arg=38,]]
        if len(args) < 1 {  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME len [jump_t=False] [arg=39,], LOAD_NAME args [jump_t=False] [arg=38,], CALL 1 [jump_t=False] [arg=1,], LOAD_CONST 1 [jump_t=False] [arg=76,], COMPARE_OP < [jump_t=False] [arg=2,], POP_JUMP_IF_FALSE to 1534 [jump_t=False] [arg=45,]]

            parser.print_help(sys.stderr) ;      # bb41 → bb42 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME parser [jump_t=False] [arg=34,], LOAD_ATTR NULL|self + print_help [jump_t=False] [arg=81,], LOAD_NAME sys [jump_t=False] [arg=10,], LOAD_ATTR stderr [jump_t=False] [arg=82,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]
            sys.exit(1) ;  # bb41 → bb42 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME sys [jump_t=False] [arg=10,], LOAD_ATTR exit [jump_t=False] [arg=84,], LOAD_CONST 1 [jump_t=False] [arg=76,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]
        }
        
        if opts.out_links and opts.out_urls {  # bb42 → bb45, bb43 [exec=0, time=0.0000s] [mem=57284 bytes] | bb43 → bb45, bb44 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME opts [jump_t=True] [arg=37,], LOAD_ATTR out_links [jump_t=False] [arg=86,], POP_JUMP_IF_FALSE to 1670 [jump_t=False] [arg=56,], LOAD_NAME opts [jump_t=False] [arg=37,], LOAD_ATTR out_urls [jump_t=False] [arg=88,], POP_JUMP_IF_FALSE to 1670 [jump_t=False] [arg=44,]]

            parser.print_help(sys.stderr) ;      # bb44 → bb45 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME parser [jump_t=False] [arg=34,], LOAD_ATTR NULL|self + print_help [jump_t=False] [arg=81,], LOAD_NAME sys [jump_t=False] [arg=10,], LOAD_ATTR stderr [jump_t=False] [arg=82,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]
            parser.error('options -L and -u are mutually exclusive') ;  # bb44 → bb45 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME parser [jump_t=False] [arg=34,], LOAD_ATTR NULL|self + error [jump_t=False] [arg=91,], LOAD_CONST 'options -L and -u are mutually exclusive' [jump_t=False] [arg=77,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]
        }
        

        url = args[0];      # bb45 → bb56, bb46 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME args [jump_t=True] [arg=38,], LOAD_CONST 0 [jump_t=False] [arg=0,], BINARY_SUBSCR [jump_t=False] [arg=None,], STORE_NAME url [jump_t=False] [arg=46,]]

        node_alias = {};  # bb45 → bb56, bb46 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [BUILD_MAP 0 [jump_t=False] [arg=0,], STORE_NAME node_alias [jump_t=False] [arg=47,]]
        
        if opts.links {  # bb45 → bb56, bb46 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME opts [jump_t=False] [arg=37,], LOAD_ATTR links [jump_t=False] [arg=96,], EXTENDED_ARG 1 [jump_t=False] [arg=1,], POP_JUMP_IF_FALSE to 2320 [jump_t=False] [arg=305,]]

            request = urllib.request.Request(url);      # bb46 → bb55, bb47 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME urllib [jump_t=False] [arg=13,], LOAD_ATTR request [jump_t=False] [arg=98,], LOAD_ATTR NULL|self + Request [jump_t=False] [arg=101,], LOAD_NAME url [jump_t=False] [arg=46,], CALL 1 [jump_t=False] [arg=1,], STORE_NAME request [jump_t=False] [arg=49,]]
            request.add_header('User-Agent', AGENT) ;      # bb46 → bb55, bb47 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME request [jump_t=False] [arg=49,], LOAD_ATTR NULL|self + add_header [jump_t=False] [arg=103,], LOAD_CONST 'User-Agent' [jump_t=False] [arg=78,], LOAD_NAME AGENT [jump_t=False] [arg=30,], CALL 2 [jump_t=False] [arg=2,], POP_TOP [jump_t=False] [arg=None,]]
            handle = urllib.request.build_opener();      # bb46 → bb55, bb47 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME urllib [jump_t=False] [arg=13,], LOAD_ATTR request [jump_t=False] [arg=98,], LOAD_ATTR NULL|self + build_opener [jump_t=False] [arg=105,], CALL 0 [jump_t=False] [arg=0,], STORE_NAME handle [jump_t=False] [arg=53,]]
            data = handle.open(request);      # bb46 → bb55, bb47 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME handle [jump_t=False] [arg=53,], LOAD_ATTR NULL|self + open [jump_t=False] [arg=109,], LOAD_NAME request [jump_t=False] [arg=49,], CALL 1 [jump_t=False] [arg=1,], STORE_NAME data [jump_t=False] [arg=55,]]
            mime_type = data.info().get_content_type();      # bb46 → bb55, bb47 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME data [jump_t=False] [arg=55,], LOAD_ATTR NULL|self + info [jump_t=False] [arg=113,], CALL 0 [jump_t=False] [arg=0,], LOAD_ATTR NULL|self + get_content_type [jump_t=False] [arg=115,], CALL 0 [jump_t=False] [arg=0,], STORE_NAME mime_type [jump_t=False] [arg=58,]]
            if mime_type == 'text/html' {  # bb46 → bb55, bb47 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME mime_type [jump_t=False] [arg=58,], LOAD_CONST 'text/html' [jump_t=False] [arg=79,], COMPARE_OP == [jump_t=False] [arg=40,], POP_JUMP_IF_FALSE to 2284 [jump_t=False] [arg=164,]]

                content = data.read().decode('utf-8', errors='replace');      # bb47 → bb48 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME data [jump_t=False] [arg=55,], LOAD_ATTR NULL|self + read [jump_t=False] [arg=119,], CALL 0 [jump_t=False] [arg=0,], LOAD_ATTR NULL|self + decode [jump_t=False] [arg=121,], LOAD_CONST 'utf-8' [jump_t=False] [arg=80,], LOAD_CONST 'replace' [jump_t=False] [arg=81,], KW_NAMES ('errors' [jump_t=False] [arg=82,], CALL 2 [jump_t=False] [arg=2,], STORE_NAME content [jump_t=False] [arg=61,]]

                soup = BeautifulSoup(content, 'html.parser');      # bb47 → bb48 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME BeautifulSoup [jump_t=False] [arg=27,], LOAD_NAME content [jump_t=False] [arg=61,], LOAD_CONST 'html.parser' [jump_t=False] [arg=83,], CALL 2 [jump_t=False] [arg=2,], STORE_NAME soup [jump_t=False] [arg=62,]]

                tags = soup('a');      # bb47 → bb48 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME soup [jump_t=False] [arg=62,], LOAD_CONST 'a' [jump_t=False] [arg=84,], CALL 1 [jump_t=False] [arg=1,], STORE_NAME tags [jump_t=False] [arg=63,]]

                out_urls = [];  # bb47 → bb48 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [BUILD_LIST 0 [jump_t=False] [arg=0,], STORE_NAME out_urls [jump_t=False] [arg=44,]]
                
                for tag in tags {  # bb52 → bb53 [exec=0, time=0.0000s] [mem=57284 bytes] | bb47 → bb48 [exec=0, time=0.0000s] [mem=57284 bytes] | bb48 → bb52, bb48 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME tags [jump_t=False] [arg=63,], GET_ITER [jump_t=False] [arg=None,], FOR_ITER to 2224 [jump_t=True] [arg=78,], STORE_NAME tag [jump_t=False] [arg=64,], END_FOR [jump_t=True] [arg=None,]]

                    href = tag.get('href');      # bb48 → bb52, bb48 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME tag [jump_t=False] [arg=64,], LOAD_ATTR NULL|self + get [jump_t=False] [arg=131,], LOAD_CONST 'href' [jump_t=False] [arg=85,], CALL 1 [jump_t=False] [arg=1,], STORE_NAME href [jump_t=False] [arg=66,]]
                    if href is not None {  # bb48 → bb52, bb48 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME href [jump_t=False] [arg=66,], POP_JUMP_IF_NOT_NONE to 2110 [jump_t=False] [arg=1,], JUMP_BACKWARD to 2064 [jump_t=False] [arg=23,]]

                        url_joined = urllib.parse.urljoin(url, escape(href));      # bb49 → bb51, bb50 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME urllib [jump_t=True] [arg=13,], LOAD_ATTR parse [jump_t=False] [arg=134,], LOAD_ATTR NULL|self + urljoin [jump_t=False] [arg=137,], LOAD_NAME url [jump_t=False] [arg=46,], PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME escape [jump_t=False] [arg=21,], LOAD_NAME href [jump_t=False] [arg=66,], CALL 1 [jump_t=False] [arg=1,], CALL 2 [jump_t=False] [arg=2,], STORE_NAME url_joined [jump_t=False] [arg=69,]]
                        if url_joined not in out_urls {  # bb49 → bb51, bb50 [exec=0, time=0.0000s] [mem=57284 bytes] | bb50 → bb48 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME url_joined [jump_t=False] [arg=69,], LOAD_NAME out_urls [jump_t=False] [arg=44,], CONTAINS_OP 1 [jump_t=False] [arg=1,], POP_JUMP_IF_TRUE to 2188 [jump_t=False] [arg=1,], JUMP_BACKWARD to 2064 [jump_t=False] [arg=62,]]

                            out_urls.append(url_joined) ;  # bb51 → bb48 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME out_urls [jump_t=True] [arg=44,], LOAD_ATTR NULL|self + append [jump_t=False] [arg=141,], LOAD_NAME url_joined [jump_t=False] [arg=69,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,], JUMP_BACKWARD to 2064 [jump_t=False] [arg=80,]]
                        }
                    }
                }
                
                for (i, url) in enumerate(out_urls) {  # bb53 → bb54, bb53 [exec=0, time=0.0000s] [mem=57284 bytes] | bb52 → bb53 [exec=0, time=0.0000s] [mem=57284 bytes] | bb54 → bb55 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME enumerate [jump_t=False] [arg=71,], LOAD_NAME out_urls [jump_t=False] [arg=44,], CALL 1 [jump_t=False] [arg=1,], GET_ITER [jump_t=False] [arg=None,], FOR_ITER to 2282 [jump_t=True] [arg=18,], UNPACK_SEQUENCE 2 [jump_t=False] [arg=2,], STORE_NAME i [jump_t=False] [arg=72,], STORE_NAME url [jump_t=False] [arg=46,], END_FOR [jump_t=True] [arg=None,]]

                    print(('%d. %s' % (i, url))) ;  # bb53 → bb54, bb53 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME print [jump_t=False] [arg=73,], LOAD_CONST '%d. %s' [jump_t=False] [arg=86,], LOAD_NAME i [jump_t=False] [arg=72,], LOAD_NAME url [jump_t=False] [arg=46,], BUILD_TUPLE 2 [jump_t=False] [arg=2,], BINARY_OP % [jump_t=False] [arg=6,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,], JUMP_BACKWARD to 2242 [jump_t=False] [arg=20,]]
                }
            }

            
            sys.exit(0) ;  # bb55 → bb56 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=True] [arg=None,], LOAD_NAME sys [jump_t=False] [arg=10,], LOAD_ATTR exit [jump_t=False] [arg=84,], LOAD_CONST 0 [jump_t=False] [arg=0,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]
        }
        

        depth_limit = opts.depth_limit;      # bb56 → bb57 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME opts [jump_t=True] [arg=37,], LOAD_ATTR depth_limit [jump_t=False] [arg=148,], STORE_NAME depth_limit [jump_t=False] [arg=74,]]

        confine_prefix = opts.confine;      # bb56 → bb57 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME opts [jump_t=False] [arg=37,], LOAD_ATTR confine [jump_t=False] [arg=150,], STORE_NAME confine_prefix [jump_t=False] [arg=76,]]

        exclude = opts.exclude;      # bb56 → bb57 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME opts [jump_t=False] [arg=37,], LOAD_ATTR exclude [jump_t=False] [arg=154,], STORE_NAME exclude [jump_t=False] [arg=77,]]

        sTime = time.time();      # bb56 → bb57 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME time [jump_t=False] [arg=11,], LOAD_ATTR time [jump_t=False] [arg=22,], CALL 0 [jump_t=False] [arg=0,], STORE_NAME sTime [jump_t=False] [arg=78,]]

        print(  # bb56 → bb57 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME print [jump_t=False] [arg=73,], CALL 2 [jump_t=False] [arg=2,], POP_TOP [jump_t=False] [arg=None,]]
            sys.stderr,  # bb56 → bb57 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME sys [jump_t=False] [arg=10,], LOAD_ATTR stderr [jump_t=False] [arg=82,]]
            ('Crawling %s (Max Depth: %d)' % (url, depth_limit))  # bb56 → bb57 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'Crawling %s (Max Depth: %d)' [jump_t=False] [arg=87,], LOAD_NAME url [jump_t=False] [arg=46,], LOAD_NAME depth_limit [jump_t=False] [arg=74,], BUILD_TUPLE 2 [jump_t=False] [arg=2,], BINARY_OP % [jump_t=False] [arg=6,]]
        ) ;
        

        host = urlparse(url)[1];      # bb56 → bb57 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME urlparse [jump_t=False] [arg=17,], LOAD_NAME url [jump_t=False] [arg=46,], CALL 1 [jump_t=False] [arg=1,], LOAD_CONST 1 [jump_t=False] [arg=76,], BINARY_SUBSCR [jump_t=False] [arg=None,], STORE_NAME host [jump_t=False] [arg=79,]]

        urls_seen = <>set();      # bb56 → bb57 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME set [jump_t=False] [arg=80,], CALL 0 [jump_t=False] [arg=0,], STORE_NAME urls_seen [jump_t=False] [arg=81,]]

        urls_remembered = <>set();      # bb56 → bb57 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME set [jump_t=False] [arg=80,], CALL 0 [jump_t=False] [arg=0,], STORE_NAME urls_remembered [jump_t=False] [arg=82,]]

        visited_links = <>set();      # bb56 → bb57 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME set [jump_t=False] [arg=80,], CALL 0 [jump_t=False] [arg=0,], STORE_NAME visited_links [jump_t=False] [arg=83,]]

        links_remembered = <>set();      # bb56 → bb57 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME set [jump_t=False] [arg=80,], CALL 0 [jump_t=False] [arg=0,], STORE_NAME links_remembered [jump_t=False] [arg=84,]]

        num_links = 0;      # bb56 → bb57 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 0 [jump_t=False] [arg=0,], STORE_NAME num_links [jump_t=False] [arg=85,]]

        num_followed = 0;      # bb56 → bb57 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 0 [jump_t=False] [arg=0,], STORE_NAME num_followed [jump_t=False] [arg=86,]]

        q = Queue();      # bb56 → bb57 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME Queue [jump_t=False] [arg=23,], CALL 0 [jump_t=False] [arg=0,], STORE_NAME q [jump_t=False] [arg=87,]]

        q.put((url, 0)) ;      # bb56 → bb57 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME q [jump_t=False] [arg=87,], LOAD_ATTR NULL|self + put [jump_t=False] [arg=177,], LOAD_NAME url [jump_t=False] [arg=46,], LOAD_CONST 0 [jump_t=False] [arg=0,], BUILD_TUPLE 2 [jump_t=False] [arg=2,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]
        class Link {  # bb56 → bb57 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_BUILD_CLASS [jump_t=False] [arg=None,], LOAD_CONST <code object Link at 0x7f6853ff0030 [jump_t=False] [arg=88,], MAKE_FUNCTION 0 [jump_t=False] [arg=0,], LOAD_CONST 'Link' [jump_t=False] [arg=89,], CALL 2 [jump_t=False] [arg=2,], STORE_NAME Link [jump_t=False] [arg=89,]]
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

        
        while not q.empty() {  # bb87 → bb58 [exec=0, time=0.0000s] [mem=57284 bytes] | bb86 → bb88, bb87 [exec=0, time=0.0000s] [mem=57284 bytes] | bb57 → bb88, bb58 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME q [jump_t=True] [arg=87,], LOAD_ATTR NULL|self + empty [jump_t=False] [arg=181,], CALL 0 [jump_t=False] [arg=0,], EXTENDED_ARG 2 [jump_t=False] [arg=2,], POP_JUMP_IF_TRUE to 3940 [jump_t=False] [arg=637,], LOAD_NAME q [jump_t=True] [arg=87,], LOAD_ATTR NULL|self + empty [jump_t=False] [arg=181,], CALL 0 [jump_t=False] [arg=0,], POP_JUMP_IF_TRUE to 3940 [jump_t=False] [arg=2,], EXTENDED_ARG 2 [jump_t=False] [arg=2,], JUMP_BACKWARD to 2666 [jump_t=False] [arg=637,]]

            (this_url, depth) = q.get();      # bb58 → bb60, bb59 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME q [jump_t=True] [arg=87,], LOAD_ATTR NULL|self + get [jump_t=False] [arg=131,], CALL 0 [jump_t=False] [arg=0,], UNPACK_SEQUENCE 2 [jump_t=False] [arg=2,], STORE_NAME this_url [jump_t=False] [arg=91,], STORE_NAME depth [jump_t=False] [arg=92,]]
            if depth > depth_limit {  # bb58 → bb60, bb59 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME depth [jump_t=False] [arg=92,], LOAD_NAME depth_limit [jump_t=False] [arg=74,], COMPARE_OP > [jump_t=False] [arg=68,], POP_JUMP_IF_FALSE to 2716 [jump_t=False] [arg=1,]]
                continue ;  # bb59 → bb57 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [JUMP_BACKWARD to 2632 [jump_t=False] [arg=42,]]
            }

            
            prefix_ok = confine_prefix is None  # bb61 → bb62 [exec=0, time=0.0000s] [mem=57284 bytes] | bb60 → bb62, bb61 [exec=0, time=0.0000s] [mem=57284 bytes] | bb62 → bb63 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME confine_prefix [jump_t=True] [arg=76,], LOAD_CONST None [jump_t=False] [arg=3,], IS_OP 0 [jump_t=False] [arg=0,], COPY 1 [jump_t=False] [arg=1,], POP_JUMP_IF_TRUE to 2760 [jump_t=False] [arg=17,], POP_TOP [jump_t=False] [arg=None,], STORE_NAME prefix_ok [jump_t=True] [arg=94,]]
                or this_url.startswith(confine_prefix);      # bb61 → bb62 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME this_url [jump_t=False] [arg=91,], LOAD_ATTR NULL|self + startswith [jump_t=False] [arg=187,], LOAD_NAME confine_prefix [jump_t=False] [arg=76,], CALL 1 [jump_t=False] [arg=1,]]
            exclude_ok = all([not this_url.startswith(p) for p in exclude]);      # bb63 → bb64, bb63 [exec=0, time=0.0000s] [mem=57284 bytes] | bb103 [exec=0, time=0.0000s] [mem=57284 bytes] | bb62 → bb63 [exec=0, time=0.0000s] [mem=57284 bytes] | bb64 → bb68, bb65 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME all [jump_t=False] [arg=95,], LOAD_NAME exclude [jump_t=False] [arg=77,], GET_ITER [jump_t=False] [arg=None,], LOAD_FAST_AND_CLEAR p [jump_t=False] [arg=0,], SWAP 2 [jump_t=False] [arg=2,], BUILD_LIST 0 [jump_t=False] [arg=0,], SWAP 2 [jump_t=False] [arg=2,], FOR_ITER to 2822 [jump_t=True] [arg=20,], STORE_FAST p [jump_t=False] [arg=0,], LOAD_NAME this_url [jump_t=False] [arg=91,], LOAD_ATTR NULL|self + startswith [jump_t=False] [arg=187,], LOAD_FAST p [jump_t=False] [arg=0,], CALL 1 [jump_t=False] [arg=1,], UNARY_NOT [jump_t=False] [arg=None,], LIST_APPEND 2 [jump_t=False] [arg=2,], JUMP_BACKWARD to 2778 [jump_t=False] [arg=22,], END_FOR [jump_t=True] [arg=None,], SWAP 2 [jump_t=False] [arg=2,], STORE_FAST p [jump_t=False] [arg=0,], CALL 1 [jump_t=False] [arg=1,], STORE_NAME exclude_ok [jump_t=False] [arg=96,], SWAP 2 [jump_t=False] [arg=2,], STORE_FAST p [jump_t=False] [arg=0,]]
            not_visited = this_url not in visited_links;      # bb64 → bb68, bb65 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME this_url [jump_t=False] [arg=91,], LOAD_NAME visited_links [jump_t=False] [arg=83,], CONTAINS_OP 1 [jump_t=False] [arg=1,], STORE_NAME not_visited [jump_t=False] [arg=97,]]
            same_host = False;      # bb64 → bb68, bb65 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST False [jump_t=False] [arg=6,], STORE_NAME same_host [jump_t=False] [arg=98,]]
            host_from_url = urlparse(this_url)[1];      # bb64 → bb68, bb65 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME urlparse [jump_t=False] [arg=17,], LOAD_NAME this_url [jump_t=False] [arg=91,], CALL 1 [jump_t=False] [arg=1,], LOAD_CONST 1 [jump_t=False] [arg=76,], BINARY_SUBSCR [jump_t=False] [arg=None,], STORE_NAME host_from_url [jump_t=False] [arg=99,]]
            same_host = re.match(('.*%s' % host), host_from_url) is not None;      # bb64 → bb68, bb65 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME re [jump_t=False] [arg=8,], LOAD_ATTR match [jump_t=False] [arg=200,], LOAD_CONST '.*%s' [jump_t=False] [arg=90,], LOAD_NAME host [jump_t=False] [arg=79,], BINARY_OP % [jump_t=False] [arg=6,], LOAD_NAME host_from_url [jump_t=False] [arg=99,], CALL 2 [jump_t=False] [arg=2,], LOAD_CONST None [jump_t=False] [arg=3,], IS_OP 1 [jump_t=False] [arg=1,], STORE_NAME same_host [jump_t=False] [arg=98,]]
            if prefix_ok  # bb64 → bb68, bb65 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME prefix_ok [jump_t=False] [arg=94,], POP_JUMP_IF_FALSE to 2936 [jump_t=False] [arg=6,]]
                and exclude_ok  # bb65 → bb68, bb66 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME exclude_ok [jump_t=False] [arg=96,], POP_JUMP_IF_FALSE to 2936 [jump_t=False] [arg=4,]]
                and not_visited  # bb66 → bb68, bb67 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME not_visited [jump_t=False] [arg=97,], POP_JUMP_IF_FALSE to 2936 [jump_t=False] [arg=2,]]
                and same_host or depth == 0 {  # bb67 → bb69, bb68 [exec=0, time=0.0000s] [mem=57284 bytes] | bb68 → bb86, bb69 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME same_host [jump_t=False] [arg=98,], POP_JUMP_IF_TRUE to 2948 [jump_t=False] [arg=6,], LOAD_NAME depth [jump_t=True] [arg=92,], LOAD_CONST 0 [jump_t=False] [arg=0,], COMPARE_OP == [jump_t=False] [arg=40,], EXTENDED_ARG 1 [jump_t=False] [arg=1,], POP_JUMP_IF_FALSE to 3904 [jump_t=False] [arg=478,]]

                visited_links.add(this_url) ;      # bb69 → bb86, bb70 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME visited_links [jump_t=True] [arg=83,], LOAD_ATTR NULL|self + add [jump_t=False] [arg=203,], LOAD_NAME this_url [jump_t=False] [arg=91,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]
                num_followed += 1;      # bb69 → bb86, bb70 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME num_followed [jump_t=False] [arg=86,], LOAD_CONST 1 [jump_t=False] [arg=76,], BINARY_OP += [jump_t=False] [arg=13,], STORE_NAME num_followed [jump_t=False] [arg=86,]]
                request = urllib.request.Request(this_url);      # bb69 → bb86, bb70 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME urllib [jump_t=False] [arg=13,], LOAD_ATTR request [jump_t=False] [arg=98,], LOAD_ATTR NULL|self + Request [jump_t=False] [arg=101,], LOAD_NAME this_url [jump_t=False] [arg=91,], CALL 1 [jump_t=False] [arg=1,], STORE_NAME request [jump_t=False] [arg=49,]]
                request.add_header('User-Agent', AGENT) ;      # bb69 → bb86, bb70 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME request [jump_t=False] [arg=49,], LOAD_ATTR NULL|self + add_header [jump_t=False] [arg=103,], LOAD_CONST 'User-Agent' [jump_t=False] [arg=78,], LOAD_NAME AGENT [jump_t=False] [arg=30,], CALL 2 [jump_t=False] [arg=2,], POP_TOP [jump_t=False] [arg=None,]]
                handle = None;      # bb69 → bb86, bb70 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST None [jump_t=False] [arg=3,], STORE_NAME handle [jump_t=False] [arg=53,]]
                handle = urllib.request.build_opener();      # bb69 → bb86, bb70 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME urllib [jump_t=False] [arg=13,], LOAD_ATTR request [jump_t=False] [arg=98,], LOAD_ATTR NULL|self + build_opener [jump_t=False] [arg=105,], CALL 0 [jump_t=False] [arg=0,], STORE_NAME handle [jump_t=False] [arg=53,]]
                if handle {  # bb69 → bb86, bb70 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME handle [jump_t=False] [arg=53,], EXTENDED_ARG 1 [jump_t=False] [arg=1,], POP_JUMP_IF_FALSE to 3904 [jump_t=False] [arg=380,]]

                    data = None;      # bb70 → bb86, bb71 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST None [jump_t=False] [arg=3,], STORE_NAME data [jump_t=False] [arg=55,]]
                    data = handle.open(request);      # bb70 → bb86, bb71 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME handle [jump_t=False] [arg=53,], LOAD_ATTR NULL|self + open [jump_t=False] [arg=109,], LOAD_NAME request [jump_t=False] [arg=49,], CALL 1 [jump_t=False] [arg=1,], STORE_NAME data [jump_t=False] [arg=55,]]
                    if data {  # bb70 → bb86, bb71 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME data [jump_t=False] [arg=55,], EXTENDED_ARG 1 [jump_t=False] [arg=1,], POP_JUMP_IF_FALSE to 3904 [jump_t=False] [arg=358,]]

                        mime_type = data.info().get_content_type();      # bb71 → bb86, bb72 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME data [jump_t=False] [arg=55,], LOAD_ATTR NULL|self + info [jump_t=False] [arg=113,], CALL 0 [jump_t=False] [arg=0,], LOAD_ATTR NULL|self + get_content_type [jump_t=False] [arg=115,], CALL 0 [jump_t=False] [arg=0,], STORE_NAME mime_type [jump_t=False] [arg=58,]]
                        url = data.geturl();      # bb71 → bb86, bb72 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME data [jump_t=False] [arg=55,], LOAD_ATTR NULL|self + geturl [jump_t=False] [arg=205,], CALL 0 [jump_t=False] [arg=0,], STORE_NAME url [jump_t=False] [arg=46,]]
                        if mime_type == 'text/html' {  # bb71 → bb86, bb72 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME mime_type [jump_t=False] [arg=58,], LOAD_CONST 'text/html' [jump_t=False] [arg=79,], COMPARE_OP == [jump_t=False] [arg=40,], EXTENDED_ARG 1 [jump_t=False] [arg=1,], POP_JUMP_IF_FALSE to 3904 [jump_t=False] [arg=306,]]

                            content = data.read().decode('utf-8', errors='replace');      # bb72 → bb73 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME data [jump_t=False] [arg=55,], LOAD_ATTR NULL|self + read [jump_t=False] [arg=119,], CALL 0 [jump_t=False] [arg=0,], LOAD_ATTR NULL|self + decode [jump_t=False] [arg=121,], LOAD_CONST 'utf-8' [jump_t=False] [arg=80,], LOAD_CONST 'replace' [jump_t=False] [arg=81,], KW_NAMES ('errors' [jump_t=False] [arg=82,], CALL 2 [jump_t=False] [arg=2,], STORE_NAME content [jump_t=False] [arg=61,]]
                            soup = BeautifulSoup(content, 'html.parser');      # bb72 → bb73 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME BeautifulSoup [jump_t=False] [arg=27,], LOAD_NAME content [jump_t=False] [arg=61,], LOAD_CONST 'html.parser' [jump_t=False] [arg=83,], CALL 2 [jump_t=False] [arg=2,], STORE_NAME soup [jump_t=False] [arg=62,]]
                            tags = soup('a');      # bb72 → bb73 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME soup [jump_t=False] [arg=62,], LOAD_CONST 'a' [jump_t=False] [arg=84,], CALL 1 [jump_t=False] [arg=1,], STORE_NAME tags [jump_t=False] [arg=63,]]
                            for tag in tags {  # bb72 → bb73 [exec=0, time=0.0000s] [mem=57284 bytes] | bb73 → bb85, bb73 [exec=0, time=0.0000s] [mem=57284 bytes] | bb85 → bb86 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME tags [jump_t=False] [arg=63,], GET_ITER [jump_t=False] [arg=None,], FOR_ITER to 3902 [jump_t=True] [arg=251,], STORE_NAME tag [jump_t=False] [arg=64,], END_FOR [jump_t=True] [arg=None,]]

                                href = tag.get('href');      # bb73 → bb85, bb73 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME tag [jump_t=False] [arg=64,], LOAD_ATTR NULL|self + get [jump_t=False] [arg=131,], LOAD_CONST 'href' [jump_t=False] [arg=85,], CALL 1 [jump_t=False] [arg=1,], STORE_NAME href [jump_t=False] [arg=66,]]
                                if href is not None {  # bb73 → bb85, bb73 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME href [jump_t=False] [arg=66,], POP_JUMP_IF_NOT_NONE to 3442 [jump_t=False] [arg=1,], JUMP_BACKWARD to 3396 [jump_t=False] [arg=23,]]

                                    link_url = urllib.parse.urljoin(this_url, escape(href));      # bb74 → bb76, bb75 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME urllib [jump_t=True] [arg=13,], LOAD_ATTR parse [jump_t=False] [arg=134,], LOAD_ATTR NULL|self + urljoin [jump_t=False] [arg=137,], LOAD_NAME this_url [jump_t=False] [arg=91,], PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME escape [jump_t=False] [arg=21,], LOAD_NAME href [jump_t=False] [arg=66,], CALL 1 [jump_t=False] [arg=1,], CALL 2 [jump_t=False] [arg=2,], STORE_NAME link_url [jump_t=False] [arg=103,]]
                                    (base, frag) = urllib.parse.urldefrag(link_url);      # bb74 → bb76, bb75 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME urllib [jump_t=False] [arg=13,], LOAD_ATTR parse [jump_t=False] [arg=134,], LOAD_ATTR NULL|self + urldefrag [jump_t=False] [arg=209,], LOAD_NAME link_url [jump_t=False] [arg=103,], CALL 1 [jump_t=False] [arg=1,], UNPACK_SEQUENCE 2 [jump_t=False] [arg=2,], STORE_NAME base [jump_t=False] [arg=105,], STORE_NAME frag [jump_t=False] [arg=106,]]
                                    link_url = base;      # bb74 → bb76, bb75 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME base [jump_t=False] [arg=105,], STORE_NAME link_url [jump_t=False] [arg=103,]]
                                    if link_url not in urls_seen {  # bb74 → bb76, bb75 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME link_url [jump_t=False] [arg=103,], LOAD_NAME urls_seen [jump_t=False] [arg=81,], CONTAINS_OP 1 [jump_t=False] [arg=1,], POP_JUMP_IF_FALSE to 3660 [jump_t=False] [arg=39,]]

                                        q.put((link_url, (depth + 1))) ;  # bb75 → bb76 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME q [jump_t=False] [arg=87,], LOAD_ATTR NULL|self + put [jump_t=False] [arg=177,], LOAD_NAME link_url [jump_t=False] [arg=103,], LOAD_NAME depth [jump_t=False] [arg=92,], LOAD_CONST 1 [jump_t=False] [arg=76,], BINARY_OP + [jump_t=False] [arg=0,], BUILD_TUPLE 2 [jump_t=False] [arg=2,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]
                                        
                                        urls_seen.add(link_url) ;  # bb75 → bb76 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME urls_seen [jump_t=False] [arg=81,], LOAD_ATTR NULL|self + add [jump_t=False] [arg=203,], LOAD_NAME link_url [jump_t=False] [arg=103,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]
                                    }
                                    
                                    out_prefix_ok = confine_prefix is None  # bb78 → bb80, bb79 [exec=0, time=0.0000s] [mem=57284 bytes] | bb77 → bb78 [exec=0, time=0.0000s] [mem=57284 bytes] | bb76 → bb78, bb77 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME confine_prefix [jump_t=True] [arg=76,], LOAD_CONST None [jump_t=False] [arg=3,], IS_OP 0 [jump_t=False] [arg=0,], COPY 1 [jump_t=False] [arg=1,], POP_JUMP_IF_TRUE to 3704 [jump_t=False] [arg=17,], POP_TOP [jump_t=False] [arg=None,], STORE_NAME out_prefix_ok [jump_t=True] [arg=107,]]
                                        or link_url.startswith(confine_prefix);      # bb77 → bb78 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME link_url [jump_t=False] [arg=103,], LOAD_ATTR NULL|self + startswith [jump_t=False] [arg=187,], LOAD_NAME confine_prefix [jump_t=False] [arg=76,], CALL 1 [jump_t=False] [arg=1,]]
                                    out_same_host = False;      # bb78 → bb80, bb79 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST False [jump_t=False] [arg=6,], STORE_NAME out_same_host [jump_t=False] [arg=108,]]
                                    out_host = urlparse(link_url)[1];      # bb78 → bb80, bb79 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME urlparse [jump_t=False] [arg=17,], LOAD_NAME link_url [jump_t=False] [arg=103,], CALL 1 [jump_t=False] [arg=1,], LOAD_CONST 1 [jump_t=False] [arg=76,], BINARY_SUBSCR [jump_t=False] [arg=None,], STORE_NAME out_host [jump_t=False] [arg=109,]]
                                    out_same_host = re.match(('.*%s' % host), out_host) is not None;      # bb78 → bb80, bb79 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME re [jump_t=False] [arg=8,], LOAD_ATTR match [jump_t=False] [arg=200,], LOAD_CONST '.*%s' [jump_t=False] [arg=90,], LOAD_NAME host [jump_t=False] [arg=79,], BINARY_OP % [jump_t=False] [arg=6,], LOAD_NAME out_host [jump_t=False] [arg=109,], CALL 2 [jump_t=False] [arg=2,], LOAD_CONST None [jump_t=False] [arg=3,], IS_OP 1 [jump_t=False] [arg=1,], STORE_NAME out_same_host [jump_t=False] [arg=108,]]
                                    if out_prefix_ok and out_same_host {  # bb78 → bb80, bb79 [exec=0, time=0.0000s] [mem=57284 bytes] | bb79 → bb73 [exec=0, time=0.0000s] [mem=57284 bytes] | bb81 → bb73 [exec=0, time=0.0000s] [mem=57284 bytes] | bb80 → bb82, bb81 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME out_prefix_ok [jump_t=False] [arg=107,], POP_JUMP_IF_TRUE to 3786 [jump_t=False] [arg=1,], JUMP_BACKWARD to 3396 [jump_t=False] [arg=195,], LOAD_NAME out_same_host [jump_t=True] [arg=108,], POP_JUMP_IF_TRUE to 3792 [jump_t=False] [arg=1,], JUMP_BACKWARD to 3396 [jump_t=False] [arg=198,]]

                                        num_links += 1;      # bb82 → bb84, bb83 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME num_links [jump_t=True] [arg=85,], LOAD_CONST 1 [jump_t=False] [arg=76,], BINARY_OP += [jump_t=False] [arg=13,], STORE_NAME num_links [jump_t=False] [arg=85,]]
                                        urls_remembered.add(link_url) ;      # bb82 → bb84, bb83 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME urls_remembered [jump_t=False] [arg=82,], LOAD_ATTR NULL|self + add [jump_t=False] [arg=203,], LOAD_NAME link_url [jump_t=False] [arg=103,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]
                                        link = Link(this_url, link_url, 'href');      # bb82 → bb84, bb83 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME Link [jump_t=False] [arg=89,], LOAD_NAME this_url [jump_t=False] [arg=91,], LOAD_NAME link_url [jump_t=False] [arg=103,], LOAD_CONST 'href' [jump_t=False] [arg=85,], CALL 3 [jump_t=False] [arg=3,], STORE_NAME link [jump_t=False] [arg=110,]]
                                        if link not in links_remembered {  # bb82 → bb84, bb83 [exec=0, time=0.0000s] [mem=57284 bytes] | bb83 → bb73 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME link [jump_t=False] [arg=110,], LOAD_NAME links_remembered [jump_t=False] [arg=84,], CONTAINS_OP 1 [jump_t=False] [arg=1,], POP_JUMP_IF_TRUE to 3866 [jump_t=False] [arg=1,], JUMP_BACKWARD to 3396 [jump_t=False] [arg=235,]]

                                            links_remembered.add(link) ;  # bb84 → bb73 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME links_remembered [jump_t=True] [arg=84,], LOAD_ATTR NULL|self + add [jump_t=False] [arg=203,], LOAD_NAME link [jump_t=False] [arg=110,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,], JUMP_BACKWARD to 3396 [jump_t=False] [arg=253,]]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if opts.out_urls {  # bb88 → bb90, bb89 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME opts [jump_t=True] [arg=37,], LOAD_ATTR out_urls [jump_t=False] [arg=88,], POP_JUMP_IF_FALSE to 4010 [jump_t=False] [arg=23,]]

            print('\n'.join(urls_remembered)) ;  # bb89 → bb90 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME print [jump_t=False] [arg=73,], LOAD_CONST [arg=91,], LOAD_ATTR NULL|self + join [jump_t=False] [arg=223,], LOAD_NAME urls_remembered [jump_t=False] [arg=82,], CALL 1 [jump_t=False] [arg=1,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]
        }
        
        if opts.out_links {  # bb90 → bb94, bb91 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME opts [jump_t=True] [arg=37,], LOAD_ATTR out_links [jump_t=False] [arg=86,], POP_JUMP_IF_FALSE to 4116 [jump_t=False] [arg=41,]]
            for link in links_remembered {  # bb92 → bb93, bb92 [exec=0, time=0.0000s] [mem=57284 bytes] | bb91 → bb92 [exec=0, time=0.0000s] [mem=57284 bytes] | bb93 → bb94 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME links_remembered [jump_t=False] [arg=84,], GET_ITER [jump_t=False] [arg=None,], FOR_ITER to 4114 [jump_t=True] [arg=36,], STORE_NAME link [jump_t=False] [arg=110,], END_FOR [jump_t=True] [arg=None,]]

                print(((link.src + ' -> ') + link.dst)) ;  # bb92 → bb93, bb92 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME print [jump_t=False] [arg=73,], LOAD_NAME link [jump_t=False] [arg=110,], LOAD_ATTR src [jump_t=False] [arg=224,], LOAD_CONST ' -> ' [jump_t=False] [arg=92,], BINARY_OP + [jump_t=False] [arg=0,], LOAD_NAME link [jump_t=False] [arg=110,], LOAD_ATTR dst [jump_t=False] [arg=226,], BINARY_OP + [jump_t=False] [arg=0,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,], JUMP_BACKWARD to 4038 [jump_t=False] [arg=38,]]
            }
        }

        
        if opts.out_dot {  # bb94 → bb102, bb95 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME opts [jump_t=True] [arg=37,], LOAD_ATTR out_dot [jump_t=False] [arg=228,], EXTENDED_ARG 1 [jump_t=False] [arg=1,], POP_JUMP_IF_FALSE to 4672 [jump_t=False] [arg=265,]]

            print('digraph Crawl {') ;      # bb95 → bb96 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME print [jump_t=False] [arg=73,], LOAD_CONST 'digraph Crawl {' [jump_t=False] [arg=93,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]
            print('\t edge [K=0.2, len=0.1];') ;      # bb95 → bb96 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME print [jump_t=False] [arg=73,], LOAD_CONST '\t edge [K=0.2 [jump_t=False] [arg=94,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]
            for link in links_remembered {  # bb95 → bb96 [exec=0, time=0.0000s] [mem=57284 bytes] | bb96 → bb101, bb98, bb97 [exec=0, time=0.0000s] [mem=57284 bytes] | bb101 → bb102 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME links_remembered [jump_t=False] [arg=84,], GET_ITER [jump_t=False] [arg=None,], FOR_ITER to 4654 [jump_t=True] [arg=236,], STORE_NAME link [jump_t=False] [arg=110,], END_FOR [jump_t=True] [arg=None,]]

                src_url = link.src;      # bb96 → bb101, bb98, bb97 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME link [jump_t=False] [arg=110,], LOAD_ATTR src [jump_t=False] [arg=224,], STORE_NAME src_url [jump_t=False] [arg=115,]]

                dst_url = link.dst;      # bb96 → bb101, bb98, bb97 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME link [jump_t=False] [arg=110,], LOAD_ATTR dst [jump_t=False] [arg=226,], STORE_NAME dst_url [jump_t=False] [arg=116,]]
                if src_url not in node_alias {  # bb96 → bb101, bb98, bb97 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME src_url [jump_t=False] [arg=115,], LOAD_NAME node_alias [jump_t=False] [arg=47,], CONTAINS_OP 1 [jump_t=False] [arg=1,], POP_JUMP_IF_FALSE to 4416 [jump_t=False] [arg=88,]]

                    m = hashlib.md5();      # bb97 → bb98 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME hashlib [jump_t=False] [arg=19,], LOAD_ATTR md5 [jump_t=False] [arg=234,], CALL 0 [jump_t=False] [arg=0,], STORE_NAME m [jump_t=False] [arg=118,]]

                    m.update(src_url.encode('utf-8')) ;      # bb97 → bb98 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME m [jump_t=False] [arg=118,], LOAD_ATTR NULL|self + update [jump_t=False] [arg=239,], LOAD_NAME src_url [jump_t=False] [arg=115,], LOAD_ATTR NULL|self + encode [jump_t=False] [arg=241,], LOAD_CONST 'utf-8' [jump_t=False] [arg=80,], CALL 1 [jump_t=False] [arg=1,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]

                    name = ('N' + m.hexdigest());  # bb97 → bb98 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'N' [jump_t=False] [arg=95,], LOAD_NAME m [jump_t=False] [arg=118,], LOAD_ATTR NULL|self + hexdigest [jump_t=False] [arg=243,], CALL 0 [jump_t=False] [arg=0,], BINARY_OP + [jump_t=False] [arg=0,], STORE_NAME name [jump_t=False] [arg=122,]]
                    

                    node_alias[src_url] = name;      # bb97 → bb98 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME name [jump_t=False] [arg=122,], LOAD_NAME node_alias [jump_t=False] [arg=47,], LOAD_NAME src_url [jump_t=False] [arg=115,], STORE_SUBSCR [jump_t=False] [arg=None,]]

                    print(('\t%s [label="%s"];' % (name, src_url))) ;  # bb97 → bb98 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME print [jump_t=False] [arg=73,], LOAD_CONST '\t' [jump_t=False] [arg=96,], LOAD_NAME name [jump_t=False] [arg=122,], FORMAT_VALUE str [jump_t=False] [arg=1,], LOAD_CONST ' [label="' [jump_t=False] [arg=97,], LOAD_NAME src_url [jump_t=False] [arg=115,], FORMAT_VALUE str [jump_t=False] [arg=1,], LOAD_CONST '"];' [jump_t=False] [arg=98,], BUILD_STRING 5 [jump_t=False] [arg=5,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]
                }

                
                if dst_url not in node_alias {  # bb98 → bb100, bb99 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME dst_url [jump_t=True] [arg=116,], LOAD_NAME node_alias [jump_t=False] [arg=47,], CONTAINS_OP 1 [jump_t=False] [arg=1,], POP_JUMP_IF_FALSE to 4600 [jump_t=False] [arg=88,]]

                    m = hashlib.md5();      # bb99 → bb100 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME hashlib [jump_t=False] [arg=19,], LOAD_ATTR md5 [jump_t=False] [arg=234,], CALL 0 [jump_t=False] [arg=0,], STORE_NAME m [jump_t=False] [arg=118,]]

                    m.update(dst_url.encode('utf-8')) ;      # bb99 → bb100 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME m [jump_t=False] [arg=118,], LOAD_ATTR NULL|self + update [jump_t=False] [arg=239,], LOAD_NAME dst_url [jump_t=False] [arg=116,], LOAD_ATTR NULL|self + encode [jump_t=False] [arg=241,], LOAD_CONST 'utf-8' [jump_t=False] [arg=80,], CALL 1 [jump_t=False] [arg=1,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]

                    name = ('N' + m.hexdigest());  # bb99 → bb100 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'N' [jump_t=False] [arg=95,], LOAD_NAME m [jump_t=False] [arg=118,], LOAD_ATTR NULL|self + hexdigest [jump_t=False] [arg=243,], CALL 0 [jump_t=False] [arg=0,], BINARY_OP + [jump_t=False] [arg=0,], STORE_NAME name [jump_t=False] [arg=122,]]
                    

                    node_alias[dst_url] = name;      # bb99 → bb100 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME name [jump_t=False] [arg=122,], LOAD_NAME node_alias [jump_t=False] [arg=47,], LOAD_NAME dst_url [jump_t=False] [arg=116,], STORE_SUBSCR [jump_t=False] [arg=None,]]

                    print(('\t%s [label="%s"];' % (name, dst_url))) ;  # bb99 → bb100 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME print [jump_t=False] [arg=73,], LOAD_CONST '\t' [jump_t=False] [arg=96,], LOAD_NAME name [jump_t=False] [arg=122,], FORMAT_VALUE str [jump_t=False] [arg=1,], LOAD_CONST ' [label="' [jump_t=False] [arg=97,], LOAD_NAME dst_url [jump_t=False] [arg=116,], FORMAT_VALUE str [jump_t=False] [arg=1,], LOAD_CONST '"];' [jump_t=False] [arg=98,], BUILD_STRING 5 [jump_t=False] [arg=5,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]
                }

                

                print(  # bb100 → bb96 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=True] [arg=None,], LOAD_NAME print [jump_t=False] [arg=73,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,], JUMP_BACKWARD to 4178 [jump_t=False] [arg=238,]]
                    (((('\t' + node_alias[src_url]) + ' -> ') + node_alias[dst_url]) + ';')  # bb100 → bb96 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST '\t' [jump_t=False] [arg=96,], LOAD_NAME node_alias [jump_t=False] [arg=47,], LOAD_NAME src_url [jump_t=False] [arg=115,], BINARY_SUBSCR [jump_t=False] [arg=None,], BINARY_OP + [jump_t=False] [arg=0,], LOAD_CONST ' -> ' [jump_t=False] [arg=92,], BINARY_OP + [jump_t=False] [arg=0,], LOAD_NAME node_alias [jump_t=False] [arg=47,], LOAD_NAME dst_url [jump_t=False] [arg=116,], BINARY_SUBSCR [jump_t=False] [arg=None,], BINARY_OP + [jump_t=False] [arg=0,], LOAD_CONST ';' [jump_t=False] [arg=99,], BINARY_OP + [jump_t=False] [arg=0,]]
                ) ;
            }

            
            print('}') ;  # bb101 → bb102 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME print [jump_t=False] [arg=73,], LOAD_CONST '}' [jump_t=False] [arg=100,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]
        }
        

        eTime = time.time();      # bb102 → bb103 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=True] [arg=None,], LOAD_NAME time [jump_t=False] [arg=11,], LOAD_ATTR time [jump_t=False] [arg=22,], CALL 0 [jump_t=False] [arg=0,], STORE_NAME eTime [jump_t=False] [arg=123,]]

        tTime = (eTime - sTime);  # bb102 → bb103 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME eTime [jump_t=False] [arg=123,], LOAD_NAME sTime [jump_t=False] [arg=78,], BINARY_OP - [jump_t=False] [arg=10,], STORE_NAME tTime [jump_t=False] [arg=124,]]
        

        print(sys.stderr, ('Found:    %d' % num_links)) ;  # bb102 → bb103 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME print [jump_t=False] [arg=73,], LOAD_NAME sys [jump_t=False] [arg=10,], LOAD_ATTR stderr [jump_t=False] [arg=82,], LOAD_CONST 'Found:    %d' [jump_t=False] [arg=101,], LOAD_NAME num_links [jump_t=False] [arg=85,], BINARY_OP % [jump_t=False] [arg=6,], CALL 2 [jump_t=False] [arg=2,], POP_TOP [jump_t=False] [arg=None,]]
        

        print(sys.stderr, ('Followed: %d' % num_followed)) ;  # bb102 → bb103 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME print [jump_t=False] [arg=73,], LOAD_NAME sys [jump_t=False] [arg=10,], LOAD_ATTR stderr [jump_t=False] [arg=82,], LOAD_CONST 'Followed: %d' [jump_t=False] [arg=102,], LOAD_NAME num_followed [jump_t=False] [arg=86,], BINARY_OP % [jump_t=False] [arg=6,], CALL 2 [jump_t=False] [arg=2,], POP_TOP [jump_t=False] [arg=None,]]
        

        print(  # bb102 → bb103 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME print [jump_t=False] [arg=73,], CALL 2 [jump_t=False] [arg=2,], POP_TOP [jump_t=False] [arg=None,], RETURN_CONST None [jump_t=False] [arg=3,]]
            sys.stderr,  # bb102 → bb103 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME sys [jump_t=False] [arg=10,], LOAD_ATTR stderr [jump_t=False] [arg=82,]]
            ('Stats:    (%d/s after %0.2fs)' % (int(math.ceil((float(num_links) / tTime)))  # bb102 → bb103 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_CONST 'Stats:    (%d/s after %0.2fs)' [jump_t=False] [arg=103,], PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME int [jump_t=False] [arg=125,], PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME math [jump_t=False] [arg=12,], LOAD_ATTR ceil [jump_t=False] [arg=252,], PUSH_NULL [jump_t=False] [arg=None,], LOAD_NAME float [jump_t=False] [arg=127,], LOAD_NAME num_links [jump_t=False] [arg=85,], CALL 1 [jump_t=False] [arg=1,], LOAD_NAME tTime [jump_t=False] [arg=124,], BINARY_OP / [jump_t=False] [arg=11,], CALL 1 [jump_t=False] [arg=1,], CALL 1 [jump_t=False] [arg=1,], BUILD_TUPLE 2 [jump_t=False] [arg=2,], BINARY_OP % [jump_t=False] [arg=6,]]
            ,tTime))  # bb102 → bb103 [exec=0, time=0.0000s] [mem=57284 bytes]
        #   [LOAD_NAME tTime [jump_t=False] [arg=124,]]
        ) ;
    }
}
