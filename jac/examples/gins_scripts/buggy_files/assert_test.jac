import:py from jaclang.runtimelib.gins.smart_assert, smart_assert ;

with entry {

    x = 10;    
    for i in range(10) {

        x = (x - 1);
    }

    smart_assert(x != 0, 'test') ;    
    
    y = (1 / x);
}

