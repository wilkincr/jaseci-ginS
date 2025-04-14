import:py time ;
import:py random ;

with entry {

    cache = {};
    
    inputs = [];
    
    
    for _ in range(100) {

        inputs.append(random.randint(1, (10 ** 6))) ;
    }

    
    start_time = time.time();    
    results = [];
    
    
    for num in inputs {

        if num in cache {
            result = cache[num];
        } else {

            time.sleep(0.01) ;    
            result = (num * num);
            
            cache[num] = result;
        }
        
        results.append(result) ;
    }
    
    print(f"{'Computed '}{len(results)}{' results.'}") ;    
    print(
        f"{'Elapsed time: '}{(time.time() - start_time)}{' seconds'}"
    ) ;
}

