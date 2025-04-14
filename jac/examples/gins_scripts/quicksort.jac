import:py time ;

with entry {

    arr = <>list(range(100));    
    start_time = time.time();    
    stack = [(0, (len(arr) - 1))];
    
    
    while stack {

        (low, high) = stack.pop();    
        if low < high {

            pivot_index = low;    
            pivot_value = arr[pivot_index];    
            i = (low + 1);
            
            j = high;    
            while True {
                while i <= j and arr[i] < pivot_value {

                    i += 1;
                }
                
                while i <= j and arr[j] >= pivot_value {

                    j -= 1;
                }
                

                if i <= j {

                    (arr[i], arr[j]) = (arr[j], arr[i]);
                } else {
                    break ;
                }
            }

            
            (arr[low], arr[j]) = (arr[j], arr[low]);    
            stack.append((low, (j - 1))) ;
            
            stack.append(((j + 1), high)) ;
        }
    }
    
    elapsed_time = (time.time() - start_time);
    
    print('Input size:', len(arr)) ;    
    print('First 10 elements of result:', arr[:10]) ;    
    print('Last 10 elements of result:', arr[-10:]) ;    
    print(f"{'Elapsed time: '}{elapsed_time}{' seconds'}") ;
}

