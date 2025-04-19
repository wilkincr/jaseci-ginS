with entry {

    n_values = [127, 128, 3005, 13, 14, 27, 834, 254, 256];    
    expected_values = [7, 1, 9, 3, 3, 4, 4, 7, 1];    
    for i in range(len(n_values)) {

        n = n_values[i];    
        expected = expected_values[i];    
        count = 0;    
        temp = n;    
        while temp {

            temp ^= (temp - 1);
            
            count += 1;
        }
        
        print('Input:', n) ;    
        print('Expected:', expected) ;    
        print('Got     :', count) ;    
        print('Pass    :', count == expected) ;    
        print() ;
    }
}

