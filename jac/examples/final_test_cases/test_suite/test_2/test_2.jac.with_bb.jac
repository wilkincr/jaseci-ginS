import:py random; # BB: 0 Execution frequency: 0 Total execution time: 0.000 ms

with entry {
    names:list[str] = ["Harry", "Kazu", "Fardeen", "Joe", "Connor", "Mary", "Sue", "Clare"]; # BB: 3 Execution frequency: 0 Total execution time: 0.000 ms

	result:list[str] = []; # BB: 3
	while len(result) != len(names) { # BB: 3
		copy:list[str] = names.copy(); # BB: 4 Execution frequency: 0 Total execution time: 0.000 ms

		for i in range(0, len(names)) { # BB: 4
			index:int = random.randrange(0,len(copy)); # BB: 5 Execution frequency: 0 Total execution time: 0.000 ms
			result.append(copy[index]);			 # BB: 5
			copy.remove(copy[index]); # BB: 5
        }
    }

    print(names); # BB: 8 Execution frequency: 0 Total execution time: 0.000 ms
    print(result); # BB: 8
}