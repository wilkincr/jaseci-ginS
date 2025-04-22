import:py random;

with entry {
    names:list[str] = ["Harry", "Kazu", "Fardeen", "Joe", "Connor", "Mary", "Sue", "Clare"];
    #   Variable values in this block:   index = 0 (seen 1×)

	result:list[str] = [];
	while len(result) != len(names) {
		copy:list[str] = names.copy();
  #   Variable values in this block:   index = 0 (seen 5×)  index = 3 (seen 1×)  index = 4 (seen 1×)  index = 2 (seen 1×)

		for i in range(0, len(names)) {
			index:int = random.randrange(0,len(copy));
   #   Variable values in this block:   index = 0 (seen 14×)  index = 3 (seen 3×)  index = 4 (seen 3×)  index = 2 (seen 3×)
			result.append(copy[index]);			
			copy.remove(copy[index]);
        }
    }

    print(names);
    #   Variable values in this block:   index = 0 (seen 1×)
    print(result);
}