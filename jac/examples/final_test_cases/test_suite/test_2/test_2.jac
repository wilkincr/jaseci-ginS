import:py random;

with entry {
    names:list[str] = ["Harry", "Kazu", "Fardeen", "Joe", "Connor", "Mary", "Sue", "Clare"];

	result:list[str] = [];
	while len(result) != len(names) {
		copy:list[str] = names.copy();

		for i in range(0, len(names)) {
			index:int = random.randrange(0,len(copy));
			result.append(copy[index]);			
			copy.remove(copy[index]);
        }
    }

    print(names);
    print(result);
}