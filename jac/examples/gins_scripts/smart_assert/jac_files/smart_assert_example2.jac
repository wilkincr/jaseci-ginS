import:py random;
import:py from jaclang.runtimelib.gins.smart_assert, smart_assert ;

with entry {
    names = ["Harry", "Kazu", "Fardeen", "Joe", "Connor", "Mary", "Sue", "Clare"];

	result = [];
	while len(result) != len(names) {
		copy = names.copy();

		for i in range(0, len(names)) {
			index = random.randrange(0,len(copy));
			result.append(copy[index]);
		
			smart_assert(copy[index] != names[i], "");
			
			copy.remove(copy[index]);
        }
    }

    print(names);
    print(result);
}