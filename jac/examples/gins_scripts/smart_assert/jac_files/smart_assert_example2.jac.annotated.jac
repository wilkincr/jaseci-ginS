import:py random;  # bb2 → bb3 [exec=0, time=0.0000s] [mem=38666 bytes] | bb0 → bb2, bb1 [exec=0, time=0.0000s] [mem=38666 bytes] | bb1 → bb3 [exec=0, time=0.0000s] [mem=38666 bytes]
import:py from jaclang.runtimelib.gins.smart_assert, smart_assert ;  # bb5 → bb6 [exec=0, time=0.0000s] [mem=38666 bytes] | bb4 → bb6 [exec=0, time=0.0000s] [mem=38666 bytes] | bb3 → bb5, bb4 [exec=0, time=0.0000s] [mem=38666 bytes]

with entry {
    names = ["Harry", "Kazu", "Fardeen", "Joe", "Connor", "Mary", "Sue", "Clare"];  # bb6 → bb11, bb7 [exec=0, time=0.0000s] [mem=38666 bytes]

	result = [];  # bb6 → bb11, bb7 [exec=0, time=0.0000s] [mem=38666 bytes]
	while len(result) != len(names) {  # bb9 → bb11, bb10 [exec=0, time=0.0000s] [mem=38666 bytes] | bb6 → bb11, bb7 [exec=0, time=0.0000s] [mem=38666 bytes] | bb10 → bb7 [exec=0, time=0.0000s] [mem=38666 bytes]
		copy = names.copy();  # bb7 → bb8 [exec=0, time=0.0000s] [mem=38666 bytes]

		for i in range(0, len(names)) {  # bb7 → bb8 [exec=0, time=0.0000s] [mem=38666 bytes] | bb9 → bb11, bb10 [exec=0, time=0.0000s] [mem=38666 bytes] | bb8 → bb9, bb8 [exec=0, time=0.0000s] [mem=38666 bytes]
			index = random.randrange(0,len(copy));  # bb8 → bb9, bb8 [exec=0, time=0.0000s] [mem=38666 bytes]
			result.append(copy[index]);  # bb8 → bb9, bb8 [exec=0, time=0.0000s] [mem=38666 bytes]
		
			smart_assert(copy[index] != names[i], "");  # bb8 → bb9, bb8 [exec=0, time=0.0000s] [mem=38666 bytes]
			
			copy.remove(copy[index]);  # bb8 → bb9, bb8 [exec=0, time=0.0000s] [mem=38666 bytes]
        }
    }

    print(names);  # bb11 [exec=0, time=0.0000s] [mem=38666 bytes]
    print(result);  # bb11 [exec=0, time=0.0000s] [mem=38666 bytes]
}