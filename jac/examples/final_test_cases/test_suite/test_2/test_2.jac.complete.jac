import:py random; # BB: 0 Execution frequency: 0 Total execution time: 0.000 ms
#   Instructions: [SETUP_ANNOTATIONS(None)]

with entry {
    names:list[str] = ["Harry", "Kazu", "Fardeen", "Joe", "Connor", "Mary", "Sue", "Clare"]; # BB: 3 Execution frequency: 0 Total execution time: 0.000 ms
    #   Instructions: [BUILD_LIST(0), LOAD_CONST(('Harry', 'Kazu', 'Fardeen', 'Joe', 'Connor', 'Mary', 'Sue', 'Clare')) [no line], LIST_EXTEND(1) [no line], STORE_NAME(names) [no line], LOAD_CONST(list[str]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(names) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(result) [no line], LOAD_CONST(list[str]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(result) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(result) [no line], CALL(1) [no line], PUSH_NULL(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(names) [no line], CALL(1) [no line], COMPARE_OP(!=) [no line], POP_JUMP_IF_FALSE(426) [no line]]
    #   Variable values in this block:   index = 0 (seen 1×)

	result:list[str] = []; # BB: 3
 #   Instructions: [BUILD_LIST(0)]
	while len(result) != len(names) { # BB: 3
 #   Instructions: [PUSH_NULL(None), PUSH_NULL(None)]
		copy:list[str] = names.copy(); # BB: 4 Execution frequency: 0 Total execution time: 0.000 ms
  #   Instructions: [LOAD_NAME(names), LOAD_ATTR(copy) [no line], CALL(0) [no line], STORE_NAME(copy) [no line], LOAD_CONST(list[str]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(copy) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(range) [no line], LOAD_CONST(0) [no line], PUSH_NULL(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(names) [no line], CALL(1) [no line], CALL(2) [no line], GET_ITER(None) [no line]]
  #   Variable values in this block:   index = 0 (seen 5×)  index = 3 (seen 1×)  index = 4 (seen 1×)  index = 2 (seen 1×)

		for i in range(0, len(names)) { # BB: 4
  #   Instructions: [PUSH_NULL(None), END_FOR(None), LOAD_NAME(len) [no line], LOAD_NAME(result) [no line], CALL(1) [no line], PUSH_NULL(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(names) [no line], CALL(1) [no line], COMPARE_OP(!=) [no line], POP_JUMP_IF_FALSE(426) [no line]]
			index:int = random.randrange(0,len(copy)); # BB: 5 Execution frequency: 0 Total execution time: 0.000 ms
   #   Instructions: [FOR_ITER(388) [no line], STORE_NAME(i) [no line], PUSH_NULL(None), LOAD_NAME(random) [no line], LOAD_ATTR(randrange) [no line], LOAD_CONST(0) [no line], PUSH_NULL(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(copy) [no line], CALL(1) [no line], CALL(2) [no line], STORE_NAME(index) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(index) [no line], STORE_SUBSCR(None) [no line], LOAD_ATTR(append) [no line], LOAD_NAME(copy) [no line], LOAD_NAME(index) [no line], BINARY_SUBSCR(None) [no line], CALL(1) [no line], POP_TOP(None) [no line], LOAD_ATTR(remove) [no line], LOAD_NAME(copy) [no line], LOAD_NAME(index) [no line], BINARY_SUBSCR(None) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(240) [no line]]
   #   Variable values in this block:   index = 0 (seen 14×)  index = 3 (seen 3×)  index = 4 (seen 3×)  index = 2 (seen 3×)
			result.append(copy[index]);			 # BB: 5
   #   Instructions: [LOAD_NAME(result)]
			copy.remove(copy[index]); # BB: 5
   #   Instructions: [LOAD_NAME(copy)]
        }
    }

    print(names); # BB: 8 Execution frequency: 0 Total execution time: 0.000 ms
    #   Instructions: [PUSH_NULL(None), LOAD_NAME(print) [no line], LOAD_NAME(names) [no line], CALL(1) [no line], POP_TOP(None) [no line], LOAD_NAME(print) [no line], LOAD_NAME(result) [no line], CALL(1) [no line], POP_TOP(None) [no line]]
    #   Variable values in this block:   index = 0 (seen 1×)
    print(result); # BB: 8
    #   Instructions: [PUSH_NULL(None)]
}