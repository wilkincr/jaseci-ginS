import:py random;
#   Instructions: [SETUP_ANNOTATIONS(None)]

with entry {
    names:list[str] = ["Harry", "Kazu", "Fardeen", "Joe", "Connor", "Mary", "Sue", "Clare"];
    #   Instructions: [BUILD_LIST(0), LOAD_CONST(('Harry', 'Kazu', 'Fardeen', 'Joe', 'Connor', 'Mary', 'Sue', 'Clare')) [no line], LIST_EXTEND(1) [no line], STORE_NAME(names) [no line], LOAD_CONST(list[str]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(names) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(result) [no line], LOAD_CONST(list[str]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(result) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(result) [no line], CALL(1) [no line], PUSH_NULL(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(names) [no line], CALL(1) [no line], COMPARE_OP(!=) [no line], POP_JUMP_IF_FALSE(426) [no line]]

	result:list[str] = [];
 #   Instructions: [BUILD_LIST(0)]
	while len(result) != len(names) {
 #   Instructions: [PUSH_NULL(None), PUSH_NULL(None)]
		copy:list[str] = names.copy();
  #   Instructions: [LOAD_NAME(names), LOAD_ATTR(copy) [no line], CALL(0) [no line], STORE_NAME(copy) [no line], LOAD_CONST(list[str]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(copy) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(range) [no line], LOAD_CONST(0) [no line], PUSH_NULL(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(names) [no line], CALL(1) [no line], CALL(2) [no line], GET_ITER(None) [no line]]

		for i in range(0, len(names)) {
  #   Instructions: [PUSH_NULL(None), END_FOR(None), LOAD_NAME(len) [no line], LOAD_NAME(result) [no line], CALL(1) [no line], PUSH_NULL(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(names) [no line], CALL(1) [no line], COMPARE_OP(!=) [no line], POP_JUMP_IF_FALSE(426) [no line]]
			index:int = random.randrange(0,len(copy));
   #   Instructions: [FOR_ITER(388) [no line], STORE_NAME(i) [no line], PUSH_NULL(None), LOAD_NAME(random) [no line], LOAD_ATTR(randrange) [no line], LOAD_CONST(0) [no line], PUSH_NULL(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(copy) [no line], CALL(1) [no line], CALL(2) [no line], STORE_NAME(index) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(index) [no line], STORE_SUBSCR(None) [no line], LOAD_ATTR(append) [no line], LOAD_NAME(copy) [no line], LOAD_NAME(index) [no line], BINARY_SUBSCR(None) [no line], CALL(1) [no line], POP_TOP(None) [no line], LOAD_ATTR(remove) [no line], LOAD_NAME(copy) [no line], LOAD_NAME(index) [no line], BINARY_SUBSCR(None) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(240) [no line]]
			result.append(copy[index]);			
   #   Instructions: [LOAD_NAME(result)]
			copy.remove(copy[index]);
   #   Instructions: [LOAD_NAME(copy)]
        }
    }

    print(names);
    #   Instructions: [PUSH_NULL(None), LOAD_NAME(print) [no line], LOAD_NAME(names) [no line], CALL(1) [no line], POP_TOP(None) [no line], LOAD_NAME(print) [no line], LOAD_NAME(result) [no line], CALL(1) [no line], POP_TOP(None) [no line]]
    print(result);
    #   Instructions: [PUSH_NULL(None)]
}