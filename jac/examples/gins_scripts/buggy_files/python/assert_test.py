from jaclang.runtimelib.gins.smart_assert import smart_assert


x = 10
for i in range(10):
    x = x - 1
smart_assert((x != 0), "test")
y = 1/x
