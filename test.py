# print("Hello World")
# import keyword
# keyword.kwlist
import keyword
print(keyword.kwlist)
# ['False', 'None', 'True', '__peg_parser__', 
# 'and', 'as', 'assert', 'async', 'await', 'break',
#  'class', 'continue', 'def', 'del', 'elif', 'else', 
# 'except', 'finally', 'for', 'from', 'global', 'if', 
# 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 
# 'or', 'pass', 'raise', 'return', 'try', 'while', 
# 'with', 'yield']

a , b  = 0 , 1
while b < 10 :
    print(b, end = " ")
    a , b = b , a + b
