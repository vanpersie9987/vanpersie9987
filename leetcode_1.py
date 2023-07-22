# print("Hello World")
# import keyword
# keyword.kwlist
# import keyword
# print(keyword.kwlist)
# ['False', 'None', 'True', '__peg_parser__', 
# 'and', 'as', 'assert', 'async', 'await', 'break',
#  'class', 'continue', 'def', 'del', 'elif', 'else', 
# 'except', 'finally', 'for', 'from', 'global', 'if', 
# 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 
# 'or', 'pass', 'raise', 'return', 'try', 'while', 
# 'with', 'yield']

# a , b  = 0 , 1
# while b < 10 :
#     print(b, end = " ")
#     a , b = b , a + b

from functools import cache
from math import inf
from typing import List

class leetcode_1 :
    # 2463. 最小移动总距离 (Minimum Total Distance Traveled)
    def minimumTotalDistance(self, robot: List[int], factory: List[List[int]]) -> int:
      robot.sort()
      factory.sort(key=lambda f : f[0])
      m , n = len(robot) , len(factory)

      @cache
      def dfs(i : int , j : int) -> int:
          if i == m : return 0
          if j == n : return inf

          dis, k = 0 , 1
          res = dfs(i, j + 1)
          while k <= factory[j][1] and i + k - 1 < m:
            dis += abs(factory[j][0] - robot[i + k - 1])
            res = min(res, dfs(i + k, j + 1) + dis)
            k += 1
          return res
      return dfs(0, 0)