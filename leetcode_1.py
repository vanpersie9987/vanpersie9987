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
    

    # 1478. 安排邮筒 (Allocate Mailboxes)
    def minDistance(self, houses: List[int], k: int) -> int:
       houses.sort()
       n = len(houses)
       @cache
       def dfs(i: int, j: int) -> int:
          if i == n :
             return 0
          if j == k :
             return inf
          res = inf
          for x in range(i, n) :
             res = min(res, dfs(x + 1, j + 1) + dis(i, x))
          return res
       def dis(i: int , j: int) -> int:
          res = 0
          while i < j:
            res += houses[j] - houses[i]
            j -= 1
            i += 1
          return res
       return dfs(0, 0)
    
    # 860. 柠檬水找零 (Lemonade Change)
    def lemonadeChange(self, bills: List[int]) -> bool:
        f ,t = 0, 0
        for x in bills :
            if x == 5 :
                f += 1
            elif x == 10 :
                if f == 0 :
                    return False
                f -= 1
                t += 1
            else :
                if f >= 1 and t >= 1 :
                    f -= 1
                    t -= 1
                elif f >= 3 :
                    f -= 3
                else :
                    return False
        return True
    
    # 1388. 3n 块披萨 (Pizza With 3n Slices)
    def maxSizeSlices(self, slices: List[int]) -> int:
       nums1 = slices[1:]
       nums2 = slices[:-1]
       n = len(nums1)
       d = (n + 1) // 3
       @cache
       def dfs1(i: int, j: int) -> int :
          if i >= n or j == d:
             return 0
          return max(dfs1(i + 1, j), dfs1(i + 2, j + 1) + nums1[i])
       @cache
       def dfs2(i: int, j: int) -> int :
          if i >= n or j == d:
             return 0
          return max(dfs2(i + 1, j), dfs2(i + 2, j + 1) + nums2[i])
       return max(dfs1(0, 0), dfs2(0, 0))
    
    # 2784. 检查数组是否是好的 (Check if Array is Good)
    def isGood(self, nums: List[int]) -> bool:
       nums.sort()
       n = len(nums)
       if nums[-1] != n - 1 :
          return False
       for i in range(0, n - 1) :
          if i + 1 != nums[i] :
             return False
       return True
    
    
    # 2786. 访问数组中的位置使分数最大 (Visit Array Positions to Maximize Score)
    def maxScore(self, nums: List[int], x: int) -> int:
       list = []
       n = len(nums)
       def getList() -> int:
          i = 0
          j = 0
          while j < n:
             sum = 0
             while j < n and nums[j] % 2 == nums[i] % 2:
                sum += nums[j]
                j += 1
             list.append(sum)
             i = j
          return len(list)
       
       @cache
       def dfs(i: int) -> int:
          if i >= n:
             return 0
          return max(dfs(i + 1) - x, dfs(i + 2)) + list[i]
       n = getList()
       return dfs(0)
