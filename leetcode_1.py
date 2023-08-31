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

from collections import Counter
import collections
from functools import cache
from math import gcd, inf
from queue import PriorityQueue
from typing import List, Optional
import heapq
import bisect

class leetcode_1 :
    class TreeNode:
       def __init__(self, val=0, left=None, right=None):
          self.val = val
          self.left = left
          self.right = right

    class ListNode:
       def __init__(self, val=0, next=None):
          self.val = val
          self.next = next
          
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
       def getList() -> None:
          i = 0
          j = 0
          n = len(nums)
          while j < n:
             sum = 0
             while j < n and nums[j] % 2 == nums[i] % 2:
                sum += nums[j]
                j += 1
             list.append(sum)
             i = j
       
       @cache
       def dfs(i: int) -> int:
          if i >= n:
             return 0
          return max(dfs(i + 1) - x, dfs(i + 2)) + list[i]
       getList()
       n = len(list)
       return dfs(0)

    # 70. 爬楼梯 (Climbing Stairs)
    # 剑指 Offer 10- II. 青蛙跳台阶问题
    def climbStairs(self, n: int) -> int:
       @cache
       def dfs(i: int) -> int:
          if i <= 1:
             return 1
          return dfs(i - 1) + dfs(i - 2)
       return dfs(n)
    
    # 62. 不同路径 (Unique Paths)
    def uniquePaths(self, m: int, n: int) -> int:
       @cache
       def dfs(i: int, j: int) -> int:
          if i == m or j == n:
             return 0
          if i == m - 1 and j == n - 1:
             return 1
          return dfs(i + 1, j) + dfs(i, j + 1) 
       return dfs(0, 0)
   
    
    # 64. 最小路径和 (Minimum Path Sum)
    # LCR 099. 最小路径和
    def minPathSum(self, grid: List[List[int]]) -> int:
       m = len(grid)
       n = len(grid[0])
       @cache
       def dfs(i: int, j: int) -> int:
          if i == m or j == n:
             return inf
          if i == m - 1 and j == n - 1:
             return grid[i][j]
          return min(dfs(i + 1, j), dfs(i, j + 1)) + grid[i][j]
       return dfs(0, 0)
    
    # 63. 不同路径 II (Unique Paths II)
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
       m = len(obstacleGrid)
       n = len(obstacleGrid[0])
       
       @cache
       def dfs(i: int, j: int) -> int:
          if i == m or j == n:
             return 0
          if obstacleGrid[i][j] == 1:
             return 0
          if i == m - 1 and j == n - 1:
             return 1
          return dfs(i + 1, j) + dfs(i, j + 1)
       return dfs(0, 0)
    
    # 91. 解码方法 (Decode Ways)
    def numDecodings(self, s: str) -> int:
       n = len(s)
       
       @cache
       def dfs(i: int) -> int:
          if i > n:
             return 0
          if i == n:
             return 1
          if s[i] == '0':
             return 0
          res = dfs(i + 1)
          if s[i] == '1':
             res += dfs(i + 2)
          elif s[i] == '2':
             if i + 1 < n and s[i + 1] <= '6':
                res += dfs(i + 2)
          return res
       return dfs(0)
    
    # 97. 交错字符串 (Interleaving String)
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
       n1 = len(s1)
       n2 = len(s2)
       n3 = len(s3)
       if n1 + n2 != n3:
          return False
       @cache
       def dfs(i: int, j: int, k: int) -> int:
          if i == n1:
             return s2[j:] == s3[k:]
          if j == n2:
             return s1[i:] == s3[k:]
          if s1[i:] + s2[j:] == s3[k:] or s2[j:] + s1[i:] == s3[k:]:
             return True
          res = False
          if s1[i] == s3[k]:
             res = res or dfs(i + 1, j, k + 1)
          if s2[j] == s3[k]:
             res = res or dfs(i, j + 1, k + 1)
          return res
       return dfs(0, 0, 0)
    
    # 120. 三角形最小路径和 (Triangle)
    # LCR 100. 三角形最小路径和
    def minimumTotal(self, triangle: List[List[int]]) -> int:
       n = len(triangle)

       @cache
       def dfs(i: int, j: int) -> int:
          if i == n:
             return 0
          return min(dfs(i + 1, j), dfs(i + 1, j + 1)) + triangle[i][j]
       return dfs(0, 0)
       
    # 121. 买卖股票的最佳时机 (Best Time to Buy and Sell Stock)
    # 剑指 Offer 63. 股票的最大利润
    def maxProfit(self, prices: List[int]) -> int:
       n = len(prices)

       @cache
       def dfs(i: int, j: int) -> int:
          if i == n:
             return 0
          if j == 0:
             return max(dfs(i + 1, 0), dfs(i + 1, 1) - prices[i])
          return max(dfs(i + 1, 1), prices[i])
       return dfs(0, 0)
    
    # 122. 买卖股票的最佳时机 II (Best Time to Buy and Sell Stock II)
    def maxProfit(self, prices: List[int]) -> int:
       n = len(prices)

       @cache
       def dfs(i: int, j: int) -> int:
          if i == n:
             return 0
          return max(dfs(i + 1, j), dfs(i + 1, j ^ 1) + (j * 2 - 1) * prices[i])
       return dfs(0, 0)
    
    # 123. 买卖股票的最佳时机 III (Best Time to Buy and Sell Stock III)
    def maxProfit(self, prices: List[int]) -> int:
       n = len(prices)

       @cache
       def dfs(i: int, j: int, k: int) -> int:
          if i == n:
             return 0
          if j == 2:
             return 0
          if k == 0:
             return max(dfs(i + 1, j, 0), dfs(i + 1, j, 1) - prices[i])
          return max(dfs(i + 1, j, 1), dfs(i + 1, j + 1, 0) + prices[i])
       return dfs(0, 0, 0)
    
   # 188. 买卖股票的最佳时机 IV (Best Time to Buy and Sell Stock IV)
    def maxProfit(self, k: int, prices: List[int]) -> int:
       n = len(prices)

       @cache
       def dfs(i: int, j: int, s: int) -> int:
          if i == n:
             return 0
          if j == k:
             return 0
          if s == 0:
             return max(dfs(i + 1, j, 0), dfs(i + 1, j, 1) - prices[i])
          return max(dfs(i + 1, j, 1), dfs(i + 1, j + 1, 0) + prices[i])
       return dfs(0, 0, 0)

    # 198. 打家劫舍 (House Robber)
    def rob(self, nums: List[int]) -> int:
       n = len(nums)

       @cache
       def dfs(i: int) -> int:
          if i >= n:
             return 0
          return max(dfs(i + 1), dfs(i + 2) + nums[i])
       return dfs(0)
       
    # 213. 打家劫舍 II (House Robber II)
    def rob(self, nums: List[int]) -> int:
       n = len(nums) - 1
       if n == 0:
          return nums[0]
       nums1 = nums[1:]
       nums2 = nums[:-1]
       @cache
       def dfs1(i: int) -> int:
          if i >= n:
             return 0
          return max(dfs1(i + 1), dfs1(i + 2) + nums1[i])
       @cache
       def dfs2(i: int) -> int:
          if i >= n:
             return 0
          return max(dfs2(i + 1), dfs2(i + 2) + nums2[i])
       return max(dfs1(0), dfs2(0))
    
   # 2500. 删除每行中的最大值 (Delete Greatest Value in Each Row)
    def deleteGreatestValue(self, grid: List[List[int]]) -> int:
       for row in grid:
            row.sort()
       return sum(max(col) for col in zip(*grid))


    # 309. 最佳买卖股票时机含冷冻期 (Best Time to Buy and Sell Stock with Cooldown)
    def maxProfit(self, prices: List[int]) -> int:
       n = len(prices)

       @cache
       def dfs(i: int, j: int) -> int:
          if i >= n:
             return 0
          if j == 0:
             return max(dfs(i + 1, j), dfs(i + 1, 1) - prices[i])
          return max(dfs(i + 1, j), dfs(i + 2, 0) + prices[i])
       return dfs(0, 0)
    
   # 714. 买卖股票的最佳时机含手续费 (Best Time to Buy and Sell Stock with Transaction Fee)
    def maxProfit(self, prices: List[int], fee: int) -> int:
       n = len(prices)

       @cache
       def dfs(i: int, j: int) -> int:
          if i == n:
             return 0
          if j == 0:
             return max(dfs(i + 1, j), dfs(i + 1, 1) - prices[i] - fee)
          return max(dfs(i + 1, j), dfs(i + 1, 0) + prices[i])
       return dfs(0, 0)

    # 329. 矩阵中的最长递增路径 (Longest Increasing Path in a Matrix)
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
       m = len(matrix)
       n = len(matrix[0])
       @cache
       def dfs(i: int, j: int) -> int:
          if i < 0 or i >= m or j < 0 or j >= n:
             return 0
          res = 1
          dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]]
          for d in dirs:
             x = i + d[0]
             y = j + d[1]
             if x >= 0 and x < m and y >= 0 and y < n and matrix[x][y] > matrix[i][j]:
                res = max(res, dfs(x, y) + 1)
          return res
       res = 0
       for i in range(m):
          for j in range(n):
             res = max(res, dfs(i, j))
       return res
    
   # 518. 零钱兑换 II (Coin Change II) 
    def change(self, amount: int, coins: List[int]) -> int:
       coins.sort()
       n = len(coins)
       
       @cache
       def dfs(i: int, j: int) -> int:
          if j == amount:
             return 1
          if j > amount:
             return 0
          if i == n:
             return 0
          return dfs(i + 1, j) + dfs(i, j + coins[i])
       return dfs(0, 0)
    
   # 322. 零钱兑换 (Coin Change)
    def coinChange(self, coins: List[int], amount: int) -> int:
       n = len(coins)

       @cache
       def dfs(i:int, j: int) -> int:
          if j == amount:
             return 0
          if i == n or j > amount:
             return inf
          return min(dfs(i, j + coins[i]) + 1, dfs(i + 1, j))
       res = dfs(0, 0)
       return res if res <= amount else -1
    

    # 576. 出界的路径数 (Out of Boundary Paths)
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
       M = 10 ** 9 + 7
       @cache
       def dfs(i: int, j: int, l: int) -> int:
          if i < 0 or i >= m or j < 0 or j >= n:
             return 1
          if l == 0:
             return 0
          res = 0
          dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]]
          for d in dirs:
             x = i + d[0]
             y = j + d[1]
             res = (res + dfs(x, y, l - 1)) % M
          return res
       return dfs(startRow, startColumn, maxMove)
    
    # 583. 两个字符串的删除操作 (Delete Operation for Two Strings)
    def minDistance(self, word1: str, word2: str) -> int:
       n = len(word1)
       m = len(word2)

       @cache
       def dfs(i: int, j: int) -> int:
          if i < 0:
             return j + 1
          if j < 0:
             return i + 1
          if word1[i] == word2[j]:
             return dfs(i - 1, j - 1)
          return min(dfs(i - 1, j), dfs(i, j - 1)) + 1
       return dfs(n - 1, m - 1)
    
   # 516. 最长回文子序列 (Longest Palindromic Subsequence)
    def longestPalindromeSubseq(self, s: str) -> int:
       n = len(s)

       @cache
       def dfs(i: int, j: int) -> int:
          if i >= j:
             return 0
          if s[i] == s[j]:
             return dfs(i + 1, j - 1)
          return min(dfs(i + 1, j), dfs(i, j - 1)) + 1
       return n - dfs(0, n - 1)
    
    # 790. 多米诺和托米诺平铺 (Domino and Tromino Tiling)
    def numTilings(self, n: int) -> int:
       # 0 1 2
       M = 10 ** 9 + 7
       @cache
       def dfs(i: int, j: int) -> int:
          if i == n:
             return j == 0
          if i > n:
             return 0
          if j == 0:
             return (dfs(i + 1, j) + dfs(i + 2, j) + dfs(i + 2, 1) + dfs(i + 2, 2)) % M
          return (dfs(i + 1, 0) + dfs(i + 1, 2 if j == 1 else 2)) % M
       return dfs(0, 0)
    
   # 2140. 解决智力问题 (Solving Questions With Brainpower) 
    def mostPoints(self, questions: List[List[int]]) -> int:
       n = len(questions)

       @cache
       def dfs(i: int) -> int:
          if i >= n:
             return 0
          return max(dfs(i + 1), dfs(i + questions[i][1] + 1) + questions[i][0])
       return dfs(0)
    
    # 2400. 恰好移动 k 步到达某一位置的方法数目 (Number of Ways to Reach a Position After Exactly k Steps)
    def numberOfWays(self, startPos: int, endPos: int, k: int) -> int:
       M = 10 ** 9 + 7
       @cache
       def dfs(i: int, j: int) -> int:
          if i == endPos and j == 0:
             return 1
          if abs(endPos - i) > j:
             return 0
          return (dfs(i + 1, j - 1) + dfs(i - 1, j - 1)) % M
       return dfs(startPos, k)
    
    # 2585. 获得分数的方法数 (Number of Ways to Earn Points)
    def waysToReachTarget(self, target: int, types: List[List[int]]) -> int:
       M = 10 ** 9 + 7
       n = len(types)
       @cache
       def dfs(i: int, j: int) -> int:
          if j == target:
             return 1
          if i == n:
             return 0
          res = 0
          k = 0
          while k <= types[i][0]:
             if k * types[i][1] + j > target:
                break
             res = (res + dfs(i + 1, k * types[i][1] + j)) % M
             k += 1
          return res

       return dfs(0, 0)
    

    # 6957. 统计范围内的步进数字数目 (Count Stepping Numbers in Range)
    def countSteppingNumbers(self, low: str, high: str) -> int:
       m = 10 ** 9 + 7
       def cal(s: str) -> int:
         @cache
         def dfs(i: int, pre: int, isLimit: bool, isNum: bool) -> int:
            if i == len(s):
               return isNum
            res = 0
            if not isNum:
               res = dfs(i + 1, pre, False, False)
            up = int(s[i]) if isLimit else 9
            for j in range(0 if isNum else 1, up + 1):
               if (not isNum) or abs(j - pre) == 1:
                  res += dfs(i + 1, j, isLimit and j == up, True)
            return res % m
         return dfs(0, 0, True, False)
       return (cal(high) - cal(str(int(low) - 1))) % m
    
    # 233. 数字 1 的个数 (Number of Digit One)
    # 剑指 Offer 43. 1～n 整数中 1 出现的次数
    def countDigitOne(self, n: int) -> int:
       s = str(n)
       m = len(s)

       @cache
       def dfs(i: int, cnt: int, isLimit: bool, isNum: bool) -> int:
          if i == m:
             return cnt if isNum else 0
          res = 0
          if not isNum:
             res = dfs(i + 1, cnt, False, False)
          up = ord(s[i]) - ord('0') if isLimit else 9
          for d in range(0 if isNum else 1, up + 1):
             res += dfs(i + 1, cnt + (d == 1), isLimit and d == up, True)
          return res
       return dfs(0, 0, True, False)
    
    # 面试题 17.06. 2出现的次数 (Number Of 2s In Range LCCI)
    def numberOf2sInRange(self, n: int) -> int:
       s = str(n)
       m = len(s)

       @cache
       def dfs(i: int, cnt: int, isLimit: bool, isNum: bool) -> int:
          if i == m:
             return cnt if isNum else 0
          res = 0
          if not isNum:
             res += dfs(i + 1, cnt, False, False)
          up = ord(s[i]) - ord('0') if isLimit else 9
          for d in range(0 if isNum else 1, up + 1):
             res += dfs(i + 1, cnt + (d == 2), isLimit and d == up, True)
          return res
       return dfs(0, 0, True, False)
    

    # 600. 不含连续1的非负整数 (Non-negative Integers without Consecutive Ones)
    def findIntegers(self, n: int) -> int:
       s = bin(n)[2:]
       m = len(s)
       @cache
       def dfs(i: int, pre: int, isLimit: bool, isNum: bool) -> int:
          if i == m:
             return isNum
          res = 0
          if not isNum:
             res = dfs(i + 1, pre, False, False)
          up = ord(s[i]) - ord('0') if isLimit else 1
          for d in range(0 if isNum else 1, up + 1):
             if pre == 0 or d == 0:
                res += dfs(i + 1, d, isLimit and d == up, True)
          return res
       return dfs(0, 0, True, False) + 1
    

    # 902. 最大为 N 的数字组合 (Numbers At Most N Given Digit Set)
    def atMostNGivenDigitSet(self, digits: List[str], n: int) -> int:
       s = str(n)
       m = len(s)

       @cache
       def dfs(i: int, isLimit: bool, isNum: bool) -> int:
          if i == m:
             return isNum
          res = 0
          if not isNum:
             res = dfs(i + 1, False, False)
          up = s[i] if isLimit else '9'
          for d in digits:
             if d > up:
                break
             res += dfs(i + 1, isLimit and d == up, True)
          return res
       return dfs(0, True, False)
    
    # 1012. 至少有 1 位重复的数字 (Numbers With Repeated Digits)
    def numDupDigitsAtMostN(self, n: int) -> int:
       s = str(n)
       m = len(s)

       @cache
       def dfs(i: int, mask: int, isLimit: bool, isNum: bool) -> int:
          if i == m:
             return isNum
          res = 0
          if not isNum:
             res = dfs(i + 1, mask, False, False)
          up = ord(s[i]) - ord('0') if isLimit else 9
          for d in range(0 if isNum else 1, up + 1):
             if not (mask >> d) & 1:
                res += dfs(i + 1, mask | (1 << d), isLimit and up == d, True)
          return res
       return n - dfs(0, 0, True, False)
    

    # 1105. 填充书架 (Filling Bookcase Shelves)
    def minHeightShelves(self, books: List[List[int]], shelfWidth: int) -> int:
       n = len(books)

       @cache
       def dfs(i: int) -> int:
          if i == n:
             return 0
          w = 0
          maxH = 0
          res = inf
          for j in range(i, n):
             w += books[j][0]
             maxH = max(maxH, books[j][1])
             if w > shelfWidth:
                break
             res = min(res, dfs(j + 1) + maxH)
          return res
       return dfs(0)
     
   # 1043. 分隔数组以得到最大和 (Partition Array for Maximum Sum)
    def maxSumAfterPartitioning(self, arr: List[int], k: int) -> int:
      n = len(arr)

      @cache
      def dfs(i: int) -> int:
         if i == n:
            return 0
         res = 0
         m = 0
         for j in range(i, min(n, i + k)):
            m = max(m, arr[j])
            res = max(res, dfs(j + 1) + m * (j - i + 1))
         return res
      return dfs(0)
    
   # 1039. 多边形三角剖分的最低得分 (Minimum Score Triangulation of Polygon)
    def minScoreTriangulation(self, values: List[int]) -> int:
       n = len(values)

       @cache
       def dfs(i: int, j: int) -> int:
          if j - i <= 1:
             return 0
          res = inf
          for k in range(i + 1, j):
             res = min(res, dfs(i, k) + dfs(k, j) + values[i] * values[k] * values[j])
          return res
       return dfs(0, n - 1)
    
    # 1048. 最长字符串链 (Longest String Chain)
    def longestStrChain(self, words: List[str]) -> int:
       n = len(words)
       words.sort(key = len)

       @cache
       def dfs(i: int) -> int:
          if i == n:
             return 0
          res = 0
          for j in range(i + 1, n):
             if len(words[j]) - len(words[i]) < 1:
                continue
             if len(words[j]) - len(words[i]) > 1:
                break
             if check(words[i], words[j]):
                res = max(res, dfs(j))
          return res + 1
       
       def check(s1: str, s2: str) -> bool:
          i = 0
          j = 0
          n1 = len(s1)
          n2 = len(s2)
          f = False
          while i < n1 and j < n2:
             if s1[i] == s2[j]:
                i += 1
                j += 1
             elif not f:
                f = True
                j += 1
             else:
                return False
          return True
       res = 0
       for i in range(0, n):
          res = max(res, dfs(i))
       return res
    
    # 1911. 最大子序列交替和 (Maximum Alternating Subsequence Sum)
    def maxAlternatingSum(self, nums: List[int]) -> int:
       n = len(nums)

       @cache
       def dfs(i: int, j: int) -> int:
          if i == n:
             return 0
          return max(dfs(i + 1, j), dfs(i + 1, j ^ 1) + (-nums[i] if j else nums[i]))
       return dfs(0, 0)
    
   # 1463. 摘樱桃 II (Cherry Pickup II)
    def cherryPickup(self, grid: List[List[int]]) -> int:
       m = len(grid)
       n = len(grid[0])

       @cache
       def dfs(i: int, j: int, k: int) -> int:
          if i == m:
             return 0
          res = 0
          for nj in range(max(0, j - 1), min(j + 2, n)):
             for nk in range(max(0, k - 1), min(k + 2, n)):
                res = max(res, dfs(i + 1, nj, nk) + grid[i][nj] + (0 if nj == nk else grid[i][nk]))
          return res
       return dfs(1, 0, n - 1) + grid[0][0] + grid[0][n - 1]
    


    # 1473. 粉刷房子 III (Paint House III)
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
       
       @cache
       def dfs(i: int, pre: int, left: int) -> int:
          if i == m:
             return 0 if left == 0 else inf
          if left > m - i:
             return inf
          if left < 0:
             return inf
          if houses[i]:
             return dfs(i + 1, houses[i], left if houses[i] == pre else left - 1)
          res = inf
          for c in range(1, n + 1):
             res = min(res, dfs(i + 1, c, left if c == pre else left - 1) + cost[i][c - 1])
          return res
       res = dfs(0, n + 1, target)
       return -1 if res == inf else res
    
   # 1524. 和为奇数的子数组数目 (Number of Sub-arrays With Odd Sum)
    def numOfSubarrays(self, arr: List[int]) -> int:
       even = 1
       odd = 0
       res = 0
       sum = 0
       m = 10 ** 9 + 7
       for x in arr:
          sum += x
          if (sum & 1) == 0:
             res += odd
             even += 1
          else:
             res += even
             odd += 1
       return res % m
    
   # 1525. 字符串的好分割数目 (Number of Good Ways to Split a String
    def numSplits(self, s: str) -> int:
       n = len(s)
       left = [0] * n
       right = [0] * n
       m = 0
       for i in range(n):
          m |= 1 << (ord(s[i]) - ord('a'))
          left[i] = m
       m = 0
       for i in range(n - 1, -1, -1):
          m |= 1 << (ord(s[i]) - ord('a'))
          right[i] = m
       res = 0
       for i in range(1, n):
          res += left[i - 1].bit_count() == right[i].bit_count()
       return res
    
    # 2466. 统计构造好字符串的方案数 (Count Ways To Build Good Strings)
    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
       m = 10 ** 9 + 7
       @cache
       def dfs(i: int) -> int:
          if i < 0:
             return 0
          if i == 0:
             return 1
          return (dfs(i - zero) + dfs(i - one)) % m
       res = 0
       for i in range(low, high + 1):
          res += dfs(i)
       return res % m
    
    # 53. 最大子数组和 (Maximum Subarray)
    # 剑指 Offer 42. 连续子数组的最大和
    def maxSubArray(self, nums: List[int]) -> int:
      n = len(nums)
      res = -inf
      pre = 0
      m = 0
      for num in nums:
         pre += num
         res = max(res, pre - m)
         m = min(m, pre)
      return res
    
    # 72. 编辑距离 (Edit Distance)
    def minDistance(self, word1: str, word2: str) -> int:
       m = len(word1)
       n = len(word2)

       @cache
       def dfs(i: int, j: int) -> int:
          if i < 0:
             return j + 1
          if j < 0:
             return i + 1
          if word1[i] == word2[j]:
             return dfs(i - 1, j - 1)
          return min(dfs(i - 1, j), dfs(i, j - 1), dfs(i - 1, j - 1)) + 1
       return dfs(m - 1, n - 1)
    
    # 87. 扰乱字符串 (Scramble String)
    def isScramble(self, s1: str, s2: str) -> bool:
       n = len(s1)
       @cache
       def dfs(i: int, j: int, l: int) -> bool:
          if l == 1:
             return s1[i] == s2[j]
          if s1[i: i + l] == s2[j: j + 1]:
             return True
          cnts = [0] * 26
          for k in range(i, i + l):
             cnts[ord(s1[k]) - ord('a')] += 1
          for k in range(j, j + l):
             cnts[ord(s2[k]) - ord('a')] -= 1
          for c in cnts:
             if c:
                return False
          for k in range(1, l):
             if dfs(i, j, k) and dfs(i + k, j + k, l - k):
                return True
             if dfs(i, j + l - k, k) and dfs(i + k, j, l - k):
                return True
          return False
       return dfs(0, 0, n)
    
    # 5. 最长回文子串 (Longest Palindromic Substring)
    def longestPalindrome(self, s: str) -> str:
       n = len(s)
       dp = [[False] * (n) for _ in range(0, n)]
       l = -1
       r = -1
       for i in range(n - 1, -1, -1):
          for j in range(i, n):
             if i == j or j - i == 1 and s[i] == s[j] or j - i > 1 and dp[i + 1][j - 1] and s[i] == s[j]:
                dp[i][j] = True
                if j - i >= r - l:
                   r = j
                   l = i
       return s[l: r + 1]
    
   # 509. 斐波那契数 (Fibonacci Number)
    def fib(self, n: int) -> int:
       
       @cache
       def dfs(i: int) -> int:
          if i <= 1:
             return i
          return dfs(i - 1) + dfs(i - 2)
       return dfs(n)
    
   # 467. 环绕字符串中唯一的子字符串 (Unique Substrings in Wraparound String)
    def findSubstringInWraproundString(self, s: str) -> int:
       n = len(s)
       dp = [0] * 26
       cnt = 0
       for i in range(n):
          if i == 0 or (ord(s[i]) - ord(s[i - 1])) % 26 != 1:
             cnt = 1
          else:
             cnt += 1
          dp[ord(s[i]) - ord('a')] = max(dp[ord(s[i]) - ord('a')], cnt)
       return sum(dp)
    

    # 879. 盈利计划 (Profitable Schemes)
    def profitableSchemes(self, n: int, minProfit: int, group: List[int], profit: List[int]) -> int:
       m = len(profit)
       mod = 10 ** 9 + 7

       @cache
       def dfs(i: int, j: int, k: int) -> int:
          if j > n:
             return 0
          if i == m:
             return k >= minProfit
          return (dfs(i + 1, j, k) + dfs(i + 1, j + group[i], min(minProfit, k + profit[i]))) % mod
       return dfs(0, 0, 0)
    
    # 688. 骑士在棋盘上的概率 (Knight Probability in Chessboard)
    def knightProbability(self, n: int, k: int, row: int, column: int) -> float:
       dirs = [[1, 2],[1, -2],[-1, 2],[-1, -2],[2, 1],[2, -1],[-2, 1],[-2, -1]]
       
       @cache
       def dfs(i: int, j: int, left: int) -> float:
          if i < 0 or i >= n or j < 0 or j >= n:
             return 0.0
          if left == 0:
             return 1.0
          res = 0.0
          for dx, dy in dirs:
             nx = i + dx
             ny = j + dy
             res += dfs(nx, ny, left - 1)
          return res / 8.0
       return dfs(row, column, k)
    
    # 712. 两个字符串的最小ASCII删除和 (Minimum ASCII Delete Sum for Two Strings)
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
       n1 = len(s1)
       n2 = len(s2)

       @cache
       def dfs(i: int, j: int) -> int:
          if i == n1:
             return sum(ord(c) for c in s2[j:])
          if j == n2:
             return sum(ord(c) for c in s1[i:])
          if s1[i] == s2[j]:
             return dfs(i + 1, j + 1)
          return min(dfs(i + 1, j) + ord(s1[i]), dfs(i, j + 1) + ord(s2[j]))
       return dfs(0, 0)
    
    # 1143. 最长公共子序列 (Longest Common Subsequence) 
    # LCR 095. 最长公共子序列
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
       n1 = len(text1)
       n2 = len(text2)

       @cache
       def dfs(i: int, j: int) -> int:
          if i == n1 or j == n2:
             return 0
          if text1[i] == text2[j]:
             return dfs(i + 1, j + 1) + 1
          return max(dfs(i + 1, j), dfs(i, j + 1))
       return dfs(0, 0)
    
    # 256. 粉刷房子 --plus
    # LCR 091. 粉刷房子 
    def minCost(self, costs: List[List[int]]) -> int:
       n = len(costs)

       @cache
       def dfs(i: int, j: int) -> int:
          if i == n:
             return 0
          res = inf
          for k in range(3):
             if j != k:
                res = min(res, dfs(i + 1, k) + costs[i][k])
          return res
       return dfs(0, -1)
    
    # 343. 整数拆分 (Integer Break)
    # 剑指 Offer 14- I. 剪绳子
    def cuttingRope(self, n: int) -> int:
       
       @cache
       def dfs(i: int) -> int:
          if i == 2:
             return 1
          if i < 2:
             return 0
          res = 0
          for j in range(1, n - 1):
             res = max(res, dfs(i - j) * j, j * (i - j))
          return res
       return dfs(n)
    
    # 剑指 Offer 47. 礼物的最大价值 
    def maxValue(self, grid: List[List[int]]) -> int:
       m = len(grid)
       n = len(grid[0])
       dirs = [[0, 1], [1, 0]]

       @cache
       def dfs(i: int, j: int) -> int:
          if i < 0 or i >= m or j < 0 or j >= n:
             return 0
          res = 0
          for dx, dy in dirs:
             nx = i + dx
             ny = j + dy
             res = max(res, dfs(nx, ny))
          return res + grid[i][j]
       return dfs(0, 0)
    
    # 面试题 08.14. 布尔运算 
    def countEval(self, s: str, result: int) -> int:
       n = len(s)
       
       @cache
       def dfs(i: int, j: int, ret: int) -> int:
          if i == j:
             return 1 if ord(s[i]) - ord('0') == ret else 0
          res = 0
          for k in range(i + 1, j, 2):
             op = ord(s[k])
             for x in range(2):
                for y in range(2):
                   if check(x, y, op, ret):
                      res += dfs(i, k - 1, x) * dfs(k + 1, j, y)
          return res
       
       def check(a: int, b: int, op: int, res: int) -> bool:
          if op == ord('|'):
             return (a | b) == res
          if op == ord('&'):
             return (a & b) == res
          return (a ^ b) == res
       return dfs(0, n - 1, result)
    
    # 1444. 切披萨的方案数 (Number of Ways of Cutting a Pizza)
    def ways2(self, pizza: List[str], k: int) -> int:
       m = len(pizza)
       n = len(pizza[0])
       MOD = 10 ** 9 + 7
       pre = [[0] * (n + 1) for _ in range(m + 1)]
       for i in range(1, m + 1):
          for j in range(1, n + 1):
             pre[i][j] = pre[i - 1][j] + pre[i][j - 1] - pre[i - 1][j - 1] + (1 if pizza[i - 1][j - 1] == 'A' else 0)
       def getCounts(x1: int, y1: int, x2: int, y2: int) -> int:
          return pre[x2 + 1][y2 + 1] - pre[x1][y2 + 1] - pre[x2 + 1][y1] + pre[x1][y1]
       @cache
       def dfs(i: int, j: int, l: int) -> int:
          c = getCounts(i, j, m - 1, n - 1)
          if l == 1:
             return 1 if c > 0 else 0
          if m - 1 - i + n - 1 - j + 1 < l:
             return 0
          if l > c:
             return 0
          res = 0
          for k in range(i, m - 1):
             cnt = getCounts(i, j, k, n - 1)
             if cnt > 0:
                res += dfs(k + 1, j, l - 1)
                res %= MOD
          for k in range(j, n - 1):
             cnt = getCounts(i, j, m - 1, k)
             if cnt > 0:
                res += dfs(i, k + 1, l - 1)
                res %= MOD
          return res
       return dfs(0, 0, k)
    
    # 1162. 地图分析 (As Far from Land as Possible)
    def maxDistance(self, grid: List[List[int]]) -> int:
       n = len(grid)
       q = []
       for i in range(n):
          for j in range(n):
             if grid[i][j] == 1:
                q.append((i, j))
                grid[i][j] = 0
             else:
                grid[i][j] = -1
       if len(q) == n * n:
          return -1
       dirs = [[0, -1],[0, 1],[1, 0],[-1, 0]]
       res = 0
       while q:
          res += 1
          size = len(q)
          for i in range(size):
            cur = q.pop(0)
            x = cur[0]
            y = cur[1]
            for dx, dy in dirs:
               nx = x + dx 
               ny = y + dy
               if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == -1:
                  grid[nx][ny] = grid[x][y] + 1
                  q.append((nx, ny))
       return res - 1
    
    # 935. 骑士拨号器 (Knight Dialer)
    def knightDialer(self, n: int) -> int:
       MOD = 10 ** 9 + 7

       @cache
       def dfs(i: int, left: int) -> int:
          if left == 0:
             return 1
          res = 0
          if i == 1:
             res += dfs(6, left - 1) + dfs(8, left - 1)
             res %= MOD
          if i == 2:
             res += dfs(7, left - 1) + dfs(9, left - 1)
             res %= MOD
          if i == 3:
             res += dfs(4, left - 1) + dfs(8, left - 1)
             res %= MOD
          if i == 4:
             res += dfs(3, left - 1) + dfs(9, left - 1) + dfs(0, left - 1)
             res %= MOD
          if i == 6:
             res += dfs(1, left - 1) + dfs(7, left - 1) + dfs(0, left - 1)
             res %= MOD
          if i == 7:
             res += dfs(2, left - 1) + dfs(6, left - 1)
             res %= MOD
          if i == 8:
             res += dfs(1, left - 1) + dfs(3, left - 1)
             res %= MOD
          if i == 9:
             res += dfs(2, left - 1) + dfs(4, left - 1)
             res %= MOD
          if i == 0:
             res += dfs(4, left - 1) + dfs(6, left - 1)
             res %= MOD
          return res
       res = 0
       for i in range(10):
          res += dfs(i, n - 1)
          res %= MOD
       return res

    # 931. 下降路径最小和 (Minimum Falling Path Sum)
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
       n = len(matrix)

       @cache
       def dfs(i: int, j: int) -> int:
          if i == n:
             return 0
          res = inf
          for k in range(max(0, j - 1), min(n - 1, j + 1) + 1):
             res = min(res, dfs(i + 1, k) + matrix[i][k])
          return res
       res = inf
       for j in range(0, n):
          res = min(res, dfs(1, j) + matrix[0][j])
       return res
    
    # 983. 最低票价 (Minimum Cost For Tickets)
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
       arr = [False] * 366
       for d in days:
          arr[d] = True

       @cache
       def dfs(i: int) -> int:
          if i > 365:
             return 0
          return min(dfs(i + 1) + (costs[0] if arr[i] else 0), dfs(i + 7) + costs[1], dfs(i + 30) + costs[2])
       return dfs(1)
    
    # 1155. 掷骰子等于目标和的方法数 (Number of Dice Rolls With Target Sum)
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
       MOD = 10 ** 9 + 7
       
       @cache
       def dfs(i: int, j: int) -> int:
          if i == n:
             return 1 if j == target else 0
          if j > target:
             return 0
          if n - i > target - j:
             return 0
          if (n - i) * k < target - j:
             return 0
          res = 0
          for x in range(1, k + 1):
             res += dfs(i + 1, j + x)
             res %= MOD
          return res
       return dfs(0, 0)
             

    # 1289. 下降路径最小和 II (Minimum Falling Path Sum II)
    def minFallingPathSum(self, grid: List[List[int]]) -> int:
       n = len(grid)

       @cache
       def dfs(i: int, j: int) -> int:
          if i == n:
             return 0
          res = inf
          for k in range(0, n):
             if k == j:
                continue
             res = min(res, dfs(i + 1, k) + grid[i][k])
          return res
       return dfs(0, n)
    

    # 845. 数组中的最长山脉 (Longest Mountain in Array)
    def longestMountain(self, arr: List[int]) -> int:
       n = len(arr)
       i = 0
       j = 1
       res = 0
       while j < n:
          f1 = False
          f2 = False
          while j < n and arr[j - 1] < arr[j]:
             f1 = True
             j += 1
          if f1:
             while j < n and arr[j] < arr[j - 1]:
                f2 = True
                j += 1
          if f1 and f2:
             res = max(res, j - i)
             i = j - 1
          else:
             i = j
             j += 1
       return res
    
    # 813. 最大平均值和的分组 (Largest Sum of Averages)
    def largestSumOfAverages(self, nums: List[int], k: int) -> float:
       n = len(nums)
       @cache
       def dfs(i: int, j: int) -> float:
          if i == n:
             return 0
          if j == 0:
             return -inf
          sum = 0
          res = 0
          for x in range(i, n):
             sum += nums[x]
             res = max(res, dfs(x + 1, j - 1) + sum / (x - i + 1))
          return res
       return dfs(0, k)
    
    # 1035. 不相交的线 (Uncrossed Lines) 
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
       n1 = len(nums1)
       n2 = len(nums2)

       @cache
       def dfs(i: int, j: int) -> int:
          if i == n1 or j == n2:
             return 0
          return dfs(i + 1, j + 1) + 1 if nums1[i] == nums2[j] else max(dfs(i + 1, j), dfs(i, j + 1))
       return dfs(0, 0)
    
    # 2547. 拆分数组的最小代价 (Minimum Cost to Split an Array)
    def minCost(self, nums: List[int], k: int) -> int:
       n = len(nums)

       @cache
       def dfs(i: int) -> int:
          if i == n:
             return 0
          res = inf
          cnts = [0] * n
          unique = 0
          for j in range(i, n):
              cnts[nums[j]] += 1
              if cnts[nums[j]] == 1:
                 unique += 1
              if cnts[nums[j]] == 2:
                 unique -= 1
              res = min(res, dfs(j + 1) + j - i + 1 - unique + k)
          return res
       return dfs(0)
    
    # 2742. 给墙壁刷油漆 (Painting the Walls)
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
       n = len(cost)
       @cache
       def dfs(i: int, j: int) -> int:
          if i == n:
             return inf if j < 0 else 0
          if n - i <= j:
             return 0
          return min(dfs(i + 1, time[i] + j) + cost[i], dfs(i + 1, j - 1))
       return dfs(0, 0)
    
    # 27. 移除元素 (Remove Element) 
    def removeElement(self, nums: List[int], val: int) -> int:
        i = 0
        j = 0
        n = len(nums)
        while j < n:
            while j < n and nums[j] == val:
                j += 1
            if j < n:
                nums[i] = nums[j]
                i += 1
                j += 1
        return i
    
    # 26. 删除有序数组中的重复项 (Remove Duplicates from Sorted Array) 
    def removeDuplicates(self, nums: List[int]) -> int:
        i = 0
        j = 0
        n = len(nums)
        while j < n:
            x = nums[j]
            while j < n and nums[j] == x:
                j += 1
            nums[i] = x
            i += 1
        return i
    
    # 1349. 参加考试的最大学生数 (Maximum Students Taking Exam)
    def maxStudents(self, seats: List[List[str]]) -> int:
       m = len(seats)
       n = len(seats[0])
       u = (1 << n) - 1
       arr = [0] * m
       for i in range(m):
          mask = 0
          for j in range(n):
             if seats[i][j] == '#':
                mask |= 1 << j
          arr[i] = mask

       @cache
       def dfs(i: int, j: int) -> int:
          if i < 0:
             return 0
          c = u ^ ((j << 1 | j >> 1 | arr[i]) & u)
          k = c
          # 不选
          res = dfs(i - 1, 0)
          while k:
             if k & (k >> 1) == 0:
                # 选
                res = max(res, dfs(i - 1, k) + k.bit_count())
             k = (k - 1) & c
          return res
       return dfs(m - 1, 0)
    
    # 2305. 公平分发饼干 (Fair Distribution of Cookies)
    def distributeCookies(self, cookies: List[int], k: int) -> int:
       n = len(cookies)
       u = (1 << n) - 1
       arr = [0] * (1 << n)
       for i in range(1, 1 << n):
          index = (i & -i).bit_length() - 1
          arr[i] = arr[i ^ (1 << index)] + cookies[index]

       @cache
       def dfs(i: int, m: int) -> int:
          if i == k or m == u:
             return 0 if i == k and m == u else inf
          res = inf
          c = m ^ u
          s = c
          while s > 0:
             res = min(res, max(dfs(i + 1, s | m), arr[s]))
             s = (s - 1) & c
          return res
       return dfs(0, 0)
    
    # 2741. 特别的排列 (Special Permutations)
    def specialPerm(self, nums: List[int]) -> int:
       n = len(nums)
       u = (1 << n) - 1
       MOD = 10 ** 9 + 7

       @cache
       def dfs(i: int, m: int) -> int:
          if m == u:
             return 1
          res = 0
          j = u ^ m
          while j > 0:
             index = (j & -j).bit_length() - 1
             if nums[index] % nums[i] == 0 or nums[i] % nums[index] == 0:
                res += dfs(index, m | (1 << index))
                res %= MOD
             j &= j - 1
          return res
       res = 0
       for i in range(n):
          res += dfs(i, 1 << i)
          res %= MOD
       return res
    
    # 2002. 两个回文子序列长度的最大乘积 (Maximum Product of the Length of Two Palindromic Subsequences)
    def maxProduct(self, s: str) -> int:
       n = len(s)
       arr = [False] * (1 << n)
       for i in range(1, 1 << n):
          cnt = i.bit_count()
          lead = i.bit_length() - 1
          trail = (i & -i).bit_length() - 1
          if s[lead] == s[trail] and (cnt <= 2 or arr[i ^ (1 << lead) ^ (1 << trail)]):
             arr[i] = True
       res = 1
       u = (1 << n) - 2
       while u:
          cnt = u.bit_count()
          if arr[u] and res < cnt * (n - cnt):
             s = ((1 << n) - 1) ^ u
             j = s
             while j > 0:
                if arr[j]:
                   res = max(res, cnt * j.bit_count())
                j = (j - 1) & s
          u -= 1
       return res
    
    # 2472. 不重叠回文子字符串的最大数目 (Maximum Number of Non-overlapping Palindrome Substrings)
    def maxPalindromes(self, s: str, k: int) -> int:
       n = len(s)
       arr = [[0] * n for _ in range(n)]
       for i in range(n - 1, -1, -1):
          for j in range(i, n):
             if s[i] == s[j] and (j - i < 2 or arr[i + 1][j - 1]):
                arr[i][j] = 1
       
       @cache
       def dfs(i: int) -> int:
          if i == n:
             return 0
          if n - i < k:
             return 0
          res = dfs(i + 1)
          for j in range(i + k - 1, n):
             if arr[i][j]:
                res = max(res, dfs(j + 1) + 1)
          return res
       return dfs(0)
    
    # 2444. 统计定界子数组的数目 (Count Subarrays With Fixed Bounds)
    def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
       n = len(nums)
       res = 0
       pos = -1
       p1 = -1
       p2 = -1
       for i in range(0, n):
          if nums[i] > maxK or nums[i] < minK:
             pos = i
             p1 = -1
             p2 = -1
          else:
             if nums[i] == minK:
                p1 = i
             if nums[i] == maxK:
                p2 = i
          if p1 != -1 and p2 != -1:
             res += min(p1, p2) - pos
       return res
          
    # 2430. 对字母串可执行的最大删除数 (Maximum Deletions on a String)
    def deleteString(self, s: str) -> int:
       n = len(s)
       if len(set(s)) == 1: return n
       lca = [[0] * (n + 1) for _ in range(n + 1)]
       for i in range(n - 1, -1, -1):
          for j in range(n - 1, i, -1):
             if s[i] == s[j]:
                lca[i][j] = lca[i + 1][j + 1] + 1
       f = [0] * n
       for i in range(n - 1, -1, -1):
           for j in range(1, (n - i) // 2 + 1):
               if lca[i][i + j] >= j:  # 说明 s[i:i+j] == s[i+j:i+2*j]
                    f[i] = max(f[i], f[i + j])
           f[i] += 1
       return f[0]
    
    # 2435. 矩阵中和能被 K 整除的路径 (Paths in Matrix Whose Sum Is Divisible by K)
    def numberOfPaths(self, grid: List[List[int]], k: int) -> int:
       m = len(grid)
       n = len(grid[0])
       MOD = 10 ** 9 + 7
       @cache
       def dfs(i: int, j: int, s: int) -> int:
          if i == m - 1 and j == n - 1:
             return 1 if (s + grid[i][j]) % k == 0 else 0
          res = 0
          if i + 1 < m:
             res += dfs(i + 1, j, (s + grid[i][j]) % k)
             res %= MOD
          if j + 1 < n:
             res += dfs(i, j + 1, (s + grid[i][j]) % k)
             res %= MOD
          return res
       res = dfs(0, 0, 0)
       dfs.cache_clear()
       return res
    

    # 2318. 不同骰子序列的数目 (Number of Distinct Roll Sequences)
    def distinctSequences(self, n: int) -> int:
       MOD = 10 ** 9 + 7
       
       @cache
       def dfs(i: int, j: int, k: int) -> int:
          if i == n:
             return 1
          res = 0
          for s in range(1, 7):
             if s != j and s != k and (i == 0 or gcd(s, j) == 1):
                res += dfs(i + 1, s, j)
                res %= MOD
          return res
       return dfs(0, 0, 0)
    
    # 2304. 网格中的最小路径代价 (Minimum Path Cost in a Grid)
    def minPathCost(self, grid: List[List[int]], moveCost: List[List[int]]) -> int:
       m = len(grid)
       n = len(grid[0])

       @cache
       def dfs(i: int, j: int):
          if i == m - 1:
             return grid[i][j]
          res = inf
          for k in range(n):
             res = min(res, dfs(i + 1, k) + moveCost[grid[i][j]][k])
          return res + grid[i][j]

       res = inf
       for j in range(0, n):
          res = min(res, dfs(0, j))
       return res
    
    # 88. 合并两个有序数组 (Merge Sorted Array)
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
       p1 = m - 1
       p2 = n - 1
       p = m + n - 1
       while p1 >= 0 and p2 >= 0:
          if nums1[p1] > nums2[p2]:
             nums1[p] = nums1[p1]
             p1 -= 1
          else:
             nums1[p] = nums2[p2]
             p2 -= 1
          p -= 1
       while p2 >= 0:
          nums1[p] = nums2[p2]
          p -= 1
          p2 -= 1

    # 2320. 统计放置房子的方式数 (Count Number of Ways to Place Houses)
    def countHousePlacements(self, n: int) -> int:
       MOD = 10 ** 9 + 7
       
       @cache
       def dfs(i: int) -> int:
          if i >= n:
             return 1
          return dfs(i + 1) + dfs(i + 2)
       res = dfs(0)
       return res ** 2 % MOD
    
    # 2550. 猴子碰撞的方法数 (Count Collisions of Monkeys on a Polygon)
    def monkeyMove(self, n: int) -> int:
        MOD = 10 ** 9 + 7
        return (pow(2, n, MOD) - 2) % MOD
    
    # 617. 合并二叉树 (Merge Two Binary Trees) 
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
       if not root1:
          return root2
       if not root2:
          return root1
       root1.val += root2.val
       root1.left = self.mergeTrees(root1.left, root2.left)
       root1.right = self.mergeTrees(root1.right, root2.right)
       return root1
    
    # 50. Pow(x, n) (Pow(x, n))
    def myPow(self, x: float, n: int) -> float:
       return pow(x, n)
    
    # 31. 下一个排列 (Next Permutation)
    def nextPermutation(self, nums: List[int]) -> None:
       n = len(nums)
       i = n - 1
       while i - 1 >= 0 and nums[i - 1] >= nums[i]:
          i -= 1
       if i == 0:
          nums.sort()
          return
       i -= 1

       j = n - 1
       while i < j:
          if nums[i] < nums[j]:
             temp = nums[i]
             nums[i] = nums[j]
             nums[j] = temp
             nums[i + 1:] = sorted(nums[i + 1:])
             break
          j -= 1

    # 1771. 由子序列构造的最长回文串的长度 (Maximize Palindrome Length From Subsequences)
    def longestPalindrome(self, word1: str, word2: str) -> int:
       n1 = len(word1)
       n2 = len(word2)
       s = word1 + word2
       @cache
       def dfs(i: int, j: int, k: int) -> int:
          if i > j:
             return 0
          if i == j:
             return 1
          if not (i < n1 and j >= n1) and not k:
             return 0
          if s[i] == s[j]:
             return dfs(i + 1, j - 1, 1) + 2
          return max(dfs(i + 1, j, k), dfs(i, j - 1, k))
       res = dfs(0, n1 + n2 - 1, 0)
       return res if res > 1 else 0

    # 2147. 分隔长廊的方案数 (Number of Ways to Divide a Long Corridor)
    def numberOfWays(self, corridor: str) -> int:
       n = len(corridor)
       res = 1
       MOD = 10 ** 9 + 7
       j = -1
       cnt = 0
      
       for i, c in enumerate(corridor):
          if c == 'S':
             cnt += 1
             if j >= 0 and (cnt & 1) == 1:
               res *= i - j
               res %= MOD
             j = i
       return res if (cnt & 1) == 0 and cnt >= 2 else 0
    
    # 2100. 适合打劫银行的日子 (Find Good Days to Rob the Bank)
    def goodDaysToRobBank(self, security: List[int], time: int) -> List[int]:
        n = len(security)
        left = [0] * n
        for i in range(n - 2, -1, -1):
            if security[i] <= security[i + 1]:
                left[i] = left[i + 1] + 1
        right = [0] * n
        for i in range(1, n):
            if security[i] <= security[i - 1]:
                right[i] = right[i - 1] + 1
        res = []
        for i in range(time, n - time):
            if left[i] >= time and right[i] >= time:
                res.append(i)
        return res
    
    # 2086. 从房屋收集雨水需要的最少水桶数 (Minimum Number of Food Buckets to Feed the Hamsters)
    def minimumBuckets(self, hamsters: str) -> int:
        n = len(hamsters)
        res = 0
        i = 0
        while i < n:
            if hamsters[i] == 'H':
                if i + 1 < n and hamsters[i + 1] == '.':
                    res += 1
                    i += 3
                elif i - 1 >= 0 and hamsters[i - 1] == '.':
                    res += 1
                    i += 1
                else:
                    return -1
            else:
                i += 1
        return res
    
    # 2063. 所有子字符串中的元音 (Vowels of All Substrings)
    def countVowels(self, word: str) -> int:
        res = 0
        m = 0
        for c in "aeiou":
           m |= 1 << (ord(c) - ord('a'))
        n = len(word)
        for i, c in enumerate(word):
            if (m >> (ord(c) - ord('a'))) & 1:
                 res += (i + 1) * (n - i)
        return res
    
    # 1986. 完成任务的最少工作时间段 (Minimum Number of Work Sessions to Finish the Tasks)
    def minSessions(self, tasks: List[int], sessionTime: int) -> int:
       n = len(tasks)
       sum = [0] * (1 << n)
       u = (1 << n) - 1
       for i in range(1, 1 << n):
         index = (i & -i).bit_length() - 1
         sum[i] = sum[i ^ (1 << index)] + tasks[index]
       
       @cache
       def dfs(i: int) -> int:
          if i == u:
             return 0
          res = inf
          c = u ^ i
          j = c
          while j:
             if sum[j] <= sessionTime:
                res = min(res, dfs(i | j) + 1)
             j = (j - 1) & c
          return res
       return dfs(0)
    
    # 1799. N 次操作后的最大分数和 (Maximize Score After N Operations)
    def maxScore(self, nums: List[int]) -> int:
       n = len(nums)
       u = (1 << n) - 1

       @cache
       def dfs(i: int, m: int) -> int:
          if m == u:
             return 0
          c = m ^ u
          j = c
          res = 0
          while j:
             if j.bit_count() == 2:
               index1 = (j & -j).bit_length() - 1
               index2 = (j & (j - 1)).bit_length() - 1
               g = gcd(nums[index1], nums[index2])
               res = max(res, dfs(i + 1, m | j) + (i + 1) * g)
             j = (j - 1) & c
          return res
       return dfs(0, 0)
    
    # 1931. 用三种不同颜色为网格涂色 (Painting a Grid With Three Different Colors)
    def colorTheGrid(self, m: int, n: int) -> int:
       MOD = 10 ** 9 + 7

       def check(i: int) -> bool:
          pre = -1
          cnt = m
          while cnt > 0:
             if pre == i % 3:
                return False
             cnt -= 1
             pre = i % 3
             i //= 3
          return True

       def legal(a: int, b: int) -> bool:
          cnt = m
          while cnt > 0:
             if a % 3 == b % 3:
                return False
             cnt -= 1
             a //= 3
             b //= 3
          return True
       
       @cache
       def dfs(j: int, i: int) -> int:
          if j == n:
             return 1
          res = 0
          for k in s:
             if legal(k, i):
                res += dfs(j + 1, k)
                res %= MOD
          return res
       s = set()
       for i in range(pow(3, m)):
          if check(i):
             s.add(i)
       res = 0 
       for i in s:
          res += dfs(1, i)
          res %= MOD
       return res
    
    # 剑指 Offer 46. 把数字翻译成字符串
    def translateNum(self, num: int) -> int:
       s = str(num)
       l = len(s)

       @cache
       def dfs(i: int) -> int:
          if i == l:
             return 1
          res = dfs(i + 1)
          if s[i] == '1' and i + 1 < l:
             res += dfs(i + 2)
          elif s[i] == '2' and i + 1 < l and '0' <= s[i + 1] <= '5':
             res += dfs(i + 2)
          return res
       return dfs(0)
    
    # LCP 19. 秋叶收藏集 
    def minimumOperations(self, leaves: str) -> int:
       n = len(leaves)

       @cache
       def dfs(i: int, s: int) -> int:
          if i == n:
             return 0 if s == 2 else inf
          if i == 0:
             return dfs(i + 1, s) if leaves[i] == 'r' else dfs(i + 1, s) + 1
          if s == 0:
             if leaves[i] == 'r':
                # 不变 // 变  
                return min(dfs(i + 1, s), dfs(i + 1, s + 1) + 1)
             # 不变 // 变 
             return min(dfs(i + 1, s + 1), dfs(i + 1, s) + 1) 
          if s == 1:
             if leaves[i] == 'r':
                # 不变 // 变
                return min(dfs(i + 1, s + 1), dfs(i + 1, s) + 1)
             # 不变 // 变
             return min(dfs(i + 1, s), dfs(i + 1, s + 1) + 1)
          if leaves[i] == 'r':
             return dfs(i + 1, s)
          return dfs(i + 1, s) + 1
       return dfs(0, 0)
    
    # 2439. 最小化数组中的最大值 (Minimize Maximum of Array)
    def minimizeArrayValue(self, nums: List[int]) -> int:
       def check(target: int) -> bool:
          have = 0
          for num in nums:
             if num <= target:
                have += target - num
             elif num - target <= have:
                have -= num - target
             else:
                return False
          return True
       left = 0
       right = 10 ** 9
       res = 0
       while left <= right:
          mid = left + ((right - left) >> 1)
          if check(mid):
             res = mid
             right = mid - 1
          else:
             left = mid + 1
       return res
    
    # 2328. 网格图中递增路径的数目 (Number of Increasing Paths in a Grid)
    def countPaths(self, grid: List[List[int]]) -> int:
       m = len(grid)
       n = len(grid[0])
       MOD = 10 ** 9 + 7
       dirs = [0, 1], [0, -1], [1, 0], [-1, 0]
      
       @cache
       def dfs(i: int, j: int) -> int:
          res = 1
          for dx, dy in dirs:
             nx = i + dx
             ny = j + dy
             if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] > grid[i][j]:
                res += dfs(nx, ny)
          return res
       res = 0
       for i in range(m):
          for j in range(n):
             res += dfs(i, j)
             res %= MOD
       return res
    
    # 1320. 二指输入的的最小距离 (Minimum Distance to Type a Word Using Two Fingers)
    def minimumDistance(self, word: str) -> int:
      n = len(word)

      def dis(i: int, j: int) -> int:
         return abs(i // 6 - j // 6) + abs(i % 6 - j % 6)

      @cache
      def dfs(i: int, p1: int, p2: int) -> int:
         if i == n:
            return 0
         return min(dfs(i + 1, ord(word[i]) - ord('A'), p2) + dis(ord(word[i]) - ord('A'), p1),
                    dfs(i + 1, p1,ord(word[i]) - ord('A')) + dis(ord(word[i]) - ord('A'), p2))
      res = inf
      for i in range(26):
         for j in range(26):
            res = min(res, dfs(0, i, j))
      return res
    
    # 2682. 找出转圈游戏输家 (Find the Losers of the Circular Game)
    def circularGameLosers(self, n: int, k: int) -> List[int]:
       vis = [False] * n
       step = 1
       i = 0
       while not vis[i]:
          vis[i] = True
          i = (i + k * step) % n
          step += 1
       res = []
       for i in range(n):
          if not vis[i]:
             res.append(i + 1)
       return res
    
    # 1402. 做菜顺序 (Reducing Dishes)
    def maxSatisfaction(self, satisfaction: List[int]) -> int:
       n = len(satisfaction)
       satisfaction.sort()

       @cache
       def dfs(i: int, j: int) -> int:
          if i == n:
             return 0
          return max(dfs(i + 1, j), dfs(i + 1, j + 1) + satisfaction[i] * (j + 1))
       return dfs(0, 0)
    
    # 1493. 删掉一个元素以后全为 1 的最长子数组 (Longest Subarray of 1's After Deleting One Element)
    def longestSubarray(self, nums: List[int]) -> int:
       n = len(nums)
       left = 0
       right = 0
       zero = 0
       res = 0
       while right < n:
          if nums[right] == 0:
             zero += 1
          while zero > 1:
             if nums[left] == 0:
                zero -= 1
             left += 1
          res = max(res, right - left)
          right += 1
       return res
    
    # 1493. 删掉一个元素以后全为 1 的最长子数组 (Longest Subarray of 1's After Deleting One Element)
    def longestSubarray2(self, nums: List[int]) -> int:
       n = len(nums)
       a = 0
       b = 0
       res = 0
       for num in nums:
          if num == 1:
             a += 1
             b += 1
             res = max(res, a)
          else:
             a = b
             b = 0
       return min(n - 1, res)
    
    # 1458. 两个子序列的最大点积 (Max Dot Product of Two Subsequences)
    def maxDotProduct(self, nums1: List[int], nums2: List[int]) -> int:
       n1 = len(nums1)
       n2 = len(nums2)

       @cache
       def dfs(i: int, j: int, k: int) -> int:
          if i == n1 or j == n2:
             return 0 if k else -inf
          return max(dfs(i + 1, j, k), dfs(i, j + 1, k), dfs(i + 1, j + 1, k), dfs(i + 1, j + 1, min(1, k + 1)) + nums1[i] * nums2[j])
       return dfs(0, 0, 0)
    
    # 1449. 数位成本和为目标值的最大数字 (Form Largest Integer With Digits That Add up to Target)
    def largestNumber(self, cost: List[int], target: int) -> str:
       
       @cache
       def dfs(i: int) -> str:
          if i == target:
             return ""
          res = ""
          for j in range(0, 9):
             if cost[j] + i <= target:
                s = str(j + 1) + dfs(cost[j] + i)
                if not s.__contains__("0"):
                   if len(s) > len(res) or len(s) == len(res) and s > res:
                      res = s
          return "0" if res == "" else res
       return dfs(0)
    
    # 389. 找不同 (Find the Difference)
    def findTheDifference(self, s: str, t: str) -> str:
        res = 0
        for c in s + t:
            res ^= ord(c)
        return chr(res)
    
    # 152. 乘积最大子数组 (Maximum Product Subarray)
    def maxProduct(self, nums: List[int]) -> int:
       n = len(nums)

       @cache
       def dfs(i: int, j: int) -> int:
          if i < 0:
             return 1
          res = nums[i]
          if j == 0:
             if nums[i] < 0:
                res = max(res, dfs(i - 1, j ^ 1) * nums[i])
             else:
                res = max(res, dfs(i - 1, j) * nums[i])
          else:
             if nums[i] < 0:
                res = min(res, dfs(i - 1, j ^ 1) * nums[i])
             else:
                res = min(res, dfs(i - 1, j) * nums[i])
          return res
       return max(dfs(i, 0) for i in range(n))
    
    # 838. 推多米诺 (Push Dominoes)
    def pushDominoes(self, dominoes: str) -> str:
       n = len(dominoes)
       res = ['.'] * n
       i = 0
       j = 0
       while j < n:
          if dominoes[j] == 'L':
             if dominoes[i] == 'R':
                for k in range(i + 1, i + (j - i - 1) // 2 + 1):
                   res[k] = 'R'
                for k in range(j - (j - i - 1) // 2, j):
                   res[k] = 'L'
             else:
                for k in range(i, j):
                   res[k] = 'L'
             res[j] = 'L'
             i = j
          elif dominoes[j] == 'R':
             if dominoes[i] == 'R':
                for k in range(i, j):
                   res[k] = 'R'
             res[j] = 'R'
             i = j
          j += 1

       j = n - 1
       while j >= 0 and dominoes[j] == '.':
          j -= 1
       if j >= 0 and dominoes[j] == 'R':
          for k in range(j, n):
             res[k] = 'R'
          
       return "".join(res)
    
    # 828. 统计子串中的唯一字符 (Count Unique Characters of All Substrings of a Given String)
    def uniqueLetterString(self, s: str) -> int:
       n = len(s)
       last = [-1] * 26
       pre = [-1] * n
       for i, c in enumerate(s):
          j = ord(c) - ord('A')
          pre[i] = last[j]
          last[j] = i
       last = [n] * 26
       suf = [n] * n
       for i in range(n - 1, -1, -1):
          j = ord(s[i]) - ord('A')
          suf[i] = last[j]
          last[j] = i
       res = 0
       for i in range(n):
          res += (i - pre[i]) * (suf[i] - i)
       return res
    
    # 1879. 两个数组最小的异或值之和 (Minimum XOR Sum of Two Arrays)
    def minimumXORSum(self, nums1: List[int], nums2: List[int]) -> int:
       n = len(nums1)
       u = (1 << n) - 1

       @cache
       def dfs(i: int, m: int) -> int:
          if i == n:
             return 0
          c = u ^ m
          res = inf
          while c:
             index = (c & -c).bit_length() - 1
             res = min(res, dfs(i + 1, m | (1 << index)) + (nums1[i] ^ nums2[index]))
             c &= c - 1
          return res
       return dfs(0, 0)

          
    # 1655. 分配重复整数 (Distribute Repeating Integers)
    def canDistribute(self, nums: List[int], quantity: List[int]) -> bool:
       c = Counter(nums)
       cnt = list(c.values())
       n = len(cnt)
       q = len(quantity)
       u = (1 << q) - 1
       sum = [0] * (1 << q)
       for i in range(1, 1 << q):
          index = (i & -i).bit_length() - 1
          sum[i] = sum[i ^ (1 << index)] + quantity[index]
       
       @cache
       def dfs(i: int, m: int) -> bool:
          if m == u:
             return True
          if i == n:
             return False
          # 不选
          if dfs(i + 1, m):
             return True
          c = u ^ m
          j = c
          while j:
             # 选
             if cnt[i] >= sum[j] and dfs(i + 1, m | j):
                return True
             j = (j - 1) & c
          return False
       return dfs(0, 0)
    
    # 713. 乘积小于 K 的子数组 (Subarray Product Less Than K)
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
       n = len(nums)
       i = 0
       j = 0
       res = 0
       mul = 1
       while j < n:
          mul *= nums[j]
          while i <= j and mul >= k:
             mul /= nums[i]
             i += 1
          res += j - i + 1
          j += 1
       return res
    

    # 8013. 范围中美丽整数的数目 (Number of Beautiful Integers in the Range)
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
       def cal(s: str) -> int:
         @cache
         def dfs(i: int, diff: int, m: int, isLimit: bool, isNum: bool) -> int:
            if i == len(s):
               return 1 if isNum and diff == 0 and m == 0 else 0
            res = 0
            if not isNum:
               res = dfs(i + 1, 0, 0, False, False)
            up = int(s[i]) if isLimit else 9
            for j in range(0 if isNum else 1, up + 1):
               res += dfs(i + 1, diff + (1 if j % 2 == 0 else -1), (m * 10 + j) % k, isLimit and j == up, True)
            return res
         return dfs(0, 0, 0, True, False)
       return cal(str(high)) - cal(str(low - 1))
    
    # 6941. 将三个组排序 (Sorting Three Groups)
    def minimumOperations(self, nums: List[int]) -> int:
       n = len(nums)

       @cache
       def dfs(i: int, j: int) -> int:
          if i == n:
             return 0
          res = inf
          if nums[i] >= j:
             res = min(res, dfs(i + 1, nums[i]))
          for k in range(j, 4):
             if nums[i] != k:
                res = min(res, dfs(i + 1, k) + 1)
          return res
       return dfs(0, 1) 
    
    # 2828. 判别首字母缩略词 (Check if a String Is an Acronym of Words)
    def isAcronym(self, words: List[str], s: str) -> bool:
       return len(words) == len(s) and all(ss[0] == c2 for ss, c2 in zip(words, s))
    
    # 2829. k-avoiding 数组的最小总和 (Determine the Minimum Sum of a k-avoiding Array)
    def minimumSum(self, n: int, k: int) -> int:
       res = 0
       cnt = 0
       i = 1
       while i <= k // 2 and cnt < n:
          res += i
          i += 1
          cnt += 1
       while cnt < n:
          res += k
          k += 1
          cnt += 1
       return res
    
    # 2830. 销售利润最大化 (Maximize the Profit as the Salesman)
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
       map = [[] for _ in range(n)]
       
       for start, end, gold in offers:
          map[start].append((end, gold))
       
       @cache
       def dfs(i: int) -> int:
          if i >= n:
             return 0
          res = dfs(i + 1)
          for end, gold in map[i]:
             res = max(res, dfs(end + 1) + gold)
          return res
       return dfs(0)
    
    # 849. 到最近的人的最大距离 (Maximize Distance to Closest Person)
    def maxDistToClosest(self, seats: List[int]) -> int:
       n = len(seats)
       last = -1
       res = 0
       for i in range(n):
          if seats[i]:
             if last == -1:
                res = max(res, i)
             else:
                res = max(res, (i - last) // 2)
             last = i
       res = max(res, n - 1 - last)
       return res
    
    # 139. 单词拆分 (Word Break)
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
       n = len(s)
       dict = set(wordDict)
       
       @cache
       def dfs(i: int) -> bool:
          if i == n:
             return True
          for j in range(i, n):
             if dict.__contains__(s[i:j + 1]) and dfs(j + 1):
                return True
          return False
       return dfs(0)
    
    # 907. 子数组的最小值之和 (Sum of Subarray Minimums)
    def sumSubarrayMins(self, arr: List[int]) -> int:
       MOD = 10 ** 9 + 7
       n = len(arr)
       st = []
       left = [-1] * n
       right = [n] * n
       for i in range(n):
          while st and arr[st[-1]] >= arr[i]:
             right[st.pop()] = i
          if st:
             left[i] = st[-1]
          st.append(i)
      #  st = []
      #  for i in range(n - 1, -1, -1):
      #     while st and arr[st[-1]] > arr[i]:
      #        st.pop()
      #     if st:
      #        right[i] = st[-1]
      #     st.append(i)
       res = 0
       for i in range(n):
          res += (right[i] - i) * (i - left[i]) * arr[i]
          res %= MOD
       return res
    
    # 1856. 子数组最小乘积的最大值 (Maximum Subarray Min-Product)
    def maxSumMinProduct(self, nums: List[int]) -> int:
       n = len(nums)
       pre = [0] * (n + 1)
       for i, v in enumerate(nums):
          pre[i + 1] = pre[i] + v
       st = []
       left = [-1] * n 
       for i in range(n):
          while st and nums[st[-1]] >= nums[i]:
             st.pop()
          if st:
             left[i] = st[-1]
          st.append(i)
       st.clear()
       right = [n] * n
       for i in range(n - 1, -1, -1):
          while st and nums[st[-1]] >= nums[i]:
             st.pop()
          if st:
             right[i] = st[-1]
          st.append(i)
       res = 0
       for i in range(n):
          res = max(res, (pre[right[i]] - pre[left[i] + 1]) * nums[i])
       MOD = 10 ** 9 + 7
       return res % MOD 
    
    # 2104. 子数组范围和 (Sum of Subarray Ranges)
    def subArrayRanges(self, nums: List[int]) -> int:
       n = len(nums)
       leftMax = [-1] * n
       rightMax = [n] * n
       leftMin = [-1] * n
       rightMin = [n] * n
       stMax = []
       stMin = []
       for i in range(n):
          while stMax and nums[stMax[-1]] <= nums[i]:
             rightMax[stMax.pop()] = i
          if stMax:
             leftMax[i] = stMax[-1]
          stMax.append(i)

          while stMin and nums[stMin[-1]] >= nums[i]:
             rightMin[stMin.pop()] = i
          if stMin:
             leftMin[i] = stMin[-1]
          stMin.append(i)

       res = 0
       for i in range(n):
          res += nums[i] * ((rightMax[i] - i) * (i - leftMax[i]) - (rightMin[i] - i) * (i - leftMin[i]))
       return res
    
    # 2818. 操作使得分最大 (Apply Operations to Maximize Score)
    def maximumScore(self, nums: List[int], k: int) -> int:
       n = len(nums)
       MAX = 10 ** 5 + 1
       MOD = 10 ** 9 + 7
       ### 需要把下面这一部分定义在class之外 否则会超时
       omega = [0] * MAX
       for i in range(2, MAX):
          if omega[i] == 0:
             for j in range(i, MAX, i):
                omega[j] += 1
       ###
       left = [-1] * n
       right = [n] * n
       st = []
       for i in range(n):
          while st and omega[nums[st[-1]]] < omega[nums[i]]:
             right[st.pop()] = i
          if st:
             left[i] = st[-1]
          st.append(i)
       res = 1
       for i, v, l, r in sorted(zip(range(n), nums, left, right), key = lambda z: -z[1]):
          tot = (r - i) * (i - l)
          if tot >= k:
             res = res * pow(v, k, MOD) % MOD
             break
          res = res * pow(v, tot, MOD) % MOD
          k -= tot
       return res

    # 1267. 统计参与通信的服务器 (Count Servers that Communicate)
    def countServers(self, grid: List[List[int]]) -> int:
       m = len(grid)
       n = len(grid[0])
       row = [0] * m
       col = [0] * n
       res = 0
       for i in range(m):
          for j in range(n):
             if grid[i][j]:
                row[i] += 1
                col[j] += 1
       for i in range(m):
          for j in range(n):
                if grid[i][j] and (row[i] > 1 or col[j] > 1):
                   res += 1
       return res 
    
    # 209. 长度最小的子数组 (Minimum Size Subarray Sum)
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        n = len(nums)
        res = inf
        i = 0
        j = 0
        sum = 0
        while j < n:
            sum += nums[j]
            while i <= j and sum >= target:
                res = min(res, j - i + 1)
                sum -= nums[i]
                i += 1
            j += 1
        return 0 if res == inf else res
    
    # 204. 计数质数 (Count Primes)
    def countPrimes(self, n: int) -> int:
        arr = [True] * n
        for i in range(2, n):
            if arr[i]:
                for j in range(i + i, n, i):
                    arr[j] = False
        res = 0
        for i in range(2, n):
            if arr[i]:
                res += 1
        return res
    
    # 207. 课程表 (Course Schedule)
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        map = [[] for _ in range(numCourses)]
        deg = [0] * numCourses
        for a, b in prerequisites:
            map[b].append(a)
            deg[a] += 1
        q = []
        for i, d in enumerate(deg):
            if not d:
                q.append(i)
        cnt = 0
        while q:
            cnt += 1
            node = q.pop()
            for nxt in map[node]:
                deg[nxt] -= 1
                if not deg[nxt]:
                    q.append(nxt)
        return cnt == numCourses
    
    # 228. 汇总区间 (Summary Ranges)
    def summaryRanges(self, nums: List[int]) -> List[str]:
       res = []
       n = len(nums)
       i = 0
       while i < n:
          cur = str(nums[i])
          j = i + 1
          while j < n and nums[j] - nums[j - 1] == 1:
             j += 1
          if j - i != 1:
             cur = cur + "->" + str(nums[j - 1])
          res.append(cur)
          i = j
       return res
    
    # 1494. 并行课程 II (Parallel Courses II)
    def minNumberOfSemesters(self, n: int, relations: List[List[int]], k: int) -> int:
       pre = [0] * n
       u = (1 << n) - 1
       for r in relations:
          pre[r[1] - 1] |= 1 << (r[0] - 1)

       @cache
       def dfs(i: int) -> int:
          if i == u:
             return 0
          c = u ^ i
          candidate = 0
          while c:
             index = (c & -c).bit_length() - 1
             if (pre[index] | i) == i:
                candidate |= 1 << index
             c &= c - 1
          if candidate.bit_count() <= k:
             return dfs(i | candidate) + 1
          res = inf
          j = candidate
          while j:
             if j.bit_count() <= k:
                res = min(res, dfs(i | j) + 1)
             j = (j - 1) & candidate
          return res
       return dfs(0)
    
    # 1416. 恢复数组 (Restore The Array)
    def numberOfArrays(self, s: str, k: int) -> int:
       n = len(s)
       MOD = 10 ** 9 + 7
       
       @cache
       def dfs(i: int) -> int:
          if i == n:
             return 1
          if s[i] == '0':
             return 0
          sum = 0
          res = 0
          for j in range(i, n):
             sum = sum * 10 + int(s[j])
             if sum > k:
                break
             res += dfs(j + 1)
             res %= MOD
          return res
       return dfs(0)
    
    # 1420. 生成数组 (Build Array Where You Can Find The Maximum Exactly K Comparisons)
    def numOfArrays(self, n: int, m: int, k: int) -> int:
       MOD = 10 ** 9 + 7

       @cache
       def dfs(i: int, maximum: int, j: int) -> int:
          if i == n:
             return j == k
          if j == k:
             return pow(maximum, n - i, MOD)
          res = maximum * dfs(i + 1, maximum, j) % MOD
          for x in range(maximum + 1, m + 1):
             res += dfs(i + 1, x, j + 1)
             res %= MOD
          return res
       return dfs(0, 0, 0)
    
    # 417. 太平洋大西洋水流问题 (Pacific Atlantic Water Flow)
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
       m = len(heights)
       n = len(heights[0])
       arr0 = [[False] * n for _ in range(m)]
       q = []
       for i in range(n):
          arr0[0][i] = True
          q.append((0, i))
       for i in range(1, m):
          arr0[i][0] = True
          q.append((i, 0))
       dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]]
       while q:
          cur = q.pop()
          x = cur[0]
          y = cur[1]
          for dx, dy in dirs:
             nx = x + dx
             ny = y + dy
             if 0 <= nx < m and 0 <= ny < n and not arr0[nx][ny] and heights[nx][ny] >= heights[x][y]:
                arr0[nx][ny] = True
                q.append((nx, ny))
       arr1 = [[False] * n for _ in range(m)]
       for i in range(n):
          arr1[m - 1][i] = True
          q.append((m - 1, i))
       for i in range(m - 1):
          arr1[i][n - 1] = True
          q.append((i, n - 1))

       while q:
          cur = q.pop()
          x = cur[0]
          y = cur[1]
          for dx, dy in dirs:
             nx = x + dx
             ny = y + dy
             if 0 <= nx < m and 0 <= ny < n and not arr1[nx][ny] and heights[nx][ny] >= heights[x][y]:
                arr1[nx][ny] = True
                q.append((nx, ny))
       res = []
       for i in range(m):
          for j in range(n):
             if arr0[i][j] and arr1[i][j]:
                res.append((i, j))
       return res
    
    # 56. 合并区间 (Merge Intervals)
    # LCR 074. 合并区间
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
       n = len(intervals)
       intervals.sort(key=lambda k: k[0])
       res = []
       i = 0
       while i < n:
          left = intervals[i][0]
          right = intervals[i][1]
          j = i + 1
          while j < n and intervals[j][0] <= right:
             right = max(right, intervals[j][1])
             j += 1
          res.append((left, right))
          i = j
       return res
    
    # 1976. 到达目的地的方案数 (Number of Ways to Arrive at Destination)
    def countPaths(self, n: int, roads: List[List[int]]) -> int:
       MOD = 10 ** 9 + 7
       # g = collections.defaultdict(list)
       g = [[] for _ in range(n)]
       for a, b, t in roads:
          g[a].append((b, t))
          g[b].append((a, t))
       dis = [inf] * n
       dis[0] = 0
       dp = [0] * n
       dp[0] = 1
       q = [[0, 0]]
       heapq.heapify(q)
       while q:
          cur = heapq.heappop(q)
          d = cur[0]
          x = cur[1]
          for nxt in g[x]:
            y = nxt[0]
            dt = nxt[1]
            if d + dt < dis[y]:
               dis[y] = d + dt
               dp[y] = dp[x]
               heapq.heappush(q, (d + dt, y))
            elif d + dt == dis[y]:
               dp[y] += dp[x]
               dp[y] %= MOD
       return dp[n - 1]
    
    # 1547. 切棍子的最小成本 (Minimum Cost to Cut a Stick)
    def minCost(self, n: int, cuts: List[int]) -> int:
       cuts.sort()

       @cache
       def dfs(l: int, r: int, i: int, j: int) -> int:
          if r <= l or j <= i:
             return 0
          return min(dfs(l, cuts[k], i, k) + dfs(cuts[k], r, k + 1, j) for k in range(i, j)) + r - l
       return dfs(0, n, 0, len(cuts))
    
    # 3. 无重复字符的最长子串 (Longest Substring Without Repeating Characters)
    def lengthOfLongestSubstring(self, s: str) -> int:
       n = len(s)
       i = 0
       j = 0
       cnt = [0] * 128
       res = 0
       while j < n:
          c = ord(s[j])
          cnt[c] += 1
          while cnt[c] > 1:
             cnt[ord(s[i])] -= 1
             i += 1
          res = max(res, j - i + 1)
          j += 1
       return res
    
    # 2. 两数相加 (Add Two Numbers)
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
       class ListNode:
          def __init__(self, val=0, next=None):
             self.val = val
             self.next = next
       dummy = cur = ListNode()
       carry = 0
       while l1 or l2 or carry:
          if l1:
             carry += l1.val
             l1 = l1.next
          if l2:
             carry += l2.val
             l2 = l2.next
          cur.next = ListNode(carry % 10)
          cur = cur.next
          carry //= 10
       if l1:
          cur.next = l1
       if l2:
          cur.next = l2
       return dummy.next
    
    # 49. 字母异位词分组 (Group Anagrams)
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
       map = collections.defaultdict(list)
       for s in strs:
          key = "".join(sorted(s))
          map[key].append(s)
       return list(map.values())
    
    # 49. 字母异位词分组 (Group Anagrams)
    def groupAnagrams2(self, strs: List[str]) -> List[List[str]]:
       map = collections.defaultdict(list)
       for s in strs:
          cnts = [0] * 26
          for c in s:
             cnts[ord(c) - ord('a')] += 1
          map[tuple(cnts)].append(s)
       return list(map.values())
    
    # 1. 两数之和 (Two Sum)
    def twoSum(self, nums: List[int], target: int) -> List[int]:
       map = collections.defaultdict(int)
       for i, v in enumerate(nums):
          if target - v in map:
             return [map[target - v], i]
          map[v] = i
       return None
    
    # 560. 和为 K 的子数组 (Subarray Sum Equals K)
    def subarraySum(self, nums: List[int], k: int) -> int:
       mp = collections.defaultdict(int)
       mp[0] = 1
       pre = 0
       res = 0
       for num in nums:
          pre += num
          res += mp[pre - k]
          mp[pre] += 1
       return res
    
    # 57. 插入区间 (Insert Interval)
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
       n = len(intervals)
       i = 0
       res = []
       f = False
       while i < n:
          l = intervals[i][0]
          r = intervals[i][1]
          L = newInterval[0]
          R = newInterval[1]
          if r < L:
             res.append(intervals[i])
             i += 1
          elif R < l:
             res.append(newInterval)
             f = True
             while i < n:
                res.append(intervals[i])
                i += 1
          else:
             f = True
             left = min(l, L)
             right = max(r, R)
             i += 1
             while i < n and intervals[i][0] <= right:
                right = max(right, intervals[i][1])
                i += 1
             res.append([left, right])
             while i < n:
                res.append(intervals[i])
                i += 1
       if not f:
          res.append(newInterval)
       return res
    
    # 342. 4的幂 (Power of Four)
    def isPowerOfFour(self, n: int) -> bool:
        return n > 0 and n.bit_count() == 1 and (n & 0b10101010101010101010101010101010) == 0
    
    # 338. 比特位计数 (Counting Bits)
    # LCR 003. 比特位计数
    def countBits(self, n: int) -> List[int]:
       res = [0] * (n + 1)
       for i in range(1, n + 1):
          res[i] = res[(i & -i) ^ i] + 1
       return res
    
    # 338. 比特位计数 (Counting Bits)
    # LCR 003. 比特位计数
    def countBits2(self, n: int) -> List[int]:
       res = [0] * (n + 1)
       for i in range(1, n + 1):
          res[i] = res[i & (i - 1)] + 1
       return res
    
    # 36. 有效的数独 (Valid Sudoku)
    def isValidSudoku(self, board: List[List[str]]) -> bool:
       row = [0] * 9
       col = [0] * 9
       mat = [[0] * 3 for _ in range(3)]
       for i in range(9):
          for j in range(9):
             if board[i][j] != '.':
                x = int(board[i][j])
                if row[i] & (1 << x):
                   return False
                row[i] |= 1 << x
                if col[j] & (1 << x):
                   return False
                col[j] |= 1 << x
                if mat[i // 3][j // 3] & (1 << x):
                   return False
                mat[i // 3][j // 3] |= 1 << x
       return True
    
    # 48. 旋转图像 (Rotate Image)
    def rotate(self, matrix: List[List[int]]) -> None:
       n = len(matrix)
       for i in range(n):
          for j in range(i):
             t = matrix[i][j]
             matrix[i][j] = matrix[j][i]
             matrix[j][i] = t
       for i in range(n):
          matrix[i].reverse()

    # 823. 带因子的二叉树 (Binary Trees With Factors)
    def numFactoredBinaryTrees(self, arr: List[int]) -> int:
       arr.sort()
       s = set(arr)
       MOD = 10 ** 9 + 7

       @cache
       def dfs(i: int) -> int:
          res = 1
          for j in arr:
             if j * arr[0] > i:
                break
             if i % j == 0 and s.__contains__(i / j):
                res += dfs(j) * dfs(i / j)
                res %= MOD
          return res
       return sum(dfs(i) for i in arr) % MOD
    
    # 1335. 工作计划的最低难度 (Minimum Difficulty of a Job Schedule)
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
       n = len(jobDifficulty)

       @cache
       def dfs(i: int, j: int) -> int:
          if i == n or j == 0:
             return 0 if i == n and j == 0 else inf
          if n - i < j:
             return inf
          res = inf
          m = 0
          for k in range(i, n):
             m = max(m, jobDifficulty[k])
             res = min(res, dfs(k + 1, j - 1) + m)
          return res
       res = dfs(0, d)
       return -1 if res == inf else res
    
    # 1312. 让字符串成为回文串的最少插入次数 (Minimum Insertion Steps to Make a String Palindrome)
    def minInsertions(self, s: str) -> int:
       n = len(s)

       @cache
       def dfs(i: int, j: int) -> int:
          if i >= j:
             return 0
          if s[i] == s[j]:
             return dfs(i + 1, j - 1)
          return min(dfs(i + 1, j), dfs(i, j - 1)) + 1
       return dfs(0, n - 1)
    
    # 1301. 最大得分的路径数目 (Number of Paths with Max Score)
    def pathsWithMaxScore(self, board: List[str]) -> List[int]:
       n = len(board)
       ways = [[0] * n for _ in range(n)]
       ways[n - 1][n - 1] = 1
       MOD = 10 ** 9 + 7

       @cache
       def dfs(i: int, j: int) -> int:
          if i == n - 1 and j == n - 1:
             return 0
          if board[i][j] == 'X':
             return -inf
          res1 = -inf
          res2 = -inf
          res3 = -inf
          if i + 1 < n:
             res1 = dfs(i + 1, j)
          if j + 1 < n:
             res2 = dfs(i, j + 1)
          if i + 1 < n and j + 1 < n:
             res3 = dfs(i + 1, j + 1)
          res = max(res1, res2, res3)
          if res != -inf:
             if res == res1:
                ways[i][j] += ways[i + 1][j]
                ways[i][j] %= MOD
             if res == res2:
                ways[i][j] += ways[i][j + 1]
                ways[i][j] %= MOD
             if res == res3:
                ways[i][j] += ways[i + 1][j + 1]
                ways[i][j] %= MOD
          if not (i == 0 and j == 0):
             res += ord(board[i][j]) - ord('0')
          return res
       res = dfs(0, 0)
       if res < 0:
          return [0, 0]
       return [res, ways[0][0]]
    
    # 1278. 分割回文串 III (Palindrome Partitioning III)
    def palindromePartition(self, s: str, k: int) -> int:

       @cache
       def dfs(i: int, j: int) -> int:
          if i >= j:
             return 0
          if s[i] == s[j]:
             return dfs(i + 1, j - 1)
          return dfs(i + 1, j - 1) + 1
       
       n = len(s)
       
       @cache
       def dfs2(i: int, j: int) -> int:
          if i == n or j == k:
             return 0 if i == n and j == k else inf
          if n - i < k - j:
             return inf
          res = inf
          for x in range(i, n):
             res = min(res, dfs2(x + 1, j + 1) + dfs(i, x))
          return res
       return dfs2(0, 0)
    
    # 1223. 掷骰子模拟 (Dice Roll Simulation)
    def dieSimulator(self, n: int, rollMax: List[int]) -> int:
       MOD = 10 ** 9 + 7
       
       @cache
       def dfs(i: int, j: int, k: int) -> int:
          if i == n:
             return 1
          res = 0
          for x in range(1, 7):
             if x != j:
                res += dfs(i + 1, x, rollMax[x - 1] - 1)
             elif k:
                res += dfs(i + 1, x, k - 1)
             res %= MOD
          return res
       return dfs(0, 0, 0)
    
    # 1220. 统计元音字母序列的数目 (Count Vowels Permutation)
    def countVowelPermutation(self, n: int) -> int:
       MOD = 10 ** 9 + 7
       dic = collections.defaultdict(list)
       dic[0].append(1)
       dic[1].append(0)
       dic[1].append(2)
       dic[2].append(0)
       dic[2].append(1)
       dic[2].append(3)
       dic[2].append(4)
       dic[3].append(2)
       dic[3].append(4)
       dic[4].append(0)
       dic[5].append(0)
       dic[5].append(1)
       dic[5].append(2)
       dic[5].append(3)
       dic[5].append(4)

       @cache
       def dfs(i: int, j: int) -> int:
          if i == n:
             return 1
          res = 0
          for k in dic[j]:
             res += dfs(i + 1, k)
             res %= MOD
          return res
       return dfs(0, 5)

    # 2833. 距离原点最远的点 (Furthest Point From Origin)
    def furthestDistanceFromOrigin(self, moves: str) -> int:
       return abs(moves.count('L') - moves.count('R')) + moves.count('_')
    
    # 2834. 找出美丽数组的最小和 (Find the Minimum Possible Sum of a Beautiful Array)
    def minimumPossibleSum(self, n: int, target: int) -> int:
       res = 0
       cnt = 0
       i = 1
       while i <= target // 2 and cnt < n:
          res += i
          i += 1
          cnt += 1
       while cnt < n:
          res += target
          target += 1
          cnt += 1
       return res
    
    # 2787. 将一个数字表示成幂的和的方案数 (Ways to Express an Integer as Sum of Powers)
    def numberOfWays(self, n: int, x: int) -> int:
       MOD = 10 ** 9 + 7
       
       @cache
       def dfs(i: int, j: int) -> int:
          if j == n:
             return 1
          if i > n:
             return 0
          s = pow(i, x)
          if s > n:
             return 0
          if j > n:
             return 0
          res = dfs(i + 1, j) 
          if j + s <= n:
             res += dfs(i + 1, j + s)
          return res % MOD
       return dfs(1, 0)
    
    # 2646. 最小化旅行的价格总和 (Minimize the Total Price of the Trips)
    def minimumTotalPrice(self, n: int, edges: List[List[int]], price: List[int], trips: List[List[int]]) -> int:
       cnts = [0] * n
       def paths(x: int, fa: int) -> bool:
          if x == end:
             cnts[x] += 1
             return True
          for y in g[x]:
             if y != fa and paths(y, x):
                cnts[x] += 1
                return True
          return False
       
       def dfs(x: int, fa: int) -> List[int]:
          a = (price[x] // 2) * cnts[x]
          b = price[x] * cnts[x]
          for y in g[x]:
             if y != fa:
                c = dfs(y, x)
                a += c[1]
                b += min(c[0], c[1])
          return [a, b]
       g = [[] for _ in range(n)]
       for a, b in edges:
          g[a].append(b)
          g[b].append(a)
       end = -1
       for a, b in trips:
          end = b
          paths(a, -1)
       return min(x for x in dfs(0, -1))
    
    # 2581. 统计可能的树根数目 (Count Number of Possible Root Nodes)
    def rootCount(self, edges: List[List[int]], guesses: List[List[int]], k: int) -> int:
       n = len(edges) + 1
       g = [[] for _ in range(n)]
       for a, b in edges:
          g[a].append(b)
          g[b].append(a)
       s = set()
       for guess in guesses:
          s.add(tuple(guess))
       cur = 0
       res = 0

       def dfs(x: int, fa: int) -> None:
          for y in g[x]:
             if y != fa :
                if s.__contains__(tuple([x, y])):
                   nonlocal cur
                   cur += 1
                dfs(y, x)

       def reroot(x: int, fa: int, c: int) -> None:
          if c >= k:
             nonlocal res
             res += 1
          for y in g[x]:
             copy = c
             if y != fa :
                if s.__contains__(tuple([x, y])):
                   copy -= 1
                if s.__contains__(tuple([y, x])):
                   copy += 1
                reroot(y, x, copy)

       dfs(0, -1)
       reroot(0, -1, cur)
       return res
    
    # 1654. 到家的最少跳跃次数 (Minimum Jumps to Reach Home)
    def minimumJumps(self, forbidden: List[int], a: int, b: int, x: int) -> int:
       s = set(forbidden)
       vis = [[False] * 2 for _ in range(6000)]
       vis[0][0] = True
       q = [[0, 0]]
       res = 0
       while q:
          size = len(q)
          for _ in range(size):
             cur = q.pop(0)
             node = cur[0]
             leftStep = cur[1]
             if node == x:
                return res

             if node + a < len(vis) and not vis[node + a][0] and not s.__contains__(node + a):
                vis[node + a][0] = True
                q.append([node + a, 0])
             
             if leftStep == 0 and node - b >= 0 and not vis[node - b][1] and not s.__contains__(node - b):
                vis[node - b][1] = True
                q.append([node - b, 1])
          res += 1
       return -1
    

    # 2522. 将字符串分割成值不超过 K 的子字符串 (将字符串分割成值不超过 K 的子字符串)
    def minimumPartition(self, s: str, k: int) -> int:
       n = len(s)

       @cache
       def dfs(i: int) -> int:
          if i == n:
             return 0
          res = inf
          val = 0
          for j in range(i, n):
             val = val * 10 + int(s[j])
             if val > k:
                break
             res = min(res, dfs(j + 1) + 1)
          return res
       res = dfs(0)
       return res if res < inf else -1
    

    # 2522. 将字符串分割成值不超过 K 的子字符串 (将字符串分割成值不超过 K 的子字符串)
    def minimumPartition2(self, s: str, k: int) -> int:
       n = len(s)
       i = 0
       res = 0
       while i < n:
          val = int(s[i])
          if val > k:
             return -1
          j = i + 1
          while j < n:
             val = val * 10 + int(s[j])
             if val > k:
                break
             j += 1
          i = j
          res += 1
       return res
    
    # 2222. 选择建筑的方案数 (Number of Ways to Select Buildings)
    def numberOfWays(self, s: str) -> int:
       n = len(s)
       right0 = [0] * n
       for i in range(n - 2, -1, -1):
          right0[i] = right0[i + 1] + (1 if s[i + 1] == '0' else 0)
       res = 0
       left0 = 0
       for i, v in enumerate(s):
          if v == '1':
             res += left0 * right0[i]
          else:
             res += (i - left0) * ((n - i - 1) - right0[i])
             left0 += 1
       return res
    
    # 2746. 字符串连接删减字母 (Decremental String Concatenation)
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
       n = len(words)

       @cache
       def dfs(i: int, l: int, r: int) -> int:
          if i == n:
             return 0
          start = ord(words[i][0]) - ord('a')
          end = ord(words[i][-1]) - ord('a')
          return min(dfs(i + 1, l, end) + len(words[i]) - (start == r), dfs(i + 1, start, r) + len(words[i]) - (l == end))
       return dfs(1, ord(words[0][0]) - ord('a'), ord(words[0][-1]) - ord('a')) + len(words[0])
    
    # 2218. 从栈中取出 K 个硬币的最大面值和 (Maximum Value of K Coins From Piles)
    def maxValueOfCoins(self, piles: List[List[int]], k: int) -> int:
       n = len(piles)

       @cache
       def dfs(i: int, j: int) -> int:
          if i == n or j == k:
             return 0
          res = dfs(i + 1, j)
          sum = 0
          cnt = 0
          for x in piles[i]:
             sum += x
             cnt += 1
             if j + cnt > k:
                break
             res = max(res, dfs(i + 1, j + cnt) + sum)
          return res
       return dfs(0, 0)
    
    # 2747. 统计没有收到请求的服务器数目 (Count Zero Request Servers)
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
       logs.sort(key=lambda k:k[1])
       res = [0] * len(queries)
       i = 0
       j = 0
       cur = 0
       cnts = [0] * (n + 1)
       for t, id in sorted(zip(queries, range(len(queries))), key=lambda k:k[0]):
          while j < len(logs) and logs[j][1] <= t:
             cnts[logs[j][0]] += 1
             if cnts[logs[j][0]] == 1:
                cur += 1
             j += 1
          while i < len(logs) and logs[i][1] < t - x:
             cnts[logs[i][0]] -= 1
             if cnts[logs[i][0]] == 0:
                cur -= 1
             i += 1
          res[id] = n - cur
       return res

    # 2370. 最长理想子序列 (Longest Ideal Subsequence)
    def longestIdealString(self, s: str, k: int) -> int:
       n = len(s)
       memo = [[-1] * 27 for _ in range(n)]

       def dfs(i: int, j: int) -> int:
          if i == n:
             return 0
          if memo[i][j] != -1:
             return memo[i][j]
          res = dfs(i + 1, j)
          if j == 26 or abs(ord(s[i]) - ord('a') - j) <= k:
             res = max(res, dfs(i + 1, ord(s[i]) - ord('a')) + 1)
          memo[i][j] = res
          return res
       return dfs(0, 26)
    
    # 2376. 统计特殊整数 (Count Special Integers)
    def countSpecialNumbers(self, n: int) -> int:
       s = str(n)
       l = len(s)

       @cache
       def dfs(i: int, m: int, isLimit: bool, isNum: bool) -> int:
          if i == l:
             return isNum
          res = 0
          if not isNum:
             res = dfs(i + 1, m, False, False)
          up = int(s[i]) if isLimit else 9
          for j in range(0 if isNum else 1, up + 1):
             if ((m >> j) & 1) == 0:
                res += dfs(i + 1, m | (1 << j), isLimit and j == up, True)
          return res
       return dfs(0, 0, True, False)
    
    # 357. 统计各位数字都不同的数字个数 (Count Numbers with Unique Digits)
    def countNumbersWithUniqueDigits(self, n: int) -> int:
       s = str(pow(10, n) - 1)
       l = len(s)

       @cache
       def dfs(i: int, m: int, isLimit: bool, isNum: bool) -> int:
          if i == l:
             return isNum
          res = 0
          if not isNum:
             res = dfs(i + 1, m, False, False)
          up = int(s[i]) if isLimit else 9
          for j in range(0 if isNum else 1, up + 1):
             if ((m >> j) & 1) == 0:
                res += dfs(i + 1, m | (1 << j), isLimit and j == up, True)
          return res
       return dfs(0, 0, True, False) + 1
    
    # 1761. 一个图中连通三元组的最小度数 (Minimum Degree of a Connected Trio in a Graph)
    def minTrioDegree(self, n: int, edges: List[List[int]]) -> int:
       connected = [[False] * n for _ in range(n)]
       deg = [0] * n
       for a, b in edges:
          a -= 1
          b -= 1
          connected[a][b] = True
          connected[b][a] = True
          deg[a] += 1
          deg[b] += 1
       res = inf
       for i in range(n):
          for j in range(i + 1, n):
             if connected[i][j]:
                for k in range(n):
                   if connected[i][k] and connected[j][k]:
                      res = min(res, deg[i] + deg[j] + deg[k] - 6)
       return -1 if res == inf else res
    
    # 1770. 执行乘法运算的最大分数 (Maximum Score from Performing Multiplication Operations)
    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
       n = len(nums)
       m = len(multipliers)

       @cache
       def dfs(i: int, j: int) -> int:
          if i + j >= m:
             return 0
          return max(dfs(i + 1, j) + multipliers[i + j] * nums[i], dfs(i, j + 1) + multipliers[i + j] * nums[n - j - 1])
       return dfs(0, 0)
    
    # 1751. 最多可以参加的会议数目 II (Maximum Number of Events That Can Be Attended II)
    def maxValue(self, events: List[List[int]], k: int) -> int:
       n = len(events)
       events.sort(key=lambda o: o[0])

       @cache
       def dfs(i: int, j: int) -> int:
          if i == n or j == k:
             return 0
          right = bisect.bisect_left(events, events[i][1] + 1, key=lambda e: e[0])
          return max(dfs(i + 1, j), dfs(right, j + 1) + events[i][2])
       return dfs(0, 0)
    
    # 1235. 规划兼职工作 (Maximum Profit in Job Scheduling)
    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
       n = len(startTime)
       jobs = sorted(zip(startTime, endTime, profit))
       
       @cache
       def dfs(i: int) -> int:
          if i == n:
             return 0
          p = bisect.bisect_left(jobs, jobs[i][1], key=lambda k: k[0])
          return max(dfs(i + 1), dfs(p) + jobs[i][2])
       return dfs(0)

