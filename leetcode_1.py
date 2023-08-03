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
    def minimumTotal(self, triangle: List[List[int]]) -> int:
       n = len(triangle)

       @cache
       def dfs(i: int, j: int) -> int:
          if i == n:
             return 0
          return min(dfs(i + 1, j), dfs(i + 1, j + 1)) + triangle[i][j]
       return dfs(0, 0)
       
    # 121. 买卖股票的最佳时机 (Best Time to Buy and Sell Stock)
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
       for i in range(0, m):
          for j in range(0, n):
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
            for j in range( 0 if isNum else 1, up + 1):
               if (not isNum) or abs(j - pre) == 1:
                  res += dfs(i + 1, j, isLimit and j == up, True)
            return res % m
         return dfs(0, 0, True, False)
       return (cal(high) - cal(str(int(low) - 1))) % m
    
    # 233. 数字 1 的个数 (Number of Digit One)
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
             
             

          
       
          
             
             
       
      

       
          

          