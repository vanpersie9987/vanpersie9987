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


from asyncio import FastChildWatcher
from audioop import reverse
from calendar import c
from collections import Counter, deque
import collections
from ctypes.wintypes import _ULARGE_INTEGER
from curses import curs_set
from decimal import Rounded
import enum
from functools import cache
from inspect import modulesbyfile
from itertools import accumulate, pairwise, permutations
from locale import DAY_4
from math import comb, cos, gcd, inf, isqrt, sqrt
from operator import le
from pickletools import read_uint1
from queue import PriorityQueue
from re import M, X
import re
from socket import NI_NUMERICSERV
from ssl import VERIFY_X509_TRUSTED_FIRST
from string import ascii_lowercase, ascii_uppercase
from tabnanny import check
from textwrap import indent
from tkinter import W
from tkinter.tix import Tree
from turtle import RawTurtle, reset, st
from typing import List, Optional
import heapq
import bisect
from zoneinfo import reset_tzpath
from collections import Counter, defaultdict, deque

# curl https://bootstrap.pypa.io/pip/get-pip.py -o get-pip.py
# sudo python3 get-pip.py
# pip3 install sortedcontainers
from sortedcontainers import SortedList


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


class leetcode_1:
    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next

    class Node:
        def __init__(self, val=None, children=None):
            self.val = val
            self.children = children

    # 2463. 最小移动总距离 (Minimum Total Distance Traveled)
    def minimumTotalDistance(self, robot: List[int], factory: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == m:
                return 0
            if j == n:
                return inf
            res = dfs(i, j + 1)
            d = 0
            for k in range(i, min(m, i + factory[j][1])):
                d += abs(factory[j][0] - robot[k])
                res = min(res, dfs(k + 1, j + 1) + d)
            return res

        m = len(robot)
        n = len(factory)
        robot.sort()
        factory.sort()
        return dfs(0, 0)

    # 1478. 安排邮筒 (Allocate Mailboxes)
    def minDistance(self, houses: List[int], k: int) -> int:
        def dis(i: int, j: int) -> int:
            res = 0
            while i < j:
                res += houses[j] - houses[i]
                i += 1
                j -= 1
            return res

        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            if j == k:
                return inf
            return min(dfs(x + 1, j + 1) + dis(i, x) for x in range(i, n))

        n = len(houses)
        houses.sort()
        return dfs(0, 0)

    # 860. 柠檬水找零 (Lemonade Change)
    def lemonadeChange(self, bills: List[int]) -> bool:
        f, t = 0, 0
        for x in bills:
            if x == 5:
                f += 1
            elif x == 10:
                if f == 0:
                    return False
                f -= 1
                t += 1
            else:
                if f >= 1 and t >= 1:
                    f -= 1
                    t -= 1
                elif f >= 3:
                    f -= 3
                else:
                    return False
        return True

    # 1388. 3n 块披萨 (Pizza With 3n Slices)
    def maxSizeSlices(self, slices: List[int]) -> int:
        def cal(nums: list) -> int:
            @cache
            def dfs(i: int, j: int) -> int:
                if j == m:
                    return 0
                if i >= n:
                    return -inf
                return max(dfs(i + 1, j), dfs(i + 2, j + 1) + nums[i])

            n = len(nums)
            m = (n + 1) // 3
            return dfs(0, 0)

        return max(cal(slices[1:]), cal(slices[: len(slices) - 1]))

    # 2784. 检查数组是否是好的 (Check if Array is Good)
    def isGood(self, nums: List[int]) -> bool:
        nums.sort()
        n = len(nums)
        if nums[-1] != n - 1:
            return False
        for i in range(0, n - 1):
            if i + 1 != nums[i]:
                return False
        return True

    # 2786. 访问数组中的位置使分数最大 (Visit Array Positions to Maximize Score)
    def maxScore(self, nums: List[int], x: int) -> int:
        @cache
        def dfs(i: int) -> int:
            if i >= n:
                return 0
            return max(dfs(i + 1) - x, dfs(i + 2)) + arr[i]

        arr = []
        n = len(nums)
        i = 0
        while i < n:
            j = i
            s = 0
            while j < n and nums[i] % 2 == nums[j] % 2:
                s += nums[j]
                j += 1
            arr.append(s)
            i = j
        n = len(arr)
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
    # LCR 098. 不同路径
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
        @cache
        def dfs(i: int, j: int) -> int:
            if i == m or j == n or obstacleGrid[i][j]:
                return 0
            if i == m - 1 and j == n - 1:
                return 1
            return dfs(i + 1, j) + dfs(i, j + 1)

        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        return dfs(0, 0)

    # 91. 解码方法 (Decode Ways)
    def numDecodings(self, s: str) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == n:
                return 1
            if s[i] == "0":
                return 0
            res = dfs(i + 1)
            if i + 1 < n and (s[i] == "1" or s[i] == "2" and s[i + 1] <= "6"):
                res += dfs(i + 2)
            return res

        n = len(s)
        return dfs(0)

    # 97. 交错字符串 (Interleaving String)
    # LCR 096. 交错字符串
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        @cache
        def dfs(i: int, j: int, k: int) -> bool:
            if i == n1:
                return s2[j:] == s3[k:]
            if j == n2:
                return s1[i:] == s3[k:]
            return (
                s1[i] == s3[k]
                and dfs(i + 1, j, k + 1)
                or s2[j] == s3[k]
                and dfs(i, j + 1, k + 1)
            )

        n1 = len(s1)
        n2 = len(s2)
        n3 = len(s3)
        return n1 + n2 == n3 and dfs(0, 0, 0)

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
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            if j == 0:
                return max(dfs(i + 1, j), dfs(i + 1, 1) - prices[i])
            return max(dfs(i + 1, j), prices[i])

        n = len(prices)
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
            if i == n or j == 2:
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
    # LCR 089. 打家劫舍
    def rob(self, nums: List[int]) -> int:
        n = len(nums)

        @cache
        def dfs(i: int) -> int:
            if i >= n:
                return 0
            return max(dfs(i + 1), dfs(i + 2) + nums[i])

        return dfs(0)

    # 213. 打家劫舍 II (House Robber II)
    # LCR 090. 打家劫舍 II
    def rob(self, nums: List[int]) -> int:
        def cal(arr: list) -> int:
            @cache
            def dfs(i: int) -> int:
                if i >= len(arr):
                    return 0
                return max(dfs(i + 1), dfs(i + 2) + arr[i])

            return dfs(0)

        n = len(nums)
        if n == 1:
            return nums[0]
        return max(cal(nums[1:]), cal(nums[: n - 1]))

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
    # LCR 112. 矩阵中的最长递增路径
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            res = 0
            for dx, dy in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                nx, ny = i + dx, j + dy
                if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] > matrix[i][j]:
                    res = max(res, dfs(nx, ny))
            return res + 1

        m = len(matrix)
        n = len(matrix[0])
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
    # LCR 103. 零钱兑换
    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)

        @cache
        def dfs(i: int, j: int) -> int:
            if j == amount:
                return 0
            if i == n or j > amount:
                return inf
            return min(dfs(i, j + coins[i]) + 1, dfs(i + 1, j))

        res = dfs(0, 0)
        return res if res <= amount else -1

    # 576. 出界的路径数 (Out of Boundary Paths)
    def findPaths(
        self, m: int, n: int, maxMove: int, startRow: int, startColumn: int
    ) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i < 0 or i >= m or j < 0 or j >= n:
                return 1
            if min(i + 1, j + 1, m - i, n - j) > k:
                return 0
            return sum(dfs(i + dx, j + dy, k - 1) for dx, dy in dirs if k) % MOD

        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        MOD = 10**9 + 7
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
        @cache
        def dfs(i: int, j: int) -> int:
            if i >= n:
                return i == n and j == 2
            return (
                dfs(i + 1, 2)
                + (
                    sum(dfs(i + 2, k) for k in range(3))
                    if j == 2
                    else dfs(i + 1, j ^ 1)
                )
            ) % MOD

        MOD = 10**9 + 7
        return dfs(0, 2)

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
        @cache
        def dfs(i: int, j: int) -> int:
            if abs(i - endPos) > j:
                return 0
            if j == 0:
                return i == endPos
            return (dfs(i + 1, j - 1) + dfs(i - 1, j - 1)) % MOD

        MOD = 10**9 + 7
        return dfs(startPos, k)

    # 2585. 获得分数的方法数 (Number of Ways to Earn Points)
    def waysToReachTarget(self, target: int, types: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j == target:
                return 1
            if i == n:
                return 0
            res = 0
            for k in range(types[i][0] + 1):
                s = k * types[i][1]
                if j + s > target:
                    break
                res += dfs(i + 1, j + s)
                res %= MOD
            return res

        n = len(types)
        MOD = 10**9 + 7
        return dfs(0, 0)

    # 2801. 统计范围内的步进数字数目 (Count Stepping Numbers in Range)
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7

        def check(arr: str) -> int:
            n = len(arr)

            @cache
            def dfs(i: int, j: int, is_limit: bool, is_num: bool) -> int:
                if i == n:
                    return int(is_num)
                res = 0
                if not is_num:
                    res = dfs(i + 1, j, False, False)
                up = int(arr[i]) if is_limit else 9
                for d in range(0 if is_num else 1, up + 1):
                    if not is_num or abs(d - j) == 1:
                        res += dfs(i + 1, d, is_limit and d == up, True)
                        res %= MOD
                return res % MOD

            return dfs(0, 0, True, False)

        return (check(high) - check(str(int(low) - 1))) % MOD

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
            up = ord(s[i]) - ord("0") if isLimit else 9
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
            up = ord(s[i]) - ord("0") if isLimit else 9
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
            up = ord(s[i]) - ord("0") if isLimit else 1
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
            up = s[i] if isLimit else "9"
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
            up = ord(s[i]) - ord("0") if isLimit else 9
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
        @cache
        def dfs(i: int, j: int) -> int:
            if j - i <= 1:
                return 0
            return min(
                dfs(i, k) + dfs(k, j) + values[i] * values[j] * values[k]
                for k in range(i + 1, j)
            )

        n = len(values)
        return dfs(0, n - 1)

    # 1048. 最长字符串链 (Longest String Chain)
    def longestStrChain(self, words: List[str]) -> int:
        n = len(words)
        words.sort(key=len)

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
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            return max(dfs(i + 1, j), dfs(i + 1, -1 * j) + j * nums[i])

        n = len(nums)
        return dfs(0, 1)

    # 1463. 摘樱桃 II (Cherry Pickup II)
    def cherryPickup(self, grid: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j0: int, j1: int) -> int:
            if i == m:
                return 0
            res = 0
            for x in range(max(0, j0 - 1), min(n, j0 + 2)):
                for y in range(max(0, j1 - 1), min(n, j1 + 2)):
                    res = max(res, dfs(i + 1, x, y))
            return res + grid[i][j0] + (0 if j0 == j1 else grid[i][j1])

        m = len(grid)
        n = len(grid[0])
        return dfs(0, 0, n - 1)

    # 1473. 粉刷房子 III (Paint House III)
    def minCost(
        self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int
    ) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i == m:
                return 0 if k == target else inf
            if target - k > m - i or k > target:
                return inf
            if houses[i]:
                return dfs(i + 1, houses[i], k + (houses[i] != j))
            return min(
                dfs(i + 1, x + 1, k + ((x + 1) != j)) + cost[i][x] for x in range(n)
            )

        res = dfs(0, 0, 0)
        return res if res < inf else -1

    # 1524. 和为奇数的子数组数目 (Number of Sub-arrays With Odd Sum)
    def numOfSubarrays(self, arr: List[int]) -> int:
        even = 1
        odd = 0
        res = 0
        sum = 0
        m = 10**9 + 7
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
            m |= 1 << (ord(s[i]) - ord("a"))
            left[i] = m
        m = 0
        for i in range(n - 1, -1, -1):
            m |= 1 << (ord(s[i]) - ord("a"))
            right[i] = m
        res = 0
        for i in range(1, n):
            res += left[i - 1].bit_count() == right[i].bit_count()
        return res

    # 2466. 统计构造好字符串的方案数 (Count Ways To Build Good Strings)
    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        MOD = 10**9 + 7

        @cache
        def dfs(i: int) -> int:
            if i < 0:
                return 0
            if i == 0:
                return 1
            return (dfs(i - zero) + dfs(i - one)) % MOD

        return sum(dfs(x) for x in range(low, high + 1)) % MOD

    # 53. 最大子数组和 (Maximum Subarray) --前缀和
    # 剑指 Offer 42. 连续子数组的最大和
    def maxSubArray(self, nums: List[int]) -> int:
        res = -inf
        pre = 0
        m = 0
        for num in nums:
            pre += num
            res = max(res, pre - m)
            m = min(m, pre)
        return res

    # 53. 最大子数组和 (Maximum Subarray) --dp
    # 剑指 Offer 42. 连续子数组的最大和
    def maxSubArray(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int) -> int:
            if i < 0:
                return 0
            return max(dfs(i - 1), 0) + nums[i]

        n = len(nums)
        return max(dfs(i) for i in range(n))

    # 72. 编辑距离 (Edit Distance)
    def minDistance(self, word1: str, word2: str) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n1:
                return n2 - j
            if j == n2:
                return n1 - i
            if word1[i] == word2[j]:
                return dfs(i + 1, j + 1)
            return min(dfs(i, j + 1), dfs(i + 1, j + 1), dfs(i + 1, j)) + 1

        n1 = len(word1)
        n2 = len(word2)
        return dfs(0, 0)

    # 87. 扰乱字符串 (Scramble String)
    def isScramble(self, s1: str, s2: str) -> bool:
        @cache
        def dfs(i: int, j: int, l: int) -> bool:
            if l == 1:
                return s1[i] == s2[j]
            if s1[i : i + l] == s2[j : j + l]:
                return True
            if Counter(s1[i : i + l]) != Counter(s2[j : j + l]):
                return False
            return any(
                dfs(i, j, k)
                and dfs(i + k, j + k, l - k)
                or dfs(i, j + l - k, k)
                and dfs(i + k, j, l - k)
                for k in range(1, l)
            )

        n = len(s1)
        return dfs(0, 0, n)

    # 5. 最长回文子串 (Longest Palindromic Substring)
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False] * (n) for _ in range(0, n)]
        l = -1
        r = -1
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if (
                    i == j
                    or j - i == 1
                    and s[i] == s[j]
                    or j - i > 1
                    and dp[i + 1][j - 1]
                    and s[i] == s[j]
                ):
                    dp[i][j] = True
                    if j - i >= r - l:
                        r = j
                        l = i
        return s[l : r + 1]

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
            dp[ord(s[i]) - ord("a")] = max(dp[ord(s[i]) - ord("a")], cnt)
        return sum(dp)

    # 879. 盈利计划 (Profitable Schemes)
    def profitableSchemes(
        self, n: int, minProfit: int, group: List[int], profit: List[int]
    ) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if j > n:
                return 0
            if i == len(group):
                return int(k == minProfit)
            return (
                dfs(i + 1, j, k)
                + dfs(i + 1, j + group[i], min(k + profit[i], minProfit))
            ) % MOD

        MOD = 10**9 + 7
        return dfs(0, 0, 0)

    # 688. 骑士在棋盘上的概率 (Knight Probability in Chessboard)
    def knightProbability(self, n: int, k: int, row: int, column: int) -> float:
        @cache
        def dfs(i: int, j: int, k: int) -> float:
            if not (i < n and i >= 0 and j < n and j >= 0):
                return 0
            if k == 0:
                return 1.0
            return sum(dfs(i + x, j + y, k - 1) for x, y in dirs) / 8

        dirs = (-1, 2), (2, -1), (1, -2), (2, 1), (-1, -2), (-2, -1), (1, 2), (-2, 1)
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

    # LCR 166. 珠宝的最高价值
    def jewelleryValue(self, frame: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == m or j == n:
                return 0
            return max(dfs(i + 1, j), dfs(i, j + 1)) + frame[i][j]

        m = len(frame)
        n = len(frame[0])
        return dfs(0, 0)

    # 面试题 08.14. 布尔运算
    def countEval(self, s: str, result: int) -> int:
        n = len(s)

        @cache
        def dfs(i: int, j: int, ret: int) -> int:
            if i == j:
                return 1 if ord(s[i]) - ord("0") == ret else 0
            res = 0
            for k in range(i + 1, j, 2):
                op = ord(s[k])
                for x in range(2):
                    for y in range(2):
                        if check(x, y, op, ret):
                            res += dfs(i, k - 1, x) * dfs(k + 1, j, y)
            return res

        def check(a: int, b: int, op: int, res: int) -> bool:
            if op == ord("|"):
                return (a | b) == res
            if op == ord("&"):
                return (a & b) == res
            return (a ^ b) == res

        return dfs(0, n - 1, result)

    # 1444. 切披萨的方案数 (Number of Ways of Cutting a Pizza)
    def ways(self, pizza: List[str], k: int) -> int:
        def check(x0: int, y0: int, x1: int, y1: int) -> int:
            return (
                pre[x1 + 1][y1 + 1] - pre[x0][y1 + 1] - pre[x1 + 1][y0] + pre[x0][y0]
                > 0
            )

        @cache
        def dfs(i: int, j: int, l: int) -> int:
            if l == 1:
                return int(check(i, j, m - 1, n - 1))
            res = 0
            for x in range(i + 1, m):
                if not check(i, j, x - 1, n - 1):
                    continue
                res += dfs(x, j, l - 1)
            for y in range(j + 1, n):
                if not check(i, j, m - 1, y - 1):
                    continue
                res += dfs(i, y, l - 1)
            return res % MOD

        m = len(pizza)
        n = len(pizza[0])
        pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                pre[i + 1][j + 1] = (
                    pre[i][j + 1] + pre[i + 1][j] - pre[i][j] + (pizza[i][j] == "A")
                )
        MOD = 10**9 + 7
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
        dirs = [[0, -1], [0, 1], [1, 0], [-1, 0]]
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
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 1
            return sum(dfs(i + 1, x) for x in d[j]) % MOD

        MOD = 10**9 + 7
        d = [[] for _ in range(10)]
        d[0].extend([4, 6])
        d[1].extend([6, 8])
        d[2].extend([7, 9])
        d[3].extend([4, 8])
        d[4].extend([0, 3, 9])
        d[6].extend([0, 1, 7])
        d[7].extend([2, 6])
        d[8].extend([1, 3])
        d[9].extend([2, 4])
        return sum(dfs(1, i) for i in range(10)) % MOD

    # 931. 下降路径最小和 (Minimum Falling Path Sum)
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            return (
                min(dfs(i + 1, k) for k in range(max(0, j - 1), min(n, j + 2)))
                + matrix[i][j]
            )

        n = len(matrix)
        return min(dfs(0, j) for j in range(n))

    # 983. 最低票价 (Minimum Cost For Tickets)
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        arr = [False] * 366
        for d in days:
            arr[d] = True

        @cache
        def dfs(i: int) -> int:
            if i > 365:
                return 0
            return min(
                dfs(i + 1) + (costs[0] if arr[i] else 0),
                dfs(i + 7) + costs[1],
                dfs(i + 30) + costs[2],
            )

        return dfs(1)

    # 1155. 掷骰子等于目标和的方法数 (Number of Dice Rolls With Target Sum)
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j > target:
                return 0
            if i == n:
                return j == target
            if n - i > target - j:
                return 0
            if target - j > (n - i) * k:
                return 0
            return sum(dfs(i + 1, j + x) for x in range(1, k + 1)) % MOD

        MOD = 10**9 + 7
        return dfs(0, 0)

    # 1289. 下降路径最小和 II (Minimum Falling Path Sum II)
    def minFallingPathSum(self, grid: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n - 1:
                return grid[i][j]
            return min(dfs(i + 1, k) if k != j else inf for k in range(n)) + grid[i][j]

        n = len(grid)
        return min(dfs(0, j) for j in range(n))

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
        @cache
        def dfs(i: int, j: int) -> float:
            if i == n:
                return 0
            if j == k:
                return -inf
            s = 0
            res = 0
            for x in range(i, n - k + j + 1):
                s += nums[x]
                res = max(res, dfs(x + 1, j + 1) + s / (x - i + 1))
            return res

        n = len(nums)
        return dfs(0, 0)

    # 1035. 不相交的线 (Uncrossed Lines)
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        n1 = len(nums1)
        n2 = len(nums2)

        @cache
        def dfs(i: int, j: int) -> int:
            if i == n1 or j == n2:
                return 0
            return (
                dfs(i + 1, j + 1) + 1
                if nums1[i] == nums2[j]
                else max(dfs(i + 1, j), dfs(i, j + 1))
            )

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
        @cache
        def dfs(i: int, j: int) -> int:
            if j >= n - i:
                return 0
            if i == n:
                return 0 if j >= 0 else inf
            if j + suf[i] < 0:
                return inf
            return min(dfs(i + 1, j + time[i]) + cost[i], dfs(i + 1, j - 1))

        n = len(cost)
        suf = [0] * n
        suf[-1] = time[-1]
        for i in range(n - 2, -1, -1):
            suf[i] = suf[i + 1] + time[i]
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
        @cache
        def dfs(i: int, j: int) -> int:
            if i < 0:
                return 0
            res = dfs(i - 1, 0)
            sub = c = ((j << 1 | j >> 1 | forbid[i]) & u) ^ u
            while sub:
                if sub & (sub >> 1) == 0:
                    res = max(res, dfs(i - 1, sub) + sub.bit_count())
                sub = (sub - 1) & c
            return res

        m = len(seats)
        n = len(seats[0])
        forbid = [0] * m
        for i in range(m):
            for j in range(n):
                forbid[i] |= (seats[i][j] == "#") << j
        u = (1 << n) - 1
        return dfs(m - 1, 0)

    # 2305. 公平分发饼干 (Fair Distribution of Cookies)
    def distributeCookies(self, cookies: List[int], k: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j == u:
                return 0
            if i == k:
                return inf
            res = inf
            sub = c = u ^ j
            while sub:
                res = min(res, max(dfs(i + 1, j | sub), s[sub]))
                sub = (sub - 1) & c
            return res

        n = len(cookies)
        s = [0] * (1 << n)
        for i in range(1, 1 << n):
            s[i] = s[i & (i - 1)] + cookies[(i & -i).bit_length() - 1]
        u = (1 << n) - 1
        return dfs(0, 0)

    # 2741. 特别的排列 (Special Permutations)
    def specialPerm(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == u:
                return 1
            c = u ^ i
            res = 0
            while c:
                lb = (c & -c).bit_length() - 1
                if nums[lb] % nums[j] == 0 or nums[j] % nums[lb] == 0:
                    res += dfs(i | (1 << lb), lb)
                c &= c - 1
            return res % MOD

        n = len(nums)
        u = (1 << n) - 1
        MOD = 10**9 + 7
        return sum(dfs(1 << i, i) for i in range(n)) % MOD

    # 2002. 两个回文子序列长度的最大乘积 (Maximum Product of the Length of Two Palindromic Subsequences)
    def maxProduct(self, s: str) -> int:
        n = len(s)
        arr = [False] * (1 << n)
        for i in range(1, 1 << n):
            cnt = i.bit_count()
            lead = i.bit_length() - 1
            trail = (i & -i).bit_length() - 1
            if s[lead] == s[trail] and (
                cnt <= 2 or arr[i ^ (1 << lead) ^ (1 << trail)]
            ):
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
        @cache
        def dfs(i: int) -> int:
            if i == n or n - i < k:
                return 0
            res = dfs(i + 1)
            for j in range(i + k - 1, n):
                if valid[i][j]:
                    res = max(res, dfs(j + 1) + 1)
                    break
            return res

        n = len(s)
        valid = [[False] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if (
                    i == j
                    or j - i == 1
                    and s[i] == s[j]
                    or j - i > 1
                    and s[i] == s[j]
                    and valid[i + 1][j - 1]
                ):
                    valid[i][j] = True
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
        if len(set(s)) == 1:
            return n
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

    # 2430. 对字母串可执行的最大删除数 (Maximum Deletions on a String)
    def deleteString(self, s: str) -> int:
        def dfs(i: int) -> int:
            if i == n:
                return 0
            if memo[i] != -1:
                return memo[i]
            res = 0
            for j in range(i + 1, n):
                if n + i < j * 2:
                    break
                if lcp[i][j] >= j - i:
                    res = max(res, dfs(j))
            memo[i] = res + 1
            return memo[i]

        n = len(s)
        if len(set(s)) == 1:
            return n
        lcp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, i, -1):
                if s[i] == s[j]:
                    lcp[i][j] = lcp[i + 1][j + 1] + 1
        memo = [-1] * n
        return dfs(0)

    # 2435. 矩阵中和能被 K 整除的路径 (Paths in Matrix Whose Sum Is Divisible by K)
    def numberOfPaths(self, grid: List[List[int]], k: int) -> int:
        @cache
        def dfs(i: int, j: int, mod: int) -> int:
            if i == m - 1 and j == n - 1:
                return int(not (grid[m - 1][n - 1] + mod) % k)
            if i == m or j == n:
                return 0
            return (
                dfs(i + 1, j, (mod + grid[i][j]) % k)
                + dfs(i, j + 1, (mod + grid[i][j]) % k)
            ) % MOD

        m = len(grid)
        n = len(grid[0])
        MOD = 10**9 + 7
        res = dfs(0, 0, 0)
        dfs.cache_clear()
        return res

    # 2318. 不同骰子序列的数目 (Number of Distinct Roll Sequences)
    def distinctSequences(self, n: int) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i == n:
                return 1
            return (
                sum(
                    dfs(i + 1, x, j)
                    for x in range(1, 7)
                    if gcd(x, j) == 1 and x != j and x != k
                )
                % MOD
            )

        MOD = 10**9 + 7
        return dfs(0, -1, -1)

    # 2304. 网格中的最小路径代价 (Minimum Path Cost in a Grid)
    def minPathCost(self, grid: List[List[int]], moveCost: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == m - 1:
                return grid[i][j]
            return (
                min(dfs(i + 1, k) + moveCost[grid[i][j]][k] for k in range(n))
                + grid[i][j]
            )

        m = len(grid)
        n = len(grid[0])
        return min(dfs(0, j) for j in range(n))

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
        MOD = 10**9 + 7

        @cache
        def dfs(i: int) -> int:
            if i >= n:
                return 1
            return (dfs(i + 1) + dfs(i + 2)) % MOD

        res = dfs(0)
        return pow(res, 2, MOD)

    # 2550. 猴子碰撞的方法数 (Count Collisions of Monkeys on a Polygon)
    def monkeyMove(self, n: int) -> int:
        MOD = 10**9 + 7
        return (pow(2, n, MOD) - 2) % MOD

    # 617. 合并二叉树 (Merge Two Binary Trees)
    def mergeTrees(
        self, root1: Optional[TreeNode], root2: Optional[TreeNode]
    ) -> Optional[TreeNode]:
        if not root1:
            return root2
        if not root2:
            return root1
        root1.val += root2.val
        root1.left = self.mergeTrees(root1.left, root2.left)
        root1.right = self.mergeTrees(root1.right, root2.right)
        return root1

    # 50. Pow(x, n) (Pow(x, n))
    # LCR 134. Pow(x, n)
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
                nums[i + 1 :] = sorted(nums[i + 1 :])
                break
            j -= 1

    # 1771. 由子序列构造的最长回文串的长度 (Maximize Palindrome Length From Subsequences)
    def longestPalindrome(self, word1: str, word2: str) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == j:
                return 1
            if i > j:
                return 0
            if s[i] == s[j]:
                return dfs(i + 1, j - 1) + 2
            return max(dfs(i + 1, j), dfs(i, j - 1))

        s = word1 + word2
        res = 0
        for i in range(len(word1)):
            for j in range(len(word2) - 1, -1, -1):
                if word1[i] == word2[j]:
                    res = max(res, dfs(i + 1, len(word1) + j - 1) + 2)
        return res

    # 2147. 分隔长廊的方案数 (Number of Ways to Divide a Long Corridor)
    def numberOfWays(self, corridor: str) -> int:
        n = len(corridor)
        res = 1
        MOD = 10**9 + 7
        j = -1
        cnt = 0

        for i, c in enumerate(corridor):
            if c == "S":
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
            if hamsters[i] == "H":
                if i + 1 < n and hamsters[i + 1] == ".":
                    res += 1
                    i += 3
                elif i - 1 >= 0 and hamsters[i - 1] == ".":
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
            m |= 1 << (ord(c) - ord("a"))
        n = len(word)
        for i, c in enumerate(word):
            if (m >> (ord(c) - ord("a"))) & 1:
                res += (i + 1) * (n - i)
        return res

    # 1986. 完成任务的最少工作时间段 (Minimum Number of Work Sessions to Finish the Tasks)
    def minSessions(self, tasks: List[int], sessionTime: int) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == u:
                return 0
            res = inf
            sub = c = u ^ i
            while sub:
                if s[sub] <= sessionTime:
                    res = min(res, dfs(i | sub) + 1)
                sub = (sub - 1) & c
            return res

        n = len(tasks)
        s = [0] * (1 << n)
        for i in range(1, 1 << n):
            s[i] = s[i & (i - 1)] + tasks[(i & -i).bit_length() - 1]
        u = (1 << n) - 1
        return dfs(0)

    # 1799. N 次操作后的最大分数和 (Maximize Score After N Operations)
    def maxScore(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j == u:
                return 0
            res = 0
            sub = c = u ^ j
            while sub:
                if sub.bit_count() == 2:
                    lb = (sub & -sub).bit_length() - 1
                    lb2 = (sub & (sub - 1)).bit_length() - 1
                    res = max(
                        res,
                        dfs(i + 1, j | (1 << lb) | (1 << lb2))
                        + (i + 1) * gcd(nums[lb], nums[lb2]),
                    )
                sub = (sub - 1) & c
            return res

        n = len(nums)
        u = (1 << n) - 1
        return dfs(0, 0)

    # 1931. 用三种不同颜色为网格涂色 (Painting a Grid With Three Different Colors)
    def colorTheGrid(self, m: int, n: int) -> int:
        MOD = 10**9 + 7

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
            if s[i] == "1" and i + 1 < l:
                res += dfs(i + 2)
            elif s[i] == "2" and i + 1 < l and "0" <= s[i + 1] <= "5":
                res += dfs(i + 2)
            return res

        return dfs(0)

    # LCP 19. 秋叶收藏集
    def minimumOperations(self, leaves: str) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0 if j == 2 else inf
            if j == 0:
                if leaves[i] == "r":
                    return min(dfs(i + 1, j), dfs(i + 1, j + 1) + 1)
                return min(dfs(i + 1, j + 1), dfs(i + 1, j) + 1)
            if j == 1:
                if leaves[i] == "r":
                    return min(dfs(i + 1, j + 1), dfs(i + 1, j) + 1)
                return min(dfs(i + 1, j), dfs(i + 1, j + 1) + 1)
            return dfs(i + 1, j) + (leaves[i] == "y")

        n = len(leaves)
        return dfs(1, 0) + (leaves[0] == "y")

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
        right = 10**9
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
        MOD = 10**9 + 7
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
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            cur = ord(word[i]) - ord("A")
            pre = ord(word[i - 1]) - ord("A")
            return min(dfs(i + 1, j) + dis(pre, cur), dfs(i + 1, pre) + dis(j, cur))

        return min(dfs(1, j) for j in range(26))

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

    # 1458. 两个子序列的最大点积 (Max Dot Product of Two Subsequences)
    def maxDotProduct(self, nums1: List[int], nums2: List[int]) -> int:
        n1 = len(nums1)
        n2 = len(nums2)

        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i == n1 or j == n2:
                return 0 if k else -inf
            return max(
                dfs(i + 1, j, k),
                dfs(i, j + 1, k),
                dfs(i + 1, j + 1, 1) + nums1[i] * nums2[j],
            )

        return dfs(0, 0, 0)

    # 1449. 数位成本和为目标值的最大数字 (Form Largest Integer With Digits That Add up to Target)
    def largestNumber(self, cost: List[int], target: int) -> str:
        @cache
        def dfs(i: int) -> int:
            if i == target:
                return 0
            if i > target:
                return -inf
            res = -inf
            for c in cost:
                res = max(res, dfs(c + i))
            return res + 1 if res > -inf else -inf

        def make_ans(i: int) -> None:
            if i == target:
                return
            final_ans = dfs(i)
            for j in range(8, -1, -1):
                if dfs(cost[j] + i) + 1 == final_ans:
                    res.append(str(j + 1))
                    make_ans(cost[j] + i)
                    break

        res = dfs(0)
        if res == -inf:
            return "0"
        res = []
        make_ans(0)
        return "".join(res)

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
        res = ["."] * n
        i = 0
        j = 0
        while j < n:
            if dominoes[j] == "L":
                if dominoes[i] == "R":
                    for k in range(i + 1, i + (j - i - 1) // 2 + 1):
                        res[k] = "R"
                    for k in range(j - (j - i - 1) // 2, j):
                        res[k] = "L"
                else:
                    for k in range(i, j):
                        res[k] = "L"
                res[j] = "L"
                i = j
            elif dominoes[j] == "R":
                if dominoes[i] == "R":
                    for k in range(i, j):
                        res[k] = "R"
                res[j] = "R"
                i = j
            j += 1

        j = n - 1
        while j >= 0 and dominoes[j] == ".":
            j -= 1
        if j >= 0 and dominoes[j] == "R":
            for k in range(j, n):
                res[k] = "R"

        return "".join(res)

    # 828. 统计子串中的唯一字符 (Count Unique Characters of All Substrings of a Given String)
    def uniqueLetterString(self, s: str) -> int:
        n = len(s)
        last = [-1] * 26
        pre = [-1] * n
        for i, c in enumerate(s):
            j = ord(c) - ord("A")
            pre[i] = last[j]
            last[j] = i
        last = [n] * 26
        suf = [n] * n
        for i in range(n - 1, -1, -1):
            j = ord(s[i]) - ord("A")
            suf[i] = last[j]
            last[j] = i
        res = 0
        for i in range(n):
            res += (i - pre[i]) * (suf[i] - i)
        return res

    # 1879. 两个数组最小的异或值之和 (Minimum XOR Sum of Two Arrays)
    def minimumXORSum(self, nums1: List[int], nums2: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j == u:
                return 0
            c = j ^ u
            res = inf
            while c:
                lb = (c & -c).bit_length() - 1
                res = min(res, dfs(i + 1, j | (1 << lb)) + (nums1[i] ^ nums2[lb]))
                c &= c - 1
            return res

        n = len(nums1)
        u = (1 << n) - 1
        return dfs(0, 0)

    # 1655. 分配重复整数 (Distribute Repeating Integers)
    def canDistribute(self, nums: List[int], quantity: List[int]) -> bool:
        @cache
        def dfs(i: int, j: int) -> bool:
            if j == u:
                return True
            if i == n:
                return False
            sub = c = j ^ u
            if dfs(i + 1, j):
                return True
            while sub:
                if arr[i] >= s[sub] and dfs(i + 1, j | sub):
                    return True
                sub = (sub - 1) & c
            return False

        arr = list(Counter(nums).values())
        if max(arr) < max(quantity):
            return False
        n = len(arr)
        m = len(quantity)
        s = [0] * (1 << m)
        u = (1 << m) - 1
        for i in range(1, 1 << m):
            s[i] = s[i & (i - 1)] + quantity[(i & -i).bit_length() - 1]
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

    # 2827. 范围中美丽整数的数目 (Number of Beautiful Integers in the Range)
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
                    res += dfs(
                        i + 1,
                        diff + (1 if j % 2 == 0 else -1),
                        (m * 10 + j) % k,
                        isLimit and j == up,
                        True,
                    )
                return res

            return dfs(0, 0, 0, True, False)

        return cal(str(high)) - cal(str(low - 1))

    # 2826. 将三个组排序 (Sorting Three Groups)
    def minimumOperations(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            res = dfs(i + 1, j)
            if nums[i] >= j:
                res = max(res, dfs(i + 1, nums[i]) + 1)
            return res

        n = len(nums)
        return n - dfs(0, 1)

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
        class Trie:

            def __init__(self) -> None:
                self.children = [None] * 26
                self.is_end = False

            def insert(self, s: str) -> None:
                node = self
                for c in s:
                    index = ord(c) - ord("a")
                    if node.children[index] is None:
                        node.children[index] = Trie()
                    node = node.children[index]
                node.is_end = True

            def check(self, s: str) -> list:
                ret = []
                node = self
                for i, c in enumerate(s):
                    index = ord(c) - ord("a")
                    if node.children[index] is None:
                        break
                    node = node.children[index]
                    if node.is_end:
                        ret.append(i)
                return ret

        trie = Trie()
        for w in wordDict:
            trie.insert(w)

        @cache
        def dfs(i: int) -> bool:
            if i == len(s):
                return True
            return any(dfs(i + j + 1) for j in trie.check(s[i:]))

        return dfs(0)

    # 907. 子数组的最小值之和 (Sum of Subarray Minimums)
    def sumSubarrayMins(self, arr: List[int]) -> int:
        MOD = 10**9 + 7
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
        MOD = 10**9 + 7
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
            res += nums[i] * (
                (rightMax[i] - i) * (i - leftMax[i])
                - (rightMin[i] - i) * (i - leftMin[i])
            )
        return res

    # 2818. 操作使得分最大 (Apply Operations to Maximize Score)
    def maximumScore(self, nums: List[int], k: int) -> int:
        n = len(nums)
        MAX = 10**5 + 1
        MOD = 10**9 + 7
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
        for i, v, l, r in sorted(zip(range(n), nums, left, right), key=lambda z: -z[1]):
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
        g = [[] for _ in range(numCourses)]
        deg = [0] * numCourses
        q = []
        for p in prerequisites:
            g[p[1]].append(p[0])
            deg[p[0]] += 1
        for i in range(numCourses):
            if not deg[i]:
                q.append(i)
        while q:
            numCourses -= 1
            x = q.pop(0)
            for y in g[x]:
                deg[y] -= 1
                if not deg[y]:
                    q.append(y)
        return not numCourses

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
        @cache
        def dfs(i: int) -> int:
            if i == u:
                return 0
            sub = i ^ u
            c = 0
            while sub:
                lb = (sub & -sub).bit_length() - 1
                if pre[lb] | i == i:
                    c |= 1 << lb
                sub &= sub - 1
            if c.bit_count() <= k:
                return dfs(i | c) + 1
            res = inf
            sub = c
            while sub:
                if sub.bit_count() <= k:
                    res = min(res, dfs(i | sub) + 1)
                sub = (sub - 1) & c
            return res

        pre = [0] * n
        for x, y in relations:
            pre[y - 1] |= 1 << (x - 1)
        u = (1 << n) - 1
        return dfs(0)

    # 1416. 恢复数组 (Restore The Array)
    def numberOfArrays(self, s: str, k: int) -> int:
        n = len(s)
        MOD = 10**9 + 7

        @cache
        def dfs(i: int) -> int:
            if i == n:
                return 1
            if s[i] == "0":
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
        @cache
        def dfs(i: int, j: int, l: int) -> int:
            if l > k or n - i < k - l or m - j < k - l:
                return 0
            if i == n:
                return l == k
            return (
                j * dfs(i + 1, j, l)
                + sum(dfs(i + 1, x, l + 1) for x in range(j + 1, m + 1))
            ) % MOD

        MOD = 10**9 + 7
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
                if (
                    0 <= nx < m
                    and 0 <= ny < n
                    and not arr0[nx][ny]
                    and heights[nx][ny] >= heights[x][y]
                ):
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
                if (
                    0 <= nx < m
                    and 0 <= ny < n
                    and not arr1[nx][ny]
                    and heights[nx][ny] >= heights[x][y]
                ):
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
        MOD = 10**9 + 7
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
        @cache
        def dfs(i: int, j: int) -> int:
            if j - i == 1:
                return 0
            return (
                min(dfs(i, k) + dfs(k, j) for k in range(i + 1, j)) + cuts[j] - cuts[i]
            )

        cuts.extend([0, n])
        cuts.sort()
        return dfs(0, len(cuts) - 1)

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
                cnts[ord(c) - ord("a")] += 1
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
    def insert(
        self, intervals: List[List[int]], newInterval: List[int]
    ) -> List[List[int]]:
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
        return (
            n > 0
            and n.bit_count() == 1
            and (n & 0b10101010101010101010101010101010) == 0
        )

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
                if board[i][j] != ".":
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
        MOD = 10**9 + 7

        @cache
        def dfs(i: int) -> int:
            res = 1
            for j in arr:
                if j * arr[0] > i:
                    break
                if i % j == 0 and (i / j in s):
                    res += dfs(j) * dfs(i / j)
                    res %= MOD
            return res

        return sum(dfs(i) for i in arr) % MOD

    # 1335. 工作计划的最低难度 (Minimum Difficulty of a Job Schedule)
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0 if j == d else inf
            if j == d:
                return inf
            res = inf
            mx = 0
            for k in range(i, n - d + j + 1):
                mx = max(mx, jobDifficulty[k])
                res = min(res, dfs(k + 1, j + 1) + mx)
            return res

        n = len(jobDifficulty)
        if d > n:
            return -1
        return dfs(0, 0)

    # 1312. 让字符串成为回文串的最少插入次数 (Minimum Insertion Steps to Make a String Palindrome)
    def minInsertions(self, s: str) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i >= j:
                return 0
            if s[i] == s[j]:
                return dfs(i + 1, j - 1)
            return min(dfs(i, j - 1), dfs(i + 1, j)) + 1

        n = len(s)
        return dfs(0, n - 1)

    # 1301. 最大得分的路径数目 (Number of Paths with Max Score)
    def pathsWithMaxScore(self, board: List[str]) -> List[int]:
        @cache
        def dfs(i: int, j: int) -> list:
            if i == n - 1 and j == n - 1:
                return [0, 1]
            if i == n or j == n or board[i][j] == "X":
                return [-inf, 0]
            res = [-inf, 0]
            for dx, dy in [[0, 1], [1, 0], [1, 1]]:
                [cur0, cur1] = dfs(i + dx, j + dy)
                if cur0 > res[0]:
                    res = [cur0, cur1]
                elif cur0 == res[0]:
                    res[1] += cur1
                    res[1] %= MOD
            res[0] += int(board[i][j]) if board[i][j] != "E" else 0
            return res

        MOD = 10**9 + 7
        n = len(board)
        res = dfs(0, 0)
        return [0, 0] if res[0] < 0 else res

    # 1278. 分割回文串 III (Palindrome Partitioning III)
    def palindromePartition(self, s: str, k: int) -> int:
        def cal(i: int, j: int) -> int:
            res = 0
            while i < j:
                res += s[i] != s[j]
                i += 1
                j -= 1
            return res

        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0 if j == k else inf
            if j == k:
                return inf
            res = inf
            for x in range(i, n - k + j + 1):
                res = min(res, dfs(x + 1, j + 1) + p[i][x])
            return res

        n = len(s)
        p = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(1, n):
                p[i][j] = cal(i, j)
        return dfs(0, 0)

    # 1223. 掷骰子模拟 (Dice Roll Simulation)
    def dieSimulator(self, n: int, rollMax: List[int]) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i == n:
                return 1
            res = 0
            for x in range(1, 7):
                if j != x:
                    res += dfs(i + 1, x, 1)
                elif k < rollMax[x - 1]:
                    res += dfs(i + 1, x, k + 1)
            return res % MOD

        MOD = 10**9 + 7
        return dfs(0, 0, 0)

    # 1220. 统计元音字母序列的数目 (Count Vowels Permutation)
    def countVowelPermutation(self, n: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 1
            return sum(dfs(i + 1, k) for k in dic[j]) % MOD

        dic = defaultdict(list)
        dic[0].extend([1])
        dic[1].extend([0, 2])
        dic[2].extend([0, 1, 3, 4])
        dic[3].extend([2, 4])
        dic[4].extend([0])
        dic[5].extend([0, 1, 2, 3, 4])
        MOD = 10**9 + 7
        return dfs(0, 5)

    # 2833. 距离原点最远的点 (Furthest Point From Origin)
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        return abs(moves.count("L") - moves.count("R")) + moves.count("_")

    # 2787. 将一个数字表示成幂的和的方案数 (Ways to Express an Integer as Sum of Powers)
    def numberOfWays(self, n: int, x: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j == n:
                return 1
            if j > n or j + pow(i, x) > n:
                return 0
            return (dfs(i + 1, j) + dfs(i + 1, j + pow(i, x))) % MOD

        MOD = 10**9 + 7
        return dfs(1, 0)

    # 2646. 最小化旅行的价格总和 (Minimize the Total Price of the Trips)
    def minimumTotalPrice(
        self, n: int, edges: List[List[int]], price: List[int], trips: List[List[int]]
    ) -> int:
        def min_dfs(x: int, fa: int) -> List[int]:
            a = 0
            b = 0
            for y in g[x]:
                if y != fa:
                    # 原价 // 打折
                    cur = min_dfs(y, x)
                    a += min(cur[0], cur[1])
                    b += cur[0]
            return [a + cnt[x] * price[x], b + cnt[x] * price[x] // 2]

        def dfs(x: int, fa: int) -> bool:
            if x == e:
                cnt[x] += 1
                return True
            for y in g[x]:
                if y != fa and dfs(y, x):
                    cnt[x] += 1
                    return True
            return False

        cnt = [0] * n
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        for s, e in trips:
            dfs(s, -1)
        return min(min_dfs(0, -1))

    # 2581. 统计可能的树根数目 (Count Number of Possible Root Nodes)
    def rootCount(
        self, edges: List[List[int]], guesses: List[List[int]], k: int
    ) -> int:
        def reroot(x: int, fa: int, c: int) -> None:
            for y in g[x]:
                d = 0
                if y != fa:
                    if (x, y) in s:
                        d -= 1
                    if (y, x) in s:
                        d += 1
                    nonlocal res
                    res += int(c + d >= k)
                    reroot(y, x, c + d)

        def dfs0(x: int, fa: int) -> None:
            for y in g[x]:
                if y != fa:
                    if (x, y) in s:
                        nonlocal cnt
                        cnt += 1
                    dfs0(y, x)

        n = len(edges) + 1
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        s = set((x, y) for x, y in guesses)
        cnt = 0
        dfs0(0, -1)
        res = int(cnt >= k)
        reroot(0, -1, cnt)
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

                if node + a < len(vis) and not vis[node + a][0] and (node + a not in s):
                    vis[node + a][0] = True
                    q.append([node + a, 0])

                if (
                    leftStep == 0
                    and node - b >= 0
                    and not vis[node - b][1]
                    and (node - b not in s)
                ):
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
        left1 = [0] * n
        for i in range(1, n):
            left1[i] = left1[i - 1] + int(s[i - 1])
        res = 0
        cnt1 = 0
        for i in range(n - 1, -1, -1):
            if s[i] == "1":
                res += (i - left1[i]) * (n - i - 1 - cnt1)
            else:
                res += left1[i] * cnt1
            cnt1 += int(s[i])
        return res

    # 2746. 字符串连接删减字母 (Decremental String Concatenation)
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)

        @cache
        def dfs(i: int, l: int, r: int) -> int:
            if i == n:
                return 0
            start = ord(words[i][0]) - ord("a")
            end = ord(words[i][-1]) - ord("a")
            return min(
                dfs(i + 1, l, end) + len(words[i]) - (start == r),
                dfs(i + 1, start, r) + len(words[i]) - (l == end),
            )

        return dfs(1, ord(words[0][0]) - ord("a"), ord(words[0][-1]) - ord("a")) + len(
            words[0]
        )

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
    def countServers(
        self, n: int, logs: List[List[int]], x: int, queries: List[int]
    ) -> List[int]:
        logs.sort(key=lambda k: k[1])
        res = [0] * len(queries)
        i = 0
        j = 0
        cur = 0
        cnts = [0] * (n + 1)
        for t, id in sorted(zip(queries, range(len(queries))), key=lambda k: k[0]):
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
            if j == 26 or abs(ord(s[i]) - ord("a") - j) <= k:
                res = max(res, dfs(i + 1, ord(s[i]) - ord("a")) + 1)
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
        @cache
        def dfs(i: int, j: int) -> int:
            if j - i + 1 == n - m:
                return 0
            return max(
                dfs(i + 1, j) + nums[i] * multipliers[n - (j - i + 1)],
                dfs(i, j - 1) + nums[j] * multipliers[n - (j - i + 1)],
            )

        n = len(nums)
        m = len(multipliers)
        return dfs(0, n - 1)

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
    def jobScheduling(
        self, startTime: List[int], endTime: List[int], profit: List[int]
    ) -> int:
        n = len(startTime)
        jobs = sorted(zip(startTime, endTime, profit))

        @cache
        def dfs(i: int) -> int:
            if i == n:
                return 0
            p = bisect.bisect_left(jobs, jobs[i][1], key=lambda k: k[0])
            return max(dfs(i + 1), dfs(p) + jobs[i][2])

        return dfs(0)

    # 2008. 出租车的最大盈利 (Maximum Earnings From Taxi)
    def maxTaxiEarnings(self, n: int, rides: List[List[int]]) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == n:
                return 0
            res = dfs(i + 1)
            for j, v in dic[i]:
                res = max(res, dfs(j) + j - i + v)
            return res

        dic = [[] for _ in range(n + 1)]
        for s, e, v in rides:
            dic[s].append((e, v))
        return dfs(1)

    # 2830. 销售利润最大化 (Maximize the Profit as the Salesman)
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == n:
                return 0
            res = dfs(i + 1)
            for j, v in dic[i]:
                res = max(res, dfs(j + 1) + v)
            return res

        dic = [[] for _ in range(n)]
        for s, e, v in offers:
            dic[s].append((e, v))
        return dfs(0)

    # 2054. 两个最好的不重叠活动 (Two Best Non-Overlapping Events)
    def maxTwoEvents(self, events: List[List[int]]) -> int:
        n = len(events)
        # 不写排序规则，默认按照第一维从小到大排序
        # events.sort(key=lambda k: k[0])
        events.sort()

        @cache
        def dfs(i: int, j: int) -> int:
            if i == n or j == 2:
                return 0
            p = bisect.bisect_left(events, events[i][1] + 1, key=lambda k: k[0])
            return max(dfs(i + 1, j), dfs(p, j + 1) + events[i][2])

        return dfs(0, 0)

    # 2050. 并行课程 III (Parallel Courses III)
    def minimumTime(self, n: int, relations: List[List[int]], time: List[int]) -> int:
        g = collections.defaultdict(list)
        deg = [0] * n
        for a, b in relations:
            g[a - 1].append(b - 1)
            deg[b - 1] += 1
        res = 0
        dp = [0] * n
        q = []
        for i in range(n):
            if not deg[i]:
                q.append(i)
        while q:
            x = q.pop()
            t = time[x] + dp[x]
            res = max(res, t)
            for y in g[x]:
                dp[y] = max(dp[y], t)
                deg[y] -= 1
                if not deg[y]:
                    q.append(y)
        return res

    # 124. 二叉树中的最大路径和 (Binary Tree Maximum Path Sum)
    # LCR 051. 二叉树中的最大路径和
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        def dfs(root: TreeNode) -> int:
            if root is None:
                return 0
            left = max(0, dfs(root.left))
            right = max(0, dfs(root.right))
            nonlocal res
            res = max(res, left + right + root.val)
            return max(left, right) + root.val

        res = -inf
        dfs(root)
        return res

    # 264. 丑数 II (Ugly Number II)
    def nthUglyNumber(self, n: int) -> int:
        q = [1]
        heapq.heapify(q)
        s = set()
        s.add(1)
        res = 1
        for i in range(n):
            pop = heapq.heappop(q)
            res = pop
            for j in [2, 3, 5]:
                nxt = j * pop
                if not (nxt in s):
                    heapq.heappush(q, nxt)
                    s.add(nxt)
        return res

    # 2240. 买钢笔和铅笔的方案数 (Number of Ways to Buy Pens and Pencils)
    def waysToBuyPensPencils(self, total: int, cost1: int, cost2: int) -> int:
        res = 0
        cnt = 0
        while cnt * cost1 <= total:
            left = total - cnt * cost1
            res += left // cost2 + 1
            cnt += 1
        return res

    # 543. 二叉树的直径 (Diameter of Binary Tree)
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        # 统计边的数量
        def dfs(node: Optional[TreeNode]) -> int:
            if node is None:
                return -1
            left = dfs(node.left) + 1
            right = dfs(node.right) + 1
            nonlocal res
            res = max(res, left + right)
            return max(left, right)

        res = 0
        dfs(root)
        return res

    # 统计点的数量
    #  def dfs(root: Optional[TreeNode]) -> int:
    #     if not root:
    #        return 0
    #     left = dfs(root.left)
    #     right = dfs(root.right)
    #     nonlocal res
    #     res = max(res, left + right + 1)
    #     return max(left, right) + 1
    #  dfs(root)
    #  return res - 1

    # 2246. 相邻字符不同的最长路径 (Longest Path With Different Adjacent Characters)
    def longestPath(self, parent: List[int], s: str) -> int:
        def dfs(x: int) -> int:
            mx = 0
            pre = 0
            for y in g[x]:
                cur = dfs(y)
                if s[x] != s[y]:
                    mx = max(mx, cur + pre)
                    pre = max(pre, cur)
            nonlocal res
            res = max(res, mx + 1)
            return pre + 1

        n = len(s)
        g = [[] for _ in range(n)]
        for i, v in enumerate(parent[1:], 1):
            g[v].append(i)
        res = 0
        dfs(0)
        return res

    # 687. 最长同值路径 (Longest Univalue Path) --边的数量
    def longestUnivaluePath(self, root: Optional[TreeNode]) -> int:
        res = 0

        # 边的数量
        def dfs(root: Optional[TreeNode]) -> int:
            if root is None:
                return -1
            left = dfs(root.left) + 1
            right = dfs(root.right) + 1
            cur = 0
            mx = 0
            if root.left and root.val == root.left.val:
                cur += left
                mx = left
            if root.right and root.val == root.right.val:
                cur += right
                mx = max(mx, right)
            nonlocal res
            res = max(res, cur)
            return mx

        dfs(root)
        return res

    # 687. 最长同值路径 (Longest Univalue Path) --点的数量
    def longestUnivaluePath(self, root: Optional[TreeNode]) -> int:
        # 点的数量
        def dfs(node: TreeNode) -> int:
            if node is None:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            mx = 0
            cur = 1
            if node.left and node.left.val == node.val:
                cur += left
                mx = max(mx, left)
            if node.right and node.right.val == node.val:
                cur += right
                mx = max(mx, right)
            nonlocal res
            res = max(res, cur)
            return mx + 1

        res = 0
        dfs(root)
        return max(0, res - 1)

    # 2538. 最大价值和与最小价值和的差值 (Difference Between Maximum and Minimum Price Sum)
    def maxOutput(self, n: int, edges: List[List[int]], price: List[int]) -> int:
        g = [[] for _ in range(n)]
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
        res = 0

        def dfs(x: int, fa: int) -> List[int]:
            m0 = price[x]
            m1 = 0
            for y in g[x]:
                if y != fa:
                    cur = dfs(y, x)
                    nonlocal res
                    res = max(res, cur[0] + m1, cur[1] + m0)
                    m0 = max(m0, cur[0] + price[x])
                    m1 = max(m1, cur[1] + price[x])
            return [m0, m1]

        dfs(0, -1)
        return res

    # 1617. 统计子树中城市之间最大距离 (Count Subtrees With Max Distance Between Cities)
    def countSubgraphsForEachDiameter(
        self, n: int, edges: List[List[int]]
    ) -> List[int]:
        def dfs(x: int, fa: int) -> None:
            nonlocal c
            c |= 1 << x
            for y in g[x]:
                if y != fa and s & (1 << y) != 0:
                    dfs(y, x)

        def dfs_diameter(x: int, fa: int) -> int:
            res = 0
            pre = 0
            for y in g[x]:
                if y != fa and s & (1 << y) != 0:
                    cur = dfs_diameter(y, x)
                    res = max(res, cur + pre)
                    pre = max(cur, pre)
            nonlocal mx
            mx = max(mx, res + 1)
            return pre + 1

        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u - 1].append(v - 1)
            g[v - 1].append(u - 1)
        res = [0] * (n - 1)
        for i in range(1, 1 << n):
            s = i
            lb = (s & -s).bit_length() - 1
            c = 0
            dfs(lb, -1)
            if s == c:
                mx = 0
                dfs_diameter(lb, -1)
                if mx - 1 >= 1:
                    res[mx - 2] += 1
        return res

    # 2511. 最多可以摧毁的敌人城堡数目 (Maximum Enemy Forts That Can Be Captured)
    def captureForts(self, forts: List[int]) -> int:
        res = 0
        i = 0
        n = len(forts)
        while i < n:
            while i < n and forts[i] == 0:
                i += 1
            j = i
            i += 1
            while i < n and forts[i] == 0:
                i += 1
            if i < n and forts[i] + forts[j] == 0:
                res = max(res, i - j - 1)
        return res

    # 66. 加一 (Plus One)
    def plusOne(self, digits: List[int]) -> List[int]:
        n = len(digits)
        for i in range(n - 1, -1, -1):
            digits[i] += 1
            if digits[i] // 10 == 0:
                return digits
            digits[i] %= 10
        digits.insert(0, 1)
        return digits

    # 58. 最后一个单词的长度 (Length of Last Word)
    def lengthOfLastWord(self, s: str) -> int:
        i = len(s) - 1
        while i >= 0 and s[i] == " ":
            i -= 1
        j = i
        while i >= 0 and s[i] != " ":
            i -= 1
        return j - i

    # 1921. 消灭怪物的最大数量 (Eliminate Maximum Number of Monsters)
    def eliminateMaximum(self, dist: List[int], speed: List[int]) -> int:
        arrivalTimes = sorted([(d - 1) // s + 1 for d, s in zip(dist, speed)])
        for attackTime, arrivalTime in enumerate(arrivalTimes):
            if arrivalTime <= attackTime:
                return attackTime
        return len(arrivalTimes)

    # 2839. 判断通过操作能否让字符串相等 I (Check if Strings Can be Made Equal With Operations I)
    # 2840. 判断通过操作能否让字符串相等 II (Check if Strings Can be Made Equal With Operations II)
    def canBeEqual(self, s1: str, s2: str) -> bool:
        return sorted(s1[::2]) == sorted(s2[::2]) and sorted(s1[1::2]) == sorted(
            s2[1::2]
        )

    # 2843. 统计对称整数的数目 (Count Symmetric Integers)
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        def cal(x: int) -> int:
            @cache
            def dfs(i: int, j: int, diff: int, is_limit: bool, is_num: bool) -> int:
                if i == n:
                    return diff == 0
                res = 0
                if not is_num:
                    res += dfs(i + 1, j, diff, False, False)
                    if (n - i) & 1:
                        return res
                up = int(s[i]) if is_limit else 9
                for d in range(0 if is_num else 1, up + 1):
                    cur = (
                        diff + d
                        if not is_num or i - j + 1 <= (n - j) >> 1
                        else diff - d
                    )
                    if cur < 0:
                        break
                    res += dfs(
                        i + 1, i if not is_num else j, cur, is_limit and up == d, True
                    )
                return res

            s = str(x)
            n = len(s)
            return dfs(0, 0, 0, True, False)

        return cal(high) - cal(low - 1)

    # 2844. 生成特殊数字的最少操作 (Minimum Operations to Make a Special Number)
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        res = n
        if "0" in num:
            res -= 1

        def check(s: str) -> int:
            i = num.rfind(s[1])
            if i < 0:
                return n
            j = num.rfind(s[0], 0, i)
            if j < 0:
                return n
            return n - j - 2

        return min(res, check("00"), check("50"), check("75"), check("25"))

    # 2845. 统计趣味子数组的数目 (Count of Interesting Subarrays)
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        d = collections.defaultdict(int)
        d[0] = 1
        pre = 0
        res = 0
        for x in nums:
            pre += x % modulo == k
            pre %= modulo
            res += d[(pre - k) % modulo]
            d[pre] += 1
        return res

    # 2605. 从两个数字数组里生成最小数字 (Form Smallest Number From Two Digit Arrays)
    def minNumber(self, nums1: List[int], nums2: List[int]) -> int:
        m1 = 0
        m2 = 0
        for x in nums1:
            m1 |= 1 << x
        for x in nums2:
            m2 |= 1 << x
        m = m1 & m2
        if m:
            return (m & -m).bit_length() - 1
        x = (m1 & -m1).bit_length() - 1
        y = (m2 & -m2).bit_length() - 1
        return min(x * 10 + y, y * 10 + x)

    # 23. 合并 K 个升序链表 (Merge k Sorted Lists)
    # 比较堆中节点的大小
    ListNode.__lt__ = lambda a, b: a.val < b.val

    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next

        cur = dummy = ListNode()
        list = [head for head in lists if head]
        heapq.heapify(list)
        while list:
            node = heapq.heappop(list)
            if node.next:
                heapq.heappush(list, node.next)
            cur.next = node
            cur = cur.next
        return dummy.next

    # 1269. 停在原地的方案数 (Number of Ways to Stay in the Same Place After Some Steps)
    def numWays(self, steps: int, arrLen: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i < 0 or i > j or i >= arrLen:
                return 0
            if j == 0:
                return i == 0
            return (dfs(i + 1, j - 1) + dfs(i - 1, j - 1) + dfs(i, j - 1)) % MOD

        MOD = 10**9 + 7
        return dfs(0, steps)

    # 2571. 将整数减少到零需要的最少操作数 (Minimum Operations to Reduce an Integer to 0)
    def minOperations(self, n: int) -> int:

        @cache
        def dfs(i: int) -> int:
            if (i & (i - 1)) == 0:
                return 1
            x = i & -i
            return min(dfs(i + x), dfs(i - x)) + 1

        return dfs(n)

    # 2571. 将整数减少到零需要的最少操作数 (Minimum Operations to Reduce an Integer to 0)
    def minOperations(self, n: int) -> int:
        res = 1
        while n & (n - 1):
            lb = n & -n
            if n & (lb << 1):
                n += lb
            else:
                n -= lb
            res += 1
        return res

    # 1964. 找出到每个位置为止最长的有效障碍赛跑路线 (Find the Longest Valid Obstacle Course at Each Position)
    def longestObstacleCourseAtEachPosition(self, obstacles: List[int]) -> List[int]:
        n = len(obstacles)
        list = []
        res = [0] * n
        for i, v in enumerate(obstacles):
            if len(list) == 0 or v >= list[-1]:
                list.append(v)
                res[i] = len(list)
            else:
                # list中大于或等于 v + 1 的最小值的最小索引
                pos = bisect.bisect_left(list, v + 1)
                list[pos] = v
                res[i] = pos + 1
        return res

    # 300. 最长递增子序列 (Longest Increasing Subsequence)
    def lengthOfLIS(self, nums: List[int]) -> int:
        list = []
        for i, v in enumerate(nums):
            if len(list) == 0 or v > list[-1]:
                list.append(v)
            else:
                # list中小于或等于 v - 1 的最大值的最大索引
                pos = bisect.bisect_right(list, v - 1)
                list[pos] = v
        return len(list)

    # 559. N 叉树的最大深度 (Maximum Depth of N-ary Tree)
    def maxDepth(self, root: Node) -> int:
        if not root:
            return 0
        res = 0
        for y in root.children:
            res = max(res, self.maxDepth(y))
        return res + 1

    # 199. 二叉树的右视图 (Binary Tree Right Side View)
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        s = set()
        res = []

        def dfs(root: Optional[TreeNode], d: int) -> None:
            if not root:
                return
            if not d in s:
                res.append(root.val)
                s.add(d)
            dfs(root.right, d + 1)
            dfs(root.left, d + 1)

        dfs(root, 0)
        return res

    # 200. 岛屿数量 (Number of Islands)
    def numIslands(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0])

        def dfs(i: int, j: int) -> None:
            if i < 0 or i >= m or j < 0 or j >= n or not int(grid[i][j]):
                return
            grid[i][j] = 0
            dfs(i + 1, j)
            dfs(i, j + 1)
            dfs(i - 1, j)
            dfs(i, j - 1)

        res = 0
        for i in range(m):
            for j in range(n):
                if int(grid[i][j]):
                    res += 1
                    dfs(i, j)
        return res

    # 865. 具有所有最深节点的最小子树 (Smallest Subtree with all the Deepest Nodes)
    # 1123. 最深叶节点的最近公共祖先 (Lowest Common Ancestor of Deepest Leaves)
    def subtreeWithAllDeepest(self, root: TreeNode) -> TreeNode:
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        res = None
        maxD = -1

        def dfs(root: TreeNode, d: int) -> int:
            nonlocal maxD, res
            if not root:
                maxD = max(maxD, d)
                return d
            left = dfs(root.left, d + 1)
            right = dfs(root.right, d + 1)
            if left == right and left == maxD:
                res = root
            return max(left, right)

        dfs(root, 0)
        return res

    # 104. 二叉树的最大深度 (Maximum Depth of Binary Tree)
    # LCR 175. 计算二叉树的深度
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        res = 0

        def dfs(root: Optional[TreeNode], d: int) -> None:
            nonlocal res
            if not root:
                res = max(res, d)
                return
            dfs(root.left, d + 1)
            dfs(root.right, d + 1)

        dfs(root, 0)
        return res

    # 236. 二叉树的最近公共祖先 (Lowest Common Ancestor of a Binary Tree)
    # 面试题 04.08. 首个共同祖先 (First Common Ancestor LCCI)
    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        if root in (None, p, q):
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        return left if left else right

    # 235. 二叉搜索树的最近公共祖先 (Lowest Common Ancestor of a Binary Search Tree)
    # 剑指 Offer 68 - I. 二叉搜索树的最近公共祖先
    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        if min(p.val, q.val) <= root.val <= max(p.val, q.val):
            return root
        if root.val < min(p.val, q.val):
            return self.lowestCommonAncestor(root.right, p, q)
        return self.lowestCommonAncestor(root.left, p, q)

    # 2096. 从二叉树一个节点到另一个节点每一步的方向 (Step-By-Step Directions From a Binary Tree Node to Another)
    def getDirections(
        self, root: Optional[TreeNode], startValue: int, destValue: int
    ) -> str:
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        def lca(root: Optional[TreeNode], start: int, dest: int) -> TreeNode:
            if not root or root.val in (start, dest):
                return root
            left = lca(root.left, start, dest)
            right = lca(root.right, start, dest)
            if left and right:
                return root
            return left if left else right

        def dfs(root: Optional[TreeNode], val: int, isStart: bool) -> bool:
            if not root:
                return False
            if root.val == val:
                return True
            nonlocal s
            if dfs(root.left, val, isStart):
                s += "U" if isStart else "L"
                return True
            if dfs(root.right, val, isStart):
                s += "U" if isStart else "R"
                return True
            return False

        s = ""
        l = lca(root, startValue, destValue)
        res = ""
        dfs(l, startValue, True)
        res = s
        s = ""
        dfs(l, destValue, False)
        for c in reversed(s):
            res += c
        return res

    # 2594. 修车的最少时间 (Minimum Time to Repair Cars)
    def repairCars(self, ranks: List[int], cars: int) -> int:
        left = 0
        right = max(x for x in ranks) * cars * cars
        res = 0
        while left <= right:
            mid = left + ((right - left) >> 1)
            if sum(int(sqrt(mid // x)) for x in ranks) >= cars:
                res = mid
                right = mid - 1
            else:
                left = mid + 1
        return res

    # 2182. 构造限制重复的字符串 (Construct String With Repeat Limit)
    def repeatLimitedString(self, s: str, repeatLimit: int) -> str:
        n = len(s)
        res = ""
        cnts = [0] * 26
        for c in s:
            cnts[ord(c) - ord("a")] += 1
        for i in range(25, -1, -1):
            while cnts[i] > 0:
                if cnts[i] <= repeatLimit:
                    while cnts[i] > 0:
                        res += chr(ord("a") + i)
                        cnts[i] -= 1
                else:
                    for j in range(repeatLimit):
                        res += chr(ord("a") + i)
                    j = i - 1
                    while j >= 0:
                        if cnts[j]:
                            res += chr(ord("a") + j)
                            cnts[j] -= 1
                            break
                        j -= 1
                    if j < 0:
                        return res
                    cnts[i] -= repeatLimit
        return res

    # 2187. 完成旅途的最少时间 (Minimum Time to Complete Trips)
    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        def check(t: int) -> bool:
            return sum(t // c for c in time) >= totalTrips

        left = 1
        right = max(time) * totalTrips
        res = 1
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = mid
                right = mid - 1
            else:
                left = mid + 1
        return res

    # 2190. 数组中紧跟 key 之后出现最频繁的数字 (Most Frequent Number Following Key In an Array)
    def mostFrequent(self, nums: List[int], key: int) -> int:
        n = len(nums)
        res = -1
        cnt = 0
        d = collections.defaultdict(int)
        for i in range(1, n):
            if nums[i - 1] == key:
                d[nums[i]] += 1
                if d[nums[i]] > cnt:
                    cnt = d[nums[i]]
                    res = nums[i]
        return res

    # 2191. 将杂乱无章的数字排序 (Sort the Jumbled Numbers)
    def sortJumbled(self, mapping: List[int], nums: List[int]) -> List[int]:
        n = len(nums)
        arr = [0] * n
        for i, v in enumerate(nums):
            sum = 0
            s = str(v)
            for j in range(len(s)):
                sum = sum * 10 + mapping[int(s[j])]
            arr[i] = sum
        res = []
        for x, _ in sorted(zip(nums, arr), key=lambda k: k[1]):
            res.append(x)
        return res

    # 2192. 有向无环图中一个节点的所有祖先 (All Ancestors of a Node in a Directed Acyclic Graph)
    def getAncestors(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        deg = [0] * n
        g = [[] for _ in range(n)]
        for f, t in edges:
            deg[t] += 1
            g[f].append(t)
        res = [set() for _ in range(n)]
        q = deque()
        for i, d in enumerate(deg):
            if d == 0:
                q.append(i)
        while q:
            x = q.popleft()
            for y in g[x]:
                res[y].add(x)
                res[y].update(res[x])
                deg[y] -= 1
                if deg[y] == 0:
                    q.append(y)
        return [sorted(res[i]) for i in range(n)]

    # 2195. 向数组中追加 K 个整数 (Append K Integers With Minimal Sum)
    def minimalKSum(self, nums: List[int], k: int) -> int:
        nums.sort()
        pre = 0
        res = 0
        for x in nums:
            if x - pre <= 1:
                pre = x
                continue
            first = pre + 1
            last = min(x - 1, first + k - 1)
            res += (first + last) * (last - first + 1) // 2
            pre = x
            k -= last - first + 1
        first = nums[-1] + 1
        last = first + k - 1
        res += (first + last) * (last - first + 1) // 2
        return res

    # 2196. 根据描述创建二叉树 (Create Binary Tree From Descriptions)
    def createBinaryTree(self, descriptions: List[List[int]]) -> Optional[TreeNode]:
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        s = set()
        d = collections.defaultdict(TreeNode)
        for i in range(1, 10**5 + 1):
            d[i] = TreeNode(i)
        for pa, son, is_left in descriptions:
            s.add(son)
        root = None
        for pa, son, is_left in descriptions:
            if not pa in s:
                root = d[pa]
                break
        for pa, son, is_left in descriptions:
            if is_left:
                d[pa].left = d[son]
            else:
                d[pa].right = d[son]
        return root

    # 2707. 字符串中的额外字符 (Extra Characters in a String)
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)

        class Trie:

            def __init__(self) -> None:
                self.child = [None] * 26
                self.is_word = False

            def insert(self, s: str) -> None:
                node = self
                for c in s:
                    index = ord(c) - ord("a")
                    if node.child[index] is None:
                        node.child[index] = Trie()
                    node = node.child[index]
                node.is_word = True

            def check(self, s: str) -> list:
                res = []
                node = self
                for i, c in enumerate(s):
                    index = ord(c) - ord("a")
                    if node.child[index] is None:
                        break
                    node = node.child[index]
                    if node.is_word:
                        res.append(i)
                return res

        @cache
        def dfs(i: int) -> int:
            if i == n:
                return 0
            res = dfs(i + 1) + 1
            for j in trie.check(s[i:]):
                res = min(res, dfs(i + j + 1))
            return res

        trie = Trie()
        for d in dictionary:
            trie.insert(d)
        return dfs(0)

    # 2651. 计算列车到站时间 (Calculate Delayed Arrival Time)
    def findDelayedArrivalTime(self, arrivalTime: int, delayedTime: int) -> int:
        return (arrivalTime + delayedTime) % 24

    # 788. 旋转数字 (Rotated Digits)
    def rotatedDigits(self, n: int) -> int:
        s = str(n)
        l = len(s)
        legal = [0, 0, 1, -1, -1, 1, 1, -1, 0, 1]

        @cache
        def dfs(i: int, hasDiff: bool, isLimit: bool, isNum: bool) -> int:
            if i == l:
                return hasDiff
            res = 0
            if not isNum:
                res += dfs(i + 1, False, False, False)
            up = int(s[i]) if isLimit else 9
            for d in range(0 if isNum else 1, up + 1):
                if legal[d] != -1:
                    res += dfs(i + 1, hasDiff or legal[d], isLimit and d == up, True)
            return res

        return dfs(0, False, True, False)

    # 611. 有效三角形的个数 (Valid Triangle Number)
    def triangleNumber(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        res = 0
        for i in range(n - 1, -1, -1):
            j = 0
            k = i - 1
            while j < k:
                if nums[j] + nums[k] > nums[i]:
                    res += k - j
                    k -= 1
                else:
                    j += 1
        return res

    # 657. 机器人能否返回原点 (Robot Return to Origin)
    def judgeCircle(self, moves: str) -> bool:
        c = Counter(moves)
        return c["L"] == c["R"] and c["U"] == c["D"]

    # 2842. 统计一个字符串的 k 子序列美丽值最大的数目 (Count K-Subsequences of a String With Maximum Beauty)
    def countKSubsequencesWithMaxBeauty(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        res = 1
        cnt = Counter(Counter(s).values())
        for c, num in sorted(cnt.items(), reverse=True):
            if num >= k:
                return res * pow(c, k, MOD) * comb(num, k) % MOD
            res *= pow(c, num, MOD)
            k -= num
        return 0

    # 1434. 每个人戴不同帽子的方案数 (Number of Ways to Wear Different Hats to Each Other)
    def numberWays(self, hats: List[List[int]]) -> int:
        MOD = 10**9 + 7
        n = len(hats)
        u = (1 << n) - 1
        g = [[] for _ in range(41)]
        for i in range(n):
            for j in range(len(hats[i])):
                g[hats[i][j]].append(i)

        @cache
        def dfs(i: int, j: int) -> int:
            if j == u:
                return 1
            if i == 41:
                return 0
            res = dfs(i + 1, j)
            for k in g[i]:
                if ((j >> k) & 1) == 0:
                    res += dfs(i + 1, j | (1 << k))
                    res %= MOD
            return res

        return dfs(0, 0)

    # 1372. 二叉树中的最长交错路径 (Longest ZigZag Path in a Binary Tree)
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        res = 0

        def dfs(root: Optional[TreeNode]) -> (int, int):
            if not root:
                return (-1, -1)
            left = dfs(root.left)
            right = dfs(root.right)
            nonlocal res
            res = max(res, left[1] + 1, right[0] + 1)
            return (left[1] + 1, right[0] + 1)

        dfs(root)
        return res

    # 1363. 形成三的最大倍数 (Largest Multiple of Three)
    def largestMultipleOfThree(self, digits: List[int]) -> str:
        def check(start: int, c: int) -> bool:
            for s in range(start, 10, 3):
                while d[s]:
                    d[s] -= 1
                    c -= 1
                    if c == 0:
                        return True
            return False

        d = [0] * 10
        s = 0
        for digit in digits:
            v = int(digit)
            d[v] += 1
            s += v
        if s == 0:
            return "0"
        if s % 3 == 1:
            if not check(1, 1):
                check(2, 2)
        if s % 3 == 2:
            if not check(2, 1):
                check(1, 2)
        res = []
        for i in range(9, -1, -1):
            res.extend()
            while d[i]:
                res.append(str(i))
                d[i] -= 1
        if len(res) == 0:
            return ""
        if res[0] == "0":
            return "0"
        return "".join(res)

    # 1334. 阈值距离内邻居最少的城市 (Find the City With the Smallest Number of Neighbors at a Threshold Distance)
    def findTheCity(
        self, n: int, edges: List[List[int]], distanceThreshold: int
    ) -> int:
        g = [[] for _ in range(n)]
        for a, b, w in edges:
            g[a].append((b, w))
            g[b].append((a, w))

        def check(i: int) -> int:
            dis = [inf] * n
            dis[i] = 0
            q = [(0, i)]
            heapq.heapify(q)
            while q:
                (d, x) = heapq.heappop(q)
                for y, w in g[x]:
                    if d + w < dis[y]:
                        dis[y] = d + w
                        heapq.heappush(q, (dis[y], y))
            return sum(d <= distanceThreshold for d in dis)

        res = -1
        m = inf
        for i in range(n):
            cur = check(i)
            if cur <= m:
                res = i
                m = cur
        return res

    # 210. 课程表 II (Course Schedule II)
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        g = [[] for _ in range(numCourses)]
        deg = [0] * numCourses
        for a, b in prerequisites:
            g[b].append(a)
            deg[a] += 1
        res = []
        q = []
        for i in range(numCourses):
            if not deg[i]:
                q.append(i)
        while q:
            x = q.pop(0)
            res.append(x)
            for y in g[x]:
                deg[y] -= 1
                if not deg[y]:
                    q.append(y)
        return res if len(res) == numCourses else []

    # 2767. 将字符串分割为最少的美丽子字符串 (Partition String Into Minimum Beautiful Substrings)
    def minimumBeautifulSubstrings(self, s: str) -> int:

        def is_valid(v: int) -> bool:
            while v % 5 == 0:
                v //= 5
            return v == 1

        @cache
        def dfs(i: int) -> int:
            if i == n:
                return 0
            if s[i] == "0":
                return inf
            res = inf
            _s = 0
            for j in range(i, n):
                _s = (_s << 1) + int(s[j])
                if is_valid(_s):
                    res = min(res, dfs(j + 1) + 1)
            return res

        n = len(s)
        res = dfs(0)
        return -1 if res == inf else res

    # 8029. 与车相交的点 (Points That Intersect With Cars)
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        max_val = max(end for (_, end) in nums)
        d = [0] * (max_val + 2)
        for start, end in nums:
            d[start] += 1
            d[end + 1] -= 1
        return sum(x > 0 for x in accumulate(d))

    # 8049. 判断能否在给定时间到达单元格 (Determine if a Cell Is Reachable at a Given Time)
    def isReachableAtTime(self, sx: int, sy: int, fx: int, fy: int, t: int) -> bool:
        max_val = max(abs(sx - fx), abs(sy - fy))
        if max_val == 0 and t == 1:
            return False
        return max_val <= t

    # 2850. 将石头分散到网格图的最少移动次数 (Minimum Moves to Spread Stones Over Grid)
    def minimumMoves(self, grid: List[List[int]]) -> int:

        @cache
        def dfs(i: int, j: int) -> int:
            if j == u:
                return 0
            res = inf
            (x, y) = give[i]
            val = grid[x][y] - 1
            candidate = u ^ j
            c = candidate
            while c:
                if c.bit_count() == val:
                    dis = 0
                    copy = c
                    while copy:
                        index = (copy & -copy).bit_length() - 1
                        (nx, ny) = need[index]
                        dis += abs(nx - x) + abs(ny - y)
                        copy &= copy - 1
                    res = min(res, dfs(i + 1, j | c) + dis)
                c = (c - 1) & candidate
            return res

        give = []
        need = []
        for i in range(3):
            for j in range(3):
                if grid[i][j] > 1:
                    give.append((i, j))
                elif grid[i][j] == 0:
                    need.append((i, j))
        m = len(give)
        n = len(need)
        u = (1 << n) - 1
        return dfs(0, 0)

    # 1462. 课程表 IV (Course Schedule IV)
    def checkIfPrerequisite(
        self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]
    ) -> List[bool]:
        @cache
        def dfs(x: int, target: int) -> bool:
            if x == target:
                return True
            for y in g[x]:
                if dfs(y, target):
                    return True
            return False

        g = [[] for _ in range(numCourses)]
        for a, b in prerequisites:
            g[a].append(b)
        res = []
        for a, b in queries:
            res.append(dfs(a, b))
        return res

    # 2596. 检查骑士巡视方案 (Check Knight Tour Configuration)
    def checkValidGrid(self, grid: List[List[int]]) -> bool:
        n = len(grid)
        q = [(0, 0)]
        cnt = 0
        dirs = [[2, 1], [1, 2], [-1, 2], [1, -2], [-1, -2], [2, -1], [-2, 1], [-2, -1]]
        while q:
            (x, y) = q.pop(0)
            cnt += 1
            for dx, dy in dirs:
                nx = x + dx
                ny = y + dy
                if (
                    nx >= 0
                    and nx < n
                    and ny >= 0
                    and ny < n
                    and grid[x][y] + 1 == grid[nx][ny]
                ):
                    q.append((nx, ny))
        return cnt == n * n

    # 1222. 可以攻击国王的皇后 (Queens That Can Attack the King)
    def queensAttacktheKing(
        self, queens: List[List[int]], king: List[int]
    ) -> List[List[int]]:
        n = 8
        s = set(map(tuple, queens))
        dirs = [0, 1], [0, -1], [1, -1], [1, 1], [-1, 1], [-1, -1], [1, 0], [-1, 0]
        res = []
        for dx, dy in dirs:
            x = king[0] + dx
            y = king[1] + dy
            while n > x >= 0 and n > y >= 0:
                if (x, y) in s:
                    res.append([x, y])
                    break
                x += dx
                y += dy
        return res

    # 1745. 分割回文串 IV (Palindrome Partitioning IV)
    def checkPartitioning(self, s: str) -> bool:
        n = len(s)
        is_valid = [[False] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if (
                    i == j
                    or j - i == 1
                    and s[i] == s[j]
                    or j - i > 1
                    and s[i] == s[j]
                    and is_valid[i + 1][j - 1]
                ):
                    is_valid[i][j] = True

        @cache
        def dfs(i: int, j: int) -> bool:
            if i == n:
                return j == 3
            if j == 3:
                return False
            return any(is_valid[i][k] and dfs(k + 1, j + 1) for k in range(i, n))

        return dfs(0, 0)

    # 1723. 完成所有工作的最短时间 (Find Minimum Time to Finish All Jobs)
    def minimumTimeRequired(self, jobs: List[int], k: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j == u:
                return 0
            if i == k or k - i > (u ^ j).bit_count():
                return inf
            sub = c = u ^ j
            res = inf
            while sub:
                if s[sub] < res and dfs(i + 1, j | sub) < res:
                    res = max(dfs(i + 1, j | sub), s[sub])
                sub = (sub - 1) & c
            return res

        n = len(jobs)
        u = (1 << n) - 1
        s = [0] * (1 << n)
        for i in range(1, 1 << n):
            s[i] = s[i & (i - 1)] + jobs[(i & -i).bit_length() - 1]
        return dfs(0, 0)

    # LCP 50. 宝石补给
    def giveGem(self, gem: List[int], operations: List[List[int]]) -> int:
        for a, b in operations:
            origin = gem[a]
            gem[a] = (gem[a] + 1) // 2
            gem[b] += origin - gem[a]
        return max(gem) - min(gem)

    # 1681. 最小不兼容性 (Minimum Incompatibility)
    def minimumIncompatibility(self, nums: List[int], k: int) -> int:
        def check(i: int) -> bool:
            c = Counter()
            for j in range(n):
                if i >> j & 1 == 1:
                    c[nums[j]] += 1
                    if c[nums[j]] > 1:
                        return False
            return True

        @cache
        def dfs(i: int) -> int:
            if i == u:
                return 0
            sub = c = i ^ u
            res = inf
            while sub:
                if d[sub] != inf:
                    res = min(res, dfs(i | sub) + d[sub])
                sub = (sub - 1) & c
            return res

        c = Counter(nums)
        if max(c.values()) > k:
            return -1
        n = len(nums)
        mx = [0] * (1 << n)
        mi = [inf] * (1 << n)
        d = [inf] * (1 << n)
        for i in range(1, 1 << n):
            mx[i] = max(mx[i & (i - 1)], nums[(i & -i).bit_length() - 1])
            mi[i] = min(mi[i & (i - 1)], nums[(i & -i).bit_length() - 1])
            if i.bit_count() == n // k and check(i):
                d[i] = mx[i] - mi[i]
        u = (1 << n) - 1
        return dfs(0)

    # 542. 01 矩阵 (01 Matrix)
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m = len(mat)
        n = len(mat[0])
        vis = [[False] * n for _ in range(m)]
        q = []
        for i in range(m):
            for j in range(n):
                if not mat[i][j]:
                    q.append((i, j))
                    vis[i][j] = True
        dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        level = 0
        while q:
            level += 1
            size = len(q)
            for _ in range(size):
                (x, y) = q.pop(0)
                for dx, dy in dirs:
                    nx = x + dx
                    ny = y + dy
                    if nx >= 0 and nx < m and ny >= 0 and ny < n and not vis[nx][ny]:
                        vis[nx][ny] = True
                        mat[nx][ny] = level
                        q.append((nx, ny))
        return mat

    # 494. 目标和 (Target Sum)
    # LCR 102. 目标和
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == len(nums):
                return int(j == target)
            return dfs(i + 1, nums[i] + j) + dfs(i + 1, j - nums[i])

        s = sum(nums)
        if s < target or target < -s:
            return 0
        return dfs(0, 0)

    # 494. 目标和 (Target Sum)
    # LCR 102. 目标和
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i < 0:
                return int(j == 0)
            res = dfs(i - 1, j)
            if j - nums[i] >= 0:
                res += dfs(i - 1, j - nums[i])
            return res

        n = len(nums)
        s = sum(nums) - abs(target)
        if s < 0 or s % 2:
            return 0
        m = s // 2
        return dfs(n - 1, m)

    # 474. 一和零 (Ones and Zeroes)
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if j > m or k > n:
                return -inf
            if i == len(arr):
                return 0
            return max(dfs(i + 1, j, k), dfs(i + 1, j + arr[i][0], k + arr[i][1]) + 1)

        arr = []
        for s in strs:
            one = sum(int(x) for x in s)
            arr.append([len(s) - one, one])
        return dfs(0, 0, 0)

    # 486. 预测赢家 (Predict the Winner)
    def predictTheWinner(self, nums: List[int]) -> bool:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i == j:
                return nums[i] * k
            return (
                max(
                    (dfs(i + 1, j, -k) + nums[i] * k) * k,
                    (dfs(i, j - 1, -k) + nums[j] * k) * k,
                )
                * k
            )

        return dfs(0, len(nums) - 1, 1) >= 0

    # 2791. 树中可以形成回文的路径数 (Count Paths That Can Form a Palindrome in a Tree)
    def countPalindromePaths(self, parent: List[int], s: str) -> int:
        def dfs(x: int, mask: int) -> None:
            nonlocal res
            res += dic[mask]
            dic[mask] += 1
            for i in range(26):
                res += dic[mask ^ (1 << i)]
            for y, m in g[x]:
                dfs(y, mask ^ m)

        n = len(s)
        g = [[] * n for _ in range(n)]
        for i in range(1, n):
            g[parent[i]].append((i, 1 << (ord(s[i]) - ord("a"))))
        res = 0
        dic = collections.defaultdict(int)
        dfs(0, 0)
        return res

    # 2583. 二叉树中的第 K 大层和 (Kth Largest Sum in a Binary Tree)
    def kthLargestLevelSum(self, root: Optional[TreeNode], k: int) -> int:
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        def dfs(root: Optional[TreeNode], level: int) -> None:
            if not root:
                return
            dic[level] += root.val
            dfs(root.left, level + 1)
            dfs(root.right, level + 1)

        dic = collections.defaultdict(int)
        dfs(root, 1)
        if k > len(dic):
            return -1
        return sorted(dic.values(), reverse=True)[k - 1]

    # 2583. 二叉树中的第 K 大层和 (Kth Largest Sum in a Binary Tree)
    def kthLargestLevelSum(self, root: Optional[TreeNode], k: int) -> int:
        arr = []
        q = deque([root])
        while q:
            size = len(q)
            s = 0
            for _ in range(size):
                x = q.popleft()
                s += x.val
                if x.left:
                    q.append(x.left)
                if x.right:
                    q.append(x.right)
            arr.append(s)
        if len(arr) < k:
            return -1
        return sorted(arr)[-k]

    # 2509. 查询树中环的长度 (Cycle Length Queries in a Tree)
    def cycleLengthQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        res = []
        for a, b in queries:
            s = 1
            while a != b:
                if a < b:
                    tmp = a
                    a = b
                    b = tmp
                a >>= 1
                s += 1
            res.append(s)
        return res

    # 2477. 到达首都的最少油耗 (Minimum Fuel Cost to Report to the Capital)
    def minimumFuelCost(self, roads: List[List[int]], seats: int) -> int:
        n = len(roads) + 1
        cnt = [1] * n
        deg = [0] * n
        g = [[] * n for _ in range(n)]
        for u, v in roads:
            g[u].append(v)
            g[v].append(u)
            deg[u] += 1
            deg[v] += 1
        res = 0
        q = []
        for i in range(n):
            if deg[i] == 1:
                q.append(i)
        while q:
            x = q.pop()
            if x == 0:
                continue
            deg[x] -= 1
            for y in g[x]:
                if deg[y] == 0:
                    continue
                res += (cnt[x] + seats - 1) // seats
                cnt[y] += cnt[x]
                deg[y] -= 1
                if deg[y] == 1:
                    q.append(y)
        return res

    # 2477. 到达首都的最少油耗 (Minimum Fuel Cost to Report to the Capital)
    def minimumFuelCost2(self, roads: List[List[int]], seats: int) -> int:
        def dfs(x: int, fa: int) -> int:
            size = 1
            for y in g[x]:
                if y != fa:
                    size += dfs(y, x)
            if x:
                nonlocal res
                res += (size + seats - 1) // seats
            return size

        n = len(roads) + 1
        g = [[] for _ in range(n)]
        for a, b in roads:
            g[a].append(b)
            g[b].append(a)
        res = 0
        dfs(0, -1)
        return res

    # 2476. 二叉搜索树最近节点查询 (Closest Nodes Queries in a Binary Search Tree)
    def closestNodes(
        self, root: Optional[TreeNode], queries: List[int]
    ) -> List[List[int]]:
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        def dfs(root: Optional[TreeNode]) -> None:
            if not root:
                return
            dfs(root.left)
            arr.append(root.val)
            dfs(root.right)

        arr = []
        dfs(root)
        n = len(arr)
        res = []
        for q in queries:
            x = bisect.bisect_right(arr, q)
            min_val = arr[x - 1] if x else -1
            y = bisect.bisect_left(arr, q)
            max_val = arr[y] if y < n else -1
            res.append([min_val, max_val])
        return res

    # 8039. 使数组成为递增数组的最少右移次数 (Minimum Right Shifts to Sort the Array)
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        cnt = 0
        j = -1
        for i in range(1, n):
            if nums[i - 1] > nums[i]:
                cnt += 1
                j = i
        if not cnt:
            return 0
        if nums[0] < nums[n - 1]:
            cnt += 1
        if cnt > 1:
            return -1
        return n - j

    # 2856. 删除数对后的最小数组长度 (Minimum Array Length After Pair Removals)
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        max_cnt = max(Counter(nums).values())
        if max_cnt >= n - max_cnt:
            return max_cnt - (n - max_cnt)
        return n & 1

    # 6988. 统计距离为 k 的点对 (Count Pairs of Points With Distance k)
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        res = 0
        dic = collections.defaultdict(int)
        for x, y in coordinates:
            for i in range(k + 1):
                tx = x ^ i
                ty = y ^ (k - i)
                res += dic[(tx, ty)]
            dic[(x, y)] += 1
        return res

    # 2858. 可以到达每一个节点的最少边反转次数 (Minimum Edge Reversals So Every Node Is Reachable)
    def minEdgeReversals(self, n: int, edges: List[List[int]]) -> List[int]:
        def reroot(x: int, fa: int) -> None:
            for y, d in g[x]:
                if y != fa:
                    res[y] = res[x] - 2 * d + 1
                    reroot(y, x)

        def dfs0(x: int, fa: int) -> None:
            for y, d in g[x]:
                if y != fa:
                    res[0] += d
                    dfs0(y, x)

        res = [0] * n
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append((v, 0))
            g[v].append((u, 1))
        dfs0(0, -1)
        reroot(0, -1)
        return res

    # 337. 打家劫舍 III (House Robber III)
    def rob(self, root: Optional[TreeNode]) -> int:
        def dfs(node: TreeNode) -> List[int]:
            if node is None:
                return [0, 0]
            # 不偷 // 偷
            [a0, b0] = dfs(node.left)
            [a1, b1] = dfs(node.right)
            return [max(a0, b0) + max(a1, b1), a0 + a1 + node.val]

        return max(dfs(root))

    # 2560. 打家劫舍 IV (House Robber IV)
    def minCapability(self, nums: List[int], k: int) -> int:
        def check(target: int) -> bool:
            cur = 0
            cnt = 0
            for num in nums:
                if num > target:
                    cur += (cnt + 1) // 2
                    cnt = 0
                    if cur >= k:
                        return True
                else:
                    cnt += 1
            cur += (cnt + 1) // 2
            return cur >= k

        left = min(nums)
        right = max(nums)
        res = 0
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = mid
                right = mid - 1
            else:
                left = mid + 1
        return res

    # 206. 反转链表 (Reverse Linked List)
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        pre = None
        cur = head
        while cur:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        return pre

    # 231. 2 的幂 (Power of Two)
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and not (n & (n - 1))

    # 231. 2 的幂 (Power of Two)
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and (n & -n) == n

    # 231. 2 的幂 (Power of Two)
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and n.bit_count() == 1

    # 242. 有效的字母异位词 (Valid Anagram)
    def isAnagram(self, s: str, t: str) -> bool:
        return sorted(s) == sorted(t)

    # 283. 移动零 (Move Zeroes)
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        j = 0
        i = 0
        while i < n:
            if nums[i]:
                nums[j] = nums[i]
                j += 1
            i += 1
        while j < n:
            nums[j] = 0
            j += 1

    # 801. 使序列递增的最小交换次数 (Minimum Swaps To Make Sequences Increasing)
    def minSwap(self, nums1: List[int], nums2: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i < 0:
                return 0
            if j == 0:
                res = inf
                if nums1[i] < nums1[i + 1] and nums2[i] < nums2[i + 1]:
                    res = min(res, dfs(i - 1, 0))
                if nums1[i] < nums2[i + 1] and nums2[i] < nums1[i + 1]:
                    res = min(res, dfs(i - 1, 1) + 1)
                return res
            res = inf
            if nums1[i] < nums2[i + 1] and nums2[i] < nums1[i + 1]:
                res = min(res, dfs(i - 1, 0))
            if nums1[i] < nums1[i + 1] and nums2[i] < nums2[i + 1]:
                res = min(res, dfs(i - 1, 1) + 1)
            return res

        n = len(nums1)
        return min(dfs(n - 2, 0), dfs(n - 2, 1) + 1)

    # LCP 06. 拿硬币
    def minCount(self, coins: List[int]) -> int:
        return sum((x + 1) // 2 for x in coins)

    # 1340. 跳跃游戏 V (Jump Game V)
    def maxJumps(self, arr: List[int], d: int) -> int:
        @cache
        def dfs(i: int) -> int:
            res = 0
            for j in range(i - 1, -1, -1):
                if i - j > d or arr[j] >= arr[i]:
                    break
                res = max(res, dfs(j))
            for j in range(i + 1, n):
                if j - i > d or arr[j] >= arr[i]:
                    break
                res = max(res, dfs(j))
            return res + 1

        n = len(arr)
        return max(dfs(x) for x in range(n))

    # 1255. 得分最高的单词集合 (Maximum Score Words Formed by Letters)
    def maxScoreWords(
        self, words: List[str], letters: List[str], score: List[int]
    ) -> int:
        def dfs(i: int, j: int) -> None:
            if i == n:
                nonlocal res
                res = max(res, j)
                return
            # 不选
            dfs(i + 1, j)
            # 选
            cur = [0] * 26
            for c in words[i]:
                cur[ord(c) - ord("a")] += 1
            if all(a >= b for a, b in zip(cnts, cur)):
                s = 0
                for k in range(26):
                    s += cur[k] * score[k]
                    cnts[k] -= cur[k]
                dfs(i + 1, j + s)
                for k in range(26):
                    cnts[k] += cur[k]

        n = len(words)
        cnts = [0] * 26
        for c in letters:
            cnts[ord(c) - ord("a")] += 1
        res = 0
        dfs(0, 0)
        return res

    # 1227. 飞机座位分配概率 (Airplane Seat Assignment Probability)
    def nthPersonGetsNthSeat(self, n: int) -> float:
        return 1 if n == 1 else 0.5

    # 面试题 08.09. 括号
    # LCR 085. 括号生成
    # 22. 括号生成 (Generate Parentheses)
    def generateParenthesis(self, n: int) -> List[str]:
        def dfs(i: int, j: int) -> None:
            nonlocal s
            if len(s) == n * 2:
                res.append(s)
                return
            if i < n:
                s += "("
                dfs(i + 1, j)
                s = s[: len(s) - 1]
            if j < i:
                s += ")"
                dfs(i, j + 1)
                s = s[: len(s) - 1]

        s = ""
        res = []
        dfs(0, 0)
        return res

    # 面试题 08.01. 三步问题
    def waysToStep(self, n: int) -> int:
        a, b, c = 4, 2, 1
        m = 10**9 + 7
        if n < 3:
            return n
        if n == 3:
            return 4
        for _ in range(n - 3):
            a, b, c = (a + b + c) % m, a, b
        return a

    # 2266. 统计打字方案数 (Count Number of Texts)
    def countTexts(self, pressedKeys: str) -> int:
        n = len(pressedKeys)
        MOD = 10**9 + 7

        @cache
        def dfs(i: int) -> int:
            if i == n:
                return 1
            c = pressedKeys[i]
            j = 3
            if c == "7" or c == "9":
                j = 4
            res = 0
            k = 0
            while i + k < n and k < j and c == pressedKeys[i + k]:
                res += dfs(i + k + 1)
                res %= MOD
                k += 1
            return res

        return dfs(0)

    # 2267. 检查是否有合法括号字符串路径 (Check if There Is a Valid Parentheses String Path)
    def hasValidPath(self, grid: List[List[str]]) -> bool:
        @cache
        def dfs(i: int, j: int, k: int) -> bool:
            if i == m - 1 and j == n - 1:
                return k == 1
            if i == m or j == n or k < 0 or m - i + n - j - 1 < k:
                return False
            return dfs(i + 1, j, k + (1 if grid[i][j] == "(" else -1)) or dfs(
                i, j + 1, k + (1 if grid[i][j] == "(" else -1)
            )

        m = len(grid)
        n = len(grid[0])
        if (m + n) % 2 == 0 or grid[0][0] == ")" or grid[-1][-1] == "(":
            return False
        return dfs(0, 0, 0)

    # 2603. 收集树中金币 (Collect Coins in a Tree)
    def collectTheCoins(self, coins: List[int], edges: List[List[int]]) -> int:
        n = len(coins)
        deg = [0] * n
        g = [[] * n for _ in range(n)]
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
            deg[a] += 1
            deg[b] += 1
        q = []
        for i in range(n):
            if coins[i] == 0 and deg[i] == 1:
                q.append(i)
        left = n
        while q:
            left -= 1
            x = q.pop(0)
            deg[x] = 0
            for y in g[x]:
                deg[y] -= 1
                if deg[y] == 1 and coins[y] == 0:
                    q.append(y)
        for i in range(n):
            if coins[i] == 1 and deg[i] == 1:
                q.append(i)
        cnt = 2
        while q and cnt > 0:
            cnt -= 1
            size = len(q)
            for _ in range(size):
                left -= 1
                x = q.pop(0)
                deg[x] = 0
                for y in g[x]:
                    deg[y] -= 1
                    if deg[y] == 1:
                        q.append(y)
        return max(0, (left - 1) * 2)

    # 1395. 统计作战单位数 (Count Number of Teams)
    def numTeams(self, rating: List[int]) -> int:
        n = len(rating)
        pre1 = [0] * n
        pre2 = [0] * n
        for i in range(1, n):
            pre1[i] = sum(rating[j] < rating[i] for j in range(i))
            pre2[i] = sum(rating[j] > rating[i] for j in range(i))
        suf1 = [0] * n
        suf2 = [0] * n
        for i in range(n - 2, -1, -1):
            suf1[i] = sum(rating[j] > rating[i] for j in range(i + 1, n))
            suf2[i] = sum(rating[j] < rating[i] for j in range(i + 1, n))
        return sum(pre1[i] * suf1[i] + pre2[i] * suf2[i] for i in range(n))

    # 1510. 石子游戏 IV (Stone Game IV)
    def winnerSquareGame(self, n: int) -> bool:
        @cache
        def dfs(i: int) -> bool:
            if i == 0:
                return False
            if i == 1:
                return True
            j = 1
            while j * j <= i:
                if not dfs(i - j * j):
                    return True
                j += 1
            return False

        return dfs(n)

    # 1477. 找两个和为目标值且不重叠的子数组 (Find Two Non-overlapping Sub-arrays Each With Target Sum)
    def minSumOfLengths(self, arr: List[int], target: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j == 2:
                return 0
            if i == n:
                return inf
            res = dfs(i + 1, j)
            for x, y in g[i]:
                res = min(res, dfs(y + 1, j + 1) + y - x + 1)
            return res

        n = len(arr)
        g = [[] for _ in range(n)]
        i = 0
        j = 0
        s = 0
        while j < n:
            s += arr[j]
            while s > target:
                s -= arr[i]
                i += 1
            if s == target:
                g[i].append([i, j])
            j += 1
        res = dfs(0, 0)
        return res if res < inf else -1

    # 2591. 将钱分给最多的儿童 (Distribute Money to Maximum Children)
    def distMoney(self, money: int, children: int) -> int:
        money -= children
        if money < 0:
            return -1
        res = min(money // 7, children)
        money -= res * 7
        children -= res
        if children == 0 and money or children == 1 and money == 3:
            res -= 1
        return res

    # 1993. 树上的操作 (Operations on Tree)
    class LockingTree:

        def __init__(self, parent: List[int]):
            self.n = len(parent)
            self.g = [[] for _ in range(self.n)]
            for i in range(1, self.n):
                self.g[parent[i]].append(i)
            self.locked = [0] * self.n
            self.parent = parent

        def lock(self, num: int, user: int) -> bool:
            if self.locked[num]:
                return False
            self.locked[num] = user
            return True

        def unlock(self, num: int, user: int) -> bool:
            if self.locked[num] != user:
                return False
            self.locked[num] = 0
            return True

        def upgrade(self, num: int, user: int) -> bool:
            def check_ancestor_locked(x: int, num: int) -> bool:
                if x == num:
                    return True
                if self.locked[x]:
                    return False
                for y in self.g[x]:
                    if check_ancestor_locked(y, num):
                        return True
                return False

            def dfs(x: int) -> bool:
                res = False
                if self.locked[x]:
                    self.locked[x] = 0
                    res = True
                for y in self.g[x]:
                    if dfs(y):
                        res = True
                return res

            if self.locked[num]:
                return False
            x = num
            while x != -1:
                if self.locked[x]:
                    return False
                x = self.parent[x]
            #  if not check_ancestor_locked(0, num):
            #     return False
            if not dfs(num):
                return False
            self.locked[num] = user
            return True

    # 1443. 收集树上所有苹果的最少时间 (Minimum Time to Collect All Apples in a Tree)
    def minTime(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        def dfs(x: int, fa: int) -> bool:
            f = hasApple[x]
            for y in g[x]:
                if y != fa:
                    if dfs(y, x):
                        nonlocal res
                        res += 2
                        f = True
            return f

        res = 0
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        dfs(0, -1)
        return res

    # 435. 无重叠区间 (Non-overlapping Intervals)
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == max_val:
                return 0
            res = dfs(i + 1)
            for y in dic[i]:
                res = max(res, dfs(y) + 1)
            return res

        dic = collections.defaultdict(list)
        min_val = inf
        max_val = -inf
        for a, b in intervals:
            min_val = min(a, min_val)
            max_val = max(b, max_val)
            dic[a].append(b)
        return len(intervals) - dfs(min_val)

    # 2580. 统计将重叠区间合并成组的方案数 (Count Ways to Group Overlapping Ranges)
    def countWays(self, ranges: List[List[int]]) -> int:
        MOD = 10**9 + 7
        res = 1
        ranges.sort()
        n = len(ranges)
        i = 0
        while i < n:
            right = ranges[i][1]
            j = i
            while j < n and ranges[j][0] <= right:
                right = max(right, ranges[j][1])
                j += 1
            res <<= 1
            res %= MOD
            i = j
        return res

    # 1031. 两个非重叠子数组的最大和 (Maximum Sum of Two Non-Overlapping Subarrays)
    def maxSumTwoNoOverlap(self, nums: List[int], firstLen: int, secondLen: int) -> int:
        def check(nums: List[int], n1: int, n2: int) -> int:
            n = len(nums)
            res = 0
            sum1 = 0
            max_sum1 = 0
            sum2 = 0
            for i in range(n1):
                sum1 += nums[i]
            max_sum1 = sum1
            for i in range(n1, n1 + n2):
                sum2 += nums[i]
            res = sum1 + sum2
            for i in range(n1 + n2, n):
                sum2 += nums[i]
                sum2 -= nums[i - n2]
                sum1 += nums[i - n2]
                sum1 -= nums[i - n1 - n2]
                max_sum1 = max(max_sum1, sum1)
                res = max(res, max_sum1 + sum2)
            return res

        return max(check(nums, firstLen, secondLen), check(nums, secondLen, firstLen))

    # 689. 三个无重叠子数组的最大和 (Maximum Sum of 3 Non-Overlapping Subarrays)
    def maxSumOfThreeSubarrays(self, nums: List[int], k: int) -> List[int]:
        def check(nums1: List[int], nums2: List[int]) -> bool:
            for a, b in zip(nums1, nums2):
                if a != b:
                    if b < a:
                        return True
                    return False
            return False

        n = len(nums)
        pre = [0] * (n + 1)
        for i, x in enumerate(nums):
            pre[i + 1] = pre[i] + x
        left = [[0] * 2 for _ in range(n)]
        s = 0
        for i in range(k):
            s += nums[i]
        left[k - 1] = [s, 0]
        for i in range(k, n):
            s += nums[i]
            s -= nums[i - k]
            if s > left[i - 1][0]:
                left[i] = [s, i - k + 1]
            else:
                left[i] = left[i - 1]
        right = [[0] * 2 for _ in range(n)]
        s = 0
        for i in range(n - 1, n - k - 1, -1):
            s += nums[i]
        right[n - k] = [s, n - k]
        for i in range(n - k - 1, -1, -1):
            s += nums[i]
            s -= nums[i + k]
            if s >= right[i + 1][0]:
                right[i] = [s, i]
            else:
                right[i] = right[i + 1]
        max_sum = 0
        res = [n, n, n]
        for i in range(k, n - k * 2 + 1):
            cur_sum = left[i - 1][0] + pre[i + k] - pre[i] + right[i + k][0]
            if cur_sum > max_sum:
                res = [left[i - 1][1], i, right[i + k][1]]
                max_sum = cur_sum
            elif cur_sum == max_sum:
                if check(res, [left[i - 1][1], i, right[i + k][1]]):
                    res = left[i - 1][1], i, right[i + k][1]
        return res

    # 2864. 最大二进制奇数 (Maximum Odd Binary Number)
    def maximumOddBinaryNumber(self, s: str) -> str:
        cnt1 = s.count("1")
        return "1" * (cnt1 - 1) + "0" * (len(s) - cnt1) + "1"

    # 2865. 美丽塔 I (Beautiful Towers I)
    # 2866. 美丽塔 II (Beautiful Towers II)
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        right = [0] * (n + 1)
        st = [n]
        sum_val = 0
        for i in range(n - 1, -1, -1):
            while len(st) > 1 and maxHeights[i] <= maxHeights[st[-1]]:
                j = st.pop()
                sum_val -= maxHeights[j] * (st[-1] - j)
            sum_val += maxHeights[i] * (st[-1] - i)
            right[i] = sum_val
            st.append(i)
        res = sum_val
        left = [0] * n
        st = [-1]
        sum_val = 0
        for i in range(n):
            while len(st) > 1 and maxHeights[i] <= maxHeights[st[-1]]:
                j = st.pop()
                sum_val -= maxHeights[j] * (j - st[-1])
            sum_val += maxHeights[i] * (i - st[-1])
            left[i] = sum_val
            st.append(i)
        for i in range(n):
            res = max(res, left[i] + right[i + 1])
        return res

    # 257. 二叉树的所有路径 (Binary Tree Paths)
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        def dfs(root: Optional[TreeNode]) -> None:
            nonlocal s
            if root.left is None and root.right is None:
                res.append(s + str(root.val))
                return
            if root.left is not None:
                s += str(root.val) + "->"
                dfs(root.left)
                s = s[: len(s) - len(str(root.val) + "->")]
            if root.right is not None:
                s += str(root.val) + "->"
                dfs(root.right)
                s = s[: len(s) - len(str(root.val) + "->")]

        res = []
        s = ""
        dfs(root)
        return res

    # 1373. 二叉搜索子树的最大键值和 (Maximum Sum BST in Binary Tree)
    def maxSumBST(self, root: Optional[TreeNode]) -> int:
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        def dfs(root: Optional[TreeNode]) -> [int, int, int]:
            nonlocal res
            if root is None:
                return [None, None, 0]
            left = dfs(root.left)
            right = dfs(root.right)
            if (
                left[1] is not None
                and left[1] >= root.val
                or right[0] is not None
                and right[0] <= root.val
            ):
                return [inf, -inf, -inf]
            res = max(res, left[2] + right[2] + root.val)
            min_val = root.val
            max_val = root.val
            sum_val = left[2] + right[2] + root.val
            if left[0] is not None:
                min_val = min(min_val, left[0])
            if right[1] is not None:
                max_val = max(max_val, right[1])
            return [min_val, max_val, sum_val]

        res = 0
        dfs(root)
        return res

    # 1664. 生成平衡数组的方案数 (Ways to Make a Fair Array)
    def waysToMakeFair(self, nums: List[int]) -> int:
        n = len(nums)
        even = sum(nums[i] for i in range(0, n, 2))
        odd = sum(nums) - even
        res = 0
        curEven = 0
        curOdd = 0
        for i in range(n):
            if i % 2 == 0:
                if curEven + odd == curOdd + even - nums[i]:
                    res += 1
                curEven += nums[i]
                even -= nums[i]
            else:
                if curOdd + even == curEven + odd - nums[i]:
                    res += 1
                curOdd += nums[i]
                odd -= nums[i]
        return res

    # 1749. 任意子数组和的绝对值的最大值 (Maximum Absolute Sum of Any Subarray)
    def maxAbsoluteSum(self, nums: List[int]) -> int:
        pre_min = 0
        pre_max = 0
        pre = 0
        res = 0
        for num in nums:
            pre += num
            res = max(res, abs(pre - pre_min), abs(pre - pre_max))
            pre_min = min(pre_min, pre)
            pre_max = max(pre_max, pre)
        return res

    # 1774. 最接近目标价格的甜点成本 (Closest Dessert Cost)
    def closestCost(
        self, baseCosts: List[int], toppingCosts: List[int], target: int
    ) -> int:
        def dfs(i: int, j: int) -> None:
            if i == len(toppingCosts):
                s.add(j)
                return
            for k in range(3):
                dfs(i + 1, toppingCosts[i] * k + j)

        s = set()
        dfs(0, 0)
        diff = inf
        res = inf
        for base in baseCosts:
            for top in s:
                _s = top + base
                if abs(_s - target) < diff:
                    res = _s
                    diff = abs(_s - target)
                elif abs(_s - target) == diff:
                    res = min(res, _s)
                if res == 0:
                    break
        return res

    # 2582. 递枕头 (Pass the Pillow)
    def passThePillow(self, n: int, time: int) -> int:
        time %= (n - 1) * 2
        if time <= n - 1:
            return time + 1
        return n * 2 - (time + 1)

    # 343. 整数拆分 (Integer Break)
    # LCR 132. 砍竹子 II
    def cuttingBamboo(self, bamboo_len: int) -> int:
        MOD = 10**9 + 7

        @cache
        def dfs(i: int) -> int:
            if i <= 2:
                return 1
            res = i
            for j in range(1, i):
                res = max(res, dfs(i - j) * j)
            return res

        if bamboo_len == 3:
            return 2
        return dfs(bamboo_len) % MOD

    # 258. 各位相加 (Add Digits)
    def addDigits(self, num: int) -> int:
        while len(str(num)) > 1:
            s = 0
            while num:
                s += num % 10
                num //= 10
            num = s
        return num

    # 260. 只出现一次的数字 III (Single Number III)
    def singleNumber(self, nums: List[int]) -> List[int]:
        xor = 0
        for num in nums:
            xor ^= num
        bit = xor & -xor
        x = 0
        y = 0
        for num in nums:
            if num & bit:
                x ^= num
            else:
                y ^= num
        return [x, y]

    # 268. 丢失的数字 (Missing Number)
    def missingNumber(self, nums: List[int]) -> int:
        s = sum(nums)
        n = len(nums)
        return n * (n + 1) // 2 - s

    # 268. 丢失的数字 (Missing Number)
    def missingNumber(self, nums: List[int]) -> int:
        res = len(nums)
        for i, x in enumerate(nums):
            res ^= i ^ x
        return res

    # 1333. 餐厅过滤器 (Filter Restaurants by Vegan-Friendly, Price and Distance)
    def filterRestaurants(
        self,
        restaurants: List[List[int]],
        veganFriendly: int,
        maxPrice: int,
        maxDistance: int,
    ) -> List[int]:
        restaurants.sort(key=lambda k: (-k[1], -k[0]))
        res = []
        for id, _, v, p, d in restaurants:
            if (
                (veganFriendly and v or not veganFriendly)
                and p <= maxPrice
                and d <= maxDistance
            ):
                res.append(id)
        return res

    # 1671. 得到山形数组的最少删除次数 (Minimum Number of Removals to Make Mountain Array)
    def minimumMountainRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        left = [0] * n
        _list = []
        for i, v in enumerate(nums):
            if len(_list) == 0 or _list[-1] < v:
                _list.append(v)
                left[i] = len(_list)
            else:
                j = bisect.bisect_left(_list, v)
                _list[j] = v
                left[i] = j + 1
        right = [0] * n
        _list.clear()
        for i in range(n - 1, -1, -1):
            if len(_list) == 0 or _list[-1] < nums[i]:
                _list.append(nums[i])
                right[i] = len(_list)
            else:
                j = bisect.bisect_left(_list, nums[i])
                _list[j] = nums[i]
                right[i] = j + 1
        res = n
        for i in range(n):
            if left[i] != 1 and right[i] != 1:
                res = min(res, n - left[i] - right[i] + 1)
        return res

    # 1639. 通过给定词典构造目标字符串的方案数 (Number of Ways to Form a Target String Given a Dictionary)
    def numWays(self, words: List[str], target: str) -> int:
        MOD = 10**9 + 7

        @cache
        def dfs(i: int, j: int) -> int:
            if j < 0:
                return 1
            if i < j:
                return 0
            return (
                dfs(i - 1, j) + cnts[i][ord(target[j]) - ord("a")] * dfs(i - 1, j - 1)
            ) % MOD

        n = len(words[0])
        cnts = [[0] * 26 for _ in range(n)]
        for w in words:
            for i, c in enumerate(w):
                cnts[i][ord(c) - ord("a")] += 1
        return dfs(n - 1, len(target) - 1)

    # 2146. 价格范围内最高排名的 K 样物品 (K Highest Ranked Items Within a Price Range)
    def highestRankedKItems(
        self, grid: List[List[int]], pricing: List[int], start: List[int], k: int
    ) -> List[List[int]]:
        m = len(grid)
        n = len(grid[0])
        _list = []
        q = [[start[0], start[1], 0]]
        dirs = [[0, -1], [1, 0], [-1, 0], [0, 1]]
        while q:
            [x, y, d] = q.pop(0)
            if grid[x][y] == 0:
                continue
            if pricing[1] >= grid[x][y] >= pricing[0]:
                _list.append([d, grid[x][y], x, y])
            grid[x][y] = 0
            for dx, dy in dirs:
                nx = x + dx
                ny = y + dy
                if nx >= 0 and nx < m and ny >= 0 and ny < n and grid[nx][ny] != 0:
                    q.append([nx, ny, d + 1])
        _list.sort(key=lambda o: (o[0], o[1], o[2], o[3]))
        res = []
        for i in range(min(len(_list), k)):
            res.append([_list[i][2], _list[i][3]])
        return res

    # 404. 左叶子之和 (Sum of Left Leaves)
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        def dfs(root: Optional[TreeNode]) -> None:
            if root is None:
                return
            if (
                root
                and root.left
                and root.left.left is None
                and root.left.right is None
            ):
                nonlocal res
                res += root.left.val
            dfs(root.left)
            dfs(root.right)

        res = 0
        dfs(root)
        return res

    # 834. 树中距离之和 (Sum of Distances in Tree)
    def sumOfDistancesInTree(self, n: int, edges: List[List[int]]) -> List[int]:
        def dfs(x: int, fa: int, d: int) -> None:
            res[0] += d
            for y in g[x]:
                if y != fa:
                    dfs(y, x, d + 1)
                    size[x] += size[y]

        def reroot(x: int, fa: int) -> None:
            for y in g[x]:
                if y != fa:
                    res[y] = res[x] + n - 2 * size[y]
                    reroot(y, x)

        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        size = [1] * n
        res = [0] * n
        dfs(0, -1, 0)
        reroot(0, -1)
        return res

    # 2049. 统计最高分的节点数目 (Count Nodes With the Highest Score)
    def countHighestScoreNodes(self, parents: List[int]) -> int:
        def dfs(x: int) -> int:
            s = 0
            cur = 1
            for y in g[x]:
                cnt = dfs(y)
                cur *= cnt
                s += cnt
            # 考虑根节点的情况
            if n - s - 1 != 0:
                cur *= n - s - 1
            nonlocal res, max_score
            if cur > max_score:
                max_score = cur
                res = 1
            elif cur == max_score:
                res += 1
            return s + 1

        n = len(parents)
        g = [[] for _ in range(n)]
        for i in range(1, n):
            g[parents[i]].append(i)
        res = 0
        max_score = 0
        dfs(0)
        return res

    # 110. 平衡二叉树 (Balanced Binary Tree)
    # 面试题 04.04. 检查平衡性 (Check Balance LCCI)
    # LCR 176. 判断是否为平衡二叉树
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        def dfs(root: Optional[TreeNode]) -> int:
            nonlocal res
            if root is None or not res:
                return 0
            left = dfs(root.left)
            right = dfs(root.right)
            if abs(left - right) > 1:
                res = False
            return max(left, right) + 1

        res = True
        dfs(root)
        return res

    # 107. 二叉树的层序遍历 II (Binary Tree Level Order Traversal II)
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        res = []
        q = collections.deque([root])
        while q:
            size = len(q)
            _list = list()
            for _ in range(size):
                x = q.popleft()
                _list.append(x.val)
                if x.left:
                    q.append(x.left)
                if x.right:
                    q.append(x.right)
            res.append(_list)
        return res[::-1]

    # 107. 二叉树的层序遍历 II (Binary Tree Level Order Traversal II)
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        def dfs(root: "TreeNode", d: int) -> None:
            if root is None:
                return
            if d == len(res):
                res.append([])
            res[d].append(root.val)
            dfs(root.left, d + 1)
            dfs(root.right, d + 1)

        res = []
        dfs(root, 0)
        return res[::-1]

    # 764. 最大加号标志 (Largest Plus Sign)
    def orderOfLargestPlusSign(self, n: int, mines: List[List[int]]) -> int:
        left = [[0] * n for _ in range(n)]
        up = [[0] * n for _ in range(n)]
        s = set()
        for x, y in mines:
            s.add((x, y))
        for i in range(n):
            for j in range(n):
                if j == 0:
                    left[i][j] = 1 if (i, j) not in s else 0
                else:
                    left[i][j] = 0 if (i, j) in s else left[i][j - 1] + 1
                if i == 0:
                    up[i][j] = 1 if (i, j) not in s else 0
                else:
                    up[i][j] = 0 if (i, j) in s else up[i - 1][j] + 1
        res = 0
        down = [0] * n
        for i in range(n - 1, -1, -1):
            right = 0
            for j in range(n - 1, -1, -1):
                if j == n - 1:
                    right = 1 if (i, j) not in s else 0
                else:
                    right = 0 if (i, j) in s else right + 1
                if i == n - 1:
                    down[j] = 1 if (i, j) not in s else 0
                else:
                    down[j] = 0 if (i, j) in s else down[j] + 1
                res = max(res, min(min(left[i][j], right), min(up[i][j], down[j])))
        return res

    # 605. 种花问题 (Can Place Flowers)
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        flowerbed = [0] + flowerbed + [0]
        for i in range(1, len(flowerbed) - 1):
            if flowerbed[i - 1] == 0 and flowerbed[i] == 0 and flowerbed[i + 1] == 0:
                flowerbed[i] = 1
                n -= 1
        return n <= 0

    # 1959. K 次调整数组大小浪费的最小总空间 (Minimum Total Space Wasted With K Resizing Operations)
    def minSpaceWastedKResizing(self, nums: List[int], k: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            if j == k:
                return arr[i][n - 1]
            return min(dfs(x + 1, j + 1) + arr[i][x] for x in range(i, n - k + j + 1))

        n = len(nums)
        arr = [[0] * n for _ in range(n)]
        s = 0
        for i in range(n):
            s += nums[i]
            m = nums[i]
            for j in range(i, n):
                m = max(m, nums[j])
                arr[i][j] = m * (j - i + 1)
        return dfs(0, 0) - s

    # 1981. 最小化目标值与所选元素的差 (Minimize the Difference Between Target and Chosen Elements)
    def minimizeTheDifference(self, mat: List[List[int]], target: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == m:
                return abs(j)
            if j <= 0:
                return abs(j) + suf[i]
            return min(dfs(i + 1, j - x) for x in mat[i])

        m = len(mat)
        suf = [min(mat[i]) for i in range(m)]
        for i in range(m - 2, -1, -1):
            suf[i] += suf[i + 1]
        return dfs(0, target)

    # 473. 火柴拼正方形 (Matchsticks to Square)
    def makesquare(self, matchsticks: List[int]) -> bool:
        @cache
        def dfs(i: int) -> bool:
            if i == u:
                return True
            c = candidate = i ^ u
            while c:
                if arr[c] == side and dfs(i | c):
                    return True
                c = (c - 1) & candidate
            return False

        s = sum(matchsticks)
        if s % 4 != 0:
            return False
        side = s // 4
        n = len(matchsticks)
        u = (1 << n) - 1
        arr = [0] * (1 << n)
        for i in range(1, 1 << n):
            bit = (i & -i).bit_length() - 1
            arr[i] = arr[i ^ (1 << bit)] + matchsticks[bit]
        return dfs(0)

    # 526. 优美的排列 (Beautiful Arrangement)
    def countArrangement(self, n: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 1
            res = 0
            c = u ^ j
            while c:
                lb = (c & -c).bit_length() - 1
                if (i + 1) % (lb + 1) == 0 or (lb + 1) % (i + 1) == 0:
                    res += dfs(i + 1, j | (1 << lb))
                c &= c - 1
            return res

        u = (1 << n) - 1
        return dfs(0, 0)

    # 8038. 收集元素的最少操作次数 (Minimum Operations to Collect Elements)
    def minOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        m = 0
        u = (1 << k) - 1
        for i in range(n - 1, -1, -1):
            m |= 1 << (nums[i] - 1)
            if (m & u) == u:
                return n - i
        return 0

    # 100032. 使数组为空的最少操作次数 (Minimum Number of Operations to Make Array Empty)
    def minOperations(self, nums: List[int]) -> int:
        dic = collections.defaultdict(int)
        for num in nums:
            dic[num] += 1
        res = 0
        for num in dic.values():
            if num == 1:
                return -1
            res += (num - 1) // 3 + 1
        return res

    # 100019. 将数组分割成最多数目的子数组 (Split Array Into Maximum Number of Subarrays)
    def maxSubarrays(self, nums: List[int]) -> int:
        res = 0
        cur = -1
        for num in nums:
            cur &= num
            if cur == 0:
                cur = -1
                res += 1
        return max(1, res)

    # 8051. 可以被 K 整除连通块的最大数目 (Maximum Number of K-Divisible Components)
    def maxKDivisibleComponents(
        self, n: int, edges: List[List[int]], values: List[int], k: int
    ) -> int:
        def dfs(x: int, fa: int) -> int:
            s = values[x]
            for y in g[x]:
                if y != fa:
                    s += dfs(y, x)
            if s % k == 0:
                nonlocal res
                res += 1
            return s % k

        g = [[] for _ in range(n)]
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
        res = 0
        dfs(0, -1)
        return res

    # 2873. 有序三元组中的最大值 I (Maximum Value of an Ordered Triplet I)
    # 2874. 有序三元组中的最大值 II (Maximum Value of an Ordered Triplet II)
    def maximumTripletValue(self, nums: List[int]) -> int:
        res = pre_max = diff = 0
        for x in nums:
            res = max(res, diff * x)
            diff = max(diff, pre_max - x)
            pre_max = max(pre_max, x)
        return res

    # 2875. 无限数组的最短子数组 (Minimum Size Subarray in Infinite Array)
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        _s = sum(nums)
        if target % _s == 0:
            return n * (target // _s)
        cnt = n * (target // _s)
        target %= _s
        arr = [0] * (n * 2)
        for i in range(n):
            arr[i] = nums[i]
            arr[i + n] = nums[i]
        i = 0
        j = 0
        cur = 0
        d = n + 1
        while j < n * 2:
            cur += arr[j]
            while cur > target:
                cur -= arr[i]
                i += 1
            if cur == target:
                d = min(d, j - i + 1)
            j += 1
        if d == n + 1:
            return -1
        return d + cnt

    # 2876. 有向图访问计数 (Count Visited Nodes in a Directed Graph)
    def countVisitedNodes(self, edges: List[int]) -> List[int]:
        n = len(edges)
        deg = [0] * n
        g = [-1] * n
        for i, v in enumerate(edges):
            g[i] = v
            deg[v] += 1
        q = collections.deque()
        for i in range(n):
            if deg[i] == 0:
                q.append(i)
        while q:
            x = q.popleft()
            y = g[x]
            deg[y] -= 1
            if deg[y] == 0:
                q.append(y)
        res = [0] * n
        for i in range(n):
            if deg[i] != 0 and res[i] == 0:
                j = i
                cnt = 1
                while g[j] != i:
                    cnt += 1
                    j = g[j]
                j = i
                res[i] = cnt
                while g[j] != i:
                    res[j] = cnt
                    j = g[j]
        rg = [[] for _ in range(n)]
        for i, v in enumerate(edges):
            if deg[i] == 0 or deg[v] == 0:
                rg[v].append(i)

        def rdfs(x: int, d: int) -> None:
            res[x] = d
            for y in rg[x]:
                rdfs(y, d + 1)

        for i in range(n):
            if deg[i] != 0 and res[i] != 0:
                rdfs(i, res[i])
        return res

    # 2719. 统计整数数目 (Count of Integers)
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        def cal(s: str) -> int:
            @cache
            def dfs(i: int, cur_sum: int, is_limit: bool, is_num) -> int:
                if i == n:
                    return is_num and cur_sum >= min_sum
                res = 0
                if not is_num:
                    res = dfs(i + 1, cur_sum, False, False)
                up = int(s[i]) if is_limit else 9
                for j in range(0 if is_num else 1, up + 1):
                    if j + cur_sum > max_sum:
                        break
                    res += dfs(i + 1, cur_sum + j, is_limit and j == up, True)
                return res % MOD

            n = len(s)
            return dfs(0, 0, True, False)

        MOD = 10**9 + 7
        return (cal(num2) - cal(str(int(num1) - 1))) % MOD

    # 1955. 统计特殊子序列的数目 (Count Number of Special Subsequences)
    def countSpecialSubsequences(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return int(j == 3)
            if j == 0:
                if nums[i] == 0:
                    return (dfs(i + 1, 0) + dfs(i + 1, 1)) % MOD
                return dfs(i + 1, 0)
            if j == 1:
                if nums[i] == 0:
                    return dfs(i + 1, 1) * 2 % MOD
                if nums[i] == 1:
                    return (dfs(i + 1, 1) + dfs(i + 1, 2)) % MOD
                return dfs(i + 1, 1)
            if j == 2:
                if nums[i] == 0:
                    return dfs(i + 1, 2)
                if nums[i] == 1:
                    return dfs(i + 1, 2) * 2 % MOD
                return (dfs(i + 1, 2) + dfs(i + 1, 3)) % MOD
            if j == 3:
                if nums[i] == 2:
                    return dfs(i + 1, 3) * 2 % MOD
                return dfs(i + 1, 3)

        n = len(nums)
        MOD = 10**9 + 7
        res = dfs(0, 0)
        dfs.cache_clear()
        return res

    # 2572. 无平方子集计数 (Count the Number of Square-Free Subsets)
    def squareFreeSubsets(self, nums: List[int]) -> int:
        MOD = 10**9 + 7

        def check(x: int) -> int:
            m = 0
            for i in range(2, isqrt(x) + 1):
                while x % i == 0:
                    if m >> i & 1:
                        return 0
                    m |= 1 << i
                    x //= i
            if x > 1:
                m |= 1 << x
            return m

        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 1
            res = dfs(i + 1, j)
            if j & dic[keys[i]] == 0:
                res += dfs(i + 1, j | dic[keys[i]]) * cnts[keys[i]] % MOD
            return res % MOD

        dic = defaultdict(int)
        for i in range(2, 31):
            x = check(i)
            if x:
                dic[i] = x
        cnts = defaultdict(int)
        cnt1 = 0
        for x in nums:
            if x == 1:
                cnt1 += 1
            elif x in dic:
                cnts[x] += 1
        keys = list(cnts.keys())
        n = len(keys)
        return (dfs(0, 0) * (pow(2, cnt1, MOD)) - 1) % MOD

    # 2035. 将数组分成两个数组并最小化数组和的差 (Partition Array Into Two Arrays to Minimize Sum Difference)
    def minimumDifference(self, nums: List[int]) -> int:
        n = len(nums) // 2
        dic = collections.defaultdict(list)
        for i in range(1 << n):
            cnt = 0
            _sum = 0
            for j in range(n):
                if ((i >> j) & 1) == 0:
                    cnt += 1
                    _sum += nums[j]
                else:
                    _sum -= nums[j]
            dic[cnt].append(_sum)
        for c in dic.values():
            c.sort()
        res = inf
        for i in range(1 << n):
            cnt = 0
            _sum = 0
            for j in range(n):
                if ((i >> j) & 1) == 0:
                    cnt += 1
                    _sum += nums[n + j]
                else:
                    _sum -= nums[n + j]
            _list = dic[n - cnt]
            left = bisect.bisect_left(_list, -_sum)
            if left != len(_list):
                res = min(res, abs(_list[left] + _sum))
            right = bisect.bisect_right(_list, -_sum)
            if right != len(_list):
                res = min(res, abs(_list[right] + _sum))
            if res == 0:
                break
        return res

    # 1994. 好子集的数目 (The Number of Good Subsets)
    def numberOfGoodSubsets(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == len(arr):
                return 1
            res = dfs(i + 1, j)
            if j & dic[arr[i]] == 0:
                res += dfs(i + 1, j | dic[arr[i]]) * cnts[arr[i]] % MOD
            return res % MOD

        def check(x: int) -> int:
            m = 0
            for i in range(2, isqrt(x) + 1):
                while x % i == 0:
                    if m >> i & 1 == 1:
                        return 0
                    m |= 1 << i
                    x //= i
            if x > 1:
                m |= 1 << x
            return m

        dic = defaultdict(int)
        for i in range(2, 31):
            x = check(i)
            if x:
                dic[i] = x
        cnts = defaultdict(int)
        cnt1 = 0
        for x in nums:
            if x == 1:
                cnt1 += 1
            elif x in dic:
                cnts[x] += 1
        arr = list(cnts.keys())
        MOD = 10**9 + 7
        return (dfs(0, 0) - 1) * pow(2, cnt1, MOD) % MOD

    # 101. 对称二叉树 (Symmetric Tree)
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        def dfs(L: Optional[TreeNode], R: Optional[TreeNode]) -> bool:
            if L is None and R is None:
                return True
            if L is None or R is None or L.val != R.val:
                return False
            return dfs(L.left, R.right) and dfs(L.right, R.left)

        return dfs(root.left, root.right)

    # 698. 划分为k个相等的子集 (Partition to K Equal Sum Subsets)
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        @cache
        def dfs(i: int, j: int) -> bool:
            if i == k:
                return True
            sub = c = u ^ j
            while sub:
                if s[sub] == p and dfs(i + 1, sub | j):
                    return True
                sub = (sub - 1) & c
            return False

        n = len(nums)
        s = sum(nums)
        if k == 1:
            return True
        if s % k != 0 or max(nums) > s // k:
            return False
        p = s // k
        s = [0] * (1 << n)
        u = (1 << n) - 1
        for i in range(1, 1 << n):
            s[i] = s[i & (i - 1)] + nums[(i & -i).bit_length() - 1]
        return dfs(0, 0)

    # 691. 贴纸拼词 (Stickers to Spell Word)
    def minStickers(self, stickers: List[str], target: str) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == u:
                return 0
            res = m + 1
            for s in stickers:
                cnt = [0] * 26
                for c in s:
                    cnt[ord(c) - ord("a")] += 1
                c = 0
                for j in range(m):
                    if ((i >> j) & 1) == 0 and cnt[ord(target[j]) - ord("a")] > 0:
                        cnt[ord(target[j]) - ord("a")] -= 1
                        c |= 1 << j
                if c:
                    res = min(res, dfs(i | c) + 1)
            return res

        m = len(target)
        u = (1 << m) - 1
        res = dfs(0)
        return res if res < m + 1 else -1

    # 464. 我能赢吗 (Can I Win)
    def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
        @cache
        def dfs(i: int, j: int) -> bool:
            c = i ^ u
            while c:
                index = (c & -c).bit_length() - 1
                if j + index + 1 >= desiredTotal:
                    return True
                if not dfs(i | (1 << index), j + index + 1):
                    return True
                c &= c - 1
            return False

        if (1 + maxChoosableInteger) * maxChoosableInteger < desiredTotal:
            return False
        u = (1 << maxChoosableInteger) - 1
        return dfs(0, 0)

    # 805. 数组的均值分割 (Split Array With Same Average) --折半搜索
    def splitArraySameAverage(self, nums: List[int]) -> bool:
        n = len(nums)
        s = sum(nums)
        dic = collections.defaultdict(set)
        dic[0].add(0)
        half_len = n // 2
        arr = [0] * (1 << half_len)
        for i in range(1, 1 << half_len):
            index = (i & -i).bit_length() - 1
            arr[i] = arr[i ^ (1 << index)] + nums[index]
            dic[i.bit_count()].add(arr[i])
        arr = [0] * (1 << (n - half_len))
        for i in range(1, 1 << (n - half_len)):
            index = (i & -i).bit_length() - 1
            arr[i] = arr[i ^ (1 << index)] + nums[index + half_len]
            cnt = i.bit_count()
            for j in range(half_len + 1):
                if (
                    j + cnt < n
                    and (j * s + cnt * s - n * arr[i]) % n == 0
                    and ((j * s + cnt * s - n * arr[i]) // n) in dic[j]
                ):
                    return True
        return False

    # 901. 股票价格跨度 (Online Stock Span)
    class StockSpanner:

        def __init__(self):
            self.st = [[-1, 0]]
            self.i = 0

        def next(self, price: int) -> int:
            while len(self.st) > 1 and self.st[-1][-1] <= price:
                self.st.pop()
            res = self.i - self.st[-1][0]
            self.st.append([self.i, price])
            self.i += 1
            return res

    # 1947. 最大兼容性评分和 (Maximum Compatibility Score Sum)
    def maxCompatibilitySum(
        self, students: List[List[int]], mentors: List[List[int]]
    ) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            c = j ^ u
            res = 0
            while c:
                index = (c & -c).bit_length() - 1
                res = max(
                    res,
                    dfs(i + 1, j | (1 << index))
                    + (stu_mask[i] ^ men_mask[index] ^ m).bit_count(),
                )
                c &= c - 1
            return res

        n = len(students)
        stu_mask = [0] * n
        men_mask = [0] * n
        for i in range(n):
            stu = 0
            men = 0
            for s, m in zip(students[i], mentors[i]):
                stu = (stu << 1) | s
                men = (men << 1) | m
            stu_mask[i] = stu
            men_mask[i] = men
        m = (1 << len(students[0])) - 1
        u = (1 << n) - 1
        return dfs(0, 0)

    # 2578. 最小和分割 (Split With Minimum Sum)
    def splitNum(self, num: int) -> int:
        cnt = [0] * 10
        while num:
            cnt[num % 10] += 1
            num //= 10
        res = [0] * 2
        i = 0
        for j in range(10):
            while cnt[j]:
                res[i] = res[i] * 10 + j
                cnt[j] -= 1
                i ^= 1
        return sum(res)

    # 2894. 分类求和并作差 (Divisible and Non-divisible Sums Difference)
    def differenceOfSums(self, n: int, m: int) -> int:
        return (1 + n) * n // 2 - m * (1 + n // m) * (n // m)

    # 2895. 最小处理时间 (Minimum Processing Time)
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        processorTime.sort(reverse=True)
        tasks.sort()
        res = 0
        for i, v in enumerate(tasks):
            res = max(res, v + processorTime[i // 4])
        return res

    # 2896. 执行操作使两个字符串相等 (Apply Operations to Make Two Strings Equal)
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == n:
                return 0
            if i == n - 1:
                return x
            return min(dfs(i + 1) + x, dfs(i + 2) + (p[i + 1] - p[i]) * 2)

        p = []
        for i, c1, c2 in zip(range(len(s1)), s1, s2):
            if c1 != c2:
                p.append(i)
        n = len(p)
        if n == 0:
            return 0
        if n % 2 == 1:
            return -1
        return dfs(0) // 2

    # 2896. 执行操作使两个字符串相等 (Apply Operations to Make Two Strings Equal)
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if n - i == j:
                return 0
            res = dfs(i + 1, j + 1) + x
            if i < n - 1:
                res = min(res, dfs(i + 2, j) + arr[i + 1] - arr[i])
            if j:
                res = min(res, dfs(i + 1, j - 1))
            return res

        arr = [i for i, x, y in zip(range(len(s1)), s1, s2) if x != y]
        n = len(arr)
        if len(arr) % 2:
            return -1
        return dfs(0, 0)

    # 2897. 对数组执行操作使平方和最大 (Apply Operations on Array to Maximize Sum of Squares)
    def maxSum(self, nums: List[int], k: int) -> int:
        cnt = [0] * 31
        for num in nums:
            for i in range(31):
                cnt[i] += (num >> i) & 1
        res = 0
        MOD = 10**9 + 7
        while k > 0:
            k -= 1
            x = 0
            for i in range(31):
                if cnt[i] > 0:
                    cnt[i] -= 1
                    x |= 1 << i
            res += x * x
            res %= MOD
        return res

    # 395. 至少有 K 个重复字符的最长子串 (Longest Substring with At Least K Repeating Characters)
    def longestSubstring(self, s: str, k: int) -> int:
        if len(s) < k:
            return 0
        for c in set(s):
            if s.count(c) < k:
                return max(self.longestSubstring(t, k) for t in s.split(c))
        return len(s)

    # 368. 最大整除子集 (Largest Divisible Subset)
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        nums.sort()
        n = len(nums)
        dp = [1] * n
        _max = 0
        max_val = 0
        for i in range(n):
            for j in range(i):
                if nums[i] % nums[j] == 0:
                    dp[i] = max(dp[i], dp[j] + 1)
            if dp[i] > _max:
                _max = dp[i]
                max_val = nums[i]
        res = []
        for i in range(n - 1, -1, -1):
            if _max == 0:
                break
            if dp[i] == _max and max_val % nums[i] == 0:
                res.append(nums[i])
                max_val = nums[i]
                _max -= 1
        return res

    # 368. 最大整除子集 (Largest Divisible Subset)
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        @cache
        def dfs(i: int) -> int:
            res = 0
            for j in range(i):
                if nums[i] % nums[j] == 0:
                    res = max(res, dfs(j))
            return res + 1

        def make_ans(i: int) -> None:
            res.append(nums[i])
            final_ans = dfs(i)
            for j in range(i):
                if nums[i] % nums[j] == 0 and dfs(j) + 1 == final_ans:
                    make_ans(j)
                    break

        n = len(nums)
        nums.sort()
        mx = 0
        f = 0
        for i in range(n):
            cur = dfs(i)
            if cur > mx:
                mx = cur
                f = i
        res = []
        make_ans(f)
        return res

    # 2731. 移动机器人 (Movement of Robots)
    def sumDistance(self, nums: List[int], s: str, d: int) -> int:
        n = len(nums)
        for i, x, dir in zip(range(n), nums, s):
            nums[i] += d if dir == "R" else -d
        nums.sort()
        res = 0
        pre = 0
        MOD = 10**9 + 7
        for i, x in enumerate(nums):
            res += x * i - pre
            pre += x
            res %= MOD
        return res

    # 2512. 奖励最顶尖的 K 名学生 (Reward Top K Students)
    def topStudents(
        self,
        positive_feedback: List[str],
        negative_feedback: List[str],
        report: List[str],
        student_id: List[int],
        k: int,
    ) -> List[int]:
        pos = set(positive_feedback)
        neg = set(negative_feedback)
        _list = []
        for r, id in zip(report, student_id):
            s = 0
            for w in r.split():
                if w in pos:
                    s += 3
                elif w in neg:
                    s -= 1
            _list.append([s, id])
        _list.sort(key=lambda k: (-k[0], k[1]))
        return [i for _, i in _list[:k]]

    # 115. 不同的子序列 (Distinct Subsequences)
    # LCR 097. 不同的子序列
    def numDistinct(self, s: str, t: str) -> int:
        MOD = 10**9 + 7

        @cache
        def dfs(i: int, j: int) -> int:
            if j == m:
                return 1
            if i == n or n - i < m - j:
                return 0
            res = dfs(i + 1, j)
            if s[i] == t[j]:
                res += dfs(i + 1, j + 1)
            return res % MOD

        n = len(s)
        m = len(t)
        return dfs(0, 0)

    # 132. 分割回文串 II (Palindrome Partitioning II)
    # LCR 094. 分割回文串 II
    def minCut(self, s: str) -> int:
        n = len(s)
        valid = [[False] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if (
                    j == i
                    or j - i == 1
                    and s[i] == s[j]
                    or j - i > 1
                    and s[i] == s[j]
                    and valid[i + 1][j - 1]
                ):
                    valid[i][j] = True

        @cache
        def dfs(i: int) -> int:
            if i == n:
                return 0
            if valid[i][n - 1]:
                return 1
            return min(dfs(j + 1) + 1 if valid[i][j] else inf for j in range(i, n))

        return dfs(0) - 1

    # 2562. 找出数组的串联值 (Find the Array Concatenation Value)
    def findTheArrayConcVal(self, nums: List[int]) -> int:
        n = len(nums)
        i = 0
        j = n - 1
        res = 0
        while i < j:
            res += int(str(nums[i]) + str(nums[j]))
            i += 1
            j -= 1
        if i == j:
            res += nums[i]
        return res

    # 10. 正则表达式匹配 (Regular Expression Matching)
    # LCR 137. 模糊搜索验证
    def isMatch(self, s: str, p: str) -> bool:
        @cache
        def dfs(i: int, j: int) -> bool:
            if j == m:
                return i == n
            match = i < n and (s[i] == p[j] or p[j] == ".")
            if j + 1 < m and p[j + 1] == "*":
                return dfs(i, j + 2) or match and dfs(i + 1, j)
            return match and dfs(i + 1, j + 1)

        n = len(s)
        m = len(p)
        return dfs(0, 0)

    # 面试题 17.16. 按摩师
    def massage(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int) -> int:
            if i >= n:
                return 0
            return max(dfs(i + 1), dfs(i + 2) + nums[i])

        n = len(nums)
        return dfs(0)

    # 926. 将字符串翻转到单调递增 (Flip String to Monotone Increasing)
    # LCR 092. 将字符串翻转到单调递增
    def minFlipsMonoIncr(self, s: str) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            if j == 0:
                return min(dfs(i + 1, int(s[i])), dfs(i + 1, 1 ^ int(s[i])) + 1)
            return dfs(i + 1, 1) + (int(s[i]) ^ 1)

        n = len(s)
        return dfs(0, 0)

    # 383. 赎金信 (Ransom Note)
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        cnt = [0] * 26
        for c in magazine:
            cnt[ord(c) - ord("a")] += 1
        for c in ransomNote:
            cnt[ord(c) - ord("a")] -= 1
            if cnt[ord(c) - ord("a")] < 0:
                return False
        return True

    # 1488. 避免洪水泛滥 (Avoid Flood in The City)
    def avoidFlood(self, rains: List[int]) -> List[int]:
        n = len(rains)
        st = SortedList()
        dic = {}
        res = [-1] * n
        for i, v in enumerate(rains):
            if v == 0:
                st.add(i)
            else:
                if v in dic:
                    id = st.bisect_right(dic[v])
                    if id == len(st):
                        return []
                    res[st[id]] = v
                    st.discard(st[id])
                dic[v] = i
        for i in range(n):
            if rains[i] == 0 and res[i] == -1:
                res[i] = 1
        return res

    # 137. 只出现一次的数字 II (Single Number II)
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for i in range(32):
            cnt = 0
            for num in nums:
                cnt += (num >> i) & 1
            if cnt % 3:
                if i == 31:
                    res -= 1 << i
                else:
                    res |= 1 << i
        return res

    # LCP 13. 寻宝 (就差最后一个用例)
    def minimalSteps(self, maze: List[str]) -> int:
        def min(a: int, b: int) -> int:
            return a if a < b else b

        def dfs(i: int, j: int) -> int:
            if i == u:
                return dis_t_to_m[j]
            if memo[i][j] != -1:
                return memo[i][j]
            # 起点
            if i == 0:
                res = inf
                for k in range(len(m_pos)):
                    min_dis = inf
                    for x in range(len(o_pos)):
                        min_dis = min(min_dis, dis_m_to_o[k][x] + o_pos[x][2])
                    res = min(res, dfs(1 << k, k) + min_dis)
                memo[i][j] = res
                return res
            c = i ^ u
            res = inf
            while c:
                index = (c & -c).bit_length() - 1
                min_dis = inf
                for x in range(len(o_pos)):
                    min_dis = min(min_dis, dis_m_to_o[index][x] + dis_m_to_o[j][x])
                res = min(res, dfs(i | (1 << index), index) + min_dis)
                c &= c - 1
            memo[i][j] = res
            return res

        m = len(maze)
        n = len(maze[0])
        m_cnt = 0
        s = []
        t = []
        for i in range(m):
            for j in range(n):
                if maze[i][j] == "M":
                    m_cnt += 1
                elif maze[i][j] == "S":
                    s = [i, j]
                elif maze[i][j] == "T":
                    t = [i, j]
        dirs = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        q = collections.deque()
        m_pos = []
        o_pos = []
        q.append((s[0], s[1]))
        vis = [[False] * n for _ in range(m)]
        vis[s[0]][s[1]] = True
        level = 0
        start_to_target = 0
        while q:
            level += 1
            size = len(q)
            for _ in range(size):
                cur = q.popleft()
                x = cur[0]
                y = cur[1]
                for dx, dy in dirs:
                    nx = x + dx
                    ny = y + dy
                    if (
                        m > nx >= 0
                        and n > ny >= 0
                        and maze[nx][ny] != "#"
                        and not vis[nx][ny]
                    ):
                        vis[nx][ny] = True
                        if maze[nx][ny] == "M":
                            m_pos.append((nx, ny, level))
                        elif maze[nx][ny] == "O":
                            o_pos.append((nx, ny, level))
                        elif maze[nx][ny] == "T":
                            start_to_target = level
                        q.append((nx, ny))
        if not vis[t[0]][t[1]]:
            return -1
        if m_cnt != len(m_pos):
            return -1
        if m_cnt == 0:
            return start_to_target
        if len(o_pos) == 0:
            return -1
        m_to_index = collections.defaultdict(int)
        o_to_index = collections.defaultdict(int)
        for i in range(len(m_pos)):
            m_to_index[(m_pos[i][0], m_pos[i][1])] = i
        for i in range(len(o_pos)):
            o_to_index[(o_pos[i][0], o_pos[i][1])] = i
        dis_m_to_o = [[0] * len(o_pos) for _ in range(len(m_pos))]
        for i in range(len(m_pos)):
            mx = m_pos[i][0]
            my = m_pos[i][1]
            vis = [[False] * n for _ in range(m)]
            vis[mx][my] = True
            q.clear()
            q.append((mx, my, 0))
            o_cnt = 0
            while q:
                if o_cnt == len(o_pos):
                    break
                cur = q.popleft()
                x = cur[0]
                y = cur[1]
                dis = cur[2]
                for dx, dy in dirs:
                    nx = x + dx
                    ny = y + dy
                    if (
                        0 <= nx < m
                        and 0 <= ny < n
                        and maze[nx][ny] != "#"
                        and not vis[nx][ny]
                    ):
                        vis[nx][ny] = True
                        if maze[nx][ny] == "O":
                            o_cnt += 1
                            dis_m_to_o[i][o_to_index[(nx, ny)]] = dis + 1
                        q.append((nx, ny, dis + 1))
        # 终点到机关的距离
        dis_t_to_m = [0] * len(m_pos)
        q.clear()
        q.append((t[0], t[1], 0))
        vis = [[False] * n for _ in range(m)]
        vis[t[0]][t[1]] = True
        m_cnt = 0
        while q:
            if m_cnt == len(m_pos):
                break
            cur = q.popleft()
            x = cur[0]
            y = cur[1]
            dis = cur[2]
            for dx, dy in dirs:
                nx = x + dx
                ny = y + dy
                if (
                    0 <= nx < m
                    and 0 <= ny < n
                    and maze[nx][ny] != "#"
                    and not vis[nx][ny]
                ):
                    vis[nx][ny] = True
                    if maze[nx][ny] == "M":
                        m_cnt += 1
                        dis_t_to_m[m_to_index[(nx, ny)]] = dis + 1
                    q.append((nx, ny, dis + 1))
        u = (1 << len(m_pos)) - 1
        memo = [[-1] * len(m_pos) for _ in range(1 << len(m_pos))]
        return dfs(0, 0)

    # 2903. 找出满足差值条件的下标 I (Find Indices With Index and Value Difference I)
    # 2905. 找出满足差值条件的下标 II (Find Indices With Index and Value Difference II)
    def findIndices(
        self, nums: List[int], indexDifference: int, valueDifference: int
    ) -> List[int]:
        min_index = max_index = 0
        n = len(nums)
        for i in range(indexDifference, n):
            j = i - indexDifference
            if nums[j] > nums[max_index]:
                max_index = j
            elif nums[j] < nums[min_index]:
                min_index = j
            if nums[max_index] - nums[i] >= valueDifference:
                return [max_index, i]
            if nums[i] - nums[min_index] >= valueDifference:
                return [min_index, i]
        return [-1, -1]

    # 2904. 最短且字典序最小的美丽子字符串 (Shortest and Lexicographically Smallest Beautiful String)
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        i = 0
        j = 0
        n = len(s)
        cnt = 0
        res = ""
        while j < n:
            cnt += int(s[j])
            while cnt > k or i <= j and s[i] == "0":
                cnt -= int(s[i])
                i += 1
            if cnt == k:
                if (
                    res == ""
                    or j - i + 1 < len(res)
                    or j - i + 1 == len(res)
                    and s[i : j + 1] < res
                ):
                    res = s[i : j + 1]
            j += 1
        return res

    # 2906. 构造乘积矩阵 (Construct Product Matrix)
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        m = len(grid)
        n = len(grid[0])
        res = [[1] * n for _ in range(m)]
        mul = 1
        MOD = 12345
        for i in range(m):
            for j in range(n):
                res[i][j] = mul
                mul *= grid[i][j]
                mul %= MOD
        mul = 1
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                res[i][j] *= mul
                res[i][j] %= MOD
                mul *= grid[i][j]
                mul %= MOD
        return res

    # 2899. 上一个遍历的整数 (Last Visited Integers)
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        res = []
        k = 0
        for w in words:
            if w[0] == "p":
                k += 1
                if len(nums) >= k:
                    res.append(nums[-k])
                else:
                    res.append(-1)
            else:
                k = 0
                nums.append(int(w))
        return res

    # 2900. 最长相邻不相等子序列 I (Longest Unequal Adjacent Groups Subsequence I)
    def getLongestSubsequence(self, words: List[str], groups: List[int]) -> List[str]:
        pre = groups[0]
        res = [words[0]]
        for w, g in zip(words, groups):
            if (g ^ pre) == 1:
                res.append(w)
                pre ^= 1
        return res

    # 2900. 最长相邻不相等子序列 I (Longest Unequal Adjacent Groups Subsequence I)
    def getLongestSubsequence(self, words: List[str], groups: List[int]) -> List[str]:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            res = dfs(i + 1, j)
            if j == -1 or j ^ groups[i] == 1:
                res = max(res, dfs(i + 1, groups[i]) + 1)
            return res

        def make_ans(i: int, j: int, mx: int) -> None:
            if i == n:
                return
            if mx == dfs(i + 1, j):
                make_ans(i + 1, j, mx)
                return
            res.append(words[i])
            make_ans(i + 1, groups[i], mx - 1)

        n = len(words)
        mx = dfs(0, -1)
        res = []
        make_ans(0, -1, mx)
        return res

    # 2652. 倍数求和 (Sum Multiples)
    # Sn=n*a1+n(n-1)d/2
    def sumOfMultiples(self, n: int) -> int:
        def s(m: int) -> int:
            return n // m * m + (n // m) * (n // m - 1) * m // 2

        return s(3) + s(5) + s(7) - s(15) - s(21) - s(35) + s(105)

    # 238. 除自身以外数组的乘积 (Product of Array Except Self)
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [0] * n
        suf = 1
        for i in range(n - 1, -1, -1):
            res[i] = suf
            suf *= nums[i]
        pre = 1
        for i in range(n):
            res[i] *= pre
            pre *= nums[i]
        return res

    # 994. 腐烂的橘子 (Rotting Oranges)
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        rotten = 0
        freshed = 0
        q = deque()
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    freshed += 1
                elif grid[i][j] == 2:
                    rotten += 1
                    q.append((i, j))
        if freshed == 0:
            return 0
        if rotten == 0:
            return -1
        res = 0
        dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        while q:
            size = len(q)
            for _ in range(size):
                (x, y) = q.popleft()
                for dx, dy in dirs:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 1:
                        grid[nx][ny] = 2
                        freshed -= 1
                        q.append((nx, ny))
            if q:
                res += 1
        return -1 if freshed else res

    # 993. 二叉树的堂兄弟节点 (Cousins in Binary Tree) --dfs
    def isCousins(self, root: Optional[TreeNode], x: int, y: int) -> bool:
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        def dfs(node: Optional[TreeNode], level: int, fa: int) -> None:
            if node is None:
                return
            d[node.val] = (level, fa)
            dfs(node.left, level + 1, node.val)
            dfs(node.right, level + 1, node.val)

        d = dict()
        dfs(root, 0, -1)
        return d[x][0] == d[y][0] and d[x][1] != d[y][1]

    # 993. 二叉树的堂兄弟节点 (Cousins in Binary Tree) --bfs
    def isCousins(self, root: Optional[TreeNode], x: int, y: int) -> bool:
        q = deque()
        q.append((root, -1))
        while q:
            size = len(q)
            f = -2
            for _ in range(size):
                (node, fa) = q.popleft()
                if f == -2 and (node.val == x or node.val == y):
                    f = fa
                elif f != -2 and (node.val == x or node.val == y):
                    return fa != f
                if node.left:
                    q.append((node.left, node.val))
                if node.right:
                    q.append((node.right, node.val))
            if f != -2:
                return False
        return False

    # 2530. 执行 K 次操作后的最大分数 (Maximal Score After Applying K Operations)
    def maxKelements(self, nums: List[int], k: int) -> int:
        q = [-x for x in nums]
        heapq.heapify(q)

        res = 0
        for _ in range(k):
            x = heapq.heappop(q)
            res += -x
            heapq.heappush(q, -((-x + 2) // 3))
        return res

    # 46. 全排列 (Permutations)
    # LCR 083. 全排列
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        used = 0
        _list = []
        res = []

        def dfs() -> None:
            if len(_list) == n:
                res.append(_list.copy())
                return
            nonlocal used
            for j in range(n):
                if ((used >> j) & 1) != 0:
                    continue
                used ^= 1 << j
                _list.append(nums[j])
                dfs()
                _list.pop()
                used ^= 1 << j

        dfs()
        return res

    # 47. 全排列 II (Permutations II)
    # LCR 084. 全排列 II
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        used = 0
        _list = []
        res = []

        def dfs() -> None:
            if len(_list) == n:
                res.append(_list.copy())
                return
            for i in range(n):
                nonlocal used
                if (
                    ((used >> i) & 1) == 1
                    or i > 0
                    and nums[i] == nums[i - 1]
                    and ((used >> (i - 1)) & 1) == 0
                ):
                    continue
                used ^= 1 << i
                _list.append(nums[i])
                dfs()
                _list.pop()
                used ^= 1 << i

        dfs()
        return res

    # LCR 129. 字母迷宫
    # 79. 单词搜索 (Word Search)
    def wordPuzzle(self, grid: List[List[str]], target: str) -> bool:
        m = len(grid)
        n = len(grid[0])
        l = len(target)
        vis = [[False] * n for _ in range(m)]
        dirs = [[0, -1], [0, 1], [-1, 0], [1, 0]]

        def dfs(i, j, k) -> bool:
            if k == l:
                return True
            if not (i >= 0 and i < m and j >= 0 and j < n):
                return False
            if vis[i][j]:
                return False
            if grid[i][j] != target[k]:
                return False
            vis[i][j] = True
            for dx, dy in dirs:
                if dfs(i + dx, j + dy, k + 1):
                    return True
            vis[i][j] = False

        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0):
                    return True
        return False

    # 78. 子集 (Subsets)
    # LCR 079. 子集
    # 面试题 08.04. 幂集
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        res = []
        _list = []

        def dfs(i: int) -> None:
            if i == n:
                res.append(_list.copy())
                return
            dfs(i + 1)
            _list.append(nums[i])
            dfs(i + 1)
            _list.pop()

        dfs(0)
        return res

    # 面试题 04.01. 节点间通路
    def findWhetherExistsPath(
        self, n: int, graph: List[List[int]], start: int, target: int
    ) -> bool:
        g = [[] for _ in range(n)]
        for a, b in graph:
            g[a].append(b)
        vis = [False] * n

        def dfs(x: int) -> bool:
            if target == x:
                return True
            if vis[x]:
                return False
            vis[x] = True
            for y in g[x]:
                if dfs(y):
                    return True
            return False

        return dfs(start)

    # 77. 组合 (Combinations)
    # LCR 080. 组合
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        _list = []

        def dfs(i: int) -> None:
            if len(_list) > k:
                return
            if len(_list) + n - i < k:
                return
            if i == n:
                res.append(_list.copy())
                return
            dfs(i + 1)
            _list.append(i + 1)
            dfs(i + 1)
            _list.pop()

        dfs(0)
        return res

    # 1726. 同积元组 (Tuple with Same Product)
    def tupleSameProduct(self, nums: List[int]) -> int:
        n = len(nums)
        dic = collections.defaultdict(int)
        res = 0
        for i in range(n):
            for j in range(i + 1, n):
                res += dic[nums[i] * nums[j]]
                dic[nums[i] * nums[j]] += 1
        return res * 8

    # 90. 子集 II (Subsets II)
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        def dfs(i: int) -> None:
            if i == n:
                res.append(arr.copy())
                return
            dfs(i + 1)
            if i == 0 or nums[i] != nums[i - 1] or used[i - 1]:
                used[i] = True
                arr.append(nums[i])
                dfs(i + 1)
                arr.pop()
                used[i] = False

        nums.sort()
        n = len(nums)
        arr = []
        res = []
        used = [False] * n
        dfs(0)
        return res

    # 39. 组合总和 (Combination Sum)
    # LCR 081. 组合总和
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort(reverse=True)
        n = len(candidates)
        res = []
        path = []

        def dfs(i: int, j: int) -> None:
            if i == n:
                if j == target:
                    res.append(path.copy())
                return
            if j > target:
                return
            dfs(i + 1, j)
            path.append(candidates[i])
            dfs(i, j + candidates[i])
            path.pop()

        dfs(0, 0)
        return res

    # 40. 组合总和 II (Combination Sum II)
    # LCR 082. 组合总和 II
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        def dfs(i: int, j: int) -> None:
            if i == n:
                if j == target:
                    res.append(path.copy())
                return
            if j > target:
                return
            dfs(i + 1, j)
            k = 1
            cur = []
            while k <= _list[i][1]:
                if j + k * _list[i][0] > target:
                    break
                cur.append(_list[i][0])
                path.extend(cur)
                dfs(i + 1, j + k * _list[i][0])
                f = k
                while f:
                    path.pop()
                    f -= 1
                k += 1

        dic = collections.defaultdict(int)
        for c in candidates:
            dic[c] += 1
        _list = []
        for k in dic.keys():
            _list.append([k, dic[k]])
        n = len(_list)
        path = []
        res = []
        dfs(0, 0)
        return res

    # 2525. 根据规则将箱子分类 (Categorize Box According to Criteria)
    def categorizeBox(self, length: int, width: int, height: int, mass: int) -> str:
        w = 10**4
        m = 10**9
        is_bulky = (
            length >= w or width >= w or height >= w or length * width * height >= m
        )
        is_heavy = mass >= 100
        if is_bulky and is_heavy:
            return "Both"
        if is_bulky:
            return "Bulky"
        if is_heavy:
            return "Heavy"
        return "Neither"

    # 279. 完全平方数 (Perfect Squares)
    def numSquares(self, n: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j >= n or i == 0:
                return 0 if j == n else inf
            return min(dfs(i, i * i + j) + 1, dfs(i - 1, j))

        res = dfs(isqrt(n), 0)
        dfs.cache_clear()
        return res

    # 221. 最大正方形 (Maximal Square)
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        m = len(matrix)
        n = len(matrix[0])
        res = 0
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if i == 0 or j == 0:
                    dp[i][j] = int(matrix[i][j])
                elif int(matrix[i][j]) == 1:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                res = max(res, dp[i][j])
        return res * res

    # 1277. 统计全为 1 的正方形子矩阵 (Count Square Submatrices with All Ones)
    def countSquares(self, matrix: List[List[int]]) -> int:
        m = len(matrix)
        n = len(matrix[0])
        res = 0
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if i == 0 or j == 0:
                    dp[i][j] = int(matrix[i][j])
                elif int(matrix[i][j]) == 1:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                res += dp[i][j]
        return res

    # 377. 组合总和 Ⅳ (Combination Sum IV)
    # LCR 104. 组合总和 Ⅳ
    def combinationSum4(self, nums: List[int], target: int) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == target:
                return 1
            res = 0
            for num in nums:
                if num + i <= target:
                    res += dfs(i + num)
            return res

        return dfs(0)

    # 647. 回文子串 (Palindromic Substrings)
    # LCR 020. 回文子串
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        res = 0
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if (
                    i == j
                    or j - i == 1
                    and s[i] == s[j]
                    or j - i > 1
                    and s[i] == s[j]
                    and dp[i + 1][j - 1]
                ):
                    dp[i][j] = True
                    res += 1
        return res

    # 2316. 统计无向图中无法互相到达点对数 (Count Unreachable Pairs of Nodes in an Undirected Graph)
    def countPairs(self, n: int, edges: List[List[int]]) -> int:
        def dfs(x: int, fa: int) -> None:
            vis[x] = True
            nonlocal cnt
            cnt += 1
            for y in g[x]:
                if y != fa and not vis[y]:
                    dfs(y, x)

        g = [[] for _ in range(n)]
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
        vis = [False] * n
        cnt = 0
        pre = 0
        res = 0
        for i in range(n):
            if not vis[i]:
                cnt = 0
                dfs(i, -1)
                res += cnt * pre
                pre += cnt
        return res

    # 2908. 元素和最小的山形三元组 I (Minimum Sum of Mountain Triplets I)
    # 2909. 元素和最小的山形三元组 II (Minimum Sum of Mountain Triplets II)
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        left = [inf] * n
        left[0] = nums[0]
        for i in range(1, n):
            left[i] = min(left[i - 1], nums[i])
        res = inf
        right = nums[n - 1]
        for i in range(n - 2, -1, -1):
            right = min(right, nums[i])
            if nums[i] > right and nums[i] > left[i]:
                res = min(res, left[i] + nums[i] + right)
        return res if res < inf else -1

    # 2911. 得到 K 个半回文串的最少修改次数 (Minimum Changes to Make K Semi-palindromes)
    def minimumChanges(self, s: str, k: int) -> int:
        def cal(i: int, j: int) -> int:
            l = j - i + 1
            ss = s[i : j + 1]
            m = inf
            for d in range(1, (l >> 1) + 1):
                if l % d == 0:
                    cur_s = 0
                    dic = [[] for _ in range(d)]
                    for x in range(l):
                        dic[x % d].append(ss[x])
                    for x in range(d):
                        cur = 0
                        left = 0
                        right = len(dic[x]) - 1
                        while left < right:
                            if dic[x][left] != dic[x][right]:
                                cur += 1
                            left += 1
                            right -= 1
                        cur_s += cur
                    m = min(m, cur_s)
            return m

        n = len(s)
        arr = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                arr[i][j] = cal(i, j)

        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0 if j == k else inf
            if j == k:
                return inf
            res = inf
            for x in range(i + 1, n):
                if n - x - 1 < (k - j - 1) * 2:
                    break
                res = min(res, dfs(x + 1, j + 1) + arr[i][x])
            return res

        return dfs(0, 0)

    # 100097. 合法分组的最少组数 (Minimum Number of Groups to Create a Valid Assignment)
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        cnt = Counter(nums)
        m = min(cnt.values())
        for k in range(m, 0, -1):
            res = 0
            for c in cnt.values():
                if c // k < c % k:
                    break
                res += (c + k) // (k + 1)
            else:
                return res

    # 2678. 老人的数目 (Number of Senior Citizens)
    def countSeniors(self, details: List[str]) -> int:
        return sum(int(x[11:13]) > 60 for x in details)

    # 2698. 求一个整数的惩罚数
    def punishmentNumber(self, n: int) -> int:
        def dfs(i: int, j: int) -> bool:
            if i == len(s):
                return j == target
            ss = 0
            for k in range(i, len(s)):
                ss = ss * 10 + int(s[k])
                if ss > target:
                    break
                if dfs(k + 1, j + ss):
                    return True
            return False

        res = 0
        target = 0
        for i in range(1, n + 1):
            s = str(i * i)
            target = i
            if dfs(0, 0):
                res += i * i
        return res

    # 2520. 统计能整除数字的位数 (Count the Digits That Divide a Number)
    def countDigits(self, num: int) -> int:
        copy = num
        res = 0
        while copy:
            m = copy % 10
            if num % m == 0:
                res += 1
            copy //= 10
        return res

    # 1465. 切割后面积最大的蛋糕 (Maximum Area of a Piece of Cake After Horizontal and Vertical Cuts)
    def maxArea(
        self, h: int, w: int, horizontalCuts: List[int], verticalCuts: List[int]
    ) -> int:
        MOD = 10**9 + 7
        horizontalCuts.sort()
        verticalCuts.sort()
        max_h = max(horizontalCuts[0], h - horizontalCuts[-1])
        for i in range(1, len(horizontalCuts)):
            max_h = max(max_h, horizontalCuts[i] - horizontalCuts[i - 1])
        max_w = max(verticalCuts[0], w - verticalCuts[-1])
        for i in range(1, len(verticalCuts)):
            max_w = max(max_w, verticalCuts[i] - verticalCuts[i - 1])
        return max_h * max_w % MOD

    # 274. H 指数 (H-Index)
    def hIndex(self, citations: List[int]) -> int:
        citations.sort()
        res = 0
        i = len(citations) - 1
        while i >= 0 and citations[i] > res:
            res += 1
            i -= 1
        return res

    #  100094. 子数组不同元素数目的平方和 I (Subarrays Distinct Element Sum of Squares I)
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        for i in range(n):
            for j in range(i, n):
                s = set()
                for k in range(i, j + 1):
                    s.add(nums[k])
                res += len(s) * len(s)
        return res

    # 2914. 使二进制字符串变美丽的最少修改次数 (Minimum Number of Changes to Make Binary String Beautiful)
    def minChanges(self, s: str) -> int:
        return sum(s[i] != s[i + 1] for i in range(0, len(s), 2))

    # 2915. 和为目标值的最长子序列的长度 (Length of the Longest Subsequence That Sums to Target)
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0 if target == j else -inf
            res = dfs(i + 1, j)
            if j + nums[i] <= target:
                res = max(res, dfs(i + 1, j + nums[i]) + 1)
            return res

        n = len(nums)
        if sum(nums) < target:
            return -1
        if sum(nums) == target:
            return n
        nums.sort(reverse=True)
        res = dfs(0, 0)
        dfs.cache_clear()
        return -1 if res <= 0 else res

    # 2917. 找出数组中的 K-or 值 (Find the K-or of an Array)
    def findKOr(self, nums: List[int], k: int) -> int:
        res = 0
        cnt = [0] * 31
        for x in nums:
            for i in range(31):
                if (x >> i) & 1:
                    cnt[i] += 1
        for i, v in enumerate(cnt):
            if v >= k:
                res |= 1 << i
        return res

    # 2918. 数组的最小相等和 (Minimum Equal Sum of Two Arrays After Replacing Zeros)
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        s1 = 0
        cnt1 = 0
        s2 = 0
        cnt2 = 0
        for x in nums1:
            s1 += x
            cnt1 += int(x == 0)
        for x in nums2:
            s2 += x
            cnt2 += int(x == 0)
        s1 += cnt1
        s2 += cnt2
        return -1 if s1 > s2 and cnt2 == 0 or s1 < s2 and cnt1 == 0 else max(s1, s2)

    # 2919. 使数组变美的最小增量运算数 (Minimum Increment Operations to Make Array Beautiful)
    def minIncrementOperations(self, nums: List[int], k: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            res = dfs(i + 1, 0) + max(k - nums[i], 0)
            if j < 2:
                res = min(res, dfs(i + 1, j + 1))
            return res

        n = len(nums)
        return dfs(0, 0)

    # 275. H 指数 II (H-Index II)
    def hIndex(self, citations: List[int]) -> int:
        n = len(citations)

        def check(target: int) -> bool:
            for i in range(n - 1, -1, -1):
                if citations[i] < target:
                    return n - i - 1 >= target
            return n >= target

        left = 0
        right = n
        res = 0
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = mid
                left = mid + 1
            else:
                right = mid - 1
        return res

    # 2920. 收集所有金币可获得的最大积分 (Maximum Points After Collecting Coins From All Nodes)
    def maximumPoints(self, edges: List[List[int]], coins: List[int], k: int) -> int:
        @cache
        def dfs(i: int, j: int, fa: int) -> int:
            res1 = (coins[i] >> j) - k
            res2 = coins[i] >> (j + 1)
            for y in g[i]:
                if y != fa:
                    res1 += dfs(y, j, i)
                    if j < 14:
                        res2 += dfs(y, j + 1, i)
            return max(res1, res2)

        n = len(coins)
        g = [[] * n for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        return dfs(0, 0, -1)

    # 2003. 每棵子树内缺失的最小基因值 (Smallest Missing Genetic Value in Each Subtree)
    def smallestMissingValueSubtree(
        self, parents: List[int], nums: List[int]
    ) -> List[int]:
        def dfs(x: int) -> None:
            vis.add(nums[x])
            for y in g[x]:
                if nums[y] not in vis:
                    dfs(y)

        n = len(nums)
        node = -1
        res = [1] * n
        for i in range(n):
            if nums[i] == 1:
                node = i
        if node < 0:
            return res
        g = [[] * n for _ in range(n)]
        for i in range(1, n):
            g[parents[i]].append(i)
        mex = 2
        vis = set()
        while node >= 0:
            dfs(node)
            while mex in vis:
                mex += 1
            res[node] = mex
            node = parents[node]
        return res

    # 2127. 参加会议的最多员工数 (Maximum Employees to Be Invited to a Meeting)
    def maximumInvitations(self, favorite: List[int]) -> int:
        n = len(favorite)
        deg = [0] * n
        for f in favorite:
            deg[f] += 1
        dp = [1] * n
        q = collections.deque()
        for i in range(n):
            if not deg[i]:
                q.append(i)
        while q:
            node = q.popleft()
            neighbor = favorite[node]
            dp[neighbor] = max(dp[neighbor], dp[node] + 1)
            deg[neighbor] -= 1
            if not deg[neighbor]:
                q.append(neighbor)
        two_nodes_ring = 0
        three_or_more_ring = 0
        for i in range(n):
            if deg[i]:
                neighbor = favorite[i]
                # 环长 == 2
                if favorite[neighbor] == i:
                    deg[i] = 0
                    deg[neighbor] = 0
                    two_nodes_ring += dp[i] + dp[neighbor]
                # 环长 > 2
                else:
                    cnt = 0
                    node = i
                    while deg[node]:
                        deg[node] = 0
                        cnt += 1
                        node = favorite[node]
                    three_or_more_ring = max(three_or_more_ring, cnt)
        return max(three_or_more_ring, two_nodes_ring)

    # 2103. 环和杆 (Rings and Rods)
    def countPoints(self, rings: str) -> int:
        res = 0
        dic = collections.defaultdict(int)
        dic["R"] = 0
        dic["G"] = 1
        dic["B"] = 2
        bit_map = [0] * 10
        n = len(rings)
        for r in range(0, n, 2):
            bit_map[int(rings[r + 1])] |= 1 << dic[rings[r]]
        for b in bit_map:
            if b == (1 << 3) - 1:
                res += 1
        return res

    # 117. 填充每个节点的下一个右侧节点指针 II (Populating Next Right Pointers in Each Node II) --bfs
    def connect(self, root: "Node") -> "Node":
        class Node:
            def __init__(
                self,
                val: int = 0,
                left: "Node" = None,
                right: "Node" = None,
                next: "Node" = None,
            ):
                self.val = val
                self.left = left
                self.right = right
                self.next = next

        q = collections.deque()
        q.append(root)
        while q:
            size = len(q)
            for i in range(size):
                x = q.popleft()
                if not x:
                    continue
                nxt = None
                if i < size - 1:
                    nxt = q.popleft()
                    x.next = nxt
                if nxt:
                    q.appendleft(nxt)
                if x.left:
                    q.append(x.left)
                if x.right:
                    q.append(x.right)
        return root

    # 117. 填充每个节点的下一个右侧节点指针 II (Populating Next Right Pointers in Each Node II) --dfs
    def connect(self, root: "Node") -> "Node":
        class Node:
            def __init__(
                self,
                val: int = 0,
                left: "Node" = None,
                right: "Node" = None,
                next: "Node" = None,
            ):
                self.val = val
                self.left = left
                self.right = right
                self.next = next

        def dfs(node: "Node", depth: int) -> None:
            if node is None:
                return
            if depth == len(pre):
                pre.append(node)
            else:
                pre[depth].next = node
                pre[depth] = node
            dfs(node.left, depth + 1)
            dfs(node.right, depth + 1)

        pre = []
        dfs(root, 0)
        return root

    # 421. 数组中两个数的最大异或值 (Maximum XOR of Two Numbers in an Array)
    # LCR 067. 数组中两个数的最大异或值
    def findMaximumXOR(self, nums: List[int]) -> int:
        res = 0
        mask = 0
        s = set()
        for i in range(max(nums).bit_length() - 1, -1, -1):
            mask |= 1 << i
            s.clear()
            new_res = res | (1 << i)
            for x in nums:
                x &= mask
                if x ^ new_res in s:
                    res = new_res
                    break
                s.add(x)
        return res

    # 187. 重复的DNA序列 (Repeated DNA Sequences)
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        cnts = collections.defaultdict(int)
        # A 00
        # C 01
        # T 10
        # G 11
        dic = collections.defaultdict(int)
        dic["A"] = 0
        dic["C"] = 1
        dic["T"] = 2
        dic["G"] = 3
        u = (1 << 20) - 1
        mask = 0
        res = []
        for i, c in enumerate(s):
            mask = (mask << 2) | dic[c]
            if i >= 9:
                mask &= u
                cnts[mask] += 1
                if cnts[mask] == 2:
                    res.append(s[i - 10 + 1 : i + 1])
        return res

    # 2923. 找到冠军 I (Find Champion I)
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        d = [0] * n
        for i in range(n):
            for j in range(i):
                if grid[i][j]:
                    d[j] += 1
                else:
                    d[i] += 1
        for i, v in enumerate(d):
            if v == 0:
                return i

    # 2924. 找到冠军 II (Find Champion II)
    def findChampion(self, n: int, edges: List[List[int]]) -> int:
        deg = [0] * n
        for _, v in edges:
            deg[v] += 1
        res = -1
        for i, v in enumerate(deg):
            if v == 0:
                if res != -1:
                    return -1
                res = i
        return res

    # 100116. 在树上执行操作以后得到的最大分数 (Maximum Score After Applying Operations on a Tree)
    def maximumScoreAfterOperations(
        self, edges: List[List[int]], values: List[int]
    ) -> int:
        def dfs_score(x: int, fa: int) -> int:
            if len(g[x]) == 1:
                return 0
            res1 = s[x] - values[x]
            res2 = values[x]
            for y in g[x]:
                if y != fa:
                    res2 += dfs_score(y, x)
            return max(res1, res2)

        def dfs(x: int, fa: int) -> int:
            s[x] = values[x]
            for y in g[x]:
                if y != fa:
                    s[x] += dfs(y, x)
            return s[x]

        n = len(values)
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        g[0].append(-1)
        s = [0] * n
        dfs(0, -1)
        return dfs_score(0, -1)

    # 318. 最大单词长度乘积 (Maximum Product of Word Lengths)
    def maxProduct(self, words: List[str]) -> int:
        dic = collections.defaultdict(int)
        res = 0
        for w in words:
            m = 0
            for c in w:
                m |= 1 << (ord(c) - ord("a"))
            for k, v in dic.items():
                if (k & m) == 0:
                    res = max(res, v * len(w))
            dic[m] = max(dic[m], len(w))
        return res

    # 2586. 统计范围内的元音字符串数 (Count the Number of Vowel Strings in Range)
    def vowelStrings(self, words: List[str], left: int, right: int) -> int:
        res = 0
        u = 0
        for c in "aeiou":
            u |= 1 << (ord(c) - ord("a"))
        for i in range(left, right + 1):
            c = (1 << (ord(words[i][0]) - ord("a"))) | (
                1 << (ord(words[i][-1]) - ord("a"))
            )
            if (c & u) == c:
                res += 1
        return res

    # 2609. 最长平衡子字符串 (Find the Longest Balanced Substring of a Binary String)
    def findTheLongestBalancedSubstring(self, s: str) -> int:
        res = 0
        n = len(s)
        i = 0
        while i < n:
            cnt0 = 0
            cnt1 = 0
            j = i
            while j < n and s[j] == "0":
                j += 1
            cnt0 = j - i
            i = j
            while j < n and s[j] == "1":
                j += 1
            cnt1 = j - i
            i = j
            res = max(res, min(cnt0, cnt1) * 2)
        return res

    # 2300. 咒语和药水的成功对数 (Successful Pairs of Spells and Potions)
    def successfulPairs(
        self, spells: List[int], potions: List[int], success: int
    ) -> List[int]:
        n = len(spells)
        m = len(potions)
        potions.sort()
        res = [0] * n
        j = 0
        for i, v in sorted(zip(range(n), spells), key=lambda z: -z[1]):
            while j < m and potions[j] * v < success:
                j += 1
            if m - j == 0:
                break
            res[i] = m - j
        return res

    # 2928. 给小朋友们分糖果 I (Distribute Candies Among Children I)
    # 2929. 给小朋友们分糖果 II (Distribute Candies Among Children II)
    def distributeCandies(self, n: int, limit: int) -> int:
        res = 0
        for i in range(min(n, limit), -1, -1):
            c = n - i
            if limit * 2 < c:
                break
            res += min(limit, c) - max(0, c - limit) + 1
        return res

    # 2930. 重新排列后包含指定子字符串的字符串数目 (Number of Strings Which Can Be Rearranged to Contain Substring)
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7

        @cache
        def dfs(i: int, j: int, k: int, l: int) -> int:
            if i == n:
                return int(j == 1 and k == 2 and l == 1)
            return (
                dfs(i + 1, 1, k, l)
                + dfs(i + 1, j, min(k + 1, 2), l)
                + dfs(i + 1, j, k, 1)
                + 23 * dfs(i + 1, j, k, l)
            ) % MOD

        return dfs(0, 0, 0, 0)

    # 2931. 购买物品的最大开销 (Maximum Spending After Buying Items)
    def maxSpending(self, values: List[List[int]]) -> int:
        m = len(values)
        n = len(values[0])
        a = []
        for i in range(m):
            for j in range(n):
                a.append(values[i][j])
        a.sort()
        res = 0
        for d, x in enumerate(a, 1):
            res += d * x
        return res

    # 2933. 高访问员工 (High-Access Employees)
    def findHighAccessEmployees(self, access_times: List[List[str]]) -> List[str]:
        dic = collections.defaultdict(list)
        for c, s in access_times:
            dic[c].append(s)
        res = []
        for k, v in dic.items():
            v.sort()
            n = len(v)
            for i in range(2, n):
                if int(v[i]) - int(v[i - 2]) < 100:
                    res.append(k)
                    break
        return res

    # 2934. 最大化数组末位元素的最少操作次数 (Minimum Operations to Maximize Last Elements in Arrays)
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        def check(mx1: int, mx2: int) -> int:
            s = 0
            for _, x, y in zip(range(n - 1), nums1, nums2):
                if x <= mx1 and y <= mx2:
                    continue
                if x <= mx2 and y <= mx1:
                    s += 1
                else:
                    return -1
            return s

        n = len(nums1)
        return min(check(nums1[-1], nums2[-1]), check(nums2[-1], nums1[-1]) + 1)

    # 2656. K 个元素的最大和 (Maximum Sum With Exactly K Elements)
    def maximizeSum(self, nums: List[int], k: int) -> int:
        a1 = max(x for x in nums)
        an = a1 + k - 1
        return (a1 + an) * k // 2

    # 2760. 最长奇偶子数组 (Longest Even Odd Subarray With Threshold)
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        res = 0
        i = 0
        n = len(nums)
        while i < n:
            j = i + 1
            if nums[i] % 2 == 0 and nums[i] <= threshold:
                while j < n and nums[j] % 2 != nums[j - 1] % 2 and nums[j] <= threshold:
                    j += 1
                res = max(res, j - i)
            i = j
        return res

    # 2342. 数位和相等数对的最大和 (Max Sum of a Pair With Equal Sum of Digits)
    def maximumSum(self, nums: List[int]) -> int:
        dic = collections.defaultdict(int)
        res = -1
        for num in nums:
            d = 0
            copy = num
            while num:
                d += num % 10
                num //= 10
            if dic[d]:
                res = max(res, dic[d] + copy)
            dic[d] = max(dic[d], copy)
        return res

    # 2216. 美化数组的最少删除数 (Minimum Deletions to Make Array Beautiful)
    def minDeletion(self, nums: List[int]) -> int:
        res = 0
        n = len(nums)
        i = 0
        while i < n - 1:
            if nums[i] == nums[i + 1]:
                res += 1
                i += 1
            else:
                i += 2
        if (n - res) % 2 == 1:
            res += 1
        return res

    # 2937. 使三个字符串相等 (Make Three Strings Equal)
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        i = 0
        for c1, c2, c3 in zip(s1, s2, s3):
            if c1 != c2 or c2 != c3:
                break
            i += 1
        if i == 0:
            return -1
        return len(s1) + len(s2) + len(s3) - 3 * i

    # 2938. 区分黑球与白球 (Separate Black and White Balls)
    def minimumSteps(self, s: str) -> int:
        res = 0
        c = 0
        for i, v in enumerate(s):
            if v == "0":
                res += i - c
                c += 1
        return res

    # 2939. 最大异或乘积 (Maximum Xor Product)
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        if a < b:
            a, b = b, a
        mask = (1 << n) - 1
        ax = a & ~mask
        bx = b & ~mask

        a &= mask
        b &= mask

        diff = a ^ b
        one = diff ^ mask
        ax |= one
        bx |= one
        if diff and ax == bx:
            high_bit = 1 << (diff.bit_length() - 1)
            ax |= high_bit
            diff ^= high_bit
        bx |= diff
        MOD = 10**9 + 7
        return ax * bx % MOD

    # 2940. 找到 Alice 和 Bob 可以相遇的建筑 (Find Building Where Alice and Bob Can Meet)
    def leftmostBuildingQueries(
        self, heights: List[int], queries: List[List[int]]
    ) -> List[int]:
        dic = collections.defaultdict(list)
        res = [-1] * len(queries)
        for i, (a, b) in enumerate(queries):
            if a > b:
                a, b = b, a
            if a == b or heights[a] < heights[b]:
                res[i] = b
            else:
                dic[b].append([heights[a], i])
        h = []
        for i, x in enumerate(heights):
            while h and h[0][0] < x:
                res[heapq.heappop(h)[1]] = i
            for v in dic[i]:
                heapq.heappush(h, v)
        return res

    # 1410. HTML 实体解析器 (HTML Entity Parser)
    def entityParser(self, text: str) -> str:
        res = ""
        n = len(text)
        i = 0
        while i < n:
            if text[i] != "&":
                res += text[i]
                i += 1
            elif text[i:].startswith("&quot;"):
                res += '"'
                i += 6
            elif text[i:].startswith("&apos;"):
                res += "'"
                i += 6
            elif text[i:].startswith("&amp;"):
                res += "&"
                i += 5
            elif text[i:].startswith("&gt;"):
                res += ">"
                i += 4
            elif text[i:].startswith("&lt;"):
                res += "<"
                i += 4
            elif text[i:].startswith("&frasl;"):
                res += "/"
                i += 7
            else:
                res += text[i]
                i += 1
        return res

    # 2824. 统计和小于目标的下标对数目 (Count Pairs Whose Sum is Less than Target)
    def countPairs(self, nums: List[int], target: int) -> int:
        res = 0
        nums.sort()
        i = 0
        j = len(nums) - 1
        while i < j:
            if nums[i] + nums[j] < target:
                res += j - i
                i += 1
            else:
                j -= 1
        return res

    # 1457. 二叉树中的伪回文路径 (Pseudo-Palindromic Paths in a Binary Tree)
    def pseudoPalindromicPaths(self, root: Optional[TreeNode]) -> int:
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        def dfs(root: Optional[TreeNode], d: int) -> None:
            if root is None:
                return
            d ^= 1 << root.val
            if root.left is None and root.right is None:
                nonlocal res
                res += int(d.bit_count() <= 1)
            dfs(root.left, d)
            dfs(root.right, d)

        res = 0
        dfs(root, 0)
        return res

    # 2942. 查找包含给定字符的单词 (Find Words Containing Character)
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        res = []
        for i, w in enumerate(words):
            if x in w:
                res.append(i)
        return res

    # 2943. 最大化网格图中正方形空洞的面积 (Maximize Area of Square Hole in Grid)
    def maximizeSquareHoleArea(
        self, n: int, m: int, hBars: List[int], vBars: List[int]
    ) -> int:
        def cal(bars: List[int]) -> int:
            res = 0
            bars.sort()
            i = 0
            n = len(bars)
            while i < n:
                j = i + 1
                while j < n and bars[j] - bars[j - 1] == 1:
                    j += 1
                res = max(res, j - i + 1)
                i = j
            return res

        s = min(cal(hBars), cal(vBars))
        return s**2

    # 2944. 购买水果需要的最少金币数 (Minimum Number of Coins for Fruits)
    def minimumCoins(self, prices: List[int]) -> int:
        @cache
        def dfs(i: int) -> int:
            if i * 2 >= n:
                return prices[i - 1]
            return (
                min(dfs(j + 1) for j in range(i, min(i + i + 1, n + 1))) + prices[i - 1]
            )

        n = len(prices)
        return dfs(1)

    # 2946. 循环移位后的矩阵相似检查 (Matrix Similarity After Cyclic Shifts)
    def areSimilar(self, mat: List[List[int]], k: int) -> bool:
        k %= len(mat[0])
        return k == 0 or all(r == r[k:] + r[:k] for r in mat)

    # 2947. 统计美丽子字符串 I (Count Beautiful Substrings I)
    def beautifulSubstrings(self, s: str, k: int) -> int:
        res = 0
        n = len(s)
        for i in range(n):
            a = 0
            b = 0
            for j in range(i, n):
                if s[j] in "aeiou":
                    a += 1
                else:
                    b += 1
                if a == b and a * b % k == 0:
                    res += 1
        return res

    # 2948. 交换得到字典序最小的数组 (Make Lexicographically Smallest Array by Swapping Elements)
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        arr = []
        for i, v in enumerate(nums):
            arr.append([i, v])
        arr.sort(key=lambda k: k[1])
        res = [0] * n
        list_id = []
        list_val = []
        for i, (id, v) in zip(range(n), arr):
            list_id.append(id)
            list_val.append(v)
            if i == n - 1 or arr[i + 1][1] - v > limit:
                list_id.sort()
                list_val.sort()
                for idx, val in zip(list_id, list_val):
                    res[idx] = val
                list_id.clear()
                list_val.clear()
        return res

    # 1657. 确定两个字符串是否接近 (Determine if Two Strings Are Close)
    def closeStrings(self, word1: str, word2: str) -> bool:
        return Counter(word1).keys() == Counter(word2).keys() and sorted(
            Counter(word1).values()
        ) == sorted(Counter(word2).values())

    # 2661. 找出叠涂元素 (First Completely Painted Row or Column)
    def firstCompleteIndex(self, arr: List[int], mat: List[List[int]]) -> int:
        dic = collections.defaultdict(tuple)
        m = len(mat)
        n = len(mat[0])
        for i in range(m):
            for j in range(n):
                dic[mat[i][j]] = (i, j)
        row = [0] * m
        col = [0] * n
        for i, v in enumerate(arr):
            (x, y) = dic[v]
            row[x] += 1
            col[y] += 1
            if row[x] == n or col[y] == m:
                return i

    # 1094. 拼车 (Car Pooling)
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        right = max(t for _, _, t in trips)
        diff = [0] * (right + 1)
        for p, f, t in trips:
            diff[f] += p
            diff[t] -= p
        return all(s <= capacity for s in accumulate(diff))

    # 1374. 从二叉搜索树到更大和树 (Binary Search Tree to Greater Sum Tree)
    def bstToGst(self, root: TreeNode) -> TreeNode:
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        sum = 0

        def dfs(root: TreeNode) -> None:
            if root is None:
                return
            dfs(root.right)
            nonlocal sum
            sum += root.val
            root.val = sum
            dfs(root.left)

        dfs(root)
        return root

    # 2952. 需要添加的硬币的最小数量 (Minimum Number of Coins to be Added)
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        res = 0
        s = 1
        i = 0
        while s <= target:
            if i < len(coins) and coins[i] <= s:
                s += coins[i]
                i += 1
            else:
                s *= 2
                res += 1
        return res

    # 2951. 找出峰值 (Find the Peaks)
    def findPeaks(self, mountain: List[int]) -> List[int]:
        res = []
        for i in range(1, len(mountain) - 1):
            if mountain[i - 1] < mountain[i] and mountain[i] > mountain[i + 1]:
                res.append(i)
        return res

    # 1466. 重新规划路线 (Reorder Routes to Make All Paths Lead to the City Zero)
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        def dfs(x: int, fa: int) -> None:
            for y in g[x]:
                if y[0] != fa:
                    nonlocal res
                    res += y[1]
                    dfs(y[0], x)

        g = [[] for _ in range(n)]
        for u, v in connections:
            g[u].append((v, 1))
            g[v].append((u, 0))
        res = 0
        dfs(0, -1)
        return res

    # 2048. 下一个更大的数值平衡数 (Next Greater Numerically Balanced Number)
    def nextBeautifulNumber(self, n: int) -> int:
        x = n + 1
        while True:
            copy = x
            cnt = [0] * 10
            while copy:
                cnt[copy % 10] += 1
                copy //= 10
            if all(not v or i == v for i, v in enumerate(cnt)):
                return x
            x += 1

    # 2956. 找到两个数组中的公共元素 (Find Common Elements Between Two Arrays)
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        return [
            sum(x in set(nums2) for x in nums1),
            sum(x in set(nums1) for x in nums2),
        ]

    # 100152. 消除相邻近似相等字符 (Remove Adjacent Almost-Equal Characters)
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        i = 0
        res = 0
        while i < n:
            j = i + 1
            while j < n and abs(ord(word[j]) - ord(word[j - 1])) <= 1:
                j += 1
            res += (j - i) // 2
            i = j
        return res

    # 2959. 关闭分部的可行集合数目 (Number of Possible Sets of Closing Branches)
    def numberOfSets(self, n: int, maxDistance: int, roads: List[List[int]]) -> int:
        def check_all_connected(i: int) -> bool:
            def dfs(x: int) -> None:
                nonlocal m
                m |= 1 << x
                for y, _ in g[x]:
                    if (m >> y) & 1 == 0 and (i >> y) & 1 == 1:
                        dfs(y)

            lb = (i & -i).bit_length() - 1
            m = 0
            dfs(lb)
            return m == i

        def check_distance(i: int) -> bool:
            def check(start: int) -> bool:
                dis = [inf] * n
                dis[start] = 0
                q = deque()
                q.append([start, 0])
                while q:
                    [x, d] = q.popleft()
                    if d > dis[x]:
                        continue
                    for y, w in g[x]:
                        if (i >> y) & 1 == 1 and d + w < dis[y]:
                            dis[y] = d + w
                            q.append([y, d + w])
                return all((i >> x) & 1 == 0 or dis[x] <= maxDistance for x in range(n))

            m = i
            while m:
                lb = (m & -m).bit_length() - 1
                if not check(lb):
                    return False
                m &= m - 1
            return True

        # 去重
        d = defaultdict(int)
        for u, v, w in roads:
            if (u, v) not in d or d[(u, v)] > w:
                d[(u, v)] = w
                d[(v, u)] = w
        g = [[] for _ in range(n)]
        for (u, v), w in d.items():
            g[u].append((v, w))
        res = 1
        for i in range(1, 1 << n):
            if not check_all_connected(i):
                continue
            if check_distance(i):
                res += 1
        return res

    # 2960. 统计已测试设备 (Count Tested Devices After Test Operations)
    def countTestedDevices(self, batteryPercentages: List[int]) -> int:
        cnt = 0
        for v in batteryPercentages:
            if v - cnt > 0:
                cnt += 1
        return cnt

    # 100155. 双模幂运算 (Double Modular Exponentiation)
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        res = []
        for i, (a, b, c, m) in enumerate(variables):
            x = pow(a, b, 10)
            if pow(x, c, m) == target:
                res.append(i)
        return res

    # 2962. 统计最大元素出现至少 K 次的子数组 (Count Subarrays Where Max Element Appears at Least K Times)
    def countSubarrays(self, nums: List[int], k: int) -> int:
        mx = max(nums)
        j = 0
        res = 0
        for v in nums:
            if v == mx:
                k -= 1
            while k <= 0:
                if nums[j] == mx:
                    k += 1
                j += 1
            res += j
        return res

    # 100136. 统计好分割方案的数目 (Count the Number of Good Partitions)
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        first = collections.defaultdict(int)
        last = collections.defaultdict(int)
        for i, v in enumerate(nums):
            if v not in first.keys():
                first[v] = i
            last[v] = i
        i = 0
        cnt = 0
        n = len(nums)
        while i < n:
            left = first[nums[i]]
            right = last[nums[i]]
            j = i
            while j < n and not (first[nums[j]] > right or last[nums[j]] < left):
                left = min(left, first[nums[j]])
                right = max(right, last[nums[j]])
                j += 1
            cnt += 1
            i = j
        MOD = 10**9 + 7
        return pow(2, cnt - 1, MOD)

    # 2697. 字典序最小回文串 (Lexicographically Smallest Palindrome)
    def makeSmallestPalindrome(self, s: str) -> str:
        s = list(s)
        left, right = 0, len(s) - 1
        while left < right:
            if s[left] != s[right]:
                s[left] = s[right] = min(s[left], s[right])
            left += 1
            right -= 1
        return "".join(s)

    # 2132. 用邮票贴满网格图 (Stamping the Grid)
    def possibleToStamp(
        self, grid: List[List[int]], stampHeight: int, stampWidth: int
    ) -> bool:
        m = len(grid)
        n = len(grid[0])
        p = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                p[i + 1][j + 1] = p[i + 1][j] + p[i][j + 1] - p[i][j] + grid[i][j]
        d = [[0] * (n + 2) for _ in range(m + 2)]
        for i2 in range(stampHeight, m + 1):
            for j2 in range(stampWidth, n + 1):
                i1 = i2 - stampHeight + 1
                j1 = j2 - stampWidth + 1
                if p[i2][j2] - p[i2][j1 - 1] - p[i1 - 1][j2] + p[i1 - 1][j1 - 1] == 0:
                    d[i1][j1] += 1
                    d[i1][j2 + 1] -= 1
                    d[i2 + 1][j1] -= 1
                    d[i2 + 1][j2 + 1] += 1
        for i in range(m):
            for j in range(n):
                d[i + 1][j + 1] += d[i][j + 1] + d[i + 1][j] - d[i][j]
                if grid[i][j] == 0 and d[i + 1][j + 1] == 0:
                    return False
        return True

    # 2415. 反转二叉树的奇数层 (Reverse Odd Levels of Binary Tree)
    def reverseOddLevels(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        q = collections.deque()
        i = 0
        q.append(root)
        while q:
            if i & 1:
                l, r = 0, len(q) - 1
                while i < r:
                    q[l].val, q[r].val = q[r].val, q[l].val
                    l += 1
                    r -= 1
            size = len(q)
            for _ in range(size):
                x = q.popleft()
                if x.left:
                    q.append(x.left)
                    q.append(x.right)
            i ^= 1
        return root

    # 746. 使用最小花费爬楼梯 (Min Cost Climbing Stairs)
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)

        @cache
        def dfs(i: int) -> int:
            if i >= n:
                return 0
            return min(dfs(i + 1), dfs(i + 2)) + cost[i]

        return min(dfs(0), dfs(1))

    # 162. 寻找峰值 (Find Peak Element)
    def findPeakElement(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 2
        while left <= right:
            mid = (left + right) >> 1
            if nums[mid] > nums[mid + 1]:
                right = mid - 1
            else:
                left = mid + 1
        return left

    # 2965. 找出缺失和重复的数字 (Find Missing and Repeated Values)
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        xor_all = 0
        for i in range(n):
            for j in range(n):
                xor_all ^= grid[i][j]
        # 如果 n 是偶数，那么 1 到 n ^ 2 的异或和等于 n ^ 2，否则等于 1。
        xor_all ^= 1 if n % 2 else n * n
        lb = (xor_all & -xor_all).bit_length() - 1
        res = [0] * 2
        for i in range(n):
            for j in range(n):
                res[(grid[i][j] >> lb) & 1] ^= grid[i][j]
        for i in range(1, n * n + 1):
            res[(i >> lb) & 1] ^= i
        for i in range(n):
            for j in range(n):
                if grid[i][j] == res[0]:
                    return res
        return [res[1], res[0]]

    # 2966. 划分数组并满足最大差限制 (Divide Array Into Arrays With Max Difference)
    def divideArray(self, nums: List[int], k: int) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        res = []
        for i in range(0, n, 3):
            if nums[i + 2] - nums[i] > k:
                return []
            res.append([nums[i], nums[i + 1], nums[i + 2]])
        return res

    # 2967. 使数组成为等数数组的最小代价 (Minimum Cost to Make Array Equalindromic)
    def minimumCost(self, nums: List[int]) -> int:
        def cal(mid: int, step: int, limit: int) -> int:
            while mid != limit:
                if check(mid):
                    return sum(abs(x - mid) for x in nums)
                mid += step
            return inf

        def check(num: int) -> bool:
            s = str(num)
            i = 0
            j = len(s) - 1
            while i < j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1
            return True

        nums.sort()
        n = len(nums)
        m = nums[n // 2]
        return min(cal(m, -1, 0), cal(m, 1, 10**9))

    # 1901. 寻找峰值 II (Find a Peak Element II)
    def findPeakGrid(self, mat: List[List[int]]) -> List[int]:
        left = 0
        right = len(mat) - 2
        while left <= right:
            i = left + ((right - left) >> 1)
            mx = max(mat[i])
            if mx > mat[i + 1][mat[i].index(mx)]:
                right = i - 1
            else:
                left = i + 1
        i = left
        return [i, mat[i].index(max(mat[i]))]

    # 1276. 不浪费原料的汉堡制作方案 (Number of Burgers with No Waste of Ingredients)
    def numOfBurgers(self, tomatoSlices: int, cheeseSlices: int) -> List[int]:
        if (
            tomatoSlices - 2 * cheeseSlices < 0
            or (tomatoSlices - 2 * cheeseSlices) % 2 != 0
            or cheeseSlices - (tomatoSlices - 2 * cheeseSlices) // 2 < 0
        ):
            return []
        return [
            (tomatoSlices - 2 * cheeseSlices) // 2,
            cheeseSlices - (tomatoSlices - 2 * cheeseSlices) // 2,
        ]

    # 2974. 最小数字游戏 (Minimum Number Game)
    def numberGame(self, nums: List[int]) -> List[int]:
        nums.sort()
        for i in range(1, len(nums), 2):
            nums[i], nums[i - 1] = nums[i - 1], nums[i]
        return nums

    # 2975. 移除栅栏得到的正方形田地的最大面积 (Maximum Square Area by Removing Fences From a Field)
    def maximizeSquareArea(
        self, m: int, n: int, hFences: List[int], vFences: List[int]
    ) -> int:
        def check(nums: List[int], n: int) -> set:
            nums.extend([1, n])
            nums.sort()
            s = set()
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    s.add(nums[j] - nums[i])
            return s

        h_set = check(hFences, m)
        v_set = check(vFences, n)
        res = 0
        for h in h_set:
            if h in v_set:
                res = max(res, h * h)
        MOD = 10**9 + 7
        return res % MOD if res else -1

    # 2976. 转换字符串的最小成本 I (Minimum Cost to Convert String I)
    def minimumCost(
        self,
        source: str,
        target: str,
        original: List[str],
        changed: List[str],
        cost: List[int],
    ) -> int:
        def dijkstra(start: int) -> List[int]:
            dis = [inf] * 26
            dis[start] = 0
            q = [(0, start)]
            heapq.heapify(q)
            while q:
                cur = heapq.heappop(q)
                d = cur[0]
                x = cur[1]
                for nxt in g[x]:
                    y = nxt[0]
                    dx = nxt[1]
                    if d + dx < dis[y]:
                        dis[y] = d + dx
                        heapq.heappush(q, (dis[y], y))
            return dis

        g = [[] for _ in range(26)]
        for ori, cha, cos in zip(original, changed, cost):
            g[ord(ori) - ord("a")].append((ord(cha) - ord("a"), cos))
        dis = [[inf] * 26 for _ in range(26)]
        for i in range(26):
            dis[i] = dijkstra(i)
        res = 0
        for s, t in zip(source, target):
            if dis[ord(s) - ord("a")][ord(t) - ord("a")] == inf:
                return -1
            res += dis[ord(s) - ord("a")][ord(t) - ord("a")]
        return res

    # 2971. 找到最大周长的多边形 (Find Polygon With the Largest Perimeter)
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        s = sum(nums)
        for i in range(len(nums) - 1, -1, -1):
            if s - nums[i] > nums[i]:
                return s
            s -= nums[i]
        return -1

    # 2973. 树中每个节点放置的金币数目 (Find Number of Coins to Place in Tree Nodes)
    def placedCoins(self, edges: List[List[int]], cost: List[int]) -> List[int]:
        def dfs(x: int, fa: int) -> List[int]:
            list = [cost[x]]
            for y in g[x]:
                if y != fa:
                    list.extend(dfs(y, x))
            list.sort()
            if len(list) >= 3:
                res[x] = max(
                    0, list[-1] * list[-2] * list[-3], list[0] * list[1] * list[-1]
                )
            if len(list) >= 5:
                list = list[:2] + list[-3:]
            return list

        n = len(cost)
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        res = [1] * n
        dfs(0, -1)
        return res

    # 2660. 保龄球游戏的获胜者 (Determine the Winner of a Bowling Game)
    def isWinner(self, player1: List[int], player2: List[int]) -> int:
        def check(nums: List[int]) -> int:
            res = 0
            for i, v in enumerate(nums):
                if i - 1 >= 0 and nums[i - 1] == 10 or i - 2 >= 0 and nums[i - 2] == 10:
                    res += v * 2
                else:
                    res += v
            return res

        res1 = check(player1)
        res2 = check(player2)
        if res1 == res2:
            return 0
        return 1 if res1 > res2 else 2

    # 2772. 使数组中的所有元素都等于零 (Apply Operations to Make All Array Elements Equal to Zero)
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        d = [0] * (n + 1)
        sum_d = 0
        for i, x in enumerate(nums):
            sum_d += d[i]
            x += sum_d
            if x == 0:
                continue
            if x < 0 or i + k > n:
                return False
            sum_d -= x
            d[i + k] += x
        return True

    # 2411. 按位或最大的最小子数组长度 (Smallest Subarrays With Maximum Bitwise OR)
    def smallestSubarrays(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [0] * n
        pos = [-1] * 31
        for i in range(n - 1, -1, -1):
            k = i
            for j in range(31):
                if nums[i] >> j & 1:
                    pos[j] = i
                if pos[j] != -1:
                    k = max(k, pos[j])
            res[i] = k - i + 1
        return res

    # 1185. 一周中的第几天 (Day of the Week)
    def dayOfTheWeek(self, day: int, month: int, year: int) -> str:
        list = [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ]
        mon = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        res = 4
        for i in range(1971, year):
            leap = i % 4 == 0 and i % 100 != 0 or i % 400 == 0
            res += 366 if leap else 365
        for i in range(1, month):
            res += mon[i - 1]
            if i == 2 and (year % 4 == 0 and year % 100 != 0 or year % 400 == 0):
                res += 1
        res += day
        return list[res % 7]

    # 1316. 不同的循环子字符串 (Distinct Echo Substrings)
    def distinctEchoSubstrings(self, text: str) -> int:
        def hash(l: int, r: int) -> int:
            return (pre[r + 1] - pre[l] * mul[r - l + 1]) % mod

        n = len(text)
        base, mod = 31, 10**9 + 7
        mul = [1] + [0] * n
        pre = [0] * (n + 1)
        for i, c in enumerate(text):
            pre[i + 1] = (pre[i] * base + ord(c) - ord("a")) % mod
            mul[i + 1] = mul[i] * base % mod
        res = 0
        seen = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                l = j - i
                if j + l <= n:
                    left_hash = hash(i, j - 1)
                    if left_hash not in seen[l - 1] and left_hash == hash(j, j + l - 1):
                        res += 1
                        seen[l - 1].add(left_hash)
        return res

    # 1937. 扣分后的最大得分 (Maximum Number of Points with Cost)
    def maxPoints(self, points: List[List[int]]) -> int:
        m = len(points)
        n = len(points[0])
        dp = [0] * n
        for i in range(m):
            best = -inf
            cur = [0] * n
            for j in range(n):
                best = max(best, dp[j] + j)
                cur[j] = best + points[i][j] - j
            best = -inf
            for j in range(n - 1, -1, -1):
                best = max(best, dp[j] - j)
                cur[j] = max(cur[j], best + points[i][j] + j)
            dp = cur
        return max(dp)

    # 1154. 一年中的第几天 (Day of the Year)
    def dayOfYear(self, date: str) -> int:
        y = int(date[:4])
        m = int(date[5:7])
        d = int(date[8:])
        leap_year = y % 4 == 0 and y % 100 != 0 or y % 400 == 0
        mon = [31, 28 + int(leap_year), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        return d + sum(mon[i - 1] for i in range(1, m))

    # 1599. 经营摩天轮的最大利润 (Maximum Profit of Operating a Centennial Wheel)
    def minOperationsMaxProfit(
        self, customers: List[int], boardingCost: int, runningCost: int
    ) -> int:
        i = 0
        round = 0
        waits = 0
        cost = 0
        max_cost = 0
        res = 0
        n = len(customers)
        while i < n or waits:
            round += 1
            if i < n:
                waits += customers[i]
            cost += min(waits, 4) * boardingCost - runningCost
            waits -= min(waits, 4)
            if cost > max_cost:
                max_cost = cost
                res = round
            i += 1
        return res if res else -1

    # 2980. 检查按位或是否存在尾随零 (Check if Bitwise OR Has Trailing Zeros)
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        return sum(int(x % 2 == 0) for x in nums) >= 2

    # 2981. 找出出现至少三次的最长特殊子字符串 I
    # 2982. 找出出现至少三次的最长特殊子字符串 II (Find Longest Special Substring That Occurs Thrice II) --分类统计
    def maximumLength(self, s: str) -> int:
        dic = collections.defaultdict(list)
        cnt = 0
        for i, c in enumerate(s):
            cnt += 1
            if i == len(s) - 1 or c != s[i + 1]:
                dic[c].append(cnt)
                cnt = 0
        res = 0
        for l in dic.values():
            l.extend([0, 0])
            l.sort(reverse=True)
            res = max(res, l[0] - 2, min(l[0] - 1, l[1]), l[2])
        return res if res else -1

    # 2981. 找出出现至少三次的最长特殊子字符串 I
    # 2982. 找出出现至少三次的最长特殊子字符串 II (Find Longest Special Substring That Occurs Thrice II) --二分 (超时)
    def maximumLength(self, s: str) -> int:
        def check(w: int) -> bool:
            dic = [0] * 26
            m = 0
            cnt = [0] * 26
            for i, c in enumerate(s):
                id = ord(c) - ord("a")
                cnt[id] += 1
                m |= 1 << id
                if i >= w:
                    id2 = ord(s[i - w]) - ord("a")
                    cnt[id2] -= 1
                    if cnt[id2] == 0:
                        m ^= 1 << id2
                if i >= w - 1:
                    if m & (m - 1) == 0:
                        dic[id] += 1
                        if dic[id] == 3:
                            return True
            return False

        res = 0
        left = 1
        right = len(s) - 2
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = mid
                left = mid + 1
            else:
                right = mid - 1
        return res if res else -1

    # 2799. 统计完全子数组的数目 (Count Complete Subarrays in an Array)
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        res = 0
        d = defaultdict(int)
        j = 0
        k = len(set(nums))
        for x in nums:
            d[x] += 1
            while len(d) == k:
                d[nums[j]] -= 1
                if d[nums[j]] == 0:
                    del d[nums[j]]
                j += 1
            res += j
        return res

    # 2825. 循环增长使字符串子序列等于另一个字符串 (Make String a Subsequence Using Cyclic Increments)
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        n1 = len(str1)
        n2 = len(str2)
        i = 0
        j = 0
        while i < n1 and j < n2:
            if str1[i] == str2[j] or (ord(str1[i]) - ord("a") + 1) % 26 == ord(
                str2[j]
            ) - ord("a"):
                j += 1
            i += 1
        return j == n2

    # 2487. 从链表中移除节点 (Remove Nodes From Linked List)
    def removeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next

        def rev(node: Optional[ListNode]) -> Optional[ListNode]:
            pre = None
            cur = node
            while cur:
                nxt = cur.next
                cur.next = pre
                pre = cur
                cur = nxt
            return pre

        head = rev(head)
        node = head
        while node:
            nxt = node.next
            while nxt and nxt.val < node.val:
                nxt = nxt.next
            node.next = nxt
            node = node.next
        return rev(head)

    # 2397. 被列覆盖的最多行数 (Maximum Rows Covered by Columns)
    def maximumRows(self, matrix: List[List[int]], numSelect: int) -> int:
        m = len(matrix)
        n = len(matrix[0])
        res = 0
        row = [0] * m
        for i in range(m):
            for j in range(n):
                row[i] |= matrix[i][j] << j
        for i in range(1 << n):
            res = max(
                res,
                (
                    sum(row[j] & i == row[j] for j in range(m))
                    if i.bit_count() == numSelect
                    else 0
                ),
            )
        return res

    # 1944. 队列中可以看到的人数 (Number of Visible People in a Queue)
    def canSeePersonsCount(self, heights: List[int]) -> List[int]:
        n = len(heights)
        res = [0] * n
        st = []
        for i in range(n - 1, -1, -1):
            while st and heights[st[-1]] < heights[i]:
                st.pop()
                res[i] += 1
            if st:
                res[i] += 1
            st.append(i)
        return res

    # 2770. 达到末尾下标所需的最大跳跃次数 (Maximum Number of Jumps to Reach the Last Index)
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)

        @cache
        def dfs(i: int) -> int:
            if i == n - 1:
                return 0
            res = -inf
            for j in range(i + 1, n):
                if abs(nums[i] - nums[j]) <= target:
                    res = max(res, dfs(j) + 1)
            return res

        res = dfs(0)
        return res if res >= 0 else -1

    # 2807. 在链表中插入最大公约数 (Insert Greatest Common Divisors in Linked List)
    def insertGreatestCommonDivisors(
        self, head: Optional[ListNode]
    ) -> Optional[ListNode]:
        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next

        cur = head
        while cur and cur.next:
            nxt = cur.next
            node = ListNode(gcd(cur.val, nxt.val))
            cur.next = node
            node.next = nxt
            cur = nxt
        return head

    # 2771. 构造最长非递减子数组 (Longest Non-decreasing Subarray From Two Arrays)
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            res = 0
            if arr[i][0] >= arr[i - 1][j]:
                res = max(res, dfs(i + 1, 0) + 1)
            if arr[i][1] >= arr[i - 1][j]:
                res = max(res, dfs(i + 1, 1) + 1)
            return res

        n = len(nums1)
        if n == 1:
            return 1
        arr = [[a, b] for a, b in zip(nums1, nums2)]
        return max(max(dfs(i, 0) + 1, dfs(i, 1) + 1) for i in range(1, n))

    # 447. 回旋镖的数量 (Number of Boomerangs)
    def numberOfBoomerangs(self, points: List[List[int]]) -> int:
        res = 0
        for x1, y1 in points:
            dic = collections.defaultdict(int)
            for x2, y2 in points:
                d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
                # 考虑顺序
                res += dic[d] * 2
                dic[d] += 1
        return res

    # 2861. 最大合金数 (Maximum Number of Alloys)
    def maxNumberOfAlloys(
        self,
        n: int,
        k: int,
        budget: int,
        composition: List[List[int]],
        stock: List[int],
        cost: List[int],
    ) -> int:
        def check(target: int) -> bool:
            def ok(com: List[int]) -> int:
                return sum(
                    max(0, c * target - s) * co for c, s, co in zip(com, stock, cost)
                )

            return any(ok(com) <= budget for com in composition)

        left = 0
        right = 10**9
        res = -1
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = mid
                left = mid + 1
            else:
                right = mid - 1
        return res

    # 2781. 最长合法子字符串的长度 (Length of the Longest Valid Substring)
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        s = set(forbidden)
        res = 0
        n = len(word)
        left = 0
        for right in range(n):
            for l in range(right, max(right - 10, left - 1), -1):
                if word[l : right + 1] in s:
                    left = l + 1
                    break
            res = max(res, right - left + 1)
        return res

    # 2996. 大于等于顺序前缀和的最小缺失整数 (Smallest Missing Integer Greater Than Sequential Prefix Sum)
    def missingInteger(self, nums: List[int]) -> int:
        n = len(nums)
        s = set(nums)
        res = nums[0]
        for i in range(1, n):
            if nums[i] - nums[i - 1] != 1:
                break
            res += nums[i]
        while res in s:
            res += 1
        return res

    # 2997. 使数组异或和等于 K 的最少操作次数 (Minimum Number of Operations to Make Array XOR Equal to K)
    def minOperations(self, nums: List[int], k: int) -> int:
        for num in nums:
            k ^= num
        return k.bit_count()

    # 2998. 使 X 和 Y 相等的最少操作次数 (Minimum Number of Operations to Make X and Y Equal)
    @cache
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x <= y:
            return y - x
        return min(
            x - y,
            self.minimumOperationsToMakeEqual(x // 11, y) + x % 11 + 1,
            self.minimumOperationsToMakeEqual(x // 11 + 1, y) + 11 - x % 11 + 1,
            self.minimumOperationsToMakeEqual(x // 5, y) + x % 5 + 1,
            self.minimumOperationsToMakeEqual(x // 5 + 1, y) + 5 - x % 5 + 1,
        )

    # 2998. 使 X 和 Y 相等的最少操作次数 (Minimum Number of Operations to Make X and Y Equal)
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        def add(v: int) -> None:
            if v < y:
                nonlocal res
                res = min(res, step + 1 + y - v)
            elif v not in s:
                s.add(v)
                q.append(v)

        if x <= y:
            return y - x
        res = x - y
        q = collections.deque()
        q.append(x)
        s = set()
        s.add(x)
        step = 0
        while q:
            size = len(q)
            for _ in range(size):
                v = q.popleft()
                if v == y:
                    return min(res, step)
                if v % 5 == 0:
                    add(v // 5)
                if v % 11 == 0:
                    add(v // 11)
                add(v + 1)
                add(v - 1)
            step += 1
        return -1

    # 2999. 统计强大整数的数目 (Count the Number of Powerful Integers)
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def cal(num: int) -> int:
            @cache
            def dfs(i: int, is_limit: bool, is_num: bool) -> int:
                if i == n - len(s):
                    return not is_limit or int(s) <= int(arr[i:])
                res = 0
                if not is_num:
                    res = dfs(i + 1, False, False)
                up = int(arr[i]) if is_limit else 9
                for d in range(0 if is_num else 1, up + 1):
                    if d > limit:
                        break
                    res += dfs(i + 1, is_limit and up == d, True)
                return res

            arr = str(num)
            n = len(arr)
            if num < int(s):
                return 0
            return dfs(0, True, False)

        return cal(finish) - cal(start - 1)

    # 2696. 删除子串后的字符串最小长度 (Minimum String Length After Removing Substrings)
    def minLength(self, s: str) -> int:
        ss = ""
        for c in s:
            ss += c
            if len(ss) >= 2 and (ss.rfind("AB") >= 0 or ss.rfind("CD") >= 0):
                ss = ss[: len(ss) - 2]
        return len(ss)

    # 2645. 构造有效字符串的最少插入数 (Minimum Additions to Make Valid String)
    def addMinimum(self, word: str) -> int:
        n = len(word)
        i = 0
        j = 0
        s = "abc"
        res = 0
        while j < n:
            if s[i] != word[j]:
                res += 1
            else:
                j += 1
            i = (i + 1) % 3
        if word[-1] == "a":
            res += 2
        elif word[-1] == "b":
            res += 1
        return res

    # 3000. 对角线最长的矩形的面积 (Maximum Area of Longest Diagonal Rectangle)
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        res = 0
        d = 0
        for x, y in dimensions:
            if d < x * x + y * y:
                d = x * x + y * y
                res = x * y
            elif d == x * x + y * y:
                res = max(res, x * y)
        return res

    # 3001. 捕获黑皇后需要的最少移动次数 (Minimum Moves to Capture The Queen)
    def minMovesToCaptureTheQueen(
        self, a: int, b: int, c: int, d: int, e: int, f: int
    ) -> int:
        def check(start: tuple, obstacle: tuple, dx: int, dy: int) -> int:
            SIZE = 8
            x0 = start[0]
            y0 = start[1]
            f = False
            while 0 <= x0 + dx < SIZE and 0 <= y0 + dy < SIZE:
                x0 += dx
                y0 += dy
                if x0 == obstacle[0] and y0 == obstacle[1]:
                    f = True
                elif x0 == queen[0] and y0 == queen[1]:
                    return 1 + int(f)
            return inf

        rock = (a - 1), (b - 1)
        bishop = (c - 1), (d - 1)
        queen = (e - 1), (f - 1)
        res = inf
        diag_dirs = (1, -1), (-1, 1), (-1, -1), (1, 1)
        flat_dirs = (1, 0), (-1, 0), (0, 1), (0, -1)
        for dx, dy in diag_dirs:
            res = min(res, check(bishop, rock, dx, dy))
        for dx, dy in flat_dirs:
            res = min(res, check(rock, bishop, dx, dy))
        return min(res, 2)

    # 3002. 移除后集合的最多元素数 (Maximum Size of a Set After Removals)
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        s1 = set(nums1)
        s2 = set(nums2)
        i = 0
        n = len(nums1)
        while i < n and len(s1) > n // 2:
            if nums1[i] in s2:
                s1.discard(nums1[i])
            i += 1
        i = 0
        while i < n and len(s2) > n // 2:
            if nums2[i] in s1:
                s2.discard(nums2[i])
            i += 1
        if len(s1) <= n // 2 and len(s2) <= n // 2:
            return len(s1 | s2)
        return min(len(s1), n // 2) + min(len(s2), n // 2)

    # 3003. 执行操作后的最大分割数量 (Maximize the Number of Partitions After Operations)
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)

        @cache
        def dfs(i: int, mask: int, changed: bool) -> int:
            if i == n:
                return 1
            res = 0
            # 不变
            bits = mask | (1 << (ord(s[i]) - ord("a")))
            if bits.bit_count() > k:
                res = max(res, dfs(i + 1, 1 << (ord(s[i]) - ord("a")), changed) + 1)
            else:
                res = max(res, dfs(i + 1, bits, changed))
            # 变
            if not changed:
                for j in range(26):
                    bits = mask | (1 << j)
                    if bits.bit_count() <= k:
                        res = max(res, dfs(i + 1, bits, True))
                    else:
                        res = max(res, dfs(i + 1, 1 << j, True) + 1)
            return res

        return dfs(0, 0, False)

    # 2085. 统计出现过一次的公共字符串 (Count Common Words With One Occurrence)
    def countWords(self, words1: List[str], words2: List[str]) -> int:
        dic1 = collections.defaultdict(int)
        dic2 = collections.defaultdict(int)
        res = 0
        for w in words1:
            dic1[w] += 1
        for w in words2:
            dic2[w] += 1
        for k, v in dic1.items():
            if v == 1 and dic2[k] == 1:
                res += 1
        return res

    # 2182. 构造限制重复的字符串 (Construct String With Repeat Limit)
    def repeatLimitedString(self, s: str, repeatLimit: int) -> str:
        cnts = [0] * 26
        for c in s:
            cnts[ord(c) - ord("a")] += 1
        res = []
        for i in range(25, -1, -1):
            while cnts[i]:
                max = min(repeatLimit, cnts[i])
                res.append(chr(ord("a") + i) * max)
                cnts[i] -= max
                if cnts[i] == 0:
                    break
                for j in range(i - 1, -1, -1):
                    if cnts[j]:
                        cnts[j] -= 1
                        res.append(chr(ord("a") + j))
                        break
                else:
                    break
        return "".join(res)

    # 83. 删除排序链表中的重复元素 (Remove Duplicates from Sorted List)
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head
        while cur:
            val = cur.val
            tmp = cur
            while cur and cur.val == val:
                cur = cur.next
            tmp.next = cur
        return head

    # 82. 删除排序链表中的重复元素 II (Remove Duplicates from Sorted List II)
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next

        dummy = ListNode(0, head)
        cur = dummy
        while cur:
            tmp = cur
            cur = cur.next
            if cur and cur.next and cur.val == cur.next.val:
                val = cur.val
                while cur and cur.val == val:
                    cur = cur.next
                tmp.next = cur
                cur = tmp
        return dummy.next

    # 3005. 最大频率元素计数 (Count Elements With Maximum Frequency)
    def maxFrequencyElements(self, nums: List[int]) -> int:
        cnts = [0] * 101
        max = 0
        res = 0
        for num in nums:
            cnts[num] += 1
            if cnts[num] > max:
                max = cnts[num]
                res = cnts[num]
            elif cnts[num] == max:
                res += cnts[num]
        return res

    # 3007. 价值和小于等于 K 的最大数字 (Maximum Number That Sum of the Prices Is Less Than or Equal to K)
    def findMaximumNumber(self, k: int, x: int) -> int:
        def check(num: int) -> int:
            s = bin(num)[2:]
            n = len(s)

            @cache
            def dfs(i: int, j: int, is_limit: bool, is_num: bool) -> int:
                if i == n:
                    return j if is_num else 0
                res = 0
                if not is_num:
                    res = dfs(i + 1, j, False, False)
                up = int(s[i]) if is_limit else 1
                for d in range(0 if is_num else 1, up + 1):
                    res += dfs(
                        i + 1,
                        j + int(d == 1 and (n - i) % x == 0),
                        is_limit and d == up,
                        True,
                    )
                return res

            return dfs(0, 0, True, False)

        left = 1
        right = 10**15
        res = 1
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid) <= k:
                res = mid
                left = mid + 1
            else:
                right = mid - 1
        return res

    # 2901. 最长相邻不相等子序列 II (Longest Unequal Adjacent Groups Subsequence II)
    def getWordsInLongestSubsequence(
        self, n: int, words: List[str], groups: List[int]
    ) -> List[str]:
        def check(s1: str, s2: str) -> bool:
            return (
                len(s1) == len(s2) and sum(int(c1 != c2) for c1, c2 in zip(s1, s2)) == 1
            )

        mx = n - 1
        f = [0] * n
        from_idx = [0] * n
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                if f[j] > f[i] and groups[i] != groups[j] and check(words[i], words[j]):
                    f[i] = f[j]
                    from_idx[i] = j
            f[i] += 1
            if f[i] > f[mx]:
                mx = i
        res = []
        m = f[mx]
        for i in range(m):
            res.append(words[mx])
            mx = from_idx[mx]
        return res

    # 2901. 最长相邻不相等子序列 II (Longest Unequal Adjacent Groups Subsequence II)
    def getWordsInLongestSubsequence(self, words: List[str], groups: List[int]) -> List[str]:
        def make_ans(i: int, j: int, mx: int) -> None:
            if i == n:
                return
            if mx == dfs(i + 1, j):
                make_ans(i + 1, j, mx)
                return
            res.append(words[i])
            make_ans(i + 1, i, mx - 1)
        def is_legal(i: int, j: int) -> bool:
            if len(words[i]) != len(words[j]):
                return False
            if groups[i] == groups[j]:
                return False
            return sum(x != y for x, y in zip(words[i], words[j])) == 1
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            res = dfs(i + 1, j)
            if j == n or is_legal(i, j):
                res = max(res, dfs(i + 1, i) + 1)
            return res
        n = len(words)
        mx = dfs(0, n)
        res = []
        make_ans(0, n, mx)
        return res

    # 2744. 最大字符串配对数目 (Find Maximum Number of String Pairs)
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        s = set()
        res = 0
        for a, b in words:
            if b + a in s:
                res += 1
            s.add(a + b)
        return res

    # 1647. 字符频次唯一的最小删除次数 (Minimum Deletions to Make Character Frequencies Unique)
    def minDeletions(self, s: str) -> int:
        cnts = [0] * 26
        for c in s:
            cnts[ord(c) - ord("a")] += 1
        cnts.sort()
        res = 0
        for i in range(24, -1, -1):
            if cnts[i] >= cnts[i + 1]:
                tmp = cnts[i]
                cnts[i] = max(0, cnts[i + 1] - 1)
                res += tmp - cnts[i]
        return res

    # 2812. 找出最安全路径 (Find the Safest Path in a Grid)
    def maximumSafenessFactor(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if grid[0][0] or grid[n - 1][n - 1]:
            return 0
        dis = [[inf] * n for _ in range(n)]
        q = []
        for i in range(n):
            for j in range(n):
                if grid[i][j]:
                    dis[i][j] = 0
                    q.append((i, j))
        dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        while q:
            size = len(q)
            for _ in range(size):
                (x, y) = q.pop(0)
                for dx, dy in dirs:
                    nx = x + dx
                    ny = y + dy
                    if nx >= 0 and nx < n and ny >= 0 and ny < n and dis[nx][ny] == inf:
                        dis[nx][ny] = dis[x][y] + 1
                        q.append((nx, ny))
        vis = [[False] * n for _ in range(n)]
        res = dis[0][0]
        q = []
        q.append((-dis[0][0], 0, 0))
        heapq.heapify(q)
        while q:
            (d, x, y) = heapq.heappop(q)
            if vis[x][y]:
                continue
            res = min(res, -d)
            if x == n - 1 and y == n - 1:
                return res
            vis[x][y] = True
            for dx, dy in dirs:
                nx = x + dx
                ny = y + dy
                if nx >= 0 and nx < n and ny >= 0 and ny < n:
                    heapq.heappush(q, (-dis[nx][ny], nx, ny))
        return res

    # 2171. 拿出最少数目的魔法豆 (Removing Minimum Number of Magic Beans)
    def minimumRemoval(self, beans: List[int]) -> int:
        beans.sort()
        s = sum(beans)
        n = len(beans)
        return min(s - (n - i) * v for i, v in enumerate(beans))

    # 2788. 按分隔符拆分字符串 (Split Strings by Separator)
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        res = []
        for w in words:
            res.extend(sub for sub in w.split(separator) if len(sub))
        return res

    # 2810. 故障键盘 (Faulty Keyboard)
    def finalString(self, s: str) -> str:
        res = []
        rev = True
        for c in s:
            if c == "i":
                rev = not rev
            elif rev:
                res.append(c)
            else:
                res.insert(0, c)
        return "".join(res if rev else reversed(res))

    # 410. 分割数组的最大值 (Split Array Largest Sum)
    def splitArray(self, nums: List[int], k: int) -> int:
        def check(s: int) -> int:
            cnt = 0
            cur_s = 0
            for num in nums:
                cur_s += num
                if cur_s > s:
                    cnt += 1
                    cur_s = num
            return cnt + 1

        left = max(nums)
        right = sum(nums)
        res = -1
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid) <= k:
                res = mid
                right = mid - 1
            else:
                left = mid + 1
        return res

    # 410. 分割数组的最大值 (Split Array Largest Sum)
    def splitArray(self, nums: List[int], k: int) -> int:
        # @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0 if j == k else inf
            if j == k:
                return inf
            if memo[i][j] != -1:
                return memo[i][j]
            s = 0
            res = inf
            for x in range(i, n - k + j + 1):
                s += nums[x]
                res = min(res, max(s, dfs(x + 1, j + 1)))
            memo[i][j] = res
            return res

        n = len(nums)
        memo = [[-1] * k for _ in range(n)]
        return dfs(0, 0)

    # 670. 最大交换 (Maximum Swap)
    def maximumSwap(self, num: int) -> int:
        s = str(num)
        n = len(s)
        arr = [0] * n
        for i in range(n):
            arr[i] = s[i]
        for i in range(n):
            mx = arr[i]
            idx = -1
            for j in range(i + 1, n):
                if arr[j] >= mx:
                    mx = arr[j]
                    idx = j
            if idx != -1 and mx != arr[i]:
                arr[i], arr[idx] = arr[idx], arr[i]
                break
        return int("".join(arr))

    # 2765. 最长交替子数组 (Longest Alternating Subarray)
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        res = -1
        i = 0
        while i < n:
            d = 1
            j = i + 1
            while j < n and nums[j] - nums[j - 1] == d:
                d = -d
                j += 1
            if j - i > 1:
                res = max(res, j - i)
                i = j - 1
            else:
                i = j
        return res

    # 2859. 计算 K 置位下标对应元素的和 (Sum of Values at Indices With K Set Bits)
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        return sum(v if i.bit_count() == k else 0 for i, v in enumerate(nums))

    # 3010. 将数组分成最小总代价的子数组 I
    def minimumCost(self, nums: List[int]) -> int:
        min1 = inf
        min2 = inf
        for i in range(1, len(nums)):
            if nums[i] <= min1:
                min2 = min1
                min1 = nums[i]
            elif nums[i] <= min2:
                min2 = nums[i]
        return nums[0] + min1 + min2

    # 3011. 判断一个数组是否可以变为有序 (Find if Array Can Be Sorted)
    def canSortArray(self, nums: List[int]) -> bool:
        n = len(nums)
        i = 0
        pre_max = -inf
        while i < n:
            _min = nums[i]
            _max = nums[i]
            j = i
            while j < n and nums[j].bit_count() == nums[i].bit_count():
                _min = min(_min, nums[j])
                _max = max(_max, nums[j])
                j += 1
            if _min < pre_max:
                return False
            pre_max = _max
            i = j
        return True

    # 3012. 通过操作使数组长度最小 (Minimize Length of Array Using Operations)
    def minimumArrayLength(self, nums: List[int]) -> int:
        mn = min(nums)
        for num in nums:
            if num % mn:
                return 1
        cnt = 0
        for num in nums:
            if num == mn:
                cnt += 1
        return (cnt + 1) // 2

    # 3014. 输入单词需要的最少按键次数 I (Minimum Number of Pushes to Type Word I)
    def minimumPushes(self, word: str) -> int:
        return sum(i // 8 + 1 for i in range(len(word)))

    # 3016. 输入单词需要的最少按键次数 II (Minimum Number of Pushes to Type Word II)
    def minimumPushes(self, word: str) -> int:
        cnt = [0] * 26
        for c in word:
            cnt[ord(c) - ord("a")] += 1
        cnt.sort(reverse=True)
        res = 0
        for i in range(26):
            res += cnt[i] * (i // 8 + 1)
        return res

    # 3015. 按距离统计房屋对数目 I (Count the Number of Houses at a Certain Distance I)
    def countOfPairs(self, n: int, x: int, y: int) -> List[int]:
        def check(start: int) -> List[int]:
            dis = [-1] * n
            dis[start] = 0
            q = collections.deque()
            q.append(start)
            while q:
                size = len(q)
                for _ in range(size):
                    x = q.popleft()
                    for y in g[x]:
                        if dis[y] == -1:
                            dis[y] = dis[x] + 1
                            q.append(y)
            return dis

        g = [[] for _ in range(n)]
        for i in range(1, n):
            g[i - 1].append(i)
            g[i].append(i - 1)
        if abs(x - y) > 1:
            g[x - 1].append(y - 1)
            g[y - 1].append(x - 1)
        res = [0] * n
        for i in range(n):
            for d in check(i):
                if d - 1 >= 0:
                    res[d - 1] += 1
        return res

    # 514. 自由之路 (Freedom Trail)
    def findRotateSteps(self, ring: str, key: str) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == len(key):
                return 0
            if key[i] == ring[j]:
                return dfs(i + 1, j) + 1
            res = inf
            for k in g[ord(key[i]) - ord("a")]:
                step = min(abs(j - k), len(ring) - abs(j - k)) + 1
                res = min(res, dfs(i + 1, k) + step)
            return res

        g = [[] for _ in range(26)]
        for i, v in enumerate(ring):
            g[ord(v) - ord("a")].append(i)
        return dfs(0, 0)

    # 3019. 按键变更的次数 (Number of Changing Keys)
    def countKeyChanges(self, s: str) -> int:
        return sum(x != y for x, y in pairwise(s.lower()))

    # 3019. 按键变更的次数 (Number of Changing Keys)
    def countKeyChanges(self, s: str) -> int:
        return sum((ord(s[i]) & 31) != (ord(s[i - 1]) & 31) for i in range(1, len(s)))

    # 3020. 子集中元素的最大数量 (Find the Maximum Number of Elements in Subset)
    def maximumLength(self, nums: List[int]) -> int:
        # 若使用 collections.defaultDict(int) 就会报错
        dic = Counter(nums)
        res = dic[1] - ((dic[1] % 2) ^ 1)
        del dic[1]
        for k in dic:
            cur = 0
            while dic[k] >= 2:
                cur += 2
                k *= k
            if dic[k] >= 1:
                res = max(res, cur + 1)
            else:
                res = max(res, cur - 1)
        return res

    # 3021. Alice 和 Bob 玩鲜花游戏 (Alice and Bob Playing Flower Game)
    def flowerGame(self, n: int, m: int) -> int:
        even1 = n // 2
        even2 = m // 2
        return (n - even1) * even2 + even1 * (m - even2)

    # 2808. 使循环数组所有元素相等的最少秒数 (Minimum Seconds to Equalize a Circular Array)
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        g = collections.defaultdict(list)
        for i, v in enumerate(nums):
            g[v].append(i)
        res = inf
        for _list in g.values():
            cur = (n - _list[-1] + _list[0]) // 2
            for x, y in pairwise(_list):
                cur = max(cur, (y - x) // 2)
            res = min(res, cur)
        return res

    # 2761. 和等于目标值的质数对 (Prime Pairs With Target Sum)
    def findPrimePairs(self, n: int) -> List[List[int]]:
        res = []
        primes = [True] * (n + 1)
        primes[1] = False
        for i in range(2, n + 1):
            if primes[i]:
                for j in range(i * i, n + 1, i):
                    primes[j] = False
        if n % 2:
            return [[2, n - 2]] if n > 4 and primes[n - 2] else []
        for i in range(2, n // 2 + 1):
            if primes[i] and primes[n - i]:
                res.append([i, n - i])
        return res

    # 2670. 找出不同元素数目差数组 (Find the Distinct Difference Array)
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        cnts = [0] * 51
        n = len(nums)
        suf = 0
        for num in nums:
            cnts[num] += 1
            if cnts[num] == 1:
                suf += 1
        res = [0] * n
        cur_cnts = [0] * 51
        pre = 0
        for i, v in enumerate(nums):
            cur_cnts[v] += 1
            if cur_cnts[v] == 1:
                pre += 1
            cnts[v] -= 1
            if cnts[v] == 0:
                suf -= 1
            res[i] = pre - suf
        return res

    # 2862. 完全子集的最大元素和 (Maximum Element-Sum of a Complete Subset of Indices)
    def maximumSum(self, nums: List[int]) -> int:
        # core(n) : n中除去完全平方数后的值
        @cache
        def core(n: int) -> int:
            res = 1
            for i in range(2, isqrt(n) + 1):
                e = 0
                while n % i == 0:
                    e ^= 1
                    n //= i
                if e:
                    res *= i
            if n > 1:
                res *= n
            return res

        cnt = [0] * (len(nums) + 1)
        for i, v in enumerate(nums, 1):
            cnt[core(i)] += v
        return max(cnt)

    # 2789. 合并后数组中的最大元素 (Largest Element in an Array after Merge Operations)
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        res = nums[-1]
        for i in range(n - 2, -1, -1):
            if nums[i] <= nums[i + 1]:
                nums[i] += nums[i + 1]
            res = max(res, nums[i])
        return res

    # 292. Nim 游戏 (Nim Game)
    def canWinNim(self, n: int) -> bool:
        return bool(n % 4)

    # 1690. 石子游戏 VII (Stone Game VII)
    def stoneGameVII(self, stones: List[int]) -> int:
        n = len(stones)
        pre = list(accumulate(stones, initial=0))

        @cache
        def dfs(i: int, j: int) -> int:
            if i == j:
                return 0
            return max(
                pre[j + 1] - pre[i + 1] - dfs(i + 1, j), pre[j] - pre[i] - dfs(i, j - 1)
            )

        res = dfs(0, n - 1)
        dfs.cache_clear()
        return res

    # 2932. 找出强数对的最大异或值 I (Maximum Strong Pair XOR I)
    # 2935. 找出强数对的最大异或值 II (Maximum Strong Pair XOR II)
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        nums.sort()
        mask = 0
        res = 0
        d = dict()
        for i in range(nums[-1].bit_length() - 1, -1, -1):
            d.clear()
            mask |= 1 << i
            new_res = res | (1 << i)
            for y in nums:
                mask_y = y & mask
                if mask_y ^ new_res in d and d[mask_y ^ new_res] * 2 >= y:
                    res = new_res
                    break
                d[mask_y] = y
        return res

    # 1696. 跳跃游戏 VI (Jump Game VI)
    def maxResult(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        q = collections.deque()
        q.append(0)
        for i in range(1, n):
            while q and i - q[0] > k:
                q.popleft()
            if q:
                dp[i] = dp[q[0]] + nums[i]
            while q and dp[q[-1]] <= dp[i]:
                q.pop()
            q.append(i)
        return dp[-1]

    # 1686. 石子游戏 VI (Stone Game VI)
    def stoneGameVI(self, aliceValues: List[int], bobValues: List[int]) -> int:
        c = [(a + b, i) for i, (a, b) in enumerate(zip(aliceValues, bobValues))]
        c.sort(reverse=True)
        x = sum(aliceValues[i] for _, i in c[::2])
        y = sum(bobValues[i] for _, i in c[1::2])
        return 1 if x > y else (0 if x == y else -1)

    # 3024. 三角形类型 II (Type of Triangle II)
    def triangleType(self, nums: List[int]) -> str:
        nums.sort()
        if nums[0] + nums[1] <= nums[2]:
            return "none"
        if nums[0] == nums[1] == nums[2]:
            return "equilateral"
        if nums[0] == nums[1] or nums[1] == nums[2]:
            return "isosceles"
        return "scalene"

    # 3025. 人员站位的方案数 I (Find the Number of Ways to Place People I)
    # 3027. 人员站位的方案数 II (Find the Number of Ways to Place People II)
    def numberOfPairs(self, points: List[List[int]]) -> int:
        points.sort(key=lambda p: (p[0], -p[1]))
        res = 0
        for i, (_, y) in enumerate(points):
            max_y = -inf
            for _, y0 in points[i + 1 :]:
                if max_y < y0 <= y:
                    res += 1
                    max_y = y0
        return res

    # 3026. 最大好子数组和 (Maximum Good Subarray Sum)
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        d = dict()
        s = list(accumulate(nums, initial=0))
        res = -inf
        for i, v in enumerate(nums):
            if v in d:
                res = max(res, s[i + 1] - s[d[v]])
            if v - k not in d or s[d[v - k]] > s[i]:
                d[v - k] = i
            if v + k not in d or s[d[v + k]] > s[i]:
                d[v + k] = i
        return 0 if res == -inf else res

    # 3028. 边界上的蚂蚁 (Ant on the Boundary)
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        return list(accumulate(nums)).count(0)

    # 3029. 将单词恢复初始状态所需的最短时间 I (Minimum Time to Revert Word to Initial State I)
    # 3031. 将单词恢复初始状态所需的最短时间 II (Minimum Time to Revert Word to Initial State II) --z函数
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        z = [0] * n
        left = 0
        right = 0
        for i in range(1, n):
            if i <= right:
                z[i] = min(z[i - left], right - i + 1)
            while i + z[i] < n and word[z[i]] == word[i + z[i]]:
                left, right = i, i + z[i]
                z[i] += 1
            if i % k == 0 and z[i] == n - i:
                return i // k
        return (n - 1) // k + 1

    # 239. 滑动窗口最大值 (Sliding Window Maximum)
    # LCR 183. 望远镜中最高的海拔
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        q = deque()
        n = len(nums)
        res = []
        for i in range(n):
            while q and i - q[0] >= k:
                q.popleft()
            while q and nums[q[-1]] <= nums[i]:
                q.pop()
            q.append(i)
            if i >= k - 1:
                res.append(nums[q[0]])
        return res

    # 145. 二叉树的后序遍历 (Binary Tree Postorder Traversal)
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []

        def dfs(root: "TreeNode") -> None:
            if root is None:
                return
            dfs(root.left)
            dfs(root.right)
            res.append(root.val)

        dfs(root)
        return res

    # 706. 设计哈希映射 (Design HashMap)
    class MyHashMap:

        def __init__(self):
            self.map = [-1] * (10**6 + 1)

        def put(self, key: int, value: int) -> None:
            self.map[key] = value

        def get(self, key: int) -> int:
            return self.map[key]

        def remove(self, key: int) -> None:
            self.map[key] = -1

    # 2145. 统计隐藏数组数目 (Count the Hidden Sequences)
    def numberOfArrays(self, differences: List[int], lower: int, upper: int) -> int:
        cur1 = upper
        cur2 = lower
        mx = inf
        mi = -inf
        for x in differences:
            cur1 += x
            cur2 += x
            mx = min(mx, min(upper, upper - (cur1 - upper)))
            mi = max(mi, max(lower, lower + (lower - cur2)))
            if mx < mi:
                return 0
        return mx - mi + 1
