from collections import Counter
import collections
from functools import cache
from itertools import accumulate, islice
from math import comb, gcd, inf, isnan, sqrt
from multiprocessing import reduction
from operator import le
from pydoc import resolve
from queue import PriorityQueue
from re import S, X
import re
from tabnanny import check
from textwrap import indent
from tkinter import NO, W
from tkinter.messagebox import RETRY
from tkinter.tix import Tree
from turtle import reset, st
from typing import List, Optional
import heapq
import bisect

class luogu1:

    # P1122 最大子树和
    def maxSubTreeSum(self, n: int, exponent: List[int], edges: List[List[int]]) -> int:
        def dfs(x: int, fa: int) -> int:
            s = exponent[x]
            for y in g[x]:
                if y != fa:
                    s += dfs(y, x)
            nonlocal res
            res = max(res, s)
            return max(s, 0)
        res = -inf
        g = [[] * n for _ in range(n)]
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
        dfs(0, -1)
        return res
    
    # P1042 [NOIP2003 普及组] 乒乓球
    def tableTennisResults(self, s: str) -> List[List[str]]:
        def get_res(s: str, limit: int) -> List[str]:
            res = []
            a = 0
            b = 0
            for c in s:
                if c == 'W':
                    a += 1
                else:
                    b += 1
                if (a >= limit or b >= limit) and abs(a - b) >= 2:
                    res.append(str(a) + ':' + str(b))
                    a = 0
                    b = 0
            if a or b:
                res.append(str(a) + ':' + str(b))
            return res
        res = []
        res.append(get_res(s, 11))
        res.append(get_res(s, 21))
        return res
    
    #  P1831 杠杆数
    def leverNumber(self, left: int, right: int) -> int:
        def solve(num: str) -> int:
            @cache
            def dfs(i: int, j: int, k: int, is_limit: bool, is_num: bool) -> int:
                if i == n:
                    return is_num and k == 0
                if k < 0:
                    return 0
                res = 0
                if not is_num:
                    res = dfs(i + 1, j, k, False, False)
                up = int(num[i]) if is_limit else 9
                for d in range(0 if is_num else 1, up + 1):
                    res += dfs(i + 1, j, k + (j - i) * d, is_limit and d == up, True)
                return res
            n = len(num)
            return sum(dfs(0, i, 0, True, False) for i in range(n))
        return solve(str(right)) - solve(str(left - 1))
    
    # P6754 [BalticOI 2013 Day1] Palindrome-Free Numbers
    def palindromeFreeNumbers(self, left: int, right: int) -> int:
        def solve(num: str) -> int:
            @cache
            def dfs(i: int, j: int, k: int, is_limit: bool, is_num: bool) -> int:
                if i == n:
                    return is_num
                res = 0
                if not is_num:
                    res = dfs(i + 1, j, k, False, False)
                up = int(num[i]) if is_limit else 9
                for d in range(0 if is_num else 1, up + 1):
                    if d != j and d != k:
                        res += dfs(i + 1, d, j, is_limit and d == up, True)
                return res
            n = len(num)
            return dfs(0, -1, -1, True, False)
        def check(num: str) -> bool:
            n = len(num)
            for i in range(1, n):
                if num[i] == num[i - 1] or i >= 2 and num[i] == num[i - 2]:
                    return False
            return True
        return solve(str(right)) - solve(str(left)) + check(str(left))
    
    # P2602 [ZJOI2010] 数字计数
    def calculateEachDigitsCounts(self, left: int, right: int) -> List[int]:
        def cal(s: str, num: int) -> int:
            @cache
            def dfs(i: int, j: int, is_limit: bool, is_num: bool):
                if i == n:
                    return j if is_num else 0
                res = 0
                if not is_num:
                    res = dfs(i + 1, j, False, False)
                up = int(s[i]) if is_limit else 9
                for d in range(0 if is_num else 1, up + 1):
                    res += dfs(i + 1, j + int(d == num), is_limit and d == up, True)
                return res
            n = len(s)
            return dfs(0, 0, True, False)
        res = [0] * 10
        for i in range(10):
            res[i] = cal(str(right), i) - cal(str(left - 1), i)
        return res
    
    #  P4999 烦人的数学作业
    def annoyingMathWork(self, left: int, right: int) -> int:
        def solve(s: str) -> int:
            @cache
            def dfs(i: int, j: int, is_limit: bool, is_num: bool) -> int:
                if i == n:
                    return j if is_num else 0
                res = 0
                if not is_num:
                    res = dfs(i + 1, j, False, False)
                up = int(s[i]) if is_limit else 9
                for d in range(0 if is_num else 1, up + 1):
                    res += dfs(i + 1, j + d, is_limit and up == d, True)
                    res %= MOD
                return res
            n = len(s)
            return dfs(0, 0, True, False)
        MOD = 10 ** 9 + 7
        return (solve(str(right)) - solve(str(left - 1))) % MOD
    
    # P8764 [蓝桥杯 2021 国 BC] 二进制问题
    def binaryProblem(self, n: int, k: int) -> int:
        def dfs(i: int, j: int, is_limit: bool, is_num: bool) -> int:
            if i == n:
                return int(is_num and j == k)
            res = 0
            if not is_num:
                res = dfs(i + 1, j, False, False)
            up = int(s[i]) if is_limit else 1
            for d in range(0 if is_num else 1, up + 1):
                if j + d <= k:
                    res += dfs(i + 1, j + d, is_limit and up == d, True)
            return res
        s = bin(n)[2:]
        n = len(s)
        return dfs(0, 0, True, False)
    
    # P1048 [NOIP2005 普及组] 采药
    def gatherHerbs(herbs: List[List[int]], t: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            res = dfs(i + 1, j)
            if j + herbs[i][0] <= t:
                res = max(res, dfs(i + 1, j + herbs[i][0]) + herbs[i][1])
            return res
        n = len(herbs)
        return dfs(0, 0)
    
    # P2657 [SCOI2009] windy 数
    def windyNumbers(self, a: int, b: int) -> int:
        def solve(s: str) -> int:
            @cache
            def dfs(i: int, j: int, is_limit: bool, is_num: bool):
                if i == n:
                    return int(is_num)
                res = 0
                if not is_num:
                    res = dfs(i + 1, j, False, False)
                up = int(s[i]) if is_limit else 9
                for d in range(0 if is_num else 1, up + 1):
                    if not is_num or abs(d - j) >= 2:
                        res += dfs(i + 1, d, is_limit and up == d, True)
                return res
            n = len(s)
            return dfs(0, 0, True, False)
        return solve(str(b)) - solve(str(a - 1))

    # P6218 [USACO06NOV] Round Numbers S
    def roundNumbers(self, left: int, right: int) -> int:
        def solve(s: str) -> int:
            @cache
            def dfs(i: int, j: int, is_limit: bool, is_num: bool) -> int:
                if i == n:
                    return 1 if is_num and j >= 0 else 0
                res = 0
                if not is_num:
                    res = dfs(i + 1, j, False, False)
                up = int(s[i]) if is_limit else 1
                for d in range(0 if is_num else 1, up + 1):
                    res += dfs(i + 1, j + (-2) * d + 1, is_limit and up == d, True)
                return res
            n = len(s)
        return solve(bin(right)[2:]) - solve(bin(left - 1)[2:])
    
    # P4317 花神的数论题
    def flowerGodNumTheory(self, N: int) -> int:
        @cache
        def dfs(i: int, j: int, is_limit: bool, is_num: bool) -> int:
            if i == n:
                return j if is_num else 1
            res = 0
            if not is_num:
                res *= dfs(i + 1, j, False, False)
            up = int(s[i]) if is_limit else 1
            for d in range(0 if is_num else 1, up + 1):
                res *= dfs(i + 1, j + d, is_limit and d == up, True)
            res %= MOD
            return res
        MOD = 10 ** 7 + 7
        s = bin(N)[2:]
        n = len(s)
        return dfs(0, 0, True, False)
    

    # P8625 [蓝桥杯 2015 省 B] 生命之树
    def treeOfLife(self, n: int, values: List[int], edges: List[List[int]]) -> int:
        def dfs(x: int, fa: int) -> int:
            max = values[x]
            for y in g[x]:
                if y != fa:
                    max += dfs(y, x)
            nonlocal res
            res = max(res, max)
            return max(0, max)
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u - 1].append(v - 1)
            g[v - 1].append(u - 1)
        res = 0
        dfs(0, -1)
        return res
    
    # P1103 书本整理
    def booksTipUp(self, n: int, k: int, books: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int, pre: int) -> int:
            if i == n:
                return 0
            res = dfs(i + 1, j, books[i][1]) + (0 if pre == -1 else abs(books[i][1] - pre))
            if j < k:
                res = min(res, dfs(i + 1, j + 1, pre))
        books.sort(key=lambda o: o[0])
        return dfs(0, 0, -1)
    
    # P1140 相似基因
    def similarGenes(self, s1: str, s2: str) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n1 and j == n2:
                return 0
            if i == n1:
                return dfs(i, j + 1) + similar[dic[s2[j]]][4]
            if j == n2:
                return dfs(i, j + 1) + similar[dic[s1[i]]][4]
            if s1[i] == s2[j]:
                return dfs(i + 1, j + 1) + 5
            return max(dfs(i + 1, j + 1) + similar[dic[s1[i]]][dic[s2[j]]], dfs(i + 1, j) + similar[dic[s1[i]]][4], dfs(i, j + 1) + similar[dic[s2[j]]][4])
        n1 = len(s1)
        n2 = len(s2)
        dic = collections.defaultdict(int)
        dic['A'] = 0
        dic['C'] = 1
        dic['G'] = 2
        dic['T'] = 3
        similar = [[0] * 5 for _ in range(5)]
        similar[0][0] = similar[1][1] = similar[2][2] = similar[3][3] = 5
        similar[0][1] = similar[1][0] = similar[0][3] = similar[3][0] = similar[3][4] = similar[4][3] = -1
        similar[0][2] = similar[2][0] = similar[1][3] = similar[3][1] = similar[2][3] = similar[3][2] = similar[2][4] = similar[4][2] = -2
        similar[1][2] = similar[2][1] = similar[0][4] = similar[4][0] = -3
        similar[1][4] = similar[4][1] = -4
        return dfs(0, 0)
    
    # P1754 球迷购票问题
    def buyTicketsProblem(self, n: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n or j == n:
                return 1
            res = dfs(i + 1, j)
            if i > j:
                res += dfs(i, j + 1)
        return dfs(0, 0)
    

    # P1364 医院设置
    def hospitalSetting(self, n: int, edges: List[List[int]]) -> int:
        def dfs(x: int, fa: int, d: int) -> None:
            s[x] == values[x]
            for y in g[x]:
                if y != fa:
                    dfs(y, x, d + 1)
                    s[x] += s[y]
            dp[0] += values[x] * d
        def reroot(x: int, fa: int) -> None:
            nonlocal res
            res = min(res, dp[x])
            for y in g[x]:
                if y != fa:
                    dp[y] = dp[x] - s[y] + s[0] - s[y]
                    reroot(y, x)
        g = [[] for _ in range(n)]
        values = [0] * n
        s = [0] * n
        dp = [0] * n
        for i, w, u, v in enumerate(edges):
            values[i] = w
            u -= 1
            if u >= 0:
                g[i].append(u)
            v -= 1
            if v >= 0:
                g[i].append(v)
        dfs(0, -1, 0)
        res = inf
        reroot(0, -1)
        return res

    # P1130 红牌
    def redCards(self, grid: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j == n:
                return 0
            return min(dfs((i + 1) % m, j + 1), dfs(i, j + 1)) + grid[i][j]
        m = len(grid)
        n = len(grid[0])
        return min(dfs(i, 0) for i in range(m))
    

    # P1002[NOIP2002 普及组]过河卒
    def acrossTheRiver(self, tx: int, ty: int, hx: int, hy: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i > tx or j > ty or (i, j) in s:
                return 0
            if i == tx and j == ty:
                return 1
            return dfs(i + 1, j) + dfs(i, j + 1)
        s = set()
        h = [[1, 2], [-1, 2], [1, -2], [-1, -2], [2, 1], [-2, 1], [2, -1], [-2, -1], [0, 0]]
        for dx, dy in h:
            s.add((hx + dx, hy + dy))
        return dfs(0, 0)
    
    # P4170 [CQOI2007] 涂色
    def paint(self, s: str) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == j:
                return 1
            if s[i] == s[j]:
                return min(dfs(i + 1, j), dfs(i, j - 1))
            res = inf
            for k in range(i, j):
                res = min(res, dfs(i + 1, k) + dfs(k + 1, j))
            return res
        return dfs(0, len(s) - 1)
    

    # P1115 最大子段和
    def maxSumOfSubsquence(self, nums: List[int]) -> int:
        pre = 0
        res = -inf
        min = 0
        for num in nums:
            pre += num
            res = max(res, pre - min)
            min = min(min, pre)
        return res
    
    # P8707 [蓝桥杯 2020 省 AB1] 走方格
    def suqareTicking(self, m: int, n: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == m + 1 or j == n + 1 or i % 2 == 0 and j % 2 == 0:
                return 0
            if i == m and j == n:
                return 1
            return dfs(i + 1, j) + dfs(i, j + 1)
        if m % 2 == 0 and n % 2 == 0:
            return 0
        return dfs(1, 1)

    # P1802 5 倍经验日
    def fiveTimesExperience(self, exp: int, opp: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            res = dfs(i + 1, j) + opp[i][0]
            if j >= opp[i][2]:
                res = max(res, dfs(i + 1, j - opp[i][2]) + opp[i][1])
            return res
        n = len(opp)
        return dfs(0, exp) * 5
    
    # P1091 [NOIP2004 提高组] 合唱队形
    def chorusFormation(self, arr: List[int]) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i == n:
                return 0
            # 出列
            res = dfs(i + 1, j, k)
            # 不出列
            if j == 0:
                res = max(res, dfs(i + 1, arr[i], 0) + 1)
            else:
                if k == 0:
                    if j < arr[i]:
                        res = max(res, dfs(i + 1, arr[i], 0) + 1)
                    elif j > arr[i]:
                        res = max(res, dfs(i + 1, arr[i], 1) + 1)
                else:
                    if j > arr[i]:
                        res = max(res, dfs(i + 1, arr[i], 1) + 1)
            return res
        n = len(arr)
        return n - dfs(0, 0, 0)

    











