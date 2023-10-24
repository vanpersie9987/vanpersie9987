from collections import Counter
import collections
from functools import cache
from itertools import accumulate, islice
from math import comb, gcd, inf, isnan, sqrt
from operator import le
from pydoc import resolve
from queue import PriorityQueue
from re import X
import re
from textwrap import indent
from tkinter import W
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
    def tableTennisResults(s: str) -> List[List[str]]:
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
    def leverNumber(left: int, right: int) -> int:
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
    def palindromeFreeNumbers(left: int, right: int) -> int:
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
    
    
        
        