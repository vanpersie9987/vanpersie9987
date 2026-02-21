from ast import Return, Tuple, literal_eval
from asyncio import FastChildWatcher
import csv
from curses.panel import bottom_panel
from doctest import FAIL_FAST
from gettext import find
import math
from platform import node
from posixpath import isabs
from pydoc import plain
from signal import valid_signals
from sqlite3 import paramstyle
import stat
from termios import CINTR
from tokenize import String
from unicodedata import numeric
from xxlimited import foo
from calendar import c
from collections import Counter, defaultdict, deque
import collections
from ctypes.wintypes import _ULARGE_INTEGER
from curses import can_change_color, curs_set, intrflush, nonl
from curses.ascii import SI, isalpha, isascii, isdigit, islower, isprint
from decimal import Rounded
import dis
import enum
from functools import cache, cached_property
from inspect import modulesbyfile
from itertools import (
    accumulate,
    combinations,
    count,
    islice,
    pairwise,
    permutations,
    starmap,
)
from locale import DAY_4
from logging import _Level, root
from math import comb, cos, e, fabs, floor, gcd, inf, isqrt, lcm, sqrt, ulp
from mimetypes import init
from multiprocessing import reduction
from operator import is_, le, ne, truediv
from os import eventfd, lseek, minor, name, pread
from pickletools import read_uint1
from queue import PriorityQueue
from re import A, L, S, T, X
import re
from socket import NI_NUMERICSERV
from ssl import VERIFY_X509_TRUSTED_FIRST
from string import ascii_lowercase
from tabnanny import check
from tarfile import tar_filter
from textwrap import indent
import time
from tkinter import LEFT, N, NO, W
from tkinter.messagebox import RETRY
from token import NL, RIGHTSHIFT
from turtle import (
    RawTurtle,
    color,
    heading,
    left,
    mode,
    pos,
    position,
    reset,
    right,
    rt,
    st,
    up,
    update,
)
from typing import List, Optional, Self
import heapq
import bisect
from unittest import result
from unittest.util import _count_diff_all_purpose
from wsgiref.simple_server import make_server
from wsgiref.util import guess_scheme, request_uri
from xml.dom import Node
from zoneinfo import reset_tzpath
import copy

# curl https://bootstrap.pypa.io/pip/get-pip.py -o get-pip.py
# sudo python3 get-pip.py
# pip3 install sortedcontainers
from sortedcontainers import SortedDict, SortedList, SortedSet


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


class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None


class leetcode_3:

    # 75. 颜色分类 (Sort Colors)
    def sortColors(self, nums: List[int]) -> None:
        l = 0
        r = len(nums) - 1
        p = 0
        while p <= r:
            if nums[p] == 2:
                nums[p], nums[r] = nums[r], nums[p]
                r -= 1
            elif nums[p] == 0:
                nums[p], nums[l] = nums[l], nums[p]
                p += 1
                l += 1
            else:
                p += 1
        return nums

    # 2131. 连接两字母单词得到的最长回文串 (Longest Palindrome by Concatenating Two Letter Words)
    def longestPalindrome(self, words: List[str]) -> int:
        d = [[0] * 26 for _ in range(26)]
        for a, b in words:
            d[ord(a) - ord("a")][ord(b) - ord("a")] += 1
        res = 0
        for i in range(26):
            for j in range(i, 26):
                if i == j:
                    res += d[i][j] // 2
                else:
                    res += min(d[i][j], d[j][i])
        return res * 4 + any(d[i][i] % 2 for i in range(26)) * 2

    # 1857. 有向图中最大颜色值 (Largest Color Value in a Directed Graph)
    def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:
        n = len(colors)
        g = [[] for _ in range(n)]
        indegree = [0] * n
        for u, v in edges:
            g[u].append(v)
            indegree[v] += 1
        queue = collections.deque()
        for i in range(n):
            if indegree[i] == 0:
                queue.append(i)
        count = 0
        dp = [[0] * 26 for _ in range(n)]
        while queue:
            count += 1
            u = queue.popleft()
            dp[u][ord(colors[u]) - ord("a")] += 1
            for v in g[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    queue.append(v)
                for i in range(26):
                    dp[v][i] = max(dp[v][i], dp[u][i])
        return -1 if count < n else max(max(dp[i]) for i in range(n))

    # 2359. 找到离给定两个节点最近的节点 (Find Closest Node to Given Two Nodes)
    def closestMeetingNode(self, edges: List[int], node1: int, node2: int) -> int:
        def cal(edges: List[int], start: int) -> List[int]:
            n = len(edges)
            dis = [n] * n
            dis[start] = 0
            x = start
            while edges[x] != -1 and dis[edges[x]] == n:
                dis[edges[x]] = dis[x] + 1
                x = edges[x]
            return dis

        n = len(edges)
        dis1 = cal(edges, node1)
        dis2 = cal(edges, node2)
        mx = n
        res = -1
        for i in range(n):
            dis = max(dis1[i], dis2[i])
            if dis < mx:
                mx = dis
                res = i
        return res

    # 909. 蛇梯棋 (Snakes and Ladders)
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        n = len(board)
        n_board = []
        d = 0
        for i in range(n - 1, -1, -1):
            if d == 0:
                for j in range(n):
                    n_board.append(board[i][j] - 1)
            else:
                for j in range(n - 1, -1, -1):
                    n_board.append(board[i][j] - 1)
            d ^= 1
        vis = [False] * (n * n)
        q = deque()
        q.append(0)
        step = 0
        vis[0] = True
        while q:
            sz = len(q)
            for _ in range(sz):
                x = q.popleft()
                if x == n * n - 1:
                    return step
                for j in range(x + 1, min(x + 7, n * n)):
                    if n_board[j] < 0 and not vis[j]:
                        vis[j] = True
                        q.append(j)
                    elif n_board[j] >= 0 and not vis[n_board[j]]:
                        nxt = n_board[j]
                        vis[nxt] = True
                        q.append(nxt)
            step += 1
        return -1

    # 3548. 等和矩阵分割 II (Equal Sum Grid Partition II)
    def canPartitionGrid(self, grid: List[List[int]]) -> bool:
        total = sum(sum(row) for row in grid)

        # 能否水平分割
        def check(a: List[List[int]]) -> bool:
            m, n = len(a), len(a[0])

            # 删除上半部分中的一个数，能否满足要求
            def f(a: List[List[int]]) -> bool:
                st = {0}  # 0 对应不删除数字
                s = 0
                for i, row in enumerate(a[:-1]):
                    for j, x in enumerate(row):
                        s += x
                        # 第一行，不能删除中间元素
                        if i > 0 or j == 0 or j == n - 1:
                            st.add(x)
                    # 特殊处理只有一列的情况，此时只能删除第一个数或者分割线上那个数
                    if n == 1:
                        if (
                            s * 2 == total
                            or s * 2 - total == a[0][0]
                            or s * 2 - total == row[0]
                        ):
                            return True
                        continue
                    if s * 2 - total in st:
                        return True
                    # 如果分割到更下面，那么可以删第一行的元素
                    if i == 0:
                        st.update(row)
                return False

            # 删除上半部分中的数 or 删除下半部分中的数
            return f(a) or f(a[::-1])

        # 水平分割 or 垂直分割
        return check(grid) or check(list(zip(*grid)))

    # 3550. 数位和等于下标的最小下标 (Smallest Index With Digit Sum Equal to Index)
    def smallestIndex(self, nums: List[int]) -> int:
        for i, v in enumerate(nums):
            s = 0
            while v:
                s += v % 10
                v //= 10
            if s == i:
                return i
        return -1

    # 3551. 数位和排序需要的最小交换次数 (Minimum Swaps to Sort by Digit Sum)
    def minSwaps(self, nums: List[int]) -> int:
        a = sorted([sum(map(int, str(x))), x, i] for i, x in enumerate(nums))
        res = len(nums)
        vis = [False] * len(nums)
        for i in range(len(nums)):
            if vis[i]:
                continue
            res -= 1
            j = i
            while not vis[j]:
                vis[j] = True
                j = a[j][2]
        return res

    # 3552. 网格传送门旅游 (Grid Teleportation Traversal)
    def minMoves(self, matrix: List[str]) -> int:
        m = len(matrix)
        n = len(matrix[0])
        dic = defaultdict(list)
        dis = [[inf] * n for _ in range(m)]
        dis[0][0] = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] != "." and matrix[i][j] != "#":
                    dic[ord(matrix[i][j]) - ord("A")].append((i, j))
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        # d, bit, x, y
        q = deque()
        q.append((0, 0, 0))
        while q:
            d, x, y = q.popleft()
            if x == m - 1 and y == n - 1:
                return d
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] != "#":
                    if dis[nx][ny] > d + 1:
                        dis[nx][ny] = d + 1
                        q.append((d + 1, nx, ny))
            if (
                matrix[x][y] != "."
                and matrix[x][y] != "#"
                and (ord(matrix[x][y]) - ord("A")) in dic
            ):
                idx = ord(matrix[x][y]) - ord("A")
                for nx, ny in dic[idx]:
                    if dis[nx][ny] > d:
                        dis[nx][ny] = d
                        q.appendleft((d, nx, ny))
                del dic[idx]  # 避免重复使用同一个传送门
        return -1

    # 3556. 最大质数子字符串之和 (Sum of Largest Prime Substrings)
    def sumOfLargestPrimes(self, s: str) -> int:
        def is_prime(n: int) -> bool:
            if n < 2:
                return False
            for i in range(2, int(sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return True

        _s = SortedSet()
        for i in range(len(s)):
            for j in range(i, len(s)):
                num = int(s[i : j + 1])
                if is_prime(num):
                    _s.add(num)
                    if len(_s) > 3:
                        _s.pop(0)
        return sum(_s)

    # 3557. 不相交子字符串的最大数量 (Find Maximum Number of Non Intersecting Substrings)
    def maxSubstrings(self, word: str) -> int:
        dic = {}
        res = 0
        for i, v in enumerate(word):
            if v in dic:
                if i - dic[v] > 2:
                    res += 1
                    dic.clear()
            else:
                dic[v] = i
        return res

    # 3558. 给边赋权值的方案数 I (Number of Ways to Assign Edge Weights I)
    def assignEdgeWeights(self, edges: List[List[int]]) -> int:
        def max_depth(x: int, fa: int, d: int) -> int:
            nonlocal mx
            mx = max(mx, d)
            for y in g[x]:
                if y != fa:
                    max_depth(y, x, d + 1)

        @cache
        def dfs(i: int, j: int) -> int:
            if i == mx:
                return j & 1
            return (dfs(i + 1, j) + dfs(i + 1, j ^ 1)) % MOD

        MOD = 10**9 + 7
        n = len(edges) + 1
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u - 1].append(v - 1)
            g[v - 1].append(u - 1)
        mx = 0
        max_depth(0, -1, 0)
        return dfs(0, 0)

    # 135. 分发糖果 (Candy)
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        right = [1] * n
        for i in range(n - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                right[i] = right[i + 1] + 1
        left = [1] * n
        for i in range(1, n):
            if ratings[i] > ratings[i - 1]:
                left[i] = left[i - 1] + 1
        return sum(max(x, y) for x, y in zip(left, right))

    # 3560. 木材运输的最小成本 (Find Minimum Log Transportation Cost)
    def minCuttingCost(self, n: int, m: int, k: int) -> int:
        res = 0
        if n > k:
            res += k * (n - k)
        if m > k:
            res += k * (m - k)
        return res

    # 3561. 移除相邻字符 (Resulting String After Adjacent Removals)
    def resultingString(self, s: str) -> str:
        def is_adjacent(c1: str, c2: str) -> bool:
            d = abs(ord(c1) - ord(c2))
            return d == 1 or d == 25

        st = []
        for c in s:
            if st and is_adjacent(st[-1], c):
                st.pop()
            else:
                st.append(c)
        return "".join(st)

    # 3563. 移除相邻字符后字典序最小的字符串 (Lexicographically Smallest String After Adjacent Removals)
    def lexicographicallySmallestString(self, s: str) -> str:
        def is_adjacent(c1: str, c2: str) -> bool:
            d = abs(ord(c1) - ord(c2))
            return d == 1 or d == 25

        @cache
        def check(i: int, j: int) -> bool:
            if i > j:
                return True
            if is_adjacent(s[i], s[j]) and check(i + 1, j - 1):
                return True
            for k in range(i + 1, j, 2):
                if check(i, k) and check(k + 1, j):
                    return True
            return False

        @cache
        def dfs(i: int) -> str:
            if i == n:
                return ""
            res = s[i] + dfs(i + 1)
            for j in range(i + 1, n, 2):
                if check(i, j):
                    res = min(res, dfs(j + 1))
            return res

        n = len(s)
        return dfs(0)

    # 1298. 你能从盒子里获得的最大糖果数 (Maximum Candies You Can Get from Boxes)
    def maxCandies(
        self,
        status: List[int],
        candies: List[int],
        keys: List[List[int]],
        containedBoxes: List[List[int]],
        initialBoxes: List[int],
    ) -> int:
        n = len(status)
        vis = [0] * n
        has_boxes = [0] * n
        q = deque()
        res = 0
        for init in initialBoxes:
            has_boxes[init] = 1
            if status[init]:
                vis[init] = 1
                res += candies[init]
                q.append(init)
        while q:
            x = q.popleft()
            for key in keys[x]:
                status[key] = 1
                if not vis[key] and has_boxes[key]:
                    vis[key] = 1
                    res += candies[key]
                    q.append(key)
            for box in containedBoxes[x]:
                has_boxes[box] = 1
                if not vis[box] and status[box]:
                    vis[box] = 1
                    res += candies[box]
                    q.append(box)
        return res

    # 3566. 等积子集的划分方案 (Partition Array into Two Equal Product Subsets)
    def checkEqualPartitions(self, nums: List[int], target: int) -> bool:
        def dfs(i: int, j: int, k: bool, l: bool) -> bool:
            if i == n:
                return k and l and target == j
            if j > target:
                return False
            return dfs(i + 1, j, True, l) or dfs(i + 1, j * nums[i], k, True)

        n = len(nums)
        mul = 1
        for num in nums:
            mul *= num
        if mul != target * target:
            return False
        return dfs(0, 1, False, False)

    # 3567. 子矩阵的最小绝对差 (Minimum Absolute Difference in Sliding Submatrix)
    def minAbsDiff(self, grid: List[List[int]], k: int) -> List[List[int]]:
        m = len(grid)
        n = len(grid[0])
        res = [[0] * (n - k + 1) for _ in range(m - k + 1)]
        for i in range(m - k + 1):
            sub_grid = grid[i : i + k]
            for j in range(n - k + 1):
                a = []
                for row in sub_grid:
                    a.extend(row[j : j + k])
                a.sort()
                _min = inf
                for x, y in pairwise(a):
                    if y > x:
                        _min = min(_min, y - x)
                if _min != inf:
                    res[i][j] = _min
        return res

    # 3568. 清理教室的最少移动 (Minimum Moves to Clean the Classroom)
    def minMoves(self, classroom: List[str], energy: int) -> int:
        m = len(classroom)
        n = len(classroom[0])
        garbage = [[0] * n for _ in range(m)]
        cnt = 0
        start_x = 0
        start_y = 0
        for i in range(m):
            for j in range(n):
                if classroom[i][j] == "S":
                    start_x, start_y = i, j
                elif classroom[i][j] == "L":
                    garbage[i][j] = 1 << cnt
                    cnt += 1
        if cnt == 0:
            return 0
        max_energy = [[[-1] * (1 << cnt) for _ in range(n)] for _ in range(m)]
        max_energy[start_x][start_y][0] = energy
        q = deque()
        q.append([start_x, start_y, energy, 0])
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        u = (1 << cnt) - 1
        res = 0
        while q:
            sz = len(q)
            for _ in range(sz):
                x, y, e, mask = q.popleft()
                if mask == u:
                    return res
                if e == 0:
                    continue
                for dx, dy in dirs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < m and 0 <= ny < n and classroom[nx][ny] != "X":
                        nmask = mask | garbage[nx][ny]
                        ne = energy if classroom[nx][ny] == "R" else e - 1
                        if ne > max_energy[nx][ny][nmask]:
                            max_energy[nx][ny][nmask] = ne
                            q.append([nx, ny, ne, nmask])
            res += 1
        return -1

    # 3565. 顺序网格路径覆盖 (Sequential Grid Path Cover)
    def findPath(self, grid: List[List[int]], k: int) -> List[List[int]]:
        def dfs(x: int, y: int, mx: int, mask: int) -> bool:
            if mask == (1 << (m * n)) - 1:
                return True
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and ((mask >> (nx * n + ny)) & 1 == 0):
                    if grid[nx][ny] - mx == 1 or grid[nx][ny] == 0:
                        if dfs(
                            nx, ny, max(mx, grid[nx][ny]), mask | (1 << (nx * n + ny))
                        ):
                            res.append([nx, ny])
                            return True
            return False

        m = len(grid)
        n = len(grid[0])
        res = []
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for i in range(m):
            for j in range(n):
                if grid[i][j] <= 1 and dfs(i, j, grid[i][j], 1 << (i * n + j)):
                    res.append([i, j])
                    return list(reversed(res))
                res.clear()
        return []

    # 3466. 最大硬币收集量 (Maximum Coin Collection) --plus
    def maxCoins(self, lane1: List[int], lane2: List[int]) -> int:
        @cache
        def dfs(i: int, j: int, k: int, l: bool) -> int:
            if i == n:
                return 0 if l else -inf
            res = -inf
            # 之前还未进入赛道，在第i位置，仍可以不进入赛道
            if not l:
                res = max(res, dfs(i + 1, j, k, l))
            # 不换赛道 max(0, dfs(x, x, x, True)) ，其中 0 表示下赛道因为已经跑了至少一英里
            res = max(res, max(0, dfs(i + 1, j, k, True)) + arr[i][j])
            # 换赛道
            if k < 2:
                # max(0, dfs(x, x, x, True)) ，其中 0 表示下赛道因为已经跑了至少一英里
                res = max(res, max(0, dfs(i + 1, j ^ 1, k + 1, True)) + arr[i][j ^ 1])
            return res

        n = len(lane1)
        arr = [[x, y] for x, y in zip(lane1, lane2)]
        # dfs(i, j, k, l) 从索引i开始，当前在赛道j，已经切换了k次赛道，已经跑了至少1英里（l == True 表示至少跑了1英里， l == False表示还未跑）时，
        # 可获得的最大硬币数
        return max(dfs(0, 0, 0, False), dfs(0, 1, 1, False))

    # 1061. 按字典序排列最小的等效字符串 (Lexicographically Smallest Equivalent String)
    def smallestEquivalentString(self, s1: str, s2: str, baseStr: str) -> str:
        class union:
            def __init__(self, n: int):
                self.rank = [1] * n
                self.parent = [i for i in range(n)]

            def get_root(self, p: int) -> int:
                if self.parent[p] == p:
                    return p
                self.parent[p] = self.get_root(self.parent[p])
                return self.parent[p]

            def is_connect(self, p1: int, p2: int) -> bool:
                return self.get_root(p1) == self.get_root(p2)

            def union(self, p1: int, p2: int) -> None:
                r1 = self.get_root(p1)
                r2 = self.get_root(p2)
                if r1 == r2:
                    return
                if self.rank[r1] < self.rank[r2]:
                    self.parent[r1] = r2
                else:
                    self.parent[r2] = r1
                    if self.rank[r1] == self.rank[r2]:
                        self.rank[r1] += 1

        u = union(26)
        for a, b in zip(s1, s2):
            u.union(ord(a) - ord("a"), ord(b) - ord("a"))
        dic = [-1] * 26
        for i in range(26):
            for j in range(i + 1):
                if u.is_connect(i, j):
                    dic[i] = j
                    break
        res = []
        for x in baseStr:
            id = ord(x) - ord("a")
            res.append(chr(ord("a") + dic[id]))
        return "".join(res)

    # 440. 字典序的第K小数字 (K-th Smallest in Lexicographical Order)
    def findKthNumber(self, n: int, k: int) -> int:
        def cal(node: int) -> int:
            sz = 0
            left = node
            right = node + 1
            while left <= n:
                sz += min(right - 1, n) - left + 1
                left *= 10
                right *= 10
            return sz

        node = 1
        k -= 1
        while k:
            sz = cal(node)
            if sz <= k:
                node += 1
                k -= sz
            else:
                node *= 10
                k -= 1
        return node

    # 2434. 使用机器人打印字典序最小的字符串 (Using a Robot to Print the Lexicographically Smallest String)
    def robotWithString(self, s: str) -> str:
        cnt = Counter(s)
        ans = []
        stk = []
        mi = "a"
        for c in s:
            cnt[c] -= 1
            while mi < "z" and cnt[mi] == 0:
                mi = chr(ord(mi) + 1)
            stk.append(c)
            while stk and stk[-1] <= mi:
                ans.append(stk.pop())
        return "".join(ans)

    # 386. 字典序排数 (Lexicographical Numbers)
    def lexicalOrder(self, n: int) -> List[int]:
        def dfs(x: int) -> None:
            if x > n:
                return
            res.append(x)
            for i in range(10):
                dfs(x * 10 + i)

        res = []
        for i in range(1, 10):
            dfs(i)
        return res

    # 3442. 奇偶频次间的最大差值 I (Maximum Difference Between Even and Odd Frequency I)
    def maxDifference(self, s: str) -> int:
        cnts = [0] * 26
        for c in s:
            cnts[ord(c) - ord("a")] += 1
        max_odd = 0
        min_even = inf
        for c in cnts:
            if c & 1:
                max_odd = max(max_odd, c)
            elif c:
                min_even = min(min_even, c)
        return max_odd - min_even

    # 3572. 选择不同 X 值三元组使 Y 值之和最大 (Maximize Y‑Sum by Picking a Triplet of Distinct X‑Values)
    def maxSumDistinctTriplet(self, x: List[int], y: List[int]) -> int:
        dic = defaultdict(int)
        for a, b in zip(x, y):
            dic[a] = max(dic[a], b)
        if len(dic) < 3:
            return -1
        max1 = 0
        max2 = 0
        max3 = 0
        for v in dic.values():
            if v >= max1:
                max3 = max2
                max2 = max1
                max1 = v
            elif v >= max2:
                max3 = max2
                max2 = v
            elif v >= max3:
                max3 = v
        return max1 + max2 + max3

    # 3573. 买卖股票的最佳时机 V (Best Time to Buy and Sell Stock V)
    def maximumProfit(self, prices: List[int], m: int) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if m == k or i == n:
                return 0 if j == 0 else -inf
            res = dfs(i + 1, j, k)
            if j == 0:
                return max(
                    res, dfs(i + 1, 1, k) - prices[i], dfs(i + 1, -1, k) + prices[i]
                )
            return max(res, dfs(i + 1, 0, k + 1) + prices[i] * j)

        n = len(prices)
        res = dfs(0, 0, 0)
        dfs.cache_clear()
        return res

    # 3576. 数组元素相等转换 (Transform Array to All Equal Elements)
    def canMakeEqual(self, nums: List[int], k: int) -> bool:
        def check(target: int, k: int) -> bool:
            n = len(nums)
            i = 0
            while i < n:
                if nums[i] == target:
                    i += 1
                    continue
                j = i + 1
                while j < n and nums[j] == target:
                    j += 1
                if j == n:
                    return False
                if k < j - i:
                    return False
                k -= j - i
                i = j + 1
            return True

        n = len(nums)
        return check(1, k) or check(-1, k)

    # 3577. 统计计算机解锁顺序排列数 (Count the Number of Computer Unlocking Permutations)
    def countPermutations(self, complexity: List[int]) -> int:
        _min = complexity[0]
        res = 1
        MOD = 10**9 + 7
        for i, x in enumerate(complexity[1:], 1):
            if x <= _min:
                return 0
            _min = min(_min, x)
            res *= i
            res %= MOD
        return res

    # 3574. 最大子数组 GCD 分数 (Maximize Subarray GCD Score)
    def maxGCDScore(self, nums: List[int], k: int) -> int:
        res = 0
        for i in range(len(nums)):
            lb_min = inf
            lb_cnt = g = 0
            for j in range(i, -1, -1):
                x = nums[j]
                lb = x & -x
                if lb < lb_min:
                    lb_min, lb_cnt = lb, 1
                elif lb == lb_min:
                    lb_cnt += 1
                g = gcd(g, x)
                new_g = g * 2 if lb_cnt <= k else g
                res = max(res, new_g * (i - j + 1))
        return res

    # 2616. 最小化数对的最大差值 (Minimize the Maximum Difference of Pairs)
    def minimizeMax(self, nums: List[int], p: int) -> int:
        def check(target: int) -> bool:
            i = 0
            cnt = 0
            while i + 1 < n and cnt < p:
                if nums[i + 1] - nums[i] <= target:
                    cnt += 1
                    i += 1
                i += 1
            return cnt >= p

        n = len(nums)
        nums.sort()
        left = 0
        right = 0 if n <= 1 else max((y - x) for x, y in pairwise(nums))
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return right + 1

    # 2566. 替换一个数字后的最大差值 (Maximum Difference by Remapping a Digit)
    def minMaxDifference(self, num: int) -> int:
        def check(t: chr) -> int:
            a = [c for c in str(num)]
            for i, v in enumerate(a):
                if v != t:
                    for j in range(i, len(a)):
                        if a[j] == v:
                            a[j] = t
                    break
            return int("".join(a))

        return check("9") - check("0")

    # 1432. 改变一个整数能得到的最大差值 (Max Difference You Can Get From Changing an Integer)
    def maxDiff(self, num: int) -> int:
        def check(t: chr) -> int:
            a = [c for c in str(num)]
            for i, v in enumerate(a):
                if v != t:
                    for j in range(i, len(a)):
                        if a[j] == v:
                            a[j] = t
                    break
            return int("".join(a))

        def check2(x: chr, y: chr) -> int:
            a = [c for c in str(num)]
            for i in range(len(a)):
                if a[i] != x and a[i] != y:
                    t = a[i]
                    for j in range(i, len(a)):
                        if a[j] == t:
                            a[j] = "0"
                    break
            return int("".join(a))

        return check("9") - (check("1") if str(num)[0] != "1" else check2("0", "1"))

    # 3405. 统计恰好有 K 个相等相邻元素的数组数目 (Count the Number of Arrays with K Matching Adjacent Elements)
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        return comb(n - 1, k) % MOD * m * pow(m - 1, n - k - 1, MOD) % MOD

    # 3582. 为视频标题生成标签 (Generate Tag for Video Caption)
    def generateTag(self, caption: str) -> str:
        a = caption.split(" ")
        res = ["#"]
        for s in a:
            for i in range(len(s)):
                if i:
                    res.append(s[i].lower())
                else:
                    res.append(s[i].upper())
                if len(res) == 100:
                    break
            if len(res) == 100:
                break
        if len(res) >= 2:
            res[1] = res[1].lower()
        return "".join(res)

    # 3583. 统计特殊三元组 (Count Special Triplets)
    def specialTriplets(self, nums: List[int]) -> int:
        n = len(nums)
        dic = defaultdict(int)
        left = [0] * n
        for i, v in enumerate(nums):
            c = v << 1
            left[i] = dic[c]
            dic[v] += 1
        dic.clear()
        res = 0
        MOD = 10**9 + 7
        for i in range(n - 1, -1, -1):
            c = nums[i] << 1
            res += left[i] * dic[c]
            res %= MOD
            dic[nums[i]] += 1
        return res

    # 3584. 子序列首尾元素的最大乘积 (Maximum Product of First and Last Elements of a Subsequence)
    def maximumProduct(self, nums: List[int], m: int) -> int:
        n = len(nums)
        right = [[0] * 2 for _ in range(n + 1)]
        right[-1][0] = inf
        right[-1][1] = -inf
        for i in range(n - 1, m - 2, -1):
            right[i][0] = min(right[i + 1][0], nums[i])
            right[i][1] = max(right[i + 1][1], nums[i])
        left = [inf, -inf]
        res = -inf
        for i in range(n - m + 1):
            left[0] = min(left[0], nums[i])
            left[1] = max(left[1], nums[i])
            res = max(res, left[0] * right[i + m - 1][0], left[1] * right[i + m - 1][1])
        return res

    # 2294. 划分数组使最大差为 K (Partition Array Such That Maximum Difference Is K)
    def partitionArray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        i = 0
        res = 0
        while i < n:
            j = i
            while j < n and nums[j] - nums[i] <= k:
                j += 1
            res += 1
            i = j
        return res

    # 3443. K 次修改后的最大曼哈顿距离 (Maximum Manhattan Distance After K Changes)
    def maxDistance(self, s: str, k: int) -> int:
        def check(a: str, b: str, k: int) -> int:
            mx = -inf
            dis = 0
            for x in s:
                if x == a or x == b:
                    dis += 1
                elif k:
                    k -= 1
                    dis += 1
                else:
                    dis -= 1
                mx = max(mx, dis)
            return mx

        return max(
            check("N", "W", k),
            check("S", "E", k),
            check("N", "E", k),
            check("S", "W", k),
        )

    # 3444. 使数组包含目标值倍数的最少增量 (Minimum Increments for Target Multiples in an Array)
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j == u:
                return 0
            if i == n:
                return inf
            # 不修改 nums[i]
            res = dfs(i + 1, j)
            # 修改 nums[i]
            sub = c = u ^ j
            while sub:
                _lcm = l[sub]
                cnt = (_lcm - nums[i] % _lcm) % _lcm
                res = min(res, dfs(i + 1, j | sub) + cnt)
                sub = (sub - 1) & c
            return res

        n = len(nums)
        target = list(set(target))
        m = len(target)
        u = (1 << m) - 1
        l = [0] * (1 << m)
        l[0] = 1
        for i in range(1, 1 << m):
            l[i] = lcm(l[i & (i - 1)], target[(i & -i).bit_length() - 1])
        return dfs(0, 0)

    # 3446. 按对角线进行矩阵排序 (Sort Matrix by Diagonals)
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        for i in range(n):
            _l = []
            x = i
            y = 0
            while x < n and y < n:
                _l.append(grid[x][y])
                x += 1
                y += 1
            _l.sort(reverse=True)
            x = i
            y = 0
            idx = 0
            while x < n and y < n:
                grid[x][y] = _l[idx]
                idx += 1
                x += 1
                y += 1
        for j in range(1, n):
            _l = []
            x = 0
            y = j
            while x < n and y < n:
                _l.append(grid[x][y])
                x += 1
                y += 1
            _l.sort()
            x = 0
            y = j
            idx = 0
            while x < n and y < n:
                grid[x][y] = _l[idx]
                idx += 1
                x += 1
                y += 1
        return grid

    # 3447. 将元素分配给有约束条件的组 (Assign Elements to Groups with Constraints)
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        mx = max(groups)
        target = [-1] * (mx + 1)
        for i, v in enumerate(elements):
            if v > mx or target[v] >= 0:
                continue
            for y in range(v, mx + 1, v):
                if target[y] < 0:
                    target[y] = i
        return [target[x] for x in groups]

    # 3448. 统计可以被最后一个数位整除的子字符串数目 (Count Substrings Divisible By Last Digit)
    def countSubstrings(self, s: str) -> int:
        res = 0
        f = [[0] * 9 for _ in range(10)]
        for d in map(int, s):
            for m in range(1, 10):
                nf = [0] * m
                nf[d % m] = 1
                for rem in range(m):
                    nf[(rem * 10 + d) % m] += f[m][rem]
                f[m] = nf
            res += f[d][0]
        return res

    # 3452. 好数字之和 (Sum of Good Numbers)
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        res = 0
        for i, x in enumerate(nums):
            if i - k >= 0 and x <= nums[i - k]:
                continue
            if i + k < len(nums) and x <= nums[i + k]:
                continue
            res += x
        return res

    # 3438. 找到字符串中合法的相邻数字 (Find Valid Pair of Adjacent Digits in String)
    def findValidPair(self, s: str) -> str:
        cnts = [0] * 10
        for c in map(int, s):
            cnts[c] += 1
        for x, y in pairwise(map(int, s)):
            if x != y and cnts[x] == x and cnts[y] == y:
                return f"{x}{y}"
        return ""

    # 3439. 重新安排会议得到最多空余时间 I (Reschedule Meetings for Maximum Free Time I)
    def maxFreeTime(
        self, eventTime: int, k: int, startTime: List[int], endTime: List[int]
    ) -> int:
        n = len(startTime)
        free = [0] * (n + 1)
        free[0] = startTime[0]
        for i in range(1, n):
            free[i] = startTime[i] - endTime[i - 1]
        free[n] = eventTime - endTime[-1]
        res = 0
        s = 0
        for i, v in enumerate(free):
            s += v
            if i < k:
                continue
            res = max(res, s)
            s -= free[i - k]
        return res

    # 3440. 重新安排会议得到最多空余时间 II (Reschedule Meetings for Maximum Free Time II)
    def maxFreeTime(
        self, eventTime: int, startTime: List[int], endTime: List[int]
    ) -> int:
        n = len(startTime)
        free = [0] * (n + 1)

        free[0] = startTime[0]
        for i in range(n - 1):
            free[i + 1] = startTime[i + 1] - endTime[i]
        free[n] = eventTime - endTime[-1]
        d = SortedList(free)
        res = 0
        for i, (s, e) in enumerate(zip(startTime, endTime)):
            res = max(res, free[i] + free[i + 1])
            d.remove(free[i])
            d.remove(free[i + 1])
            x = e - s
            if x <= d[-1]:
                res = max(res, free[i] + free[i + 1] + x)
            d.add(free[i])
            d.add(free[i + 1])
        return res

    # 2138. 将字符串拆分为若干长度为 k 的组 (Divide a String Into Groups of Size k)
    def divideString(self, s: str, k: int, fill: str) -> List[str]:
        res = []
        for i in range(0, len(s), k):
            part = s[i : i + k]
            if len(part) < k:
                part += fill * (k - len(part))
            res.append(part)
        return res

    # 2081. k 镜像数字的和 (Sum of k-Mirror Numbers)
    def kMirror(self, k: int, n: int) -> int:
        def is_palindrome(num: int, k: int) -> bool:
            s = ""
            while num:
                s += str(num % k)
                num //= k
            return s == s[::-1]

        res = 0
        cnt = 0
        left = 1
        while cnt < n:
            right = left * 10
            for i in range(2):
                num = left
                while num < right and cnt < n:
                    combined = num
                    x = num // 10 if i == 0 else num
                    while x:
                        combined = combined * 10 + x % 10
                        x //= 10
                    if is_palindrome(combined, k):
                        res += combined
                        cnt += 1
                    num += 1
            left = right
        return res

    # 2200. 找出数组中的所有 K 近邻下标 (Find All K-Distant Indices in an Array)
    def findKDistantIndices(self, nums: List[int], key: int, k: int) -> List[int]:
        n = len(nums)
        cnt = 0
        for i in range(0, min(n, k + 1)):
            if nums[i] == key:
                cnt += 1
        res = []
        for i in range(n):
            if cnt > 0:
                res.append(i)
            if i + k + 1 < n and nums[i + k + 1] == key:
                cnt += 1
            if i - k >= 0 and nums[i - k] == key:
                cnt -= 1
        return res

    # 3587. 最小相邻交换至奇偶交替 (Minimum Adjacent Swaps to Alternate Parity)
    def minSwaps(self, nums: List[int]) -> int:
        def min_swap(start: int) -> int:
            res = 0
            for x, y in zip(pos[start], range(0, n, 2)):
                res += abs(x - y)
            return res

        n = len(nums)
        pos = [[] for _ in range(2)]
        for i, x in enumerate(nums):
            pos[x & 1].append(i)
        if abs(len(pos[0]) - len(pos[1])) > 1:
            return -1
        res = inf
        if len(pos[0]) >= len(pos[1]):
            res = min(res, min_swap(0))
        if len(pos[1]) >= len(pos[0]):
            res = min(res, min_swap(1))
        return res

    # 3588. 找到最大三角形面积 (Find Maximum Area of a Triangle)
    def maxArea(self, coords: List[List[int]]) -> int:
        def check(d: int) -> int:
            _min = defaultdict(lambda: inf)
            _max = defaultdict(lambda: -inf)
            left = min(x[d] for x in coords)
            right = max(x[d] for x in coords)
            res = 0
            for x in coords:
                _min[x[d]] = min(_min[x[d]], x[d ^ 1])
                _max[x[d]] = max(_max[x[d]], x[d ^ 1])
                res = max(
                    res,
                    (_max[x[d]] - _min[x[d]]) * (right - x[d]),
                    (_max[x[d]] - _min[x[d]]) * (x[d] - left),
                )
            return res

        res = max(check(0), check(1))
        return -1 if res == 0 else res

    # 3589. 计数质数间隔平衡子数组 (Count Prime-Gap Balanced Subarrays)
    def primeSubarray(self, nums: List[int], k: int) -> int:
        mx = max(nums)
        prime = [True] * (mx + 1)
        prime[1] = False
        for i in range(2, mx + 1):
            if prime[i]:
                for j in range(i * i, mx + 1, i):
                    prime[j] = False
        pre = -1
        pre2 = -1
        j = 0
        cnt = 0
        dic = defaultdict(int)
        res = 0
        for i, x in enumerate(nums):
            if prime[x]:
                dic[x] += 1
                cnt += 1
            while dic and max(dic.keys()) - min(dic.keys()) > k:
                if prime[nums[j]]:
                    dic[nums[j]] -= 1
                    cnt -= 1
                    if dic[nums[j]] == 0:
                        del dic[nums[j]]
                j += 1
            if prime[x]:
                pre2 = pre
                pre = i
            res += pre2 - j + 1
        return res

    # 3591. 检查元素频次是否为质数 (Check if Any Element Has Prime Frequency)
    def checkPrimeFrequency(self, nums: List[int]) -> bool:
        def is_prime(n: int) -> bool:
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return n > 1

        cnts = [0] * 101
        for x in nums:
            cnts[x] += 1
        return any(is_prime(c) for c in cnts if c > 0)

    # 3593. 使叶子路径成本相等的最小增量 (Minimum Increments to Equalize Leaf Paths)
    def minIncrease(self, n: int, edges: List[List[int]], cost: List[int]) -> int:
        def dfs(x: int, fa: int) -> int:
            mx = 0
            mx_cnt = 0
            cnt = 0
            for y in g[x]:
                if y != fa:
                    cnt += 1
                    v = dfs(y, x)
                    if v > mx:
                        mx_cnt = 1
                        mx = v
                    elif v == mx:
                        mx_cnt += 1
            nonlocal res
            res += cnt - mx_cnt
            return cost[x] + mx

        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        res = 0
        dfs(0, -1)
        return res

    # 2311. 小于等于 K 的最长二进制子序列 (Longest Binary Subsequence Less Than or Equal to K)
    def longestSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        m = k.bit_length()
        if n < m:
            return n
        res = m if int(s[-m:], 2) <= k else m - 1
        return res + s[:-m].count("0")

    # 3592. 硬币面值还原 (Inverse Coin Change)
    def findCoins(self, numWays: List[int]) -> List[int]:
        @cache
        def dfs(i: int, j: int) -> int:
            if j == 0:
                return 1
            if i < 0:
                return 0
            res = dfs(i - 1, j)
            if j - _list[i] >= 0:
                res += dfs(i, j - _list[i])
            return res

        _list = []
        for i, v in enumerate(numWays, 1):
            cur_ways = dfs(len(_list) - 1, i)
            if cur_ways == v:
                continue
            if cur_ways + 1 == v:
                _list.append(i)
                continue
            return []
        return _list

    # 3594. 所有人渡河所需的最短时间 (Minimum Time to Transport All Individuals)
    def minTime(
        self, n: int, k: int, m: int, time: List[int], mul: List[float]
    ) -> float:
        mx = [0] * (1 << n)
        for i in range(1, 1 << n):
            mx[i] = max(mx[i & (i - 1)], time[(i & -i).bit_length() - 1])
        u = (1 << n) - 1
        dis = [[inf] * m for _ in range(1 << n)]
        q = []
        heapq.heapify(q)
        dis[0][0] = 0
        # time, mask, stage
        q.append((0, 0, 0))
        while q:
            (cur_time, cur_mask, cur_stage) = heapq.heappop(q)
            if cur_time > dis[cur_mask][cur_stage]:
                continue
            if cur_mask == u:
                return cur_time
            sub = c = u ^ cur_mask
            # 从未过河的人中枚举至多k个人上船
            while sub:
                if sub.bit_count() <= k:
                    pass_time = mx[sub] * mul[cur_stage]
                    return_stage = (cur_stage + floor(pass_time) % m) % m
                    cc = cur_mask | sub
                    # 从已过河的人中枚举1个人返回
                    while cc:
                        lb = (cc & -cc).bit_length() - 1
                        # cc == u 说明所有人均已过河，无需再选一个人返回
                        return_time = 0 if cc == u else time[lb] * mul[return_stage]
                        n_time = cur_time + pass_time + return_time
                        n_stage = (return_stage + floor(return_time) % m) % m
                        n_mask = (cur_mask | sub) ^ (0 if cc == u else (1 << lb))
                        if n_time < dis[n_mask][n_stage]:
                            dis[n_mask][n_stage] = n_time
                            heapq.heappush(q, (n_time, n_mask, n_stage))
                        cc &= cc - 1
                sub = (sub - 1) & c
        return -1

    # 2099. 找到和最大的长度为 K 的子序列 (Find Subsequence of Length K With the Largest Sum)
    def maxSubsequence(self, nums: List[int], k: int) -> List[int]:
        a = [[i, x] for i, x in enumerate(nums)]
        a.sort(key=lambda o: -o[1])
        return [x for _, x in sorted(a[:k])]

    # 1498. 满足条件的子序列数目 (Number of Subsequences That Satisfy the Given Sum Condition)
    def numSubseq(self, nums: List[int], target: int) -> int:
        nums.sort()
        n = len(nums)
        MOD = 10**9 + 7
        pow2 = [0] * (n + 1)
        pow2[0] = 1
        for i in range(1, n + 1):
            pow2[i] = (pow2[i - 1] << 1) % MOD

        left = 0
        right = n - 1
        res = 0
        while left <= right:
            if nums[left] + nums[right] > target:
                right -= 1
            else:
                res += pow2[right - left]
                res %= MOD
                left += 1
        return res

    # 594. 最长和谐子序列 (Longest Harmonious Subsequence)
    def findLHS(self, nums: List[int]) -> int:
        d = defaultdict(int)
        for x in nums:
            d[x] += 1
        res = 0
        for k, v in d.items():
            if k + 1 in d:
                res = max(res, v + d[k + 1])
        return res

    # 3597. 分割字符串 (Partition String)
    def partitionString(self, s: str) -> List[str]:
        _set = set()
        res = []
        cur = ""
        for c in s:
            cur += c
            if cur not in _set:
                _set.add(cur)
                res.append(cur)
                cur = ""
        return res

    # 3598. 相邻字符串之间的最长公共前缀 (Longest Common Prefix Between Adjacent Strings After Removals)
    def longestCommonPrefix(self, words: List[str]) -> List[int]:
        def longest_prefix(a: str, b: str) -> int:
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    return i
            return min(len(a), len(b))

        n = len(words)
        right = [0] * n
        for i in range(n - 2, -1, -1):
            right[i] = max(right[i + 1], longest_prefix(words[i], words[i + 1]))
        left = [0] * n
        for i in range(1, n):
            left[i] = max(left[i - 1], longest_prefix(words[i - 1], words[i]))
        res = [0] * n
        for i in range(n):
            mx = 0
            if i:
                mx = max(mx, left[i - 1])
            if i < n - 1:
                mx = max(mx, right[i + 1])
            if i and i < n - 1:
                mx = max(mx, longest_prefix(words[i - 1], words[i + 1]))
            res[i] = mx
        return res

    # 3599. 划分数组得到最小 XOR (Partition Array to Minimize XOR)
    def minXor(self, nums: List[int], k: int) -> int:
        min = lambda a, b: b if b < a else a
        max = lambda a, b: b if b > a else a

        @cache
        def dfs(i: int, j: int) -> int:
            if i == n or j == k:
                return 0 if i == n and j == k else inf
            if k - j > n - i:
                return inf
            res = inf
            xor = 0
            for x, v in enumerate(nums[i:], i):
                xor ^= v
                res = min(res, max(dfs(x + 1, j + 1), xor))
            return res

        n = len(nums)
        return dfs(0, 0)

    # 3600. 升级后最大生成树稳定性 (Maximize Spanning Tree Stability with Upgrades)
    def maxStability(self, n: int, edges: List[List[int]], k: int) -> int:
        class union:
            def __init__(self, n: int):
                self.parent = [i for i in range(n)]
                self.rank = [1] * n
                self.cnt = n

            def get_root(self, p: int) -> int:
                if self.parent[p] == p:
                    return p
                self.parent[p] = self.get_root(self.parent[p])
                return self.parent[p]

            def union(self, p1: int, p2: int) -> bool:
                r1 = self.get_root(p1)
                r2 = self.get_root(p2)
                if r1 == r2:
                    return False
                if self.rank[r1] < self.rank[r2]:
                    self.parent[r1] = r2
                else:
                    self.parent[r2] = r1
                    if self.rank[r1] == self.rank[r2]:
                        self.rank[r1] += 1
                self.cnt -= 1
                return True

            def get_cnt(self) -> int:
                return self.cnt

        def check(low: int) -> bool:
            _u = union(n)
            for u, v, s, must in edges:
                if must and s < low:  # 必选边太小了
                    return False
                if must or s >= low:
                    _u.union(u, v)
            k_left = k
            for u, v, s, must in edges:
                if k_left == 0 or _u.get_cnt() == 1:
                    break
                # k > 0 且 s扩大2倍后>=low 且 u, v 不连通
                if not must and s * 2 >= low and _u.union(u, v):
                    k_left -= 1
            return _u.get_cnt() == 1

        u_must = union(n)  # 必选边并查集
        u_all = union(n)  # 所有边并查集
        left = inf
        right = 0
        for u, v, s, must in edges:
            if must and not u_must.union(u, v):  # 必选边有环
                return -1
            u_all.union(u, v)
            left = min(left, s)
            right = max(right, s)
        if u_all.get_cnt() > 1:  # 整个图不连通
            return -1
        right <<= 1
        res = -1
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = mid
                left = mid + 1
            else:
                right = mid - 1
        return res

    # 1394. 找出数组中的幸运数 (Find Lucky Integer in an Array)
    def findLucky(self, arr: List[int]) -> int:
        cnts = Counter(arr)
        res = -1
        for k, v in cnts.items():
            if k == v:
                res = max(res, k)
        return res

    # 1865. 找出和为指定值的下标对 (Finding Pairs With a Certain Sum)
    class FindSumPairs:

        def __init__(self, nums1: List[int], nums2: List[int]):
            self.cnt1 = defaultdict(int)
            self.cnt2 = defaultdict(int)
            self.nums2 = nums2
            for x in nums1:
                self.cnt1[x] += 1
            for x in nums2:
                self.cnt2[x] += 1

        def add(self, index: int, val: int) -> None:
            origin = self.nums2[index]
            self.nums2[index] += val
            self.cnt2[origin] -= 1
            self.cnt2[origin + val] += 1

        def count(self, tot: int) -> int:
            return sum(v * self.cnt2[tot - k] for k, v in self.cnt1.items())

    # 1353. 最多可以参加的会议数目 (Maximum Number of Events That Can Be Attended)
    def maxEvents(self, events: List[List[int]]) -> int:
        mx = max(e[1] for e in events)

        # 按照开始时间分组
        groups = [[] for _ in range(mx + 1)]
        for e in events:
            groups[e[0]].append(e[1])

        ans = 0
        h = []
        for i, g in enumerate(groups):
            # 删除过期会议
            while h and h[0] < i:
                heapq.heappop(h)
            # 新增可以参加的会议
            for end_day in g:
                heapq.heappush(h, end_day)
            # 参加一个结束时间最早的会议
            if h:
                ans += 1
                heapq.heappop(h)
        return ans

    # 3602. 十六进制和三十六进制转化 (Hexadecimal and Hexatrigesimal Conversion)
    def concatHex36(self, n: int) -> str:
        def gen(x: int, radix: int) -> int:
            res = []
            while x:
                d = x % radix
                if d < 10:
                    res.append(str(d))
                else:
                    res.append((chr)(d - 10 + ord("A")))
                x //= radix
            return "".join(reversed(res))

        return gen(n * n, 16) + gen(n * n * n, 36)

    # 3603. 交替方向的最小路径代价 II (Minimum Cost Path with Alternating Directions II)
    def minCost(self, m: int, n: int, waitCost: List[List[int]]) -> int:
        q = []
        dis = [[inf] * n for _ in range(m)]
        dis[0][0] = 1
        heapq.heapify(q)
        q.append((1, 0, 0))
        dirs = (0, 1), (1, 0)
        while q:
            (d, x, y) = heapq.heappop(q)
            if x == m - 1 and y == n - 1:
                return d - waitCost[-1][-1]
            for dx, dy in dirs:
                nx = x + dx
                ny = y + dy
                if nx < m and ny < n:
                    if d + (nx + 1) * (ny + 1) + waitCost[nx][ny] < dis[nx][ny]:
                        dis[nx][ny] = d + (nx + 1) * (ny + 1) + waitCost[nx][ny]
                        heapq.heappush(
                            q, (d + (nx + 1) * (ny + 1) + waitCost[nx][ny], nx, ny)
                        )
        return -1

    # 3603. 交替方向的最小路径代价 II (Minimum Cost Path with Alternating Directions II)
    def minCost(self, m: int, n: int, waitCost: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i < 0 or j < 0:
                return inf
            if i == 0 and j == 0:
                return 1
            return (
                min(dfs(i - 1, j), dfs(i, j - 1)) + (i + 1) * (j + 1) + waitCost[i][j]
            )

        return dfs(m - 1, n - 1) - waitCost[-1][-1]

    # 3604. 有向图中到达终点的最少时间 (Minimum Time to Reach Destination in Directed Graph)
    def minTime(self, n: int, edges: List[List[int]]) -> int:
        q = [(0, 0)]
        heapq.heapify(q)
        g = [[] for _ in range(n)]
        for u, v, start, end in edges:
            g[u].append((v, start, end))
        dis = [inf] * n
        dis[0] = 0
        while q:
            t, x = heapq.heappop(q)
            if x == n - 1:
                return t
            for y, start, end in g[x]:
                if t > end:
                    continue
                nt = max(t, start) + 1
                if nt < dis[y]:
                    dis[y] = nt
                    heapq.heappush(q, (nt, y))
        return -1

    # 3606. 优惠券校验器 (Coupon Code Validator)
    def validateCoupons(
        self, code: List[str], businessLine: List[str], isActive: List[bool]
    ) -> List[str]:
        dic = defaultdict(int)
        dic["electronics"] = 0
        dic["grocery"] = 1
        dic["pharmacy"] = 2
        dic["restaurant"] = 3

        def check(s: str) -> bool:
            return len(s) and all(isalpha(c) or isdigit(c) or c == "_" for c in s)

        _list = [
            (dic[b], c)
            for c, b, i in zip(code, businessLine, isActive)
            if i and check(c) and b in dic
        ]
        return [c for _, c in sorted(_list)]

    # 3607. 电网维护 (Power Grid Maintenance)
    def processQueries(
        self, c: int, connections: List[List[int]], queries: List[List[int]]
    ) -> List[int]:

        class union:
            def __init__(self, n: int):
                self.rank = [1] * n
                self.parent = [i for i in range(n)]

            def get_root(self, p: int) -> int:
                if self.parent[p] == p:
                    return p
                self.parent[p] = self.get_root(self.parent[p])
                return self.parent[p]

            def is_connect(self, p1: int, p2: int) -> bool:
                return self.get_root(p1) == self.get_root(p2)

            def union(self, p1: int, p2: int) -> None:
                r1 = self.get_root(p1)
                r2 = self.get_root(p2)
                if r1 == r2:
                    return
                if self.rank[r1] < self.rank[r2]:
                    self.parent[r1] = r2
                else:
                    self.parent[r2] = r1
                    if self.rank[r1] == self.rank[r2]:
                        self.rank[r1] += 1

        c += 1
        u = union(c)
        for x, y in connections:
            u.union(x, y)
        g = [[] for _ in range(c)]
        for i in range(c):
            r = u.get_root(i)
            g[r].append(i)
        for v in g:
            heapq.heapify(v)
        online = [True] * c
        res = []
        for t, x in queries:
            if t == 1:
                if online[x]:
                    res.append(x)
                else:
                    r = u.get_root(x)
                    q = g[r]
                    while q and not online[q[0]]:
                        heapq.heappop(q)
                    if q:
                        res.append(q[0])
                    else:
                        res.append(-1)
            else:
                online[x] = False
        return res

    # 3608. 包含 K 个连通分量需要的最小时间 (Minimum Time for K Connected Components)
    def minTime(self, n: int, edges: List[List[int]], k: int) -> int:
        class union:
            def __init__(self, n: int):
                self.parent = [i for i in range(n)]
                self.rank = [1] * n
                self.cnt = n

            def get_root(self, p: int) -> int:
                if p == self.parent[p]:
                    return p
                self.parent[p] = self.get_root(self.parent[p])
                return self.parent[p]

            def is_connected(self, p1: int, p2: int) -> bool:
                return self.get_root(p1) == self.get_root(p2)

            def union(self, p1: int, p2: int) -> None:
                r1 = self.get_root(p1)
                r2 = self.get_root(p2)
                if r1 == r2:
                    return
                if self.rank[r1] < self.rank[r2]:
                    self.parent[r1] = r2
                else:
                    self.parent[r2] = r1
                    if self.rank[r1] == self.rank[r2]:
                        self.rank[r1] += 1
                self.cnt -= 1

            def get_cnt(self) -> int:
                return self.cnt

        def check(t: int) -> bool:
            u = union(n)
            for x, y, time in edges:
                if time > t:
                    u.union(x, y)
            return u.get_cnt() >= k

        left = 0
        right = max(t for _, _, t in edges) if edges else 0
        res = 0
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = mid
                right = mid - 1
            else:
                left = mid + 1
        return res

    # 2410. 运动员和训练师的最大匹配数 (Maximum Matching of Players With Trainers)
    def matchPlayersAndTrainers(self, players: List[int], trainers: List[int]) -> int:
        m = len(players)
        players.sort()
        trainers.sort()
        cnt = 0
        for t in trainers:
            if cnt == m:
                break
            if players[cnt] <= t:
                cnt += 1
        return cnt

    # 1290. 二进制链表转整数 (Convert Binary Number in a Linked List to Integer)
    def getDecimalValue(self, head: Optional[ListNode]) -> int:
        res = 0
        while head:
            res <<= 1
            res |= head.val
            head = head.next
        return res

    # 3485. 删除元素后 K 个字符串的最长公共前缀 (Longest Common Prefix of K Strings After Removal)
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        class trie:
            def __init__(self):
                self.children = [None] * 26
                self.cnt = 0
                self.pre = 0

            def insert(self, s: str) -> None:
                node = self
                for i, c in enumerate(s):
                    idx = ord(c) - ord("a")
                    if not node.children[idx]:
                        node.children[idx] = trie()
                    node = node.children[idx]
                    node.cnt += 1
                    node.pre = i + 1

            def call_total(self) -> None:
                self.dfs(self)

            def dfs(self, node) -> None:
                if node is None:
                    return
                if node.cnt >= k:
                    sl.add(node.pre)
                for nxt in node.children:
                    self.dfs(nxt)

            def delete(self, s: str) -> None:
                node = self
                for c in s:
                    idx = ord(c) - ord("a")
                    node = node.children[idx]
                    node.cnt -= 1
                    if node.cnt == k - 1:
                        sl.discard(node.pre)

            def add(self, s: str) -> None:
                node = self
                for c in s:
                    idx = ord(c) - ord("a")
                    node = node.children[idx]
                    node.cnt += 1
                    if node.cnt == k:
                        sl.add(node.pre)

            def get_max(self) -> int:
                return sl[-1]

        n = len(words)
        res = [0] * n
        if n <= k:
            return res
        sl = SortedList()
        sl.add(0)
        root = trie()
        for s in words:
            root.insert(s)
        root.call_total()
        for i, s in enumerate(words):
            root.delete(s)
            res[i] = root.get_max()
            root.add(s)
        return res

    # 3612. 用特殊操作处理字符串 I (Process String with Special Operations I)
    def processStr(self, s: str) -> str:
        flag = True
        q = deque()
        for c in s:
            if c.islower():
                if flag:
                    q.append(c)
                else:
                    q.appendleft(c)
            elif c == "*":
                if q:
                    if flag:
                        q.pop()
                    else:
                        q.popleft()
            elif c == "#":
                q.extend(q)
            else:
                flag = not flag
        if not flag:
            q = reversed(q)
        return "".join(q)

    # 3613. 最小化连通分量的最大成本 (Minimize Maximum Component Cost)
    def minCost(self, n: int, edges: List[List[int]], k: int) -> int:
        class Union:
            def __init__(self, n: int):
                self.parent = [i for i in range(n)]
                self.rank = [1] * n
                self.cnt = n

            def get_root(self, p: int) -> int:
                if p == self.parent[p]:
                    return p
                self.parent[p] = self.get_root(self.parent[p])
                return self.parent[p]

            def is_connected(self, p1: int, p2: int) -> bool:
                return self.get_root(p1) == self.get_root(p2)

            def union(self, p1: int, p2: int) -> None:
                r1 = self.get_root(p1)
                r2 = self.get_root(p2)
                if r1 == r2:
                    return
                if self.rank[r1] < self.rank[r2]:
                    self.parent[r1] = r2
                else:
                    self.parent[r2] = r1
                    if self.rank[r1] == self.rank[r2]:
                        self.rank[r1] += 1
                self.cnt -= 1

            def get_cnt(self) -> int:
                return self.cnt

        def check(t: int) -> bool:
            un = Union(n)
            for u, v, w in edges:
                if w <= t:
                    un.union(u, v)
                    if un.get_cnt() <= k:
                        return True
            return un.get_cnt() <= k

        left = 0
        right = 10**6
        res = 0
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = mid
                right = mid - 1
            else:
                left = mid + 1
        return res

    # 3614. 用特殊操作处理字符串 II (Process String with Special Operations II)
    def processStr(self, s: str, k: int) -> str:
        n = 0
        for c in s:
            if c == "*":
                n = max(n - 1, 0)
            elif c == "#":
                n <<= 1
            elif c != "%":
                n += 1
        if n <= k:
            return "."
        for c in reversed(s):
            if c == "*":
                n += 1
            elif c == "%":
                k = n - k - 1
            elif c == "#":
                n >>= 1
                if n <= k:
                    k -= n
            else:
                n -= 1
                if n == k:
                    return c

    # 875. 爱吃香蕉的珂珂 (Koko Eating Bananas)
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        def check(k: int) -> bool:
            return sum((x + k - 1) // k for x in piles) <= h

        left = 1
        right = max(piles)
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return right + 1

    # 911. 在线选举 (Online Election)
    class TopVotedCandidate:

        def __init__(self, persons: List[int], times: List[int]):
            self.times = times
            self.d = defaultdict(int)
            top = -1
            self.d[-1] = -1
            self.a = []
            for p in persons:
                self.d[p] += 1
                if self.d[p] >= self.d[top]:
                    top = p
                self.a.append(top)

        def q(self, t: int) -> int:
            left = 0
            right = len(self.times)
            while left < right:
                mid = left + ((right - left) >> 1)
                if self.times[mid] <= t:
                    left = mid + 1
                else:
                    right = mid
            return self.a[left - 1]

    # 2163. 删除元素后和的最小差值 (Minimum Difference in Sums After Removal of Elements)
    def minimumDifference(self, nums: List[int]) -> int:
        n = len(nums)
        q = []
        heapq.heapify(q)
        pre_min = [0] * n
        s = 0
        for i, v in enumerate(nums):
            s += v
            heapq.heappush(q, -v)
            if len(q) > n // 3:
                s += heapq.heappop(q)
            pre_min[i] = s
        q.clear()
        s = 0
        res = inf
        for i in range(n - 1, n // 3 - 1, -1):
            heapq.heappush(q, nums[i])
            s += nums[i]
            if len(q) > n // 3:
                s -= heapq.heappop(q)
            if len(q) == n // 3:
                res = min(res, pre_min[i - 1] - s)
        return res

    # 1233. 删除子文件夹 (Remove Sub-Folders from the Filesystem)
    def removeSubfolders(self, folder: List[str]) -> List[str]:
        class trie:
            def __init__(self):
                self.child = defaultdict(trie)
                self.pos = -1

            def insert(self, s: str, p: int) -> None:
                node = self
                a = s.split("/")
                for sub in a:
                    if sub not in node.child:
                        node.child[sub] = trie()
                    node = node.child[sub]
                node.pos = p

            def search(self, node) -> None:
                def dfs(node):
                    if node.pos != -1:
                        res.append(node.pos)
                        return
                    for child in node.child.values():
                        dfs(child)

                res = []
                dfs(node)
                return res

        root = trie()
        for i, f in enumerate(folder):
            root.insert(f, i)
        return [folder[i] for i in root.search(root)]

    # 1233. 删除子文件夹 (Remove Sub-Folders from the Filesystem)
    def removeSubfolders(self, folder: List[str]) -> List[str]:
        folder.sort()
        res = [folder[0]]
        for f in folder[1:]:
            if not f.startswith(res[-1] + "/"):
                res.append(f)
        return res

    # 1957. 删除字符使字符串变好 (Delete Characters to Make Fancy String)
    def makeFancyString(self, s: str) -> str:
        res = []
        pre = "_"
        cnt = 0
        for c in s:
            if pre != c:
                pre = c
                cnt = 0
            cnt += 1
            if cnt < 3:
                res.append(c)
        return "".join(res)

    # 3618. 根据质数下标分割数组 (Split Array by Prime Indices)
    def splitArray(self, nums: List[int]) -> int:
        mx = len(nums)
        is_prime = [True] * (mx + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(mx**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, mx + 1, i):
                    is_prime[j] = False

        res = 0
        for i, x in enumerate(nums):
            if is_prime[i]:
                res += x
            else:
                res -= x
        return abs(res)

    # 3619. 总价值可以被 K 整除的岛屿数目 (Count Islands With Total Value Divisible by K)
    def countIslands(self, grid: List[List[int]], k: int) -> int:
        def dfs(x: int, y: int) -> int:
            if x < 0 or x >= m or y < 0 or y >= n or not grid[x][y]:
                return 0
            dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            res = grid[x][y]
            grid[x][y] = 0  # 标记为已访问
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                res += dfs(nx, ny)
            return res % k

        m = len(grid)
        n = len(grid[0])
        res = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    res += not dfs(i, j)
        return res

    # 3620. 恢复网络路径 (Network Recovery Pathways)
    def findMaxPathScore(
        self, edges: List[List[int]], online: List[bool], k: int
    ) -> int:
        def dfs(x: int, s: int, t: int) -> bool:
            if x == n - 1:
                return True
            for y, w in g[x]:
                if online[y] and s + w <= k and w >= t and dfs(y, s + w, t):
                    return True
            return False

        n = len(online)
        g = [[] for _ in range(n)]
        for u, v, w in edges:
            g[u].append((v, w))
        left = 0
        right = k
        while left <= right:
            mid = left + ((right - left) >> 1)
            if dfs(0, 0, mid):
                left = mid + 1
            else:
                right = mid - 1
        return left - 1

    # 3621. 位计数深度为 K 的整数数目 I (Number of Integers With Popcount-Depth Equal to K I)
    def popcountDepth(self, n: int, k: int) -> int:
        @cache
        def dfs(i: int, j: int, is_limit: bool) -> int:
            if i == m:
                return j == 0
            res = 0
            up = int(s[i]) if is_limit else 1
            for d in range(min(up, j) + 1):
                res += dfs(i + 1, j - d, d == up and is_limit)
            return res

        if k == 0:
            return 1
        res = 0
        s = bin(n)[2:]
        m = len(s)
        if k == 1:
            return m - 1

        @cache
        def cal(i: int) -> int:
            if i == 1:
                return 0
            return cal(i.bit_count()) + 1

        for i in range(1, m + 1):
            if cal(i) == k - 1:
                res += dfs(0, i, True)
        return res

    # 3622. 判断整除性 (Check Divisibility by Digit Sum and Product)
    def checkDivisibility(self, n: int) -> bool:
        def check(n: int) -> int:
            s = 0
            m = 1
            while n:
                s += n % 10
                m *= n % 10
                n //= 10
            return s + m

        return n % check(n) == 0

    # 3623. 统计梯形的数目 I (Count Number of Trapezoids I)
    def countTrapezoids(self, points: List[List[int]]) -> int:
        MOD = 10**9 + 7
        d = defaultdict(int)
        for _, y in points:
            d[y] += 1
        res = 0
        cnt = 0
        for v in d.values():
            cur = v * (v - 1) // 2
            res += cur * cnt
            res %= MOD
            cnt += cur
        return res

    # 3625. 统计梯形的数目 II (Count Number of Trapezoids II)
    def countTrapezoids(self, points: List[List[int]]) -> int:
        cnt = defaultdict(lambda: defaultdict(int))  # 斜率 -> 截距 -> 个数
        cnt2 = defaultdict(lambda: defaultdict(int))  # 中点 -> 斜率 -> 个数

        for i, (x, y) in enumerate(points):
            for x2, y2 in points[:i]:
                dy = y - y2
                dx = x - x2
                k = dy / dx if dx else inf
                b = (y * dx - x * dy) / dx if dx else x
                cnt[k][b] += 1  # 按照斜率和截距分组
                cnt2[(x + x2, y + y2)][k] += 1  # 按照中点和斜率分组

        ans = 0
        for m in cnt.values():
            s = 0
            for c in m.values():
                ans += s * c
                s += c

        for m in cnt2.values():
            s = 0
            for c in m.values():
                ans -= s * c  # 平行四边形会统计两次，减去多统计的一次
                s += c

        return ans

    # 3490. 统计美丽整数的数目 (Count Beautiful Numbers)
    def beautifulNumbers(self, l: int, r: int) -> int:
        def cal(x: int) -> int:
            @cache
            def dfs(i: int, j: int, k: int, is_limit: bool, is_num: bool) -> int:
                if i == n:
                    return is_num and j % k == 0
                res = 0
                if not is_num:
                    res = dfs(i + 1, j, k, False, False)
                up = int(s[i]) if is_limit else 9
                for d in range(0 if is_num else 1, up + 1):
                    res += dfs(i + 1, j * d, k + d, d == up and is_limit, True)
                return res

            s = str(x)
            n = len(s)
            return dfs(0, 1, 0, True, False)

        return cal(r) - cal(l - 1)

    # 1717. 删除子字符串的最大得分 (Maximum Score From Removing Substrings)
    def maximumGain(self, s: str, x: int, y: int) -> int:
        def check(s: str, x: int, y: int) -> int:
            res = 0
            cnt_a = 0
            cnt_b = 0
            for c in s:
                if c == "b":
                    if cnt_a:
                        cnt_a -= 1
                        res += x
                    else:
                        cnt_b += 1
                elif c == "a":
                    cnt_a += 1
                else:
                    res += min(cnt_a, cnt_b) * y
                    cnt_a = 0
                    cnt_b = 0
            return res

        return max(check(s + "_", x, y), check(s[::-1] + "_", y, x))

    # 2322. 从树中删除边的最小分数 (Minimum Score After Removals on a Tree)
    def minimumScore(self, nums: List[int], edges: List[List[int]]) -> int:
        def dfs(x: int, fa: int) -> int:
            nonlocal clock
            _in[x] = clock
            clock += 1
            _xor[x] = nums[x]
            for y in g[x]:
                if y != fa:
                    _xor[x] ^= dfs(y, x)
            _out[x] = clock
            return _xor[x]

        n = len(nums)
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        _in = [0] * n
        _out = [0] * n
        _xor = [0] * n
        clock = 0
        dfs(0, -1)
        res = inf
        for i in range(2, n):
            for j in range(1, i):
                # j 是 i 的祖先
                if _in[j] < _in[i] < _out[j]:
                    x = _xor[i]
                    y = _xor[j] ^ _xor[i]
                    z = _xor[0] ^ _xor[j]
                # i 是 j 的祖先
                elif _in[i] < _in[j] < _out[i]:
                    x = _xor[j]
                    y = _xor[i] ^ _xor[j]
                    z = _xor[0] ^ _xor[i]
                # i 和 j 没有祖先关系
                else:
                    x = _xor[i]
                    y = _xor[j]
                    z = _xor[0] ^ _xor[i] ^ _xor[j]
                res = min(res, max(x, y, z) - min(x, y, z))
                if res == 0:
                    return 0
        return res

    # 42. 接雨水 (Trapping Rain Water)
    def trap(self, height: List[int]) -> int:
        n = len(height)
        left = [0] * n
        for i in range(1, n):
            left[i] = max(left[i - 1], height[i - 1])
        res = 0
        right = 0
        for i in range(n - 2, -1, -1):
            right = max(right, height[i + 1])
            res += max(0, min(left[i], right) - height[i])
        return res

    # 2210. 统计数组中峰和谷的数量 (Count Hills and Valleys in an Array)
    def countHillValley(self, nums: List[int]) -> int:
        a = []
        n = len(nums)
        i = 0
        while i < n:
            a.append(nums[i])
            j = i
            while j < n and nums[j] == nums[i]:
                j += 1
            i = j
        n = len(a)
        return sum(
            a[i - 1] < a[i] > a[i + 1] or a[i - 1] > a[i] < a[i + 1]
            for i in range(1, n - 1)
        )

    # 2044. 统计按位或能得到最大值的子集数目 (Count Number of Maximum Bitwise-OR Subsets)
    def countMaxOrSubsets(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return j == mx
            return dfs(i + 1, j) + dfs(i + 1, j | nums[i])

        mx = 0
        for x in nums:
            mx |= x
        n = len(nums)
        return dfs(0, 0)

    # 2044. 统计按位或能得到最大值的子集数目 (Count Number of Maximum Bitwise-OR Subsets)
    def countMaxOrSubsets(self, nums: List[int]) -> int:
        mx = 0
        for x in nums:
            mx |= x
        n = len(nums)
        s = [0] * (1 << n)
        res = 0
        for i in range(1, 1 << n):
            lb = (i & -i).bit_length() - 1
            s[i] = s[i ^ (1 << lb)] | nums[lb]
            res += s[i] == mx
        return res

    # 3627. 中位数之和的最大值 (Maximum Median Sum of Subsequences of Size 3)
    def maximumMedianSum(self, nums: List[int]) -> int:
        nums.sort()
        return sum(nums[i] for i in range(len(nums) - 2, len(nums) // 3 - 1, -2))

    # 3628. 插入一个字母的最大子序列数 (Maximum Number of Subsequences After One Inserting)
    def numOfSubsequences(self, s: str) -> int:
        def cal(s: str) -> int:
            l_cnt = 0
            c_cnt = 0
            res = 0
            for x in s:
                if x == "L":
                    l_cnt += 1
                elif x == "C":
                    c_cnt += l_cnt
                elif x == "T":
                    res += c_cnt
            return res

        def check() -> int:
            mx = 0
            n = len(s)
            left = [0] * n
            left[0] = s[0] == "L"
            for i in range(1, n):
                left[i] = left[i - 1] + (s[i] == "L")
            right = 0
            for i in range(n - 1, -1, -1):
                right += s[i] == "T"
                mx = max(mx, left[i] * right)
            return mx + cal(s)

        return max(cal("L" + s), cal(s + "T"), check())


# 3629. 通过质数传送到达终点的最少跳跃次数 (Minimum Jumps to Reach End via Prime Teleportation)
MX = 1_000_001
PRIME_FACTORS = [[] for _ in range(MX)]
for i in range(2, MX):
    if not PRIME_FACTORS[i]:  # i 是质数
        for j in range(i, MX, i):  # i 的倍数有质因子 i
            PRIME_FACTORS[j].append(i)


class Solution:
    def minJumps(self, nums: List[int]) -> int:
        n = len(nums)
        groups = defaultdict(list)
        for i, x in enumerate(nums):
            for p in PRIME_FACTORS[x]:
                groups[p].append(i)  # 对于质数 p，可以跳到下标 i
        ans = 0
        vis = [False] * n
        vis[0] = True
        q = deque()
        q.append(0)

        while True:
            s = len(q)
            for _ in range(s):
                x = q.popleft()
                if x == n - 1:
                    return ans
                idx = groups[nums[x]]
                idx.append(x + 1)
                if x:
                    idx.append(x - 1)
                for j in idx:  # 可以从 i 跳到 j
                    if not vis[j]:
                        vis[j] = True
                        q.append(j)
                idx.clear()  # 避免重复访问下标列表
            ans += 1

    # 2419. 按位与最大的最长子数组 (Longest Subarray With Maximum Bitwise AND)
    def longestSubarray(self, nums: List[int]) -> int:
        mx = max(nums)
        cnt = 0
        res = 0
        for x in nums:
            if mx == x:
                cnt += 1
            else:
                res = max(res, cnt)
                cnt = 0
        return max(res, cnt)

    # 2683. 相邻值的按位异或 (Neighboring Bitwise XOR)
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        x = 0
        for d in derived:
            x ^= d
        return x == 0

    # 118. 杨辉三角 (Pascal's Triangle)
    def generate(self, numRows: int) -> List[List[int]]:
        res = [[1]]
        for _ in range(numRows - 1):
            cur = res[-1] + [1]
            for i in range(1, len(cur) - 1):
                cur[i] = res[-1][i - 1] + res[-1][i]
            res.append(cur)
        return res

    # 2561. 重排水果 (Rearranging Fruits)
    def minCost(self, basket1: List[int], basket2: List[int]) -> int:
        cnt = defaultdict(int)
        for x, y in zip(basket1, basket2):
            cnt[x] += 1
            cnt[y] -= 1

        a = []
        for x, c in cnt.items():
            if c & 1:
                return -1
            a.extend([x] * (abs(c) // 2))

        a.sort()
        mn = min(cnt)

        return sum(min(x, mn * 2) for x in a[: len(a) // 2])

    # 3633. 最早完成陆地和水上游乐设施的时间 I (Earliest Finish Time for Land and Water Rides I)
    # 3635. 最早完成陆地和水上游乐设施的时间 II (Earliest Finish Time for Land and Water Rides II)
    def earliestFinishTime(
        self,
        landStartTime: List[int],
        landDuration: List[int],
        waterStartTime: List[int],
        waterDuration: List[int],
    ) -> int:
        res = inf
        end = inf
        for s, d in zip(landStartTime, landDuration):
            end = min(end, s + d)
        for s, d in zip(waterStartTime, waterDuration):
            res = min(res, max(s, end) + d)

        end = inf
        for s, d in zip(waterStartTime, waterDuration):
            end = min(end, s + d)
        for s, d in zip(landStartTime, landDuration):
            res = min(res, max(s, end) + d)
        return res

    # 3634. 使数组平衡的最少移除数目 (Minimum Removals to Balance Array)
    def minRemoval(self, nums: List[int], k: int) -> int:
        nums.sort()
        res = 0
        j = 0
        for i, v in enumerate(nums):
            while nums[j] * k < v:
                j += 1
            res = max(res, i - j + 1)
        return len(nums) - res

    # 3637. 三段式数组 I (Trionic Array I)
    def isTrionic(self, nums: List[int]) -> bool:
        i = 1
        n = len(nums)
        while i < n and nums[i] > nums[i - 1]:
            i += 1
        if i == 1 or i == n or nums[i] == nums[i - 1]:
            return False
        j = n - 2
        while j >= 0 and nums[j] < nums[j + 1]:
            j -= 1
        if j == n - 2 or j < 0 or nums[j] == nums[j + 1]:
            return False
        return all(nums[k] > nums[k + 1] for k in range(i - 1, j + 1))

    # 3637. 三段式数组 I (Trionic Array I)
    def isTrionic(self, nums: List[int]) -> bool:
        def dfs(i: int, j: int) -> bool:
            if i == n - 1:
                return j == 2
            if (
                (j == 0 or j == 2)
                and nums[i + 1] > nums[i]
                or j == 1
                and nums[i + 1] < nums[i]
            ):
                return dfs(i + 1, j)
            if (
                j == 0
                and nums[i + 1] < nums[i]
                and i
                or j == 1
                and nums[i + 1] > nums[i]
            ):
                return dfs(i + 1, j + 1)
            return False

        n = len(nums)
        return dfs(0, 0)

    # 3639. 变为活跃状态的最小时间 (Minimum Time to Activate String)
    def minTime(self, _: str, order: List[int], k: int) -> int:
        def check(t: int) -> bool:
            a = [-1] * n
            for i in range(t + 1):
                a[order[i]] = 0
            idx_last_star = -1
            res = 0
            for i in range(n):
                if a[i] != -1:
                    idx_last_star = i
                res += idx_last_star + 1
                if res >= k:
                    return True
            return False

        n = len(order)
        if k > (n + 1) * n // 2:
            return -1
        left = 0
        right = n - 1
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return right + 1

    # 3638. 平衡装运的最大数量 (Maximum Balanced Shipments)
    def maxBalancedShipments(self, weight: List[int]) -> int:
        res = 0
        mx = 0
        for w in weight:
            if w < mx:
                res += 1
                mx = 0
            else:
                mx = max(mx, w)
        return res

    # 3640. 三段式数组 II (Trionic Array II)
    def maxSumTrionic(self, nums: List[int]) -> int:
        # 以i结尾，且目前状态是j (j == 0，表示最后一组是第一次上升状态；j == 1，表示最后一组是下降状态，j == 2，表示最后一组是第二次上升状态)的最大和。
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n - 1:
                return 0 if j == 2 else -inf
            res = -inf
            if j == 0 and nums[i + 1] > nums[i] or j == 1 and nums[i + 1] < nums[i]:
                return dfs(i + 1, j) + nums[i + 1]
            if j == 2:
                return (
                    max(0, dfs(i + 1, j) + nums[i + 1]) if nums[i + 1] > nums[i] else 0
                )
            if j == 0 and nums[i + 1] < nums[i] or j == 1 and nums[i + 1] > nums[i]:
                return dfs(i + 1, j + 1) + nums[i + 1]
            return res

        n = len(nums)
        return max(
            dfs(i, 0) + nums[i - 1] + nums[i]
            for i in range(1, n)
            if nums[i] > nums[i - 1]
        )

    # 3477. 水果成篮 II (Fruits Into Baskets II)
    # 3479. 水果成篮 III (Fruits Into Baskets III) --线段树 （单点更新）
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        class SegmentTree:
            def __init__(self, a: List[int]):
                n = len(a)
                self.max = [0] * (n * 4)
                self.build(a, 1, 0, n - 1)

            def build(self, a: List[int], o: int, l: int, r: int):
                if l == r:
                    self.max[o] = a[l]
                    return
                m = l + ((r - l) >> 1)
                self.build(a, o * 2, l, m)
                self.build(a, o * 2 + 1, m + 1, r)
                self.maintain(o)

            def find(self, o: int, l: int, r: int, x: int) -> int:
                if self.max[o] < x:
                    return -1
                if l == r:
                    self.max[o] = -1
                    return l
                m = l + ((r - l) >> 1)
                i = self.find(o * 2, l, m, x)
                if i < 0:
                    i = self.find(o * 2 + 1, m + 1, r, x)
                self.maintain(o)
                return i

            def maintain(self, o: int):
                self.max[o] = max(self.max[o * 2], self.max[o * 2 + 1])

        t = SegmentTree(baskets)
        n = len(baskets)
        res = 0
        for x in fruits:
            if t.find(1, 0, n - 1, x) < 0:
                res += 1
        return res

    # 307. 区域和检索 - 数组可修改 (Range Sum Query - Mutable)
    class NumArray:

        def __init__(self, nums: List[int]):
            self.n = len(nums)
            self.t = SegmentTree307(nums)

        def update(self, index: int, val: int) -> None:
            self.t.update(1, index, val, 0, self.n - 1)

        def sumRange(self, left: int, right: int) -> int:
            return self.t.sumRange(1, left, right, 0, self.n - 1)


class SegmentTree307:
    def __init__(self, a: List[int]):
        n = len(a)
        self.s = [0] * (n * 4)
        self.build(a, 1, 0, n - 1)

    def build(self, a: List[int], o: int, l: int, r: int):
        if l == r:
            self.s[o] = a[l]
            return
        m = l + ((r - l) >> 1)
        self.build(a, o * 2, l, m)
        self.build(a, o * 2 + 1, m + 1, r)
        self.maintain(o)

    def maintain(self, o: int):
        self.s[o] = self.s[o * 2] + self.s[o * 2 + 1]

    def update(self, o: int, index: int, val: int, l: int, r: int):
        if l == r:
            self.s[o] = val
            return
        m = l + ((r - l) >> 1)
        if index <= m:
            self.update(o * 2, index, val, l, m)
        else:
            self.update(o * 2 + 1, index, val, m + 1, r)
        self.maintain(o)

    def sumRange(self, o: int, L: int, R: int, l: int, r: int) -> int:
        if L <= l and r <= R:
            return self.s[o]
        m = l + ((r - l) >> 1)
        if R <= m:
            return self.sumRange(o * 2, L, R, l, m)
        if L >= m + 1:
            return self.sumRange(o * 2 + 1, L, R, m + 1, r)
        return self.sumRange(o * 2, L, R, l, m) + self.sumRange(
            o * 2 + 1, L, R, m + 1, r
        )

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

    # 2940. 找到 Alice 和 Bob 可以相遇的建筑 (Find Building Where Alice and Bob Can Meet)
    def leftmostBuildingQueries(
        self, heights: List[int], queries: List[List[int]]
    ) -> List[int]:
        res = []
        n = len(heights)
        t = SegmentTree2940(heights)
        for a, b in queries:
            if b < a:
                a, b = b, a
            if a == b or heights[a] < heights[b]:
                res.append(b)
            else:
                res.append(
                    t.find(1, b + 1, n - 1, 0, n - 1, max(heights[a], heights[b]) + 1)
                )
        return res


class SegmentTree2940:

    def __init__(self, a: List[int]):
        n = len(a)
        self.mx = [0] * (n * 4)
        self.build(a, 1, 0, n - 1)

    def build(self, a: List[int], o: int, l: int, r: int):
        if l == r:
            self.mx[o] = a[l]
            return
        m = l + ((r - l) >> 1)
        self.build(a, o * 2, l, m)
        self.build(a, o * 2 + 1, m + 1, r)
        self.maintain(o)

    def maintain(self, o: int):
        self.mx[o] = max(self.mx[o * 2], self.mx[o * 2 + 1])

    def find(self, o: int, L: int, R: int, l: int, r: int, x: int) -> int:
        if self.mx[o] < x:
            return -1
        if l == r:
            return l

        m = l + ((r - l) >> 1)
        if R <= m:
            return self.find(o * 2, L, R, l, m, x)
        if L >= m + 1:
            return self.find(o * 2 + 1, L, R, m + 1, r, x)
        i = self.find(o * 2, L, R, l, m, x)
        if i < 0:
            i = self.find(o * 2 + 1, L, R, m + 1, r, x)
        return i

    # 2286. 以组为单位订音乐会的门票 (Booking Concert Tickets in Groups)
    class BookMyShow:

        def __init__(self, n: int, m: int):
            self.m = m
            self.n = n
            self.min = [0] * (n * 4)
            self.sum = [0] * (n * 4)

        def find_first(self, o: int, l: int, r: int, L: int, R: int, x: int) -> int:
            if self.min[o] > x:
                return -1
            if l == r:
                return l
            m = l + ((r - l) >> 1)
            if R <= m:
                return self.find_first(o * 2, l, m, L, R, x)
            if L >= m + 1:
                return self.find_first(o * 2 + 1, m + 1, r, L, R, x)
            i = self.find_first(o * 2, l, m, L, R, x)
            if i < 0:
                i = self.find_first(o * 2 + 1, m + 1, r, L, R, x)
            return i

        def query_sum(self, o: int, l: int, r: int, L: int, R: int) -> int:
            if L <= l and r <= R:
                return self.sum[o]
            m = l + ((r - l) >> 1)
            if R <= m:
                return self.query_sum(o * 2, l, m, L, R)
            if L >= m + 1:
                return self.query_sum(o * 2 + 1, m + 1, r, L, R)
            return self.query_sum(o * 2, l, m, L, R) + self.query_sum(
                o * 2 + 1, m + 1, r, L, R
            )

        def update(self, o: int, l: int, r: int, i: int, val: int):
            if l == r:
                self.min[o] += val
                self.sum[o] += val
                return
            m = l + ((r - l) >> 1)
            if i <= m:
                self.update(o * 2, l, m, i, val)
            else:
                self.update(o * 2 + 1, m + 1, r, i, val)
            self.min[o] = min(self.min[o * 2], self.min[o * 2 + 1])
            self.sum[o] = self.sum[o * 2] + self.sum[o * 2 + 1]

        def gather(self, k: int, maxRow: int) -> List[int]:
            r = self.find_first(1, 0, self.n - 1, 0, maxRow, self.m - k)
            if r < 0:
                return []
            c = self.query_sum(1, 0, self.n - 1, r, r)
            self.update(1, 0, self.n - 1, r, k)
            return [r, c]

        def scatter(self, k: int, maxRow: int) -> bool:
            s = self.query_sum(1, 0, self.n - 1, 0, maxRow)
            if self.m * (maxRow + 1) - s < k:
                return False
            i = self.find_first(1, 0, self.n - 1, 0, maxRow, self.m - 1)
            while k:
                left = min(k, self.m - self.query_sum(1, 0, self.n - 1, i, i))
                self.update(1, 0, self.n - 1, i, left)
                k -= left
                i += 1
            return True

    # 3459. 最长 V 形对角线段的长度 (Length of Longest V-Shaped Diagonal Segment)
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        # 当前在 (i, j) 方向 d 上，是否可以转弯can_turn，当前数字期望是 t
        @cache
        def dfs(i: int, j: int, d: int, can_turn: bool, t: int) -> int:
            if not (m > i >= 0 and n > j >= 0) or grid[i][j] != t:
                return 0
            t = 2 - t
            # 不转向
            ni = i + DIRS[d][0]
            nj = j + DIRS[d][1]
            res = dfs(ni, nj, d, can_turn, t)

            # 转向
            if can_turn:
                d = (d + 1) % 4
                d_max = (
                    min(j, i),
                    min(i, n - j - 1),
                    min(n - j - 1, m - i - 1),
                    min(m - i - 1, j),
                )
                if d_max[d] > res:  # 剪枝
                    ni = i + DIRS[d][0]
                    nj = j + DIRS[d][1]
                    res = max(res, dfs(ni, nj, d, False, t))
            return res + 1

        m, n = len(grid), len(grid[0])
        res = 0
        # 左上, 右上, 右下, 左下
        DIRS = (-1, -1), (-1, 1), (1, 1), (1, -1)
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    d_max = (i + 1, n - j, m - i, j + 1)
                    for d, (dx, dy) in enumerate(DIRS):
                        if res >= d_max[d]:  # 剪枝
                            continue
                        x, y = i + dx, j + dy
                        res = max(res, dfs(x, y, d, True, 2) + 1)
        return res

    # 869. 重新排序得到 2 的幂 (Reordered Power of 2)
    def reorderedPowerOf2(self, n: int) -> bool:
        def cal(x: int) -> List[int]:
            cnts = [0] * 10
            while x:
                cnts[x % 10] += 1
                x //= 10
            return cnts

        a = cal(n)
        return any(cal(1 << i) == a for i in range(30))

    # 2438. 二的幂数组中查询范围内的乘积 (Range Product Queries of Powers)
    def productQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        a = [i for i in range(31) if (n >> i) & 1]
        a = list(accumulate(a, initial=0))
        MOD = 10**9 + 7
        return [(1 << (a[y + 1] - a[x])) % MOD for x, y in queries]

    def numberOfWays(self, n: int, x: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == 0:
                return 1
            if j <= 0:
                return 0
            res = dfs(i, j - 1)
            left = i - pow(j, x)
            if left >= 0:
                res += dfs(left, j - 1)
            return res % MOD

        MOD = 10**9 + 7
        res = dfs(n, n)
        dfs.cache_clear()
        return res

    # 3643. 垂直翻转子矩阵 (Flip Square Submatrix Vertically)
    def reverseSubmatrix(
        self, grid: List[List[int]], x: int, y: int, k: int
    ) -> List[List[int]]:
        for i in range(x, x + k // 2):
            last = x + k - (i - x + 1)
            grid[i][y : y + k], grid[last][y : y + k] = (
                grid[last][y : y + k],
                grid[i][y : y + k],
            )
        return grid

    # 3644. 排序排列 (Maximum K to Sort a Permutation)
    def sortPermutation(self, nums: List[int]) -> int:
        res = -1
        for i, v in enumerate(nums):
            if i != v:
                res &= v
        return max(0, res)

    # 3645. 最优激活顺序得到的最大总和 (Maximum Total from Optimal Activation Order)
    def maxTotal(self, value: List[int], limit: List[int]) -> int:
        d = defaultdict(list)
        for v, l in zip(value, limit):
            d[l].append(v)
        res = 0
        for l, a in d.items():
            a.sort()
            res += sum(a[-l:])
        return res

    # 326. 3 的幂 (Power of Three)
    def isPowerOfThree(self, n: int) -> bool:
        if n <= 0:
            return False
        while n != 1:
            if n % 3:
                return False
            n //= 3
        return True

    # 1323. 6 和 9 组成的最大数字 (Maximum 69 Number)
    def maximum69Number(self, num: int) -> int:
        s = str(num)
        a = [x for x in s]
        for i, x in enumerate(a):
            if x == "6":
                a[i] = "9"
                break
        return int("".join(a))

    # 3648. 覆盖网格的最少传感器数目 (Minimum Sensors to Cover Grid)
    def minSensors(self, n: int, m: int, k: int) -> int:
        def check(a: int) -> int:
            b = 2 * k + 1
            return (a + b - 1) // b

        return check(n) * check(m)

    # 3649. 完美对的数目 (Number of Perfect Pairs)
    def perfectPairs(self, nums: List[int]) -> int:
        a = [abs(x) for x in nums]
        a.sort()
        j = 0
        res = 0
        for i, v in enumerate(a):
            while v > a[j] * 2:
                j += 1
            res += i - j
        return res

    # 3650. 边反转的最小路径总成本 (Minimum Cost Path with Edge Reversals)
    def minCost(self, n: int, edges: List[List[int]]) -> int:
        g = [[] for _ in range(n)]
        for u, v, w in edges:
            g[u].append((v, w))
            g[v].append((u, w * 2))
        dis = [inf] * n
        dis[0] = 0
        q = [(0, 0)]
        heapq.heapify(q)
        while q:
            w, x = heapq.heappop(q)
            if x == n - 1:
                return w
            for y, dw in g[x]:
                nw = w + dw
                if nw < dis[y]:
                    dis[y] = nw
                    heapq.heappush(q, (nw, y))
        return -1

    # 3651. 带传送的最小路径成本 (Minimum Cost Path with Teleportations)
    def minCost(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        a = sorted((grid[i][j], i, j) for j in range(n) for i in range(m))
        dis = [[[inf] * (k + 1) for _ in range(n)] for _ in range(m)]
        ptrs = [0] * (k + 1)
        dis[0][0][0] = 0
        q = [(0, 0, 0, 0)]
        heapq.heapify(q)
        dirs = (0, 1), (1, 0)
        while q:
            w, x, y, c = heapq.heappop(q)
            if dis[x][y][c] > w:
                continue
            if x == m - 1 and y == n - 1:
                return w
            for dx, dy in dirs:
                nx = x + dx
                ny = y + dy
                if nx < m and ny < n:
                    if w + grid[nx][ny] < dis[nx][ny][c]:
                        dis[nx][ny][c] = w + grid[nx][ny]
                        heapq.heappush(q, (w + grid[nx][ny], nx, ny, c))
            if c < k:
                v = grid[x][y]
                p = ptrs[c]
                while p < len(a) and a[p][0] <= v:
                    nx, ny = a[p][1], a[p][2]
                    if w < dis[nx][ny][c + 1]:
                        dis[nx][ny][c + 1] = w
                        heapq.heappush(q, (w, nx, ny, c + 1))
                    p += 1
                ptrs[c] = p
        return -1

    # 679. 24 点游戏 (24 Game)
    def judgePoint24(self, cards: List[int]) -> bool:
        def dfs(a: List[float]) -> bool:
            if len(a) == 1:
                return abs(a[0] - 24) <= 10 ** (-6)
            for i in range(len(a)):
                for j in range(i + 1, len(a)):
                    x = a[i]
                    y = a[j]
                    copy_a = [0] * (len(a) - 1)
                    id = 0
                    for k in range(len(a)):
                        if i != k and j != k:
                            copy_a[id] = a[k]
                            id += 1
                    copy_a[id] = x + y
                    if dfs(copy_a):
                        return True
                    copy_a[id] = x - y
                    if dfs(copy_a):
                        return True
                    copy_a[id] = y - x
                    if dfs(copy_a):
                        return True
                    copy_a[id] = x * y
                    if dfs(copy_a):
                        return True

                    if y > 10 ** (-6):
                        copy_a[id] = x / y
                        if dfs(copy_a):
                            return True
                    if x > 10 ** (-6):
                        copy_a[id] = y / x
                        if dfs(copy_a):
                            return True
            return False

        return dfs(cards)

    # 2348. 全 0 子数组的数目 (Number of Zero-Filled Subarrays)
    def zeroFilledSubarray(self, nums: List[int]) -> int:
        res = 0
        cnt = 0
        for x in nums:
            cnt = 0 if x else cnt + 1
            res += cnt
        return res

    # 1504. 统计全 1 子矩形 (Count Submatrices With All Ones)
    def numSubmat(self, mat: List[List[int]]) -> int:
        res = 0
        m, n = len(mat), len(mat[0])
        for top in range(m):
            a = [0] * n
            for bottom in range(top, m):
                h = bottom - top + 1
                last = -1
                for j in range(n):
                    a[j] += mat[bottom][j]
                    if a[j] != h:
                        last = j
                    else:
                        res += j - last
        return res

    # 3652. 按策略买卖股票的最佳时机 (Best Time to Buy and Sell Stock using Strategy)
    def maxProfit(self, prices: List[int], strategy: List[int], k: int) -> int:
        n = len(prices)
        pre_s = list(accumulate(prices, initial=0))
        pre = list(accumulate([a * b for a, b in zip(prices, strategy)], initial=0))
        res = pre[-1]
        for i in range(k, n + 1):
            res = max(res, pre[i - k] + pre_s[i] - pre_s[i - k // 2] + pre[-1] - pre[i])
        return res

    # 3653. 区间乘法查询后的异或 I (XOR After Range Multiplication Queries I)
    def xorAfterQueries(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        for l, r, k, v in queries:
            for i in range(l, r + 1, k):
                nums[i] = nums[i] * v % MOD
        res = 0
        for x in nums:
            res ^= x
        return res

    # 1504. 统计全 1 子矩形 (Count Submatrices With All Ones)
    def numSubmat(self, mat: List[List[int]]) -> int:
        heights = [0] * len(mat[0])
        ans = 0
        for row in mat:
            for j, x in enumerate(row):
                if x == 0:
                    heights[j] = 0
                else:
                    heights[j] += 1

            # (j, f, heights[j])
            st = [(-1, 0, -1)]  # 哨兵，方便处理 left=-1 的情况
            for j, h in enumerate(heights):
                while st[-1][2] >= h:
                    st.pop()
                left, f, _ = st[-1]
                # 计算底边为 row，右边界为 j 的子矩形个数
                # 左边界 <= left 的矩形，每个矩形的右边界都可以扩展到 j，一共有 f 个
                # 左边界 >  left 的矩形，左边界有 j-left 种，高度有 h 种，一共有 (j-left)*h 个
                f += (j - left) * h
                ans += f
                st.append((j, f, h))
        return ans

    # 3658. 奇数和与偶数和的最大公约数 (GCD of Odd and Even Sums)
    def gcdOfOddEvenSums(self, n: int) -> int:
        return n

    # 3197. 包含所有 1 的最小矩形面积 II (Find the Minimum Area to Cover All Ones II)
    def minimumSum(self, grid: List[List[int]]) -> int:
        # 顺时针旋转90度
        def rotate(a: List[List[int]]) -> List[List[int]]:
            return list(zip(*reversed(a)))
            # return [list(col) for col in zip(*a[::-1])]
            # return [list(reversed(col)) for col in zip(*a)]

        def solve(a: List[List[int]]) -> int:
            def minimum_area(arr: List[List[int]], l: int, r: int) -> int:
                left = top = inf
                right = bottom = 0
                for i in range(len(arr)):
                    for j in range(l, r):
                        if arr[i][j] == 1:
                            left = min(left, j)
                            right = max(right, j)
                            top = min(top, i)
                            bottom = max(bottom, i)
                return (right - left + 1) * (bottom - top + 1)

            m, n = len(a), len(a[0])
            res = inf
            if m >= 3:
                for r1 in range(1, m):
                    for r2 in range(r1 + 1, m):
                        res = min(
                            res,
                            minimum_area(a[:r1], 0, n)
                            + minimum_area(a[r1:r2], 0, n)
                            + minimum_area(a[r2:], 0, n),
                        )
            if m >= 2 and n >= 2:
                for i in range(1, m):
                    for j in range(1, n):
                        res = min(
                            res,
                            minimum_area(a[:i], 0, n)
                            + minimum_area(a[i:], 0, j)
                            + minimum_area(a[i:], j, n),
                        )
                        res = min(
                            res,
                            minimum_area(a[:i], 0, j)
                            + minimum_area(a[:i], j, n)
                            + minimum_area(a[i:], 0, n),
                        )
            return res

        return min(solve(grid), solve(rotate(grid)))

    # 498. 对角线遍历 (Diagonal Traverse)
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        m, n = len(mat), len(mat[0])
        res = []
        for s in range(m + n - 1):
            if s & 1 == 0:
                for i in range(min(s, m - 1), max(-1, s - n), -1):
                    res.append(mat[i][s - i])
            else:
                for i in range(max(0, s - n + 1), min(s + 1, m)):
                    res.append(mat[i][s - i])
        return res

    # 498. 对角线遍历 (Diagonal Traverse)
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        m, n = len(mat), len(mat[0])
        res = []
        dic = defaultdict(list)
        for i in range(m):
            for j in range(n):
                dic[i + j].append(mat[i][j])
        for k in range(m + n - 1):
            res.extend(dic[k] if k & 1 else reversed(dic[k]))
        return res

    # 3659. 数组元素分组 (Partition Array Into K-Distinct Groups)
    def partitionArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        if n % k:
            return False
        d = defaultdict(int)
        for x in nums:
            d[x] += 1
            if d[x] > n // k:
                return False
        return True

    # 3660. 跳跃游戏 IX (Jump Game IX)
    def maxValue(self, nums: List[int]) -> List[int]:
        n = len(nums)
        pre_max = [0] * n
        pre_max[0] = nums[0]
        for i in range(1, n):
            pre_max[i] = max(pre_max[i - 1], nums[i])
        suf_min = inf
        mx = 0
        for i in range(n - 1, -1, -1):
            if pre_max[i] <= suf_min:
                mx = pre_max[i]
            suf_min = min(suf_min, nums[i])
            nums[i] = mx
        return nums

    # 3661. 可以被机器人摧毁的最大墙壁数目 (Maximum Walls Destroyed by Robots)
    def maxWalls(self, robots: List[int], distance: List[int], walls: List[int]) -> int:
        # 当前第 i 个机器人，且前一个机器人向 is_right 方向发射子弹，能穿过的最多墙壁数量
        @cache
        def dfs(i: int, is_right: bool) -> int:
            if i == n - 1:
                return 0

            # 当前机器人向右发射子弹
            left = bisect.bisect_left(walls, r[i][0])
            right = bisect.bisect_right(walls, min(r[i + 1][0] - 1, r[i][0] + r[i][1]))
            res = dfs(i + 1, True) + max(0, right - left)

            # 当前机器人向左发射子弹
            left = bisect.bisect_left(
                walls,
                max(
                    r[i - 1][0] + 1 + (r[i - 1][1] if is_right else 0),
                    r[i][0] - r[i][1],
                ),
            )

            right = bisect.bisect_right(walls, r[i][0])
            res = max(res, dfs(i + 1, False) + max(0, right - left))
            return res

        r = [(0, 0)] + sorted(zip(robots, distance)) + [(inf, 0)]
        walls.sort()
        n = len(r)
        return dfs(1, True)

    # 3663. 出现频率最低的数字 (Find The Least Frequent Digit)
    def getLeastFrequentDigit(self, n: int) -> int:
        cnts = [0] * 10
        while n:
            cnts[n % 10] += 1
            n //= 10
        res = 0
        c = inf
        for i, v in enumerate(cnts):
            if v and v < c:
                res = i
                c = v
        return res

    # 37. 解数独 (Sudoku Solver)
    def solveSudoku(self, board: List[List[str]]) -> None:
        def flip(i: int, j: int, d: int):
            rows[i] ^= 1 << d
            cols[j] ^= 1 << d
            boxes[(i // 3) * 3 + j // 3] ^= 1 << d

        n = 9
        rows = [0] * n
        cols = [0] * n
        boxes = [0] * n
        u = (1 << n) - 1
        for i in range(n):
            for j in range(n):
                if board[i][j] != ".":
                    d = int(board[i][j]) - 1
                    flip(i, j, d)
        while True:
            changed = False
            for i in range(n):
                for j in range(n):
                    if board[i][j] != ".":
                        continue
                    mask = rows[i] | cols[j] | boxes[(i // 3) * 3 + j // 3]
                    mask = u ^ mask
                    if mask & (mask - 1) == 0:
                        d = mask.bit_length() - 1
                        board[i][j] = str(d + 1)
                        flip(i, j, d)
                        changed = True
            if not changed:
                break

        empties = []
        for i in range(n):
            for j in range(n):
                if board[i][j] == ".":
                    empties.append((i, j))
        m = len(empties)

        def dfs(i: int) -> bool:
            if i == m:
                return True
            x, y = empties[i]
            mask = rows[x] | cols[y] | boxes[(x // 3) * 3 + y // 3]
            c = u ^ mask
            while c:
                d = (c & -c).bit_length() - 1
                board[x][y] = str(d + 1)
                flip(x, y, d)
                if dfs(i + 1):
                    return True
                board[x][y] = "."
                flip(x, y, d)
                c &= c - 1
            return False

        dfs(0)

    # 1792. 最大平均通过率 (Maximum Average Pass Ratio)
    def maxAverageRatio(self, classes: List[List[int]], extraStudents: int) -> float:
        q = []
        for a, b in classes:
            heapq.heappush(q, (-(b - a) / (b * (b + 1)), a, b))
        for _ in range(extraStudents):
            _, a, b = heapq.heappop(q)
            a += 1
            b += 1
            heapq.heappush(q, (-(b - a) / (b * (b + 1)), a, b))
        return sum(a / b for _, a, b in q) / len(classes)

    # 3665. 统计镜子反射路径数目 (Twisted Mirror Path Count)
    def uniquePaths(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        # 当前在 (i, j) 位置，上一步是往下走(k == 0), 还是往右走(k == 1)
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i == m - 1 and j == n - 1:
                return 1
            if i == m or j == n:
                return 0
            res = 0
            if grid[i][j] == 0 or k == 0:
                res += dfs(i, j + 1, 1)
            if grid[i][j] == 0 or k == 1:
                res += dfs(i + 1, j, 0)
            return res % MOD

        MOD = 10**9 + 7
        return dfs(0, 0, 0)

    # 3668. 重排完成顺序 (Restore Finishing Order)
    def recoverOrder(self, order: List[int], friends: List[int]) -> List[int]:
        s = set(friends)
        return [o for o in order if o in s]

    # 3664. 两个字母卡牌游戏 (Two-Letter Card Game)
    def score(self, cards: List[str], x: str) -> int:
        def get_sum_and_max(cnt: Counter) -> Tuple[int, int]:
            del cnt[x]
            sum_cnt = sum(cnt.values())
            max_cnt = max(cnt.values(), default=0)
            return sum_cnt, max_cnt

        def calc_score(s: int, mx: int, k: int) -> int:
            s += k
            mx = max(mx, k)
            return min(s // 2, s - mx)

        cnt1 = Counter(b for a, b in cards if a == x)  # xa
        cnt2 = Counter(a for a, b in cards if b == x)  # ax
        cnt_xx = cnt1[x]
        sum1, max1 = get_sum_and_max(cnt1)
        sum2, max2 = get_sum_and_max(cnt2)
        res = 0
        for k in range(cnt_xx + 1):
            score1 = calc_score(sum1, max1, k)
            score2 = calc_score(sum2, max2, cnt_xx - k)
            res = max(res, score1 + score2)
        return res

    # 3669. K 因数分解 (Balanced K-Factor Decomposition)
    def minDifference(self, n: int, k: int) -> List[int]:
        def dfs(i: int, j: int, min_val: int, max_val: int):
            if len(a) == k:
                if j == 1:
                    nonlocal res, min_diff
                    if max_val - min_val < min_diff:
                        min_diff = max_val - min_val
                        res = a.copy()
                return
            if i == m:
                return
            dfs(i + 1, j, min_val, max_val)
            if j % fac[i] == 0:
                a.append(fac[i])
                dfs(i, j // fac[i], min(min_val, fac[i]), max(max_val, fac[i]))
                a.pop()

        fac = []
        for i in range(1, isqrt(n) + 1):
            if n % i == 0:
                fac.append(i)
                if i * i != n:
                    fac.append(n // i)
        m = len(fac)
        min_diff = inf
        res = []
        a = []
        dfs(0, n, inf, 0)
        return res

    # 3670. 没有公共位的整数最大乘积 (Maximum Product of Two Integers With No Common Bits)
    def maxProduct(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == 0:
                return 0
            if i in a:
                return i
            u = i
            mx = 0
            while i:
                lb = (i & -i).bit_length() - 1
                mx = max(mx, dfs(u ^ (1 << lb)))
                i &= i - 1
            return mx

        a = set(nums)
        u = 0
        for x in a:
            u |= x
        res = 0
        for x in a:
            res = max(res, x * dfs(u ^ x))
        return res

    # 1824. 最少侧跳次数 (Minimum Sideway Jumps)
    def minSideJumps(self, obstacles: List[int]) -> int:
        n = len(obstacles)

        @cache
        def dfs(i: int, j: int) -> int:
            if i == n - 1:
                return 0
            if obstacles[i + 1] - 1 != j:
                return dfs(i + 1, j)
            return (
                min(dfs(i + 1, k) for k in range(3) if k != j and obstacles[i] - 1 != k)
                + 1
            )

        return dfs(0, 1)

    # 719. 找出第 K 小的数对距离 (Find K-th Smallest Pair Distance)
    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        def check(up_limit: int) -> bool:
            cnt = 0
            j = 0
            for i, v in enumerate(nums):
                while v - nums[j] > up_limit:
                    j += 1
                cnt += i - j
                if cnt >= k:
                    return True
            return False

        nums.sort()
        left = 0
        right = nums[-1] - nums[0]
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return right + 1

    # 1304. 和为零的 N 个不同整数 (Find N Unique Integers Sum up to Zero)
    def sumZero(self, n: int) -> List[int]:
        res = [i + 1 for i in range(n)]
        res[-1] = -sum(res[: n - 1])
        return res

    # 1317. 将整数转换为两个无零整数的和 (Convert Integer to the Sum of Two No-Zero Integers)
    def getNoZeroIntegers(self, n: int) -> List[int]:
        def check(x: int) -> bool:
            while x:
                if x % 10 == 0:
                    return False
                x //= 10
            return True

        for a in range(1, n):
            b = n - a
            if check(a) and check(b):
                return [a, b]

    # 3674. 数组元素相等的最小操作次数 (Minimum Operations to Equalize Array)
    def minOperations(self, nums: List[int]) -> int:
        return int(any(x != y for x, y in pairwise(nums)))

    # 3675. 转换字符串的最小操作次数 (Minimum Operations to Transform String)
    def minOperations(self, s: str) -> int:
        b = 0
        a = ord("a")
        for c in map(ord, s):
            b |= 1 << (c - a)
        # -2的二进制 = 0b(11111111111111111111111111111110) 用于消除b中最低位的1
        b &= -2
        return 0 if b == 0 else 26 - ((b & -b).bit_length() - 1)

    # 3676. 碗子数组的数目 (Count Bowl Subarrays)
    def bowlSubarrays(self, nums: List[int]) -> int:
        st = [0] * len(nums)
        top = -1
        res = 0
        for x in nums:
            while top >= 0 and st[top] < x:
                top -= 1
                if top >= 0:
                    res += 1
            top += 1
            st[top] = x
        return res

    # 3677. 统计二进制回文数字的数目 (Count Binary Palindromic Numbers)
    def countBinaryPalindromes(self, n: int) -> int:
        if n == 0:
            return 1
        res = 1
        m = n.bit_length()
        for i in range(1, m):
            res += 1 << ((i - 1) >> 1)
        for i in range(m - 2, m // 2 - 1, -1):
            if n >> i & 1:
                res += 1 << (i - m // 2)
        pal = n >> (m >> 1)
        v = pal >> (m & 1)
        while v:
            pal = pal * 2 + v % 2
            v >>= 1
        if pal <= n:
            res += 1
        return res

    # 1733. 需要教语言的最少人数 (Minimum Number of People to Teach)
    def minimumTeachings(
        self, _: int, languages: List[List[int]], friendships: List[List[int]]
    ) -> int:
        s = set()
        for u, v in friendships:
            if not set(languages[u - 1]) & set(languages[v - 1]):
                s.add(u - 1)
                s.add(v - 1)
        if not s:
            return 0
        d = defaultdict(int)
        for c in s:
            for l in languages[c]:
                d[l] += 1
        return len(s) - max(d.values())

    # 2785. 将字符串中的元音字母排序 (Sort Vowels in a String)
    def sortVowels(self, s: str) -> str:
        a = sorted([x for x in s if x in "aeiouAEIOU"], reverse=True)
        res = []
        for x in s:
            if x in "aeiouAEIOU":
                res.append(a.pop())
            else:
                res.append(x)
        return "".join(res)

    # 966. 元音拼写检查器 (Vowel Spellchecker)
    def spellchecker(self, wordlist: List[str], queries: List[str]) -> List[str]:
        u = 0
        for c in "aeiou":
            u |= 1 << ((ord(c) - ord("a")))
        s = set(wordlist)
        cap_dic = defaultdict(int)
        vow_dic = defaultdict(int)
        for i, w in enumerate(wordlist):
            lower = w.lower()
            if lower not in cap_dic:
                cap_dic[lower] = i
            a = "".join(["_" if (u >> (ord(c) - ord("a"))) & 1 else c for c in lower])
            if a not in vow_dic:
                vow_dic[a] = i
        for i, q in enumerate(queries):
            if q in s:
                continue
            lower = q.lower()
            if lower in cap_dic:
                queries[i] = wordlist[cap_dic[lower]]
                continue
            a = "".join(["_" if (u >> (ord(c) - ord("a"))) & 1 else c for c in lower])
            if a in vow_dic:
                queries[i] = wordlist[vow_dic[a]]
                continue
            queries[i] = ""
        return queries

    # 1935. 可以输入的最大单词数 (Maximum Number of Words You Can Type)
    def canBeTypedWords(self, text: str, brokenLetters: str) -> int:
        u = 0
        a = ord("a")
        for c in map(ord, brokenLetters):
            u |= 1 << (c - a)
        return sum(
            all(u >> (c - a) & 1 == 0 for c in map(ord, s)) for s in text.split(" ")
        )

    # 3678. 大于平均值的最小未出现正整数 (Smallest Absent Positive Greater Than Average)
    def smallestAbsent(self, nums: List[int]) -> int:
        s = set(nums)
        a = max(1, sum(nums) // len(nums) + 1)
        while a in s:
            a += 1
        return a

    # 3679. 使库存平衡的最少丢弃次数 (Minimum Discards to Balance Inventory)
    def minArrivalsToDiscard(self, arrivals: List[int], w: int, m: int) -> int:
        d = defaultdict(int)
        res = 0
        for i, v in enumerate(arrivals):
            if d[v] == m:
                res += 1
                arrivals[i] = 0
            else:
                d[v] += 1
            left = i + 1 - w
            if left >= 0:
                d[arrivals[left]] -= 1
        return res

    # 3680. 生成赛程 (Generate Schedule)
    def generateSchedule(self, n: int) -> List[List[int]]:
        if n < 5:
            return []

        ans = []

        # 处理 d=2,3,...,n-2
        for d in range(2, n - 1):
            for i in range(n):
                ans.append([i, (i + d) % n])

        # 交错排列 d=1 与 d=n-1（或者说 d=-1）
        for i in range(n):
            ans.append([i, (i + 1) % n])
            ans.append([(i - 1) % n, (i - 2) % n])

        return ans

    # 3683. 完成一个任务的最早时间 (Earliest Time to Finish One Task)
    def earliestTime(self, tasks: List[List[int]]) -> int:
        return min(sum(t) for t in tasks)

    # 3684. 至多 K 个不同元素的最大和 (Maximize Sum of At Most K Distinct Elements)
    def maxKDistinct(self, nums: List[int], k: int) -> List[int]:
        return sorted(set(nums), reverse=True)[:k]

    # 3686. 稳定子序列的数量 (Number of Stable Subsequences)
    def countStableSubsequences(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i == n:
                return 1
            res = dfs(i + 1, j, k)
            if nums[i] & 1 != k:
                res += dfs(i + 1, 1, nums[i] & 1)
            elif j < 2:
                res += dfs(i + 1, j + 1, nums[i] & 1)
            return res % MOD

        n = len(nums)
        MOD = 10**9 + 7
        # dfs(i, j, k)
        # 从i开始，之前已经选了连续j个奇数（k == 1)/ 偶数（k == 0）
        res = dfs(0, 0, 0)
        dfs.cache_clear()
        return (res - 1) % MOD

    # 3685. 含上限元素的子序列和 (Subsequence Sum After Capping)
    def subsequenceSumAfterCapping(self, nums: List[int], k: int) -> List[bool]:
        nums.sort()
        n = len(nums)
        res = [False] * n
        f = [False] * (k + 1)
        f[0] = True
        i = 0
        for x in range(1, n + 1):
            while i < n and nums[i] == x:
                for j in range(k, nums[i] - 1, -1):
                    f[j] = f[j] or f[j - nums[i]]
                i += 1
            for j in range(min(n - i, k // x) + 1):
                if f[k - j * x]:
                    res[x - 1] = True
                    break
        return res

    # 2197. 替换数组中的非互质数 (Replace Non-Coprime Numbers in Array)
    def replaceNonCoprimes(self, nums: List[int]) -> List[int]:
        res = []
        for x in nums:
            res.append(x)
            while len(res) > 1:
                a = res[-1]
                b = res[-2]
                g = gcd(a, b)
                if g == 1:
                    break
                res.pop()
                res[-1] = a * b // g
        return res

    # 2349. 设计数字容器系统 (Design a Number Container System)
    class NumberContainers:

        def __init__(self):
            self.index_number = defaultdict(int)
            # self.number_indexes = defaultdict(SortedSet) 也可以
            self.number_indexes = defaultdict(SortedList)

        def change(self, index: int, number: int) -> None:
            original = self.index_number[index]
            self.number_indexes[original].discard(index)
            self.index_number[index] = number
            self.number_indexes[number].add(index)

        def find(self, number: int) -> int:
            return self.number_indexes[number][0] if self.number_indexes[number] else -1

    # 3508. 设计路由器 (Implement Router)
    class Router:

        def __init__(self, memoryLimit: int):
            self.memoryLimit = memoryLimit
            self.dq = deque()  # (source, destination, timestamp)
            self.time_dic = defaultdict(list)  # destination -> timestamp list
            self.s = set()

        def addPacket(self, source: int, destination: int, timestamp: int) -> bool:
            pack = (source, destination, timestamp)
            if pack in self.s:
                return False
            if len(self.dq) == self.memoryLimit:
                old_pack = self.dq.popleft()
                self.s.remove(old_pack)
                a = self.time_dic[old_pack[1]]
                a.pop(0)
                if not a:
                    del self.time_dic[old_pack[1]]
            self.dq.append(pack)
            self.s.add(pack)
            self.time_dic[destination].append(timestamp)
            return True

        def forwardPacket(self) -> List[int]:
            if not self.dq:
                return []
            pack = self.dq.popleft()
            self.s.remove(pack)
            a = self.time_dic[pack[1]]
            a.pop(0)
            if not a:
                del self.time_dic[pack[1]]
            return [pack[0], pack[1], pack[2]]

        def getCount(self, destination: int, startTime: int, endTime: int) -> int:
            a = self.time_dic[destination]
            left = bisect.bisect_left(a, startTime)
            right = bisect.bisect_right(a, endTime)
            return right - left

    # 165. 比较版本号 (Compare Version Numbers)
    def compareVersion(self, version1: str, version2: str) -> int:
        a1 = version1.split(".")
        a2 = version2.split(".")
        n1, n2 = len(a1), len(a2)
        n = max(n1, n2)
        for i in range(n):
            x1 = int(a1[i]) if i < n1 else 0
            x2 = int(a2[i]) if i < n2 else 0
            if x1 < x2:
                return -1
            if x1 > x2:
                return 1
        return 0

    # 3688. 偶数的按位或运算 (Bitwise OR of Even Numbers in an Array)
    def evenNumberBitwiseORs(self, nums: List[int]) -> int:
        res = 0
        for x in nums:
            if x & 1 == 0:
                res |= x
        return res

    # 3689. 最大子数组总值 I (Maximum Total Subarray Value I)
    def maxTotalValue(self, nums: List[int], k: int) -> int:
        return (max(nums) - min(nums)) * k

    # 3690. 拆分合并数组 (Split and Merge Array Transformation)
    def minSplitMerge(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        vis = set()
        vis.add(tuple(nums1))
        q = deque()
        q.append(tuple(nums1))
        res = 0
        while q:
            sz = len(q)
            for _ in range(sz):
                a = q.popleft()
                if a == tuple(nums2):
                    return res
                a = list(a)
                for i in range(n):
                    for j in range(i + 1, n + 1):
                        sub = a[i:j]
                        b = a[:i] + a[j:]
                        for k in range(len(b) + 1):
                            c = b[:k] + sub + b[k:]
                            t = tuple(c)
                            if t not in vis:
                                vis.add(t)
                                q.append(tuple(c))
            res += 1

    # 166. 分数到小数 (Fraction to Recurring Decimal)
    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        sign = "-" if numerator * denominator < 0 else ""
        numerator = abs(numerator)
        denominator = abs(denominator)

        q, r = divmod(numerator, denominator)
        if r == 0:
            return sign + str(q)
        res = [sign + str(q) + "."]
        r_to_pos = {r: 1}
        while r:
            q, r = divmod(r * 10, denominator)
            res.append(str(q))
            if r in r_to_pos:
                res.insert(r_to_pos[r], "(")
                res.append(")")
                break
            r_to_pos[r] = len(res)
        return "".join(res)

    # 812. 最大三角形面积 (Largest Triangle Area)
    def largestTriangleArea(self, points: List[List[int]]) -> float:
        ans = 0
        for p1, p2, p3 in combinations(points, 3):
            x1, y1 = p2[0] - p1[0], p2[1] - p1[1]
            x2, y2 = p3[0] - p1[0], p3[1] - p1[1]
            ans = max(ans, abs(x1 * y2 - y1 * x2))  # 注意这里没有除以 2
        return ans / 2

    # 976. 三角形的最大周长 (Largest Perimeter Triangle)
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        for i in range(len(nums) - 1, 1, -1):
            if nums[i] < nums[i - 1] + nums[i - 2]:
                return nums[i] + nums[i - 1] + nums[i - 2]
        return 0

    # 3692. 众数频率字符 (Majority Frequency Characters)
    def majorityFrequencyGroup(self, s: str) -> str:
        cnts = [0] * 26
        a = ord("a")
        for c in map(ord, s):
            cnts[c - a] += 1
        mx = max(cnts)
        freq = [0] * (mx + 1)
        max_len = 0
        for i, v in enumerate(cnts):
            if v:
                freq[v] |= 1 << i
                max_len = max(max_len, freq[v].bit_count())
        for f in range(mx, 0, -1):
            if freq[f].bit_count() == max_len:
                res = []
                v = freq[f]
                while v:
                    lb = (v & -v).bit_length() - 1
                    res.append(chr(lb + a))
                    v &= v - 1
                return "".join(res)

    # 3693. 爬楼梯 II (Climbing Stairs II)
    def climbStairs(self, n: int, costs: List[int]) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == n - 1:
                return 0
            return min(
                dfs(j) + (j - i) ** 2 + costs[j] for j in range(i + 1, min(n, i + 4))
            )

        return dfs(-1)

    # 3694. 删除子字符串后不同的终点 (Distinct Points Reachable After Substring Removal)
    def distinctPoints(self, s: str, k: int) -> int:
        n = len(s)
        _set = set()
        r = 0
        c = 0
        for i, ch in enumerate(s):
            if ch == "U":
                r += 1
            elif ch == "D":
                r -= 1
            elif ch == "L":
                c -= 1
            else:
                c += 1
            if i >= k:
                if s[i - k] == "U":
                    r -= 1
                elif s[i - k] == "D":
                    r += 1
                elif s[i - k] == "L":
                    c += 1
                else:
                    c -= 1
            if i >= k - 1:
                _set.add((r, c))
        return len(_set)

    # 3695. 交换元素后的最大交替和 (Maximize Alternating Sum Using Swaps)
    def maxAlternatingSum(self, nums: List[int], swaps: List[List[int]]) -> int:
        class union:
            def __init__(self, n: int):
                self.parent = [i for i in range(n)]
                self.rank = [1] * n

            def get_root(self, x: int) -> int:
                if self.parent[x] == x:
                    return x
                self.parent[x] = self.get_root(self.parent[x])
                return self.parent[x]

            def is_connected(self, x: int, y: int) -> bool:
                return self.get_root(x) == self.get_root(y)

            def union(self, x: int, y: int) -> None:
                root_x = self.get_root(x)
                root_y = self.get_root(y)
                if root_x == root_y:
                    return
                if self.rank[root_x] < self.rank[root_y]:
                    self.parent[root_x] = root_y
                else:
                    self.parent[root_y] = root_x
                    if self.rank[root_x] == self.rank[root_y]:
                        self.rank[root_x] += 1

        n = len(nums)
        u = union(n)
        vis = [False] * n
        for x, y in swaps:
            u.union(x, y)
            vis[x] = True
            vis[y] = True
        res = 0
        g = defaultdict(list)
        for i, x in enumerate(nums):
            if not vis[i]:
                if i & 1:
                    res -= x
                else:
                    res += x
            else:
                root = u.get_root(i)
                g[root].append(i)
        for a in g.values():
            b = sorted([nums[i] for i in a])
            odd = sum(i & 1 for i in a)
            res += sum(b[odd:]) - sum(b[:odd])
        return res

    # 2221. 数组的三角和 (Find Triangular Sum of an Array)
    def triangularSum(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(1, n):
            for j in range(n - i):
                nums[j] = (nums[j] + nums[j + 1]) % 10
        return nums[0]

    # 3697. 计算十进制表示 (Compute Decimal Representation)
    def decimalRepresentation(self, n: int) -> List[int]:
        res = []
        p = 1
        while n:
            m = n % 10 * p
            if m:
                res.append(m)
            n //= 10
            p *= 10
        return res[::-1]

    # 3698. 分割数组得到最小绝对差 (Split Array With Minimum Difference)
    def splitArray(self, nums: List[int]) -> int:
        n = len(nums)
        pre = [-1] * n
        pre[0] = nums[0]
        for i in range(1, n):
            if nums[i] <= nums[i - 1]:
                break
            pre[i] = pre[i - 1] + nums[i]
        suf = 0
        res = inf
        for i in range(n - 1, 0, -1):
            if i < n - 1 and nums[i] <= nums[i + 1]:
                break
            suf += nums[i]
            if i - 1 >= 0 and pre[i - 1] != -1:
                res = min(res, abs(pre[i - 1] - suf))
        return -1 if res == inf else res

    # 3699. 锯齿形数组的总数 I (Number of ZigZag Arrays I)
    def zigZagArrays(self, n: int, l: int, r: int) -> int:
        MOD = 10**9 + 7
        k = r - l + 1
        f0 = [1] * k  # 后两个数递增
        f1 = [1] * k  # 后两个数递减
        for _ in range(n - 1):
            s0 = list(accumulate(f0, initial=0))
            s1 = list(accumulate(f1, initial=0))
            for j in range(k):
                f0[j] = s1[j] % MOD
                f1[j] = (s0[k] - s0[j + 1]) % MOD
        return (sum(f0) + sum(f1)) % MOD

    # 1518. 换水问题 (Water Bottles)
    def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
        res = numBottles
        while numBottles >= numExchange:
            res += numBottles // numExchange
            numBottles = sum(divmod(numBottles, numExchange))
        return res

    # 407. 接雨水 II (Trapping Rain Water II)
    def trapRainWater(self, heightMap: List[List[int]]) -> int:
        m, n = len(heightMap), len(heightMap[0])
        q = []
        for i in range(m):
            for j in range(n):
                if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                    q.append((heightMap[i][j], i, j))
                    heightMap[i][j] = -1
        heapq.heapify(q)
        res = 0
        while q:
            h, x, y = heapq.heappop(q)
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx <= m - 1 and 0 <= ny <= n - 1 and heightMap[nx][ny] >= 0:
                    res += max(0, h - heightMap[nx][ny])
                    heapq.heappush(q, (max(h, heightMap[nx][ny]), nx, ny))
                    heightMap[nx][ny] = -1
        return res

    # 11. 盛最多水的容器 (Container With Most Water)
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        res = 0
        while l < r:
            res = max(res, (r - l) * min(height[l], height[r]))
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return res

    # 778. 水位上升的泳池中游泳 (Swim in Rising Water)
    def swimInWater(self, grid: List[List[int]]) -> int:
        class union:
            def __init__(self, n: int):
                self.parent = [i for i in range(n)]
                self.rank = [1] * n

            def get_root(self, p: int) -> int:
                if self.parent[p] == p:
                    return p
                self.parent[p] = self.get_root(self.parent[p])
                return self.parent[p]

            def is_connected(self, p1: int, p2: int) -> bool:
                return self.get_root(p1) == self.get_root(p2)

            def union(self, p1: int, p2: int):
                r1 = self.get_root(p1)
                r2 = self.get_root(p2)
                if r1 == r2:
                    return
                if self.rank[r1] < self.rank[r2]:
                    self.parent[r1] = r2
                else:
                    self.parent[r2] = r1
                    if self.rank[r1] == self.rank[r2]:
                        self.rank[r1] += 1

        n = len(grid)
        d = defaultdict(tuple)
        for i in range(n):
            for j in range(n):
                d[grid[i][j]] = (i, j)
        u = union(n * n)
        for h in count(0):
            x, y = d[h]
            for nx, ny in (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1):
                if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] <= h:
                    u.union(x * n + y, nx * n + ny)
            if u.is_connected(0, n * n - 1):
                return h

    # 778. 水位上升的泳池中游泳 (Swim in Rising Water)
    def swimInWater(self, grid: List[List[int]]) -> int:
        n = len(grid)
        q = [(grid[0][0], 0, 0)]
        heapq.heapify(q)
        res = 0
        while q:
            h, x, y = heapq.heappop(q)
            res = max(res, h)
            if x == n - 1 and y == n - 1:
                return res
            grid[x][y] = inf
            for nx, ny in (x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1):
                if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] < inf:
                    heapq.heappush(q, (grid[nx][ny], nx, ny))

    # 3701. 计算交替和 (Compute Alternating Sum)
    def alternatingSum(self, nums: List[int]) -> int:
        return sum((-2 * (i & 1) + 1) * x for i, x in enumerate(nums))

    # 3702. 按位异或非零的最长子序列 (Longest Subsequence With Non-Zero Bitwise XOR)
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        xor = 0
        cnt_0 = 0
        for x in nums:
            xor ^= x
            cnt_0 += x == 0
        if xor:
            return n
        return 0 if cnt_0 == n else n - 1

    # 3703. 移除K-平衡子字符串 (Remove K-Balanced Substrings)
    def removeSubstring(self, s: str, k: int) -> str:
        st = []
        for x in s:
            if st and st[-1][0] == x:
                st[-1][1] += 1
            else:
                st.append([x, 1])
            if len(st) >= 2 and st[-1][0] == ")" and st[-1][1] == k and st[-2][1] >= k:
                st.pop()
                st[-1][1] -= k
                if st[-1][1] == 0:
                    st.pop()
        return "".join([x * c for x, c in st])

    # 3380. 用点构造面积最大的矩形 I (Maximum Area Rectangle With Point Constraints I)
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        res = -1
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                ax = min(points[i][0], points[j][0])
                ay = min(points[i][1], points[j][1])
                bx = max(points[i][0], points[j][0])
                by = max(points[i][1], points[j][1])
                if ax == bx or ay == by:
                    continue
                c = 0
                f = True
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if points[k][0] < ax or points[k][0] > bx:
                        continue
                    if points[k][1] < ay or points[k][1] > by:
                        continue
                    if points[k][0] == ax and points[k][1] == by:
                        c += 1
                        continue
                    if points[k][0] == bx and points[k][1] == ay:
                        c += 1
                        continue
                    f = False
                if c == 2 and f:
                    res = max(res, abs(ax - bx) * abs(ay - by))
        return res

    # 718. 最长重复子数组 (Maximum Length of Repeated Subarray)
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i < 0 or j < 0 or nums1[i] != nums2[j]:
                return 0
            return dfs(i - 1, j - 1) + 1

        res = 0
        n1 = len(nums1)
        n2 = len(nums2)
        for i in range(n1):
            for j in range(n2):
                res = max(res, dfs(i, j))
                if res == min(n1, n2):
                    return res
        dfs.cache_clear()
        return res

    # 20. 有效的括号 (Valid Parentheses)
    def isValid(self, s: str) -> bool:
        if len(s) & 1:
            return False
        d = defaultdict(str)
        d[")"] = "("
        d["}"] = "{"
        d["]"] = "["
        st = []
        for x in s:
            if x in d:
                if not st or d[x] != st[-1]:
                    return False
                st.pop()
            else:
                st.append(x)
        return not st

    # 21. 合并两个有序链表 (Merge Two Sorted Lists)
    def mergeTwoLists(
        self, list1: Optional[ListNode], list2: Optional[ListNode]
    ) -> Optional[ListNode]:
        dummy = p = ListNode(0)
        while list1 or list2:
            if list1 is None:
                p.next = list2
                break
            if list2 is None:
                p.next = list1
                break
            if list1.val < list2.val:
                p.next = list1
                list1 = list1.next
            else:
                p.next = list2
                list2 = list2.next
            p = p.next
        return dummy.next

    # 19. 删除链表的倒数第 N 个结点 (Remove Nth Node From End of List)
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        c = 0
        dummy = p = ListNode(0, head)
        while p.next:
            c += 1
            p = p.next
        pre_id = c - n
        pre = dummy
        cur = head
        while pre_id:
            pre_id -= 1
            pre = pre.next
            cur = cur.next
        pre.next = cur.next
        return dummy.next

    # 3310. 移除可疑的方法 (Remove Methods From Project)
    def remainingMethods(
        self, n: int, k: int, invocations: List[List[int]]
    ) -> List[int]:
        def dfs(x: int):
            if x in c:
                return
            c.add(x)
            for y in g[x]:
                dfs(y)

        g = [[] for _ in range(n)]
        for u, v in invocations:
            g[u].append(v)
        # 被污染的方法
        c = set()
        dfs(k)
        for u, v in invocations:
            if u not in c and v in c:
                return list(range(n))
        return list(set(range(n)) - c)

    # 2273. 移除字母异位词后的结果数组 (Find Resultant Array After Removing Anagrams)
    def removeAnagrams(self, words: List[str]) -> List[str]:
        st = []
        for w in words:
            if not st or sorted(st[-1]) != sorted(w):
                st.append(w)
        return st

    # 3707. 相等子字符串分数 (Equal Score Substrings)
    def scoreBalance(self, s: str) -> bool:
        _sum = sum(ord(x) - ord("a") + 1 for x in s)
        _pre = 0
        for x in s:
            _pre += ord(x) - ord("a") + 1
            if _pre * 2 == _sum:
                return True
            if _pre * 2 > _sum:
                return False
        return False

    # 3708. 最长斐波那契子数组 (Longest Fibonacci Subarray)
    def longestSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        i = 1
        res = 0
        while i < n:
            pre = nums[i - 1]
            cur = nums[i]
            j = i + 1
            while j < n and pre + cur == nums[j]:
                pre = cur
                cur = nums[j]
                j += 1
            res = max(res, j - i + 1)
            i = j
        return res

    # 3709. 设计考试分数记录器 (Design Exam Scores Tracker)
    class ExamTracker:

        def __init__(self):
            self.pre = [(0, 0)]

        def record(self, time: int, score: int) -> None:
            self.pre.append([time, self.pre[-1][1] + score])

        def totalScore(self, startTime: int, endTime: int) -> int:
            end = self.bisect_right(endTime + 1)
            start = self.bisect_right(startTime)
            return end - start

        def bisect_right(self, t: int) -> int:
            left = 0
            right = len(self.pre) - 1
            while left <= right:
                mid = left + ((right - left) >> 1)
                if self.pre[mid][0] < t:
                    left = mid + 1
                else:
                    right = mid - 1
            return self.pre[left - 1][1]

    # 3712. 出现次数能被 K 整除的元素总和 (Sum of Elements With Frequency Divisible by K)
    def sumDivisibleByK(self, nums: List[int], m: int) -> int:
        c = Counter(nums)
        return sum(k * v for k, v in c.items() if v % m == 0)

    # 3713. 最长的平衡子串 I (Longest Balanced Substring I)
    def longestBalanced(self, s: str) -> int:
        n = len(s)
        res = 0
        for i in range(n):
            d = defaultdict(int)
            k = 0
            for j in range(i, n):
                d[s[j]] += 1
                if d[s[j]] == 1:
                    k += 1
                if (j - i + 1) % k == 0 and all(
                    x == (j - i + 1) // k for x in d.values()
                ):
                    res = max(res, j - i + 1)
        return res

    # 3715. 完全平方数的祖先个数总和 (Sum of Perfect Square Ancestors)
    def sumOfAncestors(self, n: int, edges: List[List[int]], nums: List[int]) -> int:
        def dfs(x: int, fa: int) -> int:
            v = nums[x]
            res = d[v]
            d[v] += 1
            for y in g[x]:
                if y != fa:
                    res += dfs(y, x)
            d[v] -= 1
            return res

        # 计算x除去所有完全平方的因子的值
        @cache
        def core(x: int) -> int:
            origin = x
            for p in range(2, isqrt(x) + 1):
                c = 0
                while x % p == 0:
                    c ^= 1
                    if c == 0:
                        origin //= p * p
                    x //= p
            return origin

        for i, x in enumerate(nums):
            nums[i] = core(x)
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        d = defaultdict(int)
        return dfs(0, -1)

    # 3710. 最大划分因子 (Maximum Partition Factor)
    def maxPartitionFactor(self, points: List[List[int]]) -> int:
        def check(low: int) -> bool:
            def dfs(x: int, c: int) -> bool:
                color[x] = c
                for y in range(n):
                    if x == y:
                        continue
                    if (
                        abs(points[x][0] - points[y][0])
                        + abs(points[x][1] - points[y][1])
                        >= low
                    ):
                        continue
                    if color[y] == c:
                        return False
                    if color[y] == 0 and not dfs(y, -c):
                        return False
                return True

            color = [0] * len(points)
            for i, c in enumerate(color):
                if c == 0 and not dfs(i, 1):
                    return False
            return True

        n = len(points)
        if n == 2:
            return 0
        mx = 0
        for (x1, y1), (x2, y2) in combinations(points, 2):
            mx = max(mx, abs(x1 - x2) + abs(y1 - y2))
        left = 0
        right = mx
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                left = mid + 1
            else:
                right = mid - 1
        return left - 1

    # 2111. 使数组 K 递增的最少操作次数 (Minimum Operations to Make the Array K-Increasing)
    def kIncreasing(self, arr: List[int], k: int) -> int:
        n = len(arr)
        res = 0
        for i in range(k):
            a = [arr[j] for j in range(i, n, k)]
            g = []
            for x in a:
                j = bisect.bisect_left(g, x + 1)
                if j == len(g):
                    g.append(x)
                else:
                    g[j] = x
            res += len(a) - len(g)
        return res

    # 2598. 执行操作后的最大 MEX (Smallest Missing Non-negative Integer After Operations)
    def findSmallestInteger(self, nums: List[int], value: int) -> int:
        cnts = [0] * value
        for x in nums:
            cnts[x % value] += 1
        res = 0
        while cnts[res % value]:
            cnts[res % value] -= 1
            res += 1
        return res

    # 2011. 执行操作后的变量值 (Final Value of Variable After Performing Operations)
    def finalValueAfterOperations(self, operations: List[str]) -> int:
        return sum(1 if x[1] == "+" else -1 for x in operations)

    # 1625. 执行操作后字典序最小的字符串 (Lexicographically Smallest String After Applying Operations)
    def findLexSmallestString(self, s: str, a: int, b: int) -> str:
        n = len(s)
        vis = set()
        q = deque()
        q.append(s)
        res = ""
        while q:
            sz = len(q)
            for _ in range(sz):
                cur = q.popleft()
                if not res or cur < res:
                    res = cur
                arr = [x for x in cur]
                # 操作1
                for i in range(1, n, 2):
                    arr[i] = str((int(arr[i]) + a) % 10)
                n_cur = "".join(arr)
                if n_cur not in vis:
                    vis.add(n_cur)
                    q.append(n_cur)
                # 操作2
                n_cur2 = cur[-b:] + cur[:-b]
                if n_cur2 not in vis:
                    vis.add(n_cur2)
                    q.append(n_cur2)
        return res

    # 3718. 缺失的最小倍数 (Smallest Missing Multiple of K)
    def missingMultiple(self, nums: List[int], k: int) -> int:
        s = set(nums)
        p = 1
        while p * k in s:
            p += 1
        return p * k

    # 3719. 最长平衡子数组 I (Longest Balanced Subarray I)
    def longestBalanced(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        for i in range(n):
            s = [set() for _ in range(2)]
            for j in range(i, n):
                s[nums[j] & 1].add(nums[j])
                if len(s[0]) == len(s[1]):
                    res = max(res, j - i + 1)
        return res

    # 3720. 大于目标字符串的最小字典序排列 (Lexicographically Smallest Permutation Greater Than Target)
    def lexGreaterPermutation(self, s: str, target: str) -> str:
        def check(cnts_s: List[int], start: int) -> bool:
            cur = []
            for i in range(25, -1, -1):
                cur.extend(chr(i + a) * cnts_s[i])
            t = target[start:]
            return "".join(cur) > t

        a = ord("a")
        cnts_s = [0] * 26
        for c in s:
            cnts_s[ord(c) - a] += 1
        res = []
        for i, c in enumerate(target):
            idx = ord(c) - a
            cnts_s[idx] -= 1
            if cnts_s[idx] >= 0 and check(cnts_s, i + 1):
                res.append(c)
                continue
            cnts_s[idx] += 1
            for j in range(idx + 1, 26):
                if cnts_s[j]:
                    cnts_s[j] -= 1
                    res.append(chr(j + a))
                    for k in range(26):
                        res.append(chr(k + a) * cnts_s[k])
                    return "".join(res)
            return ""
        return "".join(res)

    # 3346. 执行操作后元素的最高频率 I (Maximum Frequency of an Element After Performing Operations I)
    # 3347. 执行操作后元素的最高频率 II (Maximum Frequency of an Element After Performing Operations II)
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        cnt = defaultdict(int)
        diff = defaultdict(int)
        for x in nums:
            cnt[x] += 1
            diff[x]
            diff[x - k] += 1
            diff[x + k + 1] -= 1
        res = 0
        sum_d = 0
        for x, d in sorted(diff.items()):
            sum_d += d
            res = max(res, min(sum_d, cnt[x] + numOperations))
        return res

    # 354. 俄罗斯套娃信封问题 (Russian Doll Envelopes)
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        a = []
        for _, y in envelopes:
            j = bisect.bisect_left(a, y)
            if j == len(a):
                a.append(y)
            else:
                a[j] = y
        return len(a)

    # 1716. 计算力扣银行的钱 (Calculate Money in Leetcode Bank)
    def totalMoney(self, n: int) -> int:
        w, d = divmod(n, 7)
        # w个完整周
        a1 = 28 * w + 7 * (w - 1) * w // 2
        # d剩下的天数
        a2 = d * (d + 1) // 2 + d * w
        return a1 + a2

    # 3578. 统计极差最大为 K 的分割方式数 (Count Partitions With Max-Min Difference at Most K)
    def countPartitions(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        min_q = deque()
        max_q = deque()
        f = [0] * (n + 1)
        f[0] = 1
        sum_f = 0
        left = 0
        for i, x in enumerate(nums):
            sum_f += f[i]
            # 维护最小值单调队列 队列中的索引对应的值单调递增
            while min_q and nums[min_q[-1]] >= x:
                min_q.pop()
            min_q.append(i)
            # 维护最大值单调队列 队列中的索引对应的值单调递减
            while max_q and nums[max_q[-1]] <= x:
                max_q.pop()
            max_q.append(i)
            # 调整left指针 直到区间[left, i]满足条件
            while nums[max_q[0]] - nums[min_q[0]] > k:
                sum_f -= f[left]
                if min_q[0] == left:
                    min_q.popleft()
                if max_q[0] == left:
                    max_q.popleft()
                left += 1
            f[i + 1] = sum_f % MOD
        return f[n]

    # 2043. 简易银行系统 (Simple Bank System)
    class Bank:

        def __init__(self, balance: List[int]):
            self.balance = balance
            self.n = len(balance)

        def transfer(self, account1: int, account2: int, money: int) -> bool:
            return (
                self.accountIsLegal(account1)
                and self.accountIsLegal(account2)
                and self.withdraw(account1, money)
                and self.deposit(account2, money)
            )

        def deposit(self, account: int, money: int) -> bool:
            if not self.accountIsLegal(account):
                return False
            self.balance[account - 1] += money
            return True

        def withdraw(self, account: int, money: int) -> bool:
            if not self.accountIsLegal(account):
                return False
            if self.balance[account - 1] < money:
                return False
            self.balance[account - 1] -= money
            return True

        def accountIsLegal(self, account: int) -> bool:
            return 1 <= account <= self.n

    # 3722. 反转后字典序最小的字符串 (Lexicographically Smallest String After Reverse)
    def lexSmallest(self, s: str) -> str:
        n = len(s)
        res = s
        for i in range(n):
            res = min(res, s[: i + 1][::-1] + s[i + 1 :], s[:i] + s[i:][::-1])
        return res

    # 3723. 数位平方和的最大值 (Maximize Sum of Squares of Digits)
    def maxSumOfSquares(self, num: int, sum: int) -> str:
        if num * 9 < sum:
            return ""
        res = ["0"] * num
        for i in range(num):
            d = min(9, sum)
            res[i] = str(d)
            sum -= d
            if sum == 0:
                break
        return "".join(res)

    # 3724. 转换数组的最少操作次数 (Minimum Operations to Transform Array)
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        @cache
        def dfs(i: int, added: bool) -> int:
            if i < 0:
                return 0 if added else inf
            # 不追加
            res = dfs(i - 1, added) + abs(nums1[i] - nums2[i])
            # 追加
            if not added:
                d = max(nums1[i], nums2[i], nums2[-1]) - min(
                    nums1[i], nums2[i], nums2[-1]
                )
                res = min(res, dfs(i - 1, True) + d + 1)
            return res

        return dfs(len(nums1) - 1, False)

    # 3725. 统计每一行选择互质整数的方案数 (Count Ways to Choose Coprime Integers from Rows)
    def countCoprime(self, mat: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i < 0:
                return int(j == 1)
            return sum(dfs(i - 1, gcd(j, x)) for x in mat[i]) % MOD

        MOD = 10**9 + 7
        return dfs(len(mat) - 1, 0)

    # 2125. 银行中的激光束数量 (Number of Laser Beams in a Bank)
    def numberOfBeams(self, bank: List[str]) -> int:
        pre = 0
        res = 0
        for b in bank:
            c = b.count("1")
            if c:
                res += pre * c
                pre = c
        return res

    # 3726. 移除十进制表示中的所有零 (Remove Zeros in Decimal Representation)
    def removeZeros(self, n: int) -> int:
        res = 0
        p = 1
        while n:
            n, m = divmod(n, 10)
            if m:
                res += m * p
                p *= 10
        return res

    # 3727. 最大交替平方和 (Maximum Alternating Sum of Squares)
    def maxAlternatingSum(self, nums: List[int]) -> int:
        nums.sort(key=lambda k: abs(k))
        return sum(x**2 * (-1 if i < len(nums) // 2 else 1) for i, x in enumerate(nums))

    # 3728. 边界与内部和相等的稳定子数组 (Stable Subarrays With Equal Boundary and Interior Sum)
    def countStableSubarrays(self, capacity: List[int]) -> int:
        res = 0
        d = defaultdict(int)
        s = capacity[0]
        for last, x in pairwise(capacity):
            res += d[(x, s)]
            d[(last, last + s)] += 1
            s += x
        return res

    # 1526. 形成目标数组的子数组最少增加次数 (Minimum Number of Increments on Subarrays to Form a Target Array)
    def minNumberOperations(self, target: List[int]) -> int:
        return target[0] + sum(max(0, y - x) for x, y in pairwise(target))

    # 2257. 统计网格图中没有被保卫的格子数 (Count Unguarded Cells in the Grid)
    def countUnguarded(
        self, m: int, n: int, guards: List[List[int]], walls: List[List[int]]
    ) -> int:
        g = [[0] * n for _ in range(m)]
        for x, y in guards:
            g[x][y] = 1
        for x, y in walls:
            g[x][y] = 2
        d = (0, 1), (0, -1), (1, 0), (-1, 0)
        for gx, gy in guards:
            for dx, dy in d:
                nx, ny = gx + dx, gy + dy
                while (
                    m > nx >= 0 and n > ny >= 0 and (g[nx][ny] == 0 or g[nx][ny] == 3)
                ):
                    g[nx][ny] = 3
                    nx += dx
                    ny += dy
        res = 0
        for i in range(m):
            for j in range(n):
                if g[i][j] == 0:
                    res += 1
        return res

    # 1578. 使绳子变成彩色的最短时间 (Minimum Time to Make Rope Colorful)
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i < 0:
                return 0
            res = dfs(i - 1, j) + neededTime[i]
            c = ord(colors[i]) - ord("a")
            if c != j:
                res = min(res, dfs(i - 1, c))
            return res

        n = len(colors)
        res = dfs(n - 1, 26)
        dfs.cache_clear()
        return res

    # 1578. 使绳子变成彩色的最短时间 (Minimum Time to Make Rope Colorful)
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        n = len(colors)
        i = 0
        res = 0
        while i < n:
            j = i
            s = 0
            mx = 0
            while j < n and colors[i] == colors[j]:
                mx = max(mx, neededTime[j])
                s += neededTime[j]
                j += 1
            res += s - mx
            i = j
        return res

    # 3731. 找出缺失的元素 (Find Missing Elements)
    def findMissingElements(self, nums: List[int]) -> List[int]:
        _max = max(nums)
        _min = min(nums)
        s = set(nums)
        return [x for x in range(_min + 1, _max) if x not in s]

    # 3732. 一次替换后的三元素最大乘积 (Maximum Product of Three Elements After One Replacement)
    def maxProduct(self, nums: List[int]) -> int:
        nums.sort()
        return 10**5 * max(
            abs(nums[0] * nums[1]),
            abs(nums[-1] * nums[-2]),
            abs(nums[0] * nums[-1]),
        )

    # 3733. 完成所有送货任务的最少时间 (Minimum Time to Complete All Deliveries)
    def minimumTime(self, d: List[int], r: List[int]) -> int:
        def check(t: int) -> bool:
            # 第一架无人机不可用小时数
            x1 = t // r[0]
            # 第二架无人机不可用小时数
            x2 = t // r[1]
            # 两架无人机都可用的小时数
            b = t - (x1 + x2 - t // lcm(r[0], r[1]))
            # (y1 - b) 只有第一架无人机可用的小时数，d[0] - (y1 - b) 使用「只有第一架无人机可用的小时数」后，还需要送多少小时货，这些货只能使用两架无人机都能送货的小时数来消耗
            # r1 = max(0, d[0] - (y1 - b))
            r1 = max(0, d[0] - (x2 - t // lcm(r[0], r[1])))
            # (y2 - b) 只有第二架无人机可用的小时数，d[1] - (y2 - b) 使用「只有第二架无人机可用的小时数」后，还需要送多少小时货，这些货只能使用两架无人机都能送货的小时数来消耗
            # r2 = max(0, d[1] - (y2 - b))
            r2 = max(0, d[1] - (x1 - t // lcm(r[0], r[1])))
            return r1 + r2 <= b

        left = d[0] + d[1]
        right = 10**10
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return right + 1

    # 3318. 计算子数组的 x-sum I (Find X-Sum of All K-Long Subarrays I) --暴力
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        res = [0] * (n - k + 1)
        d = defaultdict(int)
        for i in range(n):
            d[nums[i]] += 1
            if i >= k:
                d[nums[i - k]] -= 1
            if i >= k - 1:
                a = [[val, key] for key, val in d.items()]
                a.sort(key=lambda x: (-x[0], -x[1]))
                res[i - k + 1] = sum(x * y for x, y in a[:x])
        return res

    # 3734. 大于目标字符串的最小字典序回文排列 (Lexicographically Smallest Palindromic Permutation Greater Than Target)
    def lexPalindromicPermutation(self, s: str, target: str) -> str:
        def check(cnt: List[int], i: int, p: int) -> bool:
            cnt[p] -= 1
            res[i] = res[n - i - 1] = chr(p + ord("a"))
            i += 1
            for j in range(25, -1, -1):
                while cnt[j]:
                    res[i] = res[n - i - 1] = chr(j + ord("a"))
                    cnt[j] -= 1
                    i += 1
            return "".join(res) > target

        n = len(s)
        if n == 1:
            if s <= target:
                return ""
            return s
        cnt = [0] * 26
        for x in s:
            cnt[ord(x) - ord("a")] += 1
        odd_cnt = 0
        ch = ""
        for i, c in enumerate(cnt):
            if c & 1:
                ch = chr(i + ord("a"))
                odd_cnt += 1
                # s无法构成回文
                if odd_cnt > 1:
                    return ""
        s_max = ""
        for i in range(25, -1, -1):
            s_max += chr(i + ord("a")) * (cnt[i] // 2)
        s_max = s_max + ch + "".join(reversed(s_max))
        # 最大的回文 <= target
        if s_max <= target:
            return ""
        res = [""] * n
        if n % 2:
            res[n // 2] = ch
            cnt[ord(ch) - ord("a")] -= 1
        for i in range(26):
            cnt[i] //= 2
        for i, t in enumerate(target[: n // 2]):
            p = ord(t) - ord("a")
            if cnt[p] == 0 or not check(cnt.copy(), i, p):
                for j in range(p + 1, 26):
                    if cnt[j]:
                        res[i] = res[n - i - 1] = chr(j + ord("a"))
                        i += 1
                        cnt[j] -= 1
                        break
                for j in range(26):
                    while cnt[j]:
                        res[i] = res[n - i - 1] = chr(j + ord("a"))
                        i += 1
                        cnt[j] -= 1
                return "".join(res)
            cnt[p] -= 1
            res[i] = res[n - i - 1] = t
        return "".join(res)

    # 2528. 最大化城市的最小电量 (Maximize the Minimum Powered City)
    def maxPower(self, stations: List[int], r: int, k: int) -> int:
        def check(t: int, diff: List[int], k) -> bool:
            d = 0
            for i in range(n):
                d += diff[i]
                need = max(0, t - d)
                if k - need < 0:
                    return False
                k -= need
                d += need
                diff[min(n, i + r + r + 1)] -= need
            return True

        n = len(stations)
        diff = [0] * (n + 1)
        for i, x in enumerate(stations):
            diff[max(0, i - r)] += x
            diff[min(i + r + 1, n)] -= x
        d = 0
        mn = inf
        for i in range(n):
            d += diff[i]
            mn = min(mn, d)
        left = mn
        right = mn + k
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid, diff.copy(), k):
                left = mid + 1
            else:
                right = mid - 1
        return left - 1

    # 2169. 得到 0 的操作数 (Count Operations to Obtain Zero)
    def countOperations(self, x: int, y: int) -> int:
        res = 0
        while y:
            res += x // y
            x, y = y, x % y
        return res

    # 3736. 最小操作次数使数组元素相等 III (Minimum Moves to Equal Array Elements III)
    def minMoves(self, nums: List[int]) -> int:
        return max(nums) * len(nums) - sum(nums)

    # 3740. 三个相等元素之间的最小距离 I (Minimum Distance Between Three Equal Elements I)
    # 3741. 三个相等元素之间的最小距离 II (Minimum Distance Between Three Equal Elements II)
    def minimumDistance(self, nums: List[int]) -> int:
        d = defaultdict(list)
        res = inf
        for i, x in enumerate(nums):
            d[x].append(i)
            if len(d[x]) >= 3:
                res = min(res, d[x][-1] - d[x][-3])
        return -1 if res == inf else res * 2

    # 3742. 网格中得分最大的路径 (Maximum Path Score in a Grid)
    def maxPathScore(self, grid: List[List[int]], k: int) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i < 0 or j < 0:
                return -inf
            k -= grid[i][j] > 0
            if k < 0:
                return -inf
            if i == 0 and j == 0:
                return grid[i][j]
            return max(dfs(i - 1, j, k), dfs(i, j - 1, k)) + grid[i][j]

        m, n = len(grid), len(grid[0])
        k = min(k, m + n - 2)
        res = dfs(m - 1, n - 1, k)
        dfs.cache_clear()
        return -1 if res < 0 else res

    # 3737. 统计主要元素子数组数目 I (Count Subarrays With Majority Element I)
    # 3739. 统计主要元素子数组数目 II (Count Subarrays With Majority Element II)
    def countMajoritySubarrays(self, nums: List[int], target: int) -> int:
        sl = SortedList([0])
        ans = s = 0
        for x in nums:
            s += 1 if x == target else -1
            ans += sl.bisect_left(s)
            sl.add(s)
        return ans

    # 2654. 使数组所有元素变成 1 的最少操作次数 (Minimum Number of Operations to Make All Array Elements Equal to 1)
    def minOperations(self, nums: List[int]) -> int:
        cnt1 = 0
        n = len(nums)
        g = 0
        for x in nums:
            g = gcd(g, x)
            if x == 1:
                cnt1 += 1
        if g != 1:
            return -1
        if cnt1:
            return n - cnt1
        _min = n
        for i in range(n):
            g = 0
            for j in range(i, n):
                g = gcd(g, nums[j])
                if g == 1:
                    _min = min(_min, j - i + 1)
                    break
        return _min - 1 + n - 1

    # 3738. 替换至多一个元素后最长非递减子数组 (Longest Non-Decreasing Subarray After Replacing at Most One Element)
    def longestSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 1
        res = 2
        suf = [0] * n
        suf[-1] = 1
        for i in range(n - 2, 0, -1):
            if nums[i] <= nums[i + 1]:
                suf[i] = suf[i + 1] + 1
                res = max(res, suf[i] + 1)
            else:
                suf[i] = 1
        pre = 1
        for i in range(1, n - 1):
            if nums[i - 1] <= nums[i + 1]:
                res = max(res, pre + 1 + suf[i + 1])
            if nums[i - 1] <= nums[i]:
                pre += 1
                res = max(res, pre + 1)
            else:
                pre = 1
        return res

    # 2536. 子矩阵元素加 1 (Increment Submatrices by One)
    def rangeAddQueries(self, n: int, queries: List[List[int]]) -> List[List[int]]:
        diff = [[0] * (n + 2) for _ in range(n + 2)]
        for r1, c1, r2, c2 in queries:
            diff[r1 + 1][c1 + 1] += 1
            diff[r1 + 1][c2 + 2] -= 1
            diff[r2 + 2][c1 + 1] -= 1
            diff[r2 + 2][c2 + 2] += 1
        res = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                diff[i + 1][j + 1] += diff[i][j + 1] + diff[i + 1][j] - diff[i][j]
                res[i][j] = diff[i + 1][j + 1]
        return res

    # 1513. 仅含 1 的子串数 (Number of Substrings With Only 1s)
    def numSub(self, s: str) -> int:
        MOD = 10**9 + 7
        res = 0
        cnt1 = 0
        for x in s:
            if int(x):
                cnt1 += 1
                res += cnt1
            else:
                cnt1 = 0
        return res % MOD

    # 3745. 三元素表达式的最大值 (Maximize Expression of Three Elements)
    def maximizeExpressionOfThree(self, nums: List[int]) -> int:
        mx1, mx2, min1 = -inf, -inf, inf
        for x in nums:
            if x >= mx1:
                mx2 = mx1
                mx1 = x
            elif x > mx2:
                mx2 = x
            if x < min1:
                min1 = x
        return mx1 + mx2 - min1

    # 3746. 等量移除后的字符串最小长度 (Minimum String Length After Balanced Removals)
    def minLengthAfterRemovals(self, s: str) -> int:
        return abs(s.count("a") - s.count("b"))

    # 3747. 统计移除零后不同整数的数目 (Count Distinct Integers After Removing Zeros)
    def countDistinct(self, n: int) -> int:
        @cache
        def dfs(i: int, j: bool, is_limit: bool, is_num: bool) -> int:
            if i == l:
                return j and is_num
            res = 0
            if not is_num:
                res += dfs(i + 1, j, False, False)
            up = int(s[i]) if is_limit else 9
            for d in range(0 if is_num else 1, up + 1):
                res += dfs(i + 1, j or d == 0, is_limit and d == up, True)
            return res

        s = str(n)
        l = len(s)
        return n - dfs(0, False, True, False)

    # 868. 二进制间距 (Binary Gap)
    def binaryGap(self, n: int) -> int:
        last, res = inf, 0
        while n:
            lb = (n & -n).bit_length() - 1
            res = max(res, lb - last)
            last = lb
            n &= n - 1
        return res

    # 476. 数字的补数 (Number Complement)
    # 1009. 十进制整数的反码 (Complement of Base 10 Integer)
    def bitwiseComplement(self, n: int) -> int:
        u = (1 << len(bin(n)[2:])) - 1
        return n ^ u

    # 1342. 将数字变成 0 的操作次数 (Number of Steps to Reduce a Number to Zero)
    def numberOfSteps(self, num: int) -> int:
        return max(0, num.bit_count() + num.bit_length() - 1)

    # 461. 汉明距离 (Hamming Distance)
    # 2220. 转换数字的最少位翻转次数 (Minimum Bit Flips to Convert Number)
    def minBitFlips(self, start: int, goal: int) -> int:
        return (start ^ goal).bit_count()

    # 1356. 根据数字二进制下 1 的数目排序 (Sort Integers by The Number of 1 Bits)
    def sortByBits(self, arr: List[int]) -> List[int]:
        return sorted(arr, key=lambda o: (o.bit_count(), o))

    # 1437. 是否所有 1 都至少相隔 k 个元素 (Check If All 1's Are at Least Length K Places Away)
    def kLengthApart(self, nums: List[int], k: int) -> bool:
        pre = -inf
        for i, x in enumerate(nums):
            if x:
                if i - pre - 1 < k:
                    return False
                pre = i
        return True

    # 3748. 统计稳定子数组的数目 (Count Stable Subarrays)
    def countStableSubarrays(
        self, nums: List[int], queries: List[List[int]]
    ) -> List[int]:
        left = []
        s = [0]
        res = []
        n = len(nums)
        start = 0
        for i, x in enumerate(nums):
            if i == n - 1 or x > nums[i + 1]:
                left.append(start)
                m = i - start + 1
                s.append(s[-1] + m * (m + 1) // 2)
                start = i + 1
        for l, r in queries:
            i = bisect.bisect_right(left, l)
            j = bisect.bisect_right(left, r) - 1
            if i > j:
                m = r - l + 1
                res.append(m * (m + 1) // 2)
                continue
            m1 = left[i] - l
            m2 = r - left[j] + 1
            res.append(m1 * (m1 + 1) // 2 + s[j] - s[i] + m2 * (m2 + 1) // 2)
        return res

    # 717. 1 比特与 2 比特字符 (1-bit and 2-bit Characters) --正序
    def isOneBitCharacter(self, bits: List[int]) -> bool:
        i = 0
        n = len(bits)
        while i < n - 1:
            i += bits[i] + 1
        return i == n - 1

    # 717. 1 比特与 2 比特字符 (1-bit and 2-bit Characters) --倒序
    def isOneBitCharacter(self, bits: List[int]) -> bool:
        n = len(bits)
        i = n - 2
        cnt1 = 0
        while i >= 0 and bits[i]:
            i -= 1
            cnt1 += 1
        return cnt1 & 1 == 0

    # 693. 交替位二进制数 (Binary Number with Alternating Bits)
    def hasAlternatingBits(self, n: int) -> bool:
        xor = (n ^ (n >> 1)) + 1
        return xor.bit_count() == 1

    # 693. 交替位二进制数 (Binary Number with Alternating Bits)
    def hasAlternatingBits(self, n: int) -> bool:
        return n ^ (n >> 1) == (1 << n.bit_length()) - 1

    # 2657. 找到两个数组的前缀公共数组 (Find the Prefix Common Array of Two Arrays)
    def findThePrefixCommonArray(self, A: List[int], B: List[int]) -> List[int]:
        res = []
        or_a = 0
        or_b = 0
        for a, b in zip(A, B):
            or_a |= 1 << a
            or_b |= 1 << b
            res.append((or_a & or_b).bit_count())
        return res

    # 面试题 05.01. 插入 (Insert Into Bits LCCI)
    def insertBits(self, N: int, M: int, i: int, j: int) -> int:
        w = ((1 << (j - i + 1)) - 1) << i
        u = (1 << N.bit_length()) - 1
        N &= u ^ w
        return N | (M << i)

    # 2154. 将找到的值乘以 2 (Keep Multiplying Found Values by Two)
    def findFinalValue(self, nums: List[int], original: int) -> int:
        mask = 0
        for x in nums:
            d, m = divmod(x, original)
            if m == 0 and d & (d - 1) == 0:
                mask |= d
        mask = ~mask
        return original * (mask & -mask)

    # 3273. 对 Bob 造成的最少伤害 (Minimum Amount of Damage Dealt to Bob)
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        s = sum(damage)
        a = [[(h + power - 1) // power, d] for h, d in zip(health, damage)]
        a.sort(key=lambda p: p[0] / p[1])
        res = s = 0
        for k, d in a:
            s += k
            res += s * d
        return res

    # 757. 设置交集大小至少为2 (Set Intersection Size At Least Two)
    def intersectionSizeTwo(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: (x[1], -x[0]))
        s = e = -1
        ans = 0
        for a, b in intervals:
            if a <= s:
                continue
            if a > e:
                ans += 2
                s, e = b - 1, b
            else:
                ans += 1
                s, e = e, b
        return ans

    # 1486. 数组异或操作 (XOR Operation in an Array)
    def xorOperation(self, n: int, start: int) -> int:
        res = 0
        for i in range(n):
            res ^= start + i * 2
        return res

    # 1720. 解码异或后的数组 (Decode XORed Array)
    def decode(self, encoded: List[int], first: int) -> List[int]:
        res = [first]
        for x in encoded:
            res.append(x ^ res[-1])
        return res

    # 3334. 数组的最大因子得分 (Find the Maximum Factor Score of Array)
    def maxScore(self, nums: List[int]) -> int:
        n = len(nums)
        pre_l = [1] * (n + 1)
        pre_g = [0] * (n + 1)
        for i in range(1, n + 1):
            pre_l[i] = lcm(pre_l[i - 1], nums[i - 1])
            pre_g[i] = gcd(pre_g[i - 1], nums[i - 1])
        res = pre_l[-1] * pre_g[-1]
        suf_l = 1
        suf_g = 0
        for i in range(n - 1, -1, -1):
            res = max(res, lcm(pre_l[i], suf_l) * gcd(pre_g[i], suf_g))
            suf_l = lcm(suf_l, nums[i])
            suf_g = gcd(suf_g, nums[i])
        return res

    # 3433. 统计用户被提及情况 (Count Mentions Per User)
    def countMentions(self, n: int, events: List[List[str]]) -> List[int]:
        res = [0] * n
        all_cnt = 0
        nxt_offline = [-inf] * n
        events.sort(key=lambda x: (int(x[1]), x[2]))
        for e, t, ids in events:
            if e == "MESSAGE":
                if ids == "ALL":
                    all_cnt += 1
                # 只提及上线用户
                elif ids == "HERE":
                    for i in range(n):
                        if nxt_offline[i] <= int(t) < nxt_offline[i] + 60:
                            continue
                        res[i] += 1
                else:
                    for id in ids.split(" "):
                        res[int(id[2:])] += 1
            # OFFLINE
            else:
                nxt_offline[int(ids)] = int(t)
        for i in range(n):
            res[i] += all_cnt
        return res

    # 3387. 两天自由外汇交易后的最大货币数 (Maximize Amount After Two Days of Conversions)
    def maxAmount(
        self,
        initialCurrency: str,
        pairs1: List[List[str]],
        rates1: List[float],
        pairs2: List[List[str]],
        rates2: List[float],
    ) -> float:
        def trans(
            pairs: List[List[str]], rates: List[float]
        ) -> defaultdict[str, float]:
            def dfs(u: str, val: float) -> None:
                dist[u] = val
                for v, r in g[u]:
                    if v not in dist:
                        dfs(v, val * r)

            g = defaultdict(list)
            for (a, b), r in zip(pairs, rates):
                g[a].append((b, r))
                g[b].append((a, 1.0 / r))
            dist = defaultdict(float)
            dfs(initialCurrency, 1.0)
            return dist

        m1 = trans(pairs1, rates1)
        m2 = trans(pairs2, rates2)
        return max(m1.get(k, 0.0) / v for k, v in m2.items())

    # 1930. 长度为 3 的不同回文子序列 (Unique Length-3 Palindromic Subsequences)
    def countPalindromicSubsequence(self, s: str) -> int:
        n = len(s)
        a = [0] * 26
        pre = [0] * n
        for i in range(1, n):
            pre[i] = pre[i - 1] | (1 << (ord(s[i - 1]) - ord("a")))
        suf = 0
        for i in range(n - 2, 0, -1):
            suf |= 1 << (ord(s[i + 1]) - ord("a"))
            a[ord(s[i]) - ord("a")] |= pre[i] & suf
        return sum(x.bit_count() for x in a)

    # 3750. 最少反转次数得到翻转二进制字符串 (Minimum Number of Flips to Reverse Binary String)
    def minimumFlips(self, n: int) -> int:
        l = n.bit_length()
        return (
            sum(((n >> i) & 1) ^ ((n >> (l - i - 1)) & 1) for i in range(l // 2)) << 1
        )

    # 3751. 范围内总波动值 I (Total Waviness of Numbers in Range I)
    # 3753. 范围内总波动值 II (Total Waviness of Numbers in Range II)
    def totalWaviness(self, num1: int, num2: int) -> int:
        def cal(x: int) -> int:
            @cache
            def dfs(
                i: int, pre: int, pre2: int, j: int, is_limit: bool, is_num: bool
            ) -> int:
                if i == n:
                    return j
                res = 0
                if not is_num:
                    res += dfs(i + 1, pre, pre2, j, False, False)
                up = int(s[i]) if is_limit else 9
                for d in range(0 if is_num else 1, up + 1):
                    add = pre != -1 and pre2 != -1 and (pre - d) * (pre - pre2) > 0
                    res += dfs(i + 1, d, pre, j + add, is_limit and d == up, True)
                return res

            s = str(x)
            n = len(s)
            if n < 3:
                return 0
            return dfs(0, -1, -1, 0, True, False)

        return cal(num2) - cal(num1 - 1)

    # 3752. 字典序最小和为目标值且绝对值是排列的数组 (Lexicographically Smallest Negated Permutation that Sums to Target)
    def lexSmallestNegatedPerm(self, n: int, target: int) -> List[int]:
        s = (1 + n) * n // 2
        if abs(target) > s or (s - target) % 2:
            return []
        res = [0] * n
        neg_s = (s - target) // 2
        left, right = 0, n - 1
        for x in range(n, 0, -1):
            if neg_s >= x:
                res[left] = -x
                neg_s -= x
                left += 1
            else:
                res[right] = x
                right -= 1
        return res

    # 3754. 连接非零数字并乘以其数字和 I (Concatenate Non-Zero Digits and Multiply by Sum I)
    def sumAndMultiply(self, n: int) -> int:
        x = 0
        p = 1
        s = 0
        while n:
            d, m = divmod(n, 10)
            if m:
                s += m
                x += p * m
                p *= 10
            n = d
        return x * s

    # 3756. 连接非零数字并乘以其数字和 II (Concatenate Non-Zero Digits and Multiply by Sum II)
    def sumAndMultiply(self, s: str, queries: List[List[int]]) -> List[int]:
        n = len(s)
        MOD = 10**9 + 7
        pre_x = [0] * (n + 1)
        pre_s = [0] * (n + 1)
        cnt = [0] * (n + 1)
        pow10 = [1] * (n + 1)
        for i in range(n):
            pre_s[i + 1] = (pre_s[i] + int(s[i])) % MOD
            pow10[i + 1] = (pow10[i] * 10) % MOD
            if s[i] != "0":
                pre_x[i + 1] = (pre_x[i] * 10 + int(s[i])) % MOD
                cnt[i + 1] = cnt[i] + 1
            else:
                pre_x[i + 1] = pre_x[i]
                cnt[i + 1] = cnt[i]
        res = []
        for l, r in queries:
            total_s = (pre_s[r + 1] - pre_s[l]) % MOD
            total_x = (pre_x[r + 1] - pre_x[l] * pow10[cnt[r + 1] - cnt[l]]) % MOD
            res.append((total_s * total_x) % MOD)
        return res

    # 3755. 最大平衡异或子数组的长度 (Find Maximum Balanced XOR Subarray Length)
    def maxBalancedSubarray(self, nums: List[int]) -> int:
        d = defaultdict(int)
        d[(0, 0)] = -1
        xor = 0
        diff = 0
        res = 0
        for i, x in enumerate(nums):
            xor ^= x
            diff += 1 if x & 1 else -1
            if (xor, diff) in d:
                res = max(res, i - d[(xor, diff)])
            else:
                d[(xor, diff)] = i
        return res

    # 1018. 可被 5 整除的二进制前缀 (Binary Prefix Divisible By 5)
    def prefixesDivBy5(self, nums: List[int]) -> List[bool]:
        res = []
        v = 0
        for x in nums:
            v = ((v << 1) | x) % 5
            res.append(v == 0)
        return res

    # 1015. 可被 K 整除的最小整数 (Smallest Integer Divisible by K)
    def smallestRepunitDivByK(self, k: int) -> int:
        if k % 2 == 0 or k % 5 == 0:
            return -1
        res = 1
        v = 1 % k
        while v != 0:
            v = (v * 10 + 1) % k
            res += 1
        return res

    # 3575. 最大好子树分数 (Maximum Good Subtree Score)
    def goodSubtreeSum(self, vals: List[int], par: List[int]) -> int:
        def check(v: int) -> int:
            mask = 0
            while v:
                d, m = divmod(v, 10)
                if (mask >> m) & 1:
                    return 0
                mask |= 1 << m
                v = d
            return mask

        def dfs_tree(x: int) -> list:
            @cache
            def dfs(i: int, j: int) -> int:
                if i == 10 or j == (1 << 10) - 1:
                    return 0
                # 不选
                res = dfs(i + 1, j)
                # 选
                if (j >> i) & 1 == 0:
                    # masks[y] 一定不是0
                    for m, v in ret[i]:
                        if j & m == 0:
                            res = max(res, dfs(i + 1, j | m) + v)
                return res

            ret = [set() for _ in range(10)]
            for y in g[x]:
                for i, l in enumerate(dfs_tree(y)):
                    ret[i].update(l)
            mask = masks[x]
            while mask:
                lb = (mask & -mask).bit_length() - 1
                ret[lb].add((masks[x], vals[x]))
                mask &= mask - 1
            nonlocal res
            res += dfs(0, 0)
            res %= MOD
            return ret

        n = len(vals)
        MOD = 10**9 + 7
        masks = [0] * n
        g = [[] for _ in range(n)]
        for i, (v, p) in enumerate(zip(vals, par)):
            if i:
                g[p].append(i)
            masks[i] = check(v)
        res = 0
        dfs_tree(0)
        return res

    # 1590. 使数组和能被 P 整除 (Make Sum Divisible by P)
    def minSubarray(self, nums: List[int], p: int) -> int:
        d = defaultdict(int)
        d[0] = -1
        s = sum(nums)
        m = s % p
        if m == 0:
            return 0
        res = inf
        pre = 0
        for i, x in enumerate(nums):
            pre += x
            res = min(res, i - d.get((pre - m) % p, -inf))
            d[pre % p] = i
        return res if res < len(nums) else -1

    # 3759. 统计合格元素的数目 (Count Elements With at Least K Greater Values)
    def countElements(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if k == 0:
            return n
        nums.sort()
        return bisect.bisect_left(nums, nums[-k])

    # 3760. 不同首字母的子字符串数目 (Maximum Substrings With Distinct Start)
    def maxDistinct(self, s: str) -> int:
        m = 0
        for x in s:
            m |= 1 << (ord(x) - ord("a"))
        return m.bit_count()

    # 3761. 镜像对之间最小绝对距离 (Minimum Absolute Distance Between Mirror Pairs)
    def minMirrorPairDistance(self, nums: List[int]) -> int:
        def reverse(x: int) -> int:
            res = 0
            while x:
                d, m = divmod(x, 10)
                res = res * 10 + m
                x = d
            return res

        d = defaultdict(int)
        res = inf
        for i, x in enumerate(nums):
            res = min(res, i - d.get(x, -inf))
            d[reverse(x)] = i
        return res if res < inf else -1

    # 2141. 同时运行 N 台电脑的最长时间 (Maximum Running Time of N Computers)
    def maxRunTime(self, n: int, batteries: List[int]) -> int:
        l, r = 0, sum(batteries) // n
        while l <= r:
            x = (l + r) // 2
            if n * x <= sum(min(b, x) for b in batteries):
                l = x + 1
            else:
                r = x - 1
        return l - 1

    # 2211. 统计道路上的碰撞次数 (Count Collisions on a Road)
    def countCollisions(self, directions: str) -> int:
        n = len(directions)
        left = 0
        right = n - 1
        while left < n and directions[left] == "L":
            left += 1
        while right >= 0 and directions[right] == "R":
            right -= 1
        return sum(x != "S" for x in directions[left : right + 1])

    # 1523. 在区间范围内统计奇数数目 (Count Odd Numbers in an Interval Range)
    def countOdds(self, low: int, high: int) -> int:
        return (high - low + 1) // 2 + ((low & 1) + (high & 1)) // 2

    # 765. 完全质数 (Complete Prime Number)
    def completePrime(self, num: int) -> bool:
        def check(x: int) -> bool:
            for i in range(2, isqrt(x) + 1):
                if x % i == 0:
                    return False
            return x > 1

        s = str(num)
        n = len(s)
        for i in range(1, n + 1):
            if not check(int(s[:i])):
                return False
        for i in range(n - 1, -1, -1):
            if not check(int(s[i:])):
                return False
        return True

    # 3766. 将数字变成二进制回文数的最少操作 (Minimum Operations to Make Binary Palindrome)
    def minOperations(self, nums: List[int]) -> List[int]:
        def binary_search_left(x: int) -> int:
            left = 0
            right = len(p) - 1
            while left <= right:
                mid = left + ((right - left) >> 1)
                if p[mid] <= x:
                    left = mid + 1
                else:
                    right = mid - 1
            return left - 1

        def binary_search_right(x: int) -> int:
            left = 0
            right = len(p) - 1
            while left <= right:
                mid = left + ((right - left) >> 1)
                if p[mid] >= x:
                    right = mid - 1
                else:
                    left = mid + 1
            return right + 1

        def rev(x) -> int:
            res = 0
            while x:
                res = (res << 1) | (x & 1)
                x >>= 1
            return res

        p = []
        for i in range(5050):
            r = rev(i)
            if i == r:
                p.append(i)
        return [
            min(x - p[binary_search_left(x)], p[binary_search_right(x)] - x)
            for x in nums
        ]

    # 3767. 选择 K 个任务的最大总分数 (Maximize Points After Choosing K Tasks)
    def maxPoints(self, technique1: List[int], technique2: List[int], k: int) -> int:
        d = sorted(
            (y - x for x, y in zip(technique1, technique2) if y > x), reverse=True
        )
        return sum(technique1) + sum(d[: len(technique1) - k])

    # 3769. 二进制反射排序 (Sort Integers by Binary Reflection)
    def sortByReflection(self, nums: List[int]) -> List[int]:
        def rev(x) -> int:
            res = 0
            while x:
                res = (res << 1) | (x & 1)
                x >>= 1
            return res

        a = [(rev(x), x) for x in nums]
        a.sort(key=lambda o: (o[0], o[1]))
        return [x for _, x in a]

    # 3770. 可表示为连续质数和的最大质数 (Largest Prime from Consecutive Prime Sum)
    def largestPrime(self, n: int) -> int:
        # 筛质数放在类外面
        MX = 10**6
        p = [True] * MX
        for i in range(2, MX):
            if p[i]:
                for j in range(i * i, MX, i):
                    p[j] = False
        s = 0
        res = 0
        for i in range(2, MX):
            if p[i]:
                s += i
                if s > n:
                    break
                if p[s]:
                    res = s
        return res

    # 1925. 统计平方和三元组的数目 (Count Square Sum Triples)
    def countTriples(self, n: int) -> int:
        s = set()
        for i in range(n + 1):
            s.add(i * i)
        res = 0
        for i in range(1, n + 1):
            for j in range(i, n + 1):
                if i * i + j * j in s:
                    res += 1
        return res * 2

    # 3771. 探索地牢的得分 (Total Score of Dungeon Runs)
    def totalScore(self, hp: int, damage: List[int], requirement: List[int]) -> int:
        def binary_search(x: int, i: int) -> int:
            left = 0
            right = i
            while left <= right:
                mid = left + ((right - left) >> 1)
                if s[mid] >= x:
                    right = mid - 1
                else:
                    left = mid + 1
            return right + 1

        n = len(damage)
        s = list(accumulate(damage, initial=0))
        res = 0
        for i in range(n):
            x = s[i + 1] + requirement[i] - hp
            j = binary_search(x, i)
            res += i - j + 1
        return res

    # 3772. 子图的最大得分 (Maximum Subgraph Score in a Tree)
    def maxSubgraphScore(
        self, n: int, edges: List[List[int]], good: List[int]
    ) -> List[int]:
        def reroot(x: int, fa: int, fa_score: int):
            res[x] = score_x = s[x] + max(fa_score, 0)
            for y in g[x]:
                if y != fa:
                    reroot(y, x, score_x - max(s[y], 0))

        def dfs(x: int, fa: int) -> int:
            for y in g[x]:
                if y != fa:
                    s[x] += max(0, dfs(y, x))
            s[x] += 1 if good[x] else -1
            return s[x]

        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        s = [0] * n
        dfs(0, -1)
        res = [0] * n
        reroot(0, -1, 0)
        return res

    # 2110. 股票平滑下跌阶段的数目 (Number of Smooth Descent Periods of a Stock)
    def getDescentPeriods(self, prices: List[int]) -> int:
        cnt = res = 0
        for i, x in enumerate(prices):
            if i == 0 or prices[i - 1] - x != 1:
                cnt = 1
            else:
                cnt += 1
            res += cnt
        return res

    # 3774. 最大和最小 K 个元素的绝对差 (Absolute Difference Between Maximum and Minimum K Elements)
    def absDifference(self, nums: List[int], k: int) -> int:
        nums.sort()
        return sum(nums[-k:]) - sum(nums[:k])

    # 3775. 反转元音数相同的单词 (Reverse Words With Same Vowel Count)
    def reverseWords(self, s: str) -> str:
        def check(w: str) -> int:
            cnt = 0
            for c in w:
                cnt += (mask >> (ord(c) - ord("a"))) & 1
            return cnt

        mask = 0
        for x in "aeiou":
            mask |= 1 << (ord(x) - ord("a"))
        cnt = -1
        split = s.split(" ")
        res = ""
        for word in split:
            if len(res):
                res += " "
            if cnt == -1:
                cnt = check(word)
                res += word
            elif check(word) == cnt:
                res += "".join(list(reversed(word)))
            else:
                res += word
        return res

    # 3776. 使循环数组余额非负的最少移动次数 (Minimum Moves to Balance Circular Array)
    def minMoves(self, balance: List[int]) -> int:
        n = len(balance)
        s = 0
        id = -1
        need = -1
        for i, x in enumerate(balance):
            s += x
            if x < 0:
                id = i
                need = x
        if s < 0:
            return -1
        if id < 0:
            return 0
        need = abs(need)
        d = 1
        res = 0
        while need:
            cur = min(balance[(id - d) % n] + balance[(id + d) % n], need)
            res += cur * d
            need -= cur
            d += 1
        return res

    # 2092. 找出知晓秘密的所有专家 (Find All People With Secret)
    def findAllPeople(
        self, n: int, meetings: List[List[int]], firstPerson: int
    ) -> List[int]:
        meetings.sort(key=lambda o: o[2])
        knows = [False] * n
        knows[0] = knows[firstPerson] = True
        i = 0
        while i < len(meetings):
            vis = set()
            g = defaultdict(list)
            j = i
            t = meetings[i][2]
            while j < len(meetings) and t == meetings[j][2]:
                x = meetings[j][0]
                y = meetings[j][1]
                g[x].append(y)
                g[y].append(x)
                j += 1

            def dfs(x: int):
                vis.add(x)
                knows[x] = True
                for y in g[x]:
                    if y not in vis:
                        dfs(y)

            for x in g:
                if knows[x] and x not in vis:
                    dfs(x)
            i = j
        return [i for i, x in enumerate(knows) if x]

    # 944. 删列造序 (Delete Columns to Make Sorted)
    def minDeletionSize(self, strs: List[str]) -> int:
        n, m = len(strs), len(strs[0])
        res = 0
        for j in range(m):
            for i in range(1, n):
                if strs[i][j] < strs[i - 1][j]:
                    res += 1
                    break
        return res

    # 955. 删列造序 II (Delete Columns to Make Sorted II)
    def minDeletionSize(self, strs: List[str]) -> int:
        n, m = len(strs), len(strs[0])
        a = [""] * n  # 最终得到的字符串数组
        ans = 0
        for j in range(m):
            for i in range(n - 1):
                if a[i] + strs[i][j] > a[i + 1] + strs[i + 1][j]:
                    # j 列不是升序，必须删
                    ans += 1
                    break
            else:
                # j 列是升序，不删更好
                for i, s in enumerate(strs):
                    a[i] += s[j]
        return ans

    # 960. 删列造序 III (Delete Columns to Make Sorted III)
    def minDeletionSize(self, strs: List[str]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == m:
                return 0
            # 删
            res = dfs(i + 1, j)
            # 不删
            if j == -1 or all(strs[k][i] >= strs[k][j] for k in range(n)):
                res = max(res, dfs(i + 1, i) + 1)
            return res

        n, m = len(strs), len(strs[0])
        return m - dfs(0, -1)

    # 3779. 得到互不相同元素的最少操作次数 (Minimum Number of Operations to Have Distinct Elements)
    def minOperations(self, nums: List[int]) -> int:
        cnts = defaultdict(int)
        for x in nums:
            cnts[x] += 1
        i = 0
        n = len(nums)
        res = 0
        while n - i != len(cnts):
            res += 1
            for j in range(i, min(i + 3, n)):
                cnts[nums[j]] -= 1
                if cnts[nums[j]] == 0:
                    del cnts[nums[j]]
            i = min(i + 3, n)
        return res

    # 3780. 能被 3 整除的三元组最大和 (Maximum Sum of Three Numbers Divisible by Three)
    def maximumSum(self, nums: List[int]) -> int:
        g = [[0] * 3 for _ in range(3)]
        for x in nums:
            m = x % 3
            if x >= g[m][0]:
                g[m][2] = g[m][1]
                g[m][1] = g[m][0]
                g[m][0] = x
            elif x >= g[m][1]:
                g[m][2] = g[m][1]
                g[m][1] = x
            elif x >= g[m][2]:
                g[m][2] = x
        res = 0
        for i in range(3):
            if g[i][-1] > 0:
                res = max(res, sum(g[i]))
        if all(g[i][0] > 0 for i in range(3)):
            res = max(res, g[0][0] + g[1][0] + g[2][0])
        return res

    # 3783. 整数的镜像距离 (Mirror Distance of an Integer)
    def mirrorDistance(self, n: int) -> int:
        def rev(x: int) -> int:
            res = 0
            while x:
                d, m = divmod(x, 10)
                res = res * 10 + m
                x = d
            return res

        return abs(n - rev(n))

    # 3784. 使所有字符相等的最小删除代价 (Minimum Deletion Cost to Make All Characters Equal)
    def minCost(self, s: str, cost: List[int]) -> int:
        _sum = sum(cost)
        d = [0] * 26
        for ch, c in zip(s, cost):
            idx = ord(ch) - ord("a")
            d[idx] += c
        return _sum - max(d)

    # 3786. 树组的交互代价总和 (Total Sum of Interaction Cost in Tree Groups)
    def interactionCosts(self, n: int, edges: List[List[int]], group: List[int]) -> int:
        def dfs(x: int, fa: int) -> int:
            nonlocal res
            cnt_x = group[x] == c
            for y in g[x]:
                if y != fa:
                    cnt_y = dfs(y, x)
                    res += (tot - cnt_y) * cnt_y
                    cnt_x += cnt_y
            return cnt_x

        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        d = defaultdict(int)
        for c in group:
            d[c] += 1
        res = 0
        for c, tot in d.items():
            dfs(0, -1)
        return res

    # 3781. 二进制交换后的最大分数 (Maximum Score After Binary Swaps)
    def maximumScore(self, nums: List[int], s: str) -> int:
        q = []
        heapq.heapify(q)
        for x, c in zip(reversed(nums), reversed(s)):
            heapq.heappush(q, x)
            if c == "0":
                heapq.heappop(q)
        return sum(q)

    # 3785. 避免禁用值的最小交换次数 (Minimum Swaps to Avoid Forbidden Values)
    def minSwaps(self, nums: List[int], forbidden: List[int]) -> int:
        n = len(nums)
        cnt = defaultdict(int)
        for x, f in zip(nums, forbidden):
            cnt[x] += 1
            cnt[f] += 1
        if any(x > n for x in cnt.values()):
            return -1
        a = defaultdict(int)
        for x, f in zip(nums, forbidden):
            if x != f:
                continue
            a[x] += 1
        return max(max(a.values(), default=0), (sum(a.values()) + 1) // 2)

    # 3782. 交替删除操作后最后剩下的整数 (Last Remaining Integer After Alternating Deletion Operations)
    def lastInteger(self, n: int) -> int:
        start = d = 1
        while n > 1:
            start += (n - 2 + n % 2) * d
            d *= -2
            n = (n + 1) // 2
        return start

    # 2483. 商店的最少代价 (Minimum Penalty for a Shop)
    def bestClosingTime(self, customers: str) -> int:
        n = len(customers)
        cost = customers.count("Y")
        min_cost = cost
        res = 0
        for i in range(1, n + 1):
            cost += -1 if customers[i - 1] == "Y" else 1
            if cost < min_cost:
                min_cost = cost
                res = i
        return res

    # 2402. 会议室 III (Meeting Rooms III)
    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        cnts = [0] * n
        idle_rooms = [i for i in range(n)]
        heapq.heapify(idle_rooms)
        # (i, j) is end time, j is room id
        busy_rooms = []
        heapq.heapify(busy_rooms)
        meetings.sort()
        for start, end in meetings:
            while busy_rooms and busy_rooms[0][0] <= start:
                _, room_id = heapq.heappop(busy_rooms)
                heapq.heappush(idle_rooms, room_id)
            if idle_rooms:
                room_id = heapq.heappop(idle_rooms)
                cnts[room_id] += 1
                heapq.heappush(busy_rooms, (end, room_id))
            else:
                free_time, room_id = heapq.heappop(busy_rooms)
                cnts[room_id] += 1
                duration = end - start
                heapq.heappush(busy_rooms, (free_time + duration, room_id))
        return cnts.index(max(cnts))

    # 1351. 统计有序矩阵中的负数 (Count Negative Numbers in a Sorted Matrix)
    def countNegatives(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        res = 0
        i, j = 0, n - 1
        while i < m and j >= 0:
            if grid[i][j] < 0:
                res += m - i
                j -= 1
            else:
                i += 1
        return res

    # 756. 金字塔转换矩阵 (Pyramid Transition Matrix)
    def pyramidTransition(self, bottom: str, allowed: List[str]) -> bool:
        groups = defaultdict(list)
        for s in allowed:
            groups[s[:2]].append(s[2])

        n = len(bottom)
        pyramid = [[] for _ in range(n)]
        pyramid[-1] = bottom

        vis = set()

        def dfs(i: int, j: int) -> bool:
            if i < 0:
                return True

            row = "".join(pyramid[i])
            if row in vis:  # 之前填过一模一样的，这个局部的金字塔无法填完
                return False  # 继续递归也无法填完，直接返回

            if j == i + 1:
                vis.add(row)
                return dfs(i - 1, 0)

            for top in groups[pyramid[i + 1][j] + pyramid[i + 1][j + 1]]:
                pyramid[i].append(top)
                if dfs(i, j + 1):
                    return True
                pyramid[i].pop()
            return False

        return dfs(n - 2, 0)

    # 788. 分割的最大得分 (Maximum Score of a Split)
    def maximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        suf_min = [inf] * n
        for i in range(n - 2, -1, -1):
            suf_min[i] = min(suf_min[i + 1], nums[i + 1])
        res = -inf
        pre_sum = 0
        for i in range(n - 1):
            pre_sum += nums[i]
            res = max(res, pre_sum - suf_min[i])
        return res

    # 3789. 采购的最小花费 (Minimum Cost to Acquire Required Items)
    def minimumCost(
        self, cost1: int, cost2: int, costBoth: int, need1: int, need2: int
    ) -> int:
        res1 = need1 * cost1 + need2 * cost2
        res2 = max(need1, need2) * costBoth
        m = min(need1, need2)
        res3 = m * costBoth
        need1 -= m
        need2 -= m
        res3 += need1 * cost1 + need2 * cost2
        return min(res1, res2, res3)

    # 3790. 最小全 1 倍数 (mallest All-Ones Multiple)
    def minAllOneMultiple(self, k: int) -> int:
        if k % 2 == 0 or k % 5 == 0:
            return -1
        rem = 0
        for i in range(1, k + 1):
            rem = (rem * 10 + 1) % k
            if rem == 0:
                return i
        return -1

    # 3791. 给定范围内平衡整数的数目 (Number of Balanced Integers in a Range)
    def countBalanced(self, low: int, high: int) -> int:
        def cal(x: int) -> int:
            @cache
            def dfs(i: int, j: int, is_limit: bool) -> int:
                if i == n:
                    return j == 0
                res = 0
                up = int(s[i]) if is_limit else 9
                for d in range(up + 1):
                    nj = j + (d if i % 2 else -d)
                    res += dfs(i + 1, nj, is_limit and d == up)
                return res

            s = str(x)
            n = len(s)
            return dfs(0, 0, True)

        return cal(high) - cal(low - 1)

    # 840. 矩阵中的幻方 (Magic Squares In Grid)
    def numMagicSquaresInside(self, grid: List[List[int]]) -> int:
        def check(x0: int, y0: int) -> bool:
            def check_unique() -> bool:
                mask = 0
                for x in range(x0, x0 + 3):
                    for y in range(y0, y0 + 3):
                        if grid[x][y] > 9 or grid[x][y] < 1:
                            return False
                        if (mask >> grid[x][y]) & 1:
                            return False
                        mask |= 1 << grid[x][y]
                return True

            def check_row() -> bool:
                for x in range(x0, x0 + 3):
                    s = 0
                    for y in range(y0, y0 + 3):
                        s += grid[x][y]
                    if s != 15:
                        return False
                return True

            def check_col() -> bool:
                for y in range(y0, y0 + 3):
                    s = 0
                    for x in range(x0, x0 + 3):
                        s += grid[x][y]
                    if s != 15:
                        return False
                return True

            return check_unique() and check_row() and check_col()

        m, n = len(grid), len(grid[0])
        res = 0
        for i in range(2, m):
            for j in range(2, n):
                if grid[i - 1][j - 1] == 5 and check(i - 2, j - 2):
                    res += 1
        return res

    # 1970. 你能穿过矩阵的最后一天 (Last Day Where You Can Still Cross)
    def latestDayToCross(self, row: int, col: int, cells: List[List[int]]) -> int:
        class union:
            def __init__(self, n: int):
                self.rank = [1] * n
                self.parent = [i for i in range(n)]

            def get_root(self, p: int) -> int:
                if self.parent[p] == p:
                    return p
                self.parent[p] = self.get_root(self.parent[p])
                return self.parent[p]

            def is_connect(self, p1: int, p2: int) -> bool:
                return self.get_root(p1) == self.get_root(p2)

            def union(self, p1: int, p2: int) -> None:
                r1 = self.get_root(p1)
                r2 = self.get_root(p2)
                if r1 == r2:
                    return
                if self.rank[r1] < self.rank[r2]:
                    self.parent[r1] = r2
                else:
                    self.parent[r2] = r1
                    if self.rank[r1] == self.rank[r2]:
                        self.rank[r1] += 1

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        n = len(cells)
        u = union(n + 2)
        g = [[0] * col for _ in range(row)]
        for i in range(col):
            u.union(n, i)
            u.union(n + 1, n - i - 1)
        for i in range(n - 1, -1, -1):
            r, c = cells[i]
            r -= 1
            c -= 1
            g[r][c] = 1
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < row and 0 <= nc < col and g[nr][nc] == 1:
                    u.union(r * col + c, nr * col + nc)
            if u.is_connect(n, n + 1):
                return i

    # 961. 在长度 2N 的数组中找出重复 N 次的元素 (N-Repeated Element in Size 2N Array)
    def repeatedNTimes(self, nums: List[int]) -> int:
        # s = set()
        # for x in nums:
        #     if x in s:
        #         return x
        #     s.add(x)
        m = 0
        for x in nums:
            if m >> x & 1:
                return x
            m |= 1 << x

    # 1411. 给 N x 3 网格图涂色的方案数 (Number of Ways to Paint N × 3 Grid)
    def numOfWays(self, n: int) -> int:
        MOD = 10**9 + 7
        a = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i != j and j != k:
                        m = 1 << i << 6 | 1 << j << 3 | 1 << k
                        a.append(m)

        @cache
        def dfs(i: int, j: tuple) -> int:
            if i == n:
                return 1
            return sum(dfs(i + 1, x) for x in a if x & j == 0) % MOD

        return dfs(0, 0)

    # 1390. 四因数 (Four Divisors)
    def sumFourDivisors(self, nums: List[int]) -> int:
        def cal(x: int) -> int:
            s = 0
            cnt = 0
            for i in range(1, isqrt(x) + 1):
                if x % i == 0:
                    s += i
                    cnt += 1
                    if i * i != x:
                        s += x // i
                        cnt += 1
                    if cnt > 4:
                        return 0
            return s if cnt == 4 else 0

        return sum(cal(x) for x in nums)

    # 3794. 反转字符串前缀 (Reverse String Prefix)
    def reversePrefix(self, s: str, k: int) -> str:
        return s[:k][::-1] + s[k:]

    # 3795. 不同元素和至少为 K 的最短子数组长度 (Minimum Subarray Length With Distinct Sum At Least K)
    def minLength(self, nums: List[int], k: int) -> int:
        cnts = defaultdict(int)
        left = s = 0
        res = inf
        for right, x in enumerate(nums):
            cnts[x] += 1
            if cnts[x] == 1:
                s += x
            while s >= k:
                res = min(res, right - left + 1)
                cnts[nums[left]] -= 1
                if cnts[nums[left]] == 0:
                    s -= nums[left]
                    del cnts[nums[left]]
                left += 1
        return res if res < inf else -1

    # 3797. 统计在矩形格子里移动的路径数目 (Count Routes to Climb a Rectangular Grid)
    def numberOfRoutes(self, grid: List[str], d: int) -> int:
        m, n = len(grid), len(grid[0])
        MOD = 10**9 + 7
        pre = list(
            accumulate([int(grid[m - 1][j] == ".") for j in range(n)], initial=0)
        )
        f = [0] * n
        for j in range(n):
            if grid[m - 1][j] == "#":
                continue
            f[j] = pre[min(n, j + d + 1)] - pre[max(0, j - d)]
        dis = isqrt(d * d - 1)
        for i in range(m - 2, -1, -1):
            new_f = [0] * n
            cur = [0] * n
            pre = list(accumulate(f, initial=0))
            for j in range(n):
                if grid[i][j] == "#":
                    continue
                cur[j] = pre[min(n, j + dis + 1)] - pre[max(0, j - dis)]
            pre = list(accumulate(cur, initial=0))
            for j in range(n):
                if grid[i][j] == "#":
                    continue
                new_f[j] = pre[min(n, j + d + 1)] - pre[max(0, j - d)]
                new_f[j] %= MOD
            f = new_f
        return sum(f) % MOD

    # 3798. 最大的偶数 (Largest Even Number)
    def largestEven(self, s: str) -> str:
        for i in range(len(s) - 1, -1, -1):
            if s[i] == "2":
                return s[: i + 1]
        return ""

    # 3799. 单词方块 II (Word Squares II)
    def wordSquares(self, words: List[str]) -> List[List[str]]:
        def dfs(m: int):
            if len(a) == 4:
                res.append(a.copy())
                return
            for id in range(n):
                if m >> id & 1:
                    continue
                if len(a) == 1 and a[-1][0] != words[id][0]:
                    continue
                if len(a) == 2 and a[0][-1] != words[id][0]:
                    continue
                if len(a) == 3 and (
                    a[1][-1] != words[id][0] or a[2][-1] != words[id][-1]
                ):
                    continue
                a.append(words[id])
                dfs(m | (1 << id))
                a.pop()

        n = len(words)
        words.sort()
        res = []
        a = []
        dfs(0)
        return res

    # 3800. 使二进制字符串相等的最小成本 (Minimum Cost to Make Two Binary Strings Equal)
    def minimumCost(
        self, s: str, t: str, flipCost: int, swapCost: int, crossCost: int
    ) -> int:
        x, y = 0, 0
        for a, b in zip(s, t):
            if a == b:
                continue
            if a == "0":
                x += 1
            else:
                y += 1
        if x < y:
            x, y = y, x
        res1 = (x + y) * flipCost
        res2 = flipCost if (x + y) & 1 else 0
        if (x + y) & 1:
            x -= 1
        res2 += y * swapCost
        x -= y
        res2 += min(x // 2 * (swapCost + crossCost), flipCost * x)
        return min(res1, res2)

    # 3796. 找到带限制序列的最大值 (Find Maximum Value in a Constrained Sequence)
    def findMaxVal(self, n: int, restrictions: List[List[int]], diff: List[int]) -> int:
        a = [0] * n
        for i, x in restrictions:
            a[i] = x
        for i, x in enumerate(diff, start=1):
            if a[i]:
                a[i] = min(a[i], a[i - 1] + x)
            else:
                a[i] = a[i - 1] + x
        res = a[-1]
        for i in range(n - 2, -1, -1):
            a[i] = min(a[i], a[i + 1] + diff[i])
            res = max(res, a[i])
        return res

    # 1975. 最大方阵和 (Maximum Matrix Sum)
    def maxMatrixSum(self, matrix: List[List[int]]) -> int:
        cnt_neg = 0
        min_abs = inf
        n = len(matrix)
        s = 0
        for i in range(n):
            for j in range(n):
                if matrix[i][j] < 0:
                    cnt_neg += 1
                min_abs = min(min_abs, abs(matrix[i][j]))
                s += abs(matrix[i][j])
        if cnt_neg & 1:
            s -= min_abs * 2
        return s

    # 3801. 合并有序列表的最小成本 (Minimum Cost to Merge Sorted Lists)
    def minMergeCost(self, lists: List[List[int]]) -> int:
        def merge(a: list, b: list) -> list:
            res = []
            i = j = 0
            while i < len(a) and j < len(b):
                if a[i] < b[j]:
                    res.append(a[i])
                    i += 1
                else:
                    res.append(b[j])
                    j += 1
            res.extend(a[i:])
            res.extend(b[j:])
            return res

        n = len(lists)
        g = [[] for _ in range(1 << n)]
        for i, a in enumerate(lists):  # 枚举不在 s 中的下标 i
            high_bit = 1 << i
            for s in range(high_bit):
                g[high_bit | s] = merge(g[s], a)
        f = [0] * (1 << n)
        for i in range(1, 1 << n):
            if i & (i - 1) == 0:
                continue
            f[i] = inf
            sub_a = i
            # 避免对称枚举
            while sub_a > (i ^ sub_a):
                if sub_a ^ i:
                    sub_b = sub_a ^ i
                    len_a = len(g[sub_a])
                    len_b = len(g[sub_b])
                    med_a = g[sub_a][(len_a - 1) // 2]
                    med_b = g[sub_b][(len_b - 1) // 2]
                    f[i] = min(
                        f[i], f[sub_a] + f[sub_b] + len_a + len_b + abs(med_a - med_b)
                    )
                sub_a = (sub_a - 1) & i
        return f[-1]

    # 1161. 最大层内元素和 (Maximum Level Sum of a Binary Tree)
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        max_s = -inf
        level = 0
        res = 0
        q = deque()
        q.append(root)
        while q:
            level += 1
            _len = len(q)
            s = 0
            for _ in range(_len):
                x = q.popleft()
                s += x.val
                if x.left:
                    q.append(x.left)
                if x.right:
                    q.append(x.right)
            if s > max_s:
                max_s = s
                res = level
        return res

    # 1339. 分裂二叉树的最大乘积 (Maximum Product of Splitted Binary Tree)
    def maxProduct(self, root: Optional[TreeNode]) -> int:
        def dfs(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            return node.val + dfs(node.left) + dfs(node.right)

        def cal(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            left_s = cal(node.left)
            right_s = cal(node.right)
            nonlocal res
            res = max(res, left_s * (total - left_s), right_s * (total - right_s))
            return node.val + left_s + right_s

        MOD = 10**9 + 7
        total = dfs(root)
        res = 0
        cal(root)
        return res % MOD

    # 85. 最大矩形 (Maximal Rectangle)
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        def cal() -> int:
            left = [-1] * n
            st = []
            for id, x in enumerate(heights):
                while st and heights[st[-1]] >= x:
                    st.pop()
                if st:
                    left[id] = st[-1]
                st.append(id)
            right = [n] * n
            st.clear()
            for id in range(n - 1, -1, -1):
                while st and heights[st[-1]] >= heights[id]:
                    st.pop()
                if st:
                    right[id] = st[-1]
                st.append(id)
            return max((r - l - 1) * h for l, r, h in zip(left, right, heights))

        m, n = len(matrix), len(matrix[0])
        heights = [0] * n
        res = 0
        for i in range(m):
            for j in range(n):
                heights[j] = heights[j] + 1 if matrix[i][j] == "1" else 0
            res = max(res, cal())
        return res

    # 1266. 访问所有点的最小时间 (Minimum Time Visiting All Points)
    def minTimeToVisitAllPoints(self, points: List[List[int]]) -> int:
        res = 0
        for (x1, y1), (x2, y2) in pairwise(points):
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            res += max(dx, dy)
        return res

    # 3803. 统计残差前缀 (Count Residue Prefixes)
    def residuePrefixes(self, s: str) -> int:
        res = 0
        mask = 0
        for i, x in enumerate(s):
            mask |= 1 << (ord(x) - ord("a"))
            if mask.bit_count() == 3:
                break
            if mask.bit_count() == (i + 1) % 3:
                res += 1
        return res

    # 3804. 中心子数组的数量 (Number of Centered Subarrays)
    def centeredSubarrays(self, nums: List[int]) -> int:
        res = 0
        n = len(nums)
        s = set()
        for i in range(n):
            s.clear()
            _sum = 0
            for j in range(i, n):
                s.add(nums[j])
                _sum += nums[j]
                if _sum in s:
                    res += 1
        return res

    # 3805. 统计凯撒加密对数目 (Count Caesar Cipher Pairs)
    def countPairs(self, words: List[str]) -> int:
        def trans(s: str) -> str:
            a = [x for x in s]
            d = 26 - (ord(a[0]) - ord("a"))
            if d == 0:
                return s
            for i in range(len(a)):
                id = ord(a[i]) - ord("a")
                n_id = (id + d) % 26
                a[i] = chr(ord("a") + n_id)
            return "".join(a)

        res = 0
        d = defaultdict(int)
        for s in words:
            trans_s = trans(s)
            res += d[trans_s]
            d[trans_s] += 1
        return res

    # 3806. 增加操作后最大按位与的结果 (Maximum Bitwise AND After Increment Operations) --试填法
    def maximumAND(self, nums: List[int], k: int, m: int) -> int:
        res = 0
        ops = [0] * (len(nums))
        _max = (max(nums) + k).bit_length()
        for bit in range(_max - 1, -1, -1):
            target = res | (1 << bit)
            for i, x in enumerate(nums):
                j = (target & ~x).bit_length()
                mask = (1 << j) - 1
                ops[i] = (target & mask) - (x & mask)
            ops.sort()
            if sum(ops[:m]) <= k:
                res = target
        return res

    # 3453. 分割正方形 I (Separate Squares I)
    def separateSquares(self, squares: List[List[int]]) -> float:
        def check(t: float) -> float:
            return sum(max(0, min(t, y + r) - y) * r for _, y, r in squares)

        left = min(y for _, y, _ in squares)
        right = max(y + r for _, y, r in squares)
        total = sum(r * r for _, _, r in squares)
        while right - left >= 10**-5:
            mid = left + (right - left) / 2
            if check(mid) * 2 >= total:
                right = mid - 10**-6
            else:
                left = mid + 10**-6
        return right + 10**-6

    # 998. 最大二叉树 II (Maximum Binary Tree II)
    def insertIntoMaxTree(
        self, root: Optional[TreeNode], val: int
    ) -> Optional[TreeNode]:
        fa, x = None, root
        while x:
            if val > x.val:
                if fa is None:
                    return TreeNode(val, root, None)
                fa.right = TreeNode(val, x, None)
                return root
            fa = x
            x = x.right
        node = TreeNode(val)
        fa.right = node
        return root

    # 829. 连续整数求和 (Consecutive Numbers Sum)
    def consecutiveNumbersSum(self, s: int) -> int:
        n = 1
        res = 0
        while n * (n + 1) <= s * 2:
            if s * 2 > n * (n - 1) and (s * 2 - n * (n - 1)) % (n * 2) == 0:
                res += 1
            n += 1
        return res

    # 1895. 最大的幻方 (Largest Magic Square)
    def largestMagicSquare(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        row_sum = [[0] * (n + 1) for _ in range(m)]  # → 前缀和
        col_sum = [[0] * n for _ in range(m + 1)]  # ↓ 前缀和
        diag_sum = [[0] * (n + 1) for _ in range(m + 1)]  # ↘ 前缀和
        anti_sum = [[0] * (n + 1) for _ in range(m + 1)]  # ↙ 前缀和

        for i, row in enumerate(grid):
            for j, x in enumerate(row):
                row_sum[i][j + 1] = row_sum[i][j] + x
                col_sum[i + 1][j] = col_sum[i][j] + x
                diag_sum[i + 1][j + 1] = diag_sum[i][j] + x
                anti_sum[i + 1][j] = anti_sum[i][j + 1] + x

        # k×k 子矩阵的左上角为 (i−k, j−k)，右下角为 (i−1, j−1)
        for k in range(min(m, n), 0, -1):
            for i in range(k, m + 1):
                for j in range(k, n + 1):
                    # 子矩阵主对角线的和
                    s = diag_sum[i][j] - diag_sum[i - k][j - k]

                    # 子矩阵反对角线的和等于 s
                    # 子矩阵每行的和都等于 s
                    # 子矩阵每列的和都等于 s
                    if (
                        anti_sum[i][j - k] - anti_sum[i - k][j] == s
                        and all(
                            row_sum[r][j] - row_sum[r][j - k] == s
                            for r in range(i - k, i)
                        )
                        and all(
                            col_sum[i][c] - col_sum[i - k][c] == s
                            for c in range(j - k, j)
                        )
                    ):
                        return k

    # 3809. 最好可到达的塔 (Best Reachable Tower)
    def bestTower(
        self, towers: List[List[int]], center: List[int], radius: int
    ) -> List[int]:
        res = (1, -1, -1)
        for x, y, q in towers:
            dx = abs(x - center[0])
            dy = abs(y - center[1])
            if dx + dy <= radius:
                res = min(res, (-q, x, y))
        return [res[1], res[2]]

    # 3810. 变成目标数组的最少操作次数 (Minimum Operations to Reach Target Array)
    def minOperations(self, nums: List[int], target: List[int]) -> int:
        s = set()
        for x, y in zip(nums, target):
            if x != y:
                s.add(x)
        return len(s)

    # 3811. 交替按位异或分割的数目 (Number of Alternating XOR Partitions)
    def alternatingXOR(self, nums: List[int], target1: int, target2: int) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i == n:
                return int(j == -1)
            nj = nums[i] if j == -1 else j ^ nums[i]
            res = dfs(i + 1, nj, k)
            if nj == a[k]:
                res += dfs(i + 1, -1, k ^ 1)
            return res % MOD

        MOD = 10**9 + 7
        n = len(nums)
        a = [target1, target2]
        res = dfs(0, -1, 0)
        dfs.cache_clear()
        return res

    # 3811. 交替按位异或分割的数目 (Number of Alternating XOR Partitions)
    def alternatingXOR(self, nums: List[int], target1: int, target2: int) -> int:
        f1, f2 = defaultdict(int), defaultdict(int)
        f2[0] = 1
        xor = 0
        MOD = 10**9 + 7
        for i, x in enumerate(nums):
            xor ^= x
            last1 = f2[target1 ^ xor]
            last2 = f1[target2 ^ xor]
            if i == len(nums) - 1:
                return (last1 + last2) % MOD
            f1[xor] = (f1[xor] + last1) % MOD
            f2[xor] = (f2[xor] + last2) % MOD

    # 3812. 翻转树上最少边 (Minimum Edge Toggles on a Tree)
    def minimumFlips(
        self, n: int, edges: List[List[int]], start: str, target: str
    ) -> List[int]:
        def dfs(x: int, fa: int) -> int:
            cnt = start[x] != target[x]
            for y, id in g[x]:
                if y != fa:
                    cnt_y = dfs(y, x)
                    if cnt_y:
                        res.append(id)
                    cnt ^= cnt_y
            return cnt

        g = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            g[u].append((v, i))
            g[v].append((u, i))
        res = []
        if dfs(0, -1):
            return [-1]
        return sorted(res)

    # 1292. 元素和小于等于阈值的正方形的最大边长 (Maximum Side Length of a Square with Sum Less than or Equal to Threshold)
    def maxSideLength(self, mat: List[List[int]], threshold: int) -> int:
        m, n = len(mat), len(mat[0])
        pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                pre[i + 1][j + 1] = (
                    pre[i][j + 1] + pre[i + 1][j] - pre[i][j] + mat[i][j]
                )

        def check(k: int) -> bool:
            for i in range(m - k + 1):
                for j in range(n - k + 1):
                    total = (
                        pre[i + k][j + k] - pre[i][j + k] - pre[i + k][j] + pre[i][j]
                    )
                    if total <= threshold:
                        return True
            return False

        left, right = 0, min(m, n)
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                left = mid + 1
            else:
                right = mid - 1
        return left - 1

    # 3813. 元音辅音得分 (Vowel-Consonant Score)
    def vowelConsonantScore(self, s: str) -> int:
        u, v, c = 0, 0, 0
        for chr in "aeiou":
            u |= 1 << (ord(chr) - ord("a"))
        for chr in s:
            if chr.isalpha():
                id = ord(chr) - ord("a")
                if (u >> id) & 1:
                    v += 1
                else:
                    c += 1
        return v // c if c else 0

    # 3814. 预算下的最大总容量 (Maximum Capacity Within Budget)
    def maxCapacity(self, costs: List[int], capacity: List[int], budget: int) -> int:
        res = 0
        # 仅一个机器
        for cost, cap in sorted(zip(costs, capacity)):
            if cost < budget:
                res = max(res, cap)
        if res == 0:
            return res
        # 两个cost相同的机器
        cnts = defaultdict(list)
        for cost, cap in sorted(zip(costs, capacity)):
            if cost >= budget:
                continue
            cnts[cost].append(cap)
        for cost, caps in cnts.items():
            caps.sort(reverse=True)
            if len(caps) >= 2 and cost * 2 < budget:
                res = max(res, caps[0] + caps[1])
        # 两个cost不同的机器
        a = []
        for cost, caps in cnts.items():
            a.append((cost, caps[0]))
        a.sort()
        pre_max = [0] * len(a)
        pre_max[0] = a[0][1]
        for i in range(1, len(a)):
            pre_max[i] = max(pre_max[i - 1], a[i][1])
        for i in range(1, len(a)):
            cost, cap = a[i]
            remain = budget - cost
            left, right = 0, i - 1
            while left <= right:
                mid = left + ((right - left) >> 1)
                if a[mid][0] < remain:
                    left = mid + 1
                else:
                    right = mid - 1
            if left - 1 >= 0:
                res = max(res, cap + pre_max[left - 1])
        return res

    # 3816. 删除重复字符后的字典序最小字符串 (Lexicographically Smallest String After Deleting Duplicate Characters)
    def lexSmallestAfterDeletion(self, s: str) -> str:
        cnts = Counter(s)
        st = []
        for chr in s:
            while st and st[-1] > chr and cnts[st[-1]] > 1:
                cnts[st.pop()] -= 1
            st.append(chr)
        while cnts[st[-1]] > 1:
            cnts[st.pop()] -= 1
        return "".join(st)

    # 3507. 移除最小数对使数组有序 I (Minimum Pair Removal to Sort Array I)
    def minimumPairRemoval(self, nums: List[int]) -> int:
        def check_order() -> bool:
            return all(y >= x for x, y in pairwise(nums))

        def remove_smallest_pair() -> List[int]:
            min_pair = inf
            min_id = -1
            for i in range(1, len(nums)):
                if nums[i] + nums[i - 1] < min_pair:
                    min_pair = nums[i] + nums[i - 1]
                    min_id = i
            return nums[: min_id - 1] + [min_pair] + nums[min_id + 1 :]

        res = 0
        while not check_order():
            res += 1
            nums = remove_smallest_pair()
        return res

    # 1877. 数组中最大数对和的最小值 (Minimize Maximum Pair Sum in Array)
    def minPairSum(self, nums: List[int]) -> int:
        nums.sort()
        return max(nums[i] + nums[-i - 1] for i in range(len(nums) >> 1))

    # 3818. 移除前缀使数组严格递增 (Minimum Prefix Removal to Make Array Strictly Increasing)
    def minimumPrefixLength(self, nums: List[int]) -> int:
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] >= nums[i + 1]:
                return i + 1
        return 0

    # 3819. 非负元素轮替 (Rotate Non Negative Elements)
    def rotateElements(self, nums: List[int], k: int) -> List[int]:
        if k == 0:
            return nums
        a = [x for x in nums if x >= 0]
        if not a:
            return nums
        k %= len(a)
        a = a[k:] + a[:k]
        i = 0
        for j, x in enumerate(nums):
            if x < 0:
                continue
            nums[j] = a[i]
            i += 1
        return nums

    # 3820. 树上的勾股距离节点 (Pythagorean Distance Nodes in a Tree)
    def specialNodes(
        self, n: int, edges: List[List[int]], x: int, y: int, z: int
    ) -> int:
        def cal(node: int) -> List[int]:
            def dfs(x: int, fa: int, d: int):
                dis[x] = d
                for y in g[x]:
                    if y != fa:
                        dfs(y, x, d + 1)

            dis = [0] * n
            dfs(node, -1, 0)
            return dis

        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        dx = cal(x)
        dy = cal(y)
        dz = cal(z)
        res = 0
        for t in zip(dx, dy, dz):
            a, b, c = sorted(t)
            if a * a + b * b == c * c:
                res += 1
        return res

    # 3821. 二进制中恰好K个1的第N小整数 (Find Nth Smallest Integer With K One Bits) --超时
    def nthSmallest(self, n: int, k: int) -> int:
        def check(x: int) -> int:
            @cache
            def dfs(i: int, cnt: int, is_limit: bool) -> int:
                if cnt == k:
                    return 1
                if cnt + (n - i) < k or i == n:
                    return 0
                res = 0
                up = int(s[i]) if is_limit else 1
                for d in range(up + 1):
                    if cnt + d > k:
                        break
                    res += dfs(i + 1, cnt + d, is_limit and d == up)
                return res

            s = bin(x)[2:]
            n = len(s)
            return dfs(0, 0, True)

        left = (1 << k) - 1
        right = 1 << 50
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid) >= n:
                right = mid - 1
            else:
                left = mid + 1
        return right + 1

    # 3821. 二进制中恰好K个1的第N小整数 (Find Nth Smallest Integer With K One Bits)
    def nthSmallest(self, n: int, k: int) -> int:
        res = 0
        for i in range(49, -1, -1):
            c = comb(i, k)
            if n > c:
                n -= c
                res |= 1 << i
                k -= 1
                if k == 0:
                    return res

    # 1200. 最小绝对差 (Minimum Absolute Difference)
    def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
        arr.sort()
        diff = inf
        res = []
        for x, y in pairwise(arr):
            if y - x < diff:
                diff = y - x
                res = [[x, y]]
            elif y - x == diff:
                res.append([x, y])
        return res

    # 156. 上下翻转二叉树 (Binary Tree Upside Down) --plus
    def upsideDownBinaryTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def dfs(node: Optional[TreeNode]) -> Optional[TreeNode]:
            if not node or not node.left:
                return node
            new_root = dfs(node.left)
            node.left.left = node.right
            node.left.right = node
            node.left = None
            node.right = None
            return new_root

        return dfs(root)

    # 255. 验证二叉搜索树的前序遍历序列 (Verify Preorder Sequence in Binary Search Tree) --plus
    def verifyPreorder(self, preorder: List[int]) -> bool:
        stack = []
        lower_bound = -inf
        for x in preorder:
            if x < lower_bound:
                return False
            while stack and x > stack[-1]:
                lower_bound = stack.pop()
            stack.append(x)
        return True

    # 291. 单词规律 II (Word Pattern II) --plus
    def wordPatternMatch(self, pattern: str, s: str) -> bool:
        def dfs(i: int, j: int) -> bool:
            if i == len(pattern) or j == len(s):
                return i == len(pattern) and j == len(s)
            if pattern[i] in d:
                v = d[pattern[i]]
                if not s[j:].startswith(v):
                    return False
                return dfs(i + 1, j + len(v))
            for end in range(j + 1, len(s) + 1):
                v = s[j:end]
                if v in _set:
                    continue
                d[pattern[i]] = v
                _set.add(v)
                if dfs(i + 1, end):
                    return True
                del d[pattern[i]]
                _set.remove(v)
            return False

        if len(pattern) > len(s):
            return False
        _set = set()
        d = defaultdict(str)
        return dfs(0, 0)

    # 302. 包含全部黑色像素的最小矩形 (Smallest Rectangle Enclosing Black Pixels) --plus
    def minArea(self, image: List[List[str]], x: int, y: int) -> int:
        left, right, top, bottom = y, y, x, x
        m, n = len(image), len(image[0])
        for i in range(m):
            for j in range(n):
                if image[i][j] == "1":
                    left = min(left, j)
                    right = max(right, j)
                    top = min(top, i)
                    bottom = max(bottom, i)
        return (right - left + 1) * (bottom - top + 1)

    # 302. 包含全部黑色像素的最小矩形 (Smallest Rectangle Enclosing Black Pixels) --plus
    def minArea(self, image: List[List[str]], x: int, y: int) -> int:
        def dfs(i: int, j: int):
            nonlocal left, right, top, bottom
            if i < 0 or i >= m or j < 0 or j >= n or image[i][j] == "0":
                return
            image[i][j] = "0"
            left = min(left, j)
            right = max(right, j)
            top = min(top, i)
            bottom = max(bottom, i)
            for dx, dy in directions:
                dfs(i + dx, j + dy)

        left, right, top, bottom = y, y, x, x
        m, n = len(image), len(image[0])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dfs(x, y)
        return (right - left + 1) * (bottom - top + 1)

    # 302. 包含全部黑色像素的最小矩形 (Smallest Rectangle Enclosing Black Pixels) --plus
    def minArea(self, image: List[List[str]], x: int, y: int) -> int:
        def bin_left(left: int, right: int) -> int:
            while left <= right:
                mid = left + ((right - left) >> 1)
                if any(image[i][mid] == "1" for i in range(m)):
                    right = mid - 1
                else:
                    left = mid + 1
            return right + 1

        def bin_right(left: int, right: int) -> int:
            while left <= right:
                mid = left + ((right - left) >> 1)
                if any(image[i][mid] == "1" for i in range(m)):
                    left = mid + 1
                else:
                    right = mid - 1
            return left - 1

        def bin_top(left: int, right: int) -> int:
            while left <= right:
                mid = left + ((right - left) >> 1)
                if any(image[mid][j] == "1" for j in range(n)):
                    right = mid - 1
                else:
                    left = mid + 1
            return right + 1

        def bin_bottom(left: int, right: int) -> int:
            while left <= right:
                mid = left + ((right - left) >> 1)
                if any(image[mid][j] == "1" for j in range(n)):
                    left = mid + 1
                else:
                    right = mid - 1
            return left - 1

        m, n = len(image), len(image[0])
        left = bin_left(0, y)
        right = bin_right(y, n - 1)
        top = bin_top(0, x)
        bottom = bin_bottom(x, m - 1)
        return (right - left + 1) * (bottom - top + 1)

    # 3491. 电话号码前缀 (Phone Number Prefix) --plus
    def phonePrefix(self, numbers: List[str]) -> bool:
        class trie:
            def __init__(self):
                self.children = [None] * 10

            def insert(self, s: str) -> bool:
                node = self
                is_prefix = True
                for chr in s:
                    id = ord(chr) - ord("0")
                    if not node.children[id]:
                        is_prefix = False
                        node.children[id] = trie()
                    node = node.children[id]
                return is_prefix

        numbers.sort(key=lambda k: -len(k))
        _trie = trie()
        return not any(_trie.insert(s) for s in numbers)

    # 356. 直线镜像 (Line Reflection)
    def isReflected(self, points: List[List[int]]) -> bool:
        def check(a: set) -> int:
            a = sorted(a)
            left, right = 0, len(a) - 1
            mid_x_2 = a[left] + a[right]
            while left < right:
                if a[left] + a[right] != mid_x_2:
                    return -inf
                left += 1
                right -= 1
            if left == right:
                return mid_x_2 if a[left] * 2 == mid_x_2 else inf
            return mid_x_2

        d = defaultdict(set)
        for x, y in points:
            d[y].add(x)
        # 中轴线x坐标*2 (防止出现小数)
        mid_x_2 = inf
        for a in d.values():
            cur = check(a)
            if cur == inf:
                return False
            if mid_x_2 != inf and mid_x_2 != cur:
                return False
            mid_x_2 = cur
        return True

    # 1644. 二叉树的最近公共祖先 II (Lowest Common Ancestor of a Binary Tree II) --plus
    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        def dfs(node: Optional[TreeNode]) -> Optional[TreeNode]:
            def check(node: Optional[TreeNode]) -> bool:
                if node is None:
                    return False
                if node in (p, q):
                    return True
                return check(node.left) or check(node.right)

            if node is None:
                return False
            if node in (p, q):
                if check(node.left) or check(node.right):
                    nonlocal res
                    res = node
                    return node
                return node
            left = dfs(node.left)
            right = dfs(node.right)
            if res:
                return res
            if left and right:
                res = node
                return node
            return left or right

        res = None
        dfs(root)
        return res

    # 510. 二叉搜索树中的中序后继 II (Inorder Successor in BST II) --plus
    def inorderSuccessor(self, node: "Node") -> "Optional[Node]":
        if node.right:
            node = node.right
            while node.left:
                node = node.left
            return node
        while node.parent:
            if node.parent.left == node:
                break
            node = node.parent
        return node.parent

    # 545. 二叉树的边界 (Boundary of Binary Tree) --plus
    def boundaryOfBinaryTree(self, root: Optional[TreeNode]) -> List[int]:
        def cal(node: Optional[TreeNode], is_left: bool):
            res = []
            while node:
                # node 是叶子
                if node.left is None and node.right is None:
                    break
                res.append(node.val)
                node = (
                    node.left
                    if is_left and node.left or not is_left and node.right is None
                    else node.right
                )
            return res

        def dfs(root: Optional[TreeNode], node: Optional[TreeNode]):
            if node is None:
                return
            if node.left is None and node.right is None and node != root:
                leaves.append(node.val)
                return
            dfs(root, node.left)
            dfs(root, node.right)

        # 左边界
        left = cal(root.left, True)
        # 右边界
        right = cal(root.right, False)
        # 叶子
        leaves = []
        dfs(root, root)
        return [root.val] + left + leaves + right[::-1]

    # 247. 中心对称数 II (Strobogrammatic Number II) --plus
    def findStrobogrammatic(self, n: int) -> List[str]:
        def dfs():
            nonlocal path
            if len(path) == (n + 1) // 2:
                cur = path
                for j in range(n // 2 - 1, -1, -1):
                    cur += d[path[j]]
                res.append(cur)
                return
            for chr in d.keys():
                # n不是一位数时，首位不能为0
                if len(path) == 0 and chr == "0" and n != 1:
                    continue
                # n 为奇数时，中间位置不能是 6 或 9，只能是 0、1、8
                if n % 2 == 1 and len(path) == n // 2 and chr in ["6", "9"]:
                    continue
                path += chr
                dfs()
                path = path[:-1]

        d = {"0": "0", "1": "1", "6": "9", "8": "8", "9": "6"}
        res = []
        path = ""
        dfs()
        return res

    # 246. 中心对称数 (Strobogrammatic Number) --plus
    def isStrobogrammatic(self, num: str) -> bool:
        d = {"0": "0", "1": "1", "6": "9", "8": "8", "9": "6"}
        left = 0
        right = len(num) - 1
        while left < right:
            if num[left] not in d or d[num[left]] != num[right]:
                return False
            left += 1
            right -= 1
        return left != right or num[left] in ["0", "1", "8"]

    # 800. 相似 RGB 颜色 (Similar RGB Color) --plus
    def similarRGB(self, color: str) -> str:
        def cal(s: str) -> str:
            val = int(s, 16)
            first = int(s[0], 16)
            diff = inf
            res = -1
            for i in range(max(0, first - 1), min(15, first + 1) + 1):
                cur_val = i * 16 + i
                if abs(val - cur_val) < diff:
                    diff = abs(val - cur_val)
                    res = cur_val
            return format(res, "02x")

        res = "#"
        for i in range(1, len(color) - 1, 2):
            res += cal(color[i : i + 2])
        return res

    # 3817. 数字字符串中的好索引 (Good Indices in a Digit String) --plus
    def goodIndices(self, s: str) -> List[int]:
        res = []
        for i, x in enumerate(s):
            # 最后一位不相等
            if int(x) != i % 10:
                continue
            l = len(str(i))
            if int(s[max(0, i - l + 1) : i + 1]) == i:
                res.append(i)
        return res

    # 3807. 修复边以遍历图的最小成本 (Minimum Cost to Repair Edges to Traverse a Graph) --plus
    def minCost(self, n: int, edges: List[List[int]], k: int) -> int:
        def check(limit: int) -> bool:
            vis = [False] * n
            vis[0] = True
            q = deque()
            q.append((0, 0))
            while q:
                x, d = q.popleft()
                if x == n - 1:
                    return True
                for y, w in g[x]:
                    if not vis[y] and w <= limit and d + 1 <= k:
                        vis[y] = True
                        q.append((y, d + 1))
            return False

        g = [[] for _ in range(n)]
        for u, v, w in edges:
            g[u].append((v, w))
            g[v].append((u, w))
        if not check(inf):
            return -1
        left = 0
        right = max(w for _, _, w in edges)
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return right + 1

    # 1794. 统计距离最小的子串对个数 (Count Pairs of Equal Substrings With Minimum Difference) --plus
    def countQuadruples(self, firstString: str, secondString: str) -> int:
        left = [-1] * 26
        for i, x in enumerate(firstString):
            id = ord(x) - ord("a")
            if left[id] == -1:
                left[id] = i
        right = [-1] * 26
        for i in range(len(secondString) - 1, -1, -1):
            x = secondString[i]
            id = ord(x) - ord("a")
            if right[id] == -1:
                right[id] = i
        res = 0
        min_dis = inf
        for i in range(26):
            if left[i] != -1 and right[i] != -1:
                dis = left[i] - right[i]
                min_dis = min(min_dis, dis)
        if min_dis == inf:
            return 0
        for i, x in enumerate(firstString):
            id = ord(x) - ord("a")
            if right[id] != -1 and i - right[id] == min_dis:
                res += 1
        return res

    # 1788. 最大化花园的美观度 (Maximize the Beauty of the Garden) --plus
    def maximumBeauty(self, flowers: List[int]) -> int:
        n = len(flowers)
        left = defaultdict(int)
        right = defaultdict(int)
        pre = [0] * (n + 1)
        for i, x in enumerate(flowers):
            pre[i + 1] = pre[i] + max(0, x)
            if x not in left:
                left[x] = i
            right[x] = i
        res = -inf
        for i, x in enumerate(flowers):
            l = left[x]
            r = right[x]
            if l == r:
                continue
            total = pre[r + 1] - pre[l] + min(0, x) * 2
            res = max(res, total)
        return res

    # 3792. 递增乘积块之和 (Sum of Increasing Product Blocks) --plus
    def sumOfBlocks(self, n: int) -> int:
        # 此段代码写在class之外可通过
        # 预处理，计算每个block的值
        mod = 10**9 + 7

        def getBlockRes(n: int) -> int:
            start = (1 + n - 1) * (n - 1) // 2 + 1
            end = start + n - 1
            res = 1
            for num in range(start, end + 1):
                res *= num
                res %= mod
            return res

        # 预处理，前缀和
        mx = 1000
        pre = [0]
        for x in range(1, mx + 1):
            pre.append((pre[-1] + getBlockRes(x)) % mod)
        return pre[n]

    # 3787. 查找树的直径端点 (Find Diameter Endpoints of a Tree) --plus
    # 找到树中一条直径的方法：对任意一个点 DFS 找到最远点，然后对这个点进行第二次相同的 DFS 找另一个最远点，得到的路径就是直径
    # 找树中所有直径端点的方法：稍作修改，对任意一个点 DFS 找到的所有最远点加入集合；从其中一个最远点进行第二次 DFS 找到的所有最远点加入集合。可以证明，两个集合的并集即为答案。
    def findSpecialNodes(self, n: int, edges: List[List[int]]) -> str:
        def dfs(x: int, fa: int, d: int, s: set):
            nonlocal max_dis, node
            if d > max_dis:
                max_dis = d
                s.clear()
                s.add(x)
                node = x
            elif d == max_dis:
                s.add(x)
            for y in g[x]:
                if y != fa:
                    dfs(y, x, d + 1, s)

        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        s = set()
        max_dis = 0
        node = -1
        dfs(0, -1, 0, s)
        max_dis = 0
        s2 = set()
        dfs(node, -1, 0, s2)
        s |= s2
        return "".join("1" if i in s else "0" for i in range(n))

    # 2036. 最大交替子数组和 (Maximum Alternating Subarray Sum) --plus
    def maximumAlternatingSubarraySum(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            return max(0, dfs(i + 1, -j) + nums[i] * j)

        n = len(nums)
        return max(dfs(i + 1, -1) + x for i, x in enumerate(nums))

    # 3822. 设计订单管理系统 (Design Order Management System) --plus
    class OrderManagementSystem:

        def __init__(self):
            self.id_type_price = defaultdict(tuple)
            self.type_price_orders = defaultdict(lambda: defaultdict(set))

        def addOrder(self, orderId: int, orderType: str, price: int) -> None:
            self.id_type_price[orderId] = (orderType, price)
            self.type_price_orders[orderType][price].add(orderId)

        def modifyOrder(self, orderId: int, newPrice: int) -> None:
            orderType, oldPrice = self.id_type_price[orderId]
            self.type_price_orders[orderType][oldPrice].remove(orderId)
            self.type_price_orders[orderType][newPrice].add(orderId)
            self.id_type_price[orderId] = (orderType, newPrice)

        def cancelOrder(self, orderId: int) -> None:
            orderType, price = self.id_type_price[orderId]
            del self.id_type_price[orderId]
            self.type_price_orders[orderType][price].remove(orderId)

        def getOrdersAtPrice(self, orderType: str, price: int) -> List[int]:
            return [id for id in self.type_price_orders[orderType][price]]

    # 3167. 字符串的更好压缩 (Better Compression of String) --plus
    def betterCompression(self, compressed: str) -> str:
        cnt = [0] * 26
        i = 0
        n = len(compressed)
        while i < n:
            j = i + 1
            while j < n and compressed[j].isdigit():
                j += 1
            count = int(compressed[i + 1 : j])
            id = ord(compressed[i]) - ord("a")
            cnt[id] += count
            i = j
        res = []
        for i in range(26):
            if cnt[i] > 0:
                res.append(chr(i + ord("a")) + str(cnt[i]))
        return "".join(res)

    # 3173. 相邻元素的按位或 (Bitwise OR of Adjacent Elements) --plus
    def orArray(self, nums: List[int]) -> List[int]:
        return [x | y for x, y in pairwise(nums)]

    # 3062. 链表游戏的获胜者 (Winner of the Linked List Game) --plus
    def gameResult(self, head: Optional[ListNode]) -> str:
        cnt = [0] * 2
        while head:
            even, odd = head.val, head.next.val
            if even != odd:
                cnt[odd > even] += 1
            head = head.next.next
        if cnt[0] > cnt[1]:
            return "Even"
        if cnt[0] < cnt[1]:
            return "Odd"
        return "Tie"

    # 2743. 计算没有重复字符的子字符串数量 (Count Substrings Without Repeating Character) --plus
    def numberOfSpecialSubstrings(self, s: str) -> int:
        cnts = [0] * 26
        j = 0
        res = 0
        for i, x in enumerate(s):
            id = ord(x) - ord("a")
            cnts[id] += 1
            while cnts[id] > 1:
                id_j = ord(s[j]) - ord("a")
                cnts[id_j] -= 1
                j += 1
            res += i - j + 1
        return res

    # 3773. 最大等长连续字符组 (Maximum Number of Equal Length Runs) --plus
    def maxSameLengthRuns(self, s: str) -> int:
        d = defaultdict(int)
        cnt = 0
        res = 0
        for i, c in enumerate(s):
            cnt += 1
            if i == len(s) - 1 or s[i + 1] != c:
                d[cnt] += 1
                res = max(res, d[cnt])
                cnt = 0
        return res

    # 3063. 链表频率 (Linked List Frequency) --plus
    def frequenciesOfElements(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cnts = defaultdict(int)
        while head:
            cnts[head.val] += 1
            head = head.next
        dummy = ListNode(0)
        cur = dummy
        for c in cnts.values():
            cur.next = ListNode(c)
            cur = cur.next
        return dummy.next

    # 3247. 奇数和子序列的数量 (Number of Subsequences with Odd Sum) --plus
    def subsequenceCount(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i < 0:
                return j
            return (dfs(i - 1, j) + dfs(i - 1, (j + nums[i]) & 1)) % MOD

        MOD = 10**9 + 7
        return dfs(len(nums) - 1, 0)

    # 3247. 奇数和子序列的数量 (Number of Subsequences with Odd Sum) --plus
    def subsequenceCount(self, nums: List[int]) -> int:
        cnts = [0] * 2
        for x in nums:
            cnts[x & 1] += 1
        MOD = 10**9 + 7
        return (pow(2, cnts[1] - 1, MOD) * pow(2, cnts[0], MOD)) % MOD if cnts[1] else 0

    # 3763. 带阈值约束的最大总和 (Maximum Total Sum with Threshold Constraints) --plus
    def maxSum(self, nums: List[int], threshold: List[int]) -> int:
        a = [(t, x) for t, x in zip(threshold, nums)]
        a.sort()
        res = 0
        for i, (t, x) in enumerate(a):
            if t > i + 1:
                break
            res += x
        return res

    # 3758. 将数字词转换为数字 (Convert Number Words to Digits) --plus
    def convertNumber(self, s: str) -> str:
        d = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
        }
        res = []
        i = 0
        n = len(s)
        while i < n:
            for w in d.keys():
                m = len(w)
                if s[i : i + m] == w:
                    res.append(d[w])
                    i += m
                    break
            else:
                i += 1
        return "".join(res)

    # 3119. 最大数量的可修复坑洼 (Maximum Number of Potholes That Can Be Fixed) --plus
    def maxPotholes(self, road: str, budget: int) -> int:
        cnts = defaultdict(int)
        i = 0
        n = len(road)
        while i < n:
            if road[i] == "x":
                j = i
                while j < n and road[j] == "x":
                    j += 1
                cnts[j - i] += 1
                i = j
            else:
                i += 1
        a = [(k, v) for k, v in cnts.items()]
        a.sort(key=lambda x: -x[0])
        res = 0
        for l, cnt in a:
            complete_cnt = budget // (l + 1)
            res += l * min(complete_cnt, cnt)
            budget -= (l + 1) * min(complete_cnt, cnt)
            cnt -= min(complete_cnt, cnt)
            if budget > 0 and cnt > 0:
                res += budget - 1
                break
        return res

    # 3183. 达到总和的方法数量 (The Number of Ways to Make the Sum) --plus
    def numberOfWays(self, n: int) -> int:
        f = [0] * (n + 1)
        f[0] = 1
        MOD = 10**9 + 7
        for c in (1, 2, 6):
            for i in range(c, n + 1):
                f[i] += f[i - c]
                f[i] %= MOD
        res = f[n]
        if n >= 4:
            res += f[n - 4]
        if n >= 8:
            res += f[n - 8]
        return res % MOD

    # 3778. 排除一个最大权重边的最小距离 (Minimum Distance Excluding One Maximum Weighted Edge) --plus
    def minCostExcludingMax(self, n: int, edges: List[List[int]]) -> int:
        g = [[] for _ in range(n)]
        for u, v, w in edges:
            g[u].append((v, w))
            g[v].append((u, w))
        dis = [[inf] * 2 for _ in range(n)]
        dis[0][0] = 0
        q = []
        # (d, is_deleted, node)
        q.append((0, 0, 0))
        heapq.heapify(q)
        while q:
            d, is_deleted, x = heapq.heappop(q)
            if d > dis[x][is_deleted]:
                continue
            for y, dx in g[x]:
                nd = d + dx
                if dis[y][is_deleted] > nd:
                    dis[y][is_deleted] = nd
                    heapq.heappush(q, (nd, is_deleted, y))
                if not is_deleted and dis[y][1] > d:
                    dis[y][1] = d
                    heapq.heappush(q, (d, 1, y))
        return dis[n - 1][1]

    # 2664. 巡逻的骑士 (The Knight’s Tour) --plus
    def tourOfKnight(self, m: int, n: int, r: int, c: int) -> List[List[int]]:
        def dfs(x: int, y: int, step: int) -> bool:
            res[x][y] = step
            if step == m * n - 1:
                return True
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < m
                    and 0 <= ny < n
                    and res[nx][ny] == -1
                    and dfs(nx, ny, step + 1)
                ):
                    return True
            res[x][y] = -1
            return False

        dirs = (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)
        res = [[-1] * n for _ in range(m)]
        dfs(r, c, 0)
        return res

    # 2852. 所有单元格的远离程度之和 (Sum of Remoteness of All Cells) --plus
    def sumRemoteness(self, grid: List[List[int]]) -> int:
        class union:
            def __init__(self, n: int):
                self.rank = [1] * n
                self.parent = [i for i in range(n)]

            def get_root(self, p: int) -> int:
                if self.parent[p] == p:
                    return p
                self.parent[p] = self.get_root(self.parent[p])
                return self.parent[p]

            def is_connected(self, p1: int, p2: int) -> bool:
                return self.get_root(p1) == self.get_root(p2)

            def union(self, p1: int, p2: int) -> None:
                r1 = self.get_root(p1)
                r2 = self.get_root(p2)
                if r1 == r2:
                    return
                if self.rank[r1] < self.rank[r2]:
                    self.parent[r1] = r2
                else:
                    self.parent[r2] = r1
                    if self.rank[r1] == self.rank[r2]:
                        self.rank[r1] += 1

        m, n = len(grid), len(grid[0])
        total = 0
        u = union(m * n)
        for i in range(m):
            for j in range(n):
                total += max(0, grid[i][j])
                if i - 1 >= 0 and grid[i][j] > 0 and grid[i - 1][j] > 0:
                    u.union(i * n + j, (i - 1) * n + j)
                if j - 1 >= 0 and grid[i][j] > 0 and grid[i][j - 1] > 0:
                    u.union(i * n + j, i * n + j - 1)
        group_sum = defaultdict(int)
        group_cnt = defaultdict(int)
        for i in range(m):
            for j in range(n):
                if grid[i][j] > 0:
                    root = u.get_root(i * n + j)
                    group_sum[root] += grid[i][j]
                    group_cnt[root] += 1
        res = 0
        for root, cnt in group_cnt.items():
            s = group_sum[root]
            res += (total - s) * cnt
        return res

    # 2847. 给定数字乘积的最小数字 (Smallest Number With Given Digit Product) --plus
    def smallestNumber(self, n: int) -> str:
        if n == 1:
            return "1"
        cnt = [0] * 10
        for i in range(9, 1, -1):
            while n % i == 0:
                n //= i
                cnt[i] += 1
        if n != 1:
            return "-1"
        res = [str(i) * c for i, c in enumerate(cnt[2:], 2)]
        return "".join(res)

    # 3481. 应用替换 (Apply Substitutions) --plus
    def applySubstitutions(self, replacements: List[List[str]], text: str) -> str:
        def dfs(s: str) -> str:
            res = ""
            i = 0
            n = len(s)
            while i < n:
                if s[i] == "%":
                    k = ord(s[i + 1]) - ord("A")
                    res += dfs(d[k])
                    i += 3
                else:
                    res += s[i]
                    i += 1
            return res

        d = [""] * 26
        for k, v in replacements:
            d[ord(k) - ord("A")] = v
        return dfs(text)

    # 3696. 不同单词间的最大距离 I (Maximum Distance Between Unequal Words in Array I) --plus
    # 3706. 不同单词间的最大距离 II (Maximum Distance Between Unequal Words in Array II) --plus
    def maxDistance(self, words: List[str]) -> int:
        d = defaultdict(list)
        for i, w in enumerate(words):
            if w not in d:
                d[w] = [i, i]
            else:
                d[w][0] = min(d[w][0], i)
                d[w][1] = max(d[w][1], i)
        res = 0
        min_id, max_id = inf, -inf
        for cur_min_id, cur_max_id in d.values():
            if min_id != inf:
                res = max(
                    res, abs(cur_min_id - max_id) + 1, abs(min_id - cur_max_id) + 1
                )
            min_id = min(min_id, cur_min_id)
            max_id = max(max_id, cur_max_id)
        return res

    # 3730. 跳跃燃烧的最大卡路里 (Maximum Calories Burnt from Jumps) --plus
    def maxCaloriesBurnt(self, heights: List[int]) -> int:
        a = [0] + sorted(heights)
        res = 0
        i = 0
        j = len(a) - 1
        flip = True
        while i < j:
            res += (a[j] - a[i]) * (a[j] - a[i])
            if flip:
                i += 1
            else:
                j -= 1
            flip = not flip
        return res

    # 3662. 按频率筛选字符 (Filter Characters by Frequency) --plus
    def filterCharacters(self, s: str, k: int) -> str:
        c = Counter(s)
        return "".join([x for x in s if c[x] < k])

    # 2184. 建造坚实的砖墙的方法数 (Number of Ways to Build Sturdy Brick Wall) --plus
    def buildWall(self, height: int, width: int, bricks: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == height:
                return 1
            return sum(dfs(i + 1, x) for x in g[j]) % MOD

        def back_trace(i: int):
            if i == width:
                a.append(path.copy())
                return
            for b in bricks:
                if i + b <= width:
                    path.append(b)
                    back_trace(i + b)
                    path.pop()

        if min(bricks) > width:
            return 0
        a = []
        path = []
        back_trace(0)
        _list = []
        for l in a:
            mask = 0
            pre_s = 0
            for x in l:
                pre_s += x
                mask |= 1 << pre_s
            _list.append(mask ^ (1 << width))
        g = defaultdict(list)
        for x, y in combinations(_list, 2):
            if x & y == 0:
                g[x].append(y)
                g[y].append(x)
        g[0] = _list
        MOD = 10**9 + 7
        return dfs(0, 0)

    # 3125. 使得按位与结果为 0 的最大数字 (Maximum Number That Makes Result of Bitwise AND Zero) --plus
    def maxNumber(self, n: int) -> int:
        return (1 << (n.bit_length() - 1)) - 1

    # 3744. 在展开字符串中查找第 K 个字符 (Find Kth Character in Expanded String) --plus
    def kthCharacter(self, s: str, k: int) -> str:
        mul = 1
        for x in s:
            if x != " ":
                k -= mul
                mul += 1
            else:
                mul = 1
                k -= mul
            if k < 0:
                return x

    # 744. 寻找比目标字母大的最小字母 (Find Smallest Letter Greater Than Target) --plus
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        left = 0
        right = len(letters) - 1
        while left <= right:
            mid = left + ((right - left) >> 1)
            if letters[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return letters[(right + 1) % len(letters)]

    # 2907. 价格递增的最大利润三元组 I (Maximum Profitable Triplets With Increasing Prices I) --plus
    def maxProfit(self, prices: List[int], profits: List[int]) -> int:
        n = len(prices)
        left = [0] * n
        right = [0] * n
        for i in range(n):
            for j in range(i):
                if prices[j] < prices[i]:
                    left[i] = max(left[i], profits[j])
            for j in range(i + 1, n):
                if prices[j] > prices[i]:
                    right[i] = max(right[i], profits[j])
        res = -1
        for i in range(1, n - 1):
            if left[i] and right[i]:
                res = max(res, left[i] + profits[i] + right[i])
        return res

    # 1088. 易混淆数 II (Confusing Number II) --plus
    def confusingNumberII(self, n: int) -> int:
        def rev(x: int) -> bool:
            r = 0
            while x:
                r = r * 10 + d[x % 10]
                x //= 10
            return r

        def dfs(i: int, j: int) -> int:
            if i == l:
                return j and j <= n and rev(j) != j
            res = 0
            if not j:
                res = dfs(i + 1, j)
            for x in a:
                if not j and x == 0:
                    continue
                res += dfs(i + 1, j * 10 + x)
            return res

        a = [0, 1, 6, 8, 9]
        d = [0] * 10
        d[0] = 0
        d[1] = 1
        d[6] = 9
        d[9] = 6
        d[8] = 8

        l = len(str(n))
        return dfs(0, 0)

    # 1063. 有效子数组的数目 (Number of Valid Subarrays) --plus
    def validSubarrays(self, nums: List[int]) -> int:
        res = 0
        st = []
        for i, x in enumerate(nums):
            while st and nums[st[-1]] > x:
                res += i - st.pop()
            st.append(i)
        while st:
            res += len(nums) - st.pop()
        return res

    # 2728. 计算一个环形街道上的房屋数量 (Count Houses in a Circular Street) --plus
    def houseCount(self, street: Optional["Street"], k: int) -> int:
        for _ in range(k):
            street.closeDoor()
            street.moveRight()
        res = 0
        while not street.isDoorOpen():
            res += 1
            street.openDoor()
            street.moveRight()
        return res

    # 2714. 找到 K 次跨越的最短路径 (Find Shortest Path with K Hops) --plus
    def shortestPathWithHops(
        self, n: int, edges: List[List[int]], s: int, e: int, k: int
    ) -> int:
        g = [[] for _ in range(n)]
        for u, v, w in edges:
            g[u].append((v, w))
            g[v].append((u, w))
        dis = [[inf] * (k + 1) for _ in range(n)]
        dis[s][0] = 0
        q = []
        # (d, used_k, node)
        q.append((0, 0, s))
        heapq.heapify(q)
        res = inf
        while q:
            d, used_k, x = heapq.heappop(q)
            if d > dis[x][used_k]:
                continue
            if x == e:
                res = min(res, d)
                if res == 0:
                    break
                continue
            for y, dx in g[x]:
                nd = d + dx
                if dis[y][used_k] > nd:
                    dis[y][used_k] = nd
                    heapq.heappush(q, (nd, used_k, y))
                if used_k < k and dis[y][used_k + 1] > d:
                    dis[y][used_k + 1] = d
                    heapq.heappush(q, (d, used_k + 1, y))
        return res

    # 536. 从字符串生成二叉树 (Construct Binary Tree from String) --plus
    def str2tree(self, s: str) -> Optional[TreeNode]:
        if len(s) == 0:
            return None
        sign = 1
        x = 0
        i = 0
        while i < len(s):
            if s[i] == "(":
                break
            if s[i] == "-":
                sign = -1
            else:
                x = x * 10 + int(s[i])
            i += 1
        node = TreeNode(x * sign)
        if i == len(s):
            return node
        start = i
        end = -1
        cnt = 0
        while i < len(s):
            if s[i] == "(":
                cnt += 1
            elif s[i] == ")":
                cnt -= 1
            if cnt == 0:
                end = i
                break
            i += 1
        node.left = self.str2tree(s[start + 1 : end])
        if i < len(s) - 1:
            node.right = self.str2tree(s[end + 2 : len(s) - 1])
        return node

    # 3294. 将双链表转换为数组 II (Convert Doubly Linked List to Array II) --plus
    def toArray(self, node: "Optional[Node]") -> List[int]:
        while node.prev:
            node = node.prev
        res = []
        while node:
            res.append(node.val)
            node = node.next
        return res

    # 3400. 右移后的最大匹配索引数 (Maximum Number of Matching Indices After Right Shifts) --plus
    def maximumMatchingIndices(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        res = 0
        for i in range(n):
            cnt = 0
            for j in range(i, i + n):
                if nums1[j % n] == nums2[j - i]:
                    cnt += 1
            res = max(res, cnt)
        return res

    # 3450. 一张长椅上的最多学生 (Maximum Students on a Single Bench) --plus
    def maxStudentsOnBench(self, students: List[List[int]]) -> int:
        cnts = defaultdict(set)
        for x, y in students:
            cnts[y].add(x)
        res = 0
        for s in cnts.values():
            res = max(res, len(s))
        return res

    # 3460. 最多删除一次后的最长公共前缀 (Longest Common Prefix After at Most One Removal) --plus
    def longestCommonPrefix(self, s: str, t: str) -> int:
        deleted = False
        res = 0
        i = 0
        j = 0
        while i < len(s) and j < len(t):
            if s[i] != t[j]:
                if deleted:
                    break
                deleted = True
                i += 1
                continue
            res += 1
            i += 1
            j += 1
        return res

    # 3616. 学生替换人数 (Number of Student Replacements) --plus
    def totalReplacements(self, ranks: List[int]) -> int:
        _min = ranks[0]
        res = 0
        for x in ranks:
            if x < _min:
                res += 1
                _min = x
        return res

    # 3032. 统计各位数字都不同的数字个数 II (Count Numbers With Unique Digits II) --plus
    def numberCount(self, a: int, b: int) -> int:
        def cal(x: int) -> int:
            @cache
            def dfs(i: int, j: int, is_limit: bool, is_num: bool) -> int:
                if i == n:
                    return is_num
                res = 0
                if not is_num:
                    res = dfs(i + 1, j, False, False)
                up = int(s[i]) if is_limit else 9
                for d in range(0 if is_num else 1, up + 1):
                    if j >> d & 1 == 0:
                        res += dfs(i + 1, j | (1 << d), is_limit and d == up, True)
                return res

            s = str(x)
            n = len(s)
            return dfs(0, 0, True, False)

        return cal(b) - cal(a - 1)

    # 2814. 避免淹死并到达目的地的最短时间 (Minimum Time Takes to Reach Destination Without Drowning) --plus
    def minimumSeconds(self, land: List[List[str]]) -> int:
        m, n = len(land), len(land[0])
        q_flood = deque()
        vis_flood = [[False] * n for _ in range(m)]
        s, d = (), ()
        for i in range(m):
            for j in range(n):
                x = land[i][j]
                if x == "S":
                    s = (i, j)
                elif x == "D":
                    d = (i, j)
                elif x == "*":
                    q_flood.append((i, j))
                    vis_flood[i][j] = True
        vis = [[False] * n for _ in range(m)]
        vis[s[0]][s[1]] = True
        q = deque()
        q.append((s[0], s[1], 0))
        while q:
            size = len(q_flood)
            for _ in range(size):
                x, y = q_flood.popleft()
                for dx, dy in (0, 1), (0, -1), (1, 0), (-1, 0):
                    nx = x + dx
                    ny = y + dy
                    if (
                        0 <= nx < m
                        and 0 <= ny < n
                        and not vis_flood[nx][ny]
                        and land[nx][ny] != "X"
                        and land[nx][ny] != "D"
                    ):
                        vis_flood[nx][ny] = True
                        q_flood.append((nx, ny))

            size = len(q)
            for _ in range(size):
                x, y, step = q.popleft()
                if x == d[0] and y == d[1]:
                    return step
                for dx, dy in (0, 1), (0, -1), (1, 0), (-1, 0):
                    nx = x + dx
                    ny = y + dy
                    if (
                        0 <= nx < m
                        and 0 <= ny < n
                        and not vis_flood[nx][ny]
                        and not vis[nx][ny]
                        and land[nx][ny] != "X"
                    ):
                        vis[nx][ny] = True
                        q.append((nx, ny, step + 1))
        return -1

    # 1746. 经过一次操作后的最大子数组和 (Maximum Subarray Sum After One Operation) --plus
    def maxSumAfterOperation(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, j: bool) -> int:
            if i < 0:
                return 0 if j else -inf
            res = dfs(i - 1, j) + nums[i]
            if not j:
                res = max(res, dfs(i - 1, True) + nums[i] * nums[i])
            return max(res, 0)

        return max(
            max(dfs(i - 1, False) + x, dfs(i - 1, True) + x * x)
            for i, x in enumerate(nums)
        )

    # 3631. 按严重性和可利用性排序威胁 (Sort Threats by Severity and Exploitability) --plus
    def sortThreats(self, threats: List[List[int]]) -> List[List[int]]:
        threats.sort(key=lambda k: (-k[1] * 2 - k[2], k[0]))
        return threats

    # 3682. 公共元素的最小索引和 (Minimum Index Sum of Common Elements) --plus
    def minimumSum(self, nums1: List[int], nums2: List[int]) -> int:
        d = defaultdict(int)
        for i, x in enumerate(nums1):
            if x not in d:
                d[x] = i
        res = inf
        for i, x in enumerate(nums2):
            res = min(res, i + d.get(x, inf))
        return res if res < inf else -1

    # 3647. 两个袋子中的最大重量 (Maximum Weight in Two Bags) --plus
    def maxWeight(self, weights: List[int], w1: int, w2: int) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i < 0:
                return 0
            w = weights[i]
            res = dfs(i - 1, j, k)
            if w <= j:
                res = max(res, dfs(i - 1, j - w, k) + w)
            if w <= k:
                res = max(res, dfs(i - 1, j, k - w) + w)
            return res

        res = dfs(len(weights) - 1, w1, w2)
        dfs.cache_clear()
        return res

    # 3581. 计算数字中的奇数字母数量 (Count Odd Letters from Number) --plus
    def countOddLetters(self, n: int) -> int:
        map = ["zero", "one", "two", "thr", "four", "five", "six", "svn", "eight", "ie"]
        mask_map = [0] * 10
        for i, s in enumerate(map):
            for c in s:
                mask_map[i] ^= 1 << (ord(c) - ord("a"))
        mask_cnt = 0
        while n:
            mask_cnt ^= 1 << (n % 10)
            n //= 10
        mask_res = 0
        while mask_cnt:
            lb = (mask_cnt & -mask_cnt).bit_length() - 1
            mask_res ^= mask_map[lb]
            mask_cnt &= mask_cnt - 1
        return mask_res.bit_count()

    # 3824. 减小数组使其满足条件的最小 K 值 (Minimum K to Reduce Array Within Limit)
    def minimumK(self, nums: List[int]) -> int:
        def check(k: int) -> int:
            return sum((x + k - 1) // k for x in nums)

        left = 1
        right = 10**4
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid) <= mid**2:
                right = mid - 1
            else:
                left = mid + 1
        return right + 1

    # 3827. 统计单比特整数 (Count Monobit Integers)
    def countMonobit(self, n: int) -> int:
        return (n + 1).bit_length()

    # 3828. 删除子数组后的最终元素 (Final Element After Subarray Deletions)
    def finalElement(self, nums: List[int]) -> int:
        return max(nums[0], nums[-1])

    # 3823. 反转一个字符串里的字母后反转特殊字符 (Reverse Letters Then Special Characters in a String)
    def reverseByType(self, s: str) -> str:
        a = [x for x in s]
        n = len(a)
        i = 0
        j = n - 1
        while i < j:
            while i < j and not a[i].islower():
                i += 1
            while i < j and not a[j].islower():
                j -= 1
            a[i], a[j] = a[j], a[i]
            i += 1
            j -= 1

        i = 0
        j = n - 1
        while i < j:
            while i < j and a[i].islower():
                i += 1
            while i < j and a[j].islower():
                j -= 1
            a[i], a[j] = a[j], a[i]
            i += 1
            j -= 1
        return "".join(a)

    # 3825. 按位与结果非零的最长上升子序列 (Longest Strictly Increasing Subsequence With Non-Zero Bitwise AND)
    def longestSubsequence(self, nums: List[int]) -> int:
        w = max(nums).bit_length()
        res = 0
        for i in range(w):
            g = []
            for x in nums:
                if x >> i & 1 == 0:
                    continue
                j = bisect.bisect_left(g, x)
                if j == len(g):
                    g.append(x)
                else:
                    g[j] = x
            res = max(res, len(g))
        return res

    # 3830. 移除至多一个元素后的最长交替子数组 (Longest Alternating Subarray After Removing At Most One Element)
    def longestAlternating(self, nums: List[int]) -> int:
        n = len(nums)
        res = 1
        # left[i][0] 以i结尾 最后一组是下降的最长子数组长度
        # left[i][1] 以i结尾 最后一组是上升的最长子数组长度
        left = [[0] * 2 for _ in range(n)]
        left[0][0] = left[0][1] = 1
        for i in range(1, n):
            if nums[i] == nums[i - 1]:
                left[i][0] = left[i][1] = 1
            elif nums[i] > nums[i - 1]:
                left[i][1] = left[i - 1][0] + 1
                left[i][0] = 1
            else:
                left[i][0] = left[i - 1][1] + 1
                left[i][1] = 1
            res = max(res, left[i][0], left[i][1])

        # right[i][0] 以i开始 第一组是下降的最长子数组长度
        # right[i][1] 以i开始 第一组是上升的最长子数组长度
        right = [[0] * 2 for _ in range(n)]
        right[n - 1][0] = right[n - 1][1] = 1
        for i in range(n - 2, -1, -1):
            if nums[i] == nums[i + 1]:
                right[i][0] = right[i][1] = 1
            elif nums[i] > nums[i + 1]:
                right[i][0] = right[i + 1][1] + 1
                right[i][1] = 1
            else:
                right[i][1] = right[i + 1][0] + 1
                right[i][0] = 1
            res = max(res, right[i][0], right[i][1])

        for i in range(1, n - 1):
            # 不移除nums[i]
            if nums[i] > nums[i + 1]:
                res = max(res, left[i][1] + right[i + 1][1])
            elif nums[i] < nums[i + 1]:
                res = max(res, left[i][0] + right[i + 1][0])

            # 移除nums[i]
            if nums[i - 1] > nums[i + 1]:
                res = max(res, left[i - 1][1] + right[i + 1][1])
            elif nums[i - 1] < nums[i + 1]:
                res = max(res, left[i - 1][0] + right[i + 1][0])
        return res

    # 3830. 移除至多一个元素后的最长交替子数组 (Longest Alternating Subarray After Removing At Most One Element)
    def longestAlternating(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, can_del: bool, inc: bool) -> int:
            if i == 0:
                return 1
            res = 1
            if nums[i - 1] != nums[i] and (nums[i - 1] < nums[i]) == inc:
                res = dfs(i - 1, can_del, not inc) + 1
            if (
                can_del
                and i > 1
                and nums[i - 2] != nums[i]
                and (nums[i - 2] < nums[i]) == inc
            ):
                res = max(res, dfs(i - 2, False, not inc) + 1)
            return res

        res = 0
        n = len(nums)
        for i in range(n):
            res = max(res, dfs(i, True, False), dfs(i, True, True))
        dfs.cache_clear()
        return res

    # 1653. 使字符串平衡的最少删除次数 (Minimum Deletions to Make String Balanced)
    def minimumDeletions(self, s: str) -> int:
        @cache
        def dfs(i: int, j: chr) -> int:
            if i == n:
                return 0
            res = dfs(i + 1, j) + 1
            if s[i] >= j:
                res = min(res, dfs(i + 1, s[i]))
            return res

        n = len(s)
        return dfs(0, "a")

    # 1653. 使字符串平衡的最少删除次数 (Minimum Deletions to Make String Balanced)
    def minimumDeletions(self, s: str) -> int:
        res = cnt_a = s.count("a")
        for c in s:
            if c == "a":
                cnt_a -= 1
            else:
                cnt_a += 1
            res = min(res, cnt_a)
        return res

    # 3833. 统计主导元素下标数 (Count Dominant Indices)
    def dominantIndices(self, nums: List[int]) -> int:
        s = 0
        res = 0
        n = len(nums)
        for i in range(n - 1, -1, -1):
            if s and nums[i] > s / (n - i - 1):
                res += 1
            s += nums[i]
        return res

    # 3834. 合并相邻且相等的元素 (Merge Adjacent Equal Elements)
    def mergeAdjacent(self, nums: List[int]) -> List[int]:
        st = []
        for x in nums:
            s = x
            while st and st[-1] == s:
                s += st.pop()
            st.append(s)
        return st

    # 3836. 恰好 K 个下标对的最大得分 (Maximum Score Using Exactly K Pairs)
    def maxScore(self, nums1: List[int], nums2: List[int], k: int) -> int:
        @cache
        def dfs(i: int, j: int, m: int) -> int:
            if m == 0:
                return 0
            if i < 0 or j < 0:
                return -inf
            return max(
                dfs(i - 1, j, m),
                dfs(i, j - 1, m),
                dfs(i - 1, j - 1, m - 1) + nums1[i] * nums2[j],
            )

        res = dfs(len(nums1) - 1, len(nums2) - 1, k)
        dfs.cache_clear()
        return res

    # 3835. 开销小于等于 K 的子数组数目 (Count Subarrays With Cost Less Than or Equal to K)
    def countSubarrays(self, nums: List[int], k: int) -> int:
        j = 0
        res = 0
        d = SortedList()
        for i, x in enumerate(nums):
            d.add(x)
            while (d[-1] - d[0]) * (i - j + 1) > k:
                d.remove(nums[j])
                j += 1
            res += i - j + 1
        return res

    # 3835. 开销小于等于 K 的子数组数目 (Count Subarrays With Cost Less Than or Equal to K)
    def countSubarrays(self, nums: List[int], k: int) -> int:
        q_min = deque()
        q_max = deque()
        res = 0
        left = 0
        for right, x in enumerate(nums):
            while q_min and x <= nums[q_min[-1]]:
                q_min.pop()
            q_min.append(right)

            while q_max and x >= nums[q_max[-1]]:
                q_max.pop()
            q_max.append(right)

            while (nums[q_max[0]] - nums[q_min[0]]) * (right - left + 1) > k:
                if q_max[0] <= left:
                    q_max.popleft()
                if q_min[0] <= left:
                    q_min.popleft()
                left += 1
            res += right - left + 1
        return res

    # 3829. 设计共享出行系统 (Design Ride Sharing System)
    class RideSharingSystem:

        def __init__(self):
            self.riders = deque()
            self.drivers = deque()
            self.waiting_riders = set()

        def addRider(self, riderId: int) -> None:
            self.riders.append(riderId)
            self.waiting_riders.add(riderId)

        def addDriver(self, driverId: int) -> None:
            self.drivers.append(driverId)

        def matchDriverWithRider(self) -> List[int]:
            while self.riders and self.riders[0] not in self.waiting_riders:
                self.riders.popleft()
            if not self.drivers or not self.riders:
                return [-1, -1]
            return [self.drivers.popleft(), self.riders.popleft()]

        def cancelRider(self, riderId: int) -> None:
            self.waiting_riders.discard(riderId)

    # 1382. 将二叉搜索树变平衡 (Balance a Binary Search Tree)
    def balanceBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def dfs(node: Optional[TreeNode]):
            if node is None:
                return
            dfs(node.left)
            a.append(node.val)
            dfs(node.right)

        def make_tree(i: int, j: int) -> Optional[TreeNode]:
            if i > j:
                return None
            mid = i + ((j - i) >> 1)
            node = TreeNode(a[mid])
            node.left = make_tree(i, mid - 1)
            node.right = make_tree(mid + 1, j)
            return node

        a = []
        dfs(root)
        n = len(a)
        return make_tree(0, n - 1)

    # 108. 将有序数组转换为二叉搜索树 (Convert Sorted Array to Binary Search Tree)
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def dfs(i: int, j: int) -> Optional[TreeNode]:
            if i > j:
                return None
            mid = i + ((j - i) >> 1)
            node = TreeNode(nums[mid])
            node.left = dfs(i, mid - 1)
            node.right = dfs(mid + 1, j)
            return node

        return dfs(0, len(nums) - 1)

    # 3831. 二叉搜索树某一层的中位数 (Median of a Binary Search Tree Level) --plus
    def levelMedian(self, root: Optional[TreeNode], level: int) -> int:
        q = deque()
        q.append(root)
        cur_level = 0
        while q:
            l = len(q)
            if cur_level == level:
                return q[l // 2].val
            for _ in range(l):
                x = q.popleft()
                if x.left:
                    q.append(x.left)
                if x.right:
                    q.append(x.right)
            cur_level += 1
        return -1

    # 3687. 图书馆逾期罚款计算器 (Library Late Fee Calculator) --plus
    def lateFee(self, daysLate: List[int]) -> int:
        res = 0
        for d in daysLate:
            mul = 1
            if d > 5:
                mul = 3
            elif d >= 2:
                mul = 2
            res += d * mul
        return res

    # 3711. 不出现负余额的最大交易额 (Maximum Transactions Without Negative Balance) --plus
    def maxTransactions(self, transactions: List[int]) -> int:
        q = []
        heapq.heapify(q)
        res = 0
        s = 0
        for t in transactions:
            if t < 0:
                heapq.heappush(q, t)
            s += t
            if s >= 0:
                res += 1
            else:
                s -= heapq.heappop(q)
        return res

    # 1246. 删除回文子数组 (Palindrome Removal) --plus
    def minimumMoves(self, arr: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == j:
                return 1
            if i + 1 == j:
                return 1 if arr[i] == arr[j] else 2
            res = inf
            if arr[i] == arr[j]:
                res = min(res, dfs(i + 1, j - 1))
            for k in range(i, j):
                res = min(res, dfs(i, k) + dfs(k + 1, j))
            return res

        n = len(arr)
        return dfs(0, n - 1)


"""
 This is ArrayReader's API interface.
 You should not implement it, or speculate about its implementation
 """


class ArrayReader(object):
    # Compares the sum of arr[l..r] with the sum of arr[x..y]
    # return 1 if sum(arr[l..r]) > sum(arr[x..y])
    # return 0 if sum(arr[l..r]) == sum(arr[x..y])
    # return -1 if sum(arr[l..r]) < sum(arr[x..y])
    def compareSub(self, l: int, r: int, x: int, y: int) -> int:
        return 0

    # Returns the length of the array
    def length(self) -> int:
        return 0

    # 1533. 找到最大整数的索引 (Find the Index of the Large Integer) --plus
    def getIndex(self, reader: "ArrayReader") -> int:
        left = 0
        right = reader.length() - 1
        while left < right:
            mid = left + ((right - left) >> 1)
            c = reader.compareSub(left, mid - (right - left + 1) % 2, mid + 1, right)
            if c == 0:
                return mid
            if c > 0:
                right = mid - ((right - left + 1) % 2)
            else:
                left = mid + 1
        return left


"""
This is BinaryMatrix's API interface.
You should not implement it, or speculate about its implementation
"""


class BinaryMatrix(object):
    def get(self, row: int, col: int) -> int:
        return 0

    def dimensions(self) -> list[int]:
        return None

    # 428. 至少有一个 1 的最左端列 (Leftmost Column with at Least a One) --plus
    def leftMostColumnWithOne(self, binaryMatrix: "BinaryMatrix") -> int:
        m, n = binaryMatrix.dimensions()
        left = 0
        right = n - 1
        while left <= right:
            mid = left + ((right - left) >> 1)
            if any(binaryMatrix.get(i, mid) for i in range(m)):
                right = mid - 1
            else:
                left = mid + 1
        return -1 if right + 1 == n else right + 1

    # 3353. 最小总操作数 (Minimum Total Operations) --plus
    def minOperations(self, nums: List[int]) -> int:
        return sum(x != y for x, y in pairwise(nums))

    # 3667. 按绝对值排序数组 (Sort Array By Absolute Value) --plus
    def sortByAbsoluteValue(self, nums: List[int]) -> List[int]:
        return sorted(nums, key=lambda o: abs(o))

    # 3476. 最大化任务分配的利润 (Maximize Profit from Task Assignment) --plus
    def maxProfit(self, workers: List[int], tasks: List[List[int]]) -> int:
        d = defaultdict(list)
        for r, b in tasks:
            d[r].append(b)
        for l in d.values():
            l.sort()
        res = 0
        for w in workers:
            if w in d:
                res += d[w].pop()
                if len(d[w]) == 0:
                    del d[w]
        mx = 0
        for l in d.values():
            mx = max(mx, l[-1])
        return res + mx

    # 2955. 同端子串的数量 (Number of Same-End Substrings) --plus
    def sameEndSubstringCount(self, s: str, queries: List[List[int]]) -> List[int]:
        n = len(s)
        pre = [[0] * (n + 1) for _ in range(26)]
        for c in range(26):
            for i, x in enumerate(s):
                id = ord(x) - ord("a")
                pre[c][i + 1] = pre[c][i] + int(id == c)
        res = []
        for l, r in queries:
            cur_s = 0
            for c in range(26):
                cur = pre[c][r + 1] - pre[c][l]
                cur_s += cur * (cur + 1) // 2
            res.append(cur_s)
        return res

    # 3199. 用偶数异或设置位计数三元组 I (Count Triplets with Even XOR Set Bits I) --plus
    # 3215. 用偶数异或设置位计数三元组 II (Count Triplets with Even XOR Set Bits II) --plus
    def tripletCount(self, a: List[int], b: List[int], c: List[int]) -> int:
        def cal(a: List[int]) -> List[int]:
            res = [0] * 2
            for x in a:
                res[x.bit_count() & 1] += 1
            return res

        cnt1 = cal(a)
        cnt2 = cal(b)
        cnt3 = cal(c)
        return (
            cnt1[0] * cnt2[0] * cnt3[0]
            + cnt1[1] * cnt2[1] * cnt3[0]
            + cnt1[1] * cnt2[0] * cnt3[1]
            + cnt1[0] * cnt2[1] * cnt3[1]
        )

    # 3641. 最长半重复子数组 (Longest Semi-Repeating Subarray) --plus
    def longestSubarray(self, nums: List[int], k: int) -> int:
        res = 0
        j = 0
        d = defaultdict(int)
        cnt = 0
        for i, x in enumerate(nums):
            d[x] += 1
            if d[x] == 2:
                cnt += 1
            while cnt > k:
                d[nums[j]] -= 1
                if d[nums[j]] == 1:
                    cnt -= 1
                j += 1
            res = max(res, i - j + 1)
        return res

    # 3610. 目标和所需的最小质数个数 (Minimum Number of Primes to Sum to Target) --plus
    def minNumberOfPrimes(self, n: int, m: int) -> int:
        def check(x: int) -> bool:
            for d in range(2, isqrt(x) + 1):
                if x % d == 0:
                    return False
            return True

        @cache
        def dfs(i: int, j: int) -> int:
            if j == n:
                return 0
            if i == l:
                return inf
            res = dfs(i + 1, j)
            if j + prime[i] <= n:
                res = min(res, dfs(i, j + prime[i]) + 1)
            return res

        prime = []
        i = 2
        while len(prime) < m and i <= n:
            if check(i):
                prime.append(i)
            i += 1
        l = len(prime)
        prime = list(reversed(prime))
        res = dfs(0, 0)
        dfs.cache_clear()
        return res if res < inf else -1

    # 2992. 自整除排列的数量 (Number of Self-Divisible Permutations) --plus
    def selfDivisiblePermutationCount(self, n: int) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == u:
                return 1
            c = u ^ i
            j = i.bit_count() + 1
            res = 0
            while c:
                lb = (c & -c).bit_length() - 1
                if gcd(j, lb) == 1:
                    res += dfs(i ^ (1 << lb))
                c &= c - 1
            return res

        u = (1 << (n + 1)) - 2
        return dfs(0)

    # 3004. 相同颜色的最大子树 (Maximum Subtree of the Same Color) --plus
    def maximumSubtreeSize(self, edges: List[List[int]], colors: List[int]) -> int:
        def dfs(x: int, fa: int) -> tuple:
            s = set()
            cnt = 0
            for y in g[x]:
                if y != fa:
                    cur = dfs(y, x)
                    s |= cur[0]
                    cnt += cur[1]
            s.add(colors[x])
            cnt += 1
            if len(s) == 1:
                nonlocal res
                res = max(res, cnt)
            return (s, cnt)

        res = 0
        n = len(colors)
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        dfs(0, -1)
        return res

    # 2838. 英雄可以获得的最大金币数 (Maximum Coins Heroes Can Collect) --plus
    def maximumCoins(
        self, heroes: List[int], monsters: List[int], coins: List[int]
    ) -> List[int]:
        a = [[x, y] for x, y in zip(monsters, coins)]
        a.sort()
        h = [[x, i] for i, x in enumerate(heroes)]
        h.sort()
        res = [0] * len(heroes)
        j = 0
        s = 0
        for x, id in h:
            while j < len(a) and a[j][0] <= x:
                s += a[j][1]
                j += 1
            res[id] = s
        return res

    # 3837. Delayed Count of Equal Elements --plus
    def delayedCount(self, nums: List[int], k: int) -> List[int]:
        def binary_search(a: list, x: int) -> int:
            left = 0
            right = len(a) - 1
            while left <= right:
                mid = left + ((right - left) >> 1)
                if a[mid] >= x:
                    right = mid - 1
                else:
                    left = mid + 1
            return right + 1

        d = defaultdict(list)
        for i, x in enumerate(nums):
            d[x].append(i)
        n = len(nums)
        res = [0] * n
        for i, x in enumerate(nums):
            a = d[x]
            j = binary_search(a, i + k + 1)
            if j != len(a):
                res[i] = len(a) - j
        return res

    # 3437. 全排列 III (Permutations III) --plus
    def permute(self, n: int) -> List[List[int]]:
        def dfs(i: int):
            if i == u:
                res.append(path.copy())
                return
            c = i ^ u
            while c:
                lb = (c & -c).bit_length() - 1
                if len(path) == 0 or (path[-1] & 1) ^ (lb & 1):
                    path.append(lb)
                    dfs(i ^ (1 << lb))
                    path.pop()
                c &= c - 1

        res = []
        path = []
        u = (1 << (n + 1)) - 2
        dfs(0)
        return res

    # 2313. 二叉树中得到结果所需的最少翻转次数 (Minimum Flips in Binary Tree to Get Result) --plus
    @cache
    def minimumFlips(self, root: Optional[TreeNode], result: bool) -> int:
        # 0: False ; 1: True
        if root.val <= 1:
            return result ^ root.val
        # not
        if root.val == 5:
            return self.minimumFlips(root.left if root.left else root.right, not result)
        cur1 = self.minimumFlips(root.left, True)
        cur2 = self.minimumFlips(root.left, False)
        cur3 = self.minimumFlips(root.right, True)
        cur4 = self.minimumFlips(root.right, False)
        # or
        if root.val == 2:
            return (
                min(cur1 + min(cur3, cur4), cur3 + min(cur1, cur2))
                if result
                else cur2 + cur4
            )
        # and
        if root.val == 3:
            return (
                cur1 + cur3
                if result
                else min(cur2 + min(cur3, cur4), cur4 + min(cur1, cur2))
            )
        # xor
        return (
            min(cur1 + cur4, cur2 + cur3) if result else min(cur1 + cur3, cur2 + cur4)
        )

    # 3496. 最大化配对删除后的得分 (Maximize Score After Pair Deletions) --plus
    def maxScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return 0
        s = sum(nums)
        _min = min(nums)
        _min_pair = min(x + y for x, y in pairwise(nums))
        return s - _min if n & 1 else s - _min_pair

    # 2832. 每个元素为最大值的最大范围 (Maximal Range That Each Element Is Maximum in It) --plus
    def maximumLengthOfRanges(self, nums: List[int]) -> List[int]:
        n = len(nums)
        left = [-1] * n
        st = []
        for i, x in enumerate(nums):
            while st and nums[st[-1]] <= x:
                st.pop()
            if st:
                left[i] = st[-1]
            st.append(i)
        st.clear()
        right = [n] * n
        for i in range(n - 1, -1, -1):
            while st and nums[st[-1]] <= nums[i]:
                st.pop()
            if st:
                right[i] = st[-1]
            st.append(i)
        return [r - l - 1 for l, r in zip(left, right)]

    # 317. 离建筑物最近的距离 (Shortest Distance from All Buildings) --plus
    def shortestDistance(self, grid: List[List[int]]) -> int:
        def cal(i: int, j: int):
            vis = [[False] * n for _ in range(m)]
            vis[i][j] = True
            q = deque()
            q.append((i, j, 0))
            while q:
                x, y, d = q.popleft()
                for dx, dy in (0, 1), (0, -1), (1, 0), (-1, 0):
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < m
                        and 0 <= ny < n
                        and grid[nx][ny] == 0
                        and not vis[nx][ny]
                    ):
                        cnt_in_each_0[nx][ny] += 1
                        dis_in_each_0[nx][ny] += d + 1
                        vis[nx][ny] = True
                        q.append((nx, ny, d + 1))

        # 从每个1出发，统计每个0，有多少1可以到达，并统计到达0的距离和
        m, n = len(grid), len(grid[0])
        # 从1出发，到达每个0的最短总距离
        dis_in_each_0 = [[0] * n for _ in range(m)]
        # 从1出发，能到达每个0的1的个数
        cnt_in_each_0 = [[0] * n for _ in range(m)]
        # 1的个数
        cnt1 = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    cnt1 += 1
                    cal(i, j)
        res = inf
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0 and cnt_in_each_0[i][j] == cnt1:
                    res = min(res, dis_in_each_0[i][j])
        return res if res < inf else -1

    # 1259. 不相交的握手 (Handshakes That Don't Cross) --plus
    def numberOfWays(self, numPeople: int) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == 0:
                return 1
            res = 0
            for k in range(2, i + 1, 2):
                res += dfs(k - 2) * dfs(i - k)
                res %= MOD
            return res

        MOD = 10**9 + 7
        return dfs(numPeople)

    # 3329. 字符至少出现 K 次的子字符串 II (Count Substrings With K-Frequency Characters II) --plus
    def numberOfSubstrings(self, s: str, k: int) -> int:
        cnts = [0] * 26
        j = 0
        res = 0
        n = len(s)
        for i, x in enumerate(s):
            cnts[ord(x) - ord("a")] += 1
            while cnts[ord(x) - ord("a")] >= k:
                cnts[ord(s[j]) - ord("a")] -= 1
                j += 1
            res += i - j + 1

        return n * (n + 1) // 2 - res

    # 3323. 通过插入区间最小化连通组 (Minimize Connected Groups by Inserting Interval) --plus
    def minConnectedGroups(self, intervals: List[List[int]], k: int) -> int:
        n = len(intervals)
        intervals.sort()
        a = []
        i = 0
        while i < n:
            l, r = intervals[i][0], intervals[i][1]
            j = i + 1
            while j < n and intervals[j][0] <= r:
                r = max(r, intervals[j][1])
                j += 1
            a.append((l, r))
            i = j
        res = 0
        i = 0
        j = 0
        for i, (l, r) in enumerate(a):
            while j < i and l - a[j][1] > k:
                j += 1
            res = max(res, i - j)
        return len(a) - res


# Definition for an Interval.
class Interval:
    def __init__(self, start: int = None, end: int = None):
        self.start = start
        self.end = end

    # 759. 员工空闲时间 (Employee Free Time) --plus
    def employeeFreeTime(self, schedule: "[[Interval]]") -> "[Interval]":
        def merge(a: "[Interval]", b: "[Interval]") -> "[Interval]":
            merged_list = []
            i, j, m, n = 0, 0, len(a), len(b)
            while i < m and j < n:
                if a[i].start < b[j].start:
                    merged_list.append(a[i])
                    i += 1
                else:
                    merged_list.append(b[j])
                    j += 1
            merged_list.extend(a[i:])
            merged_list.extend(b[j:])
            res = []
            i = 0
            while i < len(merged_list):
                l = merged_list[i].start
                r = merged_list[i].end
                j = i + 1
                while j < len(merged_list) and merged_list[j].start <= r:
                    r = max(r, merged_list[j].end)
                    j += 1
                res.append(Interval(l, r))
                i = j
            return res

        a = []
        for b in schedule:
            a = merge(a, b)
        return [Interval(x.end, y.start) for x, y in pairwise(a)]

    # 2737. 找到最近的标记节点 (Find the Closest Marked Node) --plus
    def minimumDistance(
        self, n: int, edges: List[List[int]], s: int, marked: List[int]
    ) -> int:
        g = [[] for _ in range(n)]
        for u, v, w in edges:
            g[u].append((v, w))
        dis = [inf] * n
        dis[s] = 0
        q = deque()
        q.append((0, s))
        while q:
            d, x = q.pop()
            if d > dis[x]:
                continue
            for y, dx in g[x]:
                if d + dx < dis[y]:
                    dis[y] = d + dx
                    q.append((d + dx, y))
        res = min(dis[m] for m in marked)
        return res if res < inf else -1

    # 2737. 找到最近的标记节点 (Find the Closest Marked Node) --plus
    def minimumDistance(
        self, n: int, edges: List[List[int]], s: int, marked: List[int]
    ) -> int:
        g = [[] for _ in range(n)]
        for u, v, w in edges:
            g[u].append((v, w))
        _set = set(marked)
        dis = [inf] * n
        dis[s] = 0
        q = [(0, s)]
        heapq.heapify(q)
        while q:
            d, x = heapq.heappop(q)
            if d > dis[x]:
                continue
            if x in _set:
                return d
            for y, dx in g[x]:
                if d + dx < dis[y]:
                    dis[y] = d + dx
                    heapq.heappush(q, (d + dx, y))
        return -1

    # 2950. 可整除子串的数量 (Number of Divisible Substrings) --plus
    def countDivisibleSubstrings(self, word: str) -> int:
        def cal(x: str) -> int:
            return (ord(x) - ord("a") + 1) // 3 + 1

        _max = max(cal(x) for x in word)
        _min = min(cal(x) for x in word)
        res = 0
        for l in range(_max, _min - 1, -1):
            s = 0
            d = defaultdict(int)
            d[0] = 1
            for x in word:
                s += cal(x) - l
                res += d[s]
                d[s] += 1
        return res

    # 3714. 最长的平衡子串 II (Longest Balanced Substring II)
    def longestBalanced(self, s: str) -> int:
        def cal_one(c: str) -> int:
            cnt = 0
            res = 0
            for x in s:
                if x == c:
                    cnt += 1
                    res = max(res, cnt)
                else:
                    cnt = 0
            return res

        def cal_two(c: str) -> int:
            pre = 0
            d = defaultdict(int)
            d[0] = -1
            res = 0
            for i, x in enumerate(s):
                if x not in c:
                    d.clear()
                    d[0] = i
                    pre = 0
                else:
                    pre += 1 if c[0] == x else -1
                    if pre in d:
                        res = max(res, i - d[pre])
                    else:
                        d[pre] = i
            return res

        def cal_three() -> int:
            res = 0
            pre = [0] * 3
            d = defaultdict(int)
            d[(0, 0)] = -1
            for i, x in enumerate(s):
                id = ord(x) - ord("a")
                pre[id] += 1
                k = (pre[0] - pre[1], pre[1] - pre[2])
                if k in d:
                    res = max(res, i - d[k])
                else:
                    d[k] = i
            return res

        res = 0
        # 出现恰好一种字符的最长字串
        for c in ("a", "b", "c"):
            res = max(res, cal_one(c))
        # 出现恰好两种字符的最长字串
        for c in ("ab", "ac", "bc"):
            res = max(res, cal_two(c))
        # 出现恰好三种字符的最长字串
        res = max(res, cal_three())
        return res

    # 3749. 计算有效表达式 (Evaluate Valid Expressions) --plus
    def evaluateExpression(self, expression: str) -> int:
        def cal(op: str, a: str, b: str) -> list:
            x, y = int(a), int(b)
            res = 0
            if op == "add":
                res = x + y
            elif op == "sub":
                res = x - y
            elif op == "mul":
                res = x * y
            else:
                res = x // y
            s = str(res)
            return [x for x in s]

        st = []
        for x in expression:
            if x == ")":
                b = deque()
                while st[-1] != ",":
                    b.appendleft(st.pop())
                st.pop()
                a = deque()
                while st[-1] != "(":
                    a.appendleft(st.pop())
                st.pop()
                op = deque()
                while len(op) < 3:
                    op.appendleft(st.pop())
                result = cal("".join(op), "".join(a), "".join(b))
                st.extend(result)
            else:
                st.append(x)
        return int("".join(st))

    # 3339. 查找 K 偶数数组的数量 (Find the Number of K-Even Arrays) --plus
    def countOfArrays(self, n: int, m: int, k: int) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i < 0:
                return k == 0
            if k < 0:
                return 0
            return (dfs(i - 1, 0, k - (j ^ 1)) * even + dfs(i - 1, 1, k) * odd) % MOD

        even, odd = m // 2, (m + 1) // 2
        MOD = 10**9 + 7
        return dfs(n - 1, 1, k)

    # 3155. 可升级服务器的最大数量 (Maximum Number of Upgradable Servers) --plus
    def maxUpgrades(
        self, count: List[int], upgrade: List[int], sell: List[int], money: List[int]
    ) -> List[int]:
        return [
            min(c, (c * s + m) // (s + u))
            for c, u, s, m in zip(count, upgrade, sell, money)
        ]

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

    # 2932. 找出强数对的最大异或值 I (Maximum Strong Pair XOR I)
    # 2935. 找出强数对的最大异或值 II (Maximum Strong Pair XOR II) --0-1字典树
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        class trie:
            def __init__(self):
                self.children = [None] * 2
                self.cnt = 0

            def insert(self, x: int):
                node = self
                for i in range(19, -1, -1):
                    bit = x >> i & 1
                    if node.children[bit] is None:
                        node.children[bit] = trie()
                    node = node.children[bit]
                    node.cnt += 1

            def delete(self, x: int):
                node = self
                for i in range(19, -1, -1):
                    bit = x >> i & 1
                    node = node.children[bit]
                    node.cnt -= 1

            def check(self, x: int) -> int:
                node = self
                res = 0
                for i in range(19, -1, -1):
                    bit = x >> i & 1
                    if node.children[bit ^ 1] and node.children[bit ^ 1].cnt:
                        bit ^= 1
                        res ^= 1 << i
                    node = node.children[bit]
                return res

        nums.sort()
        res, j = 0, 0
        _trie = trie()
        for x in nums:
            while nums[j] * 2 < x:
                _trie.delete(nums[j])
                j += 1
            _trie.insert(x)
            res = max(res, _trie.check(x))
        return res

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

    # 421. 数组中两个数的最大异或值 (Maximum XOR of Two Numbers in an Array) --0-1字典树
    # LCR 067. 数组中两个数的最大异或值
    def findMaximumXOR(self, nums: List[int]) -> int:
        class trie:
            def __init__(self):
                self.children = [None] * 2

            def insert(self, x: int):
                node = self
                for i in range(30, -1, -1):
                    bit = x >> i & 1
                    if node.children[bit] is None:
                        node.children[bit] = trie()
                    node = node.children[bit]

            def check(self, x: int) -> int:
                node = self
                res = 0
                for i in range(30, -1, -1):
                    bit = x >> i & 1
                    if node.children[bit ^ 1]:
                        bit ^= 1
                        res ^= 1 << i
                    node = node.children[bit]
                return res

        res = 0
        _trie = trie()
        for x in nums:
            _trie.insert(x)
            res = max(res, _trie.check(x))
        return res

    # 1707. 与数组中元素的最大异或值 (Maximum XOR With an Element From Array) --0-1字典树
    def maximizeXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        class trie:
            def __init__(self):
                self.children = [None] * 2

            def insert(self, x: int):
                node = self
                for i in range(29, -1, -1):
                    bit = x >> i & 1
                    if node.children[bit] is None:
                        node.children[bit] = trie()
                    node = node.children[bit]

            def check(self, x: int) -> int:
                node = self
                res = 0
                for i in range(29, -1, -1):
                    bit = x >> i & 1
                    if node.children[bit ^ 1]:
                        bit ^= 1
                        res ^= 1 << i
                    node = node.children[bit]
                return res

        nums.sort()
        n = len(queries)
        for i, q in enumerate(queries):
            q.append(i)
        print(queries)
        queries.sort(key=lambda o: o[1])
        res = [-1] * n
        _trie = trie()
        i = 0
        for x, m, id in queries:
            while i < len(nums) and nums[i] <= m:
                _trie.insert(nums[i])
                i += 1
            if i:
                res[id] = _trie.check(x)
        return res

    # 1938. 查询最大基因差 (Maximum Genetic Difference Query) --0-1字典树
    def maxGeneticDifference(
        self, parents: List[int], queries: List[List[int]]
    ) -> List[int]:
        class trie:
            def __init__(self):
                self.children = [None] * 2
                self.cnt = 0

            def insert(self, x: int):
                node = self
                for i in range(17, -1, -1):
                    bit = x >> i & 1
                    if node.children[bit] is None:
                        node.children[bit] = trie()
                    node = node.children[bit]
                    node.cnt += 1

            def delete(self, x: int):
                node = self
                for i in range(17, -1, -1):
                    bit = x >> i & 1
                    node = node.children[bit]
                    node.cnt -= 1

            def check(self, x: int) -> int:
                node = self
                res = 0
                for i in range(17, -1, -1):
                    bit = x >> i & 1
                    if node.children[bit ^ 1] and node.children[bit ^ 1].cnt:
                        bit ^= 1
                        res ^= 1 << i
                    node = node.children[bit]
                return res

        def dfs(x: int):
            _trie.insert(x)
            for val, i in d[x]:
                res[i] = _trie.check(val)
            del d[x]
            for y in g[x]:
                dfs(y)
            _trie.delete(x)

        d = defaultdict(list)
        for i, (node, val) in enumerate(queries):
            d[node].append((val, i))
        n = len(parents)
        g = [[] for _ in range(n)]
        r = -1
        for i, v in enumerate(parents):
            if v == -1:
                r = i
            else:
                g[v].append(i)
        _trie = trie()
        res = [0] * len(queries)
        dfs(r)
        return res

    # 2479. 两个不重叠子树的最大异或值 (Maximum XOR of Two Non-Overlapping Subtrees) --0-1字典树 --plus
    def maxXor(self, n: int, edges: List[List[int]], values: List[int]) -> int:
        class trie:
            def __init__(self):
                self.children = [None] * 2

            def insert(self, x: int):
                node = self
                for i in range(44, -1, -1):
                    bit = (x >> i) & 1
                    if node.children[bit] is None:
                        node.children[bit] = trie()
                    node = node.children[bit]

            def check(self, x: int) -> int:
                node = self
                res = 0
                for i in range(44, -1, -1):
                    bit = (x >> i) & 1
                    if node is None:
                        break
                    if node.children[bit ^ 1]:
                        bit ^= 1
                        res ^= 1 << i
                    node = node.children[bit]
                return res

        def dfs_pre(x: int, fa: int) -> int:
            s = values[x]
            for y in g[x]:
                if y != fa:
                    s += dfs_pre(y, x)
            pre[x] = s
            return s

        def dfs_xor(x: int, fa: int) -> int:
            res = _trie.check(pre[x])
            for y in g[x]:
                if y != fa:
                    res = max(res, dfs_xor(y, x))
            _trie.insert(pre[x])
            return res

        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        pre = [0] * n
        dfs_pre(0, -1)
        _trie = trie()
        return dfs_xor(0, -1)

    # 3632. 异或至少为 K 的子数组数目 (Subarrays with XOR at Least K) --0-1字典树 --plus
    def countXorSubarrays(self, nums: List[int], k: int) -> int:
        class trie:
            def __init__(self):
                self.children = [None] * 2
                self.cnt = 0

            def insert(self, x: int):
                node = self
                for i in range(29, -1, -1):
                    bit = x >> i & 1
                    if node.children[bit] is None:
                        node.children[bit] = trie()
                    node = node.children[bit]
                    node.cnt += 1

            def check(self, x: int, k: int) -> int:
                node = self
                res = 0
                for i in range(29, -1, -1):
                    x_bit, k_bit = x >> i & 1, k >> i & 1
                    if k_bit == 0:
                        if node.children[x_bit ^ 1]:
                            res += node.children[x_bit ^ 1].cnt
                        if node.children[x_bit]:
                            node = node.children[x_bit]
                        else:
                            return res
                    else:
                        if node.children[x_bit ^ 1]:
                            node = node.children[x_bit ^ 1]
                        else:
                            return res
                res += node.cnt
                return res

        _trie = trie()
        pre = list(accumulate(nums, xor, initial=0))  # type: ignore
        res = 0
        for i in range(len(nums)):
            _trie.insert(pre[i])
            res += _trie.check(pre[i + 1], k)
        return res

    # 1803. 统计异或值在范围内的数对有多少 (Count Pairs With XOR in a Range) --0-1字典树
    def countPairs(self, nums: List[int], low: int, high: int) -> int:
        class trie:
            def __init__(self):
                self.children = [None] * 2
                self.cnt = 0

            def insert(self, x: int):
                node = self
                for i in range(15, -1, -1):
                    bit = x >> i & 1
                    if node.children[bit] is None:
                        node.children[bit] = trie()
                    node = node.children[bit]
                    node.cnt += 1

            def check(self, x: int, k: int) -> int:
                node = self
                res = 0
                for i in range(15, -1, -1):
                    x_bit, k_bit = x >> i & 1, k >> i & 1
                    if k_bit == 0:
                        if node.children[x_bit]:
                            node = node.children[x_bit]
                        else:
                            return res
                    else:
                        if node.children[x_bit]:
                            res += node.children[x_bit].cnt
                        if node.children[x_bit ^ 1]:
                            node = node.children[x_bit ^ 1]
                        else:
                            return res
                res += node.cnt
                return res

        def cal(k: int) -> int:
            _trie = trie()
            res = 0
            for x in nums:
                res += _trie.check(x, k)
                _trie.insert(x)
            return res

        return cal(high) - cal(low - 1)

    # 799. 香槟塔 (Champagne Tower)
    def champagneTower(self, poured: int, query_row: int, query_glass: int) -> float:
        dp = [[0.0] * (query_row + 2) for _ in range(query_row + 2)]
        dp[0][0] = poured
        for i in range(query_row + 1):
            for j in range(query_glass + 1):
                p = max(0.0, dp[i][j] - 1) / 2
                dp[i + 1][j] += p
                dp[i + 1][j + 1] += p
        return min(1.0, dp[query_row][query_glass])


# Definition for BigArray.
class BigArray:
    def at(self, index: int) -> int:
        pass

    def size(self) -> int:
        pass

    # 2936. 包含相等值数字块的数量 (Number of Equal Numbers Blocks) --plus
    def countBlocks(self, nums: Optional["BigArray"]) -> int:
        n = nums.size()
        left = 0
        right = n - 1
        res = 0
        while left != n:
            x = nums.at(left)
            right = n - 1
            while left <= right:
                mid = left + ((right - left) >> 1)
                if nums.at(mid) != x:
                    right = mid - 1
                else:
                    left = mid + 1
            res += 1
        return res

    # 1152. 用户网站访问行为分析 (Analyze User Website Visit Pattern) --plus
    def mostVisitedPattern(
        self, username: List[str], timestamp: List[int], website: List[str]
    ) -> List[str]:
        d = defaultdict(list)
        for user, time, web in zip(username, timestamp, website):
            d[user].append((time, web))
        g = defaultdict(set)
        for user, l in d.items():
            l.sort()
            for a, b, c in combinations(l, 3):
                g[(a[1], b[1], c[1])].add(user)
        mx = 0
        res = []
        for k, v in g.items():
            if len(v) > mx:
                res.clear()
                mx = len(v)
                res.append(k)
            elif len(v) == mx:
                res.append(k)
        return sorted(res)[0]

    # 67. 二进制求和 (Add Binary)
    def addBinary(self, a: str, b: str) -> str:
        i, j, carry = len(a) - 1, len(b) - 1, 0
        res = []
        while i >= 0 or j >= 0 or carry:
            if i >= 0:
                carry += int(a[i])
                i -= 1
            if j >= 0:
                carry += int(b[j])
                j -= 1
            res.append(str(carry & 1))
            carry >>= 1
        return "".join(res[::-1])

    # 3838. 带权单词映射 (Weighted Word Mapping)
    def mapWordWeights(self, words: List[str], weights: List[int]) -> str:
        res = []
        for w in words:
            s = 0
            for c in w:
                s += weights[ord(c) - ord("a")]
            s %= 26
            res.append(chr(25 - s + ord("a")))
        return "".join(res)

    # 3839. 前缀连接组的数目 (Number of Prefix Connected Groups)
    def prefixConnected(self, words: List[str], k: int) -> int:
        d = defaultdict(int)
        for w in words:
            if len(w) < k:
                continue
            d[w[:k]] += 1
        return sum(v >= 2 for v in d.values())

    # 3840. 打家劫舍 V (House Robber V)
    def rob(self, nums: List[int], colors: List[int]) -> int:
        def rob(a: list) -> int:
            @cache
            def dfs(i: int) -> int:
                if i >= l:
                    return 0
                return max(dfs(i + 1), dfs(i + 2) + a[i])

            l = len(a)
            return dfs(0)

        res = 0
        i = 0
        n = len(nums)
        while i < n:
            j = i
            while j < n and colors[j] == colors[i]:
                j += 1
            res += rob(nums[i:j])
            i = j
        return res

    # 482. 密钥格式化 (License Key Formatting)
    def licenseKeyFormatting(self, s: str, k: int) -> str:
        a = [x.upper() for x in s if x != "-"]
        q = []
        for i in range(len(a) - 1, -1, -k):
            q.append("".join(a[max(0, i - k + 1) : i + 1]))
        res = []
        while q:
            if res:
                res.append("-")
            res.append(q.pop())
        return "".join(res)
