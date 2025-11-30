from ast import Return, Tuple, literal_eval
from asyncio import FastChildWatcher
from curses.panel import bottom_panel
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
from re import L, S, T, X
import re
from socket import NI_NUMERICSERV
from ssl import VERIFY_X509_TRUSTED_FIRST
from string import ascii_lowercase
from tabnanny import check
from tarfile import tar_filter
from textwrap import indent
import time
from tkinter import N, NO, W
from tkinter.messagebox import RETRY
from token import NL, RIGHTSHIFT
from turtle import (
    RawTurtle,
    color,
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
from wsgiref.util import guess_scheme
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
            if i == n:
                return 0 if j == 0 else -inf
            if m == k:
                return 0
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
        n = len(nums)
        i = 1
        while i < n:
            if nums[i] > nums[i - 1]:
                i += 1
            else:
                break
        if i == 1 or i == n:
            return False
        i -= 1
        j = n - 2
        while j >= 0:
            if nums[j + 1] > nums[j]:
                j -= 1
            else:
                break
        if j == n - 2:
            return False
        j += 1
        while i + 1 <= j:
            if nums[i + 1] >= nums[i]:
                return False
            i += 1
        return True

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
        # j == 0 未选
        # j == 1 已经选择了1个
        # j == 2 已经在第一个上升阶段
        # j == 3 已经在第一个下降阶段
        # j == 4 已经在第二个上升阶段
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0 if j == 4 else -inf
            if j == 0:
                return max(dfs(i + 1, j), dfs(i + 1, j + 1) + nums[i])
            if j == 1:
                if nums[i - 1] >= nums[i]:
                    return -inf
                return dfs(i + 1, j + 1) + nums[i]
            if j == 2:
                if nums[i] == nums[i - 1]:
                    return -inf
                return dfs(i + 1, j + (nums[i - 1] > nums[i])) + nums[i]
            if j == 3:
                if nums[i] == nums[i - 1]:
                    return -inf
                return dfs(i + 1, j + (nums[i - 1] < nums[i])) + nums[i]
            res = 0
            if nums[i - 1] < nums[i]:
                res = max(res, dfs(i + 1, j) + nums[i])
            return res

        n = len(nums)
        res = dfs(0, 0)
        dfs.cache_clear()
        return res

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
                if w + dw < dis[y]:
                    dis[y] = w + dw
                    heapq.heappush(q, (w + dw, y))
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
        p = list(accumulate(prices, initial=0))
        ps = list(accumulate([a * b for a, b in zip(prices, strategy)], initial=0))
        return max(
            ps[-1],
            max(
                ps[-1] - (ps[i] - ps[i - k]) + p[i] - p[i - k // 2]
                for i in range(k, n + 1)
            ),
        )

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
        def check() -> bool:
            x = -1
            for c in cnts:
                if c == 0:
                    continue
                if x != -1 and c != x:
                    return False
                x = c
            return True

        n = len(s)
        for L in range(n, 0, -1):
            cnts = [0] * 26
            for i, v in enumerate(s):
                cnts[ord(v) - ord("a")] += 1
                if i >= L:
                    cnts[ord(s[i - L]) - ord("a")] -= 1
                if i >= L - 1 and check():
                    return L

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
        res = 0
        pre = 30
        while n:
            lb = (n & -n).bit_length() - 1
            res = max(res, lb - pre)
            pre = lb
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
                    for y in ret[i]:
                        if j & masks[y] == 0:
                            res = max(res, dfs(i + 1, j | masks[y]) + vals[y])
                return res

            ret = [[] for _ in range(10)]
            for y in g[x]:
                for i, l in enumerate(dfs_tree(y)):
                    ret[i].extend(l)
            mask = masks[x]
            while mask:
                lb = (mask & -mask).bit_length() - 1
                ret[lb].append(x)
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
            m |= 1 << (ord(x) - ord('a'))
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
