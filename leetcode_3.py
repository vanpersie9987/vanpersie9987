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
from tokenize import String
from xxlimited import foo
from audioop import minmax, reverse
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
from itertools import accumulate, combinations, count, islice, pairwise, permutations
from locale import DAY_4
from logging import _Level, root
from math import comb, cos, e, fabs, floor, gcd, inf, isqrt, lcm, sqrt
from mimetypes import init
from multiprocessing import reduction
from operator import is_, le, ne, truediv
from os import eventfd, lseek, minor, name, pread
from pickletools import read_uint1
from queue import PriorityQueue
from re import L, X
import re
from socket import NI_NUMERICSERV
from ssl import VERIFY_X509_TRUSTED_FIRST
from string import ascii_lowercase
from tabnanny import check
from tarfile import tar_filter
from telnetlib import EOR
from textwrap import indent
import time
from tkinter import N, NO, W
from tkinter.messagebox import RETRY
from tkinter.tix import Tree
from token import NL, RIGHTSHIFT
from turtle import RawTurtle, left, mode, pos, reset, right, rt, st, up, update
from typing import List, Optional, Self
import heapq
import bisect
from unittest import result
from unittest.util import _count_diff_all_purpose
from wsgiref.simple_server import make_server
from wsgiref.util import guess_scheme
from xml.dom import Node
from zoneinfo import reset_tzpath

# curl https://bootstrap.pypa.io/pip/get-pip.py -o get-pip.py
# sudo python3 get-pip.py
# pip3 install sortedcontainers
from networkx import bull_graph, dfs_edges, grid_2d_graph, interval_graph, power, union
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
                self.parent = [i for i in range(n)]
                self.rank = [1] * n

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

        u = union(c)
        for x, y in connections:
            u.union(x - 1, y - 1)
        dic = defaultdict(SortedSet)
        for i in range(c):
            r = u.get_root(i)
            dic[r].add(i)
        res = []
        for k, x in queries:
            x -= 1
            r = u.get_root(x)
            _s = dic[r]
            if k == 2:
                _s.discard(x)
            else:
                if x in _s:
                    res.append(x + 1)
                elif not _s:
                    res.append(-1)
                else:
                    res.append(_s[0] + 1)
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
            cnt_L = 0
            cnt_C = 0
            res = 0
            for c in s:
                if c == "L":
                    cnt_L += 1
                elif c == "C":
                    cnt_C += cnt_L
                elif c == "T":
                    res += cnt_C
            return res

        def check(s: str) -> int:
            n = len(s)
            left = [0] * n
            left[0] = s[0] == "L"
            for i in range(1, n):
                left[i] = left[i - 1] + (s[i] == "L")
            right = s[-1] == "T"
            res = 0
            mx = 0
            for i in range(n - 2, -1, -1):
                right += s[i] == "T"
                if s[i] == "C":
                    res += left[i] * right
                mx = max(mx, left[i] * right)
            return res + mx

        return max(cal("L" + s), cal(s + "T"), check(s))


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
        # 上一步在 (i, j)，移动方向为 DIRS[d]，是否可以右转，当前位置目标值
        @cache
        def dfs(i: int, j: int, d: int, can_turn: int, t: int) -> int:
            i += DIRS[d][0]
            j += DIRS[d][1]
            if not (m > i >= 0 and n > j >= 0) or grid[i][j] != t:
                return 0
            res = dfs(i, j, d, can_turn, 2 - t)
            if can_turn:
                _mxs = (i, n - j - 1, m - i - 1, j)
                d = (d + 1) % 4
                if _mxs[d] > res:
                    res = max(res, dfs(i, j, d, False, 2 - t))
            return res + 1

        m = len(grid)
        n = len(grid[0])
        res = 0
        # ↖️ : 0
        # ↗️ : 1
        # ↘️ : 2
        # ↙️ : 3
        DIRS = (-1, -1), (-1, 1), (1, 1), (1, -1)
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    mxs = (i + 1, n - j, m - i, j + 1)
                    for d, mx in enumerate(mxs):
                        if mx > res:
                            res = max(res, dfs(i, j, d, True, 2) + 1)
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
