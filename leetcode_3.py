from ast import Return, Tuple
from asyncio import FastChildWatcher
from gettext import find
import math
from platform import node
from pydoc import plain
from signal import valid_signals
from sqlite3 import paramstyle
import stat
from xxlimited import foo
from audioop import minmax, reverse
from calendar import c
from collections import Counter, defaultdict, deque
import collections
from ctypes.wintypes import _ULARGE_INTEGER
from curses import can_change_color, curs_set, intrflush, nonl
from curses.ascii import SI, isprint
from decimal import Rounded
import dis
import enum
from functools import cache, cached_property
from inspect import modulesbyfile
from itertools import accumulate, combinations, count, islice, pairwise, permutations
from locale import DAY_4
from logging import _Level, root
from math import comb, cos, e, fabs, gcd, inf, isqrt, lcm, sqrt
from mimetypes import init
from multiprocessing import reduction
from operator import le, ne, truediv
from os import eventfd, minor, name, pread
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
from turtle import RawTurtle, left, mode, pos, reset, right, st, up
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
from networkx import bull_graph, dfs_edges, grid_2d_graph, interval_graph, union
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
                        if s * 2 == total or s * 2 - total == a[0][0] or s * 2 - total == row[0]:
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
                if matrix[i][j] != '.' and matrix[i][j] != '#':
                    dic[ord(matrix[i][j]) - ord('A')].append((i, j))
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
                if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] != '#':
                    if dis[nx][ny] > d + 1:
                        dis[nx][ny] = d + 1
                        q.append((d + 1, nx, ny))
            if matrix[x][y] != '.' and matrix[x][y] != '#' and (ord(matrix[x][y]) - ord('A')) in dic:
                idx = ord(matrix[x][y]) - ord('A')
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
                num = int(s[i:j + 1])
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
        return ''.join(st)

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
                return ''
            res = s[i] + dfs(i + 1)
            for j in range(i + 1, n, 2):
                if check(i, j):
                    res = min(res, dfs(j + 1))
            return res
        n = len(s)
        return dfs(0)

    # 1298. 你能从盒子里获得的最大糖果数 (Maximum Candies You Can Get from Boxes)
    def maxCandies(self, status: List[int], candies: List[int], keys: List[List[int]], containedBoxes: List[List[int]], initialBoxes: List[int]) -> int:
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
            sub_grid = grid[i: i + k]
            for j in range(n - k + 1):
                a = []
                for row in sub_grid:
                    a.extend(row[j: j + k])
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
                        if dfs(nx, ny, max(mx, grid[nx][ny]), mask | (1 << (nx * n + ny))):
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
        
