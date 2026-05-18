from ast import Return, Tuple, literal_eval
from asyncio import FastChildWatcher
import csv
from curses.panel import bottom_panel
from dbm import dumb
from doctest import FAIL_FAST
from errno import EHWPOISON
from gettext import find
import math
from pdb import run
from platform import node
from posixpath import isabs
from pydoc import plain
import queue
from signal import valid_signals
from sqlite3 import paramstyle
import stat
from termios import CINTR, N_PPP
from tokenize import String
from tty import CC
from unicodedata import numeric
from xxlimited import foo
from calendar import c, isleap
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
from math import comb, cos, e, fabs, floor, gcd, inf, isqrt, lcm, pi, sqrt, ulp
from mimetypes import init
from multiprocessing import dummy, reduction
from operator import is_, le, ne, truediv
from os import eventfd, lseek, minor, name, pread
from pickletools import read_uint1
from queue import PriorityQueue, Queue
from re import A, L, S, T, X
import re
from socket import NI_NUMERICSERV
from ssl import VERIFY_X509_TRUSTED_FIRST
from string import ascii_lowercase
from tabnanny import check
from tarfile import tar_filter
from textwrap import indent
import time
from tkinter import LEFT, N, NO, W, Grid
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
from pandas import isnull
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


class leetcode_4:
    # 3391. 设计一个高效的层跟踪三维二进制矩阵 (Design a 3D Binary Matrix with Efficient Layer Tracking) --plus
    class Matrix3D:

        def __init__(self, n: int):
            self.g = [[0] * (n * n) for _ in range(n)]
            self.cnts = [0] * n
            self.n = n

        def setCell(self, x: int, y: int, z: int) -> None:
            if self.g[x][y * self.n + z] == 0:
                self.g[x][y * self.n + z] = 1
                self.cnts[x] += 1

        def unsetCell(self, x: int, y: int, z: int) -> None:
            if self.g[x][y * self.n + z] == 1:
                self.g[x][y * self.n + z] = 0
                self.cnts[x] -= 1

        def largestMatrix(self) -> int:
            mx = 0
            res = self.n - 1
            for x in range(self.n - 1, -1, -1):
                if self.cnts[x] > mx:
                    mx = self.cnts[x]
                    res = x
            return res

    # 2524. 子数组的最大频率分数 (Maximum Frequency Score of a Subarray) --plus
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        d = defaultdict(int)
        res, s = 0, 0
        MOD = 10**9 + 7
        for i, x in enumerate(nums):
            if d[x]:
                s -= pow(x, d[x], MOD)
            d[x] += 1
            s += pow(x, d[x], MOD)
            if i >= k:
                y = nums[i - k]
                s -= pow(y, d[y], MOD)
                d[y] -= 1
                if d[y]:
                    s += pow(y, d[y], MOD)
            s %= MOD
            if i >= k - 1:
                res = max(res, s)
        return res

    # 3842. 灯泡开关 (Toggle Light Bulbs)
    def toggleLightBulbs(self, bulbs: list[int]) -> list[int]:
        status = [-1] * 101
        for b in bulbs:
            if status[b] == -1:
                status[b] *= -1
            else:
                status[b] ^= 1
        return [i for i, s in enumerate(status) if s > 0]

    # 3843. 频率不同的第一个元素 (First Element with Unique Frequency)
    def firstUniqueFreq(self, nums: List[int]) -> int:
        d = defaultdict(int)
        for x in nums:
            d[x] += 1
        cnts = defaultdict(int)
        for v in d.values():
            cnts[v] += 1
        for x in nums:
            if cnts[d[x]] == 1:
                return x
        return -1

    # 3845. 最大子数组异或值 (Maximum Subarray XOR with Bounded Range) --0-1字典树
    def maxXor(self, nums: list[int], k: int) -> int:
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

            def delete(self, x: int):
                node = self
                for i in range(15, -1, -1):
                    bit = x >> i & 1
                    node = node.children[bit]
                    node.cnt -= 1

            def max_xor(self, x: int) -> int:
                res = 0
                node = self
                for i in range(15, -1, -1):
                    bit = x >> i & 1
                    if node.children[bit ^ 1] and node.children[bit ^ 1].cnt:
                        res ^= 1 << i
                        bit ^= 1
                    node = node.children[bit]
                return res

        pre = list(accumulate(nums, xor, initial=0))  # type: ignore
        q_max = deque()
        q_min = deque()
        res = left = 0
        _trie = trie()
        for right, x in enumerate(nums):
            _trie.insert(pre[right])

            while q_max and x >= nums[q_max[-1]]:
                q_max.pop()
            q_max.append(right)

            while q_min and x <= nums[q_min[-1]]:
                q_min.pop()
            q_min.append(right)

            while nums[q_max[0]] - nums[q_min[0]] > k:
                _trie.delete(pre[left])

                if q_max[0] <= left:
                    q_max.popleft()
                if q_min[0] <= left:
                    q_min.popleft()
                left += 1
            res = max(res, _trie.max_xor(pre[right + 1]))
        return res

    # 3263. 将双链表转换为数组 I (Convert Doubly Linked List to Array I) --plus
    def toArray(self, root: "Optional[Node]") -> List[int]:
        res = []
        while root:
            res.append(root.val)
            root = root.next
        return res

    # 582. 杀掉进程 (Kill Process) --plus
    def killProcess(self, pid: List[int], ppid: List[int], kill: int) -> List[int]:
        def dfs(x: int):
            res.append(x)
            for y in g[x]:
                dfs(y)

        g = defaultdict(list)
        for x, fa in zip(pid, ppid):
            if fa:
                g[fa].append(x)
        res = []
        dfs(kill)
        return res

    # 190. 颠倒二进制位 (Reverse Bits)
    def reverseBits(self, n: int) -> int:
        # 没有 O(1) 的库函数，只能用字符串转换代替
        return int(bin(n)[2:].zfill(32)[::-1], 2)

    # 709. 转换成小写字母 (To Lower Case)
    def toLowerCase(self, s: str) -> str:
        return s.lower()

    # 682. 棒球比赛 (Baseball Game)
    def calPoints(self, operations: List[str]) -> int:
        st = []
        for op in operations:
            if op == "+":
                st.append(st[-1] + st[-2])
            elif op == "D":
                st.append(st[-1] * 2)
            elif op == "C":
                st.pop()
            else:
                st.append(int(op))
        return sum(st)

    # 1404. 将二进制表示减到 1 的步骤数 (Number of Steps to Reduce a Number in Binary Representation to One)
    def numSteps(self, s: str) -> int:
        n = int(s, 2)
        res = 0
        while n != 1:
            if n & 1:
                n += 1
            else:
                n >>= 1
            res += 1
        return res

    # 3209. 子数组按位与值为 K 的数目 (Number of Subarrays With AND Value of K) --plus
    def countSubarrays(self, nums: List[int], k: int) -> int:
        def cal(k: int) -> int:
            def add(x: int):
                while x:
                    lb = (x & -x).bit_length() - 1
                    cnts[lb] += 1
                    x &= x - 1

            def sub(x: int):
                while x:
                    lb = (x & -x).bit_length() - 1
                    cnts[lb] -= 1
                    x &= x - 1

            def check(l: int, r: int) -> int:
                res = 0
                for i, x in enumerate(cnts):
                    if x == r - l + 1:
                        res ^= 1 << i
                return res

            cnts = [0] * 30
            left = res = 0
            for right, x in enumerate(nums):
                add(x)
                while check(left, right) < k:
                    sub(nums[left])
                    left += 1
                res += right - left + 1
            return res

        # 按位与 >= k 的子数组个数
        return cal(k) - cal(k + 1)

    # 401. 二进制手表 (Binary Watch)
    def readBinaryWatch(self, turnedOn: int) -> List[str]:
        res = []
        for i in range(1 << 10):
            if i.bit_count() == turnedOn:
                h = i >> 6
                m = i & ((1 << 6) - 1)
                if h >= 12 or m >= 60:
                    continue
                res.append(f"{h}:{m:02d}")
        return res

    # 480. 滑动窗口中位数 (Sliding Window Median)
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        res = []
        sl = SortedList()
        for i, x in enumerate(nums):
            sl.add(x)
            if i >= k:
                sl.discard(nums[i - k])
            if i >= k - 1:
                if k & 1:
                    res.append(sl[k // 2])
                else:
                    s = sl[k // 2] + sl[k // 2 - 1]
                    res.append(s / 2)
        return res


# 最近公共祖先（LCA）、倍增算法 边权为1 LCA 模板（节点编号从 0 开始）：
class LcaBinaryLifting:
    def __init__(self, edges: List[List[int]]):
        n = len(edges) + 1
        m = n.bit_length()
        g = [[] for _ in range(n)]
        for x, y in edges:
            # 如果题目的节点编号从 1 开始，改成 x-1 和 y-1
            g[x - 1].append(y - 1)
            g[y - 1].append(x - 1)

        depth = [0] * n
        pa = [[-1] * n for _ in range(m)]

        def dfs(x: int, fa: int) -> None:
            pa[0][x] = fa
            for y in g[x]:
                if y != fa:
                    depth[y] = depth[x] + 1
                    dfs(y, x)

        dfs(0, -1)

        for i in range(m - 1):
            for x in range(n):
                if (p := pa[i][x]) != -1:
                    pa[i + 1][x] = pa[i][p]

        self.depth = depth
        self.pa = pa

    # 返回 node 的第 k 个祖先节点
    # 如果不存在，返回 -1
    def get_kth_ancestor(self, node: int, k: int) -> int:
        pa = self.pa
        for i in range(k.bit_length()):
            if k >> i & 1:
                node = pa[i][node]
                if node < 0:
                    return -1
        return node

    # 返回 x 和 y 的最近公共祖先
    def get_lca(self, x: int, y: int) -> int:
        if self.depth[x] > self.depth[y]:
            x, y = y, x
        # 使 y 和 x 在同一深度
        y = self.get_kth_ancestor(y, self.depth[y] - self.depth[x])
        if y == x:
            return x
        pa = self.pa
        for i in range(len(pa) - 1, -1, -1):
            px, py = pa[i][x], pa[i][y]
            if px != py:
                x, y = px, py  # 同时往上跳 2**i 步
        return pa[0][x]

    # 返回 x 到 y 的距离（最短路长度）
    def get_dis(self, x: int, y: int) -> int:
        return self.depth[x] + self.depth[y] - self.depth[self.get_lca(x, y)] * 2

    # 3559. 给边赋权值的方案数 II (Number of Ways to Assign Edge Weights II) --LCA 最近公共祖先
    def assignEdgeWeights(
        self, edges: List[List[int]], queries: List[List[int]]
    ) -> List[int]:
        lca = self.LcaBinaryLifting(edges)
        MOD = 10**9 + 7
        n = len(edges) + 1
        pow2 = [0] * n
        pow2[0] = 1
        for i in range(1, n):
            pow2[i] = (pow2[i - 1] * 2) % MOD
        res = [0] * len(queries)
        for i, (u, v) in enumerate(queries):
            d = lca.get_dis(u - 1, v - 1)
            if d:
                res[i] = pow2[d - 1]
        return res


# 最近公共祖先（LCA）、倍增算法 带权树 LCA 模板（节点编号从 0 开始）：
class LcaBinaryLifting:
    def __init__(self, edges: List[List[int]]):
        n = len(edges) + 1
        m = n.bit_length()
        g = [[] for _ in range(n)]
        for x, y, w in edges:
            # 如果题目的节点编号从 1 开始，改成 x-1 和 y-1
            g[x].append((y, w))
            g[y].append((x, w))

        depth = [0] * n
        dis = [0] * n  # 如果是无权树（边权为 1），dis 可以去掉，用 depth 代替
        pa = [[-1] * n for _ in range(m)]

        def dfs(x: int, fa: int) -> None:
            pa[0][x] = fa
            for y, w in g[x]:
                if y != fa:
                    depth[y] = depth[x] + 1
                    dis[y] = dis[x] + w
                    dfs(y, x)

        dfs(0, -1)

        for i in range(m - 1):
            for x in range(n):
                if (p := pa[i][x]) != -1:
                    pa[i + 1][x] = pa[i][p]

        self.depth = depth
        self.dis = dis
        self.pa = pa

    # 返回 node 的第 k 个祖先节点
    # 如果不存在，返回 -1
    def get_kth_ancestor(self, node: int, k: int) -> int:
        pa = self.pa
        for i in range(k.bit_length()):
            if k >> i & 1:
                node = pa[i][node]
                if node < 0:
                    return -1
        return node

    # 返回 x 和 y 的最近公共祖先
    def get_lca(self, x: int, y: int) -> int:
        if self.depth[x] > self.depth[y]:
            x, y = y, x
        # 使 y 和 x 在同一深度
        y = self.get_kth_ancestor(y, self.depth[y] - self.depth[x])
        if y == x:
            return x
        pa = self.pa
        for i in range(len(pa) - 1, -1, -1):
            px, py = pa[i][x], pa[i][y]
            if px != py:
                x, y = px, py  # 同时往上跳 2**i 步
        return pa[0][x]

    # 返回 x 到 y 的距离（最短路长度）
    def get_dis(self, x: int, y: int) -> int:
        return self.dis[x] + self.dis[y] - self.dis[self.get_lca(x, y)] * 2

    # 3553. 包含要求路径的最小带权子图 II (Minimum Weighted Subgraph With the Required Paths II) --LCA 最近公共祖先
    def minimumWeight(
        self, edges: List[List[int]], queries: List[List[int]]
    ) -> List[int]:
        lca = self.LcaBinaryLifting(edges)
        return [
            (lca.get_dis(a, b) + lca.get_dis(b, c) + lca.get_dis(c, a)) // 2
            for a, b, c in queries
        ]

    # 696. 计数二进制子串 (Count Binary Substrings)
    def countBinarySubstrings(self, s: str) -> int:
        n = len(s)
        res, j = 0, -1
        pre = 0
        for i, x in enumerate(s):
            if i == n - 1 or x != s[i + 1]:
                c = i - j
                res += min(c, pre)
                pre = c
                j = i
        return res

    # 125. 验证回文串 (Valid Palindrome)
    # LCR 018. 验证回文串
    def isPalindrome(self, s: str) -> bool:
        i, j = 0, len(s) - 1
        while i <= j:
            while i <= j and not s[i].isalnum():
                i += 1
            while i <= j and not s[j].isalnum():
                j -= 1
            if i <= j:
                if not (
                    s[i] == s[j]
                    or s[i].isalpha()
                    and s[j].isalpha
                    and s[i].lower() == s[j].lower()
                ):
                    return False
                i += 1
                j -= 1
        return True

    # 205. 同构字符串 (Isomorphic Strings)
    def isIsomorphic(self, s: str, t: str) -> bool:
        d1 = defaultdict(str)
        d2 = defaultdict(str)
        for a, b in zip(s, t):
            if d1[a] and d1[a] != b or d2[b] and d2[b] != a:
                return False
            d1[a] = b
            d2[b] = a

        return True

    # 423. 从英文中重建数字 (Reconstruct Original Digits from English)
    def originalDigits(self, s: str) -> str:
        cnts = [0] * 26
        for c in s:
            cnts[ord(c) - ord("a")] += 1
        a = []
        for num, d in (
            ("zero", "0"),
            ("wto", "2"),
            ("ufor", "4"),
            ("fvie", "5"),
            ("xsi", "6"),
            ("vseen", "7"),
            ("ghtei", "8"),
            ("one", "1"),
            ("three", "3"),
            ("inne", "9"),
        ):
            c = cnts[ord(num[0]) - ord("a")]
            for x in num:
                cnts[ord(x) - ord("a")] -= c
            a.extend(d * c)
        return "".join(sorted(a))

    # 面试题 17.07. 婴儿名字 (Baby Names LCCI)
    def trulyMostPopular(self, names: List[str], synonyms: List[str]) -> List[str]:
        class union:
            def __init__(self, n: int):
                self.parent = [i for i in range(n)]
                self.rank = [1] * n

            def get_root(self, p: int) -> int:
                if self.parent[p] == p:
                    return p
                self.parent[p] = self.get_root(self.parent[p])
                return self.parent[p]

            def union(self, p1: int, p2: int):
                r1 = self.get_root(p1)
                r2 = self.get_root(p2)
                if r1 < r2:
                    self.parent[r1] = r2
                else:
                    self.parent[r2] = r1
                    if self.rank[r1] == self.rank[r2]:
                        self.rank[r2] += 1

        id = 0
        name_cnt = defaultdict(int)
        name_id = defaultdict(int)
        for s in names:
            p = s.find("(")
            name = s[:p]
            cnt = int(s[p + 1 : -1])
            if name not in name_id:
                name_id[name] = id
                id += 1
                name_cnt[name] = cnt
        for syn in synonyms:
            p = syn.index(",")
            a = syn[1:p]
            b = syn[p + 1 : -1]
            if a not in name_id:
                name_id[a] = id
                id += 1
            if b not in name_id:
                name_id[b] = id
                id += 1
        u = union(id)
        for syn in synonyms:
            p = syn.find(",")
            a = syn[1:p]
            b = syn[p + 1 : -1]
            u.union(name_id[a], name_id[b])
        root_names = defaultdict(list)
        for name, id in name_id.items():
            r = u.get_root(id)
            root_names[r].append(name)
        res = []
        for names in root_names.values():
            names.sort()
            cnt = sum(name_cnt[name] for name in names)
            res.append(names[0] + "(" + str(cnt) + ")")
        return res

    # 2102. 序列顺序查询 (Sequentially Ordinal Rank Tracker)
    class SORTracker:

        def __init__(self):
            self.sl = SortedList()
            self.cnt = 0

        def add(self, name: str, score: int) -> None:
            self.sl.add(((-score, name)))

        def get(self) -> str:
            self.cnt += 1
            return self.sl[self.cnt - 1][1]

    # 761. 特殊的二进制字符串 (Special Binary String)
    def makeLargestSpecial(self, s: str) -> str:
        if len(s) <= 2:
            return s
        left = 0
        diff = 0
        res = []
        for i, x in enumerate(s):
            if x == "1":
                diff += 1
            else:
                diff -= 1
                if diff == 0:
                    res.append("1" + self.makeLargestSpecial(s[left + 1 : i]) + "0")
                    left = i + 1
        res.sort(reverse=True)
        return "".join(res)

    # 669. 修剪二叉搜索树 (Trim a Binary Search Tree)
    def trimBST(
        self, root: Optional[TreeNode], low: int, high: int
    ) -> Optional[TreeNode]:
        if root is None:
            return None
        if root.val < low:
            return self.trimBST(root.right, low, high)
        if root.val > high:
            return self.trimBST(root.left, low, high)
        root.left = self.trimBST(root.left, low, high)
        root.right = self.trimBST(root.right, low, high)
        return root

    # 2038. 如果相邻两个颜色均相同则删除当前颜色 (Remove Colored Pieces if Both Neighbors are the Same Color)
    def winnerOfGame(self, colors: str) -> bool:
        cnts = [0] * 2
        c = 0
        for i, x in enumerate(colors):
            c += 1
            if i == len(colors) - 1 or x != colors[i + 1]:
                cnts[ord(x) - ord("A")] += max(c - 2, 0)
                c = 0
        return cnts[0] > cnts[1]

    # 1945. 字符串转化后的各位数字之和 (Sum of Digits of String After Convert)
    def getLucky(self, s: str, k: int) -> int:
        def _sum(x: int) -> int:
            s = 0
            while x:
                s += x % 10
                x //= 10
            return s

        x = sum(_sum(ord(c) - ord("a") + 1) for c in s)
        k -= 1
        while k and x >= 10:
            x = _sum(x)
            k -= 1
        return x

    # 面试题 17.20. 连续中值 (Continuous Median LCCI)
    class MedianFinder:

        def __init__(self):
            """
            initialize your data structure here.
            """
            self.sl = SortedList()

        def addNum(self, num: int) -> None:
            self.sl.add(num)

        def findMedian(self) -> float:
            n = len(self.sl)
            if n & 1:
                return self.sl[n // 2]
            return (self.sl[n // 2 - 1] + self.sl[n // 2]) / 2

    # 2271. 毯子覆盖的最多白色砖块数 (Maximum White Tiles Covered by a Carpet)
    def maximumWhiteTiles(self, tiles: List[List[int]], carpetLen: int) -> int:
        tiles.sort()
        res, left = 0, 0
        s = 0
        for right, (start, end) in enumerate(tiles):
            s += end - start + 1
            while tiles[left][1] < tiles[right][1] - carpetLen + 1:
                s -= tiles[left][1] - tiles[left][0] + 1
                left += 1
            extra = (
                tiles[left][1]
                - max(tiles[right][1] - carpetLen + 1, tiles[left][0])
                + 1
            )
            res = max(res, s - (tiles[left][1] - tiles[left][0] + 1) + extra)
        return res

    # 189. 轮转数组 (Rotate Array)
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k %= n
        if k == 0:
            return
        a = nums[-k:] + nums[: n - k]
        for i, x in enumerate(a):
            nums[i] = x

    # 675. 为高尔夫比赛砍树 (Cut Off Trees for Golf Event)
    def cutOffTree(self, forest: List[List[int]]) -> int:
        def bfs(x0: int, y0: int, x1: int, y1: int) -> int:
            vis = [[False] * n for _ in range(m)]
            vis[x0][y0] = True
            q = deque()
            q.append((x0, y0, 0))
            while q:
                x, y, d = q.popleft()
                if x == x1 and y == y1:
                    return d
                for dx, dy in (0, 1), (0, -1), (1, 0), (-1, 0):
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < m
                        and 0 <= ny < n
                        and forest[nx][ny] != 0
                        and not vis[nx][ny]
                    ):
                        vis[nx][ny] = True
                        q.append((nx, ny, d + 1))
            return inf

        a = []
        m, n = len(forest), len(forest[0])
        for i in range(m):
            for j in range(n):
                if forest[i][j] > 1:
                    a.append((forest[i][j], i, j))
        # 从 (0, 0) 出发
        a.append((0, 0, 0))
        a.sort()
        res = 0
        for (_, x0, y0), (_, x1, y1) in pairwise(a):
            d = bfs(x0, y0, x1, y1)
            if d == inf:
                return -1
            res += d
        return res

    # 1156. 单字符重复子串的最大长度 (Swap For Longest Repeated Character Substring)
    def maxRepOpt1(self, text: str) -> int:
        def cal(x: int, cnt: int) -> int:
            a = []
            i = 0
            n = len(text)
            while i < n:
                id = ord(text[i]) - ord("a")
                if id != x:
                    i += 1
                    continue
                j = i
                while j < n and text[j] == text[i]:
                    j += 1
                a.append((i, j - 1))
                i = j
            res = 0
            for (s0, t0), (s1, t1) in pairwise(a):
                if s1 - t0 == 2:
                    if cnt == t0 - s0 + 1 + t1 - s1 + 1:
                        return cnt
                    res = max(res, t0 - s0 + 1 + t1 - s1 + 1 + 1)
                else:
                    res = max(res, t1 - s1 + 1 + 1, t0 - s0 + 1 + 1)
            return res

        cnts = [0] * 26
        res, c = 0, 0
        for i, x in enumerate(text):
            cnts[ord(x) - ord("a")] += 1
            c += 1
            if i == len(text) - 1 or x != text[i + 1]:
                res = max(res, c)
                c = 0
        for i, cnt in enumerate(cnts):
            if cnt:
                res = max(res, cal(i, cnt))
        return res

    # 168. Excel 表列名称 (Excel Sheet Column Title)
    def convertToTitle(self, columnNumber: int) -> str:
        res = []
        while columnNumber:
            columnNumber -= 1
            d, m = divmod(columnNumber, 26)
            res.append(chr(m + ord("A")))
            columnNumber = d
        return "".join(res[::-1])

    # 762. 二进制表示中质数个计算置位 (Prime Number of Set Bits in Binary Representation)
    def countPrimeSetBits(self, left: int, right: int) -> int:
        def cal(x: int) -> int:
            @cache
            def dfs(i: int, j: int, is_limit: bool) -> int:
                def is_prime(x: int) -> bool:
                    for m in range(2, isqrt(x) + 1):
                        if x % m == 0:
                            return False
                    return x > 1

                if i == n:
                    return is_prime(j)
                res = 0
                up = int(s[i]) if is_limit else 1
                for d in range(up + 1):
                    res += dfs(i + 1, j + d, is_limit and up == d)
                return res

            s = bin(x)[2:]
            n = len(s)
            return dfs(0, 0, True)

        return cal(right) - cal(left - 1)

    # 762. 二进制表示中质数个计算置位 (Prime Number of Set Bits in Binary Representation)
    def countPrimeSetBits(self, left: int, right: int) -> int:
        def is_prime(x: int) -> bool:
            for m in range(2, isqrt(x) + 1):
                if x % m == 0:
                    return False
            return x > 1

        res = 0
        for x in range(left, right + 1):
            if is_prime(x.bit_count()):
                res += 1
        return res

    # 989. 数组形式的整数加法 (Add to Array-Form of Integer)
    def addToArrayForm(self, num: List[int], k: int) -> List[int]:
        i = len(num) - 1
        carry = 0
        res = []
        while i >= 0 or k or carry:
            carry += (k % 10) + (num[i] if i >= 0 else 0)
            k //= 10
            i -= 1
            res.append(carry % 10)
            carry //= 10
        return res[::-1]

    # 1991. 找到数组的中间位置 (Find the Middle Index in Array)
    def findMiddleIndex(self, nums: List[int]) -> int:
        s = sum(nums)
        left = 0
        for i, x in enumerate(nums):
            s -= x
            if s == left:
                return i
            left += x
        return -1

    # 1002. 查找共用字符 (Find Common Characters)
    def commonChars(self, words: List[str]) -> List[str]:
        cnts = [inf] * 26
        for word in words:
            cur_cnts = [0] * 26
            for w in word:
                cur_cnts[ord(w) - ord("a")] += 1
            for i, c in enumerate(cur_cnts):
                cnts[i] = min(cnts[i], c)
        res = []
        for i, c in enumerate(cnts):
            x = chr(ord("a") + i)
            res.extend(c * x)
        return res

    # 1030. 距离顺序排列矩阵单元格 (Matrix Cells in Distance Order)
    def allCellsDistOrder(
        self, rows: int, cols: int, rCenter: int, cCenter: int
    ) -> List[List[int]]:
        a = []
        for i in range(rows):
            for j in range(cols):
                d = abs(i - rCenter) + abs(j - cCenter)
                a.append((d, i, j))
        a.sort()
        return [[i, j] for _, i, j in a]

    # 116. 填充每个节点的下一个右侧节点指针 (Populating Next Right Pointers in Each Node)
    def connect(self, root: "Optional[Node]") -> "Optional[Node]":
        def dfs(x: "Optional[Node]", depth: int):
            if x is None:
                return
            if d[depth]:
                d[depth][-1].next = x
            d[depth].append(x)
            dfs(x.left, depth + 1)
            dfs(x.right, depth + 1)

        d = defaultdict(list)
        dfs(root, 0)
        return root

    # 2078. 两栋颜色不同且距离最远的房子 (Two Furthest Houses With Different Colors)
    def maxDistance(self, colors: List[int]) -> int:
        n = len(colors)
        res = 0
        for i in range(n - 1, 0, -1):
            if colors[0] != colors[i]:
                res = i
                break
        for i in range(n - 1):
            if colors[i] != colors[-1]:
                res = max(res, n - 1 - i)
                break
        return res

    # 2265. 统计值等于子树平均值的节点数 (Count Nodes Equal to Average of Subtree)
    def averageOfSubtree(self, root: TreeNode) -> int:
        def dfs(root: TreeNode) -> tuple:
            if root is None:
                return (0, 0)
            s0, c0 = dfs(root.left)
            s1, c1 = dfs(root.right)
            s = s0 + s1 + root.val
            x = c0 + c1 + 1
            if root.val == s // x:
                nonlocal res
                res += 1
            return (s, x)

        res = 0
        dfs(root)
        return res

    # 433. 最小基因变化 (Minimum Genetic Mutation)
    def minMutation(self, startGene: str, endGene: str, bank: List[str]) -> int:
        if startGene == endGene:
            return 0
        s = set(bank)
        if endGene not in s:
            return -1
        s.discard(startGene)
        vis = set()
        q = deque()
        q.append((startGene, 0))
        while q:
            x, d = q.popleft()
            if x == endGene:
                return d
            for i, c in enumerate(x):
                for j in ("A", "C", "G", "T"):
                    if j != c:
                        cur = x[:i] + j + x[i + 1 :]
                        if cur in s and cur not in vis:
                            vis.add(cur)
                            q.append((cur, d + 1))
        return -1

    # 面试题 17.19. 消失的两个数字 (Missing Two LCCI)
    def missingTwo(self, nums: List[int]) -> List[int]:
        n = len(nums)
        xor = (n + 1) ^ (n + 2)
        for i, x in enumerate(nums, start=1):
            xor ^= i ^ x
        lb = (xor & -xor).bit_length() - 1
        res = [0] * 2
        res[(n + 1) >> lb & 1] ^= n + 1
        res[(n + 2) >> lb & 1] ^= n + 2
        for i, x in enumerate(nums, start=1):
            res[x >> lb & 1] ^= x
            res[i >> lb & 1] ^= i
        return res

    # 1288. 删除被覆盖区间 (Remove Covered Intervals)
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda o: (o[0], -o[1]))
        n = len(intervals)
        res = n
        right_max = intervals[0][1]
        for i in range(1, n):
            if intervals[i][1] <= right_max:
                res -= 1
            right_max = max(right_max, intervals[i][1])
        return res

    # 1816. 截断句子 (Truncate Sentence)
    def truncateSentence(self, s: str, k: int) -> str:
        a = s.split(" ")
        return " ".join(a[:k])

    # 1980. 找出不同的二进制字符串 (Find Unique Binary String)
    def findDifferentBinaryString(self, nums: List[str]) -> str:
        return "".join(str(int(x[i]) ^ 1) for i, x in enumerate(nums))

    # 1893. 检查是否区域内所有整数都被覆盖 (Check if All the Integers in a Range Are Covered)
    def isCovered(self, ranges: List[List[int]], left: int, right: int) -> bool:
        ranges.sort()
        i, n = 0, len(ranges)
        while i < n:
            l = ranges[i][0]
            r = ranges[i][1]
            j = i
            while j < n and ranges[j][0] <= r + 1:
                r = max(r, ranges[j][1])
                j += 1
            if l <= left <= right <= r:
                return True
            i = j
        return False

    # 1893. 检查是否区域内所有整数都被覆盖 (Check if All the Integers in a Range Are Covered)
    def isCovered(self, ranges: List[List[int]], left: int, right: int) -> bool:
        diff = [0] * (right + 1)
        for l, r in ranges:
            if l < right + 1:
                diff[l] += 1
            if r + 1 < right + 1:
                diff[r + 1] -= 1
        c = 0
        for i, d in enumerate(diff):
            c += d
            if left <= i <= right and c <= 0:
                return False
        return True

    # 1893. 检查是否区域内所有整数都被覆盖 (Check if All the Integers in a Range Are Covered)
    def isCovered(self, ranges: List[List[int]], left: int, right: int) -> bool:
        m = 0
        for l, r in ranges:
            m |= ((1 << (r + 1)) - 1) ^ ((1 << l) - 1)
        r = ((1 << (right + 1)) - 1) ^ ((1 << left) - 1)
        return m & r == r

    # 2194. Excel 表中某个范围内的单元格 (Cells in a Range on an Excel Sheet)
    def cellsInRange(self, s: str) -> List[str]:
        res = []
        x0 = ord(s[0]) - ord("A")
        x1 = ord(s[3]) - ord("A")
        y0 = int(s[1])
        y1 = int(s[-1])
        for i in range(x0, x1 + 1):
            s0 = chr(i + ord("A"))
            for j in range(y0, y1 + 1):
                s1 = str(j)
                res.append(s0 + s1)
        return res

    # 424. 替换后的最长重复字符 (Longest Repeating Character Replacement)
    def characterReplacement(self, s: str, k: int) -> int:
        res, left = 0, 0
        cnts = [0] * 26
        mx = 0
        for right, x in enumerate(s):
            id = ord(x) - ord("A")
            cnts[id] += 1
            mx = max(mx, cnts[id])
            while right - left + 1 - mx > k:
                left_id = ord(s[left]) - ord("A")
                cnts[left_id] -= 1
                left += 1
            res = max(res, right - left + 1)
        return res

    # 面试题 01.02. 判定是否互为字符重排 (Check Permutation LCCI)
    def CheckPermutation(self, s1: str, s2: str) -> bool:
        return Counter(s1) == Counter(s2)

    # 3847. 计算比赛分数差 (Find the Score Difference in a Game)
    def scoreDifference(self, nums: List[int]) -> int:
        res = [0] * 2
        id = 0
        for i, x in enumerate(nums):
            id ^= x & 1
            if (i + 1) % 6 == 0:
                id ^= 1
            res[id] += x
        return res[0] - res[1]

    # 3848. 阶数数字排列 (Check Digitorial Permutation)
    def isDigitorialPermutation(self, n: int) -> bool:
        mul = [1] * 10
        for i in range(2, 10):
            mul[i] = mul[i - 1] * i
        cnts = [0] * 10
        s = 0
        while n:
            n, m = divmod(n, 10)
            s += mul[m]
            cnts[m] += 1
        while s:
            s, m = divmod(s, 10)
            cnts[m] -= 1
        return cnts == [0] * 10

    # 3849. 重新排列后的最大按位异或值 (Maximum Bitwise XOR After Rearrangement)
    def maximumXor(self, s: str, t: str) -> str:
        cnts = [0] * 2
        for c in t:
            cnts[int(c)] += 1
        res = []
        for c in s:
            x = int(c)
            if cnts[x ^ 1]:
                res.append("1")
                cnts[x ^ 1] -= 1
            else:
                res.append("0")
                cnts[x] -= 1
        return "".join(res)

    # 3850. 统计结果等于 K 的序列数目 (Count Sequences to K)
    def countSequences(self, nums: List[int], k: int) -> int:
        @cache
        def dfs(i: int, j0: int, j1: int) -> int:
            if i == n:
                return j1 * k == j0
            if j0 * suf[i] < j1 * k:
                return 0
            return (
                dfs(i + 1, j0, j1)
                + dfs(i + 1, j0 * nums[i], j1)
                + dfs(i + 1, j0, j1 * nums[i])
            )

        n = len(nums)
        suf = [0] * n
        suf[-1] = nums[-1]
        for i in range(n - 2, -1, -1):
            suf[i] = suf[i + 1] * nums[i]
        return dfs(0, 1, 1)

    # 1461. 检查一个字符串是否包含所有长度为 K 的二进制子串 (Check If a String Contains All Binary Codes of Size K)
    def hasAllCodes(self, s: str, k: int) -> bool:
        if len(s) - k + 1 < 1 << k:
            return False
        _set = set()
        u = (1 << k) - 1
        v = 0
        for i, x in enumerate(s):
            v = (v << 1) ^ int(x)
            v &= u
            if i >= k - 1:
                _set.add(v)
        return len(_set) == 1 << k

    # 2357. 使数组中所有元素都等于零 (Make Array Zero by Subtracting Equal Amounts)
    def minimumOperations(self, nums: List[int]) -> int:
        return len(set(nums) - {0})

    # 2531. 使字符串中不同字符的数目相等 (Make Number of Distinct Characters Equal)
    def isItPossible(self, word1: str, word2: str) -> bool:
        cnt1 = [0] * 26
        cnt2 = [0] * 26
        c1 = 0
        c2 = 0
        for c in word1:
            cnt1[ord(c) - ord("a")] += 1
            if cnt1[ord(c) - ord("a")] == 1:
                c1 += 1
        for c in word2:
            cnt2[ord(c) - ord("a")] += 1
            if cnt2[ord(c) - ord("a")] == 1:
                c2 += 1
        for i in range(26):
            if cnt1[i] == 0:
                continue
            for j in range(26):
                if cnt2[j] == 0:
                    continue
                # 交换
                cnt2[i] += 1
                cnt1[i] -= 1
                if cnt2[i] == 1:
                    c2 += 1
                if cnt1[i] == 0:
                    c1 -= 1

                cnt1[j] += 1
                cnt2[j] -= 1
                if cnt1[j] == 1:
                    c1 += 1
                if cnt2[j] == 0:
                    c2 -= 1
                if c1 == c2:
                    return True

                # 还原
                cnt2[i] -= 1
                cnt1[i] += 1
                if cnt2[i] == 0:
                    c2 -= 1
                if cnt1[i] == 1:
                    c1 += 1

                cnt1[j] -= 1
                cnt2[j] += 1
                if cnt1[j] == 0:
                    c1 -= 1
                if cnt2[j] == 1:
                    c2 += 1
        return False

    # 1239. 串联字符串的最大长度 (Maximum Length of a Concatenated String with Unique Characters)
    def maxLength(self, arr: List[str]) -> int:
        @cache  # 当数据范围较小时，去掉记忆化，速度更快 （也就是回溯）
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            res = dfs(i + 1, j)
            if a[i] & j == 0:
                res = max(res, dfs(i + 1, j | a[i]) + a[i].bit_count())
            return res

        a = []
        for s in arr:
            m = 0
            for x in s:
                if m >> (ord(x) - ord("a")) & 1:
                    m = -1
                    break
                m |= 1 << (ord(x) - ord("a"))
            if m != -1:
                a.append(m)
        n = len(a)
        return dfs(0, 0)

    # 1346. 检查整数及其两倍数是否存在 (Check If N and Its Double Exist)
    def checkIfExist(self, arr: List[int]) -> bool:
        s = set()
        for x in arr:
            if x * 2 in s or x & 1 == 0 and x // 2 in s:
                return True
            s.add(x)
        return False

    # 668. 乘法表中第k小的数 (Kth Smallest Number in Multiplication Table)
    def findKthNumber(self, m: int, n: int, k: int) -> int:
        def check(x: int) -> int:
            i, j = 0, n - 1
            cnt = 0
            while i < m and j >= 0:
                if (i + 1) * (j + 1) <= x:
                    cnt += j + 1
                    i += 1
                else:
                    j -= 1
            return cnt

        left = 1
        right = m * n
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid) >= k:
                right = mid - 1
            else:
                left = mid + 1
        return right + 1

    # 2467. 树上最大得分和路径 (Most Profitable Path in a Tree)
    def mostProfitablePath(
        self, edges: List[List[int]], bob: int, amount: List[int]
    ) -> int:
        def dfs_from_bob(x: int, fa: int, t: int) -> bool:
            if x == 0:
                d[x] = t
                return True
            for y in g[x]:
                if y != fa:
                    if dfs_from_bob(y, x, t + 1):
                        d[x] = t
                        return True
            return False

        def dfs_from_alice(x: int, fa: int, t: int) -> int:
            s = 0
            if t < d[x]:
                s += amount[x]
            elif t == d[x]:
                s += amount[x] // 2
            mx = -inf
            for y in g[x]:
                if y != fa:
                    mx = max(mx, dfs_from_alice(y, x, t + 1))
            return s + (mx if mx > -inf else 0)

        n = len(amount)
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        # bob节点到0节点的距离
        d = [inf] * n
        dfs_from_bob(bob, -1, 0)
        return dfs_from_alice(0, -1, 0)

    # 1022. 从根到叶的二进制数之和 (Sum of Root To Leaf Binary Numbers)
    def sumRootToLeaf(self, root: Optional[TreeNode]) -> int:
        def dfs(node: Optional[TreeNode], x: int) -> int:
            if node is None:
                return 0
            x = (x << 1) | node.val
            if node.left is None and node.right is None:
                return x
            return dfs(node.left, x) + dfs(node.right, x)

        return dfs(root, 0)

    # 2437. 有效时间的数目 (Number of Valid Clock Times)
    def countTime(self, time: str) -> int:
        def check(s: str) -> bool:
            for a, b in zip(s, time):
                if b == "?":
                    continue
                if a != b:
                    return False
            return True

        res = 0
        for h in range(24):
            for m in range(60):
                if check(f"{h:02d}:{m:02d}"):
                    res += 1
        return res

    # 771. 宝石与石头 (Jewels and Stones)
    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        m = [0] * 2
        for x in jewels:
            m[ord(x) >> 5 & 1] |= 1 << (ord(x) & 31)
        return sum(m[ord(x) >> 5 & 1] >> (ord(x) & 31) & 1 for x in stones)

    # 1125. 最小的必要团队 (Smallest Sufficient Team)
    def smallestSufficientTeam(
        self, req_skills: List[str], people: List[List[str]]
    ) -> List[int]:
        @cache
        def dfs(i: int, j: int) -> int:
            if j == u:
                return 0
            if i == n:
                return inf
            return min(dfs(i + 1, j), dfs(i + 1, j | a[i]) + 1)

        def make_ans(i: int, j: int, s: int):
            if j == u:
                return
            if dfs(i + 1, j) == s:
                make_ans(i + 1, j, s)
                return
            res.append(i)
            make_ans(i + 1, j | a[i], s - 1)

        d = defaultdict(int)
        for i, x in enumerate(req_skills):
            d[x] = i
        u = (1 << len(req_skills)) - 1
        a = []
        for p in people:
            m = 0
            for x in p:
                m |= 1 << d[x]
            a.append(m)
        n = len(a)
        _min = dfs(0, 0)
        res = []
        make_ans(0, 0, _min)
        return res

    # 434. 字符串中的单词数 (Number of Segments in a String)
    def countSegments(self, s: str) -> int:
        return len(s.strip().split())

    # 766. 托普利茨矩阵 (Toeplitz Matrix)
    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        m, n = len(matrix), len(matrix[0])
        for i in range(m):
            for j in range(n):
                if i and j and matrix[i][j] != matrix[i - 1][j - 1]:
                    return False
        return True

    # 2058. 找出临界点之间的最小和最大距离 (Find the Minimum and Maximum Number of Nodes Between Critical Points)
    def nodesBetweenCriticalPoints(self, head: Optional[ListNode]) -> List[int]:
        # 上一个值
        pre_val = 0
        # 上一个合法的位置
        pre_ok_pos = -inf
        first_ok_pos = -1
        cur_pos = 0
        res = [inf, -inf]
        while head:
            if (
                pre_val
                and head.next
                and (
                    head.next.val < head.val > pre_val
                    or head.next.val > head.val < pre_val
                )
            ):
                res[0] = min(res[0], cur_pos - pre_ok_pos)
                pre_ok_pos = cur_pos
                if first_ok_pos == -1:
                    first_ok_pos = cur_pos
                res[1] = cur_pos - first_ok_pos
            pre_val = head.val
            head = head.next
            cur_pos += 1
        return [-1, -1] if res[0] == inf else res

    # 334. 递增的三元子序列 (Increasing Triplet Subsequence)
    def increasingTriplet(self, nums: List[int]) -> bool:
        n = len(nums)
        pre_min = [0] * n
        pre_min[0] = nums[0]
        for i in range(1, n):
            pre_min[i] = min(nums[i], pre_min[i - 1])
        suf_max = nums[-1]
        for i in range(n - 2, 0, -1):
            if pre_min[i - 1] < nums[i] < suf_max:
                return True
            suf_max = max(suf_max, nums[i])
        return False

    # 2145. 统计隐藏数组数目 (Count the Hidden Sequences)
    def numberOfArrays(self, differences: List[int], lower: int, upper: int) -> int:
        l = lower
        r = upper
        for d in differences:
            l += d
            r += d
            if r < lower or l > upper:
                return 0
            l = max(l, lower)
            r = min(r, upper)
        return r - l + 1

    # 1521. 找到最接近目标值的函数值 (Find a Value of a Mysterious Function Closest to Target)
    def closestToTarget(self, arr: List[int], target: int) -> int:
        def add(x: int):
            while x:
                lb = (x & -x).bit_length() - 1
                cnts[lb] += 1
                x &= x - 1

        def sub(x: int):
            while x:
                lb = (x & -x).bit_length() - 1
                cnts[lb] -= 1
                x &= x - 1

        def check(l: int, r: int) -> int:
            res = 0
            for i, c in enumerate(cnts):
                if r - l + 1 == c:
                    res ^= 1 << i
            return res

        L = 20
        cnts = [0] * L
        left = 0
        res = inf
        for right, x in enumerate(arr):
            add(x)
            while left <= right:
                cur = check(left, right)
                res = min(res, abs(cur - target))
                if cur >= target:
                    break
                sub(arr[left])
                left += 1
            if res == 0:
                break
        return res

    # 面试题 01.05. 一次编辑 (One Away LCCI)
    def oneEditAway(self, first: str, second: str) -> bool:
        @cache
        def dfs(i: int, j: int, k: bool) -> bool:
            if i == m and j == n:
                return True
            if i == m or j == n:
                return i == j and not k
            if not k and (
                dfs(i + 1, j, True) or dfs(i, j + 1, True) or dfs(i + 1, j + 1, True)
            ):
                return True
            if first[i] == second[j] and dfs(i + 1, j + 1, k):
                return True
            return False

        m, n = len(first), len(second)
        if abs(m - n) > 1:
            return False
        return dfs(0, 0, False)

    # 15. 三数之和 (3Sum)
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        res = []
        for i, x in enumerate(nums):
            if x > 0:
                break
            if i and x == nums[i - 1]:
                continue
            left = i + 1
            right = len(nums) - 1
            while left < right:
                if nums[right] + nums[left] + x > 0:
                    right -= 1
                elif nums[right] + nums[left] + x < 0:
                    left += 1
                else:
                    res.append((x, nums[left], nums[right]))
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
        return res

    # 886. 可能的二分法 (Possible Bipartition)
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        def dfs(x: int, color: int) -> bool:
            vis[x] = color
            for y in g[x]:
                if vis[y] != -1:
                    if vis[y] != color ^ 1:
                        return False
                    continue
                if not dfs(y, color ^ 1):
                    return False
            return True

        g = [[] for _ in range(n)]
        for u, v in dislikes:
            g[u - 1].append(v - 1)
            g[v - 1].append(u - 1)
        vis = [-1] * n
        for i in range(n):
            if vis[i] == -1 and not dfs(i, 0):
                return False
        return True

    # 886. 可能的二分法 (Possible Bipartition)
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        def bfs(x: int) -> bool:
            vis[x] = 0
            q = deque()
            q.append((x, 0))
            while q:
                x, color = q.popleft()
                for y in g[x]:
                    if vis[y] != -1:
                        if vis[y] != color ^ 1:
                            return False
                        continue
                    vis[y] = color ^ 1
                    q.append((y, color ^ 1))
            return True

        g = [[] for _ in range(n)]
        for u, v in dislikes:
            g[u - 1].append(v - 1)
            g[v - 1].append(u - 1)
        vis = [-1] * n
        for i in range(n):
            if vis[i] == -1 and not bfs(i):
                return False
        return True

    # 477. 汉明距离总和 (Total Hamming Distance)
    def totalHammingDistance(self, nums: List[int]) -> int:
        res = 0
        for i in range(30, -1, -1):
            cnts = [0] * 2
            for x in nums:
                cnts[x >> i & 1] += 1
            res += cnts[0] * cnts[1]
        return res

    # 1680. 连接连续二进制数字 (Concatenation of Consecutive Binary Numbers)
    def concatenatedBinary(self, n: int) -> int:
        # 把下面代码放在外面可加速
        MX = 10**5 + 1
        a = [0] * MX
        MOD = 10**9 + 7
        for i in range(1, MX):
            l = i.bit_length()
            a[i] = ((a[i - 1] << l) | i) % MOD
        return a[n]

    # LCP 01. 猜数字 (Guess Numbers)
    def game(self, guess: List[int], answer: List[int]) -> int:
        return sum(x ^ y == 0 for x, y in zip(guess, answer))

    # LCP 62. 交通枢纽
    def transportationHub(self, path: List[List[int]]) -> int:
        s = set()
        indegree = defaultdict(int)
        outdegree = defaultdict(int)
        for u, v in path:
            s.add(u)
            s.add(v)
            indegree[v] += 1
            outdegree[u] += 1
        for _, v in path:
            if indegree[v] == len(s) - 1 and v not in outdegree:
                return v
        return -1

    # 1689. 十-二进制数的最少数目 (Partitioning Into Minimum Number Of Deci-Binary Numbers)
    def minPartitions(self, n: str) -> int:
        return max(int(x) for x in n)

    # 3852. 不同频率的最小数对 (Smallest Pair With Different Frequencies)
    def minDistinctFreqPair(self, nums: list[int]) -> list[int]:
        d = defaultdict(int)
        for x in nums:
            d[x] += 1
        a = [(k, v) for k, v in d.items()]
        a.sort()
        i = 0
        n = len(a)
        while i < n:
            j = i
            while j < n and a[j][1] == a[i][1]:
                j += 1
            if j == n:
                return [-1, -1]
            return [a[i][0], a[j][0]]

    # 3853. 合并靠近字符 (Merge Close Characters)
    def mergeCharacters(self, s: str, k: int) -> str:
        res = []
        last = defaultdict(lambda: -inf)
        for x in s:
            if len(res) - last[x] > k:
                last[x] = len(res)
                res.append(x)
        return "".join(res)

    # 3854. 使数组奇偶交替的最少操作 (Minimum Operations to Make Array Parity Alternating)
    def makeParityAlternating(self, nums: List[int]) -> List[int]:
        def cal(t: int) -> List[int]:
            op = 0
            mn, mx = inf, -inf
            for i, x in enumerate(nums):
                if (x - i) & 1 != t:
                    op += 1
                    if x == _min:
                        x += 1
                    elif x == _max:
                        x -= 1
                mn = min(mn, x)
                mx = max(mx, x)
            return [op, max(mx - mn, 1)]

        n = len(nums)
        if n == 1:
            return [0, 0]
        _max, _min = max(nums), min(nums)
        return min(cal(0), cal(1))

    # 3855. 给定范围内 K 位数字之和 (Sum of K-Digit Numbers in a Range)
    def sumOfNumbers(self, l: int, r: int, k: int) -> int:
        MOD = 1_000_000_007
        m = r - l + 1
        return (
            (l + r)
            * m
            * (pow(10, k, MOD) - 1)
            * pow(18, -1, MOD)
            * pow(m, k - 1, MOD)
            % MOD
        )

    # 3856. 移除尾部元音字母 (Trim Trailing Vowels)
    def trimTrailingVowels(self, s: str) -> str:
        i = len(s) - 1
        while i >= 0 and s[i] in "aeiou":
            i -= 1
        return s[0 : i + 1]

    # 3857. 拆分到 1 的最小总代价 (Minimum Cost to Split into Ones)
    def minCost(self, n: int) -> int:
        # 把下面代码放在外面可加速
        @cache
        def dfs(n: int) -> int:
            if n == 1:
                return 0
            res = inf
            for x in range(1, n // 2 + 1):
                res = min(res, dfs(x) + dfs(n - x) + x * (n - x))
            return res

        return dfs(n)

    # 3859. 统计包含 K 个不同整数的子数组 (Count Subarrays With K Distinct Integers)
    def countSubarrays(self, nums: list[int], k: int, m: int) -> int:
        def cal(limit_k: int) -> int:
            d = defaultdict(int)
            ge_m = 0
            res = left = 0
            for x in nums:
                d[x] += 1
                if d[x] == m:
                    ge_m += 1

                while len(d) >= limit_k and ge_m >= k:
                    out = nums[left]
                    if d[out] == m:
                        ge_m -= 1
                    d[out] -= 1
                    if d[out] == 0:
                        del d[out]
                    left += 1
                res += left
            return res

        return cal(k) - cal(k + 1)

    # 1536. 排布二进制网格的最少交换次数 (Minimum Swaps to Arrange a Binary Grid)
    def minSwaps(self, grid: List[List[int]]) -> int:
        n = len(grid)
        a = []
        for i in range(n):
            for j in range(n - 1, -1, -1):
                if grid[i][j] != 0:
                    a.append(n - 1 - j)
                    break
            else:
                a.append(n)
        res = 0
        for i in range(n):
            need_zeros = n - i - 1
            for j in range(i, n):
                if a[j] >= need_zeros:
                    res += j - i
                    a[i + 1 : j + 1] = a[i:j]
                    break
            else:
                return -1
        return res

    # 1545. 找出第 N 个二进制字符串中的第 K 位 (Find Kth Bit in Nth Binary String)
    def findKthBit(self, n: int, k: int) -> str:
        def invert(x: str) -> str:
            return "1" if x == "0" else "0"

        if k == 1:
            return "0"
        mid = 1 << (n - 1)
        if mid == k:
            return "1"
        if k < mid:
            return self.findKthBit(n - 1, k)
        k = (mid << 1) - k
        return invert(self.findKthBit(n - 1, k))

    # 1582. 二进制矩阵中的特殊位置 (Special Positions in a Binary Matrix)
    def numSpecial(self, mat: List[List[int]]) -> int:
        m, n = len(mat), len(mat[0])
        row = [0] * m
        col = [0] * n
        for i in range(m):
            for j in range(n):
                row[i] += mat[i][j]
                col[j] += mat[i][j]
        res = 0
        for i in range(m):
            if row[i] == 1:
                for j in range(n):
                    if mat[i][j] and col[j] == 1:
                        res += 1
        return res

    # 1758. 生成交替二进制字符串的最少操作数 (Minimum Changes To Make Alternating Binary String)
    def minOperations(self, s: str) -> int:
        def cal(t: int) -> int:
            return sum(((i - int(s[i])) & 1) ^ t for i in range(len(s)))

        return min(cal(0), cal(1))

    # 1784. 检查二进制字符串字段 (Check if Binary String Has at Most One Segment of Ones)
    def checkOnesSegment(self, s: str) -> bool:
        return "01" not in s

    # 1888. 使二进制字符串字符交替的最少反转次数 (Minimum Number of Flips to Make the Binary String Alternating)
    def minFlips(self, s: str) -> List[int]:
        def cal_suf(t: int) -> int:
            suf = [0] * n
            suf[-1] = int(int(s[-1]) != t)
            t ^= 1
            for i in range(n - 2, -1, -1):
                if int(s[i]) != t:
                    suf[i] = suf[i + 1] + 1
                else:
                    suf[i] = suf[i + 1]
                t ^= 1
            return suf

        def cal_pre(t: int) -> int:
            pre = [0] * n
            for i, x in enumerate(s):
                if i:
                    pre[i] = pre[i - 1] + (((i - int(x)) & 1) ^ t)
                else:
                    pre[i] = ((i - int(x)) & 1) ^ t
            return pre

        n = len(s)
        # 期望末尾填1，以suf_1[i]开始的后缀变成交替，需要修改的次数
        suf_1 = cal_suf(1)
        # 期望末尾填0，以suf_0[i]开始的后缀变成交替，需要修改的次数
        suf_0 = cal_suf(0)

        # 期望首位填1，以pre_1[i]结尾的前缀变成交替，需要修改的次数
        pre_1 = cal_pre(1)
        # 期望首位填0，以pre_0[i]结尾的前缀变成交替，需要修改的次数
        pre_0 = cal_pre(0)
        res = min(suf_1[0], suf_0[0])
        if res == 0:
            return res
        for i in range(1, n):
            res = min(res, suf_1[i] + pre_0[i - 1])
            res = min(res, suf_0[i] + pre_1[i - 1])
            if res == 0:
                return res
        return res

    # 3861. 容量最小的箱子 (Minimum Capacity Box)
    def minimumIndex(self, capacity: list[int], itemSize: int) -> int:
        res = -1
        mn = inf
        for i, x in enumerate(capacity):
            if mn > x >= itemSize:
                res = i
                mn = x
        return res

    # 3862. 找出最小平衡下标 (Find the Smallest Balanced Index)
    def smallestBalancedIndex(self, nums: list[int]) -> int:
        n = len(nums)
        s = sum(nums)
        right_mul = [inf] * n
        right_mul[-1] = 1
        for i in range(n - 2, -1, -1):
            mul = nums[i + 1] * right_mul[i + 1]
            if mul > s:
                break
            right_mul[i] = mul
        left_s = 0
        for i, x in enumerate(nums):
            if left_s == right_mul[i]:
                return i
            left_s += x
        return -1

    # 3863. 将一个字符串排序的最小操作次数 (Minimum Operations to Sort a String)
    def minOperations(self, s: str) -> int:
        d = defaultdict(int)
        mn, mx = "z", "a"
        f = True
        for i, x in enumerate(s):
            d[x] += 1
            mn = min(mn, x)
            mx = max(mx, x)
            if i and x < s[i - 1]:
                f = False
        # 单调不减
        if f:
            return 0
        if len(s) == 2:
            return -1
        if s[0] == mn or s[-1] == mx:
            return 1
        if s[0] == mx and s[-1] == mn and d[mn] == 1 and d[mx] == 1:
            return 3
        return 2

    # 3864. 划分二进制字符串的最小费用 (Minimum Cost to Partition a Binary String)
    def minCost(self, s: str, encCost: int, flatCost: int) -> int:
        a = [int(x) for x in s]
        n = len(s)
        a = list(accumulate(a, initial=0))

        def dfs(i: int, j: int) -> int:
            res = flatCost if a[j] - a[i] == 0 else (j - i) * (a[j] - a[i]) * encCost
            if (j - i) % 2 == 0:
                res = min(res, dfs(i, (j + i) // 2) + dfs((j + i) // 2, j))
            return res

        return dfs(0, n)

    # 1431. 拥有最多糖果的孩子 (Kids With the Greatest Number of Candies)
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        mx = max(candies)
        return [x + extraCandies >= mx for x in candies]

    # 1415. 长度为 n 的开心字符串中字典序第 k 小的字符串 (The k-th Lexicographical String of All Happy Strings of Length n)
    def getHappyString(self, n: int, k: int) -> str:
        def dfs(i: chr) -> bool:
            if len(path) == n:
                res.append("".join(path))
                return len(res) == k
            for j in "abc":
                if i != j:
                    path.append(j)
                    if dfs(j):
                        return True
                    path.pop()
            return False

        path = []
        res = []
        return res[-1] if dfs("d") else ""

    # 3866. 找到第一个唯一偶数 (First Unique Even Element)
    def firstUniqueEven(self, nums: list[int]) -> int:
        cnts = [0] * 101
        for x in nums:
            if x & 1 == 0:
                cnts[x] += 1
        for x in nums:
            if x & 1 == 0 and cnts[x] == 1:
                return x
        return -1

    # 3867. 数对的最大公约数之和 (Sum of GCD of Formed Pairs)
    def gcdSum(self, nums: list[int]) -> int:
        n = len(nums)
        mx = 0
        prefixGcd = [0] * n
        for i, x in enumerate(nums):
            mx = max(mx, x)
            prefixGcd[i] = gcd(x, mx)
        prefixGcd.sort()
        return sum(gcd(prefixGcd[i], prefixGcd[n - i - 1]) for i in range(n // 2))

    # 3868. 通过交换使数组相等的最小花费 (Minimum Cost to Equalize Arrays Using Swaps)
    def minCost(self, nums1: list[int], nums2: list[int]) -> int:
        cnts = defaultdict(int)
        for x in nums1:
            cnts[x] += 1
        for x in nums2:
            cnts[x] += 1
        for c in cnts.values():
            if c & 1:
                return -1
        cnts1 = defaultdict(int)
        cnts2 = defaultdict(int)
        for x in nums1:
            cnts1[x] += 1
        for x in nums2:
            cnts2[x] += 1
        a = 0
        b = 0
        for k, v in cnts1.items():
            mn = min(v, cnts2[k])
            v -= mn
            cnts2[k] -= mn
            if v:
                a += v
            if cnts2[k]:
                b += cnts2[k]
            del cnts2[k]
        for c in cnts2.values():
            b += c
        res = 0
        mn = min(a, b)
        res += mn // 2
        a -= mn
        b -= mn
        res += a // 2 + b // 2
        return res

    # 3869. 统计区间内奇妙数的数目 (Count Fancy Numbers in a Range)
    def countFancy(self, l: int, r: int) -> int:
        def check_ok(x: int) -> bool:
            _s = str(x)
            return all(a < b for a, b in pairwise(_s)) or all(
                a > b for a, b in pairwise(_s)
            )

        MX = 15 * 9 + 1
        # s_ok[i] 表示 i 的各位和，是否为严格递增或严格递减的
        s_ok = [False] * MX
        for i in range(MX):
            s_ok[i] = check_ok(i)

        def cal(x: int) -> int:
            @cache
            def dfs(
                i: int, j: int, last: int, inc: int, is_limit: bool, is_num: bool
            ) -> int:
                if i == n:
                    return is_num and (inc != 0 or s_ok[j])
                res = 0
                if not is_num:
                    res = dfs(i + 1, j, last, inc, False, False)
                up = int(s[i]) if is_limit else 9
                for d in range(0 if is_num else 1, up + 1):
                    # 之前未填过数
                    if not is_num:
                        res += dfs(i + 1, j + d, d, inc, is_limit and up == d, True)
                    # 之前填了1个数
                    elif inc == 2:
                        nxt_inc = 1 if d > last else (-1 if d < last else 0)
                        res += dfs(i + 1, j + d, d, nxt_inc, is_limit and up == d, True)
                    else:
                        nxt_inc = 0
                        if d > last and inc == 1:
                            nxt_inc = 1
                        elif d < last and inc == -1:
                            nxt_inc = -1
                        res += dfs(i + 1, j + d, d, nxt_inc, is_limit and up == d, True)
                return res

            s = str(x)
            n = len(s)
            # 第i位, 数位和j, 上一位数填的数last, 此数字严格递增 inc == 1, 此数字严格递减 inc == -1, 此数字非严格递增或递减 inc == 0
            # dfs(i, j, last, inc, is_limit, is_num)
            return dfs(0, 0, 0, 2, True, False)

        return cal(r) - cal(l - 1)

    # 1878. 矩阵中最大的三个菱形和 (Get Biggest Three Rhombus Sums in a Grid)
    def getBiggestThree(self, grid: List[List[int]]) -> List[int]:
        m, n = len(grid), len(grid[0])
        res = [-1, -1, -1]
        for i in range(m):
            for j in range(n):
                # 枚举 (i, j) 为中心点的菱形
                mn = min(i, j, n - 1 - j, m - 1 - i)
                for d in range(mn + 1):
                    x = 0
                    if d == 0:
                        x = grid[i][j]
                    else:
                        i0, jl, jr = i - d, j, j
                        for r in range(i0, i + 1):
                            if jl != jr:
                                x += grid[r][jl] + grid[r][jr]
                            else:
                                x += grid[r][jl]
                            jl -= 1
                            jr += 1
                        i0, jl, jr = i + d, j, j
                        for r in range(i0, i, -1):
                            if jl != jr:
                                x += grid[r][jl] + grid[r][jr]
                            else:
                                x += grid[r][jl]
                            jl -= 1
                            jr += 1
                    if x in res:
                        continue
                    if x > res[0]:
                        res[2] = res[1]
                        res[1] = res[0]
                        res[0] = x
                    elif x > res[1]:
                        res[2] = res[1]
                        res[1] = x
                    elif x > res[2]:
                        res[2] = x
        while res and res[-1] == -1:
            res.pop()
        return res

    # 3870. 统计范围内的逗号 (Count Commas in Range)
    # 3871. 统计范围内的逗号 II (Count Commas in Range II)
    def countCommas(self, n: int) -> int:
        @cache
        def dfs(i: int, j: int, is_limit: bool, is_num: bool) -> int:
            if i == l:
                return is_num and (j - 1) // 3
            res = 0
            if not is_num:
                res = dfs(i + 1, j, False, False)
            up = int(s[i]) if is_limit else 9
            for d in range(0 if is_num else 1, up + 1):
                res += dfs(i + 1, j + 1, is_limit and up == d, True)
            return res

        s = str(n)
        l = len(s)
        return dfs(0, 0, True, False)

    # 3870. 统计范围内的逗号 (Count Commas in Range)
    # 3871. 统计范围内的逗号 II (Count Commas in Range II) --贡献法
    def countCommas(self, n: int) -> int:
        x = 1000
        res = 0
        while x <= n:
            res += n - x + 1
            x *= 1000
        return res

    # 1727. 重新排列后的最大子矩阵 (Largest Submatrix With Rearrangements)
    def largestSubmatrix(self, matrix: List[List[int]]) -> int:
        _, n = len(matrix), len(matrix[0])
        res = 0
        heights = [0] * n
        for row in matrix:
            for j, x in enumerate(row):
                if x:
                    heights[j] += 1
                else:
                    heights[j] = 0
            hs = sorted(heights)
            for j, x in enumerate(hs):
                res = max(res, (n - j) * x)
        return res

    # 3873. 添加一个点后可激活的最大点数 (Maximum Points Activated with One Addition)
    def maxActivated(self, points: list[list[int]]) -> int:
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
                if self.is_connected(p1, p2):
                    return
                r1 = self.get_root(p1)
                r2 = self.get_root(p2)
                if self.rank[r1] < self.rank[r2]:
                    self.parent[r1] = r2
                else:
                    self.parent[r2] = r1
                    if self.rank[r1] == self.rank[r2]:
                        self.rank[r2] += 1

        id = n = len(points)
        d1 = defaultdict(int)
        d2 = defaultdict(int)
        for x, _ in points:
            if x not in d1:
                d1[x] = id
                id += 1
        for _, y in points:
            if y not in d2:
                d2[y] = id
                id += 1
        u = union(id)
        for i, (x, y) in enumerate(points):
            u.union(i, d1[x])
            u.union(i, d2[y])
        cnts = defaultdict(int)
        for i in range(n):
            r = u.get_root(i)
            cnts[r] += 1
        res = [0, 0]
        for x in cnts.values():
            if x >= res[0]:
                res[1] = res[0]
                res[0] = x
            elif x >= res[1]:
                res[1] = x
        return res[0] + res[1] + 1

    # 1886. 判断矩阵经轮转后是否一致 (Determine Whether Matrix Can Be Obtained By Rotation)
    def findRotation(self, mat: List[List[int]], target: List[List[int]]) -> bool:
        def rotate(a: List[List[int]]):
            for i in range(n):
                for j in range(i):
                    a[i][j], a[j][i] = a[j][i], a[i][j]
            for r in a:
                r.reverse()

        n = len(mat)
        c = 0
        while c < 4:
            if all(x == y for x, y in zip(mat, target)):
                return True
            c += 1
            # 旋转90度
            rotate(mat)
        return False

    # 3876. 构造奇偶一致的数组 I (Construct Uniform Parity Array I)
    def uniformArray(self, _: list[int]) -> bool:
        return True

    # 3877. 构造奇偶一致的数组 II (Construct Uniform Parity Array II)
    def uniformArray(self, nums1: list[int]) -> bool:
        mn = [inf] * 2
        for x in nums1:
            mn[x & 1] = min(mn[x & 1], x)
        return mn[1] == inf or mn[1] < mn[0]

    # 3878. 达到目标异或值的最少删除次数 (Minimum Removals to Achieve Target XOR)
    def minRemovals(self, nums: List[int], target: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i < 0:
                return 0 if j == 0 else inf
            return min(dfs(i - 1, j) + 1, dfs(i - 1, j ^ nums[i]))

        n = len(nums)
        res = dfs(n - 1, target)
        return res if res <= n else -1

    # 69. x 的平方根 (Sqrt(x))
    def mySqrt(self, x: int) -> int:
        left = 0
        right = x
        while left <= right:
            mid = left + ((right - left) >> 1)
            if mid * mid <= x:
                left = mid + 1
            else:
                right = mid - 1
        return left - 1

    # 2458. 移除子树后的二叉树高度 (Height of Binary Tree After Subtree Removal Queries)
    def treeQueries(self, root: Optional[TreeNode], queries: List[int]) -> List[int]:
        def dfs(node: Optional[TreeNode], depth: int, rest_h: int):
            if node is None:
                return 0
            total[node.val] = rest_h
            dfs(node.left, depth + 1, max(rest_h, depth + 1 + heights[node.right]))
            dfs(node.right, depth + 1, max(rest_h, depth + 1 + heights[node.left]))

        def dfs_height(node: Optional[TreeNode]) -> int:
            if node is None:
                return 0
            heights[node] = 1 + max(dfs_height(node.left), dfs_height(node.right))
            return heights[node]

        heights = defaultdict(int)
        dfs_height(root)
        total = [0] * (len(heights) + 1)
        dfs(root, -1, 0)
        for i, q in enumerate(queries):
            queries[i] = total[q]
        return queries

    # 2139. 得到目标值的最少行动次数 (Minimum Moves to Reach Target Score)
    def minMoves(self, target: int, maxDoubles: int) -> int:
        res = 0
        while target != 1 and maxDoubles:
            res += 1
            if target & 1:
                target -= 1
            else:
                target >>= 1
                maxDoubles -= 1
        return res + target - 1

    # 891. 子序列宽度之和 (Sum of Subsequence Widths)
    def sumSubseqWidths(self, nums: List[int]) -> int:
        n = len(nums)
        pow2 = [0] * n
        pow2[0] = 1
        MOD = 10**9 + 7
        for i in range(1, n):
            pow2[i] = pow2[i - 1] * 2 % MOD
        nums.sort()
        res = 0
        for i, x in enumerate(nums):
            res += (pow2[i] - pow2[n - 1 - i]) * x
            res %= MOD
        return res

    # 3880. 两个值之间的最小绝对差值 (Minimum Absolute Difference Between Two Values)
    def minAbsoluteDifference(self, nums: list[int]) -> int:
        d = inf
        pre = [-inf] * 2
        for i, x in enumerate(nums):
            if x:
                x -= 1
                d = min(d, i - pre[x ^ 1])
                pre[x] = i
        return d if d != inf else -1

    # 3882. 网格图中最小异或路径 (Minimum XOR Path in a Grid)
    def minCost(self, grid: list[list[int]]) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i < 0 or j < 0:
                return inf
            k ^= grid[i][j]
            if i == 0 and j == 0:
                return k
            res = inf
            for di, dj in (-1, 0), (0, -1):
                res = min(res, dfs(i + di, j + dj, k))
                if res == 0:
                    break
            return res

        m, n = len(grid), len(grid[0])
        res = dfs(m - 1, n - 1, 0)
        dfs.cache_clear()
        return res

    # 3881. 恰好看到 K 个人的方向选择 (Direction Assignments with Exactly K Visible People)
    def countVisiblePeople(self, n: int, _: int, k: int) -> int:
        MOD = 10**9 + 7
        return comb(n - 1, k) * 2 % MOD

    # 3883. 统计满足数位和数组的非递减数组数目 (Count Non Decreasing Arrays With Given Digit Sums)
    def countArrays(self, digitSum: list[int]) -> int:
        d = [[] for _ in range(51)]
        for x in range(5001):
            s = sum(int(c) for c in str(x))
            if s <= 50:
                d[s].append(x)
        MOD = 10**9 + 7
        n = len(digitSum)
        f = [1] * len(d[digitSum[-1]])
        for i in range(n - 2, -1, -1):
            if len(f) == 0:
                return 0
            x = digitSum[i]
            g = [0] * len(d[x])
            j = len(d[x]) - 1
            k = len(f) - 1
            s = 0
            while j >= 0:
                while k >= 0 and d[digitSum[i + 1]][k] >= d[x][j]:
                    s += f[k]
                    s %= MOD
                    k -= 1
                g[j] = s
                j -= 1
            f = g
        return sum(f) % MOD

    # 3884. 双端字符匹配 (First Matching Character From Both Ends)
    def firstMatchingIndex(self, s: str) -> int:
        n = len(s)
        for i in range((n + 1) // 2):
            if s[i] == s[n - 1 - i]:
                return i
        return -1

    # 3886. 可排序整数求和 (Sum of Sortable Integers)
    def sortableIntegers(self, nums: list[int]) -> int:
        def check(d: int) -> bool:
            pre_mx = -inf
            for i in range(0, n, d):
                c = 0
                mx = -inf
                for j in range(i, i + d):
                    if nums[j] < pre_mx:
                        return False
                    mx = max(mx, nums[j])
                    if j > i and nums[j] < nums[j - 1]:
                        c += 1
                        if c > 1:
                            return False
                if not (c == 0 or nums[i] >= nums[i + d - 1]):
                    return False
                pre_mx = mx
            return True

        n = len(nums)
        a = []
        for i in range(1, isqrt(n) + 1):
            if n % i == 0:
                a.append(i)
                if i * i != n:
                    a.append(n // i)
        return sum(d for d in a if check(d))

    # 3885. 设计事件管理器 (Design Event Manager) --懒更新
    class EventManager:

        def __init__(self, events: list[list[int]]):
            self.id_to_priority = defaultdict(lambda: -1)
            self.q = []
            heapq.heapify(self.q)
            for eventId, priority in events:
                self.id_to_priority[eventId] = priority
                heapq.heappush(self.q, (-priority, eventId))

        def updatePriority(self, eventId: int, newPriority: int) -> None:
            self.id_to_priority[eventId] = newPriority
            heapq.heappush(self.q, (-newPriority, eventId))

        def pollHighest(self) -> int:
            while self.q:
                priority, eventId = heapq.heappop(self.q)
                if self.id_to_priority[eventId] == -priority:
                    del self.id_to_priority[eventId]
                    return eventId
            return -1

    # 3887. 增量偶权环查询 (Incremental Even-Weighted Cycle Queries)
    def numberOfEdgesAdded(self, n: int, edges: List[List[int]]) -> int:
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
                if self.is_connected(p1, p2):
                    return
                r1 = self.get_root(p1)
                r2 = self.get_root(p2)
                if self.rank[r1] < self.rank[r2]:
                    self.parent[r1] = r2
                else:
                    self.parent[r2] = r1
                    if self.rank[r1] == self.rank[r2]:
                        self.rank[r1] += 1

        res = 0
        _union = union(n * 2)
        for u, v, w in edges:
            # u、v 不连通 ：flag == -1
            # u、v 连通、相同颜色 ：flag == 0
            # u、v 连通、不同颜色 ：flag == 1
            flag = -1
            if _union.is_connected(u, v):
                flag = 0
            elif _union.is_connected(u, v + n):
                flag = 1
            if flag >= 0 and flag != w:
                continue
            res += 1
            if w == 0:
                _union.union(u, v)
                _union.union(u + n, v + n)
            else:
                _union.union(u, v + n)
                _union.union(u + n, v)
        return res

    # 2751. 机器人碰撞 (Robot Collisions)
    def survivedRobotsHealths(
        self, positions: List[int], healths: List[int], directions: str
    ) -> List[int]:
        robots = sorted(zip(positions, healths, directions, range(len(positions))))
        st = []
        for _, h, d, id in robots:
            if d == "R":
                st.append((h, "R", id))
            else:
                while st and st[-1][1] == "R":
                    _h, _, _id = st.pop()
                    if _h == h:
                        break
                    if _h > h:
                        st.append((_h - 1, "R", _id))
                        break
                    h -= 1
                else:
                    st.append((h, "L", id))
        return [h for h, _, _ in sorted(st, key=lambda x: x[-1])]

    # 2778. 特殊元素平方和 (Sum of Squares of Special Elements)
    def sumOfSquares(self, nums: List[int]) -> int:
        return sum(x * x for i, x in enumerate(nums, start=1) if len(nums) % i == 0)

    # 2087. 网格图中机器人回家的最小代价 (Minimum Cost Homecoming of a Robot in a Grid)
    def minCost(
        self,
        startPos: List[int],
        homePos: List[int],
        rowCosts: List[int],
        colCosts: List[int],
    ) -> int:
        res = 0
        r = sorted([startPos[0], homePos[0]])
        res += sum(rowCosts[i] for i in range(r[0], r[1] + 1))
        res -= rowCosts[startPos[0]]
        c = sorted([startPos[1], homePos[1]])
        res += sum(colCosts[i] for i in range(c[0], c[1] + 1))
        res -= colCosts[startPos[1]]
        return res

    # 3889. 镜像频次距离 (Mirror Frequency Distance)
    def mirrorFrequency(self, s: str) -> int:
        res = 0
        d = defaultdict(int)
        for x in s:
            d[x] += 1
        for x in s:
            if x in d:
                m = None
                if x.isalpha():
                    m = (chr)((25 - (ord(x) - ord("a"))) + ord("a"))
                else:
                    m = (chr)((9 - (ord(x) - ord("0"))) + ord("0"))
                res += abs(d[x] - d[m])
                del d[x]
                del d[m]
        return res

    # 3890. 可由多种立方和构造的整数 (Integers With Multiple Sum of Two Cubes)
    def findGoodIntegers(self, n: int) -> list[int]:
        d = defaultdict(int)
        for i in range(1, 1001):
            if i * i * i > n:
                break
            for j in range(i, 1001):
                m = i * i * i + j * j * j
                if m > n:
                    break
                d[m] += 1
        return sorted(k for k, v in d.items() if v >= 2)

    # 3891. 最大化特殊下标数目的最少增加次数 (Minimum Increase to Maximize Special Indices)
    def minIncrease(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i >= n - 1:
                return 0
            res = dfs(i + 2, j) + max(0, max(nums[i - 1], nums[i + 1]) - nums[i] + 1)
            if j == 0 and n & 1 == 0:
                res = min(res, dfs(i + 1, 1))
            return res

        n = len(nums)
        return dfs(1, 0)

    # 3892. 产生至少 K 个峰值的最少操作次数 (Minimum Operations to Achieve At Least K Peaks)
    def minOperations(self, nums: list[int], k: int) -> int:
        @cache
        def dfs(i: int, j: int, s: bool) -> int:
            if j >= k:
                return 0
            if i >= n or k - j > (n - i) // 2 + ((n - i) & 1 and (not s)):
                return inf
            return min(
                dfs(i + 1, j, s),
                dfs(i + 2, j + 1, s or i == 0)
                + max(0, max(nums[i - 1], nums[(i + 1) % n]) - nums[i] + 1),
            )

        n = len(nums)
        t = n // 2
        if k > t:
            return -1
        cnt = 0
        for i in range(n):
            if nums[i - 1] < nums[i] > nums[(i + 1) % n]:
                cnt += 1
        if cnt >= k:  # 优化：已经有至少 k 个峰值了，无需操作
            return 0
        # dfs(i, j, c) 表示以i为中心点，j表已经构造了j个特殊下标，c表示位置0是否被选过
        res = dfs(0, 0, False)
        dfs.cache_clear()
        return res

    # 874. 模拟行走机器人 (Walking Robot Simulation)
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        d = set(map(tuple, obstacles))
        x = y = res = 0
        # 0 北 1 东 2 南 3 西
        di = [0, 1, 0, -1]
        dj = [1, 0, -1, 0]
        d_idx = 0
        for c in commands:
            if c == -2:
                d_idx = (d_idx + 3) % 4
            elif c == -1:
                d_idx = (d_idx + 1) % 4
            else:
                for _ in range(c):
                    x += di[d_idx]
                    y += dj[d_idx]
                    if (x, y) in d:
                        x -= di[d_idx]
                        y -= dj[d_idx]
                        break
                res = max(res, x * x + y * y)
        return res

    # 2069. 模拟行走机器人 II (Walking Robot Simulation II)
    class Robot:

        def __init__(self, width: int, height: int):
            self.width = width
            self.height = height
            self.x = 0
            self.y = 0
            self.d_idx = 0
            # 0 东 1 北 2 西 3 南
            self.di = [1, 0, -1, 0]
            self.dj = [0, 1, 0, -1]
            self.dirs = ["East", "North", "West", "South"]
            self.perimeter = 2 * (width + height) - 4

        def step(self, num: int) -> None:
            num %= self.perimeter
            while num:
                # 需要走多少步能到边界
                need = 0
                if self.d_idx == 0:
                    need = self.width - 1 - self.x
                elif self.d_idx == 1:
                    need = self.height - 1 - self.y
                elif self.d_idx == 2:
                    need = self.x
                else:
                    need = self.y
                if need == 0:
                    # 转向
                    self.d_idx = (self.d_idx + 1) % 4
                _min = min(num, need)
                self.x += self.di[self.d_idx] * _min
                self.y += self.dj[self.d_idx] * _min
                num -= _min
            if self.x == 0 and self.y == 0:
                self.d_idx = 3

        def getPos(self) -> List[int]:
            return [self.x, self.y]

        def getDir(self) -> str:
            return self.dirs[self.d_idx]

    # 3894. 交通信号灯的颜色 (Traffic Signal Color)
    def trafficSignal(self, timer: int) -> str:
        if timer == 0:
            return "Green"
        if timer == 30:
            return "Orange"
        return "Red" if 30 <= timer <= 90 else "Invalid"

    # 3895. 统计数字出现总次数 (Count Digit Appearances)
    def countDigitOccurrences(self, nums: list[int], digit: int) -> int:
        res = 0
        for x in nums:
            while x:
                x, m = divmod(x, 10)
                if m == digit:
                    res += 1
        return res

    # 3896. 将数组转换为交替质数数组的最少操作次数 (Minimum Operations to Transform Array into Alternating Prime)
    def minOperations(self, nums: list[int]) -> int:
        MX = 10**5 + 50
        primes = [True] * MX
        primes[0] = primes[1] = False
        for i in range(2, MX):
            if primes[i]:
                for j in range(i * i, MX, i):
                    primes[j] = False
        cnts = [0] * MX
        last_prime = inf
        for i in range(MX - 2, 0, -1):
            if not primes[i]:
                cnts[i] = last_prime - i
            elif i == 2:
                last_prime = i
                cnts[i] = 2
            else:
                last_prime = i
                cnts[i] = 1
        res = 0
        for i, x in enumerate(nums):
            if (i & 1 == 0) == primes[x]:
                continue
            res += cnts[x]
        return res

    # 3897. 连接二进制片段得到的最大值 (Maximum Value of Concatenated Binary Segments)
    def maxValue(self, nums1: list[int], nums0: list[int]) -> int:
        MOD = 10**9 + 7
        MX = 2 * (10**5) + 1
        pow2 = [0] * MX
        pow2[0] = 1
        for i in range(1, MX):
            pow2[i] = pow2[i - 1] * 2 % MOD
        a = []
        for x, y in zip(nums1, nums0):
            a.append((x, y))
        a.sort(key=lambda o: (o[1] != 0, -o[0], o[1]))
        res = 0
        for x, y in a:
            res = ((1 + res) * pow2[x + y] % MOD - pow2[y]) % MOD
        return res

    # 3898. 统计每个顶点的度 (Find the Degree of Each Vertex)
    def findDegrees(self, matrix: list[list[int]]) -> list[int]:
        n = len(matrix)
        res = [0] * n
        for i in range(n):
            for j in range(n):
                res[i] += matrix[i][j]
        return res

    # 3899. 三角形的内角度数 (Angles of a Triangle) --余弦定理
    def internalAngles(self, sides: list[int]) -> list[float]:
        sides.sort()
        if sides[0] + sides[1] <= sides[2]:
            return []
        cos_a = (sides[1] ** 2 + sides[2] ** 2 - sides[0] ** 2) / (
            sides[1] * sides[2] * 2
        )
        cos_b = (sides[0] ** 2 + sides[2] ** 2 - sides[1] ** 2) / (
            sides[0] * sides[2] * 2
        )
        cos_c = (sides[0] ** 2 + sides[1] ** 2 - sides[2] ** 2) / (
            sides[0] * sides[1] * 2
        )
        return [
            math.degrees(math.acos(cos_a)),
            math.degrees(math.acos(cos_b)),
            math.degrees(math.acos(cos_c)),
        ]

    # 3900. 一次交换后的最长平衡子串 (Longest Balanced Substring After One Swap)
    def longestBalanced(self, s: str) -> int:
        res = 0
        n = len(s)
        cnt0 = s.count("0")
        cnt1 = n - cnt0
        d = defaultdict(list)
        d[0].append(-1)
        # 前缀和 1比0多的数量
        pre = 0
        for i, x in enumerate(s):
            pre += 1 if x == "1" else -1
            # 不交换
            if pre in d:
                res = max(res, i - d[pre][0])
            # 交换一次
            if pre - 2 in d:
                a = d[pre - 2]
                if (i - a[0] - 2) // 2 < cnt0:
                    res = max(res, i - a[0])
                elif len(a) >= 2:
                    res = max(res, i - a[1])

            if pre + 2 in d:
                a = d[pre + 2]
                if (i - a[0] - 2) // 2 < cnt1:
                    res = max(res, i - a[0])
                elif len(a) >= 2:
                    res = max(res, i - a[1])
            if len(d[pre]) < 2:
                d[pre].append(i)
        return res

    # 1848. 到目标元素的最小距离 (Minimum Distance to the Target Element)
    def getMinDistance(self, nums: List[int], target: int, start: int) -> int:
        i = j = start
        while i >= 0 or j < len(nums):
            if i >= 0:
                if target == nums[i]:
                    return start - i
                i -= 1
            if j < len(nums):
                if nums[j] == target:
                    return j - start
                j += 1

    # 2515. 到目标字符串的最短距离 (Shortest Distance to Target String in a Circular Array)
    def closestTarget(self, words: List[str], target: str, startIndex: int) -> int:
        a = [i for i, w in enumerate(words) if w == target]
        n = len(words)
        return (
            -1
            if len(a) == 0
            else min(min(n - abs(i - startIndex), abs(i - startIndex)) for i in a)
        )

    # 3488. 距离最小相等元素查询 (Closest Equal Element Queries)
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        d = defaultdict(list)
        n = len(nums)
        for i, x in enumerate(nums):
            d[x].append(i)
        a = [-1] * n
        for v in d.values():
            if len(v) <= 1:
                continue
            for i, p in enumerate(v):
                r1 = abs(v[i] - v[(i + 1) % len(v)])
                r2 = n - r1
                r3 = abs(v[i] - v[(i - 1) % len(v)])
                r4 = n - r3
                a[p] = min(r1, r2, r3, r4)
        return [a[q] for q in queries]

    # 1855. 下标对中的最大距离 (Maximum Distance Between a Pair of Values)
    def maxDistance(self, nums1: List[int], nums2: List[int]) -> int:
        m = len(nums1)
        res = 0
        i = 0
        for j, y in enumerate(nums2):
            while i < m and nums1[i] > y:
                i += 1
            if i == m:
                break
            res = max(res, j - i)
        return res

    # 3903. 最小稳定下标 I (Smallest Stable Index I)
    # 3904. 最小稳定下标 II (Smallest Stable Index II)
    def firstStableIndex(self, nums: list[int], k: int) -> int:
        n = len(nums)
        suf_min = [inf] * n
        suf_min[-1] = nums[-1]
        for i in range(n - 2, -1, -1):
            suf_min[i] = min(suf_min[i + 1], nums[i])
        pre_max = 0
        for i, x in enumerate(nums):
            pre_max = max(pre_max, x)
            if pre_max - suf_min[i] <= k:
                return i
        return -1

    # 3905. 多源洪水灌溉 (Multi Source Flood Fill)
    def colorGrid(self, n: int, m: int, sources: list[list[int]]) -> list[list[int]]:
        res = [[0] * m for _ in range(n)]
        sources.sort(key=lambda o: -o[-1])
        q = deque()
        for x, y, c in sources:
            res[x][y] = c
            q.append((x, y, c))

        while q:
            s = len(q)
            for _ in range(s):
                x, y, c = q.popleft()
                for dx, dy in (0, 1), (0, -1), (1, 0), (-1, 0):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < m and res[nx][ny] == 0:
                        res[nx][ny] = c
                        q.append((nx, ny, c))
        return res

    # 3906. 统计网格路径中好整数的数目 (Count Good Integers on a Grid Path)
    def countGoodIntegersOnPath(self, l: int, r: int, directions: str) -> int:
        def cal(x: int) -> int:
            @cache
            def dfs(i: int, j: int, is_limit: bool) -> int:
                if i == 16:
                    return 1
                res = 0
                up = int(s[i]) if is_limit else 9
                for d in range(j if in_path >> i & 1 else 0, up + 1):
                    res += dfs(
                        i + 1, d if in_path >> i & 1 else j, is_limit and up == d
                    )
                return res

            s = str(x).zfill(16)
            # dfs(i, j, is_limit) 当前第i位， 上一个数选的是j
            return dfs(0, 0, True)

        in_path = 1
        p = 0
        for d in directions:
            if d == "R":
                p += 1
            else:
                p += 4
            in_path |= 1 << p
        return cal(r) - cal(l - 1)

    # 1722. 执行交换操作后的最小汉明距离 (Minimize Hamming Distance After Swap Operations)
    def minimumHammingDistance(
        self, source: List[int], target: List[int], allowedSwaps: List[List[int]]
    ) -> int:
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

        n = len(source)
        _union = union(n)
        for u, v in allowedSwaps:
            _union.union(u, v)
        g = defaultdict(list)
        for i in range(n):
            r = _union.get_root(i)
            g[r].append(i)
        res = 0
        for l in g.values():
            d = defaultdict(int)
            for i in l:
                d[source[i]] += 1
                d[target[i]] -= 1
            res += sum(abs(x) for x in d.values()) // 2
        return res

    # 1722. 执行交换操作后的最小汉明距离 (Minimize Hamming Distance After Swap Operations)
    def minimumHammingDistance(
        self, source: List[int], target: List[int], allowedSwaps: List[List[int]]
    ) -> int:
        def dfs(x: int):
            vis[x] = True
            diff[source[x]] += 1
            diff[target[x]] -= 1
            for y in g[x]:
                if not vis[y]:
                    dfs(y)

        n = len(source)
        g = [[] for _ in range(n)]
        for u, v in allowedSwaps:
            g[u].append(v)
            g[v].append(u)
        res = 0
        vis = [False] * n
        for i in range(n):
            if not vis[i]:
                diff = defaultdict(int)
                dfs(i)
                res += sum(abs(v) for v in diff.values())
        return res // 2

    # 2452. 距离字典两次编辑以内的单词 (Words Within Two Edits of Dictionary)
    def twoEditWords(self, queries: List[str], dictionary: List[str]) -> List[str]:
        res = []
        for q in queries:
            for d in dictionary:
                if sum(x != y for x, y in zip(q, d)) <= 2:
                    res.append(q)
                    break
        return res

    # 2615. 等值距离和 (Sum of Distances)
    def distance(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [0] * n
        d = defaultdict(list)
        for i, x in enumerate(nums):
            d[x].append(i)
        for l in d.values():
            pre = [0] * len(l)
            for i in range(1, len(l)):
                pre[i] = pre[i - 1] + i * (l[i] - l[i - 1])
            res[l[-1]] = pre[-1]
            suf = 0
            for i in range(len(l) - 2, -1, -1):
                suf += (len(l) - i - 1) * (l[i + 1] - l[i])
                res[l[i]] = pre[i] + suf
        return res

    # 1559. 二维网格图中探测环 (Detect Cycles in 2D Grid)
    def containsCycle(self, grid: List[List[str]]) -> bool:
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
                if self.is_connected(p1, p2):
                    return
                r1 = self.get_root(p1)
                r2 = self.get_root(p2)
                if self.rank[r1] < self.rank[r2]:
                    self.parent[r1] = r2
                else:
                    self.parent[r2] = r1
                    if self.rank[r1] == self.rank[r2]:
                        self.rank[r2] += 1

        def cal(x: int, y: int) -> int:
            return x * n + y

        m, n = len(grid), len(grid[0])
        _u = union(m * n)
        for i in range(m):
            for j in range(n):
                if (
                    i
                    and j
                    and grid[i][j] == grid[i - 1][j] == grid[i][j - 1]
                    and _u.is_connected(cal(i - 1, j), cal(i, j - 1))
                ):
                    return True
                if i and grid[i - 1][j] == grid[i][j]:
                    _u.union(cal(i - 1, j), cal(i, j))
                if j and grid[i][j - 1] == grid[i][j]:
                    _u.union(cal(i, j - 1), cal(i, j))
        return False

    # 3908. 有效数字 (Valid Digit Number)
    def validDigit(self, n: int, x: int) -> bool:
        last = 0
        has_x = False
        while n:
            n, mod = divmod(n, 10)
            if mod == x:
                has_x = True
            last = mod
        return last != x and has_x

    # 3909. 比较双调部分的和 (Compare Sums of Bitonic Parts)
    def compareBitonicSums(self, nums: list[int]) -> int:
        mx = max(nums)
        s = sum(nums) + mx
        pre = 0
        for x in nums:
            pre += x
            if x == mx:
                if pre == s - pre:
                    return -1
                if pre > s - pre:
                    return 0
                return 1

    # 3910. 统计节点和为偶数的连通子图 (Count Connected Subgraphs with Even Node Sum)
    def evenSumSubgraphs(self, nums: list[int], edges: list[list[int]]) -> int:
        def dfs(x: int):
            nonlocal u
            u |= 1 << x
            if u == mask:
                return
            for y in g[x]:
                if mask >> y & 1 and u >> y & 1 == 0:
                    dfs(y)

        n = len(nums)
        g = [[] for _ in range(1 << n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        s = [0] * (1 << n)
        res = 0
        for mask in range(1, 1 << n):
            u = 0
            lb = (mask & -mask).bit_length() - 1
            s[mask] = (s[mask & (mask - 1)] + nums[lb]) % 2
            if s[mask] == 0:
                dfs(lb)
                if u == mask:
                    res += 1
        return res

    # 3912. 数组中的有效元素 (Valid Elements in an Array)
    def findValidElements(self, nums: list[int]) -> list[int]:
        n = len(nums)
        b = [False] * n
        b[0] = b[-1] = True
        mx = nums[0]
        for i in range(1, n):
            if nums[i] > mx:
                b[i] = True
            mx = max(mx, nums[i])
        mx = nums[-1]
        for i in range(n - 2, -1, -1):
            if nums[i] > mx:
                b[i] = True
                mx = nums[i]
        return [nums[i] for i, x in enumerate(b) if x]

    # 3913. 按频率对元音排序 (Sort Vowels by Frequency)
    def sortVowels(self, s: str) -> str:
        d = defaultdict(list)
        for i, x in enumerate(s):
            if x in "aeiou":
                if x not in d:
                    d[x] = [1, i]
                else:
                    cur = d[x]
                    cur[0] += 1
        a = []
        for k, v in d.items():
            a.append([k, v[0], v[1]])
        a.sort(key=lambda o: (-o[1], o[2]))
        j = 0
        res = []
        for x in s:
            if x not in "aeiou":
                res.append(x)
            else:
                res.append(a[j][0])
                a[j][1] -= 1
                if a[j][1] == 0:
                    j += 1
        return "".join(res)

    # 3914. 使数组非递减需要的最小累计值 (Minimum Operations to Make Array Non Decreasing)
    def minOperations(self, nums: list[int]) -> int:
        pre = 0
        res = 0
        for x in nums:
            x += res
            add = max(0, pre - x)
            res += add
            pre = x + add
        return res

    # 1391. 检查网格中是否存在有效路径 (Check if There is a Valid Path in a Grid)
    def hasValidPath(self, grid: List[List[int]]) -> bool:
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
                if self.is_connected(p1, p2):
                    return
                r1 = self.get_root(p1)
                r2 = self.get_root(p2)
                if self.rank[r1] < self.rank[r2]:
                    self.parent[r1] = r2
                else:
                    self.parent[r2] = r1
                    if self.rank[r1] == self.rank[r2]:
                        self.rank[r2] += 1

        def cal(x: int, y: int) -> int:
            return x * n + y

        m, n = len(grid), len(grid[0])
        u = union(m * n)
        for i in range(m):
            for j in range(n):
                if i:
                    if (grid[i][j] == 2 or grid[i][j] == 5 or grid[i][j] == 6) and (
                        grid[i - 1][j] == 2
                        or grid[i - 1][j] == 3
                        or grid[i - 1][j] == 4
                    ):
                        u.union(cal(i, j), cal(i - 1, j))
                if j:
                    if (grid[i][j] == 1 or grid[i][j] == 3 or grid[i][j] == 5) and (
                        grid[i][j - 1] == 1
                        or grid[i][j - 1] == 4
                        or grid[i][j - 1] == 6
                    ):
                        u.union(cal(i, j), cal(i, j - 1))
        return u.is_connected(cal(0, 0), cal(m - 1, n - 1))

    # 2033. 获取单值网格的最小操作数 (Minimum Operations to Make a Uni-Value Grid)
    def minOperations(self, grid: List[List[int]], x: int) -> int:
        m, n = len(grid), len(grid[0])
        mod = grid[0][0] % x
        a = []
        s = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] % x != mod:
                    return -1
                a.append(grid[i][j])
                s += grid[i][j]
        a.sort()
        res = inf
        pre_s = 0
        for i, v in enumerate(a):
            _a1 = (v * i - pre_s) // x
            pre_s += v
            _a2 = ((s - pre_s) - (len(a) - i - 1) * v) // x
            res = min(res, _a1 + _a2)
        return res

    # 396. 旋转函数 (Rotate Function)
    def maxRotateFunction(self, nums: list[int]) -> int:
        n = len(nums)
        f = sum(i * x for i, x in enumerate(nums))  # F(0)
        s = sum(nums)  # nums 的总和

        ans = f
        for i in range(n - 1, 0, -1):
            f += s - n * nums[i]
            ans = max(ans, f)
        return ans

    # 796. 旋转字符串 (Rotate String)
    def rotateString(self, s: str, goal: str) -> bool:
        return any(s[i:] + s[:i] == goal for i in range(len(s)))

    # 61. 旋转链表 (Rotate List)
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        n = 0
        p = head
        while p:
            n += 1
            p = p.next
        if n == 0:
            return head
        k %= n
        fast = head
        while k:
            k -= 1
            fast = fast.next
        slow = head
        while fast.next:
            slow = slow.next
            fast = fast.next
        fast.next = head
        res = slow.next
        slow.next = None
        return res

    # 3917. 统计下标的相反奇偶性得分 (Count Indices With Opposite Parity)
    def countOppositeParity(self, nums: list[int]) -> list[int]:
        n = len(nums)
        cnts = [0] * 2
        for i in range(n - 1, -1, -1):
            x = nums[i] & 1
            nums[i] = cnts[x ^ 1]
            cnts[x] += 1
        return nums

    # 3918. 区间内的质数和 (Sum of Primes Between Number and Its Reverse)
    def sumOfPrimesInRange(self, n: int) -> int:
        MX = 1001
        prime = [True] * MX
        prime[1] = False
        for i in range(2, MX):
            if prime[i]:
                for j in range(i * i, MX, i):
                    prime[j] = False

        def rev(x: int) -> int:
            res = 0
            while x:
                res = res * 10 + x % 10
                x //= 10
            return res

        r = rev(n)
        return sum(i for i in range(min(r, n), max(r, n) + 1) if prime[i])

    # 3919. 在下标间移动的最小代价 (Minimum Cost to Move Between Indices)
    def minCost(self, nums: list[int], queries: list[list[int]]) -> list[int]:
        n = len(nums)
        sum_l = [0] * n  # sum_l[i] 等于从 i 移动到 0 的代价和
        sum_r = [0] * n  # sum_r[i] 等于从 0 移动到 i 的代价和
        for i in range(1, n):
            # 往左走 i -> i-1
            if (
                i < n - 1 and nums[i] - nums[i - 1] > nums[i + 1] - nums[i]
            ):  # closest(i) = i+1
                cost = nums[i] - nums[i - 1]  # 只能用方式一往左走
            else:
                cost = 1
            sum_l[i] = sum_l[i - 1] + cost

            # 往右走 i-1 -> i
            if (
                i > 1 and nums[i - 1] - nums[i - 2] <= nums[i] - nums[i - 1]
            ):  # closest(i-1) = i-2
                cost = nums[i] - nums[i - 1]  # 只能用方式一往右走
            else:
                cost = 1
            sum_r[i] = sum_r[i - 1] + cost

        ans = [0] * len(queries)
        for i, q in enumerate(queries):
            l, r = q
            if l < r:
                # cost(0 -> r) - cost(0 -> l) = cost(l -> r)
                ans[i] = sum_r[r] - sum_r[l]
            else:
                # cost(l -> 0) - cost(r -> 0) = cost(l -> r)
                ans[i] = sum_l[l] - sum_l[r]
        return ans

    # 1861. 旋转盒子 (Rotating the Box)
    def rotateTheBox(self, boxGrid: List[List[str]]) -> List[List[str]]:
        def check(a: List[str]) -> List[int]:
            res = [0] * len(a)
            cnt = 0
            for i, x in enumerate(a):
                if x == "*":
                    res[i] = -1
                    if cnt:
                        res[i - 1] = cnt
                        cnt = 0
                    continue
                if x == "#":
                    cnt += 1
                if i == len(a) - 1:
                    res[i] = cnt
            return res

        m, n = len(boxGrid), len(boxGrid[0])
        res = [[None] * m for _ in range(n)]
        for i in range(m - 1, -1, -1):
            r = check(boxGrid[i])
            cnt = 0
            for j in range(n - 1, -1, -1):
                if r[j] == -1:
                    res[j][m - i - 1] = "*"
                elif r[j] > 0:
                    cnt = r[j] - 1
                    res[j][m - i - 1] = "#"
                elif cnt:
                    cnt -= 1
                    res[j][m - i - 1] = "#"
                else:
                    res[j][m - i - 1] = "."
        return res

    # 1914. 循环轮转矩阵 (Cyclically Rotating a Grid)
    def rotateGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
        m, n = len(grid), len(grid[0])
        i0, i1, j0, j1 = 0, m - 1, 0, n - 1
        for _ in range(min(m, n) // 2):
            a = []
            i, j = i0, j0
            while j < j1:
                a.append(grid[i][j])
                j += 1
            while i < i1:
                a.append(grid[i][j])
                i += 1
            while j > j0:
                a.append(grid[i][j])
                j -= 1
            while i > i0:
                a.append(grid[i][j])
                i -= 1
            l = len(a)
            k0 = k % l
            a = a[k0:] + a[:k0]
            id = 0
            while j < j1:
                grid[i][j] = a[id]
                id += 1
                j += 1
            while i < i1:
                grid[i][j] = a[id]
                id += 1
                i += 1
            while j > j0:
                grid[i][j] = a[id]
                id += 1
                j -= 1
            while i > i0:
                grid[i][j] = a[id]
                id += 1
                i -= 1
            i0 += 1
            i1 -= 1
            j0 += 1
            j1 -= 1
        return grid

    # 2553. 分割数组中数字的数位 (Separate the Digits in an Array)
    def separateDigits(self, nums: List[int]) -> List[int]:
        res = []
        for x in nums:
            for d in str(x):
                res.append(int(d))
        return res

    # 3921. 分数验证器 (Score Validator)
    def scoreValidator(self, events: list[str]) -> list[int]:
        score, counter = 0, 0
        for x in events:
            if x.isdigit():
                score += int(x)
            elif x == "W":
                counter += 1
            else:
                score += 1
            if counter == 10:
                break
        return [score, counter]

    # 3922. 使二进制字符串连贯的最少翻转次数 (Minimum Flips to Make Binary String Coherent)
    def minFlips(self, s: str) -> int:
        n = len(s)
        cnt1 = s.count("1")
        # 都是1或都是0
        res = min(cnt1, n - cnt1)
        # 既有1也有0的情况，若首尾都是1，则吧中间的都变成0
        # 否则需要整个字符串只保留1个1
        return min(res, max(0, max(0, cnt1 - 1 - int(s[0] == s[-1] == "1"))))

    # 3925. 连接逆序数组 (Concatenate Array With Reverse)
    def concatWithReverse(self, nums: list[int]) -> list[int]:
        return nums + list(reversed(nums))

    # 3926. 有效单词计数 (Count Valid Word Occurrences)
    def countWordOccurrences(self, chunks: list[str], queries: list[str]) -> list[int]:
        s = "".join(chunks)
        cnt = defaultdict(int)

        for t in s.split():  # 不用 split 的写法见另一份代码
            n = len(t)
            i = 0
            while i < n:
                if t[i] == "-":
                    i += 1
                    continue
                start = i
                # 遇到 "--"（连续两个 '-'）就跳出循环
                while i < n and (t[i] != "-" or i < n - 1 and t[i + 1] != "-"):
                    i += 1
                cnt[t[start:i]] += 1

        return [cnt[q] for q in queries]

    # 3927. 可整除替换后的数组最小元素和 (Minimize Array Sum Using Divisible Replacements)
    def minArraySum(self, nums: list[int]) -> int:
        d = defaultdict(int)
        n = len(nums)
        for x in nums:
            if x == 1:
                return n
            d[x] += 1
        MX = 10**5 + 1
        prime = [True] * MX
        for i in range(2, MX):
            if prime[i]:
                for j in range(i * i, MX, i):
                    prime[j] = False
        res = 0
        for k, v in d.items():
            if prime[k]:
                res += k * v
            else:
                _min = inf
                for i in range(2, isqrt(k) + 1):
                    if k % i == 0:
                        if i in d:
                            res += i * v
                            break
                        if k // i in d:
                            _min = min(_min, k // i)
                else:
                    if _min < inf:
                        res += _min * v
                    else:
                        res += k * v
        return res

    # 1665. 完成所有任务的最少初始能量 (Minimum Initial Energy to Finish Tasks)
    def minimumEffort(self, tasks: List[List[int]]) -> int:
        tasks.sort(key=lambda o: o[1] - o[0])
        res = 0
        for actual, minimum in tasks:
            res = max(res + actual, minimum)
        return res

    # 3924. 有限重边的最小阈值路径 (Minimum Threshold Path With Limited Heavy Edges)
    def minimumThreshold(
        self, n: int, edges: List[List[int]], source: int, target: int, k: int
    ) -> int:
        def check(limit: int) -> bool:
            dis = [k + 1] * n
            dis[source] = 0
            q = []
            q.append((0, source))
            heapq.heapify(q)
            while q:
                c, x = heapq.heappop(q)
                if x == target:
                    return True
                if c > dis[x]:
                    continue
                for y, w in g[x]:
                    if c + (w > limit) < dis[y]:
                        dis[y] = c + (w > limit)
                        heapq.heappush(q, (c + (w > limit), y))
            return False

        if source == target:
            return 0
        g = [[] for _ in range(n)]
        mx = 0
        for u, v, w in edges:
            g[u].append((v, w))
            g[v].append((u, w))
            mx = max(mx, w)
        left = 0
        right = mx
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return -1 if right + 1 > mx else right + 1

    # 3923. 得到目标点的最少代数 (Minimum Generations to Target Point)
    def minGenerations(self, points: List[List[int]], target: List[int]) -> int:
        tar = tuple(target)
        cur = set(map(tuple, points))
        res = 0
        for res in count(0):
            if tar in cur:
                return res
            nxt = cur.copy()
            for (x0, y0, z0), (x1, y1, z1) in combinations(cur, 2):
                nxt.add((((x0 + x1) // 2), ((y0 + y1) // 2), ((z0 + z1) // 2)))
            if len(nxt) == len(cur):
                return -1
            cur = nxt

    # 3928. 购买苹果的最低成本 II (Minimum Cost to Buy Apples II)
    def minCost(self, n: int, prices: List[int], roads: List[List[int]]) -> List[int]:
        def cal(start: int, carry_apple: bool) -> List[int]:
            dis = [inf if carry_apple else prices[start]] * n
            dis[start] = 0
            q = []
            heapq.heapify(q)
            q.append((0, start))
            while q:
                d, x = heapq.heappop(q)
                if d > dis[x]:
                    continue
                for y, cost, tax in g[x]:
                    dx = cost * tax if carry_apple else cost
                    if d + dx < dis[y]:
                        dis[y] = d + dx
                        heapq.heappush(q, (d + dx, y))
            return dis

        g = [[] for _ in range(n)]
        for u, v, cost, tax in roads:
            g[u].append((v, cost, tax))
            g[v].append((u, cost, tax))
        # dis0[i][j] : 从i到j 不携带苹果，的最小值
        dis0 = [[inf] * n for _ in range(n)]
        # dis1[i][j] : 从i到j 携带苹果，的最小值
        dis1 = [[inf] * n for _ in range(n)]
        for i in range(n):
            # 从i出发，不携带苹果
            dis0[i] = cal(i, False)
            # 从i出发，携带苹果
            dis1[i] = cal(i, True)
        res = [inf] * n
        for i in range(n):
            for j in range(n):
                res[i] = min(res[i], dis0[i][j] + dis1[j][i] + prices[j])
        return res

    # 153. 寻找旋转排序数组中的最小值 (Find Minimum in Rotated Sorted Array)
    def findMin(self, nums: List[int]) -> int:
        n = len(nums)
        x = nums[-1]
        left = 0
        right = n - 2
        while left <= right:
            mid = left + ((right - left) >> 1)
            if nums[mid] < x:
                right = mid - 1
            else:
                left = mid + 1
        return nums[right + 1]

    # 154. 寻找旋转排序数组中的最小值 II (Find Minimum in Rotated Sorted Array II)
    def findMin(self, nums: List[int]) -> int:
        left, right = -1, len(nums) - 1  # 开区间 (-1, n-1)
        while left + 1 < right:  # 开区间不为空
            mid = (left + right) // 2
            if nums[mid] == nums[right]:
                right -= 1
            elif nums[mid] < nums[right]:
                right = mid
            else:
                left = mid
        return nums[right]

    # 1306. 跳跃游戏 III (Jump Game III)
    def canReach(self, arr: List[int], start: int) -> bool:
        n = len(arr)
        q = deque()
        q.append(start)
        vis = [False] * n
        vis[start] = True
        while q:
            x = q.popleft()
            if arr[x] == 0:
                return True
            if x - arr[x] >= 0 and not vis[x - arr[x]]:
                vis[x - arr[x]] = True
                q.append(x - arr[x])
            if x + arr[x] < n and not vis[x + arr[x]]:
                vis[x + arr[x]] = True
                q.append(x + arr[x])
        return False

    # 1345. 跳跃游戏 IV (Jump Game IV)
    def minJumps(self, arr: List[int]) -> int:
        n = len(arr)
        q = deque()
        d = defaultdict(list)
        for i, x in enumerate(arr):
            d[x].append(i)
        q.append((0, 0))
        vis = [False] * n
        vis[0] = True
        while q:
            pos, step = q.popleft()
            if pos == n - 1:
                return step
            for nxt in d[arr[pos]]:
                if not vis[nxt]:
                    vis[nxt] = True
                    q.append((nxt, step + 1))
            del d[arr[pos]]
            if pos - 1 >= 0 and not vis[pos - 1]:
                vis[pos - 1] = True
                q.append((pos - 1, step + 1))
            if not vis[pos + 1]:
                vis[pos + 1] = True
                q.append((pos + 1, step + 1))
        return -1
    
    # 3931. 检查相邻数字差 (Check Adjacent Digit Differences)
    def isAdjacentDiffAtMostTwo(self, s: str) -> bool:
        return all(abs(int(x) - int(y)) <= 2 for x, y in pairwise(s))
