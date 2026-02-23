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
import queue
from signal import valid_signals
from sqlite3 import paramstyle
import stat
from termios import CINTR
from tokenize import String
from tty import CC
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
from math import comb, cos, e, fabs, floor, gcd, inf, isqrt, lcm, pi, sqrt, ulp
from mimetypes import init
from multiprocessing import reduction
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
        first = defaultdict(int)
        last = defaultdict(int)
        for i, c in enumerate(colors):
            if c not in first:
                first[c] = i
            last[c] = i
        res = 0
        _min = inf
        _max = -inf
        for color, pos in first.items():
            pos2 = last[color]
            if _min != inf:
                res = max(res, abs(pos2 - _min), abs(pos - _max))
            _min = min(_min, pos)
            _max = max(_max, pos2)
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
