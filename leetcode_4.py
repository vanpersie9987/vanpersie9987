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
                if not (s[i] == s[j] or s[i].isalpha() and s[j].isalpha and s[i].lower() == s[j].lower()):
                    return False
                i += 1
                j -= 1
        return True
