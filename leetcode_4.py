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
                res.append(f"{h:01d}" + ":" + f"{m:02d}")
        return res
