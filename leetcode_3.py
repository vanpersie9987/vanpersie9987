from ast import Return, Tuple
from asyncio import FastChildWatcher
from gettext import find
import math
from pydoc import plain
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
from networkx import dfs_edges, grid_2d_graph, interval_graph, union
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
