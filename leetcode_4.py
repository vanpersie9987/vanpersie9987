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
        
