from asyncio import FastChildWatcher
from audioop import reverse
from calendar import c
from collections import Counter, deque
import collections
from ctypes.wintypes import _ULARGE_INTEGER
from curses import curs_set
from decimal import Rounded
import enum
from functools import cache
from inspect import modulesbyfile
from itertools import accumulate, pairwise
from locale import DAY_4
from math import comb, cos, gcd, inf, isqrt, sqrt
from operator import le
from pickletools import read_uint1
from queue import PriorityQueue
from re import X
import re
from socket import NI_NUMERICSERV
from ssl import VERIFY_X509_TRUSTED_FIRST
from tabnanny import check
from textwrap import indent
from tkinter import W
from tkinter.tix import Tree
from turtle import reset, st
from typing import List, Optional
import heapq
import bisect
from zoneinfo import reset_tzpath
# curl https://bootstrap.pypa.io/pip/get-pip.py -o get-pip.py
# sudo python3 get-pip.py
# pip3 install sortedcontainers
from sortedcontainers import SortedList

class leetcode_2:

    # LCP 30. 魔塔游戏
    def magicTower(self, nums: List[int]) -> int:
        if sum(nums) < 0:
            return -1
        res = 0
        hp = 1
        q = []
        for x in nums:
            if x < 0:
                heapq.heappush(q, x)
            hp += x
            if hp <= 0:
                hp -= heapq.heappop(q)
                res += 1
        return res

    # 862. 和至少为 K 的最短子数组 (Shortest Subarray with Sum at Least K)
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        s = list(accumulate(nums, initial=0))
        q = deque()
        res = inf
        for i, v in enumerate(s):
            while q and v - s[q[0]] >= k:
                res = min(res, i - q.popleft())
            while q and s[q[-1]] >= v:
                q.pop()
            q.append(i)
        return -1 if res == inf else res
    
    # 918. 环形子数组的最大和 (Maximum Sum Circular Subarray) --分类讨论
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        s = 0
        res = -inf
        pre_min = 0
        pre_max = -inf
        # 最小子数组和
        smi = inf
        for x in nums:
            s += x
            res = max(res, s - pre_min)
            smi = min(smi, s - pre_max)
            pre_min = min(pre_min, s)
            pre_max = max(pre_max, s)
        return max(res, s - smi)

    # 918. 环形子数组的最大和 (Maximum Sum Circular Subarray) --前缀和 + 单调队列
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        n = len(nums)
        nums = nums + nums
        s = list(accumulate(nums, initial=0))
        q = deque()
        res = -inf
        for i, v in enumerate(s):
            while q and i - q[0] > n:
                q.popleft()
            if q:
                res = max(res, v - s[q[0]])
            while q and s[q[-1]] >= v:
                q.pop()
            q.append(i)
        return res
    
    # 1425. 带限制的子序列和 (Constrained Subsequence Sum)
    def constrainedSubsetSum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        q = deque()
        q.append(0)
        for i in range(1, n):
            while q and i - q[0] > k:
                q.popleft()
            if q:
                dp[i] = max(0, dp[q[0]]) + nums[i]
            while q and dp[q[-1]] <= dp[i]:
                q.pop()
            q.append(i)
        return max(dp)
            



