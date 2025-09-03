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
from math import comb, cos, fabs, gcd, inf, isqrt, lcm, sqrt
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


class Union924:

    def __init__(self, n: int) -> None:
        self.parent = [0] * n
        for i in range(n):
            self.parent[i] = i
        self.rank = [1] * n
        self.size = [1] * n

    def get_root(self, p: int) -> int:
        if self.parent[p] == p:
            return p
        self.parent[p] = self.get_root(self.parent[p])
        return self.parent[p]

    def is_connected(self, p1: int, p2: int) -> bool:
        return self.get_root(p1) == self.get_root(p2)

    def union721(self, p1: int, p2: int) -> None:
        root1 = self.get_root(p1)
        root2 = self.get_root(p2)
        if root1 == root2:
            return
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
            self.size[root2] += self.size[root1]
        else:
            self.parent[root2] = root1
            self.size[root1] += self.size[root2]
            if self.rank[root1] == self.rank[root2]:
                self.rank[root1] += 1

    def get_size(self, p: int) -> int:
        return self.size[self.get_root(p)]

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

    # 1499. 满足不等式的最大值 (Max Value of Equation)
    def findMaxValueOfEquation(self, points: List[List[int]], k: int) -> int:
        q = deque()
        res = -inf
        for x, y in points:
            while q and x - q[0][0] > k:
                q.popleft()
            if q:
                res = max(res, x + y + q[0][1])
            while q and q[-1][1] <= y - x:
                q.pop()
            q.append((x, y - x))
        return res

    # 2641. 二叉树的堂兄弟节点 II (Cousins in Binary Tree II)
    def replaceValueInTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        q = deque()
        root.val = 0
        q.append(root)
        while q:
            size = len(q)
            s = 0
            for _ in range(size):
                x = q.popleft()
                if x.left:
                    s += x.left.val
                if x.right:
                    s += x.right.val
                q.append(x)
            for _ in range(size):
                x = q.popleft()
                cur = 0
                if x.left:
                    cur += x.left.val
                if x.right:
                    cur += x.right.val
                if x.left:
                    x.left.val = s - cur
                    q.append(x.left)
                if x.right:
                    x.right.val = s - cur
                    q.append(x.right)
        return root

    # 2811. 判断是否能拆分数组 (Check if it is Possible to Split Array)
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        return len(nums) <= 2 or any(x + y >= m for x, y in pairwise(nums))

    # 2835. 使子序列的和等于目标的最少操作次数 (Minimum Operations to Form Subsequence With Target Sum)
    def minOperations(self, nums: List[int], target: int) -> int:
        if sum(nums) < target:
            return -1
        cnt = Counter(nums)
        res = s = i = 0
        while 1 << i <= target:
            s += cnt[1 << i] << i
            mask = (1 << (i + 1)) - 1
            i += 1
            if s >= target & mask:
                continue
            res += 1
            while cnt[1 << i] == 0:
                res += 1
                i += 1
        return res

    # 94. 二叉树的中序遍历 (Binary Tree Inorder Traversal)
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(root: "TreeNode") -> None:
            if root is None:
                return
            dfs(root.left)
            res.append(root.val)
            dfs(root.right)

        res = []
        dfs(root)
        return res

    # 1456. 定长子串中元音的最大数目 (Maximum Number of Vowels in a Substring of Given Length)
    def maxVowels(self, s: str, k: int) -> int:
        mask = 0
        for v in "aeiou":
            mask |= 1 << (ord(v) - ord("a"))
        res = 0
        cur = 0
        for i, v in enumerate(s):
            if (mask >> (ord(v) - ord("a"))) & 1:
                cur += 1
            if i >= k:
                if (mask >> (ord(s[i - k]) - ord("a"))) & 1:
                    cur -= 1
            if i >= k - 1:
                res = max(res, cur)
        return res

    # 2269. 找到一个数字的 K 美丽值 (Find the K-Beauty of a Number)
    def divisorSubstrings(self, num: int, k: int) -> int:
        s = str(num)
        n = len(s)
        res = 0
        for i in range(k - 1, n):
            x = int(s[i - k + 1 : i + 1])
            if x and num % x == 0:
                res += 1
        return res

    # 2762. 不间断子数组 (Continuous Subarrays)
    def continuousSubarrays(self, nums: List[int]) -> int:
        cnt = Counter()
        res = 0
        i = 0
        n = len(nums)
        j = 0
        while i < n:
            cnt[nums[i]] += 1
            while max(cnt) - min(cnt) > 2:
                cnt[nums[j]] -= 1
                if cnt[nums[j]] == 0:
                    del cnt[nums[j]]
                j += 1
            res += i - j + 1
            i += 1
        return res

    # 1766. 互质树 (Tree of Coprimes)
    def getCoprimes(self, nums: List[int], edges: List[List[int]]) -> List[int]:
        def dfs(x: int, fa: int) -> None:
            cur_time = -1
            for i in primes[nums[x]]:
                if dic[i] and dic[i][-1][1] > cur_time:
                    cur_time = dic[i][-1][1]
                    res[x] = dic[i][-1][0]
            nonlocal time
            dic[nums[x]].append((x, time))
            time += 1
            for y in g[x]:
                if y != fa:
                    dfs(y, x)
            dic[nums[x]].pop()

        n = len(nums)
        g = [[] for _ in range(n)]
        dic = [[] for _ in range(51)]
        primes = [[] for _ in range(51)]
        for i in range(1, 51):
            for j in range(1, 51):
                if gcd(i, j) == 1:
                    primes[i].append(j)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        res = [-1] * n
        time = 0
        dfs(0, -1)
        return res

    # 144. 二叉树的前序遍历 (Binary Tree Preorder Traversal)
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []

        def dfs(root: "TreeNode") -> None:
            if not root:
                return
            res.append(root.val)
            dfs(root.left)
            dfs(root.right)

        dfs(root)
        return res

    # 145. 二叉树的后序遍历 (Binary Tree Postorder Traversal)
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []

        def dfs(root: "TreeNode") -> None:
            if root is None:
                return
            dfs(root.left)
            dfs(root.right)
            res.append(root.val)

        dfs(root)
        return res

    # 3033. 修改矩阵 (Modify the Matrix)
    def modifiedMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        n = len(matrix[0])
        m = len(matrix)
        for j in range(n):
            mx = -1
            for i in range(m):
                mx = max(mx, matrix[i][j])
            for i in range(m):
                if matrix[i][j] == -1:
                    matrix[i][j] = mx
        return matrix

    # 3035. 回文字符串的最大数量 (Maximum Palindromes After Operations)
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        cnt = Counter()
        for w in words:
            cnt += Counter(w)
        left = sum(c // 2 for c in cnt.values())
        words.sort(key=len)
        res = 0
        for w in words:
            m = len(w) // 2
            if left < m:
                break
            res += 1
            left -= m
        return res

    # 2223. 构造字符串的总得分 (Sum of Scores of Built Strings) -- z函数
    def sumScores(self, s: str) -> int:
        n = len(s)
        z = [0] * n
        left = right = 0
        res = n
        for i in range(1, n):
            if i <= right:
                z[i] = min(z[i - left], right - i + 1)
            while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                left, right = i, i + z[i]
                z[i] += 1
            res += z[i]
        return res

    # 3034. 匹配模式数组的子数组数目 I (Number of Subarrays That Match a Pattern I)
    # 3036. 匹配模式数组的子数组数目 II (Number of Subarrays That Match a Pattern II) --z函数
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        m = len(pattern)
        pattern.extend((y > x) - (y < x) for x, y in pairwise(nums))
        n = len(pattern)
        left = right = 0
        z = [0] * n
        for i in range(1, n):
            if i <= right:
                z[i] = min(z[i - left], right - i + 1)
            while i + z[i] < n and pattern[z[i]] == pattern[i + z[i]]:
                left, right = i, i + z[i]
                z[i] += 1
        return sum(lcp >= m for lcp in z[m:])

    # 987. 二叉树的垂序遍历 (Vertical Order Traversal of a Binary Tree)
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        def dfs(root: "TreeNode", i: int, j: int) -> None:
            if root is None:
                return
            nonlocal min_col
            min_col = min(min_col, j)
            d[j].append([i, root.val])
            dfs(root.left, i + 1, j - 1)
            dfs(root.right, i + 1, j + 1)

        res = []
        min_col = 0
        d = collections.defaultdict(list)
        dfs(root, 0, 0)
        for i in range(min_col, min_col + len(d)):
            d[i].sort()
            res.append([y for _, y in d[i]])
        return res

    # 2713. 矩阵中严格递增的单元格数 (Maximum Strictly Increasing Cells in a Matrix)
    def maxIncreasingCells(self, mat: List[List[int]]) -> int:
        m = len(mat)
        n = len(mat[0])
        d = collections.defaultdict(list)
        row_max = [0] * m
        col_max = [0] * n
        for i in range(m):
            for j in range(n):
                d[mat[i][j]].append((i, j))
        for key in sorted(d):
            mx = [max(row_max[i], col_max[j]) + 1 for i, j in d[key]]
            for (i, j), _max in zip(d[key], mx):
                row_max[i] = max(row_max[i], _max)
                col_max[j] = max(col_max[j], _max)
        return max(row_max)

    # 2478. 完美分割的方案数 (Number of Beautiful Partitions)
    def beautifulPartitions(self, s: str, k: int, l: int) -> int:
        def is_prime(c: str) -> bool:
            return c in "2357"

        def can_partition(j: int) -> bool:
            return j == 0 or j == n or not is_prime(s[j - 1]) and is_prime(s[j])

        n = len(s)
        if k * l > n or not is_prime(s[0]) or is_prime(s[-1]):  # 剪枝
            return 0
        MOD = 10**9 + 7
        dp = [[0] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 1
        for i in range(1, k + 1):
            sum = 0
            for j in range(i * l, n - (k - i) * l + 1):
                if can_partition(j - l):
                    sum += dp[i - 1][j - l]
                    sum %= MOD
                if can_partition(j):
                    dp[i][j] = sum
        return dp[k][n]

    # 2478. 完美分割的方案数 (Number of Beautiful Partitions)
    def beautifulPartitions(self, s: str, k: int, minLength: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n or j == k:
                return i == n and j == k
            res = 0
            for x in dic[i]:
                if n - x - 1 < (k - j - 1) * minLength:
                    break
                res += dfs(x + 1, j + 1)
            return res % MOD

        def is_prime(c: str) -> bool:
            return p & (1 << int(c))

        n = len(s)
        p = (1 << 2) | (1 << 3) | (1 << 5) | (1 << 7)
        if k * minLength > n or not is_prime(s[0]) or is_prime(s[-1]):
            return 0
        dic = [[] for _ in range(n)]
        for i in range(n):
            if is_prime(s[i]):
                for j in range(i + minLength - 1, n):
                    if j == n - 1 or not is_prime(s[j]) and is_prime(s[j + 1]):
                        dic[i].append(j)
        MOD = 10**9 + 7
        return dfs(0, 0)

    # 102. 二叉树的层序遍历 (Binary Tree Level Order Traversal)
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        res = []
        q = deque()
        q.append(root)
        while q:
            size = len(q)
            l = []
            for _ in range(size):
                x = q.popleft()
                l.append(x.val)
                if x.left:
                    q.append(x.left)
                if x.right:
                    q.append(x.right)
            res.append(l)
        return res

    # 102. 二叉树的层序遍历 (Binary Tree Level Order Traversal)
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        def dfs(root: "TreeNode", d: int) -> None:
            if root is None:
                return
            if d == len(res):
                res.append([])
            res[d].append(root.val)
            dfs(root.left, d + 1)
            dfs(root.right, d + 1)

        res = []
        dfs(root, 0)
        return res

    # 1984. 学生分数的最小差值 (Minimum Difference Between Highest and Lowest of K Scores)
    def minimumDifference(self, nums: List[int], k: int) -> int:
        nums.sort()
        return min(nums[i] - nums[i - k + 1] for i in range(k - 1, len(nums)))

    # 643. 子数组最大平均数 I (Maximum Average Subarray I)
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        res = -inf
        s = 0
        for i, v in enumerate(nums):
            s += v
            if i >= k:
                s -= nums[i - k]
            if i >= k - 1:
                res = max(res, s / k)
        return res

    # 1343. 大小为 K 且平均值大于等于阈值的子数组数目 (Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold)
    def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
        s = 0
        res = 0
        for i, v in enumerate(arr):
            s += v
            if i >= k:
                s -= arr[i - k]
            if i >= k - 1 and s // k >= threshold:
                res += 1
        return res

    # 2090. 半径为 k 的子数组平均值 (K Radius Subarray Averages)
    def getAverages(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        res = [-1] * n
        s = 0
        for i, v in enumerate(nums):
            s += v
            if i > 2 * k:
                s -= nums[i - 2 * k - 1]
            if i >= 2 * k:
                res[i - k] = s // (2 * k + 1)
        return res

    # 2379. 得到 K 个黑块的最少涂色次数 (Minimum Recolors to Get K Consecutive Black Blocks)
    def minimumRecolors(self, blocks: str, k: int) -> int:
        res = inf
        s = 0
        for i, v in enumerate(blocks):
            s += v == "W"
            if i >= k:
                s -= blocks[i - k] == "W"
            if i >= k - 1:
                res = min(res, s)
        return res

    # 1052. 爱生气的书店老板 (Grumpy Bookstore Owner)
    def maxSatisfied(
        self, customers: List[int], grumpy: List[int], minutes: int
    ) -> int:
        s1 = 0
        s2 = 0
        s = 0
        for i, (c, g) in enumerate(zip(customers, grumpy)):
            if g:
                s += c
            else:
                s1 += c
            if i >= minutes:
                if grumpy[i - minutes]:
                    s -= customers[i - minutes]
            if i >= minutes - 1:
                s2 = max(s2, s)
        return s1 + s2

    # 2841. 几乎唯一子数组的最大和 (Maximum Sum of Almost Unique Subarray)
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        res = 0
        s = 0
        d = collections.defaultdict(int)
        for i, v in enumerate(nums):
            s += v
            d[v] += 1
            if i >= k:
                s -= nums[i - k]
                d[nums[i - k]] -= 1
                if d[nums[i - k]] == 0:
                    del d[nums[i - k]]
            if i >= k - 1 and len(d) >= m:
                res = max(res, s)
        return res

    # 2461. 长度为 K 子数组中的最大和 (Maximum Sum of Distinct Subarrays With Length K)
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        res = 0
        s = 0
        d = collections.defaultdict(int)
        for i, v in enumerate(nums):
            s += v
            d[v] += 1
            if i >= k:
                s -= nums[i - k]
                d[nums[i - k]] -= 1
                if d[nums[i - k]] == 0:
                    del d[nums[i - k]]
            if i >= k - 1 and len(d) == k:
                res = max(res, s)
        return res

    # 1423. 可获得的最大点数 (Maximum Points You Can Obtain from Cards)
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        n = len(cardPoints)
        res = inf
        s = 0
        for i, v in enumerate(cardPoints):
            s += v
            if i >= n - k:
                s -= cardPoints[i - (n - k)]
            if i >= n - k - 1:
                res = min(res, s)
        return sum(cardPoints) - res

    # 2134. 最少交换次数来组合所有的 1 II (Minimum Swaps to Group All 1's Together II)
    def minSwaps(self, nums: List[int]) -> int:
        res = inf
        k = sum(nums)
        s = 0
        nums = nums + nums
        for i in range(len(nums)):
            s += nums[i] ^ 1
            if i >= k:
                s -= nums[i - k] ^ 1
            if i >= k - 1:
                res = min(res, s)
                if res == 0:
                    return res
        return res

    # 2653. 滑动子数组的美丽值 (Sliding Subarray Beauty)
    def getSubarrayBeauty(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        res = [0] * (n - k + 1)
        cnt = [0] * 101
        for i, v in enumerate(nums):
            cnt[v + 50] += 1
            if i >= k:
                cnt[nums[i - k] + 50] -= 1
            if i >= k - 1:
                c = 0
                for j in range(0, 50):
                    c += cnt[j]
                    if c >= x:
                        res[i - k + 1] = j - 50
                        break
        return res

    # 567. 字符串的排列 (Permutation in String)
    def checkInclusion(self, s1: str, s2: str) -> bool:
        n1, n2 = len(s1), len(s2)
        if n1 > n2:
            return False
        c1 = Counter(s1)
        c2 = Counter()
        for i, v in enumerate(s2):
            c2[v] += 1
            if i >= n1:
                c2[s2[i - n1]] -= 1
            if i >= n1 - 1 and c1 == c2:
                return True
        return False

    # 438. 找到字符串中所有字母异位词 (Find All Anagrams in a String)
    def findAnagrams(self, s: str, p: str) -> List[int]:
        n1, n2 = len(s), len(p)
        if n2 > n1:
            return []
        res = []
        c2 = Counter(p)
        c1 = Counter()
        for i, v in enumerate(s):
            c1[v] += 1
            if i >= n2:
                c1[s[i - n2]] -= 1
            if i >= n2 - 1 and c1 == c2:
                res.append(i - n2 + 1)
        return res

    # 2156. 查找给定哈希值的子串 (Find Substring With Given Hash Value)
    def subStrHash(
        self, s: str, power: int, modulo: int, k: int, hashValue: int
    ) -> str:
        p = [0] * k
        p[0] = 1
        n = len(s)
        res = n
        for i in range(1, k):
            p[i] = p[i - 1] * power % modulo
        h = 0
        for i in range(n - k, n):
            h += (ord(s[i]) - ord("a") + 1) * p[i - n + k] % modulo
            h %= modulo
        if h == hashValue:
            res = n - k
        for i in range(n - k - 1, -1, -1):
            h -= (ord(s[i + k]) - ord("a") + 1) * p[-1] % modulo
            h %= modulo
            h *= power
            h %= modulo
            h += (ord(s[i]) - ord("a") + 1) % modulo
            h %= modulo
            if h == hashValue:
                res = i
        return s[res : res + k]

    # 2953. 统计完全子字符串 (Count Complete Substrings)
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def cal() -> None:
            cnt = [0] * 26
            w = c * k
            for i in range(len(s)):
                cnt[ord(s[i]) - ord("a")] += 1
                if i >= w:
                    cnt[ord(s[i - w]) - ord("a")] -= 1
                if i >= w - 1 and all(not x or x == k for x in cnt):
                    nonlocal res
                    res += 1

        n = len(word)
        i = 0
        res = 0
        while i < n:
            j = i + 1
            while j < n and abs(ord(word[j]) - ord(word[j - 1])) <= 2:
                j += 1
            s = word[i:j]
            for c in range(1, 27):
                if c * k > len(s):
                    break
                cal()
            i = j
        return res

    # 1493. 删掉一个元素以后全为 1 的最长子数组 (Longest Subarray of 1's After Deleting One Element)
    def longestSubarray(self, nums: List[int]) -> int:
        res = 0
        n = len(nums)
        cnt0 = 0
        j = 0
        for i, v in enumerate(nums):
            cnt0 += 1 ^ v
            while cnt0 > 1:
                cnt0 -= nums[j] ^ 1
                j += 1
            res = max(res, i - j + 1 - cnt0)
        return min(n - 1, res)

    # 2730. 找到最长的半重复子字符串 (Find the Longest Semi-Repetitive Substring)
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        repeat = 0
        res = 0
        j = 0
        for i, v in enumerate(s):
            repeat += i > 0 and v == s[i - 1]
            while repeat > 1:
                repeat -= s[j] == s[j + 1]
                j += 1
            res = max(res, i - j + 1)
        return res

    # 904. 水果成篮 (Fruit Into Baskets)
    def totalFruit(self, fruits: List[int]) -> int:
        res = 0
        d = defaultdict(int)
        j = 0
        for i, v in enumerate(fruits):
            d[v] += 1
            while len(d) > 2:
                d[fruits[j]] -= 1
                if d[fruits[j]] == 0:
                    del d[fruits[j]]
                j += 1
            res = max(res, i - j + 1)
        return res

    # 1695. 删除子数组的最大得分 (Maximum Erasure Value)
    def maximumUniqueSubarray(self, nums: List[int]) -> int:
        cnts = [0] * (max(nums) + 1)
        s = 0
        res = 0
        j = 0
        for x in nums:
            cnts[x] += 1
            s += x
            while cnts[x] > 1:
                cnts[nums[j]] -= 1
                s -= nums[j]
                j += 1
            res = max(res, s)
        return res

    # 2958. 最多 K 个重复元素的最长子数组 (Length of Longest Subarray With at Most K Frequency)
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        d = defaultdict(int)
        res = 0
        j = 0
        for i, v in enumerate(nums):
            d[v] += 1
            while d[v] > k:
                d[nums[j]] -= 1
                j += 1
            res = max(res, i - j + 1)
        return res

    # 2024. 考试的最大困扰度 (Maximize the Confusion of an Exam)
    def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
        def check(arr: List[int]) -> int:
            cnt1 = 0
            j = 0
            res = 0
            for i, v in enumerate(arr):
                cnt1 += v
                while cnt1 > k:
                    cnt1 -= arr[j]
                    j += 1
                res = max(res, i - j + 1)
            return res

        arr1 = [1 if c == "T" else 0 for c in answerKey]
        arr2 = [1 if c == "F" else 0 for c in answerKey]
        return max(check(arr1), check(arr2))

    # 1004. 最大连续1的个数 III (Max Consecutive Ones III)
    def longestOnes(self, nums: List[int], k: int) -> int:
        res = 0
        cnt0 = 0
        j = 0
        for i, v in enumerate(nums):
            cnt0 += 1 ^ v
            while cnt0 > k:
                cnt0 -= 1 ^ nums[j]
                j += 1
            res = max(res, i - j + 1)
        return res

    # 1438. 绝对差不超过限制的最长连续子数组 (Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit)
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        d = defaultdict(int)
        j = 0
        res = 0
        for i, v in enumerate(nums):
            d[v] += 1
            while max(d) - min(d) > limit:
                d[nums[j]] -= 1
                if d[nums[j]] == 0:
                    del d[nums[j]]
                j += 1
            res = max(res, i - j + 1)
        return res

    # 2401. 最长优雅子数组 (Longest Nice Subarray)
    def longestNiceSubarray(self, nums: List[int]) -> int:
        j = 0
        m = 0
        res = 0
        for i, v in enumerate(nums):
            while m & v:
                m ^= nums[j]
                j += 1
            m ^= nums[i]
            res = max(res, i - j + 1)
        return res

    # 1658. 将 x 减到 0 的最小操作数 (Minimum Operations to Reduce X to Zero)
    def minOperations(self, nums: List[int], x: int) -> int:
        n = len(nums)
        j = 0
        k = sum(nums) - x
        if k < 0:
            return -1
        res = -1
        s = 0
        for i, v in enumerate(nums):
            s += v
            while s > k:
                s -= nums[j]
                j += 1
            if s == k:
                res = max(res, i - j + 1)
        return -1 if res == -1 else n - res

    # 1838. 最高频元素的频数 (Frequency of the Most Frequent Element)
    def maxFrequency(self, nums: List[int], k: int) -> int:
        nums.sort()
        j = 0
        res = 1
        s = 0
        for i in range(1, len(nums)):
            s += (nums[i] - nums[i - 1]) * (i - j)
            while s > k:
                s -= nums[i] - nums[j]
                j += 1
            res = max(res, i - j + 1)
        return res

    # 2516. 每种字符至少取 K 个 (Take K of Each Character From Left and Right) --二分
    def takeCharacters(self, s: str, k: int) -> int:
        def check(w: int) -> bool:
            cur = [0] * 3
            for i, c in enumerate(s):
                cur[ord(c) - ord("a")] += 1
                if i >= w:
                    cur[ord(s[i - w]) - ord("a")] -= 1
                if i >= w - 1 and all(c0 - c1 >= k for c0, c1 in zip(cnt, cur)):
                    return True
            return False

        cnt = [0] * 3
        for c in s:
            cnt[ord(c) - ord("a")] += 1
        if any(c < k for c in cnt):
            return -1
        res = -1
        left = 0
        right = len(s) - 3 * k
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = len(s) - mid
                left = mid + 1
            else:
                right = mid - 1
        return res

    # 2516. 每种字符至少取 K 个 (Take K of Each Character From Left and Right) --双指针
    def takeCharacters(self, s: str, k: int) -> int:
        n = len(s)
        cnt = [0] * 3
        for c in s:
            cnt[ord(c) - ord("a")] += 1
        for i in range(3):
            cnt[i] -= k
            if cnt[i] < 0:
                return -1
        res = n
        left = 0
        for right, v in enumerate(s):
            x = ord(v) - ord("a")
            cnt[x] -= 1
            while cnt[x] < 0:
                cnt[ord(s[left]) - ord("a")] += 1
                left += 1
            res = min(res, n - (right - left + 1))
        return res

    # 103. 二叉树的锯齿形层序遍历 (Binary Tree Zigzag Level Order Traversal)
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        q = deque([root])
        res = []
        while q:
            size = len(q)
            l = []
            for _ in range(size):
                x = q.popleft()
                l.append(x.val)
                if x.left:
                    q.append(x.left)
                if x.right:
                    q.append(x.right)
            res.append(l)
        return [l[::-1] if i & 1 else l for i, l in enumerate(res)]

    # 103. 二叉树的锯齿形层序遍历 (Binary Tree Zigzag Level Order Traversal)
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        def dfs(root: "TreeNode", d: int) -> None:
            if root is None:
                return
            if d == len(res):
                res.append([])
            res[d].append(root.val)
            dfs(root.left, d + 1)
            dfs(root.right, d + 1)

        res = []
        dfs(root, 0)
        return [l[::-1] if i & 1 else l for i, l in enumerate(res)]

    # 2831. 找出最长等值子数组 (Find the Longest Equal Subarray)
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        res = 0
        g = defaultdict(list)
        for i, v in enumerate(nums):
            g[v].append(i)
        for vals in g.values():
            j = 0
            m = len(vals)
            for i in range(m):
                while vals[i] - vals[j] + 1 - (i - j + 1) > k:
                    j += 1
                res = max(res, i - j + 1)
        return res

    # 2106. 摘水果 (Maximum Fruits Harvested After at Most K Steps)
    def maxTotalFruits(self, fruits: List[List[int]], startPos: int, k: int) -> int:
        j = 0
        res = 0
        s = 0
        for i, (p, c) in enumerate(fruits):
            s += c
            while (
                j <= i
                and p
                - fruits[j][0]
                + min(abs(startPos - p), abs(startPos - fruits[j][0]))
                > k
            ):
                s -= fruits[j][1]
                j += 1
            res = max(res, s)
        return res

    # 429. N 叉树的层序遍历 (N-ary Tree Level Order Traversal)
    def levelOrder(self, root: "Node") -> List[List[int]]:
        def dfs(root: "Node", d: int) -> None:
            if root is None:
                return
            if len(res) == d:
                res.append([])
            res[d].append(root.val)
            [dfs(x, d + 1) for x in root.children]

        res = []
        dfs(root, 0)
        return res

    # 429. N 叉树的层序遍历 (N-ary Tree Level Order Traversal)
    def levelOrder(self, root: "Node") -> List[List[int]]:
        if root is None:
            return []
        res = []
        q = deque([root])
        while q:
            size = len(q)
            _l = []
            for _ in range(size):
                x = q.popleft()
                _l.append(x.val)
                for y in x.children:
                    if y is None:
                        continue
                    q.append(y)
            res.append(_l)
        return res

    # 2968. 执行操作使频率分数最大 (Apply Operations to Maximize Frequency Score)
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        s = list(accumulate(nums, initial=0))
        j = 0
        res = 0
        for i in range(n):
            m = (i + j) // 2
            while (
                s[i + 1]
                - s[m]
                - nums[m] * (i - m + 1)
                + nums[m] * (m + 1 - j)
                - (s[m + 1] - s[j])
                > k
            ):
                j += 1
                m = (i + j) // 2
            res = max(res, i - j + 1)
        return res

    # 763. 最长的美好子字符串 (Longest Nice Substring)
    def longestNiceSubstring(self, s: str) -> str:
        n = len(s)
        left = 0
        l = 0
        for k in range(1, 27):
            if k * 2 > n:
                break
            m = [0] * 2
            j = 0
            d = defaultdict(int)
            for i, v in enumerate(s):
                m[(ord(v) >> 5) & 1] |= 1 << (ord(v) & 31)
                d[v] += 1
                while m[0].bit_count() > k or m[1].bit_count() > k:
                    d[s[j]] -= 1
                    if d[s[j]] == 0:
                        m[(ord(s[j]) >> 5) & 1] ^= 1 << (ord(s[j]) & 31)
                    j += 1
                if m[0] == m[1]:
                    if i - j + 1 > l:
                        l = i - j + 1
                        left = j
                    elif i - j + 1 == l and left > j:
                        left = j
        return "" if l == 0 else s[left : left + l]

    # 590. N 叉树的后序遍历 (N-ary Tree Postorder Traversal)
    def postorder(self, root: "Node") -> List[int]:
        res = []

        def dfs(root: "Node") -> None:
            if root is None:
                return
            for x in root.children:
                if x is None:
                    continue
                dfs(x)
            res.append(root.val)

        dfs(root)
        return res

    # 3038. 相同分数的最大操作数目 I (Maximum Number of Operations With the Same Score I)
    def maxOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return 0
        res = 1
        s = nums[0] + nums[1]
        i = 2
        while i + 1 < n and nums[i] + nums[i + 1] == s:
            i += 2
            res += 1
        return res

    # 3039. 进行操作使字符串为空 (Apply Operations to Make String Empty)
    def lastNonEmptyString(self, s: str) -> str:
        cnt = Counter(s)
        mx = max(cnt.values())
        res = []
        for i in range(len(s) - 1, -1, -1):
            if cnt[s[i]] == mx and s[i] not in res:
                res.append(s[i])
        return "".join(res[::-1])

    # 3040. 相同分数的最大操作数目 II (Maximum Number of Operations With the Same Score II)
    def maxOperations(self, nums: List[int]) -> int:
        def check(x: int) -> int:
            @cache
            def dfs(i: int, j: int) -> int:
                if i >= j:
                    return 0
                res = 0
                if nums[i] + nums[j] == x:
                    res = max(res, dfs(i + 1, j - 1) + 1)
                if nums[i] + nums[i + 1] == x:
                    res = max(res, dfs(i + 2, j) + 1)
                if nums[j] + nums[j - 1] == x:
                    res = max(res, dfs(i, j - 2) + 1)
                return res

            res = dfs(0, n - 1)
            dfs.cache_clear()
            return res

        n = len(nums)
        if n < 2:
            return 0
        res1 = check(nums[0] + nums[-1])
        res2 = check(nums[0] + nums[1])
        res3 = check(nums[-1] + nums[-2])
        return max(res1, res2, res3)

    # 3041. 修改数组后最大化数组中的连续元素数目 (Maximize Consecutive Elements in an Array After Modification)
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        f = defaultdict(int)
        for x in nums:
            f[x + 1] = f[x] + 1
            f[x] = f[x - 1] + 1
        return max(f.values())

    # 3043. 最长公共前缀的长度 (Find the Length of the Longest Common Prefix)
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        s = set()
        for x in arr1:
            a = str(x)
            for i in range(1, len(a) + 1):
                s.add(a[:i])
        res = 0
        for x in arr2:
            a = str(x)
            for i in range(1, len(a) + 1):
                if a[:i] in s:
                    res = max(res, i)
        return res

    # 3044. 出现频率最高的质数 (Most Frequent Prime)
    def mostFrequentPrime(self, mat: List[List[int]]) -> int:
        def is_prime(x: int) -> bool:
            return all(x % i for i in range(2, isqrt(x) + 1))

        m = len(mat)
        n = len(mat[0])
        d = defaultdict(int)
        for i in range(m):
            for j in range(n):
                for dx, dy in [
                    [0, 1],
                    [0, -1],
                    [1, 0],
                    [-1, 0],
                    [1, 1],
                    [-1, -1],
                    [1, -1],
                    [-1, 1],
                ]:
                    nx = i + dx
                    ny = j + dy
                    x = mat[i][j]
                    while nx >= 0 and nx < m and ny >= 0 and ny < n:
                        x = x * 10 + mat[nx][ny]
                        if x in d or is_prime(x):
                            d[x] += 1
                        nx += dx
                        ny += dy
        if len(d) == 0:
            return -1
        mx = max(d.values())
        res = 0
        for k, v in d.items():
            if v == mx:
                res = max(res, k)
        return res

    # 3042. 统计前后缀下标对 I (Count Prefix and Suffix Pairs I)
    # 3045. 统计前后缀下标对 II (Count Prefix and Suffix Pairs II) --字典树
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        class Trie:
            __slots__ = "son", "cnt"

            def __init__(self):
                self.son = dict()
                self.cnt = 0

        res = 0
        root = Trie()
        for w in words:
            node = root
            for p in zip(w, reversed(w)):
                if p not in node.son:
                    node.son[p] = Trie()
                node = node.son[p]
                res += node.cnt
            node.cnt += 1
        return res

    # 3042. 统计前后缀下标对 I (Count Prefix and Suffix Pairs I)
    # 3045. 统计前后缀下标对 II (Count Prefix and Suffix Pairs II) --字典树 + z函数
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        class Trie:
            __slots__ = "son", "cnt"

            def __init__(self):
                self.son = dict()
                self.cnt = 0

        res = 0
        root = Trie()
        for w in words:
            n = len(w)
            z = [0] * n
            left = right = 0
            for i in range(1, n):
                if i <= right:
                    z[i] = min(z[i - left], right - i + 1)
                while i + z[i] < n and w[z[i]] == w[i + z[i]]:
                    left = i
                    right = i + z[i]
                    z[i] += 1
            z[0] = n
            node = root
            for i, v in enumerate(w):
                if v not in node.son:
                    node.son[v] = Trie()
                node = node.son[v]
                if z[n - i - 1] == i + 1:
                    res += node.cnt
            node.cnt += 1
        return res

    # 1234. 替换子串得到平衡字符串 (Replace the Substring for Balanced String)
    def balancedString(self, s: str) -> int:
        d = defaultdict(int)
        for v in s:
            d[v] += 1
        res = n = len(s)
        limit = n // 4
        j = 0
        cur = defaultdict(int)
        for i, v in enumerate(s):
            cur[v] += 1
            while (
                d["Q"] - cur["Q"] <= limit
                and d["W"] - cur["W"] <= limit
                and d["E"] - cur["E"] <= limit
                and d["R"] - cur["R"] <= limit
            ):
                res = min(res, i - j + 1)
                cur[s[j]] -= 1
                j += 1
            if res == 0:
                break
        return res

    # 1574. 删除最短的子数组使剩余数组有序 (Shortest Subarray to be Removed to Make Array Sorted)
    def findLengthOfShortestSubarray(self, arr: List[int]) -> int:
        n = len(arr)
        r = n - 1
        while r >= 1:
            if arr[r - 1] > arr[r]:
                break
            r -= 1
        if r == 0:
            return 0
        res = r
        for l in range(n):
            if l > 0 and arr[l - 1] > arr[l]:
                break
            while r < n and arr[l] > arr[r]:
                r += 1
            res = min(res, r - l - 1)
        return res

    # 1358. 包含所有三种字符的子字符串数目 (Number of Substrings Containing All Three Characters)
    def numberOfSubstrings(self, s: str) -> int:
        res = 0
        j = 0
        cnt = [0] * 3
        m = 0
        for i, v in enumerate(s):
            x = ord(v) - ord("a")
            cnt[x] += 1
            m |= 1 << x
            while m == (1 << 3) - 1:
                res += len(s) - i
                cnt[ord(s[j]) - ord("a")] -= 1
                if cnt[ord(s[j]) - ord("a")] == 0:
                    m ^= 1 << (ord(s[j]) - ord("a"))
                j += 1
        return res

    # 2302. 统计得分小于 K 的子数组数目 (Count Subarrays With Score Less Than K)
    def countSubarrays(self, nums: List[int], k: int) -> int:
        j = 0
        res = 0
        pre = 0
        for i, v in enumerate(nums):
            pre += v
            while pre * (i - j + 1) >= k:
                pre -= nums[j]
                j += 1
            res += i - j + 1
        return res

    # 2537. 统计好子数组的数目 (Count the Number of Good Subarrays)
    def countGood(self, nums: List[int], k: int) -> int:
        d = defaultdict(int)
        j = 0
        c = 0
        res = 0
        for v in nums:
            c += d[v]
            d[v] += 1
            while c >= k:
                d[nums[j]] -= 1
                c -= d[nums[j]]
                j += 1
            res += j
        return res

    # 2970. 统计移除递增子数组的数目 I (Count the Number of Incremovable Subarrays I)
    # 2972. 统计移除递增子数组的数目 II (Count the Number of Incremovable Subarrays II)
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        i = 0
        while i < n - 1 and nums[i] < nums[i + 1]:
            i += 1
        if i == n - 1:
            return (1 + n) * n // 2
        res = i + 2
        j = n - 1
        while j == n - 1 or nums[j] < nums[j + 1]:
            while i >= 0 and nums[i] >= nums[j]:
                i -= 1
            res += i + 2
            j -= 1
        return res

    # 3046. 分割数组 (Split the Array)
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        return all(x <= 2 for x in Counter(nums).values())

    # 3047. 求交集区域内的最大正方形面积 (Find the Largest Area of Square Inside Two Rectangles)
    def largestSquareArea(
        self, bottomLeft: List[List[int]], topRight: List[List[int]]
    ) -> int:
        n = len(bottomLeft)
        res = 0
        for i in range(n):
            for j in range(i + 1, n):
                p1 = bottomLeft[i]
                p2 = topRight[i]
                p3 = bottomLeft[j]
                p4 = topRight[j]
                l1 = max(0, min(p2[0], p4[0]) - max(p1[0], p3[0]))
                l2 = max(0, min(p2[1], p4[1]) - max(p1[1], p3[1]))
                l = min(l1, l2)
                res = max(res, l * l)
        return res

    # 3048. 标记所有下标的最早秒数 I (Earliest Second to Mark Indices I)
    def earliestSecondToMarkIndices(
        self, nums: List[int], changeIndices: List[int]
    ) -> int:
        def check(x: int) -> bool:
            last_t = [-1] * n
            for i, v in enumerate(changeIndices[:x]):
                last_t[v - 1] = i
            if -1 in last_t:
                return False
            cnt = 0
            for i, v in enumerate(changeIndices[:x]):
                v -= 1
                if i == last_t[v]:
                    if nums[v] > cnt:
                        return False
                    cnt -= nums[v]
                else:
                    cnt += 1
            return True

        n = len(nums)
        m = len(changeIndices)
        left = 1
        right = m
        res = -1
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = mid
                right = mid - 1
            else:
                left = mid + 1
        return res

    # 938. 二叉搜索树的范围和 (Range Sum of BST)
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        if root is None:
            return 0
        if root.val < low:
            return self.rangeSumBST(root.right, low, high)
        if root.val > high:
            return self.rangeSumBST(root.left, low, high)
        return (
            root.val
            + self.rangeSumBST(root.left, low, high)
            + self.rangeSumBST(root.right, low, high)
        )

    # 938. 二叉搜索树的范围和 (Range Sum of BST) --bfs
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        q = deque([root])
        res = 0
        while q:
            x = q.popleft()
            if x is None:
                continue
            if x.val > high:
                q.append(x.left)
            elif x.val < low:
                q.append(x.right)
            else:
                res += x.val
                q.append(x.left)
                q.append(x.right)
        return res

    # 2867. 统计树中的合法路径数目 (Count Valid Paths in a Tree)
    def countPaths(self, n: int, edges: List[List[int]]) -> int:
        def dfs(x: int, fa: int) -> None:
            arr.append(x)
            for y in g[x]:
                if y != fa and not is_prime[y]:
                    dfs(y, x)

        g = [[] for _ in range(n + 1)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        is_prime = [True] * (n + 1)
        is_prime[1] = False
        #  埃式筛优化
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        cnt = [0] * (n + 1)
        for x in range(1, n + 1):
            if is_prime[x]:
                for y in g[x]:
                    if not is_prime[y] and cnt[y] == 0:
                        arr = []
                        dfs(y, 0)
                        for node in arr:
                            cnt[node] = len(arr)
        res = 0
        for x in range(1, n + 1):
            if is_prime[x]:
                c = 0
                for y in g[x]:
                    if not is_prime[y]:
                        res += c * cnt[y]
                        c += cnt[y]
                res += c
        return res

    # 2673. 使二叉树所有路径值相等的最小代价 (Make Costs of Paths Equal in a Binary Tree)
    def minIncrements(self, n: int, cost: List[int]) -> int:
        res = 0
        for i in range(n - 1, 0, -2):
            res += abs(cost[i] - cost[i - 1])
            cost[(i - 1) // 2] += max(cost[i], cost[i - 1])
        return res

    # 930. 和相同的二元子数组 (Binary Subarrays With Sum)
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        left1 = 0
        left2 = 0
        s1 = 0
        s2 = 0
        res = 0
        for i, v in enumerate(nums):
            s1 += v
            s2 += v
            while left1 <= i and s1 > goal:
                s1 -= nums[left1]
                left1 += 1
            while left2 <= i and s2 >= goal:
                s2 -= nums[left2]
                left2 += 1
            res += left2 - left1
        return res

    # 2369. 检查数组是否存在有效划分 (Check if There is a Valid Partition For The Array)
    def validPartition(self, nums: List[int]) -> bool:
        @cache
        def dfs(i: int) -> bool:
            if i == n:
                return True
            if i + 1 < n and nums[i] == nums[i + 1] and dfs(i + 2):
                return True
            if (
                i + 2 < n
                and (
                    nums[i] == nums[i + 1] == nums[i + 2]
                    or nums[i + 2] - nums[i + 1] == 1
                    and nums[i + 1] - nums[i] == 1
                )
                and dfs(i + 3)
            ):
                return True
            return False

        n = len(nums)
        return dfs(0)

    # 2368. 受限条件下可到达节点的数目 (Reachable Nodes With Restrictions)
    def reachableNodes(
        self, n: int, edges: List[List[int]], restricted: List[int]
    ) -> int:
        def dfs(x: int, fa: int) -> None:
            nonlocal res
            res += 1
            for y in g[x]:
                if y != fa and y not in s:
                    dfs(y, x)

        s = set(restricted)
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        res = 0
        dfs(0, -1)
        return res

    # 225. 用队列实现栈 (Implement Stack using Queues) --两个队列
    class MyStack:
        __slots__ = "q1", "q2"

        def __init__(self):
            self.q1 = deque()
            self.q2 = deque()

        def push(self, x: int) -> None:
            self.q2.append(x)
            while self.q1:
                self.q2.append(self.q1.popleft())
            temp = self.q1
            self.q1 = self.q2
            self.q2 = temp

        def pop(self) -> int:
            return self.q1.popleft()

        def top(self) -> int:
            return self.q1[0]

        def empty(self) -> bool:
            return not self.q1

    # 225. 用队列实现栈 (Implement Stack using Queues) --一个队列
    class MyStack:
        __slots__ = "q"

        def __init__(self):
            self.q = deque()

        def push(self, x: int) -> None:
            n = len(self.q)
            self.q.append(x)
            for _ in range(n):
                self.q.append(self.q.popleft())

        def pop(self) -> int:
            return self.q.popleft()

        def top(self) -> int:
            return self.q[0]

        def empty(self) -> bool:
            return not self.q

    # 232. 用栈实现队列 (Implement Queue using Stacks)
    class MyQueue:
        __slots__ = "list1", "list2"

        def __init__(self):
            self.list1 = []
            self.list2 = []

        def push(self, x: int) -> None:
            self.list2.append(x)

        def pop(self) -> int:
            if self.list1:
                return self.list1.pop()
            self.trans()
            return self.list1.pop()

        def peek(self) -> int:
            if self.list1:
                return self.list1[-1]
            self.trans()
            return self.list1[-1]

        def empty(self) -> bool:
            return not self.list1 and not self.list2

        def trans(self) -> None:
            while self.list2:
                self.list1.append(self.list2.pop())

    # 3065. 超过阈值的最少操作数 I (Minimum Operations to Exceed Threshold Value I)
    def minOperations(self, nums: List[int], k: int) -> int:
        return sum(x < k for x in nums)

    # 3065. 超过阈值的最少操作数 I (Minimum Operations to Exceed Threshold Value I)
    def minOperations(self, nums: List[int], k: int) -> int:
        def check() -> int:
            left = 0
            right = len(nums) - 1
            res = 0
            while left <= right:
                mid = left + ((right - left) >> 1)
                if nums[mid] >= k:
                    res = mid
                    right = mid - 1
                else:
                    left = mid + 1
            return res

        nums.sort()
        return check()

    # 3066. 超过阈值的最少操作数 II (Minimum Operations to Exceed Threshold Value II)
    def minOperations(self, nums: List[int], k: int) -> int:
        q = []
        heapq.heapify(q)
        for x in nums:
            heapq.heappush(q, x)
        res = 0
        while len(q) >= 2 and q[0] < k:
            res += 1
            x = heapq.heappop(q)
            y = heapq.heappop(q)
            heapq.heappush(q, x * 2 + y)
        return res

    # 3067. 在带权树网络中统计可连接服务器对数目 (Count Pairs of Connectable Servers in a Weighted Tree Network)
    def countPairsOfConnectableServers(
        self, edges: List[List[int]], signalSpeed: int
    ) -> List[int]:
        def dfs(x: int, fa: int, m: int) -> int:
            m %= signalSpeed
            ss = m == 0
            for y, w in g[x]:
                if y != fa:
                    ss += dfs(y, x, m + w)
            return ss

        n = len(edges) + 1
        g = [[] for _ in range(n)]
        for u, v, w in edges:
            g[u].append((v, w))
            g[v].append((u, w))
        res = [0] * n
        for i in range(n):
            cnt = 0
            s = 0
            for x, w in g[i]:
                cur = dfs(x, i, w)
                s += cur * cnt
                cnt += cur
            res[i] = s
        return res

    # 3068. 最大节点价值之和 (Find the Maximum Sum of Node Values)
    def maximumValueSum(self, nums: List[int], k: int, edges: List[List[int]]) -> int:
        def dfs(x: int, fa: int) -> tuple:
            s0 = nums[x]
            s1 = nums[x] ^ k
            for y in g[x]:
                if y != fa:
                    cur = dfs(y, x)
                    s0, s1 = max(s0 + cur[0], s1 + cur[1]), max(
                        s1 + cur[0], s0 + cur[1]
                    )
            return (s0, s1)

        n = len(nums)
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        return dfs(0, -1)[0]

    # 3068. 最大节点价值之和 (Find the Maximum Sum of Node Values) --不建树，求选择偶数个值与k异或的最大和
    def maximumValueSum(self, nums: List[int], k: int, edges: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0 if j == 0 else -inf
            return max(dfs(i + 1, j) + nums[i], dfs(i + 1, j ^ 1) + (nums[i] ^ k))

        n = len(nums)
        return dfs(0, 0)

    # 2575. 找出字符串的可整除数组 (Find the Divisibility Array of a String)
    def divisibilityArray(self, word: str, m: int) -> List[int]:
        n = len(word)
        res = [0] * n
        s = 0
        for i, v in enumerate(word):
            s = (s * 10 + int(v)) % m
            res[i] = int(s == 0)
        return res

    # 3069. 将元素分配到两个数组中 I (Distribute Elements Into Two Arrays I)
    def resultArray(self, nums: List[int]) -> List[int]:
        _l1 = [nums[0]]
        _l2 = [nums[1]]
        n = len(nums)
        for i in range(2, n):
            if _l1[-1] > _l2[-1]:
                _l1.append(nums[i])
            else:
                _l2.append(nums[i])
        return _l1 + _l2

    # 3070. 元素和小于等于 k 的子矩阵的数目 (Count Submatrices with Top-Left Element and Sum Less Than k)
    def countSubmatrices(self, grid: List[List[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])
        res = 0
        row = [0] * n
        for i in range(m):
            s = 0
            for j in range(n):
                s += grid[i][j]
                if s + row[j] <= k:
                    res += 1
                row[j] += s
        return res

    # 3070. 元素和小于等于 k 的子矩阵的数目 (Count Submatrices with Top-Left Element and Sum Less Than k)
    def countSubmatrices(self, grid: List[List[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])
        res = 0
        pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                pre[i][j] = (
                    pre[i][j - 1]
                    + pre[i - 1][j]
                    - pre[i - 1][j - 1]
                    + grid[i - 1][j - 1]
                )
                if pre[i][j] <= k:
                    res += 1
        return res

    # 3071. 在矩阵上写出字母 Y 所需的最少操作次数 (Minimum Operations to Write the Letter Y on a Grid)
    def minimumOperationsToWriteY(self, grid: List[List[int]]) -> int:
        def check(i: int, j: int) -> int:
            cur = 0
            for r in range(3):
                if i != r:
                    cur += in_cnt[r]
                if j != r:
                    cur += out_cnt[r]
            return cur

        n = len(grid)
        in_cnt = [0] * 3
        out_cnt = [0] * 3
        for i in range(n):
            for j in range(n):
                if (
                    i <= n // 2
                    and (i == j or i + j == n - 1)
                    or i >= n // 2
                    and j == n // 2
                ):
                    in_cnt[grid[i][j]] += 1
                else:
                    out_cnt[grid[i][j]] += 1
        res = inf
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                res = min(res, check(i, j))
        return res

    # 3072. 将元素分配到两个数组中 II (Distribute Elements Into Two Arrays II)
    def resultArray(self, nums: List[int]) -> List[int]:
        def greater_count(arr: List[int], target: int) -> List[int]:
            n = len(arr)
            if target >= arr[-1]:
                return [0, n]
            if target < arr[0]:
                return [n, 0]
            left = 0
            right = n - 1
            res = 0
            while left <= right:
                mid = left + ((right - left) >> 1)
                if arr[mid] > target:
                    res = n - mid
                    right = mid - 1
                else:
                    left = mid + 1
            return [res, n - res]

        res1 = [nums[0]]
        res2 = [nums[1]]
        arr1 = [nums[0]]
        arr2 = [nums[1]]
        for v in nums[2:]:
            [cnt1, id1] = greater_count(arr1, v)
            [cnt2, id2] = greater_count(arr2, v)
            if cnt1 > cnt2 or cnt1 == cnt2 and len(res1) <= len(res2):
                res1.append(v)
                arr1.insert(id1, v)
            else:
                res2.append(v)
                arr2.insert(id2, v)

        res1.extend(res2)
        return res1

    # 2834. 找出美丽数组的最小和 (Find the Minimum Possible Sum of a Beautiful Array)
    def minimumPossibleSum(self, n: int, target: int) -> int:
        m = min(n, target // 2)
        res = (1 + m) * m // 2
        m = n - m
        res += (target + target + m - 1) * m // 2
        return res % (10**9 + 7)

    # 9. 回文数 (Palindrome Number)
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        ori = x
        cur = 0
        while x:
            cur = cur * 10 + x % 10
            x //= 10
        return cur == ori

    # 299. 猜数字游戏 (Bulls and Cows)
    def getHint(self, secret: str, guess: str) -> str:
        cnt_s = [0] * 10
        cnt_g = [0] * 10
        a_cnt = 0
        for s, g in zip(secret, guess):
            a_cnt += s == g
            cnt_s[int(s)] += 1
            cnt_g[int(g)] += 1
        b_cnt = 0
        for c1, c2 in zip(cnt_s, cnt_g):
            b_cnt += min(c1, c2)
        return str(a_cnt) + "A" + str(b_cnt - a_cnt) + "B"

    # 3074. 重新分装苹果 (Apple Redistribution into Boxes)
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        s = sum(apple)
        capacity.sort(reverse=True)
        res = 0
        cur = 0
        for c in capacity:
            if cur >= s:
                break
            cur += c
            res += 1
        return res

    # 3075. 幸福值最大化的选择方案 (Maximize Happiness of Selected Children)
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        happiness.sort(reverse=True)
        res = 0
        d = 0
        for h in happiness:
            if h - d <= 0 or k == 0:
                break
            res += h - d
            d += 1
            k -= 1
        return res

    # 3076. 数组中的最短非公共子字符串 (Shortest Uncommon Substring in an Array)
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        d = defaultdict(int)
        for a in arr:
            for i in range(len(a)):
                for j in range(i, len(a)):
                    d[a[i : j + 1]] += 1
        res = []
        for a in arr:
            n = len(a)
            for i in range(n):
                for j in range(i, n):
                    d[a[i : j + 1]] -= 1
            cur = ""
            for i in range(n):
                for j in range(i, n):
                    s = a[i : j + 1]
                    if d[s] == 0:
                        if cur == "":
                            cur = s
                        elif len(cur) > len(s):
                            cur = s
                        elif len(cur) == len(s) and s < cur:
                            cur = s
            for i in range(n):
                for j in range(i, n):
                    d[a[i : j + 1]] += 1
            res.append(cur)
        return res

    # 2129. 将标题首字母大写 (Capitalize the Title)
    def capitalizeTitle(self, s: str) -> str:
        res = []
        i = 0
        while i < len(s):
            j = i
            while j < len(s) and s[j] != " ":
                res.append(s[j].lower())
                j += 1
            if j - i > 2:
                res[i] = s[i].upper()
            if j < len(s):
                res.append(" ")
            i = j + 1
        return "".join(res)

    # 3077. K 个不相交子数组的最大能量值 (Maximum Strength of K Disjoint Subarrays)
    def maximumStrength(self, nums: List[int], k: int) -> int:
        @cache
        def dfs(i: int, j: int, p: int) -> int:
            if j > k:
                return 0
            if i == n:
                return 0 if j == k else -inf
            if n - i < k - j:
                return -inf
            if p == 0:
                return max(
                    dfs(i + 1, j, p),
                    dfs(i + 1, j + 1, 1)
                    + nums[i] * (1 if (j + 1) % 2 == 1 else -1) * (k - j),
                )
            return max(
                dfs(i + 1, j, 0),
                dfs(i + 1, j, p) + nums[i] * (1 if j % 2 == 1 else -1) * (k - j + 1),
                dfs(i + 1, j + 1, p)
                + nums[i] * (1 if (j + 1) % 2 == 1 else -1) * (k - j),
            )

        n = len(nums)
        res = dfs(0, 0, 0)
        dfs.cache_clear()
        return res

    # 1261. 在受污染的二叉树中查找元素 (Find Elements in a Contaminated Binary Tree)
    class FindElements:
        __slots__ = "s"

        def dfs(self, root: "TreeNode", x: int) -> None:
            if root is None:
                return
            self.s.add(x)
            self.dfs(root.left, x * 2 + 1)
            self.dfs(root.right, x * 2 + 2)

        def __init__(self, root: Optional[TreeNode]):
            self.s = set()
            self.dfs(root, 0)

        def find(self, target: int) -> bool:
            return target in self.s

    # 1261. 在受污染的二叉树中查找元素 (Find Elements in a Contaminated Binary Tree)
    class FindElements:

        def __init__(self, root: Optional[TreeNode]):
            self.root = root

        def find(self, target: int) -> bool:
            node = self.root
            target += 1
            for i in range(target.bit_length() - 2, -1, -1):
                bit = (target >> i) & 1
                node = node.right if bit else node.left
                if node is None:
                    return False
            return True

    # 2312. 卖木头块 (Selling Pieces of Wood)
    def sellingWood(self, m: int, n: int, prices: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == 0 or j == 0:
                return 0
            res = s[(i, j)]
            for x in range(1, i):
                res = max(res, dfs(x, j) + dfs(i - x, j))
            for y in range(1, j):
                res = max(res, dfs(i, y) + dfs(i, j - y))
            return res

        s = defaultdict(int)
        for h, w, p in prices:
            s[(h, w)] = p
        return dfs(m, n)

    # 2684. 矩阵中移动的最大次数 (Maximum Number of Moves in a Grid)
    def maxMoves(self, grid: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j == n - 1:
                return 0
            return max(
                dfs(x, j + 1) + 1 if grid[x][j + 1] > grid[i][j] else 0
                for x in range(max(i - 1, 0), min(i + 2, m))
            )

        m = len(grid)
        n = len(grid[0])
        return max(dfs(i, 0) for i in range(m))

    # 310. 最小高度树 (Minimum Height Trees)
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n == 1:
            return [0]
        g = [[] for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
            deg[u] += 1
            deg[v] += 1
        res = []
        q = deque()
        for i in range(n):
            if deg[i] == 1:
                q.append(i)
        while q:
            s = len(q)
            res.clear()
            for _ in range(s):
                x = q.popleft()
                res.append(x)
                for y in g[x]:
                    deg[y] -= 1
                    if deg[y] == 1:
                        q.append(y)
        return res

    # 303. 区域和检索 - 数组不可变 (Range Sum Query - Immutable)
    class NumArray:

        def __init__(self, nums: List[int]):
            self.s = list(accumulate(nums, initial=0))

        def sumRange(self, left: int, right: int) -> int:
            return self.s[right + 1] - self.s[left]

    # 3079. 求出加密整数的和 (Find the Sum of Encrypted Integers)
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        res = 0
        for x in nums:
            cur = 0
            mx = max(int(c) for c in str(x))
            for _ in range(len(str(x))):
                cur = cur * 10 + mx
            res += cur
        return res

    # 3080. 执行操作标记数组中的元素 (Mark Elements on Array by Performing Queries)
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        m = len(queries)
        vis = [False] * n
        s = sum(nums)
        res = [0] * m
        j = 0
        arr = sorted(zip(range(n), nums), key=lambda o: (o[1], o[0]))
        for i, (index, k) in enumerate(queries):
            if not vis[index]:
                vis[index] = True
                s -= nums[index]
            while k and j < n:
                if not vis[arr[j][0]]:
                    vis[arr[j][0]] = True
                    s -= arr[j][1]
                    k -= 1
                j += 1
            res[i] = s
        return res

    # 3082. 求出所有子序列的能量和 (Find the Sum of the Power of All Subsequences)
    def sumOfPower(self, nums: List[int], k: int) -> int:
        @cache
        def dfs(i: int, j: int, c: int) -> int:
            if j >= k or c == 0 or i == n or n - i < c:
                return j == k and c == 0
            return (dfs(i + 1, j, c) + dfs(i + 1, j + nums[i], c - 1)) % MOD

        n = len(nums)
        MOD = 10**9 + 7
        return (
            sum(dfs(0, 0, i) * pow(2, n - i, MOD) % MOD for i in range(1, n + 1)) % MOD
        )

    # 1793. 好子数组的最大分数 (Maximum Score of a Good Subarray) --单调栈
    def maximumScore(self, nums: List[int], k: int) -> int:
        n = len(nums)
        left = [-1] * n
        st = []
        for i, v in enumerate(nums):
            while st and nums[st[-1]] >= v:
                st.pop()
            if st:
                left[i] = st[-1]
            st.append(i)
        right = [n] * n
        st = []
        for i in range(n - 1, -1, -1):
            while st and nums[st[-1]] >= nums[i]:
                st.pop()
            if st:
                right[i] = st[-1]
            st.append(i)
        res = 0
        for i, l, r in zip(range(n), left, right):
            if l < k < r:
                res = max(res, nums[i] * (r - l - 1))
        return res

    # 84. 柱状图中最大的矩形 (Largest Rectangle in Histogram)
    def largestRectangleArea(self, heights: List[int]) -> int:
        n = len(heights)
        st = []
        left = [-1] * n
        for i in range(n):
            while st and heights[st[-1]] >= heights[i]:
                st.pop()
            if st:
                left[i] = st[-1]
            st.append(i)
        st.clear()
        right = [n] * n
        for i in range(n - 1, -1, -1):
            while st and heights[st[-1]] >= heights[i]:
                st.pop()
            if st:
                right[i] = st[-1]
            st.append(i)
        return max(v * (r - l - 1) for v, l, r in zip(heights, left, right))

    # 3083. 字符串及其反转中是否存在同一子字符串 (Existence of a Substring in a String and Its Reverse)
    def isSubstringPresent(self, s: str) -> bool:
        vis = [0] * 26
        for x, y in pairwise(s):
            vis[ord(x) - ord("a")] |= 1 << (ord(y) - ord("a"))
            if vis[ord(y) - ord("a")] >> (ord(x) - ord("a")) & 1:
                return True
        return False

    # 3084. 统计以给定字符开头和结尾的子字符串总数 (Count Substrings Starting and Ending with Given Character)
    def countSubstrings(self, s: str, c: str) -> int:
        cnt = s.count(c)
        return (cnt + 1) * cnt // 2

    # 3085. 成为 K 特殊字符串需要删除的最少字符数 (Minimum Deletions to Make String K-Special)
    def minimumDeletions(self, word: str, k: int) -> int:
        cnts = [0] * 26
        for c in word:
            cnts[ord(c) - ord("a")] += 1
        cnts.sort()
        left = 0
        s = 0
        res = 0
        for right, v in enumerate(cnts):
            s += v
            while cnts[left] < cnts[right]:
                s -= cnts[left]
                left += 1
            cur = 0
            for i in range(right + 1, 26):
                cur += min(cnts[i], cnts[right] + k)
            res = max(res, s + cur)
        return len(word) - res

    # 1969. 数组元素的最小非零乘积 (Minimum Non-Zero Product of the Array Elements)
    def minNonZeroProduct(self, p: int) -> int:
        MOD = 10**9 + 7
        k = (1 << p) - 1
        return k * pow(k - 1, k >> 1, MOD) % MOD

    # 2671. 频率跟踪器 (Frequency Tracker)
    class FrequencyTracker:
        __slots__ = "cnt", "freq"

        def __init__(self):
            self.cnt = defaultdict(int)
            self.freq = defaultdict(int)

        def add(self, number: int) -> None:
            c = self.cnt[number]
            self.freq[c] -= 1
            self.freq[c + 1] += 1
            self.cnt[number] += 1

        def deleteOne(self, number: int) -> None:
            c = self.cnt[number]
            self.freq[c] -= 1
            self.freq[c - 1] += 1
            self.cnt[number] = max(0, self.cnt[number] - 1)

        def hasFrequency(self, frequency: int) -> bool:
            return self.freq[frequency] > 0

    # 740. 删除并获得点数 (Delete and Earn)
    def deleteAndEarn(self, nums: List[int]) -> int:
        c = Counter(nums)
        mx = max(nums)

        @cache
        def dfs(i: int) -> int:
            if i > mx:
                return 0
            return max(dfs(i + 2) + c[i] * i, dfs(i + 1))

        return dfs(1)

    # 2597. 美丽子集的数目 (The Number of Beautiful Subsets)
    def beautifulSubsets(self, nums: List[int], k: int) -> int:
        def dfs(i: int, j: int) -> None:
            if i == n:
                nonlocal res
                res += j
                return
            dfs(i + 1, j)
            if d[nums[i] - k] == 0 and d[nums[i] + k] == 0:
                d[nums[i]] += 1
                dfs(i + 1, 1)
                d[nums[i]] -= 1

        n = len(nums)
        d = defaultdict(int)
        res = 0
        dfs(0, 0)
        return res

    # 2597. 美丽子集的数目 (The Number of Beautiful Subsets)
    def beautifulSubsets(self, nums: List[int], k: int) -> int:
        @cache
        def dfs(i: int) -> int:
            if i < 0:
                return 1
            # 不选
            res = dfs(i - 1)
            # 选
            if i and g[i][0] - g[i - 1][0] == k:
                res += dfs(i - 2) * ((1 << g[i][1]) - 1)
            else:
                res += dfs(i - 1) * ((1 << g[i][1]) - 1)
            return res

        d = defaultdict(Counter)
        for x in nums:
            d[x % k][x] += 1
        res = 1
        for x in d.values():
            g = sorted(x.items())
            res *= dfs(len(g) - 1)
            # 需要清缓存 否则会复用上一个dfs的结果
            dfs.cache_clear()
        return res - 1

    # 2606. 找到最大开销的子字符串 (Find the Substring With Maximum Cost) --前缀和
    def maximumCostSubstring(self, s: str, chars: str, vals: List[int]) -> int:
        val = [i + 1 for i in range(26)]
        for c, v in zip(chars, vals):
            val[ord(c) - ord("a")] = v
        res = 0
        pre = 0
        m = 0
        for c in s:
            pre += val[ord(c) - ord("a")]
            res = max(res, pre - m)
            m = min(m, pre)
        return res

    # 2606. 找到最大开销的子字符串 (Find the Substring With Maximum Cost) --dp
    def maximumCostSubstring(self, s: str, chars: str, vals: List[int]) -> int:
        @cache
        def dfs(i: int) -> int:
            if i < 0:
                return 0
            return max(dfs(i - 1), 0) + val[ord(s[i]) - ord("a")]

        val = [i + 1 for i in range(26)]
        for c, v in zip(chars, vals):
            val[ord(c) - ord("a")] = v
        return max(0, max(dfs(i) for i in range(len(s))))

    # 2549. 统计桌面上的不同数字 (Count Distinct Numbers on Board)
    def distinctIntegers(self, n: int) -> int:
        return max(1, n - 1)

    # 1742. 盒子中小球的最大数量 (Maximum Number of Balls in a Box)
    def countBalls(self, lowLimit: int, highLimit: int) -> int:
        def check(x: int, _s: int) -> int:
            @cache
            def dfs(i: int, j: int, is_limit: bool) -> int:
                if i == n:
                    return j == 0
                res = 0
                up = int(s[i]) if is_limit else 9
                for d in range(min(j, up) + 1):
                    res += dfs(i + 1, j - d, d == up and is_limit)
                return res

            s = str(x)
            n = len(s)
            return dfs(0, _s, True)

        l = len(str(highLimit))
        return max(
            check(highLimit, i) - check(lowLimit - 1, i) for i in range(1, l * 9 + 1)
        )

    # 3090. 每个字符最多出现两次的最长子字符串 (Maximum Length Substring With Two Occurrences)
    def maximumLengthSubstring(self, s: str) -> int:
        res = 0
        j = 0
        cnt = [0] * 26
        for i, v in enumerate(s):
            cnt[ord(v) - ord("a")] += 1
            while cnt[ord(v) - ord("a")] > 2:
                cnt[ord(s[j]) - ord("a")] -= 1
                j += 1
            res = max(res, i - j + 1)
        return res

    # 3091. 执行操作使数据元素之和大于等于 K (Apply Operations to Make Sum of Array Greater Than or Equal to k)
    def minOperations(self, k: int) -> int:
        res = inf
        for add in range(k + 1):
            v = add + 1
            cur = add + (k - 1) // v
            res = min(res, cur)
        return res

    # 3092. 最高频率的 ID (Most Frequent IDs)
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        cnt = Counter()
        sl = SortedList()
        res = []
        for x, f in zip(nums, freq):
            if cnt[x] in sl:
                sl.remove(cnt[x])
            cnt[x] += f
            sl.add(cnt[x])
            res.append(sl[-1])
        return res

    # 3093. 最长公共后缀查询 (Longest Common Suffix Queries)
    def stringIndices(
        self, wordsContainer: List[str], wordsQuery: List[str]
    ) -> List[int]:
        class Trie:
            def __init__(self) -> None:
                self.children = [None] * 26
                self.index = -1
                self.len = inf

            def insert(self, s: str, id: int, l: int) -> None:
                node = self
                if l < node.len:
                    node.len = l
                    node.index = id
                for i in range(len(s) - 1, -1, -1):
                    j = ord(s[i]) - ord("a")
                    if node.children[j] is None:
                        node.children[j] = Trie()
                    node = node.children[j]
                    if l < node.len:
                        node.len = l
                        node.index = id

            def check(self, s: str) -> int:
                node = self
                for i in range(len(s) - 1, -1, -1):
                    j = ord(s[i]) - ord("a")
                    if node.children[j] is None:
                        break
                    node = node.children[j]
                return node.index

        root = Trie()
        for id, s in enumerate(wordsContainer):
            root.insert(s, id, len(s))
        res = []
        for w in wordsQuery:
            res.append(root.check(w))
        return res

    # 363. 矩形区域不超过 K 的最大数值和 (Max Sum of Rectangle No Larger Than K)
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        res = -inf
        m = len(matrix)
        n = len(matrix[0])
        for left in range(n):
            pre = [0] * m
            for right in range(left, n):
                cur_pre = 0
                s = SortedList([0])
                for i in range(m):
                    pre[i] += matrix[i][right]
                    cur_pre += pre[i]
                    lb = s.bisect_left(cur_pre - k)
                    if lb != len(s):
                        res = max(res, cur_pre - s[lb])
                    s.add(cur_pre)
        return res

    # 2642. 设计可以求最短路径的图类 (Design Graph With Shortest Path Calculator)
    class Graph:

        def __init__(self, n: int, edges: List[List[int]]):
            self.g = [[] for _ in range(n)]
            self.n = n
            for u, v, c in edges:
                self.g[u].append((v, c))

        def addEdge(self, edge: List[int]) -> None:
            self.g[edge[0]].append((edge[1], edge[2]))

        def shortestPath(self, node1: int, node2: int) -> int:
            d = [inf] * self.n
            d[node1] = 0
            q = [node1]
            while q:
                x = q.pop()
                for y, c in self.g[x]:
                    if d[x] + c < d[y]:
                        d[y] = d[x] + c
                        q.append(y)
            return -1 if d[node2] == inf else d[node2]

    # 1594. 矩阵的最大非负积 (Maximum Non Negative Product in a Matrix)
    def maxProductPath(self, grid: List[List[int]]) -> int:
        #  从[i, j, 1] 出发，到右下角。可构成的最大积
        #  从[i, j, 0] 出发，到右下角。可构成的最小积
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i == m - 1 and j == n - 1:
                return grid[i][j]
            res = -inf if k else inf
            if k:
                if grid[i][j] < 0:
                    if i + 1 < m:
                        res = max(res, dfs(i + 1, j, 0) * grid[i][j])
                    if j + 1 < n:
                        res = max(res, dfs(i, j + 1, 0) * grid[i][j])
                else:
                    if i + 1 < m:
                        res = max(res, dfs(i + 1, j, 1) * grid[i][j])
                    if j + 1 < n:
                        res = max(res, dfs(i, j + 1, 1) * grid[i][j])
            else:
                if grid[i][j] < 0:
                    if i + 1 < m:
                        res = min(res, dfs(i + 1, j, 1) * grid[i][j])
                    if j + 1 < n:
                        res = min(res, dfs(i, j + 1, 1) * grid[i][j])
                else:
                    if i + 1 < m:
                        res = min(res, dfs(i + 1, j, 0) * grid[i][j])
                    if j + 1 < n:
                        res = min(res, dfs(i, j + 1, 0) * grid[i][j])
            return res

        m = len(grid)
        n = len(grid[0])
        MOD = 10**9 + 7
        res = dfs(0, 0, 1)
        return -1 if res < 0 else res % MOD
    
    # 1594. 矩阵的最大非负积 (Maximum Non Negative Product in a Matrix)
    def maxProductPath(self, grid: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> List[int]:
            if i == m or j == n:
                return [inf, -inf]
            s = grid[i][j]
            if i == m - 1 and j == n - 1:
                return [s, s]
            _min1, _max1 = dfs(i + 1, j)
            _min2, _max2 = dfs(i, j + 1)
            res = [min(_min1, _min2) * s, max(_max1, _max2) * s]
            return res if s >= 0 else res[::-1]

        m, n = len(grid), len(grid[0])
        _, _max = dfs(0, 0)
        MOD = 10**9 + 7
        return -1 if _max < 0 else _max % MOD

    # 741. 摘樱桃 (Cherry Pickup)
    def cherryPickup(self, grid: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            l = i + j - k
            if (
                i == n
                or j == n
                or k == n
                or l == n
                or grid[i][j] == -1
                or grid[k][l] == -1
            ):
                return -inf
            if i == n - 1 and j == n - 1:
                return grid[i][j]
            return (
                max(
                    dfs(i + 1, j, k + 1),
                    dfs(i + 1, j, k),
                    dfs(i, j + 1, k + 1),
                    dfs(i, j + 1, k),
                )
                + grid[i][j]
                + (grid[k][l] if i != k or j != l else 0)
            )

        n = len(grid)
        return max(0, dfs(0, 0, 0))

    # 2321. 拼接数组的最大分数 (Maximum Score Of Spliced Array)
    def maximumsSplicedArray(self, nums1: List[int], nums2: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            if j == 0:
                return max(dfs(i + 1, j) + nums1[i], dfs(i + 1, 1) + nums2[i])
            if j == 1:
                return max(dfs(i + 1, j) + nums2[i], dfs(i + 1, 2) + nums1[i])
            return dfs(i + 1, j) + nums1[i]

        n = len(nums1)
        res1 = dfs(0, 0)
        nums1, nums2 = nums2, nums1
        dfs.cache_clear()
        res2 = dfs(0, 0)
        return max(res1, res2)

    # 2272. 最大波动的子字符串 (Substring With Largest Variance)
    def largestVariance(self, s: str) -> int:
        res = 0
        for a, b in permutations(ascii_lowercase, 2):
            diff = 0
            diff_with_b = -inf
            for c in s:
                if c == a:
                    diff += 1
                    diff_with_b += 1
                elif c == b:
                    diff -= 1
                    diff_with_b = diff
                    diff = max(diff, 0)
            res = max(res, diff_with_b)
        return res

    # 顺丰02. 小哥派件装载问题
    def minRemainingSpace(self, N: List[int], V: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i < 0:
                return j
            res = dfs(i - 1, j)
            if j - N[i] >= 0:
                res = min(res, dfs(i - 1, j - N[i]))
            return res

        return dfs(len(N) - 1, V)

    # 416. 分割等和子集 (Partition Equal Subset Sum)
    def canPartition(self, nums: List[int]) -> bool:
        @cache
        def dfs(i: int, j: int) -> bool:
            if j == s:
                return True
            if i == n or j > s:
                return False
            return dfs(i + 1, j) or dfs(i + 1, j + nums[i])

        s = sum(nums)
        if s & 1:
            return False
        n = len(nums)
        s >>= 1
        return dfs(0, 0)

    # 174. 地下城游戏 (Dungeon Game)
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == m - 1 and j == n - 1:
                return max(1, 1 - dungeon[i][j])
            if i == m or j == n:
                return inf
            return max(1, min(dfs(i + 1, j), dfs(i, j + 1)) - dungeon[i][j])

        m = len(dungeon)
        n = len(dungeon[0])
        return dfs(0, 0)

    # 1049. 最后一块石头的重量 II (Last Stone Weight II)
    def lastStoneWeightII(self, stones: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> bool:
            if i == n:
                return abs(j)
            return min(dfs(i + 1, j - stones[i]), dfs(i + 1, j + stones[i]))

        n = len(stones)
        return dfs(0, 0)

    # 3098. 求出所有子序列的能量和 (Find the Sum of Subsequence Powers)
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        @cache
        def dfs(i: int, j: int, pre: int, min_diff: int) -> int:
            if i + 1 < j:
                return 0
            if j == 0:
                return min_diff
            return (
                dfs(i - 1, j, pre, min_diff)
                + dfs(i - 1, j - 1, nums[i], min(min_diff, pre - nums[i]))
            ) % MOD

        n = len(nums)
        nums.sort()
        MOD = 10**9 + 7
        return dfs(n - 1, k, inf, inf)

    # 3095. 或值至少 K 的最短子数组 I (Shortest Subarray With OR at Least K I)
    # 3097. 或值至少为 K 的最短子数组 II (Shortest Subarray With OR at Least K II)
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        cnt = [0] * 30
        res = inf
        j = 0
        cur = 0
        for i, x in enumerate(nums):
            cur |= x
            while x:
                cnt[(x & -x).bit_length() - 1] += 1
                x &= x - 1
            while cur >= k and j <= i:
                res = min(res, i - j + 1)
                while nums[j]:
                    lb = (nums[j] & -nums[j]).bit_length() - 1
                    cnt[lb] -= 1
                    if cnt[lb] == 0:
                        cur ^= 1 << lb
                    nums[j] &= nums[j] - 1
                j += 1
        return -1 if res == inf else res

    # 3096. 得到更多分数的最少关卡数目 (Minimum Levels to Gain More Points)
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        s = sum(possible) * 2 - n
        pre = 0
        for i, v in enumerate(possible[:-1]):
            pre += v * 2 - 1
            if pre > s - pre:
                return i + 1
        return -1

    # 3099. 哈沙德数 (Harshad Number)
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        c = x
        s = 0
        while c:
            s += c % 10
            c //= 10
        return -1 if x % s else s

    # 3100. 换水问题 II (Water Bottles II)
    def maxBottlesDrunk(self, numBottles: int, numExchange: int) -> int:
        res = numBottles
        full = 0
        empty = numBottles
        while empty >= numExchange:
            while empty >= numExchange:
                full += 1
                empty -= numExchange
                numExchange += 1
            res += full
            empty += full
            full = 0
        return res

    # 3101. 交替子数组计数 (Count Alternating Subarrays)
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        res = 0
        cnt = 0
        pre = -1
        for x in nums:
            cnt = cnt + 1 if pre != x else 1
            res += cnt
            pre = x
        return res

    # 2518. 好分区的数目 (Number of Great Partitions)
    def countPartitions(self, nums: List[int], k: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j >= k:
                return 0
            if i == n:
                return 1
            return (dfs(i + 1, j) + dfs(i + 1, j + nums[i])) % MOD

        if sum(nums) < k * 2:
            return 0
        MOD = 10**9 + 7
        n = len(nums)
        return (pow(2, n, MOD) - dfs(0, 0) * 2) % MOD

    # 3102. 最小化曼哈顿距离 (Minimize Manhattan Distances) --曼哈顿距离、切比雪夫距离
    def minimumDistance(self, points: List[List[int]]) -> int:
        sx = SortedList()
        sy = SortedList()
        for x, y in points:
            sx.add(x + y)
            sy.add(y - x)
        res = inf
        for x, y in points:
            x, y = x + y, y - x
            sx.remove(x)
            sy.remove(y)
            res = min(res, max(sx[-1] - sx[0], sy[-1] - sy[0]))
            sx.add(x)
            sy.add(y)
        return res

    # 894. 所有可能的真二叉树 (All Possible Full Binary Trees)
    def allPossibleFBT(self, n: int) -> List[Optional[TreeNode]]:
        @cache
        def dfs(c: int) -> List[Optional[TreeNode]]:
            l = []
            if c == 1:
                l.append(TreeNode(0))
            elif c % 2 == 1:
                for i in range(1, c, 2):
                    l1 = dfs(i)
                    l2 = dfs(c - i - 1)
                    for left in l1:
                        for right in l2:
                            node = TreeNode(0)
                            node.left = left
                            node.right = right
                            l.append(node)
            return l

        return dfs(n)

    # 956. 最高的广告牌 (Tallest Billboard)
    def tallestBillboard(self, rods: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0 if j == 0 else -inf
            if abs(j) > s // 2:
                return -inf
            return max(
                dfs(i + 1, j + rods[i]) + rods[i],
                dfs(i + 1, j - rods[i]),
                dfs(i + 1, j),
            )

        s = sum(rods)
        n = len(rods)
        return dfs(0, 0)

    # LCP 47. 入场安检
    def securityCheck(self, capacities: List[int], k: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j < 0:
                return 0
            if i < 0:
                return j == 0
            if pre[i] - (i + 1) < j:
                return 0
            return (dfs(i - 1, j - (capacities[i] - 1)) + dfs(i - 1, j)) % MOD

        n = len(capacities)
        MOD = 10**9 + 7
        pre = list(accumulate(capacities))
        return dfs(n - 1, k)

    # 1379. 找出克隆二叉树中的相同节点 (Find a Corresponding Node of a Binary Tree in a Clone of That Tree)
    def getTargetCopy(
        self, original: TreeNode, cloned: TreeNode, target: TreeNode
    ) -> TreeNode:
        def dfs(x: TreeNode, y: TreeNode) -> TreeNode:
            if x is None:
                return None
            if x == target:
                return y
            res1 = dfs(x.left, y.left)
            if res1 is not None:
                return res1
            return dfs(x.right, y.right)

        return dfs(original, cloned)

    # 1026. 节点与其祖先之间的最大差值 (Maximum Difference Between Node and Ancestor)
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        def dfs(node: "TreeNode") -> tuple:
            if node is None:
                return ()
            left = dfs(node.left)
            right = dfs(node.right)
            ret = [node.val, node.val]
            nonlocal res
            if left:
                res = max(res, abs(node.val - left[0]), abs(node.val - left[1]))
                ret[0] = min(ret[0], left[0])
                ret[1] = max(ret[1], left[1])
            if right:
                res = max(res, abs(node.val - right[0]), abs(node.val - right[1]))
                ret[0] = min(ret[0], right[0])
                ret[1] = max(ret[1], right[1])
            return ret

        res = 0
        dfs(root)
        return res

    # 1483. 树节点的第 K 个祖先 (Kth Ancestor of a Tree Node)
    class TreeAncestor:

        def __init__(self, n: int, parent: List[int]):
            self.t = 0
            self.n = n
            self.g = [[] for _ in range(n)]
            for i, v in enumerate(parent[1:], 1):
                self.g[v].append(i)
            self.level = [[] for _ in range(n)]
            self.node_to_layer = defaultdict(tuple)
            self.dfs(0, 0, -1)

        def dfs(self, x: int, d: int, fa: int) -> None:
            self.level[d].append((x, self.t))
            self.node_to_layer[x] = (d, self.t)
            self.t += 1
            for y in self.g[x]:
                if y != fa:
                    self.dfs(y, d + 1, x)

        def getKthAncestor(self, node: int, k: int) -> int:
            (d, t) = self.node_to_layer[node]
            if d - k < 0:
                return -1
            return self.binary_search(self.level[d - k], t)

        def binary_search(self, arr: List[tuple], target: int) -> int:
            res = -1
            left = 0
            right = len(arr) - 1
            while left <= right:
                mid = left + ((right - left) >> 1)
                if arr[mid][1] < target:
                    res = arr[mid][0]
                    left = mid + 1
                else:
                    right = mid - 1
            return res

    # 86. 分隔链表 (Partition List)
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        cur0 = dummy0 = ListNode(0)
        cur1 = dummy1 = ListNode(0)
        while head:
            if head.val < x:
                cur0.next = head
                cur0 = cur0.next
            else:
                cur1.next = head
                cur1 = cur1.next
            head = head.next
        cur1.next = None
        cur0.next = dummy1.next
        return dummy0.next

    # 1600. 王位继承顺序 (Throne Inheritance)
    class ThroneInheritance:

        def __init__(self, kingName: str):
            self.dic = defaultdict(list)
            self.death_name = set()
            self.king = kingName

        def birth(self, parentName: str, childName: str) -> None:
            self.dic[parentName].append(childName)

        def death(self, name: str) -> None:
            self.death_name.add(name)

        def getInheritanceOrder(self) -> List[str]:
            self.res = []
            self.dfs(self.king)
            return self.res

        def dfs(self, name: str) -> None:
            if name not in self.death_name:
                self.res.append(name)
            for child in self.dic[name]:
                self.dfs(child)

    # 3105. 最长的严格递增或递减子数组 (Longest Strictly Increasing or Strictly Decreasing Subarray)
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        res = 1
        cnt = 1
        n = len(nums)
        for i in range(1, n):
            if nums[i] > nums[i - 1]:
                cnt += 1
            else:
                cnt = 1
            res = max(res, cnt)
        cnt = 1
        for i in range(1, n):
            if nums[i] < nums[i - 1]:
                cnt += 1
            else:
                cnt = 1
            res = max(res, cnt)
        return res

    # 3106. 满足距离约束且字典序最小的字符串 (Lexicographically Smallest String After Operations With Constraint)
    def getSmallestString(self, s: str, k: int) -> str:
        t = list(s)
        for i, c in enumerate(t):
            m = min(26 - (ord(c) - ord("a")), ord(c) - ord("a"))
            if m > k:
                t[i] = chr(ord(c) - k)
                break
            t[i] = "a"
            k -= m
        return "".join(t)

    # 3107. 使数组中位数等于 K 的最少操作数 (Minimum Operations to Make Median of Array Equal to K)
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        i = 0
        j = n - 1
        p = n // 2
        while i < j:
            if nums[i] <= k <= nums[j]:
                i += 1
                j -= 1
            else:
                break
        if i >= j:
            return abs(nums[p] - k)
        res = 0
        if nums[j] < k:
            while j >= p:
                res += abs(nums[j] - k)
                j -= 1
        else:
            while i <= p:
                res += abs(nums[i] - k)
                i += 1
        return res

    # 3108. 带权图里旅途的最小代价 (Minimum Cost Walk in Weighted Graph)
    def minimumCost(
        self, n: int, edges: List[List[int]], query: List[List[int]]
    ) -> List[int]:
        class Union:

            def __init__(self, n: int):
                self.parent = [0] * n
                for i in range(n):
                    self.parent[i] = i
                self.rank = [1] * n

            def get_root(self, p: int) -> int:
                if self.parent[p] == p:
                    return p
                self.parent[p] = self.get_root(self.parent[p])
                return self.parent[p]

            def is_connected(self, p1: int, p2: int) -> bool:
                return self.get_root(p1) == self.get_root(p2)

            def union(self, p1: int, p2: int) -> None:
                root1 = self.get_root(p1)
                root2 = self.get_root(p2)
                if root1 == root2:
                    return
                if self.rank[root1] < self.rank[root2]:
                    self.parent[root1] = root2
                else:
                    self.parent[root2] = root1
                    if self.rank[root1] == self.rank[root2]:
                        self.rank[root1] += 1

        union = Union(n)
        for u, v, _ in edges:
            union.union(u, v)
        dic = defaultdict(int)
        for u, v, w in edges:
            root = union.get_root(u)
            if root in dic:
                dic[root] &= w
            else:
                dic[root] = w
        res = [0] * len(query)
        for i, (u, v) in enumerate(query):
            if u == v:
                continue
            if union.is_connected(u, v):
                res[i] = dic[union.get_root(u)]
            else:
                res[i] = -1
        return res

    # 3108. 带权图里旅途的最小代价 (Minimum Cost Walk in Weighted Graph)
    def minimumCost(
        self, n: int, edges: List[List[int]], query: List[List[int]]
    ) -> List[int]:
        def dfs(x: int) -> int:
            ids[x] = len(_list)
            res = -1
            for y, w in g[x]:
                res &= w
                if ids[y] < 0:
                    res &= dfs(y)
            return res

        g = [[] for _ in range(n)]
        for u, v, w in edges:
            g[u].append((v, w))
            g[v].append((u, w))
        ids = [-1] * n
        _list = []
        for i in range(n):
            if ids[i] < 0:
                _list.append(dfs(i))
        return [
            0 if s == t else -1 if ids[s] != ids[t] else _list[ids[s]] for s, t in query
        ]

    # 2009. 使数组连续的最少操作数 (Minimum Number of Operations to Make Array Continuous)
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        nums = sorted(set(nums))
        res = 0
        j = 0
        for i, v in enumerate(nums):
            while nums[j] < v - n + 1:
                j += 1
            res = max(res, i - j + 1)
        return n - res

    # 2529. 正整数和负整数的最大计数 (Maximum Count of Positive Integer and Negative Integer)
    def maximumCount(self, nums: List[int]) -> int:
        def bin_lower() -> int:
            if nums[0] >= 0:
                return 0
            if nums[-1] < 0:
                return n
            res = 0
            left = 0
            right = n - 1
            while left <= right:
                mid = left + ((right - left) >> 1)
                if nums[mid] < 0:
                    res = mid + 1
                    left = mid + 1
                else:
                    right = mid - 1
            return res

        def bin_higher() -> int:
            if nums[0] > 0:
                return n
            if nums[-1] <= 0:
                return 0
            res = 0
            left = 0
            right = n - 1
            while left <= right:
                mid = left + ((right - left) >> 1)
                if nums[mid] > 0:
                    res = n - mid
                    right = mid - 1
                else:
                    left = mid + 1
            return res

        n = len(nums)
        return max(bin_lower(), bin_higher())

    # 1702. 修改后的最大二进制字符串 (Maximum Binary String After Change)
    def maximumBinaryString(self, binary: str) -> str:
        n = len(binary)
        i = binary.find("0")
        if i < 0:
            return binary
        zeros = binary.count("0")
        s = ["1"] * n
        s[i + zeros - 1] = "0"
        return "".join(s)

    # 44. 通配符匹配 (Wildcard Matching)
    def isMatch(self, s: str, p: str) -> bool:
        @cache
        def dfs(i: int, j: int) -> bool:
            if i == n1 and j == n2:
                return True
            if i == n1:
                return p[j:].count("*") == n2 - j
            if j == n2:
                return False
            if s[i:] == p[j:]:
                return True
            if p[j].islower():
                return s[i] == p[j] and dfs(i + 1, j + 1)
            if p[j] == "?":
                return dfs(i + 1, j + 1)
            return any(dfs(k, j + 1) for k in range(i, n1 + 1))

        n1 = len(s)
        n2 = len(p)
        return dfs(0, 0)

    # 705. 设计哈希集合 (Design HashSet)
    class MyHashSet:

        def __init__(self):
            self.vis = [False] * 1000001

        def add(self, key: int) -> None:
            self.vis[key] = True

        def remove(self, key: int) -> None:
            self.vis[key] = False

        def contains(self, key: int) -> bool:
            return self.vis[key]

    # 3110. 字符串的分数 (Score of a String)
    def scoreOfString(self, s: str) -> int:
        return sum(abs(x - y) for x, y in pairwise(map(ord, s)))

    # 3111. 覆盖所有点的最少矩形数目 (Minimum Rectangles to Cover Points)
    def minRectanglesToCoverPoints(self, points: List[List[int]], w: int) -> int:
        points.sort()
        res = 0
        r = -1
        for p in points:
            if p[0] > r:
                res += 1
                r = p[0] + w
        return res

    # 3112. 访问消失节点的最少时间 (Minimum Time to Visit Disappearing Nodes)
    def minimumTime(
        self, n: int, edges: List[List[int]], disappear: List[int]
    ) -> List[int]:
        g = [[] for _ in range(n)]
        for u, v, w in edges:
            g[u].append((v, w))
            g[v].append((u, w))
        res = [-1] * n
        res[0] = 0
        q = [(0, 0)]
        heapq.heapify(q)

        while q:
            d, x = heapq.heappop(q)
            if d > res[x]:
                continue
            for y, w in g[x]:
                if (res[y] == -1 or d + w < res[y]) and d + w < disappear[y]:
                    res[y] = d + w
                    heapq.heappush(q, (d + w, y))
        return res

    # 3113. 边界元素是最大值的子数组数目 (Find the Number of Subarrays Where Boundary Elements Are Maximum)
    def numberOfSubarrays(self, nums: List[int]) -> int:
        st = [[inf, 0]]
        res = len(nums)
        for x in nums:
            while st[-1][0] < x:
                st.pop()
            if st[-1][0] == x:
                res += st[-1][1]
                st[-1][1] += 1
            else:
                st.append([x, 1])
        return res

    # 3114. 替换字符可以得到的最晚时间 (Latest Time You Can Obtain After Replacing Characters)
    def findLatestTime(self, s: str) -> str:
        for i in range(11, -1, -1):
            for j in range(59, -1, -1):
                t = f"{i:02d}:{j:02d}"
                if all(x == "?" or x == y for x, y in zip(s, t)):
                    return t

    # 3115. 素数的最大距离 (Maximum Prime Difference)
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        def is_prime(x: int) -> bool:
            for i in range(2, isqrt(x) + 1):
                if x % i == 0:
                    return False
            return x != 1

        n = len(nums)
        i = 0
        while i < n:
            if is_prime(nums[i]):
                break
            i += 1
        j = n - 1
        while j >= 0:
            if is_prime(nums[j]):
                return j - i
            j -= 1

    # 3117. 划分数组得到最小的值之和 (Minimum Sum of Values by Dividing Array)
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i == n or j == m:
                return 0 if i == n and j == m else inf
            k &= nums[i]
            if n - i < m - j or k < andValues[j]:
                return inf
            res = dfs(i + 1, j, k)
            if k == andValues[j]:
                res = min(res, dfs(i + 1, j + 1, -1) + nums[i])
            return res

        n = len(nums)
        m = len(andValues)
        res = dfs(0, 0, -1)
        return -1 if res == inf else res

    # 924. 尽量减少恶意软件的传播 (Minimize Malware Spread)
    def minMalwareSpread(self, graph: List[List[int]], initial: List[int]) -> int:
        n = len(graph)
        union = Union924(n)
        for i in range(n):
            for j in range(n):
                if graph[i][j]:
                    union.union721(i, j)
        res = min(initial)
        s = 0
        _list = [[] for _ in range(n)]
        for i in initial:
            root = union.get_root(i)
            _list[root].append(i)
        for i in initial:
            root = union.get_root(i)
            if len(_list[root]) == 1:
                if union.get_size(root) > s:
                    s = union.get_size(root)
                    res = i
                elif union.get_size(root) == s and res > i:
                    res = i
        return res

    # 2172. 数组的最大与和 (Maximum AND Sum of Array)
    def maximumANDSum(self, nums: List[int], numSlots: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j == u:
                return 0
            if numSlots - i + 1 < (u ^ j).bit_count():
                return -inf
            res = dfs(i + 1, j)
            c = u ^ j
            while c:
                lb = (c & -c).bit_length() - 1
                res = max(res, dfs(i + 1, j | (1 << lb)) + (((i + 1) >> 1) & nums[lb]))
                c &= c - 1
            return res

        n = len(nums)
        numSlots <<= 1
        u = (1 << n) - 1
        return dfs(1, 0)

    # 928. 尽量减少恶意软件的传播 II (Minimize Malware Spread II)
    def minMalwareSpread(self, graph: List[List[int]], initial: List[int]) -> int:
        def check(x: int) -> int:
            union = Union924(n)
            for i in range(n):
                for j in range(n):
                    if i != x and j != x and graph[i][j]:
                        union.union721(i, j)
            s = set()
            for i in initial:
                if i == x:
                    continue
                s.add(union.get_root(i))
            res = 0
            for i in s:
                res += union.get_size(i)
            return res

        n = len(graph)
        initial.sort()
        s = inf
        res = inf
        for i in initial:
            cur = check(i)
            if cur < s:
                s = cur
                res = i
        return res

    # 996. 正方形数组的数目 (Number of Squareful Arrays)
    def numSquarefulPerms(self, nums: List[int]) -> int:
        def dfs() -> int:
            if len(_list) == n:
                return 1
            nonlocal used
            res = 0
            for j, v in enumerate(nums):
                if (
                    (used >> j & 1)
                    or j
                    and v == nums[j - 1]
                    and (used >> (j - 1) & 1 == 0)
                ):
                    continue
                s = 0
                if len(_list):
                    s = isqrt(_list[-1] + v)
                    if s * s != _list[-1] + v:
                        continue
                used ^= 1 << j
                _list.append(v)
                res += dfs()
                used ^= 1 << j
                _list.pop()
            return res

        nums.sort()
        n = len(nums)
        _list = []
        used = 0
        return dfs()

    # 2007. 从双倍数组中还原原数组 (Find Original Array From Doubled Array)
    def findOriginalArray(self, changed: List[int]) -> List[int]:
        n = len(changed)
        if n & 1:
            return []
        changed.sort()
        c = Counter(changed)
        if c[0] & 1:
            return []
        res = []
        res.extend([0] * (c[0] // 2))
        for ch in changed:
            if ch == 0 or c[ch] == 0:
                continue
            if ch * 2 not in c:
                return []
            c[ch] -= 1
            if c[ch] == 0:
                del c[ch]
            c[ch * 2] -= 1
            if c[ch * 2] == 0:
                del c[ch * 2]
            res.append(ch)
        return res

    # 847. 访问所有节点的最短路径 (Shortest Path Visiting All Nodes)
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        n = len(graph)
        q = deque()
        vis = [[False] * (1 << n) for _ in range(n)]
        u = (1 << n) - 1
        for i in range(n):
            q.append([i, 1 << i, 0])
            vis[i][1 << i] = True
        while q:
            [x, m, d] = q.popleft()
            if m == u:
                return d
            for y in graph[x]:
                nm = m | (1 << y)
                if not vis[y][nm]:
                    vis[y][nm] = True
                    q.append([y, nm, d + 1])

    # 216. 组合总和 III (Combination Sum III)
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        def dfs(i: int, j: int) -> None:
            if i == 10 or j >= n or len(arr) == k:
                if j == n and len(arr) == k:
                    res.append(arr.copy())
                return
            dfs(i + 1, j)
            arr.append(i)
            dfs(i + 1, j + i)
            arr.pop()

        arr = []
        res = []
        dfs(1, 0)
        return res

    # 3123. 最短路径中的边 (Find Edges in Shortest Paths)
    def findAnswer(self, n: int, edges: List[List[int]]) -> List[bool]:
        g = [[] for _ in range(n)]
        for i, (u, v, w) in enumerate(edges):
            g[u].append([v, w, i])
            g[v].append([u, w, i])
        q = [[0, 0]]
        dis = [inf] * n
        dis[0] = 0
        heapq.heapify(q)
        while q:
            [d, x] = heapq.heappop(q)
            if d > dis[x]:
                continue
            for nxt in g[x]:
                [y, w, _] = nxt
                if d + w < dis[y]:
                    dis[y] = d + w
                    heapq.heappush(q, [dis[y], y])
        res = [False] * len(edges)
        if dis[n - 1] == inf:
            return res

        def dfs(x: int) -> None:
            for nxt in g[x]:
                [y, w, i] = nxt
                if dis[y] + w != dis[x]:
                    continue
                res[i] = True
                dfs(y)

        dfs(n - 1)
        return res

    # 3122. 使矩阵满足条件的最少操作次数 (Minimum Number of Operations to Satisfy Conditions)
    def minimumOperations(self, grid: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            return min(dfs(i + 1, x) - cnts[i][x] for x in range(10) if x != j) + m

        m = len(grid)
        n = len(grid[0])
        cnts = [[0] * 10 for _ in range(n)]
        for i in range(m):
            for j in range(n):
                cnts[j][grid[i][j]] += 1
        return dfs(0, -1)

    # 3121. 统计特殊字母的数量 II (Count the Number of Special Characters II)
    def numberOfSpecialChars(self, word: str) -> int:
        status = [0] * 26
        for c in map(ord, word):
            x = (c & 31) - 1
            if status[x] == -1:
                continue
            # 小写
            if c >> 5 & 1 == 1:
                if status[x] == 2:
                    status[x] = -1
                else:
                    status[x] = 1
            else:
                if status[x] == 0:
                    status[x] = -1
                else:
                    status[x] = 2
        return sum(int(s == 2) for s in status)

    # 3120. 统计特殊字母的数量 I (Count the Number of Special Characters I)
    def numberOfSpecialChars(self, word: str) -> int:
        cnt = [0] * 2
        for c in map(ord, word):
            cnt[c >> 5 & 1] |= 1 << (c & 31)
        return (cnt[0] & cnt[1]).bit_count()

    # 2385. 感染二叉树需要的总时间 (Amount of Time for Binary Tree to Be Infected)
    def amountOfTime(self, root: Optional[TreeNode], start: int) -> int:
        def dfs(root: Optional[TreeNode]) -> None:
            if root is None:
                return
            for son in [root.left, root.right]:
                if son:
                    g[root.val].append(son.val)
                    g[son.val].append(root.val)
                    dfs(son)

        g = defaultdict(list)
        dfs(root)
        res = 0
        vis = set([start])
        q = deque([start])
        while q:
            s = len(q)
            for _ in range(s):
                x = q.popleft()
                for y in g[x]:
                    if y not in vis:
                        vis.add(y)
                        q.append(y)
            if q:
                res += 1
        return res

    # 2739. 总行驶距离 (Total Distance Traveled)
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        res = 0
        while mainTank >= 5 and additionalTank:
            res += 50
            mainTank -= 4
            additionalTank -= 1
        res += mainTank * 10
        return res

    # 1146. 快照数组 (Snapshot Array)
    class SnapshotArray:

        def __init__(self, length: int):
            self.g = defaultdict(list)
            self.id = -1

        def set(self, index: int, val: int) -> None:
            if len(self.g[index]) == 0 or self.g[index][-1][1] != self.id:
                self.g[index].append([val, self.id])
            else:
                self.g[index][-1] = [val, self.id]

        def snap(self) -> int:
            self.id += 1
            return self.id

        def get(self, index: int, snap_id: int) -> int:
            return self.bis(self.g[index], snap_id)

        def bis(self, arr: list, snap_id: int) -> int:
            n = len(arr)
            left = 0
            right = n - 1
            res = 0
            while left <= right:
                mid = left + ((right - left) >> 1)
                if arr[mid][1] < snap_id:
                    res = arr[mid][0]
                    left = mid + 1
                else:
                    right = mid - 1
            return res

    # 2639. 查询网格图中每一列的宽度 (Find the Width of Columns of a Grid)
    def findColumnWidth(self, grid: List[List[int]]) -> List[int]:
        n = len(grid[0])
        m = len(grid)
        res = [0] * n
        for j in range(n):
            for i in range(m):
                res[j] = max(res[j], len(str(grid[i][j])))
        return res

    # 1017. 负二进制转换 (Convert to Base -2)
    def baseNeg2(self, n: int) -> str:
        return (
            str(n) if n in (0, 1) else self.baseNeg2((n - (n & 1)) // -2) + str(n & 1)
        )

    # 3127. 构造相同颜色的正方形 (Make a Square with the Same Color)
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        for i in range(1, 3):
            for j in range(1, 3):
                d = 0
                for x in range(i - 1, i + 1):
                    for y in range(j - 1, j + 1):
                        d += 1 if grid[x][y] == "W" else -1
                if abs(d) >= 2:
                    return True
        return False

    # 3128. 直角三角形 (Right Triangles)
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        res = 0
        row = [sum(r) for r in grid]
        col = [sum(c) for c in zip(*grid)]
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    res += (row[i] - 1) * (col[j] - 1)
        return res

    # 3129. 找出所有稳定的二进制数组 I (Find All Possible Stable Binary Arrays I)
    # 3130. 找出所有稳定的二进制数组 II (Find All Possible Stable Binary Arrays II)
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i == 0:
                return 1 if k == 1 and j <= limit else 0
            if j == 0:
                return 1 if k == 0 and i <= limit else 0
            if k == 0:
                return (
                    dfs(i - 1, j, 1)
                    + dfs(i - 1, j, 0)
                    - (dfs(i - limit - 1, j, 1) if i > limit else 0)
                ) % MOD
            return (
                dfs(i, j - 1, 0)
                + dfs(i, j - 1, 1)
                - (dfs(i, j - limit - 1, 0) if j > limit else 0)
            ) % MOD

        MOD = 10**9 + 7
        res = (dfs(zero, one, 0) + dfs(zero, one, 1)) % MOD
        dfs.cache_clear()
        return res

    # 3131. 找出与数组相加的整数 I (Find the Integer Added to Array I)
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        return min(nums2) - min(nums1)

    # 3132. 找出与数组相加的整数 II (Find the Integer Added to Array II)
    def minimumAddedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        def check(d: int) -> int:
            c = 0
            i = 0
            j = 0
            while i < len(nums1) and j < len(nums2):
                if nums2[j] - nums1[i] != d:
                    c += 1
                    if c > 2:
                        return inf
                else:
                    j += 1
                i += 1
            return d

        nums1.sort()
        nums2.sort()
        return min(check(nums2[0] - nums1[i]) for i in range(3))

    # 3133. 数组最后一个元素的最小值 (Minimum Array End)
    def minEnd(self, n: int, x: int) -> int:
        j = 0
        t = ~x
        n -= 1
        while n >> j:
            lb = t & -t
            bit = n >> j & 1
            x |= bit * lb
            j += 1
            t ^= lb
        return x

    # 3134. 找出唯一性数组的中位数 (Find the Median of the Uniqueness Array)
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        def check(upper: int) -> bool:
            cnt = 0
            c = Counter()
            l = 0
            for r, x in enumerate(nums):
                c[x] += 1
                while len(c) > upper:
                    c[nums[l]] -= 1
                    if c[nums[l]] == 0:
                        del c[nums[l]]
                    l += 1
                cnt += r - l + 1
                if cnt >= k:
                    return True
            return False

        n = len(nums)
        k = ((1 + n) * n // 2 + 1) // 2
        left = 1
        right = len(set(nums))
        res = 1
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = mid
                right = mid - 1
            else:
                left = mid + 1
        return res

    # 1329. 将矩阵按对角线排序 (Sort the Matrix Diagonally)
    def diagonalSort(self, mat: List[List[int]]) -> List[List[int]]:
        m = len(mat)
        n = len(mat[0])
        d = defaultdict(list)
        for i in range(m):
            for j in range(n):
                d[i - j].append(mat[i][j])
        for c in d.values():
            c.sort(reverse=True)
        for i in range(m):
            for j in range(n):
                mat[i][j] = d[i - j].pop()
        return mat

    # 1652. 拆炸弹 (Defuse the Bomb)
    def decrypt(self, code: List[int], k: int) -> List[int]:
        n = len(code)
        res = [0] * n
        r = k + 1 if k > 0 else n
        k = abs(k)
        s = sum(code[r - k : r])
        for i in range(n):
            res[i] = s
            s += code[r % n] - code[(r - k) % n]
            r += 1
        return res

    # 3136. 有效单词 (Valid Word)
    def isValid(self, word: str) -> bool:
        return (
            len(word) > 2
            and all(c.isalpha() or c.isdigit() for c in word)
            and any(c in "aeiouAEIOU" for c in word)
            and any(c.isalpha() and c not in "aeiouAEIOU" for c in word)
        )

    # 3137. K 周期字符串需要的最少操作次数 (Minimum Number of Operations to Make Word K-Periodic)
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        d = defaultdict(int)
        for i in range(0, n, k):
            d[word[i : i + k]] += 1
        return n // k - max(d.values())

    # 3138. 同位字符串连接的最小长度 (Minimum Length of Anagram Concatenation)
    def minAnagramLength(self, s: str) -> int:
        def check(c: int) -> bool:
            for i in range(c, len(s), c):
                cur = [0] * 26
                for j in range(i, i + c):
                    cur[ord(s[j]) - ord("a")] += 1
                if pre != cur:
                    return False
            return True

        pre = [0] * 26
        for i, v in enumerate(s):
            pre[ord(v) - ord("a")] += 1
            if len(s) % (i + 1) == 0 and check(i + 1):
                return i + 1

    # 2079. 给植物浇水 (Watering Plants)
    def wateringPlants(self, plants: List[int], capacity: int) -> int:
        res = 0
        c = capacity
        for i, v in enumerate(plants):
            res += 1
            c -= v
            if i < len(plants) - 1 and c < plants[i + 1]:
                res += (i + 1) << 1
                c = capacity
        return res

    # 2745. 构造最长的新字符串 (Construct the Longest New String)
    def longestString(self, x: int, y: int, z: int) -> int:
        @cache
        def dfs(i: int, j: int, k: int, l: int) -> int:
            if l == 0:
                return dfs(i, j - 1, k, 1) + 2 if j else 0
            if l == 1:
                return max(
                    dfs(i - 1, j, k, 0) + 2 if i else 0,
                    dfs(i, j, k - 1, 2) + 2 if k else 0,
                )
            return max(
                dfs(i - 1, j, k, 0) + 2 if i else 0, dfs(i, j, k - 1, 2) + 2 if k else 0
            )

        return max(dfs(x, y, z, 0), dfs(x, y, z, 1), dfs(x, y, z, 2))

    # 376. 摆动序列 (Wiggle Subsequence)
    def wiggleMaxLength(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            res = 0
            if j:
                for k in range(i + 1, n):
                    if nums[k] > nums[i]:
                        res = max(res, dfs(k, 0))
                return res + 1
            for k in range(i + 1, n):
                if nums[k] < nums[i]:
                    res = max(res, dfs(k, 1))
            return res + 1

        n = len(nums)
        return max(max(dfs(i, 0), dfs(i, 1)) for i in range(n))

    # 1186. 删除一次得到子数组最大和 (Maximum Subarray Sum with One Deletion)
    def maximumSum(self, arr: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i < 0:
                return 0
            # 不删
            res = max(dfs(i - 1, j) + arr[i], 0)
            # 删
            if j == 0:
                res = max(res, dfs(i - 1, j + 1))
            return res

        n = len(arr)
        return max(dfs(i - 1, 0) + arr[i] for i in range(n))

    # 2105. 给植物浇水 II (Watering Plants II)
    def minimumRefill(self, plants: List[int], capacityA: int, capacityB: int) -> int:
        n = len(plants)
        i = 0
        j = n - 1
        res = 0
        a = capacityA
        b = capacityB
        while i < j:
            if a < plants[i]:
                res += 1
                a = capacityA
            a -= plants[i]
            if b < plants[j]:
                res += 1
                b = capacityB
            b -= plants[j]
            i += 1
            j -= 1
        return res + (i == j and max(a, b) < plants[i])

    # 1262. 可被三整除的最大和 (Greatest Sum Divisible by Three)
    def maxSumDivThree(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0 if j == 0 else -inf
            return max(dfs(i + 1, j), dfs(i + 1, (j + nums[i]) % 3) + nums[i])

        n = len(nums)
        return dfs(0, 0)

    # 1567. 乘积为正数的最长子数组长度 (Maximum Length of Subarray With Positive Product)
    def getMaxLen(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i < 0 or nums[i] == 0:
                return 0 if j == 0 else -inf
            return max(0 if j == 0 else -inf, dfs(i - 1, j ^ (nums[i] < 0)) + 1)

        n = len(nums)
        return max(
            0, max(dfs(i - 1, int(nums[i] < 0)) + 1 if nums[i] else 0 for i in range(n))
        )

    # 2391. 收集垃圾的最少总时间 (Minimum Amount of Time to Collect Garbage)
    def garbageCollection(self, garbage: List[str], travel: List[int]) -> int:
        def cal(c: chr) -> int:
            res = 0
            s = 0
            for i in range(len(garbage) - 1, -1, -1):
                s += garbage[i].count(c)
                if s and i:
                    res += travel[i - 1]
            return res + s

        return sum(cal(c) for c in ("M", "G", "P"))

    # 639. 解码方法 II (Decode Ways II)
    def numDecodings(self, s: str) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == n:
                return 1
            if s[i] == "0":
                return 0
            res = dfs(i + 1) * (9 if s[i] == "*" else 1)
            if i + 1 < n:
                if s[i] == "1":
                    res += dfs(i + 2) * (9 if s[i + 1] == "*" else 1)
                elif s[i] == "2":
                    if "0" <= s[i + 1] <= "6":
                        res += dfs(i + 2)
                    elif s[i + 1] == "*":
                        res += dfs(i + 2) * 6
                elif s[i] == "*":
                    if s[i + 1] == "*":
                        res += dfs(i + 2) * 15
                    else:
                        res += dfs(i + 2) * (2 if "0" <= s[i + 1] <= "6" else 1)
            return res % MOD

        n = len(s)
        MOD = 10**9 + 7
        return dfs(0)

    # 3006. 找出数组中的美丽下标 I (Find Beautiful Indices in the Given Array I)
    # 3008. 找出数组中的美丽下标 II (Find Beautiful Indices in the Given Array II) --z函数
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        def check(t: str) -> list:
            ss = t + s
            res = []
            n = len(ss)
            z = [0] * n
            left = 0
            right = 0
            for i in range(1, n):
                if i <= right:
                    z[i] = min(z[i - left], right - i + 1)
                while i + z[i] < n and ss[z[i]] == ss[i + z[i]]:
                    left, right = i, i + z[i]
                    z[i] += 1
                if i >= len(t) and z[i] >= len(t):
                    res.append(i - len(t))
            return res

        arr_a = check(a)
        arr_b = check(b)
        res = []
        j = 0
        for x in arr_a:
            while j < len(arr_b) and x - arr_b[j] > k:
                j += 1
            if j < len(arr_b) and abs(x - arr_b[j]) <= k:
                res.append(x)
        return res

    # 1553. 吃掉 N 个橘子的最少天数 (Minimum Number of Days to Eat N Oranges)
    def minDays(self, n: int) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == 1:
                return 1
            res = inf
            if i % 2 == 0:
                res = min(res, dfs(i // 2) + 1)
            if i % 3 == 0:
                res = min(res, dfs(i // 3) + 1)
            if i % 6:
                res = min(res, dfs(i - 1) + 1)
            return res

        return dfs(n)

    # 3142. 判断矩阵是否满足条件 (Check if Grid Satisfies Conditions)
    def satisfiesConditions(self, grid: List[List[int]]) -> bool:
        return all(x != y for x, y in pairwise(grid[0])) and all(
            x == y for x, y in pairwise(grid)
        )

    # 3143. 正方形中的最多点数 (Maximum Points Inside the Square)
    def maxPointsInsideSquare(self, points: List[List[int]], s: str) -> int:
        d = defaultdict(list)
        for (x, y), c in zip(points, s):
            d[max(abs(x), abs(y))].append(ord(c) - ord("a"))
        res = 0
        m = 0
        for k in sorted(d.keys()):
            cur_m = 0
            for v in d[k]:
                if (cur_m >> v) & 1 or (m >> v) & 1:
                    return res
                cur_m |= 1 << v
            res += len(d[k])
            m |= cur_m
        return res

    # 3144. 分割字符频率相等的最少子字符串 (Minimum Substring Partition of Equal Character Frequency)
    def minimumSubstringsInPartition(self, s: str) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == n:
                return 0
            res = inf
            d = defaultdict(int)
            cnt = 0
            for j in range(i, n):
                d[s[j]] += 1
                if d[s[j]] == 1:
                    cnt += 1
                if (j - i + 1) % len(d):
                    continue
                if all(v == (j - i + 1) // len(d) for _, v in d.items()):
                    res = min(res, dfs(j + 1) + 1)
            return res

        n = len(s)
        return dfs(0)

    # 3146. 两个字符串的排列差 (Permutation Difference between Two Strings)
    def findPermutationDifference(self, s: str, t: str) -> int:
        pos = [0] * 26
        for i, v in enumerate(map(ord, s)):
            pos[v - ord("a")] = i
        return sum(abs(i - pos[v - ord("a")]) for i, v in enumerate(map(ord, t)))

    # 3147. 从魔法师身上吸取的最大能量 (Taking Maximum Energy From the Mystic Dungeon)
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        res = -inf
        for i in range(n - k, n):
            res = max(res, max(accumulate(energy[j] for j in range(i, -1, -k))))
        return res

    # 3148. 矩阵中的最大得分 (Maximum Difference Score in a Grid)
    def maxScore(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        res = -inf
        pre = [[inf] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                _min = min(pre[i - 1][j], pre[i][j - 1])
                res = max(res, grid[i - 1][j - 1] - _min)
                pre[i][j] = min(_min, grid[i - 1][j - 1])
        return res

    # 3149. 找出分数最低的排列 (Find the Minimum Cost Array Permutation)
    def findPermutation(self, nums: List[int]) -> List[int]:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == u:
                return abs(j - nums[0])
            res = inf
            c = i ^ u
            while c:
                lb = (c & -c).bit_length() - 1
                res = min(res, dfs(i | (1 << lb), lb) + abs(j - nums[lb]))
                c &= c - 1
            return res

        def mask_ans(i: int, j: int) -> int:
            res.append(j)
            if i == u:
                return
            final_ans = dfs(i, j)
            c = i ^ u
            while c:
                lb = (c & -c).bit_length() - 1
                if dfs(i | (1 << lb), lb) + abs(j - nums[lb]) == final_ans:
                    mask_ans(i | (1 << lb), lb)
                    break
                c &= c - 1

        n = len(nums)
        u = (1 << n) - 1
        dfs(1, 0)
        res = []
        mask_ans(1, 0)
        return res

    # 2244. 完成所有任务需要的最少轮数 (Minimum Rounds to Complete All Tasks)
    def minimumRounds(self, tasks: List[int]) -> int:
        c = Counter(tasks)
        return -1 if 1 in c.values() else sum((v + 2) // 3 for v in c.values())

    # 1092. 最短公共超序列 (Shortest Common Supersequence)
    def shortestCommonSupersequence(self, str1: str, str2: str) -> str:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n1:
                return n2 - j
            if j == n2:
                return n1 - i
            if str1[i] == str2[j]:
                return dfs(i + 1, j + 1) + 1
            return min(dfs(i + 1, j), dfs(i, j + 1)) + 1

        def make_ans(i: int, j: int) -> None:
            if i == n1:
                res.extend(str2[j:])
                return
            if j == n2:
                res.extend(str1[i:])
                return
            if str1[i] == str2[j]:
                res.append(str1[i])
                make_ans(i + 1, j + 1)
                return
            final_ans = dfs(i, j)
            if dfs(i + 1, j) + 1 == final_ans:
                res.append(str1[i])
                make_ans(i + 1, j)
                return
            res.append(str2[j])
            make_ans(i, j + 1)

        n1 = len(str1)
        n2 = len(str2)
        dfs(0, 0)
        res = []
        make_ans(0, 0)
        return "".join(res)

    # 943. 最短超级串 (Find the Shortest Superstring)
    def shortestSuperstring(self, words: List[str]) -> str:
        def cal(i: int, j: int) -> int:
            nn = min(len(words[i]), len(words[j]))
            for k in range(len(words[i]) - nn, len(words[i])):
                if words[j].startswith(words[i][k:]):
                    return len(words[i]) - k
            return 0

        @cache
        def dfs(i: int, j: int) -> int:
            if i == u:
                return 0
            c = i ^ u
            res = inf
            while c:
                lb = (c & -c).bit_length() - 1
                res = min(res, dfs(i | (1 << lb), lb) + len(words[lb]) - lcp[j][lb])
                c &= c - 1
            return res

        def make_ans(i: int, j: int) -> None:
            if i == u:
                return
            c = i ^ u
            final_ans = dfs(i, j)
            while c:
                lb = (c & -c).bit_length() - 1
                if dfs(i | (1 << lb), lb) + len(words[lb]) - lcp[j][lb] == final_ans:
                    res.append(words[lb][lcp[j][lb] :])
                    make_ans(i | (1 << lb), lb)
                    break
                c &= c - 1

        n = len(words)
        lcp = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                lcp[i][j] = cal(i, j)
        u = (1 << n) - 1
        m = inf
        f = -1
        for i in range(n):
            cur = dfs(1 << i, i) + len(words[i])
            if cur < m:
                m = cur
                f = i
        res = [words[f]]
        make_ans(1 << f, f)
        return "".join(res)

    # 2589. 完成所有任务的最少时间 (Minimum Time to Complete All Tasks)
    def findMinimumTime(self, tasks: List[List[int]]) -> int:
        res = 0
        idle = [0] * 2001
        tasks.sort(key=lambda k: k[1])
        for start, end, duration in tasks:
            duration -= sum(idle[start : end + 1])
            i = end
            while i >= start and duration > 0:
                if idle[i] == 0:
                    idle[i] = 1
                    duration -= 1
                    res += 1
                i -= 1
        return res

    # 1953. 你可以工作的最大周数 (Maximum Number of Weeks for Which You Can Work)
    def numberOfWeeks(self, milestones: List[int]) -> int:
        s = sum(milestones)
        mx = max(milestones)
        if mx <= s // 2:
            return s
        return (s - mx) * 2 + 1

    # 826. 安排工作以达到最大收益 (Most Profit Assigning Work)
    def maxProfitAssignment(
        self, difficulty: List[int], profit: List[int], worker: List[int]
    ) -> int:
        n = len(difficulty)
        arr = sorted(zip(difficulty, profit), key=lambda k: k[0])
        p = SortedList(profit)
        j = n - 1
        res = 0
        for x in sorted(worker, key=lambda k: -k):
            while j >= 0 and arr[j][0] > x:
                p.remove(arr[j][1])
                j -= 1
            if len(p):
                res += p[-1]
        return res

    # 2644. 找出可整除性得分最大的整数 (Find the Maximum Divisibility Score)
    def maxDivScore(self, nums: List[int], divisors: List[int]) -> int:
        score = -1
        res = 0
        for d in divisors:
            s = sum(x % d == 0 for x in nums)
            if s > score:
                score = s
                res = d
            elif s == score and res > d:
                res = d
        return res

    # 1535. 找出数组游戏的赢家 (Find the Winner of an Array Game)
    def getWinner(self, arr: List[int], k: int) -> int:
        res = arr[0]
        times = 0
        for i in range(1, len(arr)):
            if arr[i] > res:
                res = arr[i]
                times = 1
            else:
                times += 1
            if times == k:
                return res
        return max(arr)

    # 3154. 到达第 K 级台阶的方案数 (Find Number of Ways to Reach the K-th Stair)
    def waysToReachStair(self, k: int) -> int:
        @cache
        def dfs(i: int, j: int, last: int) -> int:
            if i > k + 1:
                return 0
            res = dfs(i + (1 << j), j + 1, 1)
            if i and last == 1:
                res += dfs(i - 1, j, 0)
            return res + (i == k)

        return dfs(1, 0, 1)

    # 3153. 所有数对中数位不同之和 (Sum of Digit Differences of All Pairs)
    def sumDigitDifferences(self, nums: List[int]) -> int:
        res = 0
        l = len(str(nums[0]))
        for _ in range(l):
            s = 0
            cnts = [0] * 10
            for j in range(len(nums)):
                b = nums[j] % 10
                nums[j] //= 10
                res += s - cnts[b]
                cnts[b] += 1
                s += 1
        return res

    # 3152. 特殊数组 II (Special Array II)
    def isArraySpecial(self, nums: List[int], queries: List[List[int]]) -> List[bool]:
        s = list(accumulate(((x ^ y ^ 1) & 1 for x, y in pairwise(nums)), initial=0))
        return [s[from_] == s[to] for from_, to in queries]

    # 3151. 特殊数组 I (Special Array I)
    def isArraySpecial(self, nums: List[int]) -> bool:
        return all((x ^ y) & 1 for x, y in pairwise(nums))

    # 1542. 找出最长的超赞子字符串 (Find Longest Awesome Substring)
    def longestAwesome(self, s: str) -> int:
        n = len(s)
        pos = [n] * (1 << 10)
        pos[0] = -1
        m = 0
        res = 0
        for i, v in enumerate(s):
            m ^= 1 << int(v)
            res = max(res, i - pos[m], max(i - pos[m ^ (1 << j)] for j in range(10)))
            pos[m] = min(pos[m], i)
        return res

    # 2769. 找出最大的可达成数字 (Find the Maximum Achievable Number)
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        return num + t * 2

    # 2225. 找出输掉零场或一场比赛的玩家 (Find Players With Zero or One Losses)
    def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
        c = Counter()
        s0 = set()
        s1 = set()
        for _, l in matches:
            c[l] += 1
        for w, l in matches:
            if w not in c:
                s0.add(w)
            if c[l] == 1:
                s1.add(l)
        return [sorted(list(s0)), sorted(list(s1))]

    # 1673. 找出最具竞争力的子序列 (Find the Most Competitive Subsequence)
    def mostCompetitive(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        st = []
        for i, v in enumerate(nums):
            while st and v < st[-1] and len(st) + n - i - 1 >= k:
                st.pop()
            st.append(v)
        return st[:k]

    # 1738. 找出第 K 大的异或坐标值 (Find Kth Largest XOR Coordinate Value)
    def kthLargestValue(self, matrix: List[List[int]], k: int) -> int:
        m = len(matrix)
        n = len(matrix[0])
        arr = []
        for i in range(m):
            for j in range(n):
                if i == 0 and j:
                    matrix[i][j] ^= matrix[i][j - 1]
                elif i and j == 0:
                    matrix[i][j] ^= matrix[i - 1][j]
                elif i and j:
                    matrix[i][j] ^= (
                        matrix[i][j - 1] ^ matrix[i - 1][j] ^ matrix[i - 1][j - 1]
                    )
                arr.append(matrix[i][j])
        arr.sort(reverse=True)
        return arr[k - 1]

    # 3158. 求出出现两次数字的 XOR 值 (Find the XOR of Numbers Which Appear Twice)
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        mask = 0
        res = 0
        for x in nums:
            if mask >> x & 1:
                res ^= x
            else:
                mask |= 1 << x
        return res

    # 3159. 查询数组中元素的出现位置 (Find Occurrences of an Element in an Array)
    def occurrencesOfElement(
        self, nums: List[int], queries: List[int], x: int
    ) -> List[int]:
        p = [i for i, v in enumerate(nums) if v == x]
        return [p[q - 1] if q <= len(p) else -1 for q in queries]

    # 3160. 所有球里面不同颜色的数目 (Find the Number of Distinct Colors Among the Balls)
    def queryResults(self, _: int, queries: List[List[int]]) -> List[int]:
        id_to_color = Counter()
        color_to_cnt = Counter()
        res = []
        for id, color in queries:
            if id in id_to_color:
                pre_color = id_to_color[id]
                color_to_cnt[pre_color] -= 1
                if color_to_cnt[pre_color] == 0:
                    del color_to_cnt[pre_color]
            color_to_cnt[color] += 1
            id_to_color[id] = color
            res.append(len(color_to_cnt))
        return res

    # 2028. 找出缺失的观测数据 (Find Missing Observations)
    def missingRolls(self, rolls: List[int], mean: int, n: int) -> List[int]:
        m = len(rolls)
        s = mean * (m + n) - sum(rolls)
        if s > 6 * n or s < n:
            return []
        return [s // n + 1] * (s % n) + [s // n] * (n - s % n)

    # 3162. 优质数对的总数 I (Find the Number of Good Pairs I)
    # 3164. 优质数对的总数 II (Find the Number of Good Pairs II)
    def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:
        c = Counter()
        for x in nums1:
            if x % k:
                continue
            x //= k
            for i in range(1, isqrt(x) + 1):
                if x % i == 0:
                    c[i] += 1
                    if i * i < x:
                        c[x // i] += 1
        return sum(c[x] for x in nums2)

    # 3163. 压缩字符串 III (String Compression III)
    def compressedString(self, word: str) -> str:
        res = []
        i = 0
        while i < len(word):
            j = i
            while j < len(word) and word[i] == word[j] and j - i + 1 <= 9:
                j += 1
            res.append(str(j - i))
            res.append(word[i])
            i = j
        return "".join(res)

    # 2167. 移除所有载有违禁货物车厢所需的最少时间 (Minimum Time to Remove All Cars Containing Illegal Goods)
    def minimumTime(self, s: str) -> int:
        @cache
        def dfs_pre(i: int) -> int:
            if i < 0:
                return 0
            if s[i] == "0":
                return dfs_pre(i - 1)
            return min(i + 1, dfs_pre(i - 1) + 2)

        @cache
        def dfs_suf(i: int) -> int:
            if i == n:
                return 0
            if s[i] == "0":
                return dfs_suf(i + 1)
            return min(n - i, dfs_suf(i + 1) + 2)

        n = len(s)
        res = n
        for i in range(n):
            res = min(res, dfs_pre(i) + dfs_suf(i + 1))
        return res

    # 2501. 数组中最长的方波 (Longest Square Streak in an Array)
    def longestSquareStreak(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int) -> int:
            if i not in s:
                return 0
            return 1 + dfs(i * i)

        s = set(nums)
        res = max(dfs(i) for i in s)
        return -1 if res < 2 else res

    # 1218. 最长定差子序列 (Longest Arithmetic Subsequence of Given Difference)
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        d = defaultdict(int)
        for x in arr:
            d[x] = d[x - difference] + 1
        return max(d.values())

    # 873. 最长的斐波那契子序列的长度 (Length of Longest Fibonacci Subsequence)
    def lenLongestFibSubseq(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if (nums[i] + nums[j]) in dic:
                return dfs(j, dic[nums[i] + nums[j]]) + 1
            return 0

        dic = defaultdict(int)
        for i, v in enumerate(nums):
            dic[v] = i
        n = len(nums)
        res = 0
        for i in range(n):
            for j in range(i + 1, n):
                res = max(res, dfs(i, j) + 2)
        return res if res >= 3 else 0

    # 552. 学生出勤记录 II (Student Attendance Record II)
    def checkRecord(self, n: int) -> int:
        @cache
        def dfs(i: int, a: int, l: int) -> int:
            if a == 2 or l == 3:
                return 0
            if i == n:
                return 1
            # P / A / L
            return (
                dfs(i + 1, a, 0) + dfs(i + 1, a + 1, 0) + dfs(i + 1, a, l + 1)
            ) % MOD

        MOD = 10**9 + 7
        res = dfs(0, 0, 0)
        dfs.cache_clear()
        return res

    # 403. 青蛙过河 (Frog Jump)
    def canCross(self, stones: List[int]) -> bool:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n - 1:
                return True
            for id, v in enumerate(stones[i + 1 :], i + 1):
                if (v - stones[i]) - j < -1:
                    continue
                if (v - stones[i]) - j > 1:
                    break
                if dfs(id, v - stones[i]):
                    return True
            return False

        if stones[1] != 1:
            return False
        n = len(stones)
        return dfs(1, 1)

    # 1575. 统计所有可行路径 (Count All Possible Routes)
    def countRoutes(
        self, locations: List[int], start: int, finish: int, fuel: int
    ) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if abs(locations[i] - locations[finish]) > j:
                return 0
            return (
                sum(
                    dfs(k, j - abs(locations[k] - locations[i]))
                    for k in range(n)
                    if i != k
                )
                + (i == finish)
            ) % MOD

        n = len(locations)
        MOD = 10**9 + 7
        return dfs(start, fuel)

    # 2209. 用地毯覆盖后的最少白色砖块 (Minimum White Tiles After Covering With Carpets)
    def minimumWhiteTiles(self, floor: str, numCarpets: int, carpetLen: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if n - i <= (numCarpets - j) * carpetLen:
                return 0
            res = dfs(i + 1, j) + int(floor[i])
            if j < numCarpets:
                res = min(res, dfs(i + carpetLen, j + 1))
            return res

        n = len(floor)
        return dfs(0, 0)

    # 575. 分糖果 (Distribute Candies)
    def distributeCandies(self, candyType: List[int]) -> int:
        return min(len(set(candyType)), len(candyType) // 2)

    # 1103. 分糖果 II (Distribute Candies to People)
    def distributeCandies(self, candies: int, num_people: int) -> List[int]:
        res = [0] * num_people
        i = 0
        while candies:
            cnt = i + 1
            res[i % num_people] += min(cnt, candies)
            candies -= min(candies, cnt)
            i += 1
        return res

    # 3168. 候诊室中的最少椅子数 (Minimum Number of Chairs in a Waiting Room)
    def minimumChairs(self, s: str) -> int:
        res = 0
        cnt = 0
        for c in s:
            cnt += 1 if c == "E" else -1
            res = max(res, cnt)
        return res

    # 3169. 无需开会的工作日 (Count Days Without Meetings)
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        meetings.append([0, 0])
        meetings.append([days + 1, days + 1])
        meetings.sort()
        n = len(meetings)
        i = 0
        res = 0
        while i < n:
            right = meetings[i][1]
            j = i + 1
            while j < n and meetings[j][0] <= right + 1:
                right = max(right, meetings[j][1])
                j += 1
            if j == n:
                break
            res += meetings[j][0] - right - 1
            i = j
        return res

    # 3170. 删除星号以后字典序最小的字符串 (Lexicographically Minimum String After Removing Stars)
    def clearStars(self, s: str) -> str:
        a = [x for x in s]
        dic = [[] for _ in range(26)]
        bits = 0
        for i, c in enumerate(a):
            if c != "*":
                idx = ord(c) - ord("a")
                dic[idx].append(i)
                bits |= 1 << idx
            else:
                lb = (bits & -bits).bit_length() - 1
                a[dic[lb].pop()] = "*"
                if len(dic[lb]) == 0:
                    bits ^= 1 << lb
        return "".join([x for x in a if x != "*"])

    # 3171. 找到按位与最接近 K 的子数组 (Find Subarray With Bitwise AND Closest to K)
    def minimumDifference(self, nums: List[int], k: int) -> int:
        res = inf
        for i in range(len(nums)):
            _and = -1
            for j in range(i, max(-1, i - 28), -1):
                _and &= nums[j]
                res = min(res, abs(_and - k))
        return res

    # 419. 甲板上的战舰 (Battleships in a Board)
    def countBattleships(self, board: List[List[str]]) -> int:
        res = 0
        for i in range(len(board)):
            for j in range(len(board[0])):
                if (
                    board[i][j] == "X"
                    and (i == 0 or board[i - 1][j] == ".")
                    and (j == 0 or board[i][j - 1] == ".")
                ):
                    res += 1
        return res

    # 3174. 清除数字 (Clear Digits)
    def clearDigits(self, s: str) -> str:
        res = []
        for c in s:
            if c.isdigit():
                if res:
                    res.pop()
            else:
                res.append(c)
        return "".join(res)

    # 3175. 找到连续赢 K 场比赛的第一位玩家 (Find The First Player to win K Games in a Row)
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        mx_i = 0
        win = -1
        for i, v in enumerate(skills):
            if v > skills[mx_i]:
                mx_i = i
                win = 0
            win += 1
            if k == win:
                break
        return mx_i

    # 3176. 求出最长好子序列 I (Find the Maximum Length of a Good Subsequence I)
    def maximumLength(self, nums: List[int], k: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j < 0:
                return -inf
            res = 0
            for x in range(i - 1, -1, -1):
                if nums[x] != nums[i]:
                    res = max(res, dfs(x, j - 1))
                else:
                    res = max(res, dfs(x, j))
            return res + 1

        res = 0
        for i in range(len(nums)):
            for j in range(k + 1):
                res = max(res, dfs(i, j))
        return res

    # 2806. 取整购买后的账户余额 (Account Balance After Rounded Purchase)
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        return 100 - ((purchaseAmount + 5) // 10) * 10

    # 3178. 找出 K 秒后拿着球的孩子 (Find the Child Who Has the Ball After K Seconds)
    def numberOfChild(self, n: int, k: int) -> int:
        k %= (n - 1) << 1
        if k <= n - 1:
            return k
        k -= n - 1
        return n - 1 - k

    # 3179. K 秒后第 N 个元素的值 (Find the N-th Value After K Seconds)
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        return comb(k + n - 1, k) % (10**9 + 7)

    # 3180. 执行操作可获得的最大总奖励 I (Maximum Total Reward Using Operations I)
    def maxTotalReward(self, rewardValues: List[int]) -> int:

        def binary_search(target: int) -> int:
            if rewardValues[-1] <= target:
                return n
            if rewardValues[0] > target:
                return 0
            left = 0
            right = n - 1
            res = 0
            while left <= right:
                mid = left + ((right - left) >> 1)
                if rewardValues[mid] > target:
                    res = mid
                    right = mid - 1
                else:
                    left = mid + 1
            return res

        @cache
        def dfs(i: int, j: int) -> int:
            if i >= n:
                return j
            res = dfs(i + 1, j)
            if rewardValues[i] > j:
                res = max(
                    res, dfs(binary_search(j + rewardValues[i]), j + rewardValues[i])
                )
            return res

        rewardValues = list(set(rewardValues))
        rewardValues.sort()
        n = len(rewardValues)
        res = dfs(0, 0)
        dfs.cache_clear()
        return res

    # 730. 统计不同回文子序列 (Count Different Palindromic Subsequences)
    def countPalindromicSubsequences(self, s: str) -> int:
        @cache
        def dfs(c: int, i: int, j: int) -> int:
            if i > j:
                return 0
            if i == j:
                return int(ord(s[i]) - ord("a") == c)
            if s[i] == s[j] and ord(s[i]) - ord("a") == c:
                return (sum(dfs(x, i + 1, j - 1) for x in range(4)) + 2) % MOD
            if ord(s[i]) - ord("a") == c:
                return dfs(c, i, j - 1)
            if ord(s[j]) - ord("a") == c:
                return dfs(c, i + 1, j)
            return dfs(c, i + 1, j - 1)

        n = len(s)
        MOD = 10**9 + 7
        return sum(dfs(x, 0, n - 1) for x in range(4)) % MOD

    # 2779. 数组的最大美丽值 (Maximum Beauty of an Array After Applying Operation)
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        res = 0
        j = 0
        for i, v in enumerate(nums):
            while j < n and v + 2 * k >= nums[j]:
                j += 1
            res = max(res, j - i)
            if j == n:
                break
        return res

    # 521. 最长特殊序列 Ⅰ (Longest Uncommon Subsequence I)
    def findLUSlength(self, a: str, b: str) -> int:
        return -1 if a == b else max(len(a), len(b))

    # 522. 最长特殊序列 II (Longest Uncommon Subsequence II)
    def findLUSlength(self, strs: List[str]) -> int:
        def check(s: str, t: str) -> bool:
            i = 0
            j = 0
            while i < len(s) and j < len(t):
                if s[i] == t[j]:
                    i += 1
                j += 1
            return i == len(s)

        strs.sort(key=lambda k: -len(k))
        for i, s in enumerate(strs):
            if all(j == i or not check(s, t) for j, t in enumerate(strs)):
                return len(s)
        return -1

    # 3184. 构成整天的下标对数目 I (Count Pairs That Form a Complete Day I)
    # 3185. 构成整天的下标对数目 II (Count Pairs That Form a Complete Day II)
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        res = 0
        cnts = [0] * 24
        for v in hours:
            v %= 24
            if v:
                res += cnts[24 - v]
            cnts[v] += 1
        return res + cnts[0] * (cnts[0] - 1) // 2

    # 3186. 施咒的最大总伤害 (Maximum Total Damage With Spell Casting)
    def maximumTotalDamage(self, power: List[int]) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == n:
                return 0
            res = dfs(i + 1)
            j = i + 1
            while j < n and arr[j][0] - arr[i][0] <= 2:
                j += 1
            res = max(res, dfs(j) + arr[i][0] * arr[i][1])
            return res

        c = Counter(power)
        arr = []
        for k, v in c.items():
            arr.append((k, v))
        arr.sort(key=lambda k: k[0])
        n = len(arr)
        return dfs(0)

    # 2288. 价格减免 (Apply Discount to Prices)
    def discountPrices(self, sentence: str, discount: int) -> str:
        arr = sentence.split()
        for i, a in enumerate(arr):
            if a[0] == "$" and a[1:].isdigit():
                d = int(a[1:])
                d -= d * discount * 0.01
                arr[i] = "$" + f"{d:.2f}"
        return " ".join(arr)

    # 2748. 美丽下标对的数目 (Number of Beautiful Pairs)
    def countBeautifulPairs(self, nums: List[int]) -> int:
        cnts = [0] * 10
        res = 0
        for x in nums:
            res += sum(cnts[y] for y in range(1, 10) if gcd(x % 10, y) == 1)
            while x >= 10:
                x //= 10
            cnts[x] += 1
        return res

    # LCP 61. 气温变化趋势
    def temperatureTrend(self, temperatureA: List[int], temperatureB: List[int]) -> int:
        res = 0
        cnt = 0
        for (x0, y0), (x1, y1) in zip(pairwise(temperatureA), pairwise(temperatureB)):
            if x0 - y0 == x1 - y1 or (x0 - y0) * (x1 - y1) > 0:
                cnt += 1
                res = max(res, cnt)
            else:
                cnt = 0
        return res

    # 520. 检测大写字母 (Detect Capital)
    def detectCapitalUse(self, word: str) -> bool:
        return (
            word.islower()
            or word.isupper()
            or (word[0].isupper() and word[1:].islower())
        )

    # 503. 下一个更大元素 II (Next Greater Element II)
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [-1] * n
        nums = nums + nums
        st = []
        for i, v in enumerate(nums):
            while st and nums[st[-1]] < v:
                res[st.pop() % n] = v
            st.append(i)
        return res

    # 3190. 使所有元素都可以被 3 整除的最少操作数 (Find Minimum Operations to Make All Elements Divisible by Three)
    def minimumOperations(self, nums: List[int]) -> int:
        return sum(min(x % 3, 1) for x in nums)

    # 3191. 使二进制数组全部等于 1 的最少操作次数 I
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        for i in range(n):
            if nums[i]:
                continue
            if i + 2 >= n:
                return -1
            nums[i + 1] ^= 1
            nums[i + 2] ^= 1
            res += 1
        return res

    # 3192. 使二进制数组全部等于 1 的最少操作次数 II (Minimum Operations to Make Binary Array Elements Equal to One II)
    def minOperations(self, nums: List[int]) -> int:
        res = 0
        cnt = 0
        for x in nums:
            x ^= cnt
            if x == 0:
                res += 1
                cnt ^= 1
        return res

    # 2732. 找到矩阵中的好子集 (Find a Good Subset of the Matrix)
    def goodSubsetofBinaryMatrix(self, grid: List[List[int]]) -> List[int]:
        m = len(grid)
        n = len(grid[0])
        idx = [-1] * (1 << n)
        u = (1 << n) - 1
        for i in range(m):
            s = 0
            for j in range(n):
                s |= grid[i][j] << j
            if s == 0:
                return [i]
            sub = c = u ^ s
            while sub:
                if idx[sub] != -1:
                    return [idx[sub], i]
                sub = (sub - 1) & c
            idx[s] = i
        return []

    # 3194. 最小元素和最大元素的最小平均值 (Minimum Average of Smallest and Largest Elements)
    def minimumAverage(self, nums: List[int]) -> float:
        nums.sort()
        n = len(nums)
        return min(nums[i] + nums[n - i - 1] for i in range(n // 2)) / 2

    # 3195. 包含所有 1 的最小矩形面积 I (Find the Minimum Area to Cover All Ones I)
    def minimumArea(self, grid: List[List[int]]) -> int:
        l, r, t, b = inf, -inf, inf, -inf
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    l = min(l, j)
                    r = max(r, j)
                    t = min(t, i)
                    b = max(b, i)
        return (r - l + 1) * (b - t + 1)

    # 3196. 最大化子数组的总成本 (Maximize Total Cost of Alternating Subarrays)
    def maximumTotalCost(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n:
                return 0
            return max(dfs(i + 1, -1) + nums[i], dfs(i + 1, -j) + nums[i] * j)

        n = len(nums)
        return dfs(0, 1)

    # 3193. 统计逆序对的数目 (Count the Number of Inversions)
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == 0:
                return 1
            r = req[i - 1]
            if r != -1:
                return dfs(i - 1, r) if r <= j <= i + r else 0
            return sum(dfs(i - 1, j - k) for k in range(min(i, j) + 1)) % MOD

        req = [-1] * n
        req[0] = 0
        for end, cnt in requirements:
            req[end] = cnt
        if req[0] != 0:
            return 0
        MOD = 10**9 + 7
        return dfs(n - 1, req[-1])

    # 2734. 执行子串操作后的字典序最小字符串 (Lexicographically Smallest String After Substring Operation)
    def smallestString(self, s: str) -> str:
        arr = [x for x in s]
        n = len(arr)
        for i, v in enumerate(s):
            if v != "a":
                j = i
                while j < n and arr[j] != "a":
                    arr[j] = chr(ord(arr[j]) - 1)
                    j += 1
                return "".join(arr)
        arr[-1] = "z"
        return "".join(arr)

    # 2710. 移除字符串中的尾随零 (Remove Trailing Zeros From a String)
    def removeTrailingZeros(self, num: str) -> str:
        for i in range(len(num) - 1, -1, -1):
            if num[i] != "0":
                return num[: i + 1]

    # 2065. 最大化一张图中的路径价值 (Maximum Path Quality of a Graph)
    def maximalPathQuality(
        self, values: List[int], edges: List[List[int]], maxTime: int
    ) -> int:
        def dfs(i: int, t: int, k: int) -> None:
            if i == 0:
                nonlocal res
                res = max(res, k)
            for y, dt in g[i]:
                if t + dt > maxTime:
                    continue
                if not vis[y]:
                    vis[y] = True
                    dfs(y, t + dt, k + values[y])
                    vis[y] = False
                else:
                    dfs(y, t + dt, k)

        n = len(values)
        g = [[] for _ in range(n)]
        for u, v, t in edges:
            g[u].append((v, t))
            g[v].append((u, t))
        vis = [False] * n
        vis[0] = True
        res = 0
        dfs(0, 0, values[0])
        return res

    # 3200. 三角形的最大高度 (Maximum Height of a Triangle)
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        def check(x: int, y: int) -> int:
            left = [x, y]
            i = 1
            while left[i % 2] >= i:
                left[i % 2] -= i
                i += 1
            return i - 1

        return max(check(red, blue), check(blue, red))

    # 3201. 找出有效子序列的最大长度 I (Find the Maximum Length of Valid Subsequence I)
    def maximumLength(self, nums: List[int]) -> int:
        def cal_sum(d: int) -> int:
            res = 0
            for x in nums:
                if x & 1 != d:
                    res += 1
                    d ^= 1
            return res

        s = sum(x & 1 for x in nums)
        return max(s, len(nums) - s, cal_sum(0), cal_sum(1))

    # 3202. 找出有效子序列的最大长度 II (Find the Maximum Length of Valid Subsequence II)
    def maximumLength(self, nums: List[int], k: int) -> int:
        res = 0
        for m in range(k):
            f = [0] * k
            for x in nums:
                x %= k
                f[x] = f[m - x] + 1
            res = max(res, max(f))
        return res

    # 3203. 合并两棵树后的最小直径 (Find Minimum Diameter After Merging Two Trees)
    def minimumDiameterAfterMerge(
        self, edges1: List[List[int]], edges2: List[List[int]]
    ) -> int:
        def check(edges: List[List[int]]) -> int:
            n = len(edges) + 1
            g = [[] for _ in range(n)]
            for u, v in edges:
                g[u].append(v)
                g[v].append(u)
            res = 0

            def dfs(x: int, fa: int) -> int:
                pre = 0
                mx = 0
                for y in g[x]:
                    if y != fa:
                        cur = dfs(y, x)
                        mx = max(mx, cur + pre)
                        pre = max(pre, cur)
                nonlocal res
                res = max(res, mx + 1)
                return pre + 1

            dfs(0, -1)
            return res - 1

        d1 = check(edges1)
        d2 = check(edges2)
        return max(d1, d2, ((d1 + 1) >> 1) + ((d2 + 1) >> 1) + 1)

    # 1958. 检查操作是否合法 (Check if Move is Legal)
    def checkMove(
        self, board: List[List[str]], rMove: int, cMove: int, color: str
    ) -> bool:
        def check(dx: int, dy: int) -> bool:
            f = False
            x = rMove
            y = cMove
            while x + dx >= 0 and x + dx < n and y + dy >= 0 and y + dy < n:
                x += dx
                y += dy
                if board[x][y] == ".":
                    return False
                if board[x][y] == color:
                    return f
                f = True
            return False

        n = len(board)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if check(i, j):
                    return True
        return False

    # 724. 寻找数组的中心下标 (Find Pivot Index)
    def pivotIndex(self, nums: List[int]) -> int:
        s = sum(nums)
        l = 0
        r = s
        for i, v in enumerate(nums):
            r -= v
            if l == r:
                return i
            l += v
        return -1

    # 3206. 交替组 I (Alternating Groups I)
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        return sum(
            colors[i] != colors[i - 1] and colors[i] != colors[(i + 1) % len(colors)]
            for i in range(len(colors))
        )

    # 3207. 与敌人战斗后的最大分数 (Maximum Points After Enemy Battles)
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        n = len(enemyEnergies)
        res = 0
        enemyEnergies.sort()
        if enemyEnergies[0] > currentEnergy:
            return 0
        i = 0
        j = n - 1
        while i <= j:
            if currentEnergy >= enemyEnergies[i]:
                d, m = divmod(currentEnergy, enemyEnergies[i])
                res += d
                currentEnergy = m
            else:
                currentEnergy += enemyEnergies[j]
                j -= 1
        return res

    # 3208. 交替组 II (Alternating Groups II)
    def numberOfAlternatingGroups(self, colors: List[int], k: int) -> int:
        n = len(colors)
        res = 0
        d = 0
        for i in range(k - 1):
            if colors[i] != colors[i + 1]:
                d += 1
        if d == k - 1:
            res += 1
        for i in range(1, n):
            if colors[i] != colors[i - 1]:
                d -= 1
            if colors[(i + k - 2) % n] != colors[(i + k - 1) % n]:
                d += 1
            if d == k - 1:
                res += 1
        return res

    # 3210. 找出加密后的字符串 (Find the Encrypted String)
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        k %= n
        return s[k:] + s[:k]

    # 3211. 生成不含相邻零的二进制字符串 (Generate Binary Strings Without Adjacent Zeros)
    def validStrings(self, n: int) -> List[str]:
        def dfs(i: int, j: int) -> None:
            if i == n:
                res.append("".join(_list))
                return
            _list.append("1")
            dfs(i + 1, 1)
            _list.pop()
            if j == 1:
                _list.append("0")
                dfs(i + 1, 0)
                _list.pop()

        _list = []
        res = []
        dfs(0, 1)
        return res

    # 3211. 生成不含相邻零的二进制字符串 (Generate Binary Strings Without Adjacent Zeros)
    def validStrings(self, n: int) -> List[str]:
        res = []
        u = (1 << n) - 1
        for i in range(1 << n):
            c = i ^ u
            if c & (c >> 1) == 0:
                s = bin(i)[2:]
                # zfill 左边补0直到s长度为n位
                res.append(s.zfill(n))
        return res

    # 3212. 统计 X 和 Y 频数相等的子矩阵数量 (Count Submatrices With Equal Frequency of X and Y)
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0])
        res = 0
        xs = [[0] * (n + 1) for _ in range(m + 1)]
        ys = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                xs[i + 1][j + 1] = (
                    xs[i][j + 1] + xs[i + 1][j] - xs[i][j] + (grid[i][j] == "X")
                )
                ys[i + 1][j + 1] = (
                    ys[i][j + 1] + ys[i + 1][j] - ys[i][j] + (grid[i][j] == "Y")
                )
                if xs[i + 1][j + 1] == ys[i + 1][j + 1] != 0:
                    res += 1
        return res

    # 807. 保持城市天际线 (Max Increase to Keep City Skyline)
    def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
        n = len(grid)
        res = 0
        row = [max(row) for row in grid]
        col = [max(col) for col in zip(*grid)]
        for i in range(n):
            for j in range(n):
                res += max(0, min(row[i], col[j]) - grid[i][j])
        return res

    # 3216. 交换后字典序最小的字符串 (Lexicographically Smallest String After a Swap)
    def getSmallestString(self, s: str) -> str:
        n = len(s)
        arr = [x for x in s]
        for i in range(n - 1):
            if ord(arr[i]) % 2 == ord(arr[i + 1]) % 2 and int(arr[i]) > int(arr[i + 1]):
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                break
        return "".join(arr)

    # 3217. 从链表中移除在数组中存在的节点 (Delete Nodes From Linked List Present in Array)
    def modifiedList(
        self, nums: List[int], head: Optional[ListNode]
    ) -> Optional[ListNode]:
        s = set(nums)
        cur = dummy = ListNode(0, head)
        while head:
            while head and head.val in s:
                head = head.next
            cur.next = head
            cur = cur.next
            if head:
                head = head.next
        return dummy.next

    # 3218. 切蛋糕的最小总开销 I (Minimum Cost for Cutting Cake I) --O((mn)^2)
    def minimumCost(
        self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]
    ) -> int:
        @cache
        def dfs(i0: int, j0: int, i1: int, j1: int) -> int:
            if i0 == i1 and j0 == j1:
                return 0
            res = inf
            for i in range(i0, i1):
                res = min(
                    res, dfs(i0, j0, i, j1) + dfs(i + 1, j0, i1, j1) + horizontalCut[i]
                )
            for j in range(j0, j1):
                res = min(
                    res, dfs(i0, j0, i1, j) + dfs(i0, j + 1, i1, j1) + verticalCut[j]
                )
            return res

        return dfs(0, 0, m - 1, n - 1)

    # 3219. 切蛋糕的最小总开销 II (Minimum Cost for Cutting Cake II) --O(log(m) + log(n))
    def minimumCost(
        self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]
    ) -> int:
        horizontalCut.sort()
        verticalCut.sort()
        res = 0
        h = 1
        v = 1
        i = len(horizontalCut) - 1
        j = len(verticalCut) - 1
        while i >= 0 or j >= 0:
            if i >= 0 and (j >= 0 and horizontalCut[i] > verticalCut[j] or j < 0):
                res += horizontalCut[i] * v
                h += 1
                i -= 1
            else:
                res += verticalCut[j] * h
                v += 1
                j -= 1
        return res

    # 721. 账户合并 (Accounts Merge)
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        class union:

            def __init__(self, n: int) -> None:
                self.parent = [i for i in range(n)]
                self.rank = [1] * n

            def get_root(self, p: int) -> int:
                if self.parent[p] == p:
                    return p
                self.parent[p] = self.get_root(self.parent[p])
                return self.parent[p]

            def is_connected(self, p1: int, p2: int) -> bool:
                return self.get_root(p1) == self.get_root(p2)

            def union(self, p1: int, p2: int) -> None:
                root1 = self.get_root(p1)
                root2 = self.get_root(p2)
                if root1 == root2:
                    return
                if self.rank[root1] < self.rank[root2]:
                    self.parent[root1] = root2
                else:
                    self.parent[root2] = root1
                    if self.rank[root1] == self.rank[root2]:
                        self.rank[root1] += 1

        # name <-- 账户 <--> id
        post_to_id = defaultdict(int)
        id_to_post = defaultdict(str)
        post_to_name = defaultdict(str)
        id = 0
        for account in accounts:
            name = account[0]
            for i in range(1, len(account)):
                if account[i] not in post_to_id:
                    post_to_id[account[i]] = id
                    id_to_post[id] = account[i]
                    post_to_name[account[i]] = name
                    id += 1
        u = union(id)
        for account in accounts:
            for i in range(1, len(account)):
                u.union(post_to_id[account[1]], post_to_id[account[i]])
        root_to_list = defaultdict(list)
        for i in range(id):
            root = u.get_root(i)
            root_to_list[root].append(id_to_post[i])
        res = []
        for l in root_to_list.values():
            l.sort()
            name = post_to_name[l[0]]
            l.insert(0, name)
            res.append(l)
        return res

    # 3222. 求出硬币游戏的赢家 (Find the Winning Player in Coin Game)
    def losingPlayer(self, x: int, y: int) -> str:
        return "Alice" if min(x, y // 4) % 2 else "Bob"

    # 3223. 操作后字符串的最短长度 (Minimum Length of String After Operations)
    def minimumLength(self, s: str) -> int:
        cnt = Counter(s)
        return sum((c - 1) % 2 + 1 for c in cnt.values())

    # 3224. 使差值相等的最少数组改动次数 (Minimum Array Changes to Make Differences Equal)
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        diff = [0] * (k + 2)
        for i in range(n // 2):
            p = nums[i]
            q = nums[n - i - 1]
            if p > q:
                p, q = q, p
            diff[0] += 1
            mx = max(q, k - p)
            diff[q - p] -= 1
            diff[q - p + 1] += 1
            diff[mx + 1] += 1
        return min(accumulate(diff))

    # 3226. 使两个整数相等的位更改次数 (Number of Bit Changes to Make Two Integers Equal)
    def minChanges(self, n: int, k: int) -> int:
        if n & k != k:
            return -1
        return (n ^ k).bit_count()

    # 3227. 字符串元音游戏 (Vowels Game in a String)
    def doesAliceWin(self, s: str) -> bool:
        return any(c in "aeiou" for c in s)

    # 3228. 将 1 移动到末尾的最大操作次数 (Maximum Number of Operations to Move Ones to the End)
    def maxOperations(self, s: str) -> int:
        cnt1 = 0
        res = 0
        for i in range(len(s) - 1):
            if s[i] == "1":
                cnt1 += 1
                if s[i + 1] == "0":
                    res += cnt1
        return res

    # 2101. 引爆最多的炸弹 (Detonate the Maximum Bombs)
    def maximumDetonation(self, bombs: List[List[int]]) -> int:
        def check(start: int) -> int:

            vis = [False] * n
            vis[start] = True
            q = deque()
            q.append(start)
            res = 0
            while q:
                res += 1
                x = q.popleft()
                for y in g[x]:
                    if not vis[y]:
                        vis[y] = True
                        q.append(y)
            return res

        def dis(x, y) -> bool:
            return pow(bombs[y][1] - bombs[x][1], 2) + pow(
                bombs[y][0] - bombs[x][0], 2
            ) <= pow(bombs[x][2], 2)

        n = len(bombs)
        g = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if dis(i, j):
                    g[i].append(j)
        return max(check(i) for i in range(n))

    # 2766. 重新放置石块 (Relocate Marbles)
    def relocateMarbles(
        self, nums: List[int], moveFrom: List[int], moveTo: List[int]
    ) -> List[int]:
        s = set(nums)
        for f, t in zip(moveFrom, moveTo):
            s.discard(f)
            s.add(t)
        return sorted(s)

    # 2740. 找出分区值 (Find the Value of the Partition)
    def findValueOfPartition(self, nums: List[int]) -> int:
        return min(y - x for x, y in pairwise(sorted(nums)))

    # 3232. 判断是否可以赢得数字游戏 (Find if Digit Game Can Be Won)
    def canAliceWin(self, nums: List[int]) -> bool:
        a = sum(x for x in nums if x < 10)
        return sum(nums) != a * 2

    # 3233. 统计不是特殊数字的数字数量 (Find the Count of Numbers Which Are Not Special)
    def nonSpecialCount(self, l: int, r: int) -> int:
        is_prime = [True] * (isqrt(r) + 1)
        is_prime[1] = False
        for i in range(2, isqrt(r) + 1):
            if is_prime[i]:
                for j in range(i * i, isqrt(r) + 1, i):
                    is_prime[j] = False

        def check(x: int) -> int:
            res = 0
            for d in range(2, isqrt(x) + 1):
                if is_prime[d] and d * d <= x:
                    res += 1
            return x - res

        return check(r) - check(l - 1)

    # 3235. 判断矩形的两个角落是否可达 (Check if the Rectangle Corner Is Reachable)
    def canReachCorner(self, X: int, Y: int, circles: List[List[int]]) -> bool:
        class union:
            def __init__(self, n: int) -> None:
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

        n = len(circles)
        u = union(n + 2)
        for i, (x1, y1, r1) in enumerate(circles):
            if r1 >= x1 or y1 + r1 >= Y:
                u.union(i, n)
            if r1 >= y1 or x1 + r1 >= X:
                u.union(i, n + 1)
            for j, (x2, y2, r2) in enumerate(circles[:i]):
                if (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) <= (r1 + r2) * (
                    r1 + r2
                ):
                    u.union(i, j)
            if u.is_connected(n, n + 1):
                return False
        return True

    # LCP 40. 心算挑战
    def maxmiumScore(self, cards: List[int], cnt: int) -> int:
        l = [[] for _ in range(2)]
        cards.sort(reverse=True)
        for x in cards:
            l[x % 2].append(x)
        l[0] = list(accumulate(l[0], initial=0))
        l[1] = list(accumulate(l[1], initial=0))
        res = 0
        for k in range(0, len(l[1]), 2):
            if cnt >= k and cnt - k < len(l[0]):
                res = max(res, l[1][k] + l[0][cnt - k])
        return res

    # 572. 另一棵树的子树 (Subtree of Another Tree)
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        def is_same(root, subRoot) -> bool:
            return (
                root is None
                and subRoot is None
                or root
                and subRoot
                and root.val == subRoot.val
                and is_same(root.left, subRoot.left)
                and is_same(root.right, subRoot.right)
            )

        if root is None:
            return False
        return (
            root.val == subRoot.val
            and is_same(root, subRoot)
            or self.isSubtree(root.left, subRoot)
            or self.isSubtree(root.right, subRoot)
        )

    # 3238. 求出胜利玩家的数目 (Find the Number of Winning Players)
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        cnt = [[0] * 11 for _ in range(n)]
        for x, y in pick:
            cnt[x][y] += 1
        return sum(any(j > i for j in cnt[i]) for i in range(n))

    # 3239. 最少翻转次数使二进制矩阵回文 I (Minimum Number of Flips to Make Binary Grid Palindromic I)
    def minFlips(self, grid: List[List[int]]) -> int:
        def check(g: List[List[int]]) -> int:
            res = 0
            m = len(g)
            n = len(g[0])
            for i in range(m):
                for j in range(n // 2):
                    res += g[i][j] != g[i][n - j - 1]
            return res

        return min(check(grid), check([list(row) for row in zip(*grid)]))

    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        res = 0
        for i in range(m // 2):
            for j in range(n // 2):
                cnt1 = (
                    grid[i][j]
                    + grid[m - 1 - i][j]
                    + grid[i][n - j - 1]
                    + grid[m - 1 - i][n - j - 1]
                )
                res += min(cnt1, 4 - cnt1)
        if m % 2 and n % 2:
            res += grid[m // 2][n // 2]
        d = 0
        cnt1 = 0
        if m % 2:
            r = grid[m // 2]
            for j in range(n // 2):
                if r[j] != r[n - j - 1]:
                    d += 1
                else:
                    cnt1 += r[j] * 2
        if n % 2:
            for i in range(m // 2):
                if grid[i][n // 2] != grid[m - i - 1][n // 2]:
                    d += 1
                else:
                    cnt1 += grid[i][n // 2] * 2
        return res + (d if d else cnt1 % 4)

    # 3242. 设计相邻元素求和服务 (Design Neighbor Sum Service)
    class neighborSum:

        def __init__(self, grid: List[List[int]]):
            self.grid = grid
            self.d = defaultdict(tuple)
            self.n = len(grid)
            for i in range(self.n):
                for j in range(self.n):
                    self.d[grid[i][j]] = (i, j)

        def adjacentSum(self, value: int) -> int:
            res = 0
            (x, y) = self.d[value]
            for i in range(max(0, x - 1), min(self.n, x + 2)):
                for j in range(max(0, y - 1), min(self.n, y + 2)):
                    if i == x or j == y:
                        res += self.grid[i][j]
            return res - self.grid[x][y]

        def diagonalSum(self, value: int) -> int:
            res = 0
            x, y = self.d[value]
            for i in range(max(0, x - 1), min(self.n, x + 2)):
                for j in range(max(0, y - 1), min(self.n, y + 2)):
                    if i != x and j != y:
                        res += self.grid[i][j]
            return res

    # 3243. 新增道路查询后的最短距离 I (Shortest Distance After Road Addition Queries I)
    def shortestDistanceAfterQueries(
        self, n: int, queries: List[List[int]]
    ) -> List[int]:
        @cache
        def dfs(i: int) -> int:
            if i == n - 1:
                return 0
            return min(dfs(j) + 1 for j in g[i])

        g = [[] * n for _ in range(n)]
        for i in range(n - 1):
            g[i].append(i + 1)
        res = []
        for u, v in queries:
            g[u].append(v)
            res.append(dfs(0))
            dfs.cache_clear()
        return res

    # 676. 实现一个魔法字典 (Implement Magic Dictionary)
    class MagicDictionary:

        def __init__(self):
            self.d = defaultdict(set)

        def buildDict(self, dictionary: List[str]) -> None:
            for dic in dictionary:
                self.d[len(dic)].add(dic)

        def search(self, searchWord: str) -> bool:
            for s in self.d[len(searchWord)]:
                diff = 0
                for x, y in zip(s, searchWord):
                    if x != y:
                        diff += 1
                        if diff > 1:
                            break
                if diff == 1:
                    return True
            return False

    # 3248. 矩阵中的蛇 (Snake in Matrix)
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        x = 0
        y = 0
        for c in commands:
            if c == "UP":
                x -= 1
            elif c == "DOWN":
                x += 1
            elif c == "LEFT":
                y -= 1
            else:
                y += 1
        return x * n + y

    # 3249. 统计好节点的数目 (Count the Number of Good Nodes)
    def countGoodNodes(self, edges: List[List[int]]) -> int:
        def dfs(x: int, fa: int) -> int:
            pre = 0
            s = 0
            ret = True
            for y in g[x]:
                if y != fa:
                    cnt = dfs(y, x)
                    if pre and pre != cnt:
                        ret = False
                    pre = cnt
                    s += cnt
            if ret:
                nonlocal res
                res += 1
            return s + 1

        n = len(edges) + 1
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        res = 0
        dfs(0, -1)
        return res

    # 3250. 单调数组对的数目 I (Find the Count of Monotonic Pairs I)
    def countOfPairs(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, x: int, y: int) -> int:
            if i == n:
                return 1
            return (
                sum(
                    dfs(i + 1, c, nums[i] - c)
                    for c in range(max(x, nums[i] - y), nums[i] + 1)
                )
                % MOD
            )

        n = len(nums)
        MOD = 10**9 + 7
        return dfs(0, 0, 50)

    # 551. 学生出勤记录 I (Student Attendance Record I)
    def checkRecord(self, s: str) -> bool:
        return s.count("A") < 2 and "LLL" not in s

    # 3254. 长度为 K 的子数组的能量值 I (Find the Power of K-Size Subarrays I)
    # 3255. 长度为 K 的子数组的能量值 II (Find the Power of K-Size Subarrays II)
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        cnt = 1
        n = len(nums)
        res = [-1] * (n - k + 1)
        for i, x in enumerate(nums):
            if i and x - nums[i - 1] == 1:
                cnt += 1
            if i >= k and nums[i - k + 1] - nums[i - k] == 1:
                cnt -= 1
            if cnt == k:
                res[i - k + 1] = x
        return res

    # 3258. 统计满足 K 约束的子字符串数量 I (Count Substrings That Satisfy K-Constraint I)
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        res = 0
        cnt = [0] * 2
        j = 0
        for i, v in enumerate(s):
            cnt[int(v)] += 1
            while cnt[0] > k and cnt[1] > k:
                cnt[int(s[j])] -= 1
                j += 1
            res += i - j + 1
        return res

    # 3259. 超级饮料的最大强化能量 (Maximum Energy Boost From Two Drinks)
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i >= n:
                return 0
            return max(dfs(i + 1, j), dfs(i + 2, j ^ 1)) + (
                energyDrinkA[i] if j == 0 else energyDrinkB[i]
            )

        n = len(energyDrinkA)
        return max(dfs(0, 0), dfs(0, 1))

    # 3270. 求出数字答案 (Find the Key of the Numbers)
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        res = 0
        p = 1
        arr = [num1, num2, num3]
        while any(x for x in arr):
            d = 9
            for i in range(3):
                d = min(d, arr[i] % 10)
                arr[i] //= 10
            res += p * d
            p *= 10
        return res

    # 3271. 哈希分割字符串 (Hash Divided String)
    def stringHash(self, s: str, k: int) -> str:
        _s = 0
        res = []
        for i, v in enumerate(s):
            _s += ord(v) - ord("a")
            _s %= 26
            if (i + 1) % k == 0:
                res.append(chr(ord("a") + _s))
                _s = 0
        return "".join(res)

    # 3274. 检查棋盘方格颜色是否相同 (Check if Two Chessboard Squares Have the Same Color)
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        return (ord(coordinate1[0]) - ord(coordinate2[0])) % 2 == (
            ord(coordinate1[1]) - ord(coordinate2[1])
        ) % 2

    # 3275. 第 K 近障碍物查询 (K-th Nearest Obstacle Queries)
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        q = []
        res = [-1] * len(queries)
        heapq.heapify(q)
        for i, (x, y) in enumerate(queries):
            d = abs(x) + abs(y)
            heapq.heappush(q, -d)
            if len(q) > k:
                heapq.heappop(q)
            if len(q) == k:
                res[i] = -q[0]
        return res

    # 3276. 选择矩阵中单元格的最大得分 (Select Cells in Grid With Maximum Score)
    def maxScore(self, grid: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == len(_list):
                return 0
            res = dfs(i + 1, j)
            c = (u ^ j) & _list[i][1]
            while c:
                lb = (c & -c).bit_length() - 1
                res = max(res, dfs(i + 1, j | (1 << lb)) + _list[i][0])
                c &= c - 1
            return res

        m = len(grid)
        n = len(grid[0])
        d = defaultdict(int)
        for i in range(m):
            for j in range(n):
                d[grid[i][j]] |= 1 << i
        _list = []
        for i, v in d.items():
            _list.append([i, v])
        u = (1 << m) - 1
        return dfs(0, 0)

    # 2708. 一个小组的最大实力值 (Maximum Strength of a Group)
    def maxStrength(self, nums: List[int]) -> int:
        def dfs(i: int, j: int, k: bool) -> None:
            if i == n:
                nonlocal res
                if k:
                    res = max(res, j)
                return
            dfs(i + 1, j, k)
            dfs(i + 1, j * nums[i], True)

        n = len(nums)
        res = -inf
        dfs(0, 1, False)
        return res

    # 2708. 一个小组的最大实力值 (Maximum Strength of a Group)
    def maxStrength(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, j: int, k: bool) -> int:
            if i == len(nums):
                return 1 if k else -inf
            res = dfs(i + 1, j, k)
            if j:
                return max(res, dfs(i + 1, j ^ int(nums[i] < 0), True) * nums[i])
            else:
                return min(res, dfs(i + 1, j ^ int(nums[i] < 0), True) * nums[i])

        return dfs(0, 1, False)

    # 2708. 一个小组的最大实力值 (Maximum Strength of a Group)
    def maxStrength(self, nums: List[int]) -> int:
        nums.sort()
        for i in range(1, len(nums), 2):
            if nums[i] >= 0:
                break
            nums[i] *= -1
            nums[i - 1] *= -1
        nums.sort()
        if nums[-1] > 0:
            res = 1
            for i in range(len(nums) - 1, -1, -1):
                if nums[i] <= 0:
                    break
                res *= nums[i]
            return res
        if nums[-1] < 0:
            return nums[-1]
        return 0

    # 2860. 让所有学生保持开心的分组方法数 (Happy Students)
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        ans = nums[0] > 0  # 一个学生都不选
        for i, (x, y) in enumerate(pairwise(nums), 1):
            if x < i < y:
                ans += 1
        return ans + 1  # 一定可以都选

    # 2181. 合并零之间的节点 (Merge Nodes in Between Zeros)
    def mergeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        res = head
        cur = head.next
        s = 0
        while cur:
            if cur.val:
                s += cur.val
            else:
                cur.val = s
                s = 0
                head.next = cur
                head = head.next
            cur = cur.next
        return res.next

    # 3280. 将日期转换为二进制表示 (Convert Date to Binary)
    def convertDateToBinary(self, date: str) -> str:
        y, m, d = map(int, date.split("-"))
        return bin(y)[2:] + "-" + bin(m)[2:] + "-" + bin(d)[2:]

    # 3281. 范围内整数的最大得分 (Maximize Score of Numbers in Ranges)
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        def check(t: int) -> bool:
            left = start[0]
            for i in range(len(start) - 1):
                if left + t > start[i + 1] + d:
                    return False
                left = max(left + t, start[i + 1])
            return True

        start.sort()
        left = 0
        right = start[-1] + d
        res = 0
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = mid
                left = mid + 1
            else:
                right = mid - 1
        return res

    # 3282. 到达数组末尾的最大得分 (Reach End of Array With Max Score)
    def findMaximumScore(self, nums: List[int]) -> int:
        nums.pop(-1)
        res = 0
        mx = 0
        for x in nums:
            mx = max(mx, x)
            res += mx
        return res

    # 3283. 吃掉所有兵需要的最多移动次数 (Maximum Number of Moves to Kill All Pawns)
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j == u:
                return 0
            res = inf if j.bit_count() % 2 else 0
            op = min if j.bit_count() % 2 else max
            c = u ^ j
            while c:
                lb = (c & -c).bit_length() - 1
                d = dis[i][positions[lb][0] * 50 + positions[lb][1]]
                res = op(res, dfs(lb, j | (1 << lb)) + d)
                c &= c - 1
            return res

        def cal(i: int, start_x: int, start_y: int) -> None:
            vis = [[False] * 50 for _ in range(50)]
            vis[start_x][start_y] = True
            q = deque()
            cnt = int(i < n)
            q.append((start_x, start_y))
            step = 0
            while q and cnt < n:
                step += 1
                size = len(q)
                for _ in range(size):
                    (x, y) = q.popleft()
                    for dx, dy in (
                        [-1, -2],
                        [-1, 2],
                        [1, -2],
                        [1, 2],
                        [2, -1],
                        [2, 1],
                        [-2, 1],
                        [-2, -1],
                    ):
                        nx, ny = x + dx, y + dy
                        if 50 > nx >= 0 and 50 > ny >= 0 and not vis[nx][ny]:
                            vis[nx][ny] = True
                            dis[i][nx * 50 + ny] = step
                            if (nx, ny) in s:
                                cnt += 1
                                if cnt == n:
                                    return
                            q.append((nx, ny))

        n = len(positions)
        s = set()
        for x, y in positions:
            s.add((x, y))
        dis = [[0] * 2500 for _ in range(n + 1)]
        for i, (x, y) in enumerate(positions):
            cal(i, x, y)
        cal(n, kx, ky)
        u = (1 << n) - 1
        return dfs(n, 0)

    # 2576. 求出最多标记下标 (Find the Maximum Number of Marked Indices)
    def maxNumOfMarkedIndices(self, nums: List[int]) -> int:
        nums.sort()
        i = 0
        for j in range((len(nums) + 1) // 2, len(nums)):
            if nums[i] * 2 <= nums[j]:
                i += 1
        return i * 2

    # 3285. 找到稳定山的下标 (Find Indices of Stable Mountains)
    def stableMountains(self, height: List[int], threshold: int) -> List[int]:
        return [i for i in range(1, len(height)) if height[i - 1] > threshold]

    # 3286. 穿越网格图的安全路径 (Find a Safe Walk Through a Grid) -- 0-1bfs
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m = len(grid)
        n = len(grid[0])
        q = deque()
        q.append([0, 0])
        dis = [[inf] * n for _ in range(m)]
        dis[0][0] = grid[0][0]
        while True:
            [x, y] = q.popleft()
            if dis[x][y] >= health:
                return False
            if x == m - 1 and y == n - 1:
                return True
            for dx, dy in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < m
                    and 0 <= ny < n
                    and dis[x][y] + grid[nx][ny] < dis[nx][ny]
                ):
                    dis[nx][ny] = dis[x][y] + grid[nx][ny]
                    if grid[nx][ny]:
                        q.append([nx, ny])
                    else:
                        q.appendleft([nx, ny])

    # 3289. 数字小镇中的捣蛋鬼 (The Two Sneaky Numbers of Digitville)
    def getSneakyNumbers(self, nums: List[int]) -> List[int]:
        n = len(nums) - 2
        xor = n ^ (n + 1)
        for i, x in enumerate(nums):
            xor ^= x ^ i
        lb = (xor & -xor).bit_length() - 1
        res = [0] * 2
        for i, x in enumerate(nums):
            if i < n:
                res[(i >> lb) & 1] ^= i
            res[(x >> lb) & 1] ^= x
        return res

    # 3290. 最高乘法得分 (Maximum Multiplication Score)
    def maxScore(self, a: List[int], b: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == 4:
                return 0
            if n - j < 4 - i:
                return -inf
            return max(dfs(i, j + 1), dfs(i + 1, j + 1) + a[i] * b[j])

        n = len(b)
        res = dfs(0, 0)
        dfs.cache_clear()
        return res

    # 3291. 形成目标字符串需要的最少字符串数 I (Minimum Number of Valid Strings to Form Target I)
    def minValidStrings(self, words: List[str], target: str) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == n:
                return 0
            node = trie
            res = inf
            for j in range(i, n):
                id = ord(target[j]) - ord("a")
                if node.children[id] is None:
                    break
                res = min(res, dfs(j + 1) + 1)
                node = node.children[id]
            return res

        class Trie:
            def __init__(self) -> None:
                self.children = [None] * 26

            def insert(self, s: str) -> None:
                node = self
                for i in range(len(s)):
                    j = ord(s[i]) - ord("a")
                    if node.children[j] is None:
                        node.children[j] = Trie()
                    node = node.children[j]

        n = len(target)
        trie = Trie()
        for s in words:
            trie.insert(s)
        res = dfs(0)
        return res if res <= n else -1

    # 2414. 最长的字母序连续子字符串的长度 (Length of the Longest Alphabetical Continuous Substring)
    def longestContinuousSubstring(self, s: str) -> int:
        res = 1
        cnt = 1
        for i in range(1, len(s)):
            if ord(s[i]) - ord(s[i - 1]) == 1:
                cnt += 1
            else:
                cnt = 1
            res = max(res, cnt)
        return res

    # 2374. 边积分最高的节点 (Node With Highest Edge Score)
    def edgeScore(self, edges: List[int]) -> int:
        n = len(edges)
        score = [0] * n
        for i, v in enumerate(edges):
            score[v] += i
        mx = -1
        res = -1
        for i, v in enumerate(score):
            if v > mx:
                mx = v
                res = i
        return res

    # 997. 找到小镇的法官 (Find the Town Judge)
    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        t = [0] * (n + 1)
        for u, v in trust:
            t[v] += 1
            t[u] -= 1
        for i, v in enumerate(t[1:], 1):
            if v == n - 1:
                return i
        return -1

    # 1014. 最佳观光组合 (Best Sightseeing Pair)
    def maxScoreSightseeingPair(self, values: List[int]) -> int:
        res, mx = -inf, -inf
        for i, v in enumerate(values):
            res = max(res, v - i + mx)
            mx = max(mx, v + i)
        return res

    # 3295. 举报垃圾信息 (Report Spam Message)
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        s = set(bannedWords)
        return sum(m in s for m in message) >= 2

    # 3296. 移山所需的最少秒数 (Minimum Number of Seconds to Make Mountain Height Zero)
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def check2(w: int, t: int) -> int:
            left = 0
            right = mountainHeight
            res = 0
            while left <= right:
                mid = left + ((right - left) >> 1)
                if w * (mid + 1) * mid // 2 <= t:
                    res = mid
                    left = mid + 1
                else:
                    right = mid - 1
            return res

        def check(t: int) -> bool:
            h = 0
            for w in workerTimes:
                h += check2(w, t)
                if h >= mountainHeight:
                    return True
            return False

        left = 0
        right = 10**16
        res = -1
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = mid
                right = mid - 1
            else:
                left = mid + 1
        return res

    # 3297. 统计重新排列后包含另一个字符串的子字符串数目 I (Count Substrings That Can Be Rearranged to Contain a String I)
    # 3298. 统计重新排列后包含另一个字符串的子字符串数目 II (Count Substrings That Can Be Rearranged to Contain a String II)
    def validSubstringCount(self, word1: str, word2: str) -> int:
        cnt = [0] * 26
        c = 0
        for w in word2:
            cnt[ord(w) - ord("a")] += 1
            if cnt[ord(w) - ord("a")] == 1:
                c += 1
        i = 0
        res = 0
        for w in word1:
            cnt[ord(w) - ord("a")] -= 1
            if cnt[ord(w) - ord("a")] == 0:
                c -= 1
            while c == 0:
                cnt[ord(word1[i]) - ord("a")] += 1
                if cnt[ord(word1[i]) - ord("a")] == 1:
                    c += 1
                i += 1
            res += i
        return res

    # 2207. 字符串中最多数目的子序列 (Maximize Number of Subsequences in a String)
    def maximumSubsequenceCount(self, text: str, pattern: str) -> int:
        def check(s: str) -> int:
            cnt0 = 0
            res = 0
            for v in s:
                if v == pattern[1]:
                    res += cnt0
                if v == pattern[0]:
                    cnt0 += 1
            return res

        return max(check(pattern[0] + text), check(text + pattern[1]))

    # 76. 最小覆盖子串 (Minimum Window Substring)
    def minWindow(self, s: str, t: str) -> str:
        if len(s) < len(t):
            return ""
        d = defaultdict(int)
        less = 0
        for c in t:
            d[ord(c) - ord("a")] -= 1
            if d[ord(c) - ord("a")] == -1:
                less -= 1
        res_left = -1
        res_right = -1
        left = 0
        for right, c in enumerate(s):
            d[ord(c) - ord("a")] += 1
            if d[ord(c) - ord("a")] == 0:
                less += 1
            while less == 0:
                if res_left == -1 or right - left < res_right - res_left:
                    res_left, res_right = left, right
                d[ord(s[left]) - ord("a")] -= 1
                if d[ord(s[left]) - ord("a")] == -1:
                    less -= 1
                left += 1
        return "" if res_left == -1 else s[res_left : res_right + 1]

    # 2306. 公司命名 (Naming a Company)
    def distinctNames(self, ideas: List[str]) -> int:
        res = 0
        d = defaultdict(set)
        for s in ideas:
            d[s[0]].add(s[1:])
        for a, b in combinations(d.values(), 2):
            m = len(a & b)
            res += (len(a) - m) * (len(b) - m)
        return res << 1

    # 2535. 数组元素和与数字和的绝对差 (Difference Between Element Sum and Digit Sum of an Array)
    def differenceOfSum(self, nums: List[int]) -> int:
        s = 0
        for x in nums:
            s += x
            while x:
                s -= x % 10
                x //= 10
        return s

    # 2073. 买票需要的时间 (Time Needed to Buy Tickets)
    def timeRequiredToBuy(self, tickets: List[int], k: int) -> int:
        return sum(
            min(v, tickets[k] if i <= k else tickets[k] - 1)
            for i, v in enumerate(tickets)
        )

    # 3300. 替换为数位和以后的最小元素 (Minimum Element After Replacement With Digit Sum)
    def minElement(self, nums: List[int]) -> int:
        res = inf
        for x in nums:
            cur = 0
            while x:
                cur += x % 10
                x //= 10
            res = min(res, cur)
        return res

    # 3301. 高度互不相同的最大塔高和 (Maximize the Total Height of Unique Towers)
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        maximumHeight.sort()
        res = 0
        mx = inf
        for i in range(len(maximumHeight) - 1, -1, -1):
            mx = min(mx - 1, maximumHeight[i])
            if mx == 0:
                return -1
            res += mx
        return res

    # 3304. 找出第 K 个字符 I (Find the K-th Character in String Game I)
    def kthCharacter(self, k: int) -> str:
        s = "a"
        while len(s) < k:
            tmp = []
            for i in range(len(s)):
                tmp.append((chr)((ord(s[i]) - ord("a") + 1 % 26) + ord("a")))
            s += "".join(tmp)
        return s[k - 1]

    # 2565. 最少得分子序列 (Subsequence With the Minimum Score)
    def minimumScore(self, s: str, t: str) -> int:
        n, m = len(s), len(t)
        suf = [m] * (n + 1)
        j = m - 1
        for i in range(n - 1, -1, -1):
            if s[i] == t[j]:
                j -= 1
            if j < 0:  # t 是 s 的子序列
                return 0
            suf[i] = j + 1

        ans = suf[0]  # 删除 t[:suf[0]]
        j = 0
        for i, c in enumerate(s):
            if c == t[j]:  # 注意上面判断了 t 是 s 子序列的情况，这里 j 不会越界
                j += 1
                ans = min(ans, suf[i + 1] - j)  # 删除 t[j:suf[i+1]]
        return ans

    # 3302. 字典序最小的合法序列 (Find the Lexicographically Smallest Valid Sequence)
    def validSequence(self, s: str, t: str) -> List[int]:
        n, m = len(s), len(t)
        suf = [0] * (n + 1)
        suf[n] = m
        j = m - 1
        for i in range(n - 1, -1, -1):
            if j >= 0 and s[i] == t[j]:
                j -= 1
            suf[i] = j + 1

        ans = []
        changed = False  # 是否修改过
        j = 0
        for i, c in enumerate(s):
            if c == t[j] or not changed and suf[i + 1] <= j + 1:
                if c != t[j]:
                    changed = True
                ans.append(i)
                j += 1
                if j == m:
                    return ans
        return []

    # 3303. 第一个几乎相等子字符串的下标 (Find the Occurrence of First Almost Equal Substring) --z函数
    def minStartingIndex(self, s: str, pattern: str) -> int:
        def calc_z(s: str) -> list[int]:
            n = len(s)
            z = [0] * n
            box_l = box_r = 0  # z-box 左右边界
            for i in range(1, n):
                if i <= box_r:
                    z[i] = min(z[i - box_l], box_r - i + 1)  # 改成手动 if 可以加快速度
                while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                    box_l, box_r = i, i + z[i]
                    z[i] += 1
            return z

        pre_z = calc_z(pattern + s)
        suf_z = calc_z(pattern[::-1] + s[::-1])
        m = len(pattern)
        for i in range(m, len(s) + 1):
            if pre_z[i] + suf_z[-i] + 1 >= m:
                return i - m
        return -1

    # 3305. 元音辅音字符串计数 I (Count of Substrings Containing Every Vowel and K Consonants I)
    # 3306. 元音辅音字符串计数 II (Count of Substrings Containing Every Vowel and K Consonants II)
    def countOfSubstrings(self, word: str, k: int) -> int:
        def check(k: int) -> int:
            d = defaultdict(int)
            left = 0
            consonant = 0
            res = 0
            for v in word:
                # 元音
                if v in "aeiou":
                    d[v] += 1
                else:
                    consonant += 1
                while len(d) == 5 and consonant >= k:
                    if word[left] in "aeiou":
                        d[word[left]] -= 1
                        if d[word[left]] == 0:
                            del d[word[left]]
                    else:
                        consonant -= 1
                    left += 1
                res += left
            return res

        return check(k) - check(k + 1)

    # 3307. 找出第 K 个字符 II (Find the K-th Character in String Game II)
    def kthCharacter(self, k: int, operations: List[int]) -> str:
        if not operations:
            return "a"
        n = len(operations)
        op = operations.pop()
        if k <= 1 << (n - 1):
            return self.kthCharacter(k, operations)
        res = self.kthCharacter(k - (1 << (n - 1)), operations)
        res = (ord(res) - ord("a") + op) % 26
        return ascii_lowercase[res]

    # 1845. 座位预约管理系统 (Seat Reservation Manager)
    class SeatManager:

        def __init__(self, n: int):
            self.avaliable = list(range(1, n + 1))

        def reserve(self) -> int:
            return heapq.heappop(self.avaliable)

        def unreserve(self, seatNumber: int) -> None:
            heapq.heappush(self.avaliable, seatNumber)

    # 1870. 准时到达的列车最小时速 (Minimum Speed to Arrive on Time)
    def minSpeedOnTime(self, dist: List[int], hour: float) -> int:
        def check(s: int) -> bool:
            h = 0
            for i in range(len(dist) - 1):
                h += (dist[i] - 1) // s + 1
            h += dist[len(dist) - 1] / s
            return h <= hour

        left = 1
        right = 10**7
        res = -1
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = mid
                right = mid - 1
            else:
                left = mid + 1
        return res

    # 1928. 规定时间内到达终点的最小花费 (Minimum Cost to Reach Destination in Time)
    def minCost(
        self, maxTime: int, edges: List[List[int]], passingFees: List[int]
    ) -> int:
        n = len(passingFees)
        dis = [[inf] * (maxTime + 1) for _ in range(n)]
        g = [[] for _ in range(n)]
        for u, v, t in edges:
            g[u].append((v, t))
            g[v].append((u, t))
        for i in range(maxTime + 1):
            dis[0][i] = passingFees[0]
        q = []
        # fee, node, time
        q.append((passingFees[0], 0, 0))
        heapq.heapify(q)
        while q:
            (fee, x, time) = heapq.heappop(q)
            if time > maxTime:
                continue
            if x == n - 1:
                return fee
            for y, dt in g[x]:
                if time + dt > maxTime:
                    continue
                if fee + passingFees[y] < dis[y][time + dt]:
                    dis[y][time + dt] = fee + passingFees[y]
                    heapq.heappush(q, (fee + passingFees[y], y, time + dt))
        return -1

    # 134. 加油站 (Gas Station)
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        m = inf
        min_idx = 0
        cur = 0
        for i, (x, y) in enumerate(zip(gas, cost)):
            cur += x - y
            if cur < m:
                m = cur
                min_idx = i
        return -1 if cur < 0 else (min_idx + 1) % len(gas)

    # 1436. 旅行终点站 (Destination City)
    def destCity(self, paths: List[List[str]]) -> str:
        d = defaultdict(str)
        for u, v in paths:
            d[u] = v
        res = paths[0][0]
        while d[res]:
            res = d[res]
        return res

    # 3314. 构造最小位运算数组 I (Construct the Minimum Bitwise Array I)
    # 3315. 构造最小位运算数组 II (Construct the Minimum Bitwise Array II)
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        for i, x in enumerate(nums):
            if x == 2:
                nums[i] = -1
            else:
                t = ~x
                nums[i] ^= (t & -t) >> 1
        return nums

    # 3316. 从原字符串里进行删除操作的最多次数 (Find Maximum Removals From Source String)
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i == n or j == m:
                if i == n and j == m:
                    return 0
                if j == m:
                    return dfs(i + 1, j) + s[i]
                return -inf
            if not s[i]:
                return dfs(i + 1, j + int(source[i] == pattern[j]))
            if source[i] != pattern[j]:
                return dfs(i + 1, j) + 1
            return max(dfs(i + 1, j + 1), dfs(i + 1, j) + 1)

        n = len(source)
        m = len(pattern)
        s = [False] * n
        for t in targetIndices:
            s[t] = True
        return dfs(0, 0)

    # 1884. 鸡蛋掉落-两枚鸡蛋 (Egg Drop With 2 Eggs and N Floors)
    def twoEggDrop(self, n: int) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == 0:
                return 0
            return min(max(j, dfs(i - j) + 1) for j in range(1, i + 1))

        return dfs(n)

    # 3319. 第 K 大的完美二叉子树的大小 (K-th Largest Perfect Subtree Size in Binary Tree)
    def kthLargestPerfectSubtree(self, root: Optional[TreeNode], k: int) -> int:
        def dfs(root: Optional[TreeNode]) -> tuple:
            if root is None:
                return (0, 0)
            left = dfs(root.left)
            right = dfs(root.right)
            if left[0] == -1 or right[0] == -1 or left[1] != right[1]:
                return (-1, -1)
            _l.append(left[0] + right[0] + 1)
            return (left[0] + right[0] + 1, left[1] + 1)

        _l = []
        dfs(root)
        if len(_l) < k:
            return -1
        _l.sort(reverse=True)
        return _l[k - 1]

    # 3320. 统计能获胜的出招序列数 (Count The Number of Winning Sequences)
    def countWinningSequences(self, s: str) -> int:
        @cache
        def dfs(i: int, j: int, k: chr) -> int:
            if i == n:
                return int(j > 0)
            if n - i + j <= 0:
                return 0
            res = 0
            for c in ["F", "W", "E"]:
                if c == k:
                    continue
                if c == s[i]:
                    res += dfs(i + 1, j, c)
                elif (
                    c == "F"
                    and s[i] == "E"
                    or c == "E"
                    and s[i] == "W"
                    or c == "W"
                    and s[i] == "F"
                ):
                    res += dfs(i + 1, j + 1, c)
                else:
                    res += dfs(i + 1, j - 1, c)
            return res % MOD

        n = len(s)
        MOD = 10**9 + 7
        return dfs(0, 0, "A")

    # 908. 最小差值 I (Smallest Range I)
    def smallestRangeI(self, nums: List[int], k: int) -> int:
        return max(max(nums) - min(nums) - 2 * k, 0)

    # 910. 最小差值 II (Smallest Range II)
    def smallestRangeII(self, nums: List[int], k: int) -> int:
        nums.sort()
        res = nums[-1] - nums[0]
        for x, y in pairwise(nums):
            mx = max(nums[-1] - k, x + k)
            mi = min(nums[0] + k, y - k)
            res = min(res, mx - mi)
        return res

    # 3324. 出现在屏幕上的字符串序列 (Find the Sequence of Strings Appeared on the Screen)
    def stringSequence(self, target: str) -> List[str]:
        res = []
        for t in target:
            pre = res[-1] if len(res) else ""
            for c in range(0, ord(t) - ord("a") + 1):
                res.append(pre + chr(c + ord("a")))
        return res

    # 3325. 字符至少出现 K 次的子字符串 I (Count Substrings With K-Frequency Characters I)
    def numberOfSubstrings(self, s: str, k: int) -> int:
        left = 0
        res = 0
        cnt = [0] * 26
        for c in s:
            cnt[ord(c) - ord("a")] += 1
            while cnt[ord(c) - ord("a")] >= k:
                cnt[ord(s[left]) - ord("a")] -= 1
                left += 1
            res += left
        return res

    # 3326. 使数组非递减的最少除法操作次数 (Minimum Division Operations to Make Array Non Decreasing)
    def minOperations(self, nums: List[int]) -> int:
        # 埃氏筛放在class外部可以通过
        #############################
        p = [-1] * (10**6 + 1)
        for i in range(2, 10**6 + 1):
            if p[i] == -1:
                for j in range(i * i, 10**6 + 1, i):
                    if p[j] == -1:
                        p[j] = i
        #############################
        res = 0
        for i in range(len(nums) - 2, -1, -1):
            x = nums[i]
            while x > nums[i + 1]:
                x = p[x]
                res += 1
            if x == -1:
                return -1
            nums[i] = x
        return res

    # 684. 冗余连接 (Redundant Connection) --dfs
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        def check(i: int) -> bool:
            def dfs(x: int, fa: int) -> bool:
                vis[x] = True
                for y in g[x]:
                    if y != fa:
                        if vis[y]:
                            return False
                        if not dfs(y, x):
                            return False
                return True

            g = [[] for _ in range(len(edges))]
            for id, (u, v) in enumerate(edges):
                if i == id:
                    continue
                g[u - 1].append(v - 1)
                g[v - 1].append(u - 1)
            vis = [False] * len(edges)
            res = dfs(0, -1)
            # 有环
            if not res:
                return False
            # 是森林
            return all(v for v in vis)

        for i in range(len(edges) - 1, -1, -1):
            if check(i):
                return edges[i]

    # 684. 冗余连接 (Redundant Connection) --并查集
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        class union:
            def __init__(self, n: int):
                self.parent = [i for i in range(n)]
                self.rank = [1] * n

            def get_root(self, p: int) -> int:
                if self.parent[p] == p:
                    return p
                self.parent[p] = self.get_root(self.parent[p])
                return self.parent[p]

            def is_conncted(self, p1: int, p2: int) -> bool:
                return self.get_root(p1) == self.get_root(p2)

            def union(self, p1: int, p2: int) -> None:
                r1 = self.get_root(p1)
                r2 = self.get_root(p2)
                if self.rank[r1] < self.rank[r2]:
                    self.parent[r1] = r2
                else:
                    self.parent[r2] = r1
                    if self.rank[r1] == self.rank[r2]:
                        self.rank[r1] += 1

        n = len(edges)
        u = union(n)
        for e in edges:
            if u.is_conncted(e[0] - 1, e[1] - 1):
                return e
            u.union(e[0] - 1, e[1] - 1)

    # 685. 冗余连接 II (Redundant Connection II)
    def findRedundantDirectedConnection(self, edges: List[List[int]]) -> List[int]:
        def check(i: int) -> bool:
            def dfs(x: int, fa: int) -> bool:
                vis[x] = True
                for y in g[x]:
                    if y != fa:
                        if vis[y]:
                            return False
                        if not dfs(y, x):
                            return False
                return True

            g = [[] for _ in range(len(edges))]
            deg = [0] * len(edges)
            for id, (u, v) in enumerate(edges):
                if i == id:
                    continue
                g[u - 1].append(v - 1)
                deg[v - 1] += 1
            vis = [False] * len(edges)
            root = -1
            for x in range(len(edges)):
                if deg[x] == 0:
                    root = x
                    break
            res = dfs(root, -1)
            # 有环
            if not res:
                return False
            # 是森林
            return all(v for v in vis)

        for i in range(len(edges) - 1, -1, -1):
            if check(i):
                return edges[i]

    # 3330. 找到初始输入字符串 I (Find the Original Typed String I)
    def possibleStringCount(self, word: str) -> int:
        return sum(x == y for x, y in pairwise(word)) + 1

    # 3331. 修改后子树的大小 (Find Subtree Sizes After Changes)
    def findSubtreeSizes(self, parent: List[int], s: str) -> List[int]:
        def dfs2(x: int) -> int:
            for y in g2[x]:
                res[x] += dfs2(y)
            res[x] += 1
            return res[x]

        def dfs(x: int, fa: int) -> None:
            id = ord(s[x]) - ord("a")
            if st[id]:
                g2[st[id][-1]].append(x)
            elif fa != -1:
                g2[fa].append(x)
            st[id].append(x)
            for y in g[x]:
                dfs(y, x)
            st[id].pop()

        n = len(s)
        g = [[] for _ in range(n)]
        for i in range(1, n):
            g[parent[i]].append(i)
        g2 = [[] for _ in range(n)]
        st = [[] for _ in range(26)]
        dfs(0, -1)
        res = [0] * n
        dfs2(0)
        return res

    # 3332. 旅客可以得到的最多点数 (Maximum Points Tourist Can Earn)
    def maxScore(
        self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]
    ) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j == k:
                return 0
            return max(
                dfs(i, j + 1) + stayScore[j][i],
                max(dfs(x, j + 1) + travelScore[i][x] for x in range(n)),
            )

        return max(dfs(x, 0) for x in range(n))

    # 638. 大礼包 (Shopping Offers)
    def shoppingOffers(
        self, price: List[int], special: List[List[int]], needs: List[int]
    ) -> int:
        @cache
        def dfs(i: int, j: tuple) -> int:
            if i == n:
                res = 0
                for a, b in zip(price, j):
                    res += a * b
                return res
            res = dfs(i + 1, j)
            added = 0
            k = 1
            while True:
                if any(x * k > y for (x, y) in zip(_list[i], j)):
                    break
                cur = [x for x in j]
                for id, (x, _) in enumerate(zip(_list[i], j)):
                    cur[id] -= x * k
                added += _list[i][-1]
                res = min(res, dfs(i + 1, tuple(cur)) + added)
                k += 1
            return res

        _list = []
        for s in special:
            if all(a <= b for a, b in zip(s, needs)):
                _list.append(s)
        n = len(_list)
        return dfs(0, tuple(needs))

    # 633. 平方数之和 (Sum of Square Numbers)
    def judgeSquareSum(self, c: int) -> bool:
        i = 0
        j = isqrt(c)
        while i <= j:
            s = i * i + j * j
            if s == c:
                return True
            elif s < c:
                i += 1
            else:
                j -= 1
        return False

    # 3340. 检查平衡字符串 (Check Balanced String)
    def isBalanced(self, num: str) -> bool:
        s = [0] * 2
        for i, c in enumerate(num):
            s[i & 1] += int(c)
        return s[0] == s[1]

    # 3341. 到达最后一个房间的最少时间 I (Find Minimum Time to Reach Last Room I)
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        m = len(moveTime)
        n = len(moveTime[0])
        dis = [[inf] * n for _ in range(m)]
        dis[0][0] = 0
        q = []
        heapq.heapify(q)
        q.append((0, 0, 0))
        while q:
            (t, x, y) = heapq.heappop(q)
            if t > dis[x][y]:
                continue
            if x == m - 1 and y == n - 1:
                return t
            for dx, dy in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                nx = x + dx
                ny = y + dy
                if m > nx >= 0 and n > ny >= 0:
                    dt = max(0, moveTime[nx][ny] - t) + 1
                    if t + dt < dis[nx][ny]:
                        dis[nx][ny] = t + dt
                        heapq.heappush(q, (t + dt, nx, ny))
        return -1

    # 3342. 到达最后一个房间的最少时间 II (Find Minimum Time to Reach Last Room II)
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        m = len(moveTime)
        n = len(moveTime[0])
        dis = [[inf] * n for _ in range(m)]
        dis[0][0] = 0
        q = [(0, 0, 0, 0)]
        heapq.heapify(q)
        while q:
            t, d, x, y = heapq.heappop(q)
            if t > dis[x][y]:
                continue
            if x == m - 1 and y == n - 1:
                return t
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n:
                    nt = max(moveTime[nx][ny], t) + 1 + d
                    if nt < dis[nx][ny]:
                        dis[nx][ny] = nt
                        heapq.heappush(q, (nt, d ^ 1, nx, ny))
        return -1

    # 3345. 最小可整除数位乘积 I (Smallest Divisible Digit Product I)
    def smallestNumber(self, n: int, t: int) -> int:
        def check(x: int) -> bool:
            res = 1
            while x:
                res *= x % 10
                x //= 10
            return res % t == 0

        while True:
            if check(n):
                return n
            n += 1

    # 3349. 检测相邻递增子数组 I (Adjacent Increasing Subarrays Detection I)
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        cnt = 1
        a = [False] * n
        for i, v in enumerate(nums):
            if i and v - nums[i - 1] > 0:
                cnt += 1
            if i >= k and nums[i - k + 1] - nums[i - k] > 0:
                cnt -= 1
            if cnt >= k:
                a[i] = True
                if i >= k and a[i - k]:
                    return True
        return False

    # 3350. 检测相邻递增子数组 II (Adjacent Increasing Subarrays Detection II)
    def maxIncreasingSubarrays(self, nums: List[int]) -> int:
        def check(k: int) -> bool:
            n = len(nums)
            cnt = 1
            a = [False] * n
            for i, v in enumerate(nums):
                if i and v - nums[i - 1] > 0:
                    cnt += 1
                if i >= k and nums[i - k + 1] - nums[i - k] > 0:
                    cnt -= 1
                if cnt >= k:
                    a[i] = True
                    if i >= k and a[i - k]:
                        return True
            return False

        n = len(nums)
        left = 1
        right = n // 2
        res = 1
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = mid
                left = mid + 1
            else:
                right = mid - 1
        return res

    # 3351. 好子序列的元素之和 (Sum of Good Subsequences)
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        mx = max(nums) + 2
        cnt = [0] * mx
        f = [0] * mx
        for x in nums:
            c = cnt[x - 1] + cnt[x + 1] + 1
            f[x] = (f[x] + f[x - 1] + f[x + 1] + x * c) % MOD
            cnt[x] = (cnt[x] + c) % MOD
        return sum(f) % MOD

    # 3352. 统计小于 N 的 K 可约简整数 (Count K-Reducible Numbers Less Than N)
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        @cache
        def dfs(i: int, j: int, is_limit: bool) -> int:
            if i == n:
                return 0 if is_limit or j else 1
            res = 0
            up = int(s[i]) if is_limit else 1
            for d in range(up + 1):
                if j - d < 0:
                    continue
                res += dfs(i + 1, j - d, is_limit and d == up)
            return res % MOD

        n = len(s)
        f = [0] * (n + 1)
        res = 0
        MOD = 10**9 + 7
        for i in range(1, n + 1):
            f[i] = f[i.bit_count()] + 1 if i > 1 else 0
            if f[i] < k:
                res += dfs(0, i, True)
                res %= MOD
        dfs.cache_clear()
        return res

    # 825. 适龄的朋友 (Friends Of Appropriate Ages)
    def numFriendRequests(self, ages: List[int]) -> int:
        cnt = [0] * 121
        for a in ages:
            cnt[a] += 1
        res = 0
        for i, x in enumerate(cnt):
            if x == 0:
                continue
            for j, y in enumerate(cnt):
                if y == 0:
                    continue
                if not (j <= 0.5 * i + 7 or j > i or j > 100 and i < 100):
                    res += x * y
                    if i == j:
                        res -= x
        return res

    # 661. 图片平滑器 (Image Smoother)
    def imageSmoother(self, img: List[List[int]]) -> List[List[int]]:
        m = len(img)
        n = len(img[0])
        res = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                s = 0
                cnt = 0
                for x in range(i - 1, i + 2):
                    for y in range(j - 1, j + 2):
                        if 0 <= x < m and 0 <= y < n:
                            s += img[x][y]
                            cnt += 1
                res[i][j] = s // cnt
        return res

    # 3354. 使数组元素等于零 (Make Array Elements Equal to Zero)
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        right = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            right[i] = right[i + 1] + nums[i]
        res = 0
        left = 0
        for i in range(n):
            if nums[i] == 0:
                res += max(0, 2 - abs(left - right[i + 1]))
            left += nums[i]
        return res

    # 3355. 零数组变换 I (Zero Array Transformation I)
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        for l, r in queries:
            diff[l] -= 1
            diff[r + 1] += 1
        return all(x + y <= 0 for x, y in zip(list(accumulate(diff)), nums))

    # 3356. 零数组变换 II (Zero Array Transformation II)
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        def check(t: int) -> bool:
            diff = [0] * (len(nums) + 1)
            for l, r, v in queries[:t]:
                diff[l] -= v
                diff[r + 1] += v
            return all(x + y <= 0 for x, y in zip(list(accumulate(diff)), nums))

        res = -1
        left = 0
        right = len(queries)
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = mid
                right = mid - 1
            else:
                left = mid + 1
        return res

    # 632. 最小区间 (Smallest Range Covering Elements from K Lists)
    def smallestRange(self, nums: List[List[int]]) -> List[int]:
        n = len(nums)
        arr = []
        for i in range(n):
            for x in nums[i]:
                arr.append((x, i))
        arr.sort()
        left = 0
        k = 0
        cnt = [0] * n
        res = [0, 0]
        for right in range(len(arr)):
            cnt[arr[right][1]] += 1
            if cnt[arr[right][1]] == 1:
                k += 1
            while k == n:
                if (
                    res[0] == 0
                    and res[1] == 0
                    or arr[right][0] - arr[left][0] < res[1] - res[0]
                ):
                    res[0] = arr[left][0]
                    res[1] = arr[right][0]
                cnt[arr[left][1]] -= 1
                if cnt[arr[left][1]] == 0:
                    k -= 1
                left += 1
        return res

    # 743. 网络延迟时间 (Network Delay Time)
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        g = [[] for _ in range(n)]
        for u, v, t in times:
            g[u - 1].append((v - 1, t))
        q = []
        q.append((0, k - 1))
        dis = [inf] * n
        dis[k - 1] = 0
        cnt = 0
        heapq.heapify(q)
        while q:
            (t, x) = heapq.heappop(q)
            if t > dis[x]:
                continue
            cnt += 1
            if cnt == n:
                return t
            for y, dt in g[x]:
                if t + dt < dis[y]:
                    dis[y] = t + dt
                    heapq.heappush(q, (t + dt, y))
        return -1

    # 3360. 移除石头游戏 (Stone Removal Game)
    def canAliceWin(self, n: int) -> bool:
        x = 10
        while n >= x:
            n -= x
            x -= 1
        return bool(x & 1)

    # 3361. 两个字符串的切换距离 (Shift Distance Between Two Strings)
    def shiftDistance(
        self, s: str, t: str, nextCost: List[int], previousCost: List[int]
    ) -> int:
        a = list(accumulate(nextCost, initial=0))
        b = list(accumulate(previousCost, initial=0))
        res = 0
        for x, y in zip(s, t):
            if x < y:
                res += min(
                    a[ord(y) - ord("a")] - a[ord(x) - ord("a")],
                    b[-1] - (b[ord(y) - ord("a") + 1] - b[ord(x) - ord("a") + 1]),
                )
            else:
                x, y = y, x
                res += min(
                    a[-1] - (a[ord(y) - ord("a")] - a[ord(x) - ord("a")]),
                    b[ord(y) - ord("a") + 1] - b[ord(x) - ord("a") + 1],
                )
        return res

    # 3363. 最多可收集的水果数目 (Find the Maximum Number of Fruits Collected)
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j == n - 1:
                return 0
            return (
                max(dfs(i0, j + 1) for i0 in range(max(j + 1, i - 1), min(n, i + 2)))
                + fruits[i][j]
            )

        n = len(fruits)
        res = 0
        for i in range(n):
            res += fruits[i][i]
            fruits[i][i] = 0
        res += dfs(n - 1, 0)
        # 转置
        fruits = list(zip(*fruits))
        dfs.cache_clear()
        res += dfs(n - 1, 0)
        return res

    # 3365. 重排子字符串以形成目标字符串 (Rearrange K Substrings to Form Target String)
    def isPossibleToRearrange(self, s: str, t: str, k: int) -> bool:
        d = defaultdict(int)
        l = len(s) // k
        for i in range(l, len(s) + 1, l):
            d[s[i - l : i]] += 1
            d[t[i - l : i]] -= 1
        return all(c == 0 for c in d.values())

    # 3366. 最小数组和 (Minimum Array Sum)
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        @cache
        def dfs(i: int, x: int, y: int) -> int:
            if i == n:
                return 0
            res = dfs(i + 1, x, y) + nums[i]
            if x:
                add = (nums[i] + 1) // 2
                res = min(res, dfs(i + 1, x - 1, y) + add)
                if add >= k and y:
                    res = min(res, dfs(i + 1, x - 1, y - 1) + add - k)
            if y and nums[i] >= k:
                add = nums[i] - k
                res = min(res, dfs(i + 1, x, y - 1) + add)
                if x:
                    res = min(res, dfs(i + 1, x - 1, y - 1) + (add + 1) // 2)
            return res

        n = len(nums)
        return dfs(0, op1, op2)

    # 51. N 皇后 (N-Queens)
    def solveNQueens(self, n: int) -> List[List[str]]:
        # ！！不能加 @cache
        def dfs(i: int, j: int, k: int, l: int):
            if i == n:
                cur = [["."] * n for _ in range(n)]
                for r, c in enumerate(a):
                    cur[r][c] = "Q"
                    cur[r] = "".join(cur[r])
                res.append(cur)
                return
            c = (j | k | l) ^ u
            while c:
                lb = (c & -c).bit_length() - 1
                a.append(lb)
                dfs(
                    i + 1,
                    j | (1 << lb),
                    u & ((k | (1 << lb)) << 1),
                    (l | (1 << lb)) >> 1,
                )
                a.pop()
                c &= c - 1

        u = (1 << n) - 1
        res = []
        a = []
        dfs(0, 0, 0, 0)
        return res

    # 3370. 仅含置位位的最小整数 (Smallest Number With All Set Bits)
    def smallestNumber(self, n: int) -> int:
        return (1 << n.bit_length()) - 1

    # 3371. 识别数组中的最大异常值 (Identify the Largest Outlier in an Array)
    def getLargestOutlier(self, nums: List[int]) -> int:
        s = sum(nums)
        d = defaultdict(int)
        res = -inf
        for x in nums:
            d[x] += 1
        for x in nums:
            d[x] -= 1
            if (s - x) % 2 == 0 and d[(s - x) // 2]:
                res = max(res, x)
            d[x] += 1
        return res

    # 3372. 连接两棵树后最大目标节点数目 I (Maximize the Number of Target Nodes After Connecting Trees I)
    def maxTargetNodes(
        self, edges1: List[List[int]], edges2: List[List[int]], k: int
    ) -> List[int]:
        def dfs(x: int, fa: int, g: List[List[int]], k: int) -> int:
            if k < 0:
                return 0
            res = 0
            for y in g[x]:
                if y != fa and k:
                    res += dfs(y, x, g, k - 1)
            return res + 1

        def generate(edges: List[List[int]]) -> List[List[int]]:
            g = [[] for _ in range(len(edges) + 1)]
            for u, v in edges:
                g[u].append(v)
                g[v].append(u)
            return g

        g2 = generate(edges2)
        mx = max(dfs(i, -1, g2, k - 1) for i in range(len(g2)))
        g1 = generate(edges1)
        return [dfs(i, -1, g1, k) + mx for i in range(len(g1))]

    # 3373. 连接两棵树后最大目标节点数目 II (Maximize the Number of Target Nodes After Connecting Trees II)
    def maxTargetNodes(
        self, edges1: List[List[int]], edges2: List[List[int]]
    ) -> List[int]:
        def tree(x: int, fa: int, d: int) -> None:
            res[x] += cnts1[d]
            for y in g[x]:
                if y != fa:
                    tree(y, x, d ^ 1)

        def cal(edges: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
            def dfs(x: int, fa: int, d: int) -> None:
                cnts[d] += 1
                for y in g[x]:
                    if y != fa:
                        dfs(y, x, d ^ 1)

            n = len(edges) + 1
            g = [[] for _ in range(n)]
            for u, v in edges:
                g[u].append(v)
                g[v].append(u)
            cnts = [0] * 2
            dfs(0, -1, 0)
            return (g, cnts)

        _, cnts = cal(edges2)
        mx = max(cnts)
        g, cnts1 = cal(edges1)
        res = [mx] * (len(edges1) + 1)
        tree(0, -1, 0)
        return res

    # 52. N 皇后 II (N-Queens II)
    def totalNQueens(self, n: int) -> int:
        @cache
        def dfs(i: int, j: int, k: int, l: int) -> int:
            if i == n:
                return 1
            res = 0
            c = (j | k | l) ^ u
            while c:
                lb = (c & -c).bit_length() - 1
                res += dfs(
                    i + 1,
                    j | (1 << lb),
                    u & ((k | (1 << lb)) << 1),
                    (l | (1 << lb)) >> 1,
                )
                c &= c - 1
            return res

        u = (1 << n) - 1
        return dfs(0, 0, 0, 0)

    # 2056. 棋盘上有效移动组合的数目 (Number of Valid Move Combinations On Chessboard) --回溯
    def countCombinations(self, pieces: List[str], positions: List[List[int]]) -> int:
        def generate_all_moves(dirs: List[tuple], x0: int, y0: int) -> List[tuple]:
            SIZE = 8
            cur_moves = [(x0, y0, 0, 0, 0)]
            for dx, dy in dirs:
                x = x0
                y = y0
                step = 0
                while 0 <= x + dx < SIZE and 0 <= y + dy < SIZE:
                    x += dx
                    y += dy
                    step += 1
                    cur_moves.append((x0, y0, dx, dy, step))
            return cur_moves

        def dfs(i: int) -> int:
            if i == n:
                return 1
            res = 0
            for move1 in all_moves[i]:
                if all(check(move1, move2) for move2 in move[:i]):
                    move[i] = move1
                    res += dfs(i + 1)
            return res

        def check(m0: tuple, m1: tuple) -> bool:
            x0, y0, dx0, dy0, step0 = m0
            x1, y1, dx1, dy1, step1 = m1
            for i in range(max(step0, step1)):
                if i < step0:
                    x0 += dx0
                    y0 += dy0
                if i < step1:
                    x1 += dx1
                    y1 += dy1
                if x0 == x1 and y0 == y1:
                    return False
            return True

        n = len(pieces)
        flat = (0, 1), (0, -1), (1, 0), (-1, 0)
        diagnal = (1, -1), (1, 1), (-1, -1), (-1, 1)
        d = {"r": flat, "q": flat + diagnal, "b": diagnal}
        all_moves = [
            generate_all_moves(d[piece[0]], x - 1, y - 1)
            for piece, (x, y) in zip(pieces, positions)
        ]
        move = [None] * n
        return dfs(0)

    # 999. 可以被一步捕获的棋子数 (Available Captures for Rook)
    def numRookCaptures(self, board: List[List[str]]) -> int:
        r = None
        SIZE = 8
        r = None
        for i in range(SIZE):
            for j in range(SIZE):
                if board[i][j] == "R":
                    r = (i, j)
                    break
            if r is not None:
                break
        res = 0
        dirs = (-1, 0), (1, 0), (0, -1), (0, 1)
        for dx, dy in dirs:
            x = r[0] + dx
            y = r[1] + dy
            while SIZE > x >= 0 and SIZE > y >= 0:
                if board[x][y] == "B":
                    break
                elif board[x][y] == "p":
                    res += 1
                    break
                x += dx
                y += dy
        return res

    # 3375. 使数组的值全部为 K 的最少操作次数 (Minimum Operations to Make Array Values Equal to K)
    def minOperations(self, nums: List[int], k: int) -> int:
        s = set()
        for x in nums:
            if k > x:
                return -1
            if x > k:
                s.add(x)
        return len(s)

    # 3377. 使两个整数相等的数位操作 (Digit Operations to Make Two Integers Equal)
    def minOperations(self, n: int, m: int) -> int:
        p = len(str(n))
        SIZE = 10**p
        dis = [inf] * SIZE
        dis[n] = n
        prime = [True] * SIZE
        prime[1] = False
        for i in range(2, SIZE):
            if prime[i]:
                for j in range(i * i, SIZE, i):
                    prime[j] = False
        if prime[n] or prime[m]:
            return -1
        q = []
        q.append((n, n))
        heapq.heapify(q)
        while q:
            s, x = heapq.heappop(q)
            if x == m:
                return s
            if s > dis[x]:
                continue
            _pow = 1
            v = x
            while v:
                v, d = divmod(v, 10)
                if d > 0:
                    y = x - _pow
                    if not prime[y] and s + y < dis[y]:
                        dis[y] = s + y
                        heapq.heappush(q, (s + y, y))
                if d < 9:
                    y = x + _pow
                    if not prime[y] and s + y < dis[y]:
                        dis[y] = s + y
                        heapq.heappush(q, (s + y, y))
                _pow *= 10
        return -1

    # 100489. 破解锁的最少时间 I (Minimum Time to Break Locks I)
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == u:
                return 0
            j = 1 + i.bit_count() * K
            c = i ^ u
            res = inf
            while c:
                lb = (c & -c).bit_length() - 1
                res = min(res, dfs(i | (1 << lb)) + (strength[lb] - 1) // j + 1)
                c &= c - 1
            return res

        n = len(strength)
        u = (1 << n) - 1
        return dfs(0)

    # 3379. 转换数组 (Transformed Array)
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = []
        for i, x in enumerate(nums):
            res.append(nums[(i + x) % n])
        return res

    # 3381. 长度可被 K 整除的子数组的最大元素和 (Maximum Subarray Sum With Length Divisible by K)
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        pre = list(accumulate(nums, initial=0))
        res = -inf
        mod_k = [inf] * k
        for j, v in enumerate(pre):
            i = j % k
            res = max(res, v - mod_k[i])
            mod_k[i] = min(mod_k[i], v)
        return res

    # 1812. 判断国际象棋棋盘中一个格子的颜色 (Determine Color of a Chessboard Square)
    def squareIsWhite(self, coordinates: str) -> bool:
        return (ord(coordinates[0]) - ord("a") - int(coordinates[1])) % 2 == 0

    # 2717. 半有序排列 (Semi-Ordered Permutation)
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        i = -1
        j = -1
        for id, v in enumerate(nums):
            if v == 1:
                i = id
            if v == n:
                j = id
        return n - 1 - j + i - int(i > j)

    # 3264. K 次乘运算后的最终数组 I (Final Array State After K Multiplication Operations I)
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        q = [(x, i) for i, x in enumerate(nums)]
        heapq.heapify(q)
        while k:
            k -= 1
            x, i = heapq.heappop(q)
            heapq.heappush(q, (x * multiplier, i))
        while q:
            x, i = heapq.heappop(q)
            nums[i] = x
        return nums

    # 1338. 数组大小减半 (Reduce Array Size to The Half)
    def minSetSize(self, arr: List[int]) -> int:
        d = Counter(arr)
        _list = [v for v in d.values()]
        _list.sort(reverse=True)
        res = 0
        c = 0
        for v in _list:
            c += v
            res += 1
            if c * 2 >= len(arr):
                return res

    # 3386. 按下时间最长的按钮 (Button with Longest Push Time)
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        res = events[0][0]
        d = events[0][1]
        for i in range(len(events)):
            if events[i][1] - events[i - 1][1] > d:
                d = events[i][1] - events[i - 1][1]
                res = events[i][0]
            elif events[i][1] - events[i - 1][1] == d and events[i][0] < res:
                res = events[i][0]
        return res

    # 2545. 根据第 K 场考试的分数排序 (Sort the Students by Their Kth Score)
    def sortTheStudents(self, score: List[List[int]], k: int) -> List[List[int]]:
        score.sort(key=lambda x: -x[k])
        return score

    # 1387. 将整数按权重排序 (Sort Integers by The Power Value)
    def getKth(self, lo: int, hi: int, k: int) -> int:
        def get_power(x: int) -> int:
            res = 0
            while x != 1:
                if x & 1:
                    x = 3 * x + 1
                else:
                    x >>= 1
                res += 1
            return res

        return sorted(range(lo, hi + 1), key=lambda x: (get_power(x), x))[k - 1]

    # 1387. 将整数按权重排序 (Sort Integers by The Power Value)
    def getKth(self, lo: int, hi: int, k: int) -> int:
        @cache
        def dfs(x: int) -> int:
            if x == 1:
                return 0
            if x % 2:
                return dfs((x * 3 + 1) // 2) + 2
            return dfs(x // 2) + 1

        return sorted(range(lo, hi + 1), key=lambda x: (dfs(x), x))[k - 1]

    # 3392. 统计符合条件长度为 3 的子数组数目 (Count Subarrays of Length Three With a Condition)
    def countSubarrays(self, nums: List[int]) -> int:
        return sum(
            nums[i] == (nums[i - 1] + nums[i + 1]) * 2 for i in range(1, len(nums) - 1)
        )

    # 3393. 统计异或值为给定值的路径数目 (Count Paths With the Given XOR Value)
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        @cache
        def dfs(i: int, j: int, x: int) -> int:
            if i == m or j == n:
                return 0
            x ^= grid[i][j]
            if i == m - 1 and j == n - 1:
                return int(x == k)
            return (dfs(i + 1, j, x) + dfs(i, j + 1, x)) % MOD

        MOD = 10**9 + 7
        m = len(grid)
        n = len(grid[0])
        return dfs(0, 0, 0)

    # 3394. 判断网格图能否被切割成块 (Check if Grid can be Cut into Sections)
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        def check(i0: int, i1: int) -> bool:
            rectangles.sort(key=lambda x: x[i0])
            i = 0
            res = 0
            while i < len(rectangles):
                x = rectangles[i][i1]
                j = i + 1
                while j < len(rectangles) and rectangles[j][i0] < x:
                    x = max(x, rectangles[j][i1])
                    j += 1
                i = j
                res += 1
                if res >= 3:
                    return True
            return False

        return check(0, 2) or check(1, 3)

    # 1705. 吃苹果的最大数目 (Maximum Number of Eaten Apples)
    def eatenApples(self, apples: List[int], days: List[int]) -> int:
        n = len(apples)
        q = []
        heapq.heapify(q)
        res = 0
        i = 0
        while i < n or q:
            if i < n:
                heapq.heappush(q, (i + days[i], apples[i]))
            while q and (q[0][1] == 0 or q[0][0] <= i):
                heapq.heappop(q)
            if q:
                res += 1
                d, c = heapq.heappop(q)
                c -= 1
                heapq.heappush(q, (d, c))
            i += 1
        return res

    # 1366. 通过投票对团队排名 (Rank Teams by Votes)
    def rankTeams(self, votes: List[str]) -> str:
        n = len(votes[0])
        cnt = defaultdict(lambda: [0] * (n))
        for v in votes:
            for i, c in enumerate(v):
                cnt[c][i] -= 1
        return "".join(sorted(cnt, key=lambda ch: (cnt[ch], ch)))

    # 3396. 使数组元素互不相同所需的最少操作次数 (Minimum Number of Operations to Make Elements in Array Distinct)
    def minimumOperations(self, nums: List[int]) -> int:
        d = defaultdict(int)
        for x in nums:
            d[x] += 1
        res = 0
        n = len(nums)
        for i in range(0, n - 2, 3):
            if len(d) == n - i:
                break
            res += 1
            d[nums[i]] -= 1
            if d[nums[i]] == 0:
                del d[nums[i]]
            d[nums[i + 1]] -= 1
            if d[nums[i + 1]] == 0:
                del d[nums[i + 1]]
            d[nums[i + 2]] -= 1
            if d[nums[i + 2]] == 0:
                del d[nums[i + 2]]
        return res + int(any(v > 1 for v in d.values()))

    # 3397. 执行操作后不同元素的最大数量 (Maximum Number of Distinct Elements After Operations)
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        res = 0
        nums.sort()
        pre = -inf
        for x in nums:
            x = min(max(x - k, pre + 1), x + k)
            if x > pre:
                res += 1
                pre = x
        return res

    # 3398. 字符相同的最短子字符串 I (Smallest Substring With Identical Characters I)
    # 3399. 字符相同的最短子字符串 II (Smallest Substring With Identical Characters II)
    def minLength(self, s: str, numOps: int) -> int:
        def check(m: int) -> bool:
            cnt = 0
            if m == 1:
                for i, c in enumerate(s):
                    cnt += ((ord(c) - ord("0")) ^ i) & 1
                cnt = min(cnt, n - cnt)
                return cnt <= numOps
            k = 0
            for i, c in enumerate(s):
                k += 1
                if i == n - 1 or s[i] != s[i + 1]:
                    cnt += k // (m + 1)
                    k = 0
            return cnt <= numOps

        n = len(s)
        left = 1
        right = n
        res = n
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = mid
                right = mid - 1
            else:
                left = mid + 1
        return res

    # 1367. 二叉树中的链表 (Linked List in Binary Tree)
    def isSubPath(self, head: Optional[ListNode], root: Optional[TreeNode]) -> bool:
        def dfs(l_node: Optional[ListNode], t_node: Optional[TreeNode]) -> bool:
            if l_node is None:
                return True
            if t_node is None or l_node.val != t_node.val:
                return False
            return dfs(l_node.next, t_node.left) or dfs(l_node.next, t_node.right)

        def dfs_tree(node: Optional[TreeNode]) -> bool:
            if node is None:
                return False
            if head.val == node.val and dfs(head, node):
                return True
            return dfs_tree(node.left) or dfs_tree(node.right)

        return dfs_tree(root)

    # 3402. 使每一列严格递增的最少操作次数 (Minimum Operations to Make Columns Strictly Increasing)
    def minimumOperations(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        res = 0
        for j in range(n):
            for i in range(1, m):
                add = max(0, grid[i - 1][j] + 1 - grid[i][j])
                res += add
                grid[i][j] += add
        return res

    # 3403. 从盒子中找出字典序最大的字符串 I (Find the Lexicographically Largest String From the Box I)
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        if numFriends == 1:
            return word
        res = ""
        for i in range(n):
            s = word[i : min(n, n - numFriends + i + 1)]
            if s > res:
                res = s
        return res

    # 729. 我的日程安排表 I (My Calendar I)
    class MyCalendar:

        def __init__(self):
            self.calendar = []

        def book(self, startTime: int, endTime: int) -> bool:
            for s, e in self.calendar:
                if startTime >= e or endTime <= s:
                    continue
                return False
            self.calendar.append((startTime, endTime))
            return True

    # 731. 我的日程安排表 II (My Calendar II)
    class MyCalendarTwo:

        def __init__(self):
            self.d = SortedList()

        def book(self, startTime: int, endTime: int) -> bool:
            self.d.add((startTime, 1))
            self.d.add((endTime, -1))
            cnt = 0
            for _, v in self.d:
                cnt += v
                if cnt > 2:
                    self.d.remove((startTime, 1))
                    self.d.remove((endTime, -1))
                    return False
            return True

    # 732. 我的日程安排表 III (My Calendar III)
    class MyCalendarThree:

        def __init__(self):
            self.d = SortedList()

        def book(self, startTime: int, endTime: int) -> int:
            self.d.add((startTime, 1))
            self.d.add((endTime, -1))
            cur = 0
            res = 0
            for _, v in self.d:
                cur += v
                res = max(res, cur)
            return res

    # 2241. 设计一个 ATM 机器 (Design an ATM Machine)
    class ATM:

        def __init__(self):
            self.d = (20, 50, 100, 200, 500)
            self.cnt = [0] * 5

        def deposit(self, banknotesCount: List[int]) -> None:
            for i in range(5):
                self.cnt[i] += banknotesCount[i]

        def withdraw(self, amount: int) -> List[int]:
            res = [0] * 5
            for i in range(4, -1, -1):
                cur = min(self.cnt[i], amount // self.d[i])
                res[i] += cur
                amount -= cur * self.d[i]
            if amount:
                return [-1]
            for i in range(5):
                self.cnt[i] -= res[i]
            return res

    # 2274. 不含特殊楼层的最大连续楼层数 (Maximum Consecutive Floors Without Special Floors)
    def maxConsecutive(self, bottom: int, top: int, special: List[int]) -> int:
        special.sort()
        res = max(special[0] - bottom, top - special[-1])
        for i in range(1, len(special)):
            res = max(res, special[i] - special[i - 1] - 1)
        return res

    # 3411. 最长乘积等价子数组
    def maxLength(self, nums: List[int]) -> int:
        res = 0
        for i in range(len(nums)):
            m, l, g = 1, 1, 0
            for j in range(i, len(nums)):
                x = nums[j]
                m *= x
                l = lcm(l, x)
                g = gcd(g, x)
                if m == l * g:
                    res = max(res, j - i + 1)

    # 2264. 字符串中最大的 3 位相同数字 (Largest 3-Same-Digit Number in String)
    def largestGoodInteger(self, num: str) -> str:
        res = ""
        for i in range(1, len(num) - 1):
            if num[i] == num[i - 1] == num[i + 1] and res < num[i - 1 : i + 2]:
                res = num[i - 1 : i + 2]
        return res

    # 2275. 按位与结果大于零的最长组合 (Largest Combination With Bitwise AND Greater Than Zero)
    def largestCombination(self, candidates: List[int]) -> int:
        res = 0
        MAX_BIT = 24
        for i in range(MAX_BIT):
            res = max(res, sum((x >> i) & 1 for x in candidates))
        return res

    # 3417. 跳过交替单元格的之字形遍历 (Zigzag Grid Traversal With Skip)
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        m = len(grid)
        n = len(grid[0])
        res = []
        d = 1
        for i in range(m):
            if not (i & 1):
                for j in range(n):
                    if d:
                        res.append(grid[i][j])
                    d ^= 1
            else:
                for j in range(n - 1, -1, -1):
                    if d:
                        res.append(grid[i][j])
                    d ^= 1
        return res

    # 3418. 机器人可以获得的最大金币数 (Maximum Amount of Money Robot Can Earn)
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m, n = len(coins), len(coins[0])

        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i == m or j == n:
                return -inf
            if i == m - 1 and j == n - 1:
                return max(coins[i][j], -inf if k == 2 else 0)
            res = max(dfs(i + 1, j, k), dfs(i, j + 1, k)) + coins[i][j]
            if k < 2:
                res = max(res, dfs(i + 1, j, k + 1), dfs(i, j + 1, k + 1))
            return res

        return dfs(0, 0, 0)

    # 3419. 图的最大边权的最小值 (Minimize the Maximum Edge Weight of Graph)
    def minMaxWeight(self, n: int, edges: List[List[int]], _: int) -> int:
        def check(upper: int) -> bool:
            def dfs(x: int) -> int:
                cnt = 1
                vis[x] = True
                for y, w in g[x]:
                    if w <= upper and not vis[y]:
                        cnt += dfs(y)
                return cnt

            g = [[] for _ in range(n)]
            for u, v, w in edges:
                g[v].append((u, w))
            vis = [False] * n
            return dfs(0) == n

        left = 1
        right = max(x for _, _, x in edges)
        res = -1
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = mid
                right = mid - 1
            else:
                left = mid + 1
        return res

    # 2239. 找到最接近 0 的数字 (Find Closest Number to Zero)
    def findClosestNumber(self, nums: List[int]) -> int:
        res = -inf
        d = inf
        for x in nums:
            if abs(x) < d or abs(x) == d and x > res:
                d = abs(x)
                res = x
        return res

    # 2239. 找到最接近 0 的数字 (Find Closest Number to Zero)
    def findClosestNumber(self, nums: List[int]) -> int:
        nums.sort(key=lambda x: ([abs(x), -x], x))
        return nums[0]

    # 3423. 循环数组中相邻元素的最大差值 (Maximum Difference Between Adjacent Elements in a Circular Array)
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        return max(max(abs(x - y) for x, y in pairwise(nums)), abs(nums[0] - nums[-1]))

    # 3424. 将数组变相同的最小代价 (Minimum Cost to Make Arrays Identical)
    def minCost(self, arr: List[int], brr: List[int], k: int) -> int:
        res = sum(abs(x - y) for x, y in zip(arr, brr))
        arr.sort()
        brr.sort()
        res2 = sum(abs(x - y) for x, y in zip(arr, brr))
        return min(res, res2 + k)

    # 3427. 变长子数组求和 (Sum of Variable Length Subarrays)
    def subarraySum(self, nums: List[int]) -> int:
        pre = list(accumulate(nums, initial=0))
        res = 0
        for i, v in enumerate(nums):
            start = max(0, i - v)
            res += pre[i + 1] - pre[start]
        return res

    # 3429. 粉刷房子 IV (Paint House IV)
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i == n // 2:
                return 0
            res = inf
            for x in range(3):
                for y in range(3):
                    if x == y or x == j or y == k:
                        continue
                    res = min(res, dfs(i + 1, x, y) + cost[i][x] + cost[n - i - 1][y])
            return res

        return dfs(0, -1, -1)

    # 3428. 最多 K 个元素的子序列的最值之和 (Maximum and Minimum Sums of at Most Size K Subsequences)
    # --需要用递推算逆元 以下代码无法通过最后一个测试用例
    def minMaxSums(self, nums: List[int], k: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j == 0 or i == j:
                return 1
            j = min(j, i - j)
            return dfs(i - 1, j - 1) + dfs(i - 1, j)

        n = len(nums)
        nums.sort()
        MOD = 10**9 + 7
        res = 0
        for i, v in enumerate(nums):
            for j in range(min(k, i + 1)):
                res += (dfs(i, j) * (v + nums[n - i - 1])) % MOD
                res %= MOD
        dfs.cache_clear()
        return res

    # 3425. 最长特殊路径 (Longest Special Path)
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        def dfs(x: int, fa: int, top_depth: int) -> int:
            color = nums[x]
            old_depth = last_depth.get(color, 0)
            top_depth = max(top_depth, old_depth)
            s = pre_sum[-1] - pre_sum[top_depth]
            cnt = len(pre_sum) - top_depth
            nonlocal max_s, min_cnt
            if s > max_s or s == max_s and cnt < min_cnt:
                max_s = s
                min_cnt = cnt
            last_depth[color] = len(pre_sum)
            for y, w in g[x]:
                if y != fa:
                    pre_sum.append(pre_sum[-1] + w)
                    dfs(y, x, top_depth)
                    pre_sum.pop()
            last_depth[color] = old_depth

        n = len(nums)
        max_s = -1
        min_cnt = 0
        g = [[] for _ in range(n)]
        for u, v, w in edges:
            g[u].append((v, w))
            g[v].append((u, w))
        last_depth = defaultdict()
        pre_sum = [0]
        dfs(0, -1, 0)
        return (max_s, min_cnt)

    # 1561. 你可以获得的最大硬币数目 (Maximum Number of Coins You Can Get)
    def maxCoins(self, piles: List[int]) -> int:
        piles.sort()
        return sum(piles[len(piles) // 3 :: 2])

    # 45. 跳跃游戏 II (Jump Game II)
    def jump(self, nums: List[int]) -> int:
        res = 0
        right = 0
        cur = 0
        for i, v in enumerate(nums):
            if cur >= len(nums) - 1:
                return res
            right = max(right, i + v)
            if i == cur:
                cur = right
                res += 1
        return res

    # 119. 杨辉三角 II (Pascal's Triangle II)
    def getRow(self, rowIndex: int) -> List[int]:
        res = [1]
        for _ in range(rowIndex):
            pre = res.copy()
            res.append(1)
            for j in range(1, len(res) - 1):
                res[j] = pre[j - 1] + pre[j]
        return res

    # 219. 存在重复元素 II (Contains Duplicate II)
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        d = defaultdict(int)
        for i, v in enumerate(nums):
            if v in d and i - d[v] <= k:
                return True
            d[v] = i
        return False

    # 350. 两个数组的交集 II (Intersection of Two Arrays II)
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()
        nums2.sort()
        res = []
        i = j = 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] == nums2[j]:
                res.append(nums1[i])
                i += 1
                j += 1
            elif nums1[i] < nums2[j]:
                i += 1
            else:
                j += 1
        return res

    # 541. 反转字符串 II (Reverse String II)
    def reverseStr(self, s: str, k: int) -> str:
        arr = [x for x in s]
        n = len(arr)
        for i in range(0, n, 2 * k):
            arr[i : min(i + k, n)] = arr[i : min(i + k, n)][::-1]
        return "".join(arr)

    # 598. 区间加法 II (Range Addition II)
    def maxCount(self, m: int, n: int, ops: List[List[int]]) -> int:
        for x, y in ops:
            m = min(m, x)
            n = min(n, y)
        return m * n

    # 680. 验证回文串 II (Valid Palindrome II)
    def validPalindrome(self, s: str) -> bool:
        def check(i: int, j: int) -> bool:
            while i < j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1
            return True

        i = 0
        j = len(s) - 1
        while i < j:
            if s[i] != s[j]:
                break
            i += 1
            j -= 1
        if i >= j:
            return True
        return check(i + 1, j) or check(i, j - 1)

    # 922. 按奇偶排序数组 II (Sort Array By Parity II)
    def sortArrayByParityII(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [0] * n
        e_id = 0
        o_id = 1
        for v in nums:
            if v & 1:
                res[o_id] = v
                o_id += 2
            else:
                res[e_id] = v
                e_id += 2
        return res

    # 59. 螺旋矩阵 II (Spiral Matrix II)
    def generateMatrix(self, n: int) -> List[List[int]]:
        res = [[0] * n for _ in range(n)]
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        d = 0
        x = y = 0
        for i in range(n * n):
            res[x][y] = i + 1
            if (
                x + dirs[d][0] == n
                or x + dirs[d][0] < 0
                or y + dirs[d][1] == n
                or y + dirs[d][1] < 0
                or res[x + dirs[d][0]][y + dirs[d][1]]
            ):
                d = (d + 1) % 4
            x += dirs[d][0]
            y += dirs[d][1]
        return res

    # 80. 删除有序数组中的重复项 II (Remove Duplicates from Sorted Array II)
    def removeDuplicates(self, nums: List[int]) -> int:
        n = len(nums)
        i = 0
        while i < n:
            j = i
            while j < n and nums[j] == nums[i]:
                if j - i >= 2:
                    nums[j] = inf
                j += 1
            i = j
        i = j = 0
        while i < n:
            if nums[i] != inf:
                nums[j] = nums[i]
                j += 1
            i += 1
        return j

    # 1760. 袋子里最少数目的球 (Minimum Limit of Balls in a Bag)
    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
        def check(target: int) -> bool:
            cnt = 0
            for x in nums:
                cnt += (x - 1) // target
                if cnt > maxOperations:
                    return False
            return True

        left = 1
        right = max(nums)
        res = 1
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = mid
                right = mid - 1
            else:
                left = mid + 1
        return res

    # 1552. 两球之间的磁力 (Magnetic Force Between Two Balls)
    def maxDistance(self, position: List[int], m: int) -> int:
        def check(target: int) -> bool:
            pre = -inf
            cnt = 0
            for x in position:
                if x - pre >= target:
                    cnt += 1
                    pre = x
            return cnt >= m

        position.sort()
        left = min(y - x for x, y in pairwise(position))
        right = position[-1] - position[0]
        res = 1
        while left <= right:
            mid = left + ((right - left) >> 1)
            if check(mid):
                res = mid
                left = mid + 1
            else:
                right = mid - 1
        return res

    # 1706. 球会落何处 (Where Will the Ball Fall)
    def findBall(self, grid: List[List[int]]) -> List[int]:
        m = len(grid)
        n = len(grid[0])
        res = [-1] * n
        for j in range(n):
            cur_j = j
            for i in range(m):
                if (
                    cur_j == 0
                    and grid[i][cur_j] == -1
                    or cur_j == n - 1
                    and grid[i][cur_j] == 1
                    or grid[i][cur_j] == 1
                    and grid[i][cur_j + 1] == -1
                    or grid[i][cur_j] == -1
                    and grid[i][cur_j - 1] == 1
                ):
                    cur_j = -1
                    break
                cur_j += grid[i][cur_j]
            res[j] = cur_j
        return res

    # 1299. 将每个元素替换为右侧最大元素 (Replace Elements with Greatest Element on Right Side)
    def replaceElements(self, arr: List[int]) -> List[int]:
        mx = -1
        for i in range(len(arr) - 1, -1, -1):
            cur_mx = mx
            mx = max(mx, arr[i])
            arr[i] = cur_mx
        return arr

    # 1287. 有序数组中出现次数超过25%的元素 (Element Appearing More Than 25% In Sorted Array)
    def findSpecialInteger(self, arr: List[int]) -> int:
        pre = -1
        cnt = 0
        for x in arr:
            if x != pre:
                cnt = 1
            else:
                cnt += 1
            if cnt > len(arr) // 4:
                return x
            pre = x

    # 2080. 区间内查询数字的频率 (Range Frequency Queries)
    class RangeFreqQuery:

        def __init__(self, arr: List[int]):
            self.d = defaultdict(list)
            for i, x in enumerate(arr):
                self.d[x].append(i)

        def query(self, left: int, right: int, value: int) -> int:
            def check(left: int, right: int) -> int:
                if not list or left > list[-1] or right < list[0]:
                    return 0
                l = 0
                r = len(list) - 1
                res_l = -1
                while l <= r:
                    mid = l + ((r - l) >> 1)
                    if list[mid] >= left:
                        res_l = mid
                        r = mid - 1
                    else:
                        l = mid + 1
                l = 0
                r = len(list) - 1
                res_r = -1
                while l <= r:
                    mid = l + ((r - l) >> 1)
                    if list[mid] <= right:
                        res_r = mid
                        l = mid + 1
                    else:
                        r = mid - 1
                return res_r - res_l + 1

            list = self.d[value]
            return check(left, right)

    # 624. 数组列表中的最大距离 (Maximum Distance in Arrays)
    def maxDistance(self, arrays: List[List[int]]) -> int:
        res = 0
        _min = inf
        _max = -inf
        for arr in arrays:
            first = arr[0]
            last = arr[-1]
            if _min != inf:
                res = max(res, abs(_min - last), abs(_max - first))
            _min = min(_min, first)
            _max = max(_max, last)
        return res

    # 2595. 奇偶位数 (Number of Even and Odd Bits)
    def evenOddBit(self, n: int) -> List[int]:
        MASK = 0b1010101010
        return [((MASK >> 1) & n).bit_count(), ((MASK & n).bit_count())]

    # 2506. 统计相似字符串对的数目 (Count Pairs Of Similar Strings)
    def similarPairs(self, words: List[str]) -> int:
        d = defaultdict(int)
        res = 0
        for word in words:
            bits = 0
            for c in word:
                bits |= 1 << (ord(c) - ord("a"))
            res += d[bits]
            d[bits] += 1
        return res

    # 1656. 设计有序流 (Design an Ordered Stream)
    class OrderedStream:

        def __init__(self, n: int):
            self.d = [None] * (n + 1)
            self.ptr = 1

        def insert(self, idKey: int, value: str) -> List[str]:
            self.d[idKey] = value
            res = []
            while self.ptr < len(self.d) and self.d[self.ptr]:
                res.append(self.d[self.ptr])
                self.ptr += 1
            return res

    # 3461. 判断操作后字符串中的数字是否相等 I (Check If Digits Are Equal in String After Operations I)
    def hasSameDigits(self, s: str) -> bool:
        while len(s) != 2:
            arr = [x for x in s]
            for i in range(1, len(arr)):
                arr[i - 1] = str((int(arr[i - 1]) + int(arr[i])) % 10)
            s = "".join(arr[:-1])
        return s[0] == s[1]

    # 3462. 提取至多 K 个元素的最大总和 (Maximum Sum With at Most K Elements)
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        res = 0
        m = len(grid)
        n = len(grid[0])
        arr = []
        for i in range(m):
            grid[i].sort(reverse=True)
            for j in range(0, min(limits[i], n)):
                arr.append(grid[i][j])
        arr.sort(reverse=True)
        for i in range(min(k, len(arr))):
            res += arr[i]
        return res

    # 2502. 设计内存分配器 (Design Memory Allocator)
    class Allocator:

        def __init__(self, n: int):
            self.s = SortedDict()
            self.s[n] = 0
            self.mid_pid = defaultdict(list)

        def allocate(self, size: int, mID: int) -> int:
            pre = 0
            for x, y in self.s.items():
                if x - pre >= size:
                    self.s[pre] = size
                    self.mid_pid[mID].append(pre)
                    return pre
                pre = x + y
            return -1

        def freeMemory(self, mID: int) -> int:
            s = sum(self.s.pop(x) for x in self.mid_pid[mID])
            self.mid_pid.pop(mID)
            return s

    # 1472. 设计浏览器历史记录 (Design Browser History)
    class BrowserHistory:

        def __init__(self, homepage: str):
            self.pages = [homepage]
            self.ptr = 0

        def visit(self, url: str) -> None:
            self.pages = self.pages[: self.ptr + 1]
            self.pages.append(url)
            self.ptr += 1

        def back(self, steps: int) -> str:
            self.ptr = max(0, self.ptr - steps)
            return self.pages[self.ptr]

        def forward(self, steps: int) -> str:
            self.ptr = min(len(self.pages) - 1, self.ptr + steps)
            return self.pages[self.ptr]

    # 2353. 设计食物评分系统 (Design a Food Rating System)
    class FoodRatings:

        def __init__(self, foods: List[str], cuisines: List[str], ratings: List[int]):
            self.food_to_rating_cuisine = defaultdict(tuple)
            self.cuisine_to_food_score = defaultdict(SortedList)
            self.n = len(foods)
            for food, cuisine, rating in zip(foods, cuisines, ratings):
                self.food_to_rating_cuisine[food] = (rating, cuisine)
                self.cuisine_to_food_score[cuisine].add((self.n - rating, food))

        def changeRating(self, food: str, newRating: int) -> None:
            (old_rating, cuisine) = self.food_to_rating_cuisine[food]
            self.food_to_rating_cuisine[food] = (newRating, cuisine)
            self.cuisine_to_food_score[cuisine].remove((self.n - old_rating, food))
            self.cuisine_to_food_score[cuisine].add((self.n - newRating, food))

        def highestRated(self, cuisine: str) -> str:
            return self.cuisine_to_food_score[cuisine][0][1]

    # 31. 分割回文串 (Palindrome Partitioning)
    def partition(self, s: str) -> List[List[str]]:
        def dfs(i: int) -> None:
            if i == n:
                res.append(path.copy())
                return
            for j in range(i, n):
                if is_valid[i][j]:
                    path.append(s[i : j + 1])
                    dfs(j + 1)
                    path.pop()

        n = len(s)
        is_valid = [[False] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if (
                    j == i
                    or j - i == 1
                    and s[i] == s[j]
                    or j - i > 1
                    and s[i] == s[j]
                    and is_valid[i + 1][j - 1]
                ):
                    is_valid[i][j] = True
        res = []
        path = []
        dfs(0)
        return res

    # 1328. 破坏回文串 (Break a Palindrome)
    def breakPalindrome(self, palindrome: str) -> str:
        arr = [x for x in palindrome]
        left = 0
        right = len(arr) - 1
        while left < right:
            if arr[left] != "a":
                arr[left] = "a"
                return "".join(arr)
            left += 1
            right -= 1
        if len(arr) == 1:
            return ""
        arr[-1] = "b"
        return "".join(arr)

    # 2588. 统计美丽子数组数目 (Count the Number of Beautiful Subarrays)
    def beautifulSubarrays(self, nums: List[int]) -> int:
        cnts = defaultdict(int)
        cnts[0] = 1
        res = 0
        xor = 0
        for x in nums:
            xor ^= x
            res += cnts[xor]
            cnts[xor] += 1
        return res

    # 2070. 每一个查询的最大美丽值 (Most Beautiful Item for Each Query)
    def maximumBeauty(self, items: List[List[int]], queries: List[int]) -> List[int]:
        items.sort()
        idx = sorted(range(len(queries)), key=lambda i: queries[i])
        res = [0] * len(queries)
        mx = 0
        i = 0
        for id in idx:
            while i < len(items) and items[i][0] <= queries[id]:
                mx = max(mx, items[i][1])
                i += 1
            res[id] = mx
        return res

    # 3478. 选出和最大的 K 个元素 (Choose K Elements With Maximum Sum)
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        a = sorted((x, y, i) for i, (x, y) in enumerate(zip(nums1, nums2)))

        n = len(a)
        ans = [0] * n
        h = []
        # 分组循环模板
        s = i = 0
        while i < n:
            start = i
            x = a[start][0]
            # 找到所有相同的 nums1[i]，这些数的答案都是一样的
            while i < n and a[i][0] == x:
                ans[a[i][2]] = s
                i += 1
            # 把这些相同的 nums1[i] 对应的 nums2[i] 入堆
            for j in range(start, i):
                y = a[j][1]
                s += y
                heapq.heappush(h, y)
                if len(h) > k:
                    s -= heapq.heappop(h)
        return ans

    # 3471. 找出最大的几近缺失整数 (Find the Largest Almost Missing Integer)
    def f(self, nums: List[int], x: int) -> int:
        return -1 if x in nums else x

    def largestInteger(self, nums: List[int], k: int) -> int:
        if k == len(nums):
            return max(nums)
        if k == 1:
            ans = -1
            for x, c in Counter(nums).items():
                if c == 1:
                    ans = max(ans, x)
            return ans
        # nums[0] 不能出现在其他地方，nums[-1] 同理
        return max(self.f(nums[1:], nums[0]), self.f(nums[:-1], nums[-1]))

    # 3472. 至多 K 次操作后的最长回文子序列 (Longest Palindromic Subsequence After at Most K Operations)
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i == j:
                return 1
            if i > j:
                return 0
            res = max(dfs(i + 1, j, k), dfs(i, j - 1, k))
            a = ord(s[i]) - ord("a")
            b = ord(s[j]) - ord("a")
            c = min(abs(a - b), 26 - abs(a - b))
            if c <= k:
                res = max(res, dfs(i + 1, j - 1, k - c) + 2)
            return res

        n = len(s)
        res = dfs(0, n - 1, k)
        dfs.cache_clear()
        return res

    # 3467. 将数组按照奇偶性转化 (Transform Array by Parity)
    def transformArray(self, nums: List[int]) -> List[int]:
        for i in range(len(nums)):
            nums[i] &= 1
        return sorted(nums)

    # 3468. 可行数组的数目 (Find the Number of Copy Arrays)
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        _min = -inf
        _max = inf
        for o, (l, r) in zip(original, bounds):
            if o + _max < l or o + _min > r:
                return 0
            _max = min(_max, r - o)
            _min = max(_min, l - o)
        return _max - _min + 1

    # 3469. 移除所有数组元素的最小代价 (Find Minimum Cost to Remove Array Elements)
    def minCost(self, nums: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j >= n - 1:
                res = nums[i]
                if j < n:
                    res = max(res, nums[j])
                return res
            return min(
                dfs(j + 1, j + 2) + max(nums[i], nums[j]),
                dfs(j, j + 2) + max(nums[i], nums[j + 1]),
                dfs(i, j + 2) + max(nums[j], nums[j + 1]),
            )

        n = len(nums)
        if n < 3:
            return max(nums)
        return min(
            dfs(2, 3) + max(nums[0], nums[1]),
            dfs(0, 3) + max(nums[1], nums[2]),
            dfs(1, 3) + max(nums[0], nums[2]),
        )

    # 3456. 找出长度为 K 的特殊子字符串 (Find Special Substring of Length K)
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        cnt = [0] * 26
        n = len(s)
        for i, c in enumerate(s):
            d = ord(c) - ord("a")
            cnt[d] += 1
            if i >= k:
                cnt[ord(s[i - k]) - ord("a")] -= 1
            if (
                i >= k - 1
                and cnt[d] == k
                and (i == k - 1 or s[i - k] != s[i])
                and (i == n - 1 or s[i] != s[i + 1])
            ):
                return True
        return False

    # 3457. 吃披萨 (Eat Pizzas!)
    def maxWeight(self, pizzas: List[int]) -> int:
        n = len(pizzas)
        pizzas.sort()
        res = 0
        d = n // 4
        odd = (d + 1) // 2
        for i in range(odd):
            res += pizzas[n - 1 - i]
        for i in range(d // 2):
            res += pizzas[n - 2 - odd - i * 2]
        return res

    # 2012. 数组美丽值求和 (Sum of Beauty in the Array)
    def sumOfBeauties(self, nums: List[int]) -> int:
        n = len(nums)
        valid_right = [False] * n
        mx = nums[-1]
        for i in range(n - 1, -1, -1):
            if nums[i] < mx:
                valid_right[i] = True
            mx = min(mx, nums[i])
        res = 0
        mx = nums[0]
        for i in range(1, n - 1):
            if nums[i] > mx and valid_right[i]:
                res += 2
            elif nums[i - 1] < nums[i] < nums[i + 1]:
                res += 1
            mx = max(mx, nums[i])
        return res

    # 3432. 统计元素和差值为偶数的分区方案 (Count Partitions with Even Sum Difference)
    def countPartitions(self, nums: List[int]) -> int:
        n = len(nums)
        s = sum(nums)
        res = 0
        pre = 0
        for i in range(0, n - 1):
            pre += nums[i]
            if (pre - (s - pre)) % 2 == 0:
                res += 1
        return res

    # 3483. 不同三位偶数的数目 (Unique 3-Digit Even Numbers)
    def totalNumbers(self, digits: List[int]) -> int:
        def dfs() -> None:
            nonlocal cur, vis
            if vis.bit_count() == 3:
                if 100 <= cur <= 999 and cur % 2 == 0:
                    s.add(cur)
                return
            for i in range(n):
                if (vis >> i) & 1 == 0:
                    cur = cur * 10 + digits[i]
                    vis ^= 1 << i
                    dfs()
                    vis ^= 1 << i
                    cur //= 10

        n = len(digits)
        vis = 0
        s = set()
        cur = 0
        dfs()
        return len(s)

    # 3484. 设计电子表格 (Design Spreadsheet)
    class Spreadsheet:

        def __init__(self, rows: int):
            self.sheets = [[0] * 26 for _ in range(rows + 1)]

        def setCell(self, cell: str, value: int) -> None:
            (r, c) = self.check(cell)
            self.sheets[r][c] = value

        def resetCell(self, cell: str) -> None:
            (r, c) = self.check(cell)
            self.sheets[r][c] = 0

        def getValue(self, formula: str) -> int:
            plus_pos = formula.find("+")
            x = formula[1:plus_pos]
            a = 0
            if 9 >= ord(x[0]) - ord("0") >= 0:
                a = int(x)
            else:
                (r, c) = self.check(x)
                a = self.sheets[r][c]

            y = formula[plus_pos + 1 :]
            b = 0
            if 9 >= ord(y[0]) - ord("0") >= 0:
                b = int(y)
            else:
                (r, c) = self.check(y)
                b = self.sheets[r][c]
            return a + b

        def check(self, cell: str) -> tuple:
            c = ord(cell[0]) - ord("A")
            r = int(cell[1:])
            return (r, c)

    # 1963. 使字符串平衡的最小交换次数 (Minimum Number of Swaps to Make the String Balanced)
    def minSwaps(self, s: str) -> int:
        s = list(s)
        ans = c = 0
        j = len(s) - 1
        for b in s:
            if b == "[":
                c += 1
            elif c > 0:
                c -= 1
            else:  # c == 0
                # 找最右边的左括号交换
                while s[j] == "]":
                    j -= 1
                s[j] = "]"  # s[i] = '[' 可以省略
                ans += 1
                c += 1  # s[i] 变成左括号，c 加一
        return ans

    # 2614. 对角线上的质数 (Prime In Diagonal)
    def diagonalPrime(self, nums: List[List[int]]) -> int:
        def check(x: int) -> bool:
            i = 2
            while i * i <= x:
                if x % i == 0:
                    return False
                i += 1
            return x != 1

        n = len(nums)
        res = 0
        for i in range(n):
            if check(nums[i][i]):
                res = max(res, nums[i][i])
            if check(nums[i][n - i - 1]):
                res = max(res, nums[i][n - i - 1])
        return res

    # 2610. 转换二维数组 (Convert an Array Into a 2D Array With Conditions)
    def findMatrix(self, nums: List[int]) -> List[List[int]]:
        d = defaultdict(int)
        for x in nums:
            d[x] += 1
        n = max(d.values())
        res = [[] for _ in range(n)]
        for k in d.keys():
            v = d[k]
            while v:
                res[v - 1].append(k)
                v -= 1
        return res

    # 3487. 删除后的最大子数组元素和 (Maximum Unique Subarray Sum After Deletion)
    def maxSum(self, nums: List[int]) -> int:
        n = len(nums)
        s = set()
        mx = -inf
        for x in nums:
            if x > 0:
                s.add(x)
            else:
                n -= 1
                mx = max(mx, x)
        return sum(s) if n else mx

    # 2680. 最大或值 (Maximum OR)
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        res = 0
        suf = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suf[i] = suf[i + 1] | nums[i]
        pre = 0
        for i, x in enumerate(nums):
            res = max(res, pre | suf[i + 1] | x << k)
            pre |= x
        return res

    # 2643. 一最多的行 (Row With Maximum Ones)
    def rowAndMaximumOnes(self, mat: List[List[int]]) -> List[int]:
        i = -1
        mx = -1
        for r, m in enumerate(mat):
            s = sum(m)
            if s > mx:
                i = r
                mx = s
        return [i, mx]

    # 3488. 距离最小相等元素查询 (Closest Equal Element Queries)
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        def check(_list: List[int], q: int) -> int:
            left = 0
            right = len(_list) - 1
            while left <= right:
                mid = left + ((right - left) >> 1)
                if _list[mid] == q:
                    return mid
                if _list[mid] < q:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1

        res = []
        d = defaultdict(list)
        for i, x in enumerate(nums):
            d[x].append(i)

        for q in queries:
            x = nums[q]
            _list = d[x]
            if len(_list) == 1:
                res.append(-1)
                continue
            p = check(_list, q)
            a = abs(_list[p - 1] - _list[p])
            a = min(a, len(nums) - a)
            b = abs(_list[(p + 1) % len(_list)] - _list[p])
            b = min(b, len(nums) - b)
            res.append(min(a, b))
        return res

    # 3489. 零数组变换 IV (Zero Array Transformation IV)
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if j == 0:
                return i
            if i == len(queries):
                return inf
            # 不选
            res = dfs(i + 1, j)
            # 选
            if queries[i][0] <= pos <= queries[i][1] and j >= queries[i][2]:
                res = min(res, dfs(i + 1, j - queries[i][2]))
            return res

        res = -1
        for i in range(len(nums)):
            pos = i
            cur = dfs(0, nums[i])
            dfs.cache_clear()
            res = max(res, cur)
            if res == inf:
                return -1
        return res

    # 2116. 判断一个括号字符串是否有效 (Check if a Parentheses String Can Be Valid)
    def canBeValid(self, s: str, locked: str) -> bool:
        def check(s: str, locked: str) -> bool:
            cnt = 0
            for c, l in zip(s, locked):
                if c == "(" or l == "0":
                    cnt += 1
                elif cnt:
                    cnt -= 1
                else:
                    return False
            return True

        def check2(s: str, locked: str) -> bool:
            cnt = 0
            for i in range(len(s) - 1, -1, -1):
                if s[i] == ")" or locked[i] == "0":
                    cnt += 1
                elif cnt:
                    cnt -= 1
                else:
                    return False
            return True

        n = len(s)
        if n & 1:
            return False
        return check(s, locked) and check2(s, locked)

    # 3492. 船上可以装载的最大集装箱数量 (Maximum Containers on a Ship)
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        maxWeight = min(n * n * w, maxWeight)
        return maxWeight // w

    # 3493. 属性图 (Properties Graph)
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        class union:
            def __init__(self, n):
                self.parent = [i for i in range(n)]
                self.rank = [1] * n
                self.n = n

            def find(self, x):
                if self.parent[x] == x:
                    return x
                self.parent[x] = self.find(self.parent[x])
                return self.parent[x]

            def is_connected(self, x, y):
                return self.find(x) == self.find(y)

            def union(self, x, y):
                root_x = self.find(x)
                root_y = self.find(y)
                if root_x == root_y:
                    return
                if self.rank[root_x] < self.rank[root_y]:
                    self.parent[root_x] = root_y
                else:
                    self.parent[root_y] = root_x
                    if self.rank[root_x] == self.rank[root_y]:
                        self.rank[root_x] += 1
                self.n -= 1

            def get_count(self):
                return self.n

        n = len(properties)
        u = union(n)
        for i in range(n):
            s1 = set(properties[i])
            for j in range(i + 1, n):
                s2 = set(properties[j])
                if len(s1 & s2) >= k:
                    u.union(i, j)
        return u.get_count()

    # 3494. 酿造药水需要的最少总时间 (Find the Minimum Amount of Time to Brew Potions)
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        pre = [0] * n
        for m in mana:
            s = 0
            for j in range(n):
                s = max(s, pre[j])
                s += skill[j] * m
            pre[-1] = s
            for j in range(n - 2, -1, -1):
                s -= skill[j + 1] * m
                pre[j] = s
        return pre[-1]

    # 2255. 统计是给定字符串前缀的字符串数目 (Count Prefixes of a Given String)
    def countPrefixes(self, words: List[str], s: str) -> int:
        class trie:
            def __init__(self):
                self.children = defaultdict(trie)
                self.cnt = 0

            def insert(self, word: str) -> None:
                cur = self
                for c in word:
                    if cur.children[c] is None:
                        cur.children[c] = trie()
                    cur = cur.children[c]
                cur.cnt += 1

            def query(self, s: str) -> int:
                cur = self
                res = 0
                for c in s:
                    if cur.children[c] is None:
                        break
                    cur = cur.children[c]
                    res += cur.cnt
                return res

        root = trie()
        for word in words:
            root.insert(word)
        return root.query(s)

    # 2711. 对角线上不同值的数量差 (Difference of Number of Distinct Values on Diagonals)
    def differenceOfDistinctValues(self, grid: List[List[int]]) -> List[List[int]]:
        m = len(grid)
        n = len(grid[0])
        suf = [[0] * n for _ in range(m)]
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                suf[i][j] = (
                    suf[i + 1][j + 1] | (1 << grid[i + 1][j + 1])
                    if i < m - 1 and j < n - 1
                    else 0
                )
        res = [[0] * n for _ in range(m)]
        pre = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                pre[i][j] = (
                    pre[i - 1][j - 1] | (1 << grid[i - 1][j - 1]) if i and j else 0
                )
                res[i][j] = abs(suf[i][j].bit_count() - pre[i][j].bit_count())
        return res

    # 2712. 使所有字符相等的最小成本 (Minimum Cost to Make All Characters Equal)
    def minimumCost(self, s: str) -> int:
        n = len(s)
        return sum(min(i, n - i) if s[i] != s[i - 1] else 0 for i in range(1, n))

    # 2716. 最小化字符串长度 (Minimize String Length)
    def minimizedStringLength(self, s: str) -> int:
        res = 0
        for c in s:
            res |= 1 << (ord(c) - ord("a"))
        return res.bit_count()

    # 2360. 图中的最长环 (Longest Cycle in a Graph)
    def longestCycle(self, edges: List[int]) -> int:
        n = len(edges)
        g = [[] for _ in range(n)]
        deg = [0] * n
        for i, v in enumerate(edges):
            if v != -1:
                g[i].append(v)
                deg[v] += 1
        q = deque()
        for i in range(n):
            if deg[i] == 0:
                q.append(i)
        while q:
            x = q.popleft()
            for y in g[x]:
                deg[y] -= 1
                if deg[y] == 0:
                    q.append(y)
        res = -1
        for i in range(n):
            if deg[i] != 0:
                x = i
                cnt = 0
                while deg[x] != 0:
                    deg[x] -= 1
                    x = edges[x]
                    cnt += 1
                res = max(res, cnt)
        return res

    # 2109. 向字符串添加空格 (Adding Spaces to a String)
    def addSpaces(self, s: str, spaces: List[int]) -> str:
        res = []
        j = 0
        for i, c in enumerate(s):
            if j < len(spaces) and i == spaces[j]:
                res.append(" ")
                j += 1
            res.append(c)
        return "".join(res)

    # 3498. 字符串的反转度 (Reverse Degree of a String)
    def reverseDegree(self, s: str) -> int:
        return sum(i * (26 - (ord(c) - ord("a"))) for i, c in enumerate(s, 1))

    # 3499. 操作后最大活跃区段数 I (Maximize Active Section with Trade I)
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        n = len(s)
        cnt1 = 0
        pre = -1
        res = 0
        i = 0
        while i < n:
            if s[i] == "1":
                i += 1
                cnt1 += 1
            else:
                cnt = 0
                while i < n and s[i] == "0":
                    cnt += 1
                    i += 1
                if pre != -1:
                    res = max(res, cnt + pre)
                pre = cnt
        return res + cnt1

    # 3502. 到达每个位置的最小费用 (Minimum Cost to Reach Every Position)
    def minCosts(self, cost: List[int]) -> List[int]:
        n = len(cost)
        res = [0] * n
        res[0] = cost[0]
        for i in range(1, n):
            res[i] = min(res[i - 1], cost[i])
        return res

    # 3503. 子字符串连接后的最长回文串 I (Longest Palindrome After Substring Concatenation I)
    # 3504. 子字符串连接后的最长回文串 II (Longest Palindrome After Substring Concatenation II)
    def calc(self, s: str, t: str) -> int:
        n, m = len(s), len(t)
        f = [[0] * (m + 1) for _ in range(n + 1)]
        for i, x in enumerate(s):
            for j, y in enumerate(t):
                if x == y:
                    f[i + 1][j] = f[i][j + 1] + 1
        mx = list(map(max, f))
        ans = max(mx) * 2  # |x| = |y| 的情况

        # 计算 |x| > |y| 的情况，中心扩展法
        for i in range(2 * n - 1):
            l, r = i // 2, (i + 1) // 2
            while l >= 0 and r < n and s[l] == s[r]:
                l -= 1
                r += 1
            if l + 1 <= r - 1:  # s[l+1] 到 s[r-1] 是非空回文串
                ans = max(ans, r - l - 1 + mx[l + 1] * 2)
        return ans

    def longestPalindrome(self, s: str, t: str) -> int:
        return max(self.calc(s, t), self.calc(t[::-1], s[::-1]))

    # 2278. 字母在字符串中的百分比 (Percentage of Letter in String)
    def percentageLetter(self, s: str, letter: str) -> int:
        return sum(c == letter for c in s) * 100 // len(s)

    # 1863. 找出所有子集的异或总和再求和 (Sum of All Subset XOR Totals)
    def subsetXORSum(self, nums: List[int]) -> int:
        def dfs(i: int, cur: int) -> None:
            if i == n:
                nonlocal res
                res += cur
                return
            dfs(i + 1, cur)
            dfs(i + 1, cur ^ nums[i])

        res = 0
        n = len(nums)
        dfs(0, 0)
        return res

    # 1922. 统计好数字的数目 (Count Good Numbers)
    def countGoodNumbers(self, n: int) -> int:
        even = (n + 1) // 2
        odd = n - even
        mod = 10**9 + 7
        return pow(5, even, mod) * pow(4, odd, mod) % mod

    # 3512. 使数组和能被 K 整除的最少操作次数 (Minimum Operations to Make Array Sum Divisible by K)
    def minOperations(self, nums: List[int], k: int) -> int:
        return sum(nums) % k

    # 3513. 不同 XOR 三元组的数目 I (Number of Unique XOR Triplets I)
    def uniqueXorTriplets(self, nums: List[int]) -> int:
        n = len(nums)
        return n if n <= 2 else 1 << n.bit_length()

    # 3514. 不同 XOR 三元组的数目 II (Number of Unique XOR Triplets II)
    def uniqueXorTriplets(self, nums: List[int]) -> int:
        nums = list(set(nums))  # 优化：去重，减少循环次数
        st = {x ^ y for x, y in combinations(nums, 2)} | {0}
        return len({xy ^ z for xy in st for z in nums})

    # 3516. 找到最近的人 (Find Closest Person)
    def findClosest(self, x: int, y: int, z: int) -> int:
        return 0 if x == y or x - z == z - y else (1 if abs(x - z) < abs(z - y) else 2)

    # 3517. 最小回文排列 I (Smallest Palindromic Rearrangement I)
    def smallestPalindrome(self, s: str) -> str:
        n = len(s)
        cnt = [0] * 26
        for c in s:
            cnt[ord(c) - ord("a")] += 1
        l = 0
        r = n - 1
        res = ["a"] * n
        for i in range(26):
            while cnt[i] > 1:
                cnt[i] -= 2
                res[l] = res[r] = chr(i + ord("a"))
                l += 1
                r -= 1
            if cnt[i] == 1:
                res[n // 2] = chr(i + ord("a"))
        return "".join(res)

    # 3519. 统计逐位非递减的整数 (Count Numbers with Non-Decreasing Digits)
    def countNumbers(self, l: str, r: str, b: int) -> int:
        def cal(x: int) -> int:
            @cache
            def dfs(i: int, j: int, is_limit: bool) -> int:
                if i == n:
                    return 1
                res = 0
                up = int(s[i]) if is_limit else b - 1
                for d in range(j, up + 1):
                    res += dfs(i + 1, d, is_limit and up == d)
                return res % MOD

            a = []
            while x:
                a.append(str(x % b))
                x //= b
            s = "".join(a[::-1])
            n = len(s)
            return dfs(0, 0, True)

        MOD = 10**9 + 7
        return (cal(int(r)) - cal(int(l) - 1)) % MOD

    # 1534. 统计好三元组 (ount Good Triplets)
    def countGoodTriplets(self, arr: List[int], a: int, b: int, c: int) -> int:
        res = 0
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if (
                        abs(arr[i] - arr[j]) <= a
                        and abs(arr[j] - arr[k]) <= b
                        and abs(arr[i] - arr[k]) <= c
                    ):
                        res += 1
        return res

    # 2179. 统计数组中好三元组数目 (Count Good Triplets in an Array)
    def goodTriplets(self, nums1: List[int], nums2: List[int]) -> int:
        def bisect(_list: List[int], x: int) -> int:
            n = len(_list)
            if not _list or _list[0] > x:
                _list.insert(0, x)
                return 0
            if _list[-1] < x:
                _list.append(x)
                return n
            left = 0
            right = len(_list) - 1
            res = -1
            while left <= right:
                mid = left + ((right - left) >> 1)
                if _list[mid] < x:
                    res = mid + 1
                    left = mid + 1
                else:
                    right = mid - 1
            _list.insert(res, x)
            return res

        n = len(nums1)
        arr = [0] * n
        for i, v in enumerate(nums1):
            arr[v] = i
        t = [0] * n
        for i, v in enumerate(nums2):
            t[i] = arr[v]
        res = 0
        _list = []
        for i in range(n):
            less = bisect(_list, t[i])
            more = n - i - 1 - (t[i] - less)
            res += more * less
        return res

    # 2364. 统计坏数对的数目 (Count Number of Bad Pairs)
    def countBadPairs(self, nums: List[int]) -> int:
        d = defaultdict(int)
        n = len(nums)
        res = 0
        for i, v in enumerate(nums):
            c = i - v
            res += d[c]
            d[c] += 1
        return n * (n - 1) // 2 - res

    # 2563. 统计公平数对的数目 (Count the Number of Fair Pairs)
    def countFairPairs(self, nums: List[int], lower: int, upper: int) -> int:
        def cal(target: int) -> int:
            s = SortedList()
            res = 0
            for x in nums:
                res += s.bisect_right(target - x)
                s.add(x)
            return res

        nums.sort()
        return cal(upper) - cal(lower - 1)

    # 781. 森林中的兔子 (Rabbits in Forest)
    def numRabbits(self, answers: List[int]) -> int:
        d = defaultdict(int)
        res = 0
        for x in answers:
            d[x] += 1
            if d[x] - 1 == x:
                res += d[x]
                del d[x]
        for k in d.keys():
            res += k + 1
        return res

    # 3522. 执行指令后的得分 (Calculate Score After Performing Instructions)
    def calculateScore(self, instructions: List[str], values: List[int]) -> int:
        n = len(instructions)
        vis = [False] * n
        res = 0
        i = 0
        while i < n and i >= 0 and not vis[i]:
            vis[i] = True
            if instructions[i] == "add":
                res += values[i]
                i += 1
            else:
                i += values[i]
        return res

    # 3523. 非递减数组的最大长度 (Make Array Non-decreasing)
    def maximumPossibleSize(self, nums: List[int]) -> int:
        ans = mx = 0
        for x in nums:
            if x >= mx:
                mx = x
                ans += 1
        return ans

    # 1399. 统计最大组的数目 (Count Largest Group)
    def countLargestGroup(self, n: int) -> int:
        @cache
        def dfs(i: int, j: int, is_limit: bool) -> int:
            if i == l:
                return j == 0
            res = 0
            up = int(s[i]) if is_limit else 9
            for d in range(min(j, up) + 1):
                res += dfs(i + 1, j - d, d == up and is_limit)
            return res

        res = 0
        mx = 0
        s = str(n)
        l = len(s)
        for i in range(1, l * 9 + 1):
            cnt = dfs(0, i, True)
            if cnt > mx:
                mx = cnt
                res = 1
            elif cnt == mx:
                res += 1
        return res

    # 3527. 找到最常见的回答 (Find the Most Common Response)
    def findCommonResponse(self, responses: List[List[str]]) -> str:
        d = defaultdict(int)
        mx = 0
        for response in responses:
            s = set(response)
            for x in s:
                d[x] += 1
                mx = max(mx, d[x])
        res = ""
        for x in d.keys():
            if d[x] == mx:
                if res == "" or x < res:
                    res = x
        return res

    # 3528. 单位转换 I (Unit Conversion I)
    def baseUnitConversions(self, conversions: List[List[int]]) -> List[int]:
        def dfs(x: int, v: int) -> int:
            res[x] = v
            for y, w in g[x]:
                dfs(y, v * w % MOD)

        n = len(conversions) + 1
        g = [[] for _ in range(n)]
        for x, y, w in conversions:
            g[x].append((y, w))
        MOD = 10**9 + 7
        res = [0] * n
        dfs(0, 1)
        return res

    # 3529. 统计水平子串和垂直子串重叠格子的数目 (Count Cells in Overlapping Horizontal and Vertical Substrings) --z函数 + 差分数组
    def countCells(self, grid: List[List[str]], pattern: str) -> int:
        def algorithm_z(s: str) -> List[int]:
            z = [0] * len(s)
            left, right = 0, 0
            diff = [0] * (len(s) - len(pattern) + 1)
            for i in range(1, len(s)):
                if i <= right:
                    z[i] = min(right - i + 1, z[i - left])
                while i + z[i] < len(s) and s[z[i]] == s[i + z[i]]:
                    left, right = i, i + z[i]
                    z[i] += 1
                if min(i, z[i]) >= len(pattern):
                    diff[i - len(pattern)] += 1
                    diff[i] -= 1
            for i in range(1, len(diff)):
                diff[i] += diff[i - 1]
            return diff

        def check() -> str:
            s = pattern
            for i in range(m):
                for j in range(n):
                    s += grid[i][j]
            return s

        def check2() -> str:
            s = pattern
            for j in range(n):
                for i in range(m):
                    s += grid[i][j]
            return s

        m = len(grid)
        n = len(grid[0])
        g0 = algorithm_z(check())
        g1 = algorithm_z(check2())
        res = 0
        for j in range(n):
            for i in range(m):
                if g0[i * n + j] and g1[j * m + i]:
                    res += 1
        return res

    # 3530. 有向无环图中合法拓扑排序的最大利润 (Maximum Profit from Valid Topological Order in DAG)
    def maxProfit(self, n: int, edges: List[List[int]], score: List[int]) -> int:
        @cache
        def dfs(i: int) -> int:
            if i == u:
                return 0
            c = i.bit_count() + 1
            res = 0
            j = u ^ i
            while j:
                lb = (j & -j).bit_length() - 1
                # lb未选择、且lb的直接祖先节点均已选择
                if pre[lb] | i == i:
                    res = max(res, dfs(i | (1 << lb)) + score[lb] * c)
                j &= j - 1

        if not edges:
            score.sort()
            res = 0
            for i, v in enumerate(score, 1):
                res += v * i
            return res
        pre = [0] * n
        for x, y in edges:
            pre[y] |= 1 << x
        u = (1 << n) - 1
        return dfs(0)

    # 3531. 统计被覆盖的建筑 (Count Covered Buildings)
    def countCoveredBuildings(self, n: int, buildings: List[List[int]]) -> int:
        res = 0
        col = [[inf] * (n + 1) if i == 0 else [0] * (n + 1) for i in range(2)]
        row = [[inf] * (n + 1) if i == 0 else [0] * (n + 1) for i in range(2)]
        for c, r in buildings:
            col[0][c] = min(col[0][c], r)
            col[1][c] = max(col[1][c], r)
            row[0][r] = min(row[0][r], c)
            row[1][r] = max(row[1][r], c)
        for c, r in buildings:
            if col[0][c] == r or col[1][c] == r or row[0][r] == c or row[1][r] == c:
                continue
            res += 1
        return res

    # 3532. 针对图的路径存在性查询 I (Path Existence Queries in a Graph I)
    def pathExistenceQueries(
        self, n: int, nums: List[int], maxDiff: int, queries: List[List[int]]
    ) -> List[bool]:
        class union:
            def __init__(self, n):
                self.parent = [i for i in range(n)]
                self.rank = [1] * n

            def find(self, x) -> int:
                if self.parent[x] == x:
                    return x
                self.parent[x] = self.find(self.parent[x])
                return self.parent[x]

            def is_connected(self, x, y) -> bool:
                return self.find(x) == self.find(y)

            def union(self, x, y):
                root_x = self.find(x)
                root_y = self.find(y)
                if root_x == root_y:
                    return
                if self.rank[root_x] < self.rank[root_y]:
                    self.parent[root_x] = root_y
                else:
                    self.parent[root_y] = root_x
                    if self.rank[root_x] == self.rank[root_y]:
                        self.rank[root_x] += 1

        idx = sorted(range(n), key=lambda i: nums[i])
        union = union(n)
        for x, y in pairwise(idx):
            if nums[y] - nums[x] <= maxDiff:
                union.union(x, y)
        return [union.is_connected(x, y) for x, y in queries]

    # 3533. 判断连接可整除性 (Concatenated Divisibility)
    def concatenatedDivisibility(self, nums: List[int], k: int) -> List[int]:
        @cache
        def dfs(i: int, j: int) -> bool:
            if i == (1 << n) - 1:
                return j == 0
            for id, (p10, x) in enumerate(zip(pow10, nums)):
                if (i >> id) & 1 == 0 and dfs(i | (1 << id), (j * p10 + x) % k):
                    res.append(x)
                    return True
            return False

        res = []
        n = len(nums)
        nums.sort()
        pow10 = [10 ** len(str(x)) for x in nums]
        if not dfs(0, 0):
            return []
        return res

    # 1295. 统计位数为偶数的数字 (Find Numbers with Even Number of Digits)
    def findNumbers(self, nums: List[int]) -> int:
        return sum(len(str(x)) & 1 == 0 for x in nums)

    # 1007. 行相等的最少多米诺旋转 (Minimum Domino Rotations For Equal Row)
    def minDominoRotations(self, tops: List[int], bottoms: List[int]) -> int:
        def check(target: int) -> int:
            to_top = to_bottom = 0
            for x, y in zip(tops, bottoms):
                if x != target and y != target:
                    return inf
                if x != target:
                    to_top += 1
                elif y != target:
                    to_bottom += 1
            return min(to_top, to_bottom)

        res = min(check(tops[0]), check(bottoms[0]))
        return res if res != inf else -1

    # 1128. 等价多米诺骨牌对的数量 (Number of Equivalent Domino Pairs)
    def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
        res = 0
        cnts = [0] * (1 << 10)
        for x, y in dominoes:
            b = (1 << x) | (1 << y)
            res += cnts[b]
            cnts[b] += 1
        return res

    # 2094. 找出 3 位偶数 (Finding 3-Digit Even Numbers)
    def findEvenNumbers(self, digits: List[int]) -> List[int]:
        def dfs(i: int) -> None:
            nonlocal cur
            if i == 3:
                s.add(cur)
                return
            for j, x in enumerate(cnts):
                if i == 0 and j == 0 or x == 0 or i == 2 and j % 2 == 1:
                    continue
                cnts[j] -= 1
                cur = cur * 10 + j
                dfs(i + 1)
                cur //= 10
                cnts[j] += 1

        cnts = [0] * 10
        for d in digits:
            cnts[d] += 1
        s = set()
        cur = 0
        dfs(0)
        return sorted(list(s))

    # 3536. 两个数字的最大乘积 (Maximum Product of Two Digits)
    def maxProduct(self, n: int) -> int:
        cnts = [0] * 10
        while n:
            cnts[n % 10] += 1
            n //= 10
        i = 0
        ret = [0] * 2
        for j in range(9, -1, -1):
            while cnts[j] and i < 2:
                ret[i] = j
                cnts[j] -= 1
                i += 1
            if i == 2:
                break
        return ret[0] * ret[1]

    # 3537. 填充特殊网格 (Fill a Special Grid)
    def specialGrid(self, n: int) -> List[List[int]]:
        def dfs(i0: int, j0: int, i1: int, j1: int) -> None:
            if i0 == i1 and j0 == j1:
                nonlocal id
                res[i0][j0] = id
                id += 1
                return
            # 右上
            dfs(i0, ((j0 + j1) >> 1) + 1, (i0 + i1) >> 1, j1)
            # 右下
            dfs(((i0 + i1) >> 1) + 1, ((j0 + j1) >> 1) + 1, i1, j1)
            # 左下
            dfs(((i0 + i1) >> 1) + 1, j0, i1, (j0 + j1) >> 1)
            # 左上
            dfs(i0, j0, (i0 + i1) >> 1, (j0 + j1) >> 1)

        res = [[0] * (1 << n) for _ in range(1 << n)]
        id = 0
        dfs(0, 0, (1 << n) - 1, (1 << n) - 1)
        return res

    # 3538. 合并得到最小旅行时间 (Merge Operations for Minimum Travel Time)
    def minTravelTime(
        self, _, n: int, k: int, position: List[int], time: List[int]
    ) -> int:
        s = list(accumulate(time, initial=0))  # 计算 time 的前缀和

        @cache
        def dfs(i: int, j: int, left_k: int) -> int:
            if j == n - 1:  # 到达终点
                return inf if left_k else 0
            t = s[j + 1] - s[i]  # 合并到 time[j] 的时间
            # 枚举下一个子数组 [j+1, k]
            return min(
                dfs(j + 1, k, left_k - (k - j - 1)) + (position[k] - position[j]) * t
                for k in range(j + 1, min(n, j + 2 + left_k))
            )

        return dfs(0, 0, k)  # 第一个子数组是 [0, 0]

    # 3335. 字符串转换后的长度 I (Total Characters in String After Transformations I)
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        cnts = [0] * 26
        for c in s:
            cnts[ord(c) - ord("a")] += 1
        MOD = 10**9 + 7
        for _ in range(t):
            nxt = [0] * 26
            nxt[0] = cnts[25]
            nxt[1] = (cnts[0] + cnts[25]) % MOD
            for i in range(2, 26):
                nxt[i] = cnts[i - 1]
            cnts = nxt
        return sum(cnts) % MOD

    # 3541. 找到频率最高的元音和辅音 (Find Most Frequent Vowel and Consonant)
    def maxFreqSum(self, s: str) -> int:
        cnts = [0] * 26
        c0 = 0
        c1 = 0
        for x in s:
            cnts[ord(x) - ord("a")] += 1
            if x in "aeiou":
                c0 = max(c0, cnts[ord(x) - ord("a")])
            else:
                c1 = max(c1, cnts[ord(x) - ord("a")])
        return c0 + c1

    # 3543. K 条边路径的最大边权和 (Maximum Weighted K-Edge Path) --dfs
    def maxWeight(self, n: int, edges: List[List[int]], k: int, t: int) -> int:
        @cache
        def dfs(x: int, s: int, w: int) -> None:
            if s == k:
                nonlocal res
                res = max(res, w)
                return
            for y, dw in g[x]:
                if w + dw < t:
                    dfs(y, s + 1, w + dw)

        g = [[] for _ in range(n)]
        for u, v, w in edges:
            g[u].append([v, w])
        res = -1
        for i in range(n):
            dfs(i, 0, 0)
        return res

    # 3542. 将所有元素变为 0 的最少操作次数 (Minimum Operations to Convert All Elements to Zero)
    def minOperations(self, nums: List[int]) -> int:
        ans = 0
        st = []
        for x in nums:
            while st and x < st[-1]:
                st.pop()
                ans += 1
            # 如果 x 与栈顶相同，那么 x 与栈顶可以在同一次操作中都变成 0，x 无需入栈
            if not st or x != st[-1]:
                st.append(x)
        return ans + len(st) - (st[0] == 0)  # 0 不需要操作

    # 3544. 子树反转和 (Subtree Inversion Sum)
    def subtreeInversionSum(
        self, edges: List[List[int]], nums: List[int], k: int
    ) -> int:
        @cache
        def dfs(x: int, fa: int, left: int, p: int) -> int:
            t = (x, left, p)
            if t in memo:
                return memo[t]
            # 不反转 x
            res = nums[x] * p
            for y in g[x]:
                if y != fa:
                    res += dfs(y, x, max(left - 1, 0), p)
            # 反转 x
            if left == 0:
                p = -p
                s = nums[x] * p
                for y in g[x]:
                    if y != fa:
                        s += dfs(y, x, k - 1, p)
                if s > res:
                    res = s
            memo[t] = res
            return res

        n = len(nums)
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        memo = {}
        res = dfs(0, -1, 0, 1)
        dfs.cache_clear()
        return res

    # 3545. 不同字符数量最多为 K 时的最少删除数 (Minimum Deletions for At Most K Distinct Characters)
    def minDeletion(self, s: str, k: int) -> int:
        cnts = [0] * 26
        for c in s:
            cnts[ord(c) - ord("a")] += 1
        return sum(sorted(cnts)[: 26 - k])

    # 3546. 等和矩阵分割 I (Equal Sum Grid Partition I)
    def canPartitionGrid(self, grid: List[List[int]]) -> bool:
        def check(arr: List[int]) -> bool:
            suf = sum(arr)
            pre = 0
            for x in arr:
                pre += x
                suf -= x
                if pre == suf:
                    return True
            return False

        m = len(grid)
        n = len(grid[0])
        col = [0] * n
        row = [0] * m
        for i in range(m):
            for j in range(n):
                col[j] += grid[i][j]
                row[i] += grid[i][j]
        return check(col) or check(row)

    # 3547. 图中边值的最大和 (Maximum Sum of Edge Values in a Graph)
    def maxScore(self, n: int, edges: List[List[int]]) -> int:
        ans = (n * n * 2 + n * 5 - 6) * (n - 1) // 6
        if n == len(edges):  # 环
            ans += 2
        return ans
