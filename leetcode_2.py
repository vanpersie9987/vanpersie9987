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
from turtle import reset, right, st
from typing import List, Optional
import heapq
import bisect
from xml.dom import Node
from zoneinfo import reset_tzpath
# curl https://bootstrap.pypa.io/pip/get-pip.py -o get-pip.py
# sudo python3 get-pip.py
# pip3 install sortedcontainers
from sortedcontainers import SortedList

class leetcode_2:
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
        def dfs(root: 'TreeNode') -> None:
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
        for v in 'aeiou':
            mask |= 1 << (ord(v) - ord('a'))
        res = 0
        cur = 0
        for i, v in enumerate(s):
            if (mask >> (ord(v) - ord('a'))) & 1:
                cur += 1
            if i >= k:
                if (mask >> (ord(s[i - k]) - ord('a'))) & 1:
                    cur -= 1
            if i >= k - 1:
                res = max(res, cur)
        return res
    
    # 2269. 找到一个数字的 K 美丽值 (Find the K-Beauty of a Number)
    def divisorSubstrings(self, num: int, k: int) -> int:
        s = str(num)
        cur = ''
        res = 0
        for i, v in enumerate(s):
            cur += v
            if i >= k:
                cur = cur[1:]
            if i >= k - 1:
                if int(cur) and int(num) % int(cur) == 0:
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
            val = nums[x]
            p = -1
            cur = -1
            for v in s[val]:
                if len(val_to_node[v]) > 0 and node_to_pos[val_to_node[v][-1]] > p:
                    p = node_to_pos[val_to_node[v][-1]]
                    cur = val_to_node[v][-1]
            res[x] = cur
            val_to_node[val].append(x)
            nonlocal pos
            node_to_pos[x] = pos
            pos += 1
            for y in g[x]:
                if y != fa:
                    dfs(y, x)
            val_to_node[val].pop()
            del node_to_pos[x]
            pos -= 1
        n = len(nums)
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        s = [[] for _ in range(51)]
        for i in range(1, 51):
            for j in range(1, 51):
                if gcd(i, j) == 1:
                    s[i].append(j)
        val_to_node = [[] for _ in range(51)]
        node_to_pos = dict()
        pos = 0
        res = [-1] * n
        dfs(0, -1)
        return res
    
    # 144. 二叉树的前序遍历 (Binary Tree Preorder Traversal)
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        def dfs(root: 'TreeNode') -> None:
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
        def dfs(root: 'TreeNode') -> None:
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
    

    # 3034. 匹配模式数组的子数组数目 I (Number of Subarrays That Match a Pattern I)
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        def check(i: int, j: int) -> int:
            for k in range(i + 1, j + 1):
                p = 1 if nums[k] - nums[k - 1] > 0 else (-1 if nums[k] - nums[k - 1] < 0 else 0)
                if p != pattern[k - i - 1]:
                    return 0
            return 1
        n = len(nums)
        m = len(pattern)
        return sum(check(i - m, i) for i in range(m, n))
    

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
        def dfs(root: 'TreeNode', i: int, j: int) -> None:
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