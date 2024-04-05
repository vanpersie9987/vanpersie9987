from ast import Return, Tuple
from asyncio import FastChildWatcher
from audioop import minmax, reverse
from calendar import c
from collections import Counter, defaultdict, deque
import collections
from ctypes.wintypes import _ULARGE_INTEGER
from curses import curs_set, intrflush
from curses.ascii import isprint
from decimal import Rounded
import enum
from functools import cache
from inspect import modulesbyfile
from itertools import accumulate, count, islice, pairwise, permutations
from locale import DAY_4
from math import comb, cos, fabs, gcd, inf, isqrt, sqrt
from mimetypes import init
from operator import le, truediv
from pickletools import read_uint1
from queue import PriorityQueue
from re import X
import re
from socket import NI_NUMERICSERV
from ssl import VERIFY_X509_TRUSTED_FIRST
from string import ascii_lowercase
from tabnanny import check
from textwrap import indent
from tkinter import N, NO, W
from tkinter.tix import Tree
from turtle import mode, pos, reset, right, st
from typing import List, Optional
import heapq
import bisect
from unittest.util import _count_diff_all_purpose
from wsgiref.util import guess_scheme
from xml.dom import Node
from zoneinfo import reset_tzpath

# curl https://bootstrap.pypa.io/pip/get-pip.py -o get-pip.py
# sudo python3 get-pip.py
# pip3 install sortedcontainers
from sortedcontainers import SortedList, SortedSet


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
        cur = ""
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
        res = 0
        j = 0
        s = 0
        d = defaultdict(int)
        for v in nums:
            d[v] += 1
            s += v
            while d[v] > 1:
                d[nums[j]] -= 1
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

    # 2516. 每种字符至少取 K 个 (Take K of Each Character From Left and Right)
    def takeCharacters(self, s: str, k: int) -> int:
        n = len(s)
        cnt = [0] * 3
        for c in s:
            cnt[ord(c) - ord("a")] += 1
        for i in range(3):
            cnt[i] -= k
            if cnt[i] < 0:
                return -1
        res = -1
        j = 0
        for i, v in enumerate(s):
            x = ord(v) - ord("a")
            cnt[x] -= 1
            while cnt[x] < 0:
                cnt[ord(s[j]) - ord("a")] += 1
                j += 1
            res = max(res, i - j + 1)
        return -1 if res == -1 else n - res

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
        s = list(accumulate(nums, initial=0))
        j = 0
        res = 0
        for i in range(len(nums)):
            cur = (s[i + 1] - s[j]) * (i - j + 1)
            while j <= i and cur >= k:
                j += 1
                cur = (s[i + 1] - s[j]) * (i - j + 1)
            if cur < k:
                res += i - j + 1
        return res

    # 2537. 统计好子数组的数目 (Count the Number of Good Subarrays)
    def countGood(self, nums: List[int], k: int) -> int:
        n = len(nums)
        res = 0
        j = 0
        d = defaultdict(int)
        s = 0
        for i, v in enumerate(nums):
            s += d[v]
            d[v] += 1
            while s >= k:
                res += n - i
                d[nums[j]] -= 1
                s -= d[nums[j]]
                j += 1
        return res

    # 2970. 统计移除递增子数组的数目 I (Count the Number of Incremovable Subarrays I)
    # 2972. 统计移除递增子数组的数目 II (Count the Number of Incremovable Subarrays II)
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        i = 0
        while i < n - 1:
            if nums[i] >= nums[i + 1]:
                break
            i += 1
        if i == n - 1:
            return (1 + n) * n // 2
        j = n - 1
        while j >= 1:
            if nums[j - 1] >= nums[j]:
                break
            j -= 1
        res = 1 + n - j
        for k in range(i + 1):
            while j < n and nums[k] >= nums[j]:
                j += 1
            res += n - j + 1
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
            res1 = False
            res2 = False
            if i + 1 < n and nums[i] == nums[i + 1]:
                res1 = dfs(i + 2)
            if i + 2 < n and (
                nums[i] == nums[i + 1] == nums[i + 2]
                or nums[i] + 1 == nums[i + 1]
                and nums[i + 1] + 1 == nums[i + 2]
            ):
                res2 = dfs(i + 3)
            return res1 or res2

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

    # 3066. 超过阈值的最少操作数 II (Minimum Operations to Exceed Threshold Value II)
    def minOperations(self, nums: List[int], k: int) -> int:
        q = []
        heapq.heapify(q)
        for x in nums:
            heapq.heappush(q, x)
        res = 0
        while len(q) >= 2:
            if q[0] >= k:
                break
            res += 1
            x = heapq.heappop(q)
            y = heapq.heappop(q)
            heapq.heappush(q, min(x, y) * 2 + max(x, y))
        return res

    # 3067. 在带权树网络中统计可连接服务器对数目 (Count Pairs of Connectable Servers in a Weighted Tree Network)
    def countPairsOfConnectableServers(
        self, edges: List[List[int]], signalSpeed: int
    ) -> List[int]:
        def dfs(x: int, fa: int, w: int) -> int:
            sum = w % signalSpeed == 0
            for y, nw in g[x]:
                if y != fa:
                    sum += dfs(y, x, w + nw)
            return sum

        n = len(edges) + 1
        g = [[] for _ in range(n)]
        for u, v, w in edges:
            g[u].append((v, w))
            g[v].append((u, w))
        res = [0] * n
        for i in range(n):
            s = 0
            cur = 0
            for x, w in g[i]:
                c = dfs(x, i, w)
                s += cur * c
                cur += c
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
        def greater_count(arr: List[List[int]], x: int) -> int:
            n = len(arr)
            if arr[0][0] > x:
                return n
            if arr[-1][0] <= x:
                return 0
            left = 0
            right = n - 1
            res = -1
            while left <= right:
                mid = left + ((right - left) >> 1)
                if arr[mid][0] > x:
                    res = n - mid
                    right = mid - 1
                else:
                    left = mid + 1
            return res

        arr1 = [[nums[0], 0]]
        arr2 = [[nums[1], 1]]
        for i in range(2, len(nums)):
            l1 = len(arr1)
            l2 = len(arr2)
            g1 = greater_count(arr1, nums[i])
            g2 = greater_count(arr2, nums[i])
            if g1 > g2:
                arr1.insert(l1 - g1, [nums[i], i])
            elif g1 < g2:
                arr2.insert(l2 - g2, [nums[i], i])
            elif l1 > l2:
                arr2.insert(l2 - g2, [nums[i], i])
            else:
                arr1.insert(l1 - g1, [nums[i], i])
        return [x for x, _ in sorted(arr1, key=lambda k: k[1])] + [
            x for x, _ in sorted(arr2, key=lambda k: k[1])
        ]

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
        n = len(nums)

        @cache
        def dfs(i: int, j: int, p: int) -> int:
            if i == n:
                return 0 if j == k else -inf
            if n - i < k - j:
                return -inf
            # 不选
            res = dfs(i + 1, j, 0)
            # 选
            if j < k:
                res = max(
                    res,
                    dfs(i + 1, j + 1, 1)
                    + (-1 if (j + 1) % 2 == 0 else 1) * nums[i] * (k - j),
                )
            if p == 1:
                res = max(
                    res,
                    dfs(i + 1, j, 1)
                    + (-1 if j % 2 == 0 else 1) * nums[i] * (k - j + 1),
                )
            return res

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
        cnt = [0] * 26
        for c in word:
            cnt[ord(c) - ord("a")] += 1
        cnt.sort()
        j = 0
        s = 0
        res = 0
        for i, v in enumerate(cnt):
            s += v
            while cnt[j] < v:
                s -= cnt[j]
                j += 1
            cur = 0
            for x in range(i + 1, 26):
                cur += min(cnt[x], v + k)
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
        def cal(nums: list) -> int:
            @cache
            def dfs(i: int) -> int:
                if i < 0:
                    return 1
                if i and nums[i][0] - nums[i - 1][0] == k:
                    return dfs(i - 1) + dfs(i - 2) * ((1 << nums[i][1]) - 1)
                return dfs(i - 1) << nums[i][1]

            m = len(nums)
            return dfs(m - 1)

        d = defaultdict(Counter)
        for x in nums:
            d[x % k][x] += 1
        res = 1
        for i in d.values():
            g = sorted(i.items())
            res *= cal(list(g))
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
        def ret(_s: int) -> int:
            def cal(num: int) -> int:
                @cache
                def dfs(i: int, j: int, is_limit: bool, is_num: bool) -> int:
                    if i == n:
                        return is_num and j == _s
                    res = 0
                    if not is_num:
                        res = dfs(i + 1, j, False, False)
                    up = int(s[i]) if is_limit else 9
                    for d in range(0 if is_num else 1, up + 1):
                        if j + d > _s:
                            break
                        res += dfs(i + 1, j + d, is_limit and up == d, True)
                    return res

                s = str(num)
                n = len(s)
                return dfs(0, 0, True, False)

            return cal(highLimit) - cal(lowLimit - 1)

        return max(ret(i) for i in range(1, 46))

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

    # 741. 摘樱桃 (Cherry Pickup)
    def cherryPickup(self, grid: List[List[int]]) -> int:
        @cache
        def dfs(i0: int, j0: int, j1: int) -> int:
            i1 = i0 + j0 - j1
            if (
                i0 == n
                or i1 == n
                or j0 == n
                or j1 == n
                or grid[i0][j0] == -1
                or grid[i1][j1] == -1
            ):
                return -inf
            if i0 == n - 1 and i1 == n - 1 and j0 == n - 1 and j1 == n - 1:
                return grid[i0][j0]
            return max(
                dfs(i0 + 1, j0, j1),
                dfs(i0 + 1, j0, j1 + 1),
                dfs(i0, j0 + 1, j1),
                dfs(i0, j0 + 1, j1 + 1),
            ) + (grid[i0][j0] if i0 == i1 and j0 == j1 else grid[i0][j0] + grid[i1][j1])

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
        def operate(x: int, sign: int) -> None:
            i = 0
            while x:
                cnt[i] += (x & 1) * sign
                x >>= 1
                i += 1

        def check() -> bool:
            b = 0
            for i, c in enumerate(cnt):
                if c:
                    b |= 1 << i
            return b >= k

        res = inf
        i = 0
        j = 0
        n = len(nums)
        cnt = [0] * 31
        while i < n:
            operate(nums[i], 1)
            while j <= i and check():
                res = min(res, i - j + 1)
                operate(nums[j], -1)
                j += 1
            i += 1
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
        def dfs(node: 'TreeNode') -> tuple:
            if node is None:
                return ()
            left = dfs(node.left)
            right = dfs(node.right)
            ret = [node.val, node.val]
            if left:
                nonlocal res
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
