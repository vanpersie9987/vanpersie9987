import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;

public class LeetCode_2 {

   public static void main(final String[] args) {

   }

   // 2032. 至少在两个数组中出现的值 (Two Out of Three)
   public List<Integer> twoOutOfThree(int[] nums1, int[] nums2, int[] nums3) {
      int[] count1 = new int[101];
      for (int num : nums1) {
         count1[num] = 1;
      }
      int[] count2 = new int[101];
      for (int num : nums2) {
         count2[num] = 1;
      }
      int[] count3 = new int[101];
      for (int num : nums3) {
         count3[num] = 1;
      }
      List<Integer> res = new ArrayList<>();
      for (int i = 1; i <= 100; ++i) {
         if (count1[i] + count2[i] + count3[i] > 1) {
            res.add(i);
         }
      }
      return res;

   }

   // 2032. 至少在两个数组中出现的值 (Two Out of Three)
   public List<Integer> twoOutOfThree2(int[] nums1, int[] nums2, int[] nums3) {
      int[] arr = new int[101];
      for (int num : nums1) {
         arr[num] |= 1 << 0;
      }
      for (int num : nums2) {
         arr[num] |= 1 << 1;
      }
      for (int num : nums3) {
         arr[num] |= 1 << 2;
      }
      List<Integer> res = new ArrayList<>();
      for (int i = 1; i <= 100; ++i) {
         if ((arr[i] & (arr[i] - 1)) != 0) {
            res.add(i);
         }
      }
      return res;

   }

   // 215. 数组中的第K个最大元素 (Kth Largest Element in an Array) 需掌握官方题解的方法
   // 剑指 Offer II 076. 数组中的第 k 大的数字
   public int findKthLargest(int[] nums, int k) {
      int counts[] = new int[20001];
      for (int num : nums) {
         ++counts[num + 10000];
      }
      for (int i = 20000; i >= 0; --i) {
         if (k <= counts[i]) {
            return i - 10000;
         }
         k -= counts[i];
      }
      return k;

   }

   // 852. 山脉数组的峰顶索引 (Peak Index in a Mountain Array)
   // 剑指 Offer II 069. 山峰数组的顶部
   public int peakIndexInMountainArray(int[] arr) {
      int left = 1;
      int right = arr.length - 2;
      int ans = -1;
      while (left <= right) {
         int mid = left + ((right - left) >>> 1);
         if (arr[mid] > arr[mid + 1]) {
            ans = mid;
            right = mid - 1;
         } else {
            left = mid + 1;
         }
      }
      return ans;

   }

   // 1371. 每个元音包含偶数次的最长子字符串 (Find the Longest Substring Containing Vowels in Even
   // Counts)
   public int findTheLongestSubstring(String s) {
      int status = 0;
      int[] arr = new int[1 << 5];
      Arrays.fill(arr, Integer.MAX_VALUE);
      arr[0] = -1;
      int res = 0;
      for (int i = 0; i < s.length(); ++i) {
         char c = s.charAt(i);
         if (c == 'a') {
            status ^= 1 << 0;
         } else if (c == 'e') {
            status ^= 1 << 1;
         } else if (c == 'i') {
            status ^= 1 << 2;
         } else if (c == 'o') {
            status ^= 1 << 3;
         } else if (c == 'u') {
            status ^= 1 << 4;
         }
         if (arr[status] == Integer.MAX_VALUE) {
            arr[status] = i;
         } else {
            res = Math.max(res, i - arr[status]);
         }
      }
      return res;
   }

   // 3. 无重复字符的最长子串 (Longest Substring Without Repeating Characters)
   // 剑指 Offer 48. 最长不含重复字符的子字符串
   public int lengthOfLongestSubstring(String s) {
      Set<Character> set = new HashSet<>();
      int res = 0;
      int l = 0;
      int r = 0;
      while (r < s.length()) {
         if (!set.contains(s.charAt(r))) {
            set.add(s.charAt(r++));
            res = Math.max(res, r - l);
         } else {
            set.remove(s.charAt(l++));
         }
      }
      return res;

   }

   // 525. 连续数组 (Contiguous Array)
   // 剑指 Offer II 011. 0 和 1 个数相同的子数组
   public int findMaxLength(int[] nums) {
      //// key : 前缀和，遇1则+1，遇0则减1
      // value : 数组的当前索引
      // 存储第一次出现的key
      Map<Integer, Integer> map = new HashMap<>();
      map.put(0, -1);
      int count = 0;
      int res = 0;
      for (int i = 0; i < nums.length; ++i) {
         if (nums[i] == 0) {
            --count;
         } else {
            ++count;
         }
         if (map.containsKey(count)) {
            res = Math.max(res, i - map.get(count));
         } else {
            map.put(count, i);
         }
      }
      return res;

   }

   // 146. LRU 缓存机制 (LRU Cache)
   // 面试题 16.25. LRU 缓存 (LRU Cache LCCI)
   // 剑指 Offer II 031. 最近最少使用缓存
   class LRUCache {
      class CacheNode {
         CacheNode prev;
         CacheNode next;
         int value;
         int key;

         public CacheNode() {
         }

         public CacheNode(int _key, int _value) {
            key = _key;
            value = _value;
         }

      }

      private int capacity;
      private int size;
      private CacheNode head;
      private CacheNode tail;
      private Map<Integer, CacheNode> cache;

      public LRUCache(int capacity) {
         cache = new HashMap<>();
         head = new CacheNode();
         tail = new CacheNode();
         this.capacity = capacity;
         this.size = 0;
         head.next = tail;
         tail.prev = head;

      }

      public int get(int key) {
         CacheNode curNode = cache.get(key);
         if (curNode == null) {
            return -1;
         }
         moveToHead(curNode);
         return curNode.value;

      }

      public void put(int key, int value) {
         CacheNode curNode = cache.get(key);
         if (curNode == null) {
            curNode = new CacheNode(key, value);
            cache.put(key, curNode);
            addToHead(curNode);
            ++size;
            if (size > capacity) {
               cache.remove(tail.prev.key);
               removeNode(tail.prev);
               --size;
            }
         } else {
            curNode.value = value;
            moveToHead(curNode);
         }
      }

      private void removeNode(CacheNode node) {
         node.prev.next = node.next;
         node.next.prev = node.prev;
      }

      private void addToHead(CacheNode node) {
         node.next = head.next;
         head.next = node;
         node.prev = head;
         node.next.prev = node;
      }

      private void moveToHead(CacheNode node) {
         removeNode(node);
         addToHead(node);
      }
   }

   // 676. 实现一个魔法字典 (Implement Magic Dictionary)
   // 剑指 Offer II 064. 神奇的字典
   class MagicDictionary {
      private Map<Integer, List<String>> map;

      /** Initialize your data structure here. */
      public MagicDictionary() {
         map = new HashMap<>();

      }

      public void buildDict(String[] dictionary) {
         for (String d : dictionary) {
            map.computeIfAbsent(d.length(), k -> new ArrayList<>()).add(d);
         }
      }

      public boolean search(String searchWord) {
         if (!map.containsKey(searchWord.length())) {
            return false;
         }
         for (String word : map.get(searchWord.length())) {
            int diff = 0;
            for (int i = 0; i < word.length(); ++i) {
               if (word.charAt(i) != searchWord.charAt(i)) {
                  if (++diff > 1) {
                     break;
                  }
               }
            }
            if (diff == 1) {
               return true;
            }
         }
         return false;
      }
   }

   // 676. 实现一个魔法字典 (Implement Magic Dictionary)
   // 剑指 Offer II 064. 神奇的字典
   class MagicDictionary2 {
      private Map<String, Integer> map;
      private Set<String> set;

      /** Initialize your data structure here. */
      public MagicDictionary2() {
         map = new HashMap<>();
         set = new HashSet<>();

      }

      public void buildDict(String[] dictionary) {
         for (String word : dictionary) {
            set.add(word);
            for (String neighbor : generateNeighbors(word)) {
               map.put(neighbor, map.getOrDefault(neighbor, 0) + 1);
            }
         }
      }

      public List<String> generateNeighbors(String word) {
         List<String> neighbors = new ArrayList<>();
         char[] chars = word.toCharArray();
         for (int i = 0; i < chars.length; ++i) {
            char temp = chars[i];
            chars[i] = '_';
            neighbors.add(String.valueOf(chars));
            chars[i] = temp;
         }
         return neighbors;
      }

      public boolean search(String searchWord) {
         for (String word : generateNeighbors(searchWord)) {
            int count = map.getOrDefault(word, 0);
            if (count > 1 || count == 1 && !set.contains(searchWord)) {
               return true;
            }
         }
         return false;
      }
   }

   // 540. 有序数组中的单一元素 (Single Element in a Sorted Array)
   // 剑指 Offer II 070. 排序数组中只出现一次的数字 还需要掌握二分查找法
   public int singleNonDuplicate(int[] nums) {
      int res = 0;
      for (int num : nums) {
         res ^= num;
      }
      return res;

   }

   // 560. 和为K的子数组 (Subarray Sum Equals K) 前缀和
   // 剑指 Offer II 010. 和为 k 的子数组
   public int subarraySum(int[] nums, int k) {
      Map<Integer, Integer> map = new HashMap<>();
      int res = 0;
      int preSum = 0;
      map.put(0, 1);
      for (int i = 0; i < nums.length; ++i) {
         preSum += nums[i];
         res += map.getOrDefault(preSum - k, 0);
         map.put(preSum, map.getOrDefault(preSum, 0) + 1);
      }
      return res;
   }

   // 724. 寻找数组的中心下标 前缀和
   // 1991. 找到数组的中间位置 (Find the Middle Index in Array)
   // 剑指 Offer II 012. 左右两边子数组的和相等
   public int pivotIndex(final int[] nums) {
      int leftSum = 0;
      int sum = Arrays.stream(nums).sum();
      for (int i = 0; i < nums.length; ++i) {
         if (leftSum == sum - nums[i] - leftSum) {
            return i;
         }
         leftSum += nums[i];
      }
      return -1;
   }

   // 930. 和相同的二元子数组 (Binary Subarrays With Sum) 前缀和 还需掌握滑动窗口
   public int numSubarraysWithSum(int[] nums, int goal) {
      Map<Integer, Integer> map = new HashMap<>();
      int res = 0;
      map.put(0, 1);
      int preSum = 0;
      for (int i = 0; i < nums.length; ++i) {
         preSum += nums[i];
         res += map.getOrDefault(preSum - goal, 0);
         map.put(preSum, map.getOrDefault(preSum, 0) + 1);
      }
      return res;
   }

   // 1248. 统计「优美子数组」(Count Number of Nice Subarrays) 前缀和
   public int numberOfSubarrays(int[] nums, int k) {
      Map<Integer, Integer> map = new HashMap<>();
      int oddNum = 0;
      map.put(0, 1);
      int res = 0;
      for (int i = 0; i < nums.length; ++i) {
         oddNum += nums[i] & 1;
         res += map.getOrDefault(oddNum - k, 0);
         map.put(oddNum, map.getOrDefault(oddNum, 0) + 1);
      }
      return res;

   }

   // 1248. 统计「优美子数组」(Count Number of Nice Subarrays) 前缀和
   public int numberOfSubarrays2(int[] nums, int k) {
      int[] arr = new int[nums.length + 1];
      int res = 0;
      int oddNum = 0;
      arr[0] = 1;
      for (int i = 0; i < nums.length; ++i) {
         oddNum += nums[i] & 1;
         if (oddNum - k >= 0) {
            res += arr[oddNum - k];
         }
         ++arr[oddNum];
      }
      return res;
   }

   // 974. 和可被 K 整除的子数组 --前缀和
   public int subarraysDivByK(int[] nums, int k) {
      Map<Integer, Integer> map = new HashMap<>();
      map.put(0, 1);
      int res = 0;
      int preSum = 0;
      for (int i = 0; i < nums.length; ++i) {
         preSum += nums[i];
         preSum = (preSum % k + k) % k;
         res += map.getOrDefault(preSum, 0);
         map.put(preSum, map.getOrDefault(preSum, 0) + 1);
      }
      return res;

   }

   // 974. 和可被 K 整除的子数组 --前缀和
   public int subarraysDivByK_2(int[] nums, int k) {
      int[] arr = new int[k];
      arr[0] = 1;
      int res = 0;
      int preSum = 0;
      for (int i = 0; i < nums.length; ++i) {
         preSum += nums[i];
         preSum = (preSum % k + k) % k;
         res += arr[preSum];
         ++arr[preSum];
      }
      return res;

   }

   // 1292. 元素和小于等于阈值的正方形的最大边长
   public int maxSideLength(int[][] mat, int threshold) {
      int[][] P = new int[mat.length + 1][mat[0].length + 1];
      for (int i = 1; i < P.length; ++i) {
         for (int j = 1; j < P[0].length; ++j) {
            P[i][j] = P[i - 1][j] + P[i][j - 1] + mat[i - 1][j - 1] - P[i - 1][j - 1];
         }
      }
      int left = 1;
      int ans = 0;
      int right = Math.min(mat.length, mat[0].length);
      while (left <= right) {
         boolean find = false;
         int mid = left + ((right - left) >> 1);
         for (int i = 1; i <= mat.length + 1 - mid; ++i) {
            for (int j = 1; j <= mat[0].length + 1 - mid; ++j) {
               if (getRect(P, i, j, i + mid - 1, j + mid - 1) <= threshold) {
                  find = true;
               }

            }
         }
         if (find) {
            ans = mid;
            left = mid + 1;
         } else {
            right = mid - 1;
         }
      }
      return ans;
   }

   // 1292. 元素和小于等于阈值的正方形的最大边长
   public int maxSideLength2(int[][] mat, int threshold) {
      int m = mat.length;
      int n = mat[0].length;
      int[][] preSum = new int[m + 1][n + 1];
      for (int i = 1; i <= m; ++i) {
         for (int j = 1; j <= n; ++j) {
            preSum[i][j] = preSum[i - 1][j] + preSum[i][j - 1] - preSum[i - 1][j - 1] + mat[i - 1][j - 1];
         }
      }
      int r = Math.min(m, n);
      int res = 0;
      for (int i = 1; i <= m; ++i) {
         for (int j = 1; j <= n; ++j) {
            for (int side = res + 1; side <= r; ++side) {
               if (i + side - 1 <= m && j + side - 1 <= n
                     && getRect(preSum, i, j, i + side - 1, j + side - 1) <= threshold) {
                  ++res;
               } else {
                  break;
               }
            }
         }
      }
      return res;

   }

   private int getRect(int[][] preSum, int startI, int startJ, int endI, int endJ) {
      return preSum[endI][endJ] - preSum[endI][startJ - 1] - preSum[startI - 1][endJ] + preSum[startI - 1][startJ - 1];
   }

   // 1314. 矩阵区域和 (Matrix Block Sum)
   public int[][] matrixBlockSum(int[][] mat, int k) {
      int m = mat.length;
      int n = mat[0].length;
      int[][] prefix = new int[m + 1][n + 1];
      for (int i = 1; i <= m; ++i) {
         for (int j = 1; j <= n; ++j) {
            prefix[i][j] = prefix[i - 1][j] + prefix[i][j - 1] + mat[i - 1][j - 1] - prefix[i - 1][j - 1];
         }
      }
      for (int i = 0; i < m; ++i) {
         for (int j = 0; j < n; ++j) {
            int endI = Math.min(i + k + 1, m);
            int endJ = Math.min(j + k + 1, n);
            int startI = Math.max(1, i + 1 - k);
            int startJ = Math.max(1, j + 1 - k);
            mat[i][j] = getRect1314(prefix, startI, startJ, endI, endJ);
         }
      }
      return mat;

   }

   private int getRect1314(int[][] prefix, int startI, int startJ, int endI, int endJ) {
      return prefix[endI][endJ] - prefix[endI][startJ - 1] - prefix[startI - 1][endJ] + prefix[startI - 1][startJ - 1];
   }

   // 209. 长度最小的子数组 (Minimum Size Subarray Sum) --O(n)
   // 剑指 Offer II 008. 和大于等于 target 的最短子数组 --O(n)
   public int minSubArrayLen(int target, int[] nums) {
      int res = Integer.MAX_VALUE;
      int left = 0;
      int right = 0;
      int curSum = 0;
      while (right < nums.length) {
         curSum += nums[right];
         while (curSum >= target) {
            res = Math.min(res, right - left + 1);
            curSum -= nums[left++];
         }
         ++right;
      }
      return res == Integer.MAX_VALUE ? 0 : res;

   }

   // 209. 长度最小的子数组 (Minimum Size Subarray Sum) --O(nlog(n))
   // 剑指 Offer II 008. 和大于等于 target 的最短子数组 --O(nlog(n))
   public int minSubArrayLen2(int target, int[] nums) {
      int[] prefix = new int[nums.length + 1];
      int res = Integer.MAX_VALUE;
      for (int i = 1; i < prefix.length; ++i) {
         prefix[i] = prefix[i - 1] + nums[i - 1];
      }
      for (int i = 1; i < prefix.length; ++i) {
         int find = target + prefix[i - 1];
         int bound = Arrays.binarySearch(prefix, find);
         if (bound < 0) {
            bound = -bound - 1;
         }
         if (bound <= nums.length) {
            res = Math.min(res, bound - i + 1);
         }
      }
      return res == Integer.MAX_VALUE ? 0 : res;

   }

   // 303. 区域和检索 - 数组不可变 (Range Sum Query - Immutable)
   class NumArray {
      private int[] prefix;

      public NumArray(int[] nums) {
         prefix = new int[nums.length + 1];
         for (int i = 1; i < prefix.length; ++i) {
            prefix[i] = prefix[i - 1] + nums[i - 1];
         }

      }

      public int sumRange(int left, int right) {
         return prefix[right + 1] - prefix[left];
      }
   }

   // 304. 二维区域和检索 - 矩阵不可变 (Range Sum Query 2D - Immutable)
   // 剑指 Offer II 013. 二维子矩阵的和
   class NumMatrix {
      int[][] prefix;

      public NumMatrix(int[][] matrix) {
         prefix = new int[matrix.length + 1][matrix[0].length + 1];
         for (int i = 1; i < prefix.length; ++i) {
            for (int j = 1; j < prefix[0].length; ++j) {
               prefix[i][j] = prefix[i - 1][j] + prefix[i][j - 1] + matrix[i - 1][j - 1] - prefix[i - 1][j - 1];
            }
         }
      }

      public int sumRegion(int row1, int col1, int row2, int col2) {
         return prefix[row2 + 1][col2 + 1] - prefix[row2 + 1][col1] - prefix[row1][col2 + 1] + prefix[row1][col1];
      }
   }

   // 238. 除自身以外数组的乘积
   // 剑指 Offer 66. 构建乘积数组
   public int[] constructArr(int[] a) {
      int[] res = new int[a.length];
      int k = 1;
      for (int i = 0; i < res.length; ++i) {
         res[i] = k;
         k *= a[i];
      }
      k = 1;
      for (int i = res.length - 1; i >= 0; --i) {
         res[i] *= k;
         k *= a[i];
      }
      return res;

   }

   // 2024. 考试的最大困扰度 (Maximize the Confusion of an Exam) --滑动窗口
   public int maxConsecutiveAnswers(String answerKey, int k) {
      char[] keys = answerKey.toCharArray();
      int left = 0;
      int right = 0;
      int res = 0;
      int tCounts = 0;
      int fCounts = 0;
      while (right < keys.length) {
         if (keys[right] == 'T') {
            ++tCounts;
         } else {
            ++fCounts;
         }
         while (left < keys.length && tCounts > k && fCounts > k) {
            if (keys[left++] == 'T') {
               --tCounts;
            } else {
               --fCounts;
            }
         }
         res = Math.max(res, right - left + 1);
         ++right;
      }
      return res;

   }

   // 1004.最大连续1的个数 III (Max Consecutive Ones III)
   public int longestOnes(int[] nums, int k) {
      int res = 0;
      int zeroCount = 0;
      int left = 0;
      int right = 0;
      while (right < nums.length) {
         if (nums[right] == 0) {
            ++zeroCount;
         }
         while (left < nums.length && zeroCount > k) {
            if (nums[left++] == 0) {
               --zeroCount;
            }
         }
         res = Math.max(res, right - left + 1);
         ++right;
      }
      return res;

   }

   // 1930. 长度为 3 的不同回文子序列 (Unique Length-3 Palindromic Subsequences)
   public int countPalindromicSubsequence(String s) {
      int res = 0;
      char[] chars = s.toCharArray();
      for (char c = 'a'; c <= 'z'; ++c) {
         int left = 0;
         int right = chars.length - 1;
         while (left < right && chars[left] != c) {
            ++left;
         }
         while (left < right && chars[right] != c) {
            --right;
         }
         if (right - left < 2) {
            continue;
         }
         Set<Character> set = new HashSet<>();
         for (int i = left + 1; i <= right - 1; ++i) {
            set.add(chars[i]);
         }
         res += set.size();
      }
      return res;

   }

   // 1930. 长度为 3 的不同回文子序列 (Unique Length-3 Palindromic Subsequences)
   public int countPalindromicSubsequence2(String s) {
      int res = 0;
      int n = s.length();
      char[] chars = s.toCharArray();
      int[] leftPrefix = new int[n];
      int[] rightPrefix = new int[n];
      for (int i = 1; i < n; ++i) {
         leftPrefix[i] = leftPrefix[i - 1] | (1 << (chars[i - 1] - 'a'));
      }
      for (int i = n - 2; i >= 0; --i) {
         rightPrefix[i] = rightPrefix[i + 1] | (1 << (chars[i + 1] - 'a'));
      }
      int[] ans = new int[26];
      for (int i = 1; i < n - 1; ++i) {
         ans[chars[i] - 'a'] |= leftPrefix[i] & rightPrefix[i];
      }
      for (int i = 0; i < ans.length; ++i) {
         res += Integer.bitCount(ans[i]);
      }
      return res;

   }

   // 1074. 元素和为目标值的子矩阵数量 (Number of Submatrices That Sum to Target)
   public int numSubmatrixSumTarget(int[][] matrix, int target) {
      int res = 0;
      int m = matrix.length;
      int n = matrix[0].length;
      for (int i = 0; i < m; ++i) {
         int[] prefix = new int[n];
         for (int j = i; j < m; ++j) {
            for (int c = 0; c < n; ++c) {
               prefix[c] += matrix[j][c];
            }
            res += getSum1074(prefix, target);
         }
      }
      return res;

   }

   private int getSum1074(int[] nums, int target) {
      Map<Integer, Integer> map = new HashMap<>();
      map.put(0, 1);
      int prefix = 0;
      int res = 0;
      for (int i = 0; i < nums.length; ++i) {
         prefix += nums[i];
         res += map.getOrDefault(prefix - target, 0);
         map.put(prefix, map.getOrDefault(prefix, 0) + 1);
      }
      return res;
   }

   // 1915. 最美子字符串的数目 (Number of Wonderful Substrings)
   public long wonderfulSubstrings(String word) {
      int[] state = new int[1 << 10];
      state[0] = 1;
      long res = 0;
      int mask = 0;
      for (int i = 0; i < word.length(); ++i) {
         mask ^= 1 << (word.charAt(i) - 'a');
         res += state[mask];
         for (int j = 0; j < 10; ++j) {
            res += state[mask ^ (1 << j)];
         }
         ++state[mask];
      }
      return res;
   }

   // 1838. 最高频元素的频数 (Frequency of the Most Frequent Element)
   public int maxFrequency(int[] nums, int k) {
      Arrays.sort(nums);
      int left = 0;
      int right = 1;
      long sum = 0;
      int res = 1;
      while (right < nums.length) {
         sum += (nums[right] - nums[right - 1]) * (right - left);
         while (sum > k) {
            sum -= nums[right] - nums[left];
            ++left;
         }
         res = Math.max(res, right - left + 1);
         ++right;
      }
      return res;

   }

   // 1546. 和为目标值且不重叠的非空子数组的最大数目 (Maximum Number of Non-Overlapping Subarrays With
   // Sum Equals Target)
   public int maxNonOverlapping(int[] nums, int target) {
      Set<Integer> set = new HashSet<>();
      set.add(0);
      int res = 0;
      int prefix = 0;
      for (int i = 0; i < nums.length; ++i) {
         prefix += nums[i];
         if (set.contains(prefix - target)) {
            ++res;
            set.clear();
            set.add(0);
            prefix = 0;
         } else {
            set.add(prefix);
         }
      }
      return res;

   }

   // 1865. 找出和为指定值的下标对 (Finding Pairs With a Certain Sum)
   class FindSumPairs {
      private int[] nums1;
      private int[] nums2;
      private Map<Integer, Integer> map;

      public FindSumPairs(int[] nums1, int[] nums2) {
         this.nums1 = nums1;
         this.nums2 = nums2;
         this.map = new HashMap<>();
         for (int num : nums2) {
            map.put(num, map.getOrDefault(num, 0) + 1);
         }
      }

      public void add(int index, int val) {
         map.put(nums2[index], map.get(nums2[index]) - 1);
         nums2[index] += val;
         map.put(nums2[index], map.getOrDefault(nums2[index], 0) + 1);
      }

      public int count(int tot) {
         int res = 0;
         for (int num : nums1) {
            res += map.getOrDefault(tot - num, 0);
         }
         return res;
      }
   }

   // 1456. 定长子串中元音的最大数目 (Maximum Number of Vowels in a Substring of Given Length)
   // --滑动窗口
   public int maxVowels(String s, int k) {
      char[] chars = s.toCharArray();
      int cur = 0;
      int i = 0;
      while (i < k) {
         if (chars[i] == 'a' || chars[i] == 'e' || chars[i] == 'i' || chars[i] == 'o' || chars[i] == 'u') {
            ++cur;
         }
         ++i;
      }
      int res = cur;
      while (i < chars.length) {
         if (chars[i] == 'a' || chars[i] == 'e' || chars[i] == 'i' || chars[i] == 'o' || chars[i] == 'u') {
            ++cur;
         }
         if (chars[i - k] == 'a' || chars[i - k] == 'e' || chars[i - k] == 'i' || chars[i - k] == 'o'
               || chars[i - k] == 'u') {
            --cur;
         }
         res = Math.max(res, cur);
         ++i;
      }
      return res;

   }

   // 1974. 使用特殊打字机键入单词的最少时间 (Minimum Time to Type Word Using Special Typewriter)
   // --贪心
   public int minTimeToType(String word) {
      char[] chars = word.toCharArray();
      int res = Math.min(chars[0] - 'a', 26 - (chars[0] - 'a'));
      for (int i = 1; i < chars.length; ++i) {
         char x = chars[i - 1];
         char y = chars[i];
         int abs = Math.abs(x - y);
         res += Math.min(abs, 26 - abs);
      }
      return res + word.length();

   }

   // 1358. 包含所有三种字符的子字符串数目 (Number of Substrings Containing All Three Characters)
   public int numberOfSubstrings(String s) {
      int res = 0;
      char[] chars = s.toCharArray();
      int[] counts = new int[3];
      int left = 0;
      for (int right = 0; right < chars.length; ++right) {
         ++counts[chars[right] - 'a'];
         while (counts[0] > 0 && counts[1] > 0 && counts[2] > 0) {
            res += chars.length - right;
            --counts[chars[left++] - 'a'];
         }
      }
      return res;

   }

   // 1297. 子串的最大出现次数 (Maximum Number of Occurrences of a Substring) --滑动窗口
   // 还需掌握滚动哈希
   public int maxFreq(String s, int maxLetters, int minSize, int maxSize) {
      Map<String, Integer> map = new HashMap<>();
      for (int i = 0; i < s.length() - minSize + 1; ++i) {
         String sub = s.substring(i, i + minSize);
         if (isLegal1297(sub, maxLetters)) {
            map.put(sub, map.getOrDefault(sub, 0) + 1);
         }
      }
      if (map.isEmpty()) {
         return 0;
      }
      return Collections.max(map.values());

   }

   private boolean isLegal1297(String s, int maxLetters) {
      Set<Character> set = new HashSet<>();
      for (char c : s.toCharArray()) {
         set.add(c);
         if (set.size() > maxLetters) {
            return false;
         }
      }
      return true;
   }

   // 1218. 最长定差子序列 (Longest Arithmetic Subsequence of Given Difference) --动态规划
   public int longestSubsequence(int[] arr, int difference) {
      Map<Integer, Integer> map = new HashMap<>();
      for (int num : arr) {
         map.put(num, map.getOrDefault(num - difference, 0) + 1);
      }
      return Collections.max(map.values());

   }

   // 85. 最大矩形 (Maximal Rectangle)
   // 剑指 Offer II 040. 矩阵中最大的矩形
   public int maximalRectangle(String[] matrix) {
      int res = 0;
      if (matrix == null || matrix.length == 0 || matrix[0].length() == 0) {
         return res;
      }
      int[] dp = new int[matrix[0].length()];
      for (String row : matrix) {
         char[] chars = row.toCharArray();
         for (int i = 0; i < chars.length; ++i) {
            if (chars[i] - '0' == 1) {
               ++dp[i];
            } else {
               dp[i] = 0;
            }
         }
         res = Math.max(res, getMax040(dp));
      }
      return res;

   }

   private int getMax040(int[] heights) {
      Stack<Integer> stack = new Stack<>();
      stack.push(-1);
      int res = 0;
      for (int i = 0; i < heights.length; ++i) {
         while (stack.peek() != -1 && heights[i] < heights[stack.peek()]) {
            int h = heights[stack.pop()];
            res = Math.max(res, h * (i - stack.peek() - 1));
         }
         stack.push(i);
      }
      while (stack.peek() != -1) {
         int h = heights[stack.pop()];
         res = Math.max(res, h * (heights.length - stack.peek() - 1));
      }
      return res;
   }

   // 84. 柱状图中最大的矩形 (Largest Rectangle in Histogram)
   // 剑指 Offer II 039. 直方图最大矩形面积 --单调栈
   public int largestRectangleArea(int[] heights) {
      Stack<Integer> stack = new Stack<>();
      stack.push(-1);
      int res = 0;
      for (int i = 0; i < heights.length; ++i) {
         while (stack.peek() != -1 && heights[i] < heights[stack.peek()]) {
            int h = heights[stack.pop()];
            res = Math.max(res, h * (i - stack.peek() - 1));
         }
         stack.push(i);
      }
      while (stack.peek() != -1) {
         int h = heights[stack.pop()];
         res = Math.max(res, h * (heights.length - stack.peek() - 1));
      }
      return res;

   }

   // 84. 柱状图中最大的矩形 (Largest Rectangle in Histogram)
   // 剑指 Offer II 039. 直方图最大矩形面积 --分治
   public int largestRectangleArea2(int[] heights) {
      if (heights == null) {
         return 0;
      }
      return calculateArea(heights, 0, heights.length - 1);

   }

   private int calculateArea(int[] heights, int start, int end) {
      if (end < start) {
         return 0;
      }
      int minI = start;
      for (int i = start; i <= end; ++i) {
         if (heights[i] < heights[minI]) {
            minI = i;
         }
      }
      return Math.max(heights[minI] * (end - start + 1),
            Math.max(calculateArea(heights, start, minI - 1), calculateArea(heights, minI + 1, end)));

   }

   // TODO
   // 523. 连续的子数组和 (Continuous Subarray Sum)
   // public boolean checkSubarraySum(int[] nums, int k) {

   // }

}