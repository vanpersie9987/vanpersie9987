import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
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

   // 1138. 字母板上的路径 (Alphabet Board Path)
   public String alphabetBoardPath(String target) {
      StringBuilder res = new StringBuilder();
      int preX = 0;
      int preY = 0;
      for (char c : target.toCharArray()) {
         int curX = (c - 'a') / 5;
         int curY = (c - 'a') % 5;
         if (curX < preX) {
            int count = preX - curX;
            while (count-- > 0) {
               res.append('U');
            }
         }
         if (curY < preY) {
            int count = preY - curY;
            while (count-- > 0) {
               res.append('L');
            }
         }
         if (curX > preX) {
            int count = curX - preX;
            while (count-- > 0) {
               res.append('D');
            }
         }
         if (curY > preY) {
            int count = curY - preY;
            while (count-- > 0) {
               res.append('R');
            }
         }
         res.append('!');
         preX = curX;
         preY = curY;
      }
      return res.toString();

   }

   // 1402. 做菜顺序 (Reducing Dishes)
   public int maxSatisfaction(int[] satisfaction) {
      int preSum = 0;
      int res = 0;
      Arrays.sort(satisfaction);
      for (int i = satisfaction.length - 1; i >= 0; --i) {
         preSum += satisfaction[i];
         if (preSum > 0) {
            res += preSum;
         } else {
            break;
         }
      }
      return res;

   }

   // 1605. 给定行和列的和求可行矩阵 (Find Valid Matrix Given Row and Column Sums)
   public int[][] restoreMatrix(int[] rowSum, int[] colSum) {
      int m = rowSum.length;
      int n = colSum.length;
      int[][] res = new int[m][n];
      for (int i = 0; i < m; ++i) {
         for (int j = 0; j < n; ++j) {
            res[i][j] = Math.min(rowSum[i], colSum[j]);
            rowSum[i] -= res[i][j];
            colSum[j] -= res[i][j];
         }
      }
      return res;

   }

   // 1710. 卡车上的最大单元数 (Maximum Units on a Truck)
   public int maximumUnits(int[][] boxTypes, int truckSize) {
      Arrays.sort(boxTypes, (o1, o2) -> o2[1] - o1[1]);
      int res = 0;
      for (int[] boxType : boxTypes) {
         if (truckSize <= 0) {
            break;
         }
         int min = Math.min(truckSize, boxType[0]);
         res += boxType[1] * min;
         truckSize -= min;
      }
      return res;

   }

   // 面试题 08.11. 硬币 (Coin LCCI)
   public int waysToChange(int n) {
      final int MOD = 1000000007;
      int[] dp = new int[n + 1];
      dp[0] = 1;
      int[] coins = { 1, 5, 10, 25 };
      for (int coin : coins) {
         for (int i = coin; i <= n; ++i) {
            dp[i] = (dp[i] + dp[i - coin]) % MOD;
         }
      }
      return dp[dp.length - 1];

   }

   // 518. 零钱兑换 II (Coin Change 2)
   public int change(int amount, int[] coins) {
      int[] dp = new int[amount + 1];
      dp[0] = 1;
      for (int coin : coins) {
         for (int i = coin; i <= amount; ++i) {
            dp[i] += dp[i - coin];
         }
      }
      return dp[dp.length - 1];

   }

   // 322. 零钱兑换 (Coin Change) --动态规划
   // 剑指 Offer II 103. 最少的硬币数目
   public int coinChange(int[] coins, int amount) {
      int[] dp = new int[amount + 1];
      Arrays.fill(dp, amount + 1);
      dp[0] = 0;
      for (int i = 1; i < dp.length; ++i) {
         for (int coin : coins) {
            if (i - coin >= 0) {
               dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
         }
      }
      return dp[dp.length - 1] == amount + 1 ? -1 : dp[dp.length - 1];

   }

   // 520. 检测大写字母 (Detect Capital)
   public boolean detectCapitalUse(String word) {
      return allUpperCases(word) || allLowerCases(word) || onlyLeadingCharUpperCase(word);
   }

   private boolean onlyLeadingCharUpperCase(String word) {
      char leadingChar = word.charAt(0);
      if (Character.isLowerCase(leadingChar)) {
         return false;
      }
      for (int i = 1; i < word.length(); ++i) {
         if (Character.isUpperCase(word.charAt(i))) {
            return false;
         }
      }
      return true;
   }

   private boolean allLowerCases(String word) {
      for (char c : word.toCharArray()) {
         if (Character.isUpperCase(c)) {
            return false;
         }
      }
      return true;
   }

   private boolean allUpperCases(String word) {
      for (char c : word.toCharArray()) {
         if (Character.isLowerCase(c)) {
            return false;
         }
      }
      return true;
   }

   // 677. 键值映射 (Map Sum Pairs) --还需掌握字典树
   class MapSum {
      private Map<String, Integer> map;
      private Map<String, Integer> prefixMap;

      public MapSum() {
         map = new HashMap<>();
         prefixMap = new HashMap<>();

      }

      public void insert(String key, int val) {
         int delta = val - map.getOrDefault(key, 0);
         map.put(key, val);
         for (int i = 1; i <= key.length(); ++i) {
            String prefix = key.substring(0, i);
            prefixMap.put(prefix, prefixMap.getOrDefault(prefix, 0) + delta);
         }
      }

      public int sum(String prefix) {
         return prefixMap.get(prefix);
      }
   }

   // 5898. 数组中第 K 个独一无二的字符串 (Kth Distinct String in an Array)
   public String kthDistinct(String[] arr, int k) {
      Map<String, Integer> map = new HashMap<>();
      for (String s : arr) {
         map.put(s, map.getOrDefault(s, 0) + 1);
      }
      for (String s : arr) {
         if (map.get(s) == 1) {
            if (--k == 0) {
               return s;
            }
         }
      }
      return "";

   }

   // 2062. 统计字符串中的元音子字符串 (Count Vowel Substrings of a String)
   public int countVowelSubstrings(String word) {
      int res = 0;
      if (word.length() < 5) {
         return res;
      }
      char[] chars = word.toCharArray();
      Set<Character> set = new HashSet<>();
      for (int i = 0; i < chars.length; ++i) {
         set.clear();
         for (int j = i; j < chars.length; ++j) {
            if (!isVowel(chars[j])) {
               break;
            }
            set.add(chars[j]);
            if (set.size() == 5) {
               ++res;
            }
         }
      }
      return res;

   }

   private boolean isVowel(char c) {
      return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
   }

   // LCP 39. 无人机方阵
   public int minimumSwitchingTimes(int[][] source, int[][] target) {
      int[] counts = new int[10001];
      for (int i = 0; i < source.length; ++i) {
         for (int j = 0; j < source[0].length; ++j) {
            ++counts[source[i][j]];
            --counts[target[i][j]];
         }
      }
      int res = 0;
      for (int count : counts) {
         res += Math.abs(count);
      }
      return res >> 1;

   }

   public class ListNode {
      int val;
      ListNode next;

      ListNode() {
      }

      ListNode(int val) {
         this.val = val;
      }

      ListNode(int val, ListNode next) {
         this.val = val;
         this.next = next;
      }
   }

   // 206. 反转链表 (Reverse Linked List)
   // 剑指 Offer II 024. 反转链表
   // 剑指 Offer 24. 反转链表
   public ListNode reverseList(ListNode head) {
      ListNode prev = null;
      ListNode curr = head;
      while (curr != null) {
         ListNode temp = curr.next;
         curr.next = prev;
         prev = curr;
         curr = temp;
      }
      return prev;

   }

   // 92. 反转链表 II (Reverse Linked List II)
   public ListNode reverseBetween(ListNode head, int left, int right) {
      ListNode dummy = new ListNode(0, head);
      ListNode pre = dummy;
      for (int i = 0; i < left - 1; ++i) {
         pre = pre.next;
      }
      ListNode subTail = pre;
      for (int i = 0; i < right - left + 1; ++i) {
         subTail = subTail.next;
      }
      ListNode succ = subTail.next;
      ListNode subHead = pre.next;

      pre.next = null;
      subTail.next = null;

      reverseLinkedList(subHead);

      pre.next = subTail;
      subHead.next = succ;

      return dummy.next;

   }

   private void reverseLinkedList(ListNode head) {
      ListNode pre = null;
      ListNode cur = head;
      while (cur != null) {
         ListNode temp = cur.next;
         cur.next = pre;
         pre = cur;
         cur = temp;
      }
   }

   // 92. 反转链表 II (Reverse Linked List II)
   public ListNode reverseBetween2(ListNode head, int left, int right) {
      ListNode dummy = new ListNode(0, head);
      ListNode guard = dummy;

      for (int i = 0; i < left - 1; ++i) {
         guard = guard.next;
      }
      ListNode point = guard.next;
      for (int i = 0; i < right - left; ++i) {
         ListNode removed = point.next;
         point.next = point.next.next;

         removed.next = guard.next;
         guard.next = removed;
      }
      return dummy.next;

   }

   // 328. 奇偶链表 (Odd Even Linked List)
   public ListNode oddEvenList(ListNode head) {
      if (head == null) {
         return head;
      }
      ListNode odd = head;
      ListNode evenHead = head.next;
      ListNode even = evenHead;
      while (even != null && even.next != null) {
         odd.next = even.next;
         odd = odd.next;
         even.next = odd.next;
         even = even.next;
      }
      odd.next = evenHead;
      return head;

   }

   // 725. 分隔链表 (Split Linked List in Parts)
   public ListNode[] splitListToParts(ListNode head, int k) {
      ListNode[] res = new ListNode[k];

      ListNode temp = head;
      int count = 0;
      while (temp != null) {
         temp = temp.next;
         ++count;
      }

      ListNode cur = head;

      int q = count / k;
      int r = count % k;

      for (int i = 0; i < k && cur != null; ++i) {
         res[i] = cur;
         int partSize = q + (i < r ? 1 : 0);
         for (int j = 0; j < partSize - 1; ++j) {
            cur = cur.next;
         }
         ListNode temp1 = cur.next;
         cur.next = null;
         cur = temp1;
      }
      return res;

   }

   // 2058. 找出临界点之间的最小和最大距离
   // 2058.Find the Minimum and Maximum Number of Nodes Between Critical Points
   public int[] nodesBetweenCriticalPoints(ListNode head) {
      if (head == null || head.next == null) {
         return new int[] { -1, -1 };
      }
      ListNode pre = head;
      head = head.next;
      int min = Integer.MAX_VALUE;
      int max = Integer.MIN_VALUE;
      int lastCriticalIndex = -1;
      int firstCriticalIndex = -1;
      int index = 1;

      while (head != null && head.next != null) {
         if ((pre.val > head.val && head.val < head.next.val) || (pre.val < head.val && head.val > head.next.val)) {
            if (lastCriticalIndex == -1) {
               firstCriticalIndex = index;
            } else {
               min = Math.min(min, index - lastCriticalIndex);
               max = Math.max(max, index - firstCriticalIndex);
            }
            lastCriticalIndex = index;
         }
         ++index;
         pre = pre.next;
         head = head.next;
      }
      return min == Integer.MAX_VALUE ? new int[] { -1, -1 } : new int[] { min, max };

   }

   // 1670. 设计前中后队列 (Design Front Middle Back Queue)
   class FrontMiddleBackQueue {
      private List<Integer> queue;

      public FrontMiddleBackQueue() {
         queue = new LinkedList<>();

      }

      public void pushFront(int val) {
         queue.add(0, val);
      }

      public void pushMiddle(int val) {
         int insertIndex = queue.size() >> 1;
         queue.add(insertIndex, val);
      }

      public void pushBack(int val) {
         queue.add(val);
      }

      public int popFront() {
         if (queue.isEmpty()) {
            return -1;
         }
         return queue.remove(0);
      }

      public int popMiddle() {
         if (queue.isEmpty()) {
            return -1;
         }
         int deleteIndex = (queue.size() - 1) >> 1;
         return queue.remove(deleteIndex);
      }

      public int popBack() {
         if (queue.isEmpty()) {
            return -1;
         }
         return queue.remove(queue.size() - 1);
      }
   }

   // 641. 设计循环双端队列 (Design Circular Deque)
   class MyCircularDeque {
      private int[] arr;
      private int headIndex;
      private int lastIndex;
      private int size;
      private int k;

      public MyCircularDeque(int k) {
         arr = new int[k];
         this.k = k;
      }

      public boolean insertFront(int value) {
         if (isFull()) {
            return false;
         }
         headIndex = (headIndex - 1 + k) % k;
         arr[headIndex] = value;
         ++size;
         return true;
      }

      public boolean insertLast(int value) {
         if (size == k) {
            return false;
         }
         arr[lastIndex] = value;
         lastIndex = (lastIndex + 1) % k;
         ++size;
         return true;
      }

      public boolean deleteFront() {
         if (isEmpty()) {
            return false;
         }
         headIndex = (headIndex + 1) % k;
         --size;
         return true;
      }

      public boolean deleteLast() {
         if (isEmpty()) {
            return false;
         }
         lastIndex = (lastIndex - 1 + k) % k;
         --size;
         return true;
      }

      public int getFront() {
         if (isEmpty()) {
            return -1;
         }
         return arr[headIndex];

      }

      public int getRear() {
         if (isEmpty()) {
            return -1;
         }
         return arr[(lastIndex - 1 + k) % k];
      }

      public boolean isEmpty() {
         return size == 0;

      }

      public boolean isFull() {
         return size == k;
      }
   }

   // 622. 设计循环队列 (Design Circular Queue) --follow up ：线程安全？单链表？
   class MyCircularQueue {
      private int[] arr;
      private int headIndex;
      private int size;
      private int k;

      public MyCircularQueue(int k) {
         this.k = k;
         this.arr = new int[k];

      }

      public boolean enQueue(int value) {
         if (isFull()) {
            return false;
         }
         ++size;
         arr[(headIndex + size - 1) % k] = value;
         return true;
      }

      public boolean deQueue() {
         if (isEmpty()) {
            return false;
         }
         --size;
         headIndex = (headIndex + 1) % k;
         return true;
      }

      public int Front() {
         if (isEmpty()) {
            return -1;
         }
         return arr[headIndex];

      }

      public int Rear() {
         if (isEmpty()) {
            return -1;
         }
         return arr[(headIndex + size - 1) % k];
      }

      public boolean isEmpty() {
         return size == 0;

      }

      public boolean isFull() {
         return size == k;

      }
   }

   // 707. 设计链表 (Design Linked List) --单链表
   class MyLinkedList {
      class Node {
         Node next;
         int val;

         Node() {

         }

         Node(int val) {
            this.val = val;
         }

      }

      private Node head;
      private int size;

      public MyLinkedList() {
         head = new Node();

      }

      public int get(int index) {
         if (index < 0 || index >= size) {
            return -1;
         }
         Node cur = head;
         for (int i = 0; i <= index; ++i) {
            cur = cur.next;
         }
         return cur.val;
      }

      public void addAtHead(int val) {
         addAtIndex(0, val);
      }

      public void addAtTail(int val) {
         addAtIndex(size, val);
      }

      public void addAtIndex(int index, int val) {
         if (index < 0 || index > size) {
            return;
         }
         Node add = new Node(val);
         Node cur = head;
         for (int i = 0; i < index; ++i) {
            cur = cur.next;
         }
         add.next = cur.next;
         cur.next = add;
         ++size;
      }

      public void deleteAtIndex(int index) {
         if (index < 0 || index >= size) {
            return;
         }
         Node cur = head;
         for (int i = 0; i < index; ++i) {
            cur = cur.next;
         }
         cur.next = cur.next.next;
         --size;
      }
   }

   // 707. 设计链表 (Design Linked List) --双链表
   class MyLinkedList2 {
      class Node {
         Node next;
         Node prev;
         int val;

         Node() {

         }

         Node(int val) {
            this.val = val;
         }

      }

      private Node head;
      private Node tail;
      private int size;

      public MyLinkedList2() {
         head = new Node();
         tail = new Node();
         head.next = tail;
         tail.prev = head;
      }

      public int get(int index) {
         if (index < 0 || index >= size) {
            return -1;
         }
         if (index + 1 < size - index) {
            Node cur = head;
            for (int i = 0; i < index + 1; ++i) {
               cur = cur.next;
            }
            return cur.val;
         } else {
            Node cur = tail;
            for (int i = 0; i < size - index; ++i) {
               cur = cur.prev;
            }
            return cur.val;
         }
      }

      public void addAtHead(int val) {
         addAtIndex(0, val);
      }

      public void addAtTail(int val) {
         addAtIndex(size, val);
      }

      public void addAtIndex(int index, int val) {
         if (index < 0 || index > size) {
            return;
         }
         Node add = new Node(val);
         if (index < size - index) {
            Node cur = head;
            for (int i = 0; i < index; ++i) {
               cur = cur.next;
            }
            add.next = cur.next;
            cur.next = add;

            add.next.prev = add;
            add.prev = cur;
         } else {
            Node cur = tail;
            for (int i = 0; i < size - index; ++i) {
               cur = cur.prev;
            }
            add.prev = cur.prev;
            cur.prev = add;

            add.prev.next = add;
            add.next = cur;

         }
         ++size;
      }

      public void deleteAtIndex(int index) {
         if (index < 0 || index >= size) {
            return;
         }
         if (index + 1 < size - index) {
            Node cur = head;
            for (int i = 0; i < index; ++i) {
               cur = cur.next;
            }
            cur.next = cur.next.next;
            cur.next.prev = cur;
         } else {
            Node cur = tail;
            for (int i = 0; i < size - index - 1; ++i) {
               cur = cur.prev;
            }
            cur.prev = cur.prev.prev;
            cur.prev.next = cur;
         }
         --size;
      }
   }

   // 2074. 反转偶数长度组的节点 (Reverse Nodes in Even Length Groups)
   public ListNode reverseEvenLengthGroups(ListNode head) {
      ListNode dummy = new ListNode(0, head);
      // 前驱节点
      ListNode pre = dummy;
      // 当前节点
      ListNode cur = head;
      // 当前子链表数量
      int count = 0;
      while (cur != null) {
         // 当前子链表本应该具有的长度
         ++count;
         // 试探当前子链表的长度
         ListNode tryIt = cur;
         int length = 0;
         // 应该具有的长度 == 实际长度 或者 链表到头时 ，终止遍历
         while (length < count && tryIt != null) {
            tryIt = tryIt.next;
            ++length;
         }
         // 实际子链表长度为偶数
         if (length % 2 == 0) {
            for (int i = 0; i < length - 1; ++i) {
               // 把当前cur节点的next节点删除
               ListNode removed = cur.next;
               cur.next = cur.next.next;
               // 把删除的节点添加到前驱节点之后
               removed.next = pre.next;
               pre.next = removed;
            }
            // 此时 cur节点指向下一组子链表的前驱 所以该位置为pre的指向
            pre = cur;
            // 再把cur后移一个节点，指向下一组子链表的表头
            cur = cur.next;
         }

         // 长度为奇数
         else {
            // 将pre和cur分别后移length次
            for (int i = 0; i < length; ++i) {
               cur = cur.next;
               pre = pre.next;
            }
         }
      }
      return dummy.next;

   }

   // 2057. 值相等的最小索引 (Smallest Index With Equal Value)
   public int smallestEqual(int[] nums) {
      for (int i = 0; i < nums.length; ++i) {
         if (i % 10 == nums[i]) {
            return i;
         }
      }
      return -1;

   }

   // 1880. 检查某单词是否等于两单词之和 (Check if Word Equals Summation of Two Words)
   public boolean isSumEqual(String firstWord, String secondWord, String targetWord) {
      int num1 = transferToNum(firstWord);
      int num2 = transferToNum(secondWord);
      int num3 = transferToNum(targetWord);
      return num1 + num2 == num3;

   }

   private int transferToNum(String s) {
      StringBuilder builder = new StringBuilder();
      for (char c : s.toCharArray()) {
         builder.append(c - 'a');
      }
      return Integer.parseInt(builder.toString());
   }

   // 1859. 将句子排序 (Sorting the Sentence)
   public String sortSentence(String s) {
      String[] strings = s.split("\\s+");
      Arrays.sort(strings, new Comparator<String>() {

         @Override
         public int compare(String o1, String o2) {
            int index1 = Integer.parseInt(o1.substring(o1.length() - 1));
            int index2 = Integer.parseInt(o2.substring(o2.length() - 1));
            return index1 - index2;
         }

      });

      StringBuilder res = new StringBuilder();
      for (String string : strings) {
         res.append(string.subSequence(0, string.length() - 1)).append(" ");
      }
      return res.toString().trim();

   }

   // 1859. 将句子排序 (Sorting the Sentence)
   public String sortSentence2(String s) {
      String[] arr = new String[9];
      int lastStartPos = 0;
      int count = 0;
      for (int i = 0; i < s.length(); ++i) {
         char c = s.charAt(i);
         if (Character.isDigit(c)) {
            arr[c - '0' - 1] = s.substring(lastStartPos, i);
            ++count;
         } else if (Character.isWhitespace(c)) {
            lastStartPos = i + 1;
         }
      }
      StringBuilder res = new StringBuilder(arr[0]);
      for (int i = 1; i < count; ++i) {
         res.append(" ").append(arr[i]);
      }
      return res.toString();

   }

   // 2068. 检查两个字符串是否几乎相等 (Check Whether Two Strings are Almost Equivalent)
   public boolean checkAlmostEquivalent(String word1, String word2) {
      int[] counts = new int[26];
      for (int i = 0; i < word1.length(); ++i) {
         ++counts[word1.charAt(i) - 'a'];
         --counts[word2.charAt(i) - 'a'];
      }
      for (int count : counts) {
         if (Math.abs(count) > 3) {
            return false;
         }
      }
      return true;

   }

   // 2043. 简易银行系统 (Simple Bank System)
   class Bank {
      private long[] balance;
      private int n;

      public Bank(long[] balance) {
         this.balance = balance;
         this.n = balance.length;

      }

      public boolean transfer(int account1, int account2, long money) {
         if (account1 <= 0 || account1 > n || account2 <= 0 || account2 > n) {
            return false;
         }
         long res = balance[account1 - 1] - money;
         if (res < 0) {
            return false;
         }
         balance[account1 - 1] = res;
         balance[account2 - 1] += money;
         return true;

      }

      public boolean deposit(int account, long money) {
         if (account <= 0 || account > n) {
            return false;
         }
         balance[account - 1] += money;
         return true;

      }

      public boolean withdraw(int account, long money) {
         if (account <= 0 || account > n) {
            return false;
         }
         long res = balance[account - 1] - money;
         if (res < 0) {
            return false;
         }
         balance[account - 1] = res;
         return true;
      }
   }

   // 剑指 Offer II 074. 合并区间
   // 56. 合并区间
   public int[][] merge(int[][] intervals) {
      Arrays.sort(intervals, (o1, o2) -> o1[0] - o2[0]);
      List<int[]> res = new ArrayList<>();
      for (int i = 0; i < intervals.length; ++i) {
         int left = intervals[i][0];
         int right = intervals[i][1];
         while (i + 1 < intervals.length && right >= intervals[i + 1][0]) {
            right = Math.max(intervals[i + 1][1], right);
            ++i;
         }
         res.add(new int[] { left, right });
      }
      return res.toArray(new int[0][]);

   }

   // 剑指 Offer 45. 把数组排成最小的数 --(思路与179相同)
   public String minNumber(int[] nums) {
      String[] arr = new String[nums.length];
      for (int i = 0; i < nums.length; ++i) {
         arr[i] = String.valueOf(nums[i]);
      }
      Arrays.sort(arr, (o1, o2) -> (o1 + o2).compareTo(o2 + o1));
      StringBuilder res = new StringBuilder();
      for (String s : arr) {
         res.append(s);
      }
      return res.toString();

   }

   // 179. 最大数 (Largest Number)
   public String largestNumber(int[] nums) {
      String[] strings = new String[nums.length];
      for (int i = 0; i < nums.length; ++i) {
         strings[i] = String.valueOf(nums[i]);
      }
      Arrays.sort(strings, (o1, o2) -> (o2 + o1).compareTo(o1 + o2));
      if ("0".equals(strings[0])) {
         return "0";
      }
      StringBuilder res = new StringBuilder();
      for (int i = 0; i < strings.length; ++i) {
         res.append(strings[i]);
      }

      return res.toString();

   }

   // TODO
   // 523. 连续的子数组和 (Continuous Subarray Sum)
   // public boolean checkSubarraySum(int[] nums, int k) {

   // }

}