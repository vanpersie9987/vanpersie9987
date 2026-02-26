import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@SuppressWarnings("unchecked")
public class LeetCode_2 {

   public static void main(final String[] args) {
      // String s = "4w31am0ets6sl5go5ufytjtjpb7b0sxqbee2blg9ss";
      // int res = numDifferentIntegers(s);
      // int res = maxSumTwoNoOverlap(new int[] { 1, 0, 3 }, 1, 2);
      // boolean[] res = friendRequests(3, new int[][] { { 0, 1 } }, new int[][] { {
      // 0, 2 }, { 2, 1 } });
      // int x = nextBeautifulNumber(1);
      // String sub = subStrHash("xmmhdakfursinye", 96, 45, 15, 21);

   }

   // 2032. 至少在两个数组中出现的值 (Two Out of Three)
   public List<Integer> twoOutOfThree(int[] nums1, int[] nums2, int[] nums3) {
      int[] counts = new int[101];
      getCounts2032(nums1, 1 << 0, counts);
      getCounts2032(nums2, 1 << 1, counts);
      getCounts2032(nums3, 1 << 2, counts);
      List<Integer> res = new ArrayList<>();
      for (int i = 1; i <= 100; ++i) {
         if (Integer.bitCount(counts[i]) >= 2) {
            res.add(i);
         }
      }
      return res;
   }

   private void getCounts2032(int[] nums, int mask, int[] counts) {
      for (int num : nums) {
         counts[num] |= mask;
      }
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

   // 676. 实现一个魔法字典 (Implement Magic Dictionary)
   // 剑指 Offer II 064. 神奇的字典
   class MagicDictionary {
      private Map<Integer, Set<String>> map;

      /** Initialize your data structure here. */
      public MagicDictionary() {
         map = new HashMap<>();

      }

      public void buildDict(String[] dictionary) {
         for (String d : dictionary) {
            map.computeIfAbsent(d.length(), k -> new HashSet<>()).add(d);
         }
      }

      public boolean search(String searchWord) {
         for (String word : map.getOrDefault(searchWord.length(), new HashSet<>())) {
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
   // 剑指 Offer II 070. 排序数组中只出现一次的数字
   public int singleNonDuplicate(int[] nums) {
      int res = 0;
      for (int num : nums) {
         res ^= num;
      }
      return res;

   }

   // 540. 有序数组中的单一元素 (Single Element in a Sorted Array)
   // 剑指 Offer II 070. 排序数组中只出现一次的数字 --二分查找
   public int singleNonDuplicate2(int[] nums) {
      int left = 0;
      int right = nums.length - 1;
      while (left < right) {
         int mid = left + ((right - left) >>> 1);
         if (nums[mid] == nums[mid ^ 1]) {
            left = mid + 1;
         } else {
            right = mid;
         }
      }
      return nums[left];

   }

   // 1248. 统计「优美子数组」(Count Number of Nice Subarrays) --前缀和 还需掌握数学法
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

   // 1248. 统计「优美子数组」(Count Number of Nice Subarrays) --前缀和 还需掌握数学法
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

   // 1930. 长度为 3 的不同回文子序列 (Unique Length-3 Palindromic Subsequences)
   public int countPalindromicSubsequence(String s) {
      int n = s.length();
      char[] arr = s.toCharArray();
      int[] pre = new int[n];

      for (int i = 1; i < n; ++i) {
         pre[i] = pre[i - 1] | (1 << (arr[i - 1] - 'a'));
      }
      int suf = 0;
      int[] a = new int[26];
      for (int i = n - 2; i >= 1; --i) {
         suf |= (1 << (arr[i + 1] - 'a'));
         a[arr[i] - 'a'] |= pre[i] & suf;
      }
      int res = 0;
      for (int x : a) {
         res += Integer.bitCount(x);
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
   public int maximalRectangle(char[][] matrix) {
      int res = 0;
      int m = matrix.length;
      int n = matrix[0].length;
      int[] height = new int[n];
      for (int i = 0; i < m; ++i) {
         for (int j = 0; j < n; ++j) {
            if (matrix[i][j] == '1') {
               ++height[j];
            } else {
               height[j] = 0;
            }
         }
         res = Math.max(res, cal85(height));
      }
      return res;

   }

   private int cal85(int[] heights) {
      Stack<Integer> stack = new Stack<>();
      int n = heights.length;
      stack.push(-1);
      int res = 0;
      int[] left = new int[n];
      Arrays.fill(left, -1);
      for (int i = 0; i < n; ++i) {
         while (stack.peek() != -1 && heights[stack.peek()] >= heights[i]) {
            stack.pop();
         }
         left[i] = stack.peek();
         stack.push(i);
      }
      stack.clear();
      stack.push(n);
      int[] right = new int[n];
      Arrays.fill(right, n);
      for (int i = n - 1; i >= 0; --i) {
         while (stack.peek() != n && heights[stack.peek()] >= heights[i]) {
            stack.pop();
         }
         right[i] = stack.peek();
         stack.push(i);
      }
      for (int i = 0; i < n; ++i) {
         res = Math.max(res, heights[i] * (right[i] - left[i] - 1));
      }
      return res;
   }

   // 84. 柱状图中最大的矩形 (Largest Rectangle in Histogram)
   // 剑指 Offer II 039. 直方图最大矩形面积 --单调栈
   public int largestRectangleArea(int[] heights) {
      int n = heights.length;
      int[] left = new int[n];
      Arrays.fill(left, -1);
      Stack<Integer> st = new Stack<>();
      for (int i = 0; i < n; ++i) {
         while (!st.isEmpty() && heights[st.peek()] >= heights[i]) {
            st.pop();
         }
         if (!st.isEmpty()) {
            left[i] = st.peek();
         }
         st.push(i);
      }

      int[] right = new int[n];
      Arrays.fill(right, n);
      st.clear();
      for (int i = n - 1; i >= 0; --i) {
         while (!st.isEmpty() && heights[st.peek()] >= heights[i]) {
            st.pop();
         }
         if (!st.isEmpty()) {
            right[i] = st.peek();
         }
         st.push(i);
      }
      int res = 0;
      for (int i = 0; i < n; ++i) {
         res = Math.max(res, heights[i] * (right[i] - left[i] - 1));
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
      int x = 0;
      int y = 0;
      StringBuilder res = new StringBuilder();
      for (char c : target.toCharArray()) {
         int nx = (c - 'a') / 5;
         int ny = (c - 'a') % 5;
         while (x > nx) {
            res.append('U');
            --x;
         }
         while (y < ny) {
            res.append('R');
            ++y;
         }
         while (ny < y) {
            res.append('L');
            --y;
         }
         while (nx > x) {
            res.append('D');
            ++x;
         }
         res.append('!');
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

   // 1402. 做菜顺序 (Reducing Dishes)
   private int n1402;
   private int[] satisfaction1402;
   private int[][] memo1402;

   public int maxSatisfaction2(int[] satisfaction) {
      Arrays.sort(satisfaction);
      this.n1402 = satisfaction.length;
      this.satisfaction1402 = satisfaction;
      this.memo1402 = new int[n1402][n1402];
      for (int i = 0; i < n1402; ++i) {
         Arrays.fill(memo1402[i], (int) -1e8);
      }
      return dfs1402(0, 0);

   }

   private int dfs1402(int i, int j) {
      if (i == n1402) {
         return 0;
      }
      if (memo1402[i][j] != (int) -1e8) {
         return memo1402[i][j];
      }
      return memo1402[i][j] = Math.max(dfs1402(i + 1, j), dfs1402(i + 1, j + 1) + satisfaction1402[i] * (j + 1));
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

   // 1605. 给定行和列的和求可行矩阵 (Find Valid Matrix Given Row and Column Sums)
   public int[][] restoreMatrix2(int[] rowSum, int[] colSum) {
      int m = rowSum.length;
      int n = colSum.length;
      int[][] res = new int[m][n];
      int i = 0;
      int j = 0;
      while (i < m && j < n) {
         if (rowSum[i] < colSum[j]) {
            res[i][j] = rowSum[i];
            colSum[j] -= rowSum[i];
            ++i;
         } else {
            res[i][j] = colSum[j];
            rowSum[i] -= colSum[j];
            ++j;
         }
      }
      return res;

   }

   // 1710. 卡车上的最大单元数 (Maximum Units on a Truck)
   public int maximumUnits(int[][] boxTypes, int truckSize) {
      Arrays.sort(boxTypes, new Comparator<int[]>() {

         @Override
         public int compare(int[] o1, int[] o2) {
            return o2[1] - o1[1];
         }

      });
      int res = 0;
      int i = 0;
      while (truckSize > 0 && i < boxTypes.length) {
         int count = Math.min(truckSize, boxTypes[i][0]);
         res += count * boxTypes[i][1];
         truckSize -= count;
         ++i;
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

   // 面试题 08.11. 硬币 (Coin LCCI)
   private int[][] memo_08_11;
   private int n_08_11;
   private int[] map_08_11 = { 25, 10, 5, 1 };

   public int waysToChange2(int n) {
      this.n_08_11 = n;
      this.memo_08_11 = new int[n + 1][3];
      for (int i = 0; i < n + 1; ++i) {
         Arrays.fill(memo_08_11[i], -1);
      }
      return dfs_08_11(0, 0);

   }

   private int dfs_08_11(int sum, int coin) {
      if (sum == n_08_11) {
         return 1;
      }
      if (coin == 2) {
         return (n_08_11 - sum + map_08_11[coin]) / map_08_11[coin];
      }
      // if (coin == 3) {
      // return sum <= n ? 1 : 0;
      // }
      if (memo_08_11[sum][coin] != -1) {
         return memo_08_11[sum][coin];
      }
      final int MOD = (int) (1e9 + 7);
      int res = 0;
      // 不选
      res = (res + dfs_08_11(sum, coin + 1)) % MOD;
      // 选
      if (sum + map_08_11[coin] <= n_08_11) {
         res = (res + dfs_08_11(sum + map_08_11[coin], coin)) % MOD;
      }
      return memo_08_11[sum][coin] = res;
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

   // 677. 键值映射 (Map Sum Pairs) --哈希表
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
         return prefixMap.getOrDefault(prefix, 0);
      }
   }

   // 677. 键值映射 (Map Sum Pairs) --字典树
   class MapSum2 {
      private Trie667 trie;

      public MapSum2() {
         trie = new Trie667();
      }

      public void insert(String key, int val) {
         trie.insert(key, val);
      }

      public int sum(String prefix) {
         return trie.getPrefixCount(prefix);
      }
   }

   class Trie667 {
      private Trie667[] children;
      private int count;
      private Map<String, Integer> map;

      public Trie667() {
         children = new Trie667[26];
         count = 0;
         map = new HashMap<>();
      }

      public void insert(String s, int val) {
         Trie667 node = this;
         int delta = val - map.getOrDefault(s, 0);
         map.put(s, val);
         for (char c : s.toCharArray()) {
            int index = c - 'a';
            if (node.children[index] == null) {
               node.children[index] = new Trie667();
            }
            node = node.children[index];
            node.count += delta;
         }
      }

      public int getPrefixCount(String s) {
         Trie667 node = this;
         for (char c : s.toCharArray()) {
            int index = c - 'a';
            if (node.children[index] == null) {
               return 0;
            }
            node = node.children[index];
         }
         if (node == null) {
            return 0;
         }
         return node.count;
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

      public Bank(long[] balance) {
         this.balance = balance;
      }

      public boolean transfer(int account1, int account2, long money) {
         return accountIsLegal(account1) && accountIsLegal(account2) && withdraw(account1, money)
               && deposit(account2, money);
      }

      public boolean deposit(int account, long money) {
         if (!accountIsLegal(account)) {
            return false;
         }
         balance[account - 1] += money;
         return true;

      }

      public boolean withdraw(int account, long money) {
         if (!accountIsLegal(account)) {
            return false;
         }
         if (balance[account - 1] < money) {
            return false;
         }
         balance[account - 1] -= money;
         return true;
      }

      private boolean accountIsLegal(int account) {
         return account >= 1 && account <= balance.length;
      }
   }

   // 56. 合并区间 (Merge Intervals)
   // LCR 074. 合并区间
   public int[][] merge(int[][] intervals) {
      int n = intervals.length;
      Arrays.sort(intervals, new Comparator<int[]>() {

         @Override
         public int compare(int[] o1, int[] o2) {
            return Integer.compare(o1[0], o2[0]);
         }

      });
      List<int[]> res = new ArrayList<>();
      int i = 0;
      while (i < n) {
         int min = intervals[i][0];
         int max = intervals[i][1];
         int j = i + 1;
         while (j < n && intervals[j][0] <= max) {
            max = Math.max(max, intervals[j][1]);
            ++j;
         }
         res.add(new int[] { min, max });
         i = j;
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

   // 2022. 将一维数组转变成二维数组 (Convert 1D Array Into 2D Array)
   public int[][] construct2DArray(int[] original, int m, int n) {
      if (m * n != original.length) {
         return new int[0][];
      }
      int[][] res = new int[m][n];
      for (int i = 0; i < original.length; ++i) {
         res[i / n][i % n] = original[i];
      }
      return res;
   }

   // 2. 两数相加 (Add Two Numbers)
   // 面试题 02.05. 链表求和 (Sum Lists LCCI)
   public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
      ListNode dummy = new ListNode(0);
      ListNode cur = dummy;
      int carry = 0;
      while (l1 != null || l2 != null || carry != 0) {
         if (l1 != null) {
            carry += l1.val;
            l1 = l1.next;
         }
         if (l2 != null) {
            carry += l2.val;
            l2 = l2.next;
         }
         ListNode added = new ListNode(carry % 10);
         cur.next = added;
         cur = cur.next;
         carry /= 10;
      }
      return dummy.next;

   }

   // 19. 删除链表的倒数第 N 个结点 (Remove Nth Node From End of List) --快慢指针
   // 剑指 Offer II 021. 删除链表的倒数第 n 个结点
   public ListNode removeNthFromEnd(ListNode head, int n) {
      ListNode dummy = new ListNode(0, head);
      ListNode pre = dummy;
      ListNode cur = dummy;
      while (n-- > 0) {
         cur = cur.next;
      }
      while (cur.next != null) {
         pre = pre.next;
         cur = cur.next;
      }
      pre.next = pre.next.next;

      return dummy.next;

   }

   // 21. 合并两个有序链表 (Merge Two Sorted Lists)
   // 剑指 Offer 25. 合并两个排序的链表
   public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
      ListNode dummy = new ListNode(0);
      ListNode p = dummy;
      while (list1 != null || list2 != null) {
         if (list1 == null) {
            p.next = list2;
            break;
         }
         if (list2 == null) {
            p.next = list1;
            break;
         }
         if (list1.val < list2.val) {
            p.next = list1;
            list1 = list1.next;
         } else {
            p.next = list2;
            list2 = list2.next;
         }
         p = p.next;
      }
      return dummy.next;

   }

   // 24. 两两交换链表中的节点 (Swap Nodes in Pairs)
   public ListNode swapPairs(ListNode head) {
      ListNode dummy = new ListNode(0, head);
      ListNode guard = dummy;
      ListNode cur = head;
      while (guard.next != null && guard.next.next != null) {
         ListNode removed = cur.next;
         cur.next = cur.next.next;

         removed.next = guard.next;
         guard.next = removed;

         guard = cur;
         cur = cur.next;
      }
      return dummy.next;

   }

   // 61. 旋转链表 (Rotate List)
   public ListNode rotateRight(ListNode head, int k) {
      if (k == 0 || head == null || head.next == null) {
         return head;
      }
      int count = 1;
      ListNode cur = head;
      while (cur.next != null) {
         cur = cur.next;
         ++count;
      }
      int move = count - k % count;

      if (move == count) {
         return head;
      }
      cur.next = head;

      while (move-- > 0) {
         cur = cur.next;
      }
      ListNode res = cur.next;
      cur.next = null;

      return res;
   }

   // 82. 删除排序链表中的重复元素 II (Remove Duplicates from Sorted List II)
   public ListNode deleteDuplicates(ListNode head) {
      ListNode dummy = new ListNode(0, head);
      ListNode cur = dummy;
      while (cur != null) {
         ListNode tmp = cur;
         cur = cur.next;
         if (cur != null && cur.next != null && cur.val == cur.next.val) {
            int val = cur.val;
            while (cur != null && cur.val == val) {
               cur = cur.next;
            }
            tmp.next = cur;
            cur = tmp;
         }
      }
      return dummy.next;

   }

   // 83. 删除排序链表中的重复元素 (Remove Duplicates from Sorted List)
   public ListNode deleteDuplicates83(ListNode head) {
      ListNode cur = head;
      while (cur != null) {
         int val = cur.val;
         ListNode tmp = cur;
         while (cur != null && cur.val == val) {
            cur = cur.next;
         }
         tmp.next = cur;
      }
      return head;

   }

   // 86. 分隔链表 (Partition List)
   // 面试题 02.04. 分割链表 (Partition List LCCI)
   public ListNode partition(ListNode head, int x) {
      ListNode small = new ListNode(0, null);
      ListNode smallHead = small;
      ListNode large = new ListNode(0, null);
      ListNode largeHead = large;
      while (head != null) {
         if (head.val < x) {
            small.next = head;
            small = small.next;
         } else {
            large.next = head;
            large = large.next;
         }
         head = head.next;
      }
      small.next = largeHead.next;
      large.next = null;
      return smallHead.next;

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

   // 141. 环形链表 (Linked List Cycle)
   public boolean hasCycle(ListNode head) {
      ListNode fast = head;
      ListNode slow = head;
      while (fast != null && fast.next != null) {
         fast = fast.next.next;
         slow = slow.next;
         if (slow == fast) {
            return true;
         }
      }
      return false;
   }

   // 142. 环形链表 II (Linked List Cycle II)
   // 面试题 02.08. 环路检测 (Linked List Cycle LCCI)
   // 剑指 Offer II 022. 链表中环的入口节点
   public ListNode detectCycle2(ListNode head) {
      ListNode fast = head;
      ListNode slow = head;
      while (fast != null && fast.next != null) {
         fast = fast.next.next;
         slow = slow.next;
         if (fast == slow) {
            ListNode ptr1 = head;
            ListNode ptr2 = slow;
            while (ptr1 != ptr2) {
               ptr1 = ptr1.next;
               ptr2 = ptr2.next;
            }
            return ptr1;
         }
      }
      return null;

   }

   // 143. 重排链表 (Reorder List)
   // 剑指 Offer II 026. 重排链表
   public void reorderList(ListNode head) {
      ListNode m = getMid143(head);
      ListNode r = rev143(m);
      getMix143(head, r);
   }

   private void getMix143(ListNode l1, ListNode l2) {
      while (l1 != null && l2 != null) {
         ListNode t1 = l1.next;
         ListNode t2 = l2.next;

         l1.next = l2;
         l1 = t1;
         l2.next = l1;
         l2 = t2;
      }
   }

   private ListNode rev143(ListNode head) {
      ListNode pre = null;
      ListNode cur = head;
      while (cur != null) {
         ListNode nxt = cur.next;
         cur.next = pre;
         pre = cur;
         cur = nxt;
      }
      return pre;
   }

   private ListNode getMid143(ListNode head) {
      ListNode s = head;
      ListNode f = head;
      while (f != null && f.next != null && f.next.next != null) {
         s = s.next;
         f = f.next.next;
      }
      ListNode res = s.next;
      s.next = null;
      return res;
   }

   // 146. LRU 缓存机制 (LRU Cache)
   // 面试题 16.25. LRU 缓存 (LRU Cache LCCI)
   // 剑指 Offer II 031. 最近最少使用缓存
   class LRUCache {
      class Node {
         Node next;
         Node prev;
         int val;
         int key;

         Node() {

         }

         Node(int key, int val) {
            this.key = key;
            this.val = val;
         }

      }

      private Node head;
      private Node tail;
      private int size;
      private int capacity;
      private Map<Integer, Node> map;

      public LRUCache(int capacity) {
         map = new HashMap<>();
         head = new Node();
         tail = new Node();
         head.next = tail;
         tail.prev = head;
         this.capacity = capacity;

      }

      public int get(int key) {
         Node curNode = map.get(key);
         if (curNode == null) {
            return -1;
         }
         moveToHead(curNode);
         return curNode.val;

      }

      public void put(int key, int value) {
         Node curNode = map.get(key);
         if (curNode == null) {
            curNode = new Node(key, value);
            map.put(key, curNode);
            addToHead(curNode);
            if (++size > capacity) {
               map.remove(tail.prev.key);
               removeNode(tail.prev);
               --size;
            }
         } else {
            moveToHead(curNode);
            curNode.val = value;
         }

      }

      private void moveToHead(Node node) {
         removeNode(node);
         addToHead(node);
      }

      private void addToHead(Node node) {
         node.next = head.next;
         head.next = node;
         node.prev = head;
         node.next.prev = node;
      }

      private void removeNode(Node node) {
         node.prev.next = node.next;
         node.next.prev = node.prev;
      }

   }

   // 148. 排序链表 (Sort List) --归并排序
   // 剑指 Offer II 077. 链表排序
   public ListNode sortList(ListNode head) {
      if (head == null || head.next == null) {
         return head;
      }
      ListNode slow = head;
      ListNode fast = head;
      while (fast.next != null && fast.next.next != null) {
         slow = slow.next;
         fast = fast.next.next;
      }
      ListNode head2 = slow.next;
      slow.next = null;

      ListNode sorted1 = sortList(head);
      ListNode sorted2 = sortList(head2);

      ListNode dummy = new ListNode(0);
      ListNode cur = dummy;
      while (sorted1 != null && sorted2 != null) {
         if (sorted1.val < sorted2.val) {
            cur.next = sorted1;
            sorted1 = sorted1.next;
         } else {
            cur.next = sorted2;
            sorted2 = sorted2.next;
         }
         cur = cur.next;
      }
      cur.next = sorted1 == null ? sorted2 : sorted1;
      return dummy.next;
   }

   // 160. 相交链表 (Intersection of Two Linked Lists)
   // 剑指 Offer II 023. 两个链表的第一个重合节点
   // 剑指 Offer 52. 两个链表的第一个公共节点
   // 面试题 02.07. 链表相交
   public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
      if (headA == null || headB == null) {
         return null;
      }
      ListNode ptrA = headA;
      ListNode ptrB = headB;
      while (ptrA != ptrB) {
         ptrA = ptrA == null ? headB : ptrA.next;
         ptrB = ptrB == null ? headA : ptrB.next;
      }
      return ptrA;
   }

   // 203. 移除链表元素 (Remove Linked List Elements)
   // 剑指 Offer 18. 删除链表的节点
   public ListNode removeElements(ListNode head, int val) {
      ListNode dummy = new ListNode(0, head);
      ListNode cur = dummy;
      while (cur.next != null) {
         if (cur.next.val == val) {
            cur.next = cur.next.next;
            break;
         } else {
            cur = cur.next;
         }
      }
      return dummy.next;

   }

   // 206. 反转链表 (Reverse Linked List)
   // 剑指 Offer II 024. 反转链表
   // 剑指 Offer 24. 反转链表
   public ListNode reverseList(ListNode head) {
      ListNode ptr = null;
      ListNode cur = head;
      while (cur != null) {
         ListNode temp = cur.next;
         cur.next = ptr;

         ptr = cur;
         cur = temp;
      }
      return ptr;

   }

   // 206. 反转链表 (Reverse Linked List) 头插法
   // 剑指 Offer II 024. 反转链表
   // 剑指 Offer 24. 反转链表
   public ListNode reverseList2(ListNode head) {
      ListNode guard = new ListNode(0, head);
      ListNode point = head;
      while (point != null && point.next != null) {
         ListNode removed = point.next;
         point.next = point.next.next;

         removed.next = guard.next;
         guard.next = removed;

      }
      return guard.next;

   }

   // 234. 回文链表 --还需要掌握递归法
   // 递归法
   // ：https://leetcode-cn.com/problems/palindrome-linked-list-lcci/solution/hui-wen-lian-biao-by-leetcode-solution-6cp3/
   // 面试题 02.06. 回文链表
   // 剑指 Offer II 027. 回文链表
   // (还需要理解递归和快慢指针实现)
   public boolean isPalindrome(ListNode head) {
      ListNode midPre = getMiddleNode(head);
      ListNode l2 = midPre.next;
      midPre.next = null;
      ListNode l1 = head;
      l2 = reverseList234(l2);
      return judge234(l1, l2);
   }

   private boolean judge234(ListNode l1, ListNode l2) {
      while (l1 != null && l2 != null) {
         if (l1.val != l2.val) {
            return false;
         }
         l1 = l1.next;
         l2 = l2.next;
      }
      return true;
   }

   private ListNode reverseList234(ListNode head) {
      ListNode guard = new ListNode(0, head);
      ListNode point = head;
      while (point != null && point.next != null) {
         ListNode removed = point.next;
         point.next = point.next.next;

         removed.next = guard.next;
         guard.next = removed;

      }
      return guard.next;

   }

   private ListNode getMiddleNode(ListNode head) {
      ListNode dummy = new ListNode(0, head);
      ListNode slow = dummy;
      ListNode fast = dummy;
      while (fast != null && fast.next != null) {
         slow = slow.next;
         fast = fast.next.next;
      }
      return slow;
   }

   // 237. 删除链表中的节点 (Delete Node in a Linked List)
   // 面试题 02.03. 删除中间节点 (Delete Middle Node LCCI)
   public void deleteNode(ListNode node) {
      node.val = node.next.val;
      node.next = node.next.next;

   }

   // 328. 奇偶链表 (Odd Even Linked List)
   public ListNode oddEvenList(ListNode head) {
      if (head == null || head.next == null) {
         return head;
      }
      ListNode curOdd = head;
      ListNode even = head.next;
      ListNode curEven = even;
      while (curEven != null && curEven.next != null) {
         curOdd.next = curEven.next;
         curOdd = curOdd.next;

         curEven.next = curOdd.next;
         curEven = curEven.next;
      }
      curOdd.next = even;
      return head;

   }

   // 445. 两数相加 II (Add Two Numbers II)
   // 剑指 Offer II 025. 链表中的两数相加
   public ListNode addTwoNumbers445(ListNode l1, ListNode l2) {
      l1 = reverse445(l1);
      l2 = reverse445(l2);
      ListNode dummy = new ListNode(0);
      ListNode cur = dummy;
      int carry = 0;
      while (l1 != null || l2 != null || carry != 0) {
         if (l1 != null) {
            carry += l1.val;
            l1 = l1.next;
         }
         if (l2 != null) {
            carry += l2.val;
            l2 = l2.next;
         }
         ListNode node = new ListNode(carry % 10);
         carry /= 10;
         cur.next = node;
         cur = cur.next;
      }
      return reverse445(dummy.next);

   }

   private ListNode reverse445(ListNode node) {
      ListNode pre = null;
      while (node != null) {
         ListNode next = node.next;
         node.next = pre;
         pre = node;
         node = next;
      }
      return pre;
   }

   // 1721. 交换链表中的节点 (Swapping Nodes in a Linked List)
   public ListNode swapNodes(ListNode head, int k) {
      ListNode dummy = new ListNode(0, head);
      ListNode slow = dummy;
      ListNode fast = dummy;
      for (int i = 0; i < k; ++i) {
         fast = fast.next;
      }
      ListNode swapedNode1 = fast;

      while (fast != null) {
         slow = slow.next;
         fast = fast.next;
      }
      ListNode swapedNode2 = slow;

      int temp = swapedNode1.val;
      swapedNode1.val = swapedNode2.val;
      swapedNode2.val = temp;

      return dummy.next;

   }

   // 725. 分隔链表 (Split Linked List in Parts)
   public ListNode[] splitListToParts(ListNode head, int k) {
      ListNode[] res = new ListNode[k];
      ListNode itr = head;
      int count = 0;
      while (itr != null) {
         ++count;
         itr = itr.next;
      }
      int q = count / k;
      int r = count % k;
      ListNode cur = head;
      for (int i = 0; i < k && cur != null; ++i) {
         res[i] = cur;
         int size = q + ((i < r) ? 1 : 0);
         for (int j = 0; j < size - 1; ++j) {
            cur = cur.next;
         }
         ListNode temp = cur.next;
         cur.next = null;
         cur = temp;
      }
      return res;

   }

   // 705. 设计哈希集合 (Design HashSet) -- 还需掌握链表地址法
   class MyHashSet {
      private boolean[] set;

      public MyHashSet() {
         set = new boolean[1000001];
      }

      public void add(int key) {
         set[key] = true;
      }

      public void remove(int key) {
         set[key] = false;

      }

      public boolean contains(int key) {
         return set[key];
      }
   }

   // 706. 设计哈希映射 (Design HashMap) -- 还需掌握链表地址法
   class MyHashMap {
      private int[] map;

      public MyHashMap() {
         map = new int[1000001];
         Arrays.fill(map, -1);

      }

      public void put(int key, int value) {
         map[key] = value;
      }

      public int get(int key) {
         return map[key];

      }

      public void remove(int key) {
         map[key] = -1;
      }
   }

   // 817. 链表组件 (Linked List Components)
   public int numComponents(ListNode head, int[] nums) {
      Set<Integer> set = Arrays.stream(nums).boxed().collect(Collectors.toSet());
      ListNode dummy = new ListNode(-1);
      dummy.next = head;
      ListNode node = dummy;
      int res = 0;
      while (node.next != null) {
         while (node.next != null && !set.contains(node.next.val)) {
            node = node.next;
         }
         if (node.next == null) {
            break;
         }
         ++res;
         while (node.next != null && set.contains(node.next.val)) {
            node = node.next;
         }
      }
      return res;

   }

   // 707. 设计链表 (Design Linked List) --单链表
   class MyLinkedList {
      class Node {
         int val;
         Node next;

         public Node(int val) {
            this.val = val;

         }

         public Node(int val, Node next) {
            this.val = val;
            this.next = next;
         }
      }

      private Node head;
      private int size;

      public MyLinkedList() {
         head = new Node(0);
         size = 0;

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
         if (index > size) {
            return;
         }
         Node cur = head;
         for (int i = 0; i < index; ++i) {
            cur = cur.next;
         }
         Node node = new Node(val);
         node.next = cur.next;
         cur.next = node;
         ++size;
      }

      public void deleteAtIndex(int index) {
         if (index >= size || index < 0) {
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
         int val;
         Node prev;
         Node next;

         Node(int val) {
            this.val = val;
         }

         Node(int val, Node prev, Node next) {
            this.val = val;
            this.prev = prev;
            this.next = next;
         }

      }

      private Node head;
      private Node tail;
      private int size;

      public MyLinkedList2() {
         head = new Node(0);
         tail = new Node(0);
         head.next = tail;
         tail.prev = head;
         size = 0;
      }

      public int get(int index) {
         if (index < 0 || index >= size) {
            return -1;
         }

         if (index < size / 2) {
            Node cur = head;
            for (int i = 0; i <= index; ++i) {
               cur = cur.next;
            }
            return cur.val;
         } else {
            index = size - index - 1;
            Node cur = tail;
            for (int i = 0; i <= index; ++i) {
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
         if (index > size) {
            return;
         }

         if (index < size / 2) {
            Node cur = head;
            for (int i = 0; i < index; ++i) {
               cur = cur.next;
            }
            Node node = new Node(val);
            node.next = cur.next;
            node.next.prev = node;
            cur.next = node;
            node.prev = cur;
         } else {
            index = size - index;
            Node cur = tail;
            for (int i = 0; i < index; ++i) {
               cur = cur.prev;
            }
            Node node = new Node(val);
            node.prev = cur.prev;
            node.prev.next = node;
            cur.prev = node;
            node.next = cur;
         }

         ++size;

      }

      public void deleteAtIndex(int index) {
         if (index < 0 || index >= size) {
            return;
         }
         if (index < size / 2) {
            Node cur = head;
            for (int i = 0; i < index; ++i) {
               cur = cur.next;
            }
            cur.next = cur.next.next;
            cur.next.prev = cur;
         } else {
            Node cur = tail;
            index = size - index - 1;
            for (int i = 0; i < index; ++i) {
               cur = cur.prev;
            }
            cur.prev = cur.prev.prev;
            cur.prev.next = cur;
         }
         --size;

      }
   }

   // 876. 链表的中间结点 (Middle of the Linked List)
   public ListNode middleNode(ListNode head) {
      ListNode dummy = new ListNode(0, head);
      ListNode fast = dummy;
      ListNode slow = dummy;
      while (fast.next != null && fast.next.next != null) {
         slow = slow.next;
         fast = fast.next.next;
      }
      return slow.next;

   }

   // 1019. 链表中的下一个更大节点 (Next Greater Node In Linked List)
   public int[] nextLargerNodes(ListNode head) {
      Stack<Integer> stack = new Stack<>();
      List<Integer> list = new ArrayList<>();
      while (head != null) {
         list.add(head.val);
         head = head.next;
      }
      int[] res = new int[list.size()];
      for (int i = 0; i < list.size(); ++i) {
         while (!stack.isEmpty() && list.get(i) > list.get(stack.peek())) {
            res[stack.pop()] = list.get(i);
         }
         stack.push(i);
      }
      return res;

   }

   // 1171. 从链表中删去总和值为零的连续节点 (Remove Zero Sum Consecutive Nodes from Linked List)
   public ListNode removeZeroSumSublists(ListNode head) {
      Map<Integer, ListNode> map = new HashMap<>();
      int preSum = 0;
      ListNode dummy = new ListNode(0, head);
      ListNode cur = dummy;
      while (cur != null) {
         preSum += cur.val;
         map.put(preSum, cur);
         cur = cur.next;
      }
      preSum = 0;
      cur = dummy;
      while (cur != null) {
         preSum += cur.val;
         cur.next = map.get(preSum).next;
         cur = cur.next;
      }
      return dummy.next;

   }

   // 1290. 二进制链表转整数 (Convert Binary Number in a Linked List to Integer)
   public int getDecimalValue(ListNode head) {
      int res = 0;
      while (head != null) {
         res = (res << 1) | head.val;
         head = head.next;
      }
      return res;

   }

   // 1669. 合并两个链表 (Merge In Between Linked Lists)
   public ListNode mergeInBetween(ListNode list1, int a, int b, ListNode list2) {
      int count = 0;
      ListNode cur = list1;
      ListNode d1 = null;
      ListNode d2 = null;
      while (cur != null) {
         if (count == a - 1) {
            d1 = cur;
         }
         if (count == b) {
            d2 = cur;
            break;
         }
         ++count;
         cur = cur.next;
      }
      cur = list2;
      while (cur.next != null) {
         cur = cur.next;
      }
      cur.next = d2.next;
      d1.next = list2;
      return list1;

   }

   // 1670. 设计前中后队列 (Design Front Middle Back Queue)
   class FrontMiddleBackQueue {
      class Node {
         Node next;
         Node prev;
         int val;

         Node(int val) {
            this.val = val;
         }

      }

      private Node head;
      private Node tail;
      private int size;

      public FrontMiddleBackQueue() {
         head = new Node(0);
         tail = new Node(0);
         head.next = tail;
         tail.prev = head;

      }

      public void pushFront(int val) {
         pushAtIndex(0, val);

      }

      public void pushMiddle(int val) {
         pushAtIndex(size / 2, val);
      }

      public void pushBack(int val) {
         pushAtIndex(size, val);
      }

      private void pushAtIndex(int index, int val) {
         if (index < 0 || index > size) {
            return;
         }
         if (index * 2 >= size) {
            Node cur = tail;
            for (int i = 0; i < size - index; ++i) {
               cur = cur.prev;
            }
            Node added = new Node(val);
            added.prev = cur.prev;
            added.next = cur;
            cur.prev = added;
            added.prev.next = added;
         } else {
            Node cur = head;
            for (int i = 0; i < index; ++i) {
               cur = cur.next;
            }
            Node added = new Node(val);
            added.next = cur.next;
            added.prev = cur;
            cur.next = added;
            added.next.prev = added;
         }
         ++size;
      }

      public int popFront() {
         return popAtIndex(0);

      }

      public int popMiddle() {
         return popAtIndex((size - 1) / 2);

      }

      public int popBack() {
         return popAtIndex(size - 1);
      }

      private int popAtIndex(int index) {
         if (index < 0 || index >= size) {
            return -1;
         }
         if (index * 2 >= size) {
            Node cur = tail;
            for (int i = 0; i < size - index - 1; ++i) {
               cur = cur.prev;
            }
            int res = cur.prev.val;
            cur.prev = cur.prev.prev;
            cur.prev.next = cur;
            --size;
            return res;
         } else {
            Node cur = head;
            for (int i = 0; i < index; ++i) {
               cur = cur.next;
            }
            int res = cur.next.val;
            cur.next = cur.next.next;
            cur.next.prev = cur;
            --size;
            return res;
         }
      }
   }

   // 2058. 找出临界点之间的最小和最大距离
   // 2058.Find the Minimum and Maximum Number of Nodes Between Critical Points
   public int[] nodesBetweenCriticalPoints(ListNode head) {
      int min = Integer.MAX_VALUE;
      int max = Integer.MIN_VALUE;
      int firstCriticalIndex = -1;
      int lastCriticalIndex = -1;
      int curIndex = 1;
      ListNode pre = head;
      ListNode cur = head.next;
      while (cur != null && cur.next != null) {
         if ((cur.val > pre.val && cur.val > cur.next.val) || (cur.val < pre.val && cur.val < cur.next.val)) {
            if (firstCriticalIndex == -1) {
               firstCriticalIndex = curIndex;
            } else {
               min = Math.min(min, curIndex - lastCriticalIndex);
               max = Math.max(max, curIndex - firstCriticalIndex);
            }
            lastCriticalIndex = curIndex;
         }
         ++curIndex;
         cur = cur.next;
         pre = pre.next;
      }
      return min == Integer.MAX_VALUE ? new int[] { -1, -1 } : new int[] { min, max };

   }

   // 2074. 反转偶数长度组的节点 (Reverse Nodes in Even Length Groups)
   public ListNode reverseEvenLengthGroups(ListNode head) {
      ListNode dummy = new ListNode(0, head);
      ListNode pre = dummy;
      ListNode cur = head;
      int count = 0;
      while (cur != null) {
         ++count;
         int length = 0;
         ListNode tryIt = cur;
         while (length < count && tryIt != null) {
            tryIt = tryIt.next;
            ++length;
         }
         if ((length & 1) == 0) {
            for (int i = 0; i < length - 1; ++i) {
               ListNode removed = cur.next;
               cur.next = cur.next.next;

               removed.next = pre.next;
               pre.next = removed;
            }
            pre = cur;
            cur = cur.next;
         } else {
            for (int i = 0; i < length; ++i) {
               pre = pre.next;
               cur = cur.next;
            }
         }
      }
      return dummy.next;

   }

   // 2095. 删除链表的中间节点 (Delete the Middle Node of a Linked List)
   public ListNode deleteMiddle(ListNode head) {
      ListNode dummy = new ListNode(0, head);
      ListNode slow = dummy;
      ListNode fast = dummy;
      while (fast.next != null && fast.next.next != null) {
         slow = slow.next;
         fast = fast.next.next;
      }
      slow.next = slow.next.next;
      return dummy.next;

   }

   // 面试题 02.01. 移除重复节点 (Remove Duplicate Node LCCI) --O(n)
   public ListNode removeDuplicateNodes(ListNode head) {
      Set<Integer> set = new HashSet<>();
      ListNode dummy = new ListNode(0, head);
      ListNode cur = dummy;
      while (cur.next != null) {
         if (set.add(cur.next.val)) {
            cur = cur.next;
         } else {
            cur.next = cur.next.next;
         }
      }
      return dummy.next;

   }

   // 面试题 02.01. 移除重复节点 (Remove Duplicate Node LCCI) --O(n^2)
   public ListNode removeDuplicateNodes2(ListNode head) {
      ListNode cur = head;
      while (cur != null) {
         int val = cur.val;
         ListNode innerCur = cur;
         while (innerCur.next != null) {
            if (innerCur.next.val == val) {
               innerCur.next = innerCur.next.next;
            } else {
               innerCur = innerCur.next;
            }
         }
         cur = cur.next;
      }
      return head;

   }

   // 面试题 03.03. 堆盘子
   class StackOfPlates {
      private List<Stack<Integer>> mList;
      private int mCapacity;

      public StackOfPlates(int cap) {
         this.mCapacity = cap;
         this.mList = new ArrayList<>();

      }

      public void push(int val) {
         if (mCapacity <= 0) {
            return;
         }
         if (mList.isEmpty() || mList.get(mList.size() - 1).size() == mCapacity) {
            Stack<Integer> stack = new Stack<>();
            stack.push(val);
            mList.add(stack);
         } else {
            Stack<Integer> curStack = mList.get(mList.size() - 1);
            curStack.push(val);
         }
      }

      public int pop() {
         if (mList.isEmpty()) {
            return -1;
         }
         Stack<Integer> curStack = mList.get(mList.size() - 1);
         int res = curStack.pop();
         if (curStack.isEmpty()) {
            mList.remove(mList.size() - 1);
         }
         return res;
      }

      public int popAt(int index) {
         if (index < 0 || index >= mList.size()) {
            return -1;
         }
         Stack<Integer> curStack = mList.get(index);
         int res = curStack.pop();
         if (curStack.isEmpty()) {
            mList.remove(index);
         }
         return res;

      }
   }

   // 面试题 02.02. 返回倒数第 k 个节点 (Kth Node From End of List LCCI)
   // 剑指 Offer 22. 链表中倒数第k个节点
   public int kthToLast(ListNode head, int k) {
      ListNode dummy = new ListNode(0, head);
      ListNode former = dummy;
      ListNode latter = dummy;
      while (k-- > 0) {
         latter = latter.next;
      }
      while (latter != null) {
         former = former.next;
         latter = latter.next;
      }
      return former.val;

   }

   // 剑指 Offer 06. 从尾到头打印链表 --递归
   private List<Integer> list = new ArrayList<>();

   public int[] reversePrint(ListNode head) {
      recur(head);
      int[] res = new int[list.size()];
      for (int i = 0; i < res.length; ++i) {
         res[i] = list.get(i);
      }
      return res;
   }

   private void recur(ListNode head) {
      if (head == null) {
         return;
      }
      reversePrint(head.next);
      list.add(head.val);
   }

   // 1985. 找出数组中的第 K 大整数 (Find the Kth Largest Integer in the Array)
   public String kthLargestNumber(String[] nums, int k) {
      Arrays.sort(nums, new Comparator<String>() {

         @Override
         public int compare(String o1, String o2) {
            if (o1.length() != o2.length()) {
               return o2.length() - o1.length();
            }
            return o2.compareTo(o1);
         }

      });
      return nums[k - 1];

   }

   // 2079. 给植物浇水 (Watering Plants)
   public int wateringPlants(int[] plants, int capacity) {
      int res = 0;
      int curCapacity = capacity;
      for (int i = 0; i < plants.length; ++i) {
         curCapacity -= plants[i];
         ++res;
         if (i + 1 < plants.length && curCapacity < plants[i + 1]) {
            res += (i + 1) * 2;
            curCapacity = capacity;
         }

      }
      return res;

   }

   // 剑指 Offer 05. 替换空格
   public String replaceSpace(String s) {
      StringBuilder res = new StringBuilder();
      for (char c : s.toCharArray()) {
         res.append(Character.isWhitespace(c) ? "%20" : c);
      }
      return res.toString();

   }

   // 2078. 两栋颜色不同且距离最远的房子 (Two Furthest Houses With Different Colors)
   public int maxDistance(int[] colors) {
      int res = 0;
      int n = colors.length;
      for (int i = n - 1; i >= 1; --i) {
         if (colors[i] != colors[0]) {
            res = Math.max(res, i);
            break;
         }
      }
      for (int i = 0; i < n - 1; ++i) {
         if (colors[i] != colors[n - 1]) {
            res = Math.max(res, n - 1 - i);
            break;
         }
      }
      return res;

   }

   // 238. 除自身以外数组的乘积 (Product of Array Except Self)
   // 剑指 Offer 66. 构建乘积数组
   public int[] productExceptSelf(int[] nums) {
      int n = nums.length;
      int[] res = new int[n];
      int suf = 1;
      for (int i = n - 1; i >= 0; --i) {
         res[i] = suf;
         suf *= nums[i];
      }
      int pre = 1;
      for (int i = 0; i < n; ++i) {
         res[i] *= pre;
         pre *= nums[i];
      }
      return res;

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
      private int[][] preSum;
      private int m;
      private int n;

      public NumMatrix(int[][] matrix) {
         m = matrix.length;
         n = matrix[0].length;
         preSum = new int[m + 1][n + 1];
         for (int i = 1; i < m + 1; ++i) {
            for (int j = 1; j < n + 1; ++j) {
               preSum[i][j] = preSum[i - 1][j] + preSum[i][j - 1] - preSum[i - 1][j - 1] + matrix[i - 1][j - 1];
            }
         }

      }

      public int sumRegion(int row1, int col1, int row2, int col2) {
         return preSum[row2 + 1][col2 + 1] - preSum[row2 + 1][col1] - preSum[row1][col2 + 1] + preSum[row1][col1];
      }
   }

   // 525. 连续数组 (Contiguous Array) -- 前缀和
   // 剑指 Offer II 011. 0 和 1 个数相同的子数组
   public int findMaxLength(int[] nums) {
      Map<Integer, Integer> map = new HashMap<>();
      map.put(0, -1);
      int count = 0;
      int res = 0;
      for (int i = 0; i < nums.length; ++i) {
         count += nums[i] == 0 ? -1 : 1;
         if (map.containsKey(count)) {
            res = Math.max(res, i - map.get(count));
         } else {
            map.put(count, i);
         }
      }
      return res;

   }

   // 560. 和为K的子数组 (Subarray Sum Equals K) --前缀和
   // 剑指 Offer II 010. 和为 k 的子数组
   public int subarraySum(int[] nums, int k) {
      Map<Integer, Integer> map = new HashMap<>();
      map.put(0, 1);
      int res = 0;
      int pre = 0;
      for (int x : nums) {
         pre += x;
         res += map.getOrDefault(pre - k, 0);
         map.merge(pre, 1, Integer::sum);
      }
      return res;

   }

   // 724. 寻找数组的中心下标 (Find Pivot Index) --前缀和
   // 1991. 找到数组的中间位置 (Find the Middle Index in Array)
   // 剑指 Offer II 012. 左右两边子数组的和相等
   public int pivotIndex(int[] nums) {
      int s = Arrays.stream(nums).sum();
      int l = 0;
      for (int i = 0; i < nums.length; ++i) {
         s -= nums[i];
         if (l == s) {
            return i;
         }
         l += nums[i];
      }
      return -1;
   }

   // 2089. 找出数组排序后的目标下标 (Find Target Indices After Sorting Array) -O(n)
   public List<Integer> targetIndices2(int[] nums, int target) {
      int less = 0;
      int equal = 0;
      for (int num : nums) {
         if (num < target) {
            ++less;
         } else if (num == target) {
            ++equal;
         }
      }
      List<Integer> res = new ArrayList<>();
      for (int i = less; i < less + equal; ++i) {
         res.add(i);
      }
      return res;

   }

   // 930. 和相同的二元子数组 (Binary Subarrays With Sum) --前缀和
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

   // 930. 和相同的二元子数组 (Binary Subarrays With Sum) --滑动窗口
   public int numSubarraysWithSum2(int[] nums, int goal) {
      int left1 = 0;
      int left2 = 0;
      int right = 0;
      int preSum1 = 0;
      int preSum2 = 0;
      int res = 0;
      while (right < nums.length) {
         preSum1 += nums[right];
         while (left1 <= right && preSum1 > goal) {
            preSum1 -= nums[left1++];
         }
         preSum2 += nums[right];
         while (left2 <= right && preSum2 >= goal) {
            preSum2 -= nums[left2++];
         }
         res += left2 - left1;
         ++right;
      }
      return res;
   }

   // 974. 和可被 K 整除的子数组 (Subarray Sums Divisible by K) --前缀和
   public int subarraysDivByK(int[] nums, int k) {
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

   // 1004.最大连续1的个数 III (Max Consecutive Ones III)
   public int longestOnes(int[] nums, int k) {
      int res = 0;
      int j = 0;
      int cnt0 = 0;
      for (int i = 0; i < nums.length; ++i) {
         cnt0 += 1 ^ nums[i];
         while (cnt0 > k) {
            cnt0 -= nums[j++] ^ 1;
         }
         res = Math.max(res, i - j + 1);
      }
      return res;

   }

   // 1094. 拼车 (Car Pooling) --差分数组 前缀和
   public boolean carPooling(int[][] trips, int capacity) {
      int[] diff = new int[1002];
      for (int[] trip : trips) {
         diff[trip[1]] += trip[0];
         diff[trip[2]] -= trip[0];
      }
      int preSum = diff[0];
      if (preSum > capacity) {
         return false;
      }
      for (int i = 1; i < diff.length; ++i) {
         preSum += diff[i];
         if (preSum > capacity) {
            return false;
         }
      }
      return true;

   }

   // 1854. 人口最多的年份 (Maximum Population Year) --差分数组 前缀和
   public int maximumPopulation(int[][] logs) {
      int[] diff = new int[101];
      for (int[] log : logs) {
         ++diff[log[0] - 1950];
         --diff[log[1] - 1950];
      }
      int maxYear = 0;
      int max = diff[0];
      for (int i = 1; i < diff.length; ++i) {
         diff[i] += diff[i - 1];
         if (max < diff[i]) {
            maxYear = i;
            max = diff[i];
         }
      }
      return maxYear + 1950;

   }

   // 1109. 航班预订统计 (Corporate Flight Bookings)
   public int[] corpFlightBookings(int[][] bookings, int n) {
      int[] diff = new int[n];
      for (int[] booking : bookings) {
         diff[booking[0] - 1] += booking[2];
         if (booking[1] < n) {
            diff[booking[1]] -= booking[2];
         }
      }
      for (int i = 1; i < diff.length; ++i) {
         diff[i] += diff[i - 1];
      }
      return diff;

   }

   // 1314. 矩阵区域和 (Matrix Block Sum)
   public int[][] matrixBlockSum2(int[][] mat, int k) {
      int m = mat.length;
      int n = mat[0].length;
      int[][] preSum = new int[m + 1][n + 1];
      for (int i = 1; i <= m; ++i) {
         for (int j = 1; j <= n; ++j) {
            preSum[i][j] = preSum[i - 1][j] + preSum[i][j - 1] - preSum[i - 1][j - 1] + mat[i - 1][j - 1];
         }
      }

      int[][] res = new int[m][n];
      for (int i = 0; i < m; ++i) {
         for (int j = 0; j < n; ++j) {
            int x1 = Math.max(i - k, 0);
            int x2 = Math.min(i + k, m - 1);
            int y1 = Math.max(j - k, 0);
            int y2 = Math.min(j + k, n - 1);
            res[i][j] = preSum[x2 + 1][y2 + 1] - preSum[x2 + 1][y1] - preSum[x1][y2 + 1] + preSum[x1][y1];
         }
      }
      return res;

   }

   // 1446. 连续字符 (Consecutive Characters)
   public int maxPower(String s) {
      int res = 1;
      int cur = 1;
      char[] chars = s.toCharArray();
      for (int i = 1; i < chars.length; ++i) {
         if (chars[i] == chars[i - 1]) {
            ++cur;
            res = Math.max(res, cur);
         } else {
            cur = 1;
         }
      }
      return res;

   }

   // 1694. 重新格式化电话号码 (Reformat Phone Number)
   public String reformatNumber(String number) {
      StringBuilder builder = new StringBuilder();
      for (char c : number.toCharArray()) {
         if (Character.isDigit(c)) {
            builder.append(c);
         }
      }
      StringBuilder res = new StringBuilder();
      int i = 0;
      while (i < builder.length() - 4) {
         res.append(builder.substring(i, i + 3)).append("-");
         i += 3;
      }
      if (i == builder.length() - 4) {
         res.append(builder.substring(i, i + 2)).append("-");
         i += 2;
      }
      res.append(builder.substring(i));

      return res.toString();

   }

   // 944. 删列造序 (Delete Columns to Make Sorted)
   public int minDeletionSize(String[] strs) {
      int res = 0;
      int n = strs.length;
      int m = strs[0].length();
      for (int j = 0; j < m; ++j) {
         for (int i = 1; i < n; ++i) {
            if (strs[i].charAt(j) < strs[i - 1].charAt(j)) {
               ++res;
               break;
            }
         }
      }
      return res;

   }

   // 2023. 连接后等于目标字符串的字符串对 --暴力 还需掌握更优的方法
   // 2023. Number of Pairs of Strings With Concatenation Equal to Target
   public int numOfPairs(String[] nums, String target) {
      int res = 0;
      for (int i = 0; i < nums.length; ++i) {
         for (int j = 0; j < nums.length; ++j) {
            if (i != j && target.equals(nums[i] + nums[j])) {
               ++res;
            }
         }
      }
      return res;

   }

   // 1292. 元素和小于等于阈值的正方形的最大边长 (Maximum Side Length of a Square with Sum Less than
   // or Equal to Threshold) --二分查找
   public int maxSideLength(int[][] mat, int threshold) {
      int m = mat.length;
      int n = mat[0].length;
      int[][] pre = new int[m + 1][n + 1];
      for (int i = 0; i < m; ++i) {
         for (int j = 0; j < n; ++j) {
            pre[i + 1][j + 1] = pre[i][j + 1] + pre[i + 1][j] - pre[i][j] + mat[i][j];
         }
      }
      int left = 0;
      int right = Math.min(m, n);
      while (left <= right) {
         int mid = left + ((right - left) >> 1);
         if (check1292(mid, pre, threshold)) {
            left = mid + 1;
         } else {
            right = mid - 1;
         }

      }
      return left - 1;

   }

   private boolean check1292(int k, int[][] pre, int threshold) {
      int m = pre.length - 1;
      int n = pre[0].length - 1;
      for (int i = 0; i <= m - k; ++i) {
         for (int j = 0; j <= n - k; ++j) {
            int sum = pre[i + k][j + k] - pre[i][j + k] - pre[i + k][j] + pre[i][j];
            if (sum <= threshold) {
               return true;
            }
         }
      }
      return false;
   }

   // 1292. 元素和小于等于阈值的正方形的最大边长 (Maximum Side Length of a Square with Sum Less than
   // or Equal to Threshold) --贪心 前缀和
   public int maxSideLength2(int[][] mat, int threshold) {
      int m = mat.length;
      int n = mat[0].length;
      int[][] preSum = new int[m + 1][n + 1];
      for (int i = 1; i <= m; ++i) {
         for (int j = 1; j <= n; ++j) {
            preSum[i][j] = preSum[i - 1][j] + preSum[i][j - 1] - preSum[i - 1][j - 1] + mat[i - 1][j - 1];
         }
      }
      int side = 0;
      int maxSide = Math.min(m, n);
      for (int i = 0; i < m; ++i) {
         for (int j = 0; j < n; ++j) {
            for (int k = side + 1; k <= maxSide; ++k) {
               if (i + k - 1 >= m || j + k - 1 >= n) {
                  break;
               }
               int curSum = preSum[i + k][j + k] - preSum[i + k][j] - preSum[i][j + k] + preSum[i][j];
               if (curSum > threshold) {
                  break;
               }
               side = k;
            }
         }
      }
      return side;

   }

   // 1310. 子数组异或查询 (XOR Queries of a Subarray)
   public int[] xorQueries(int[] arr, int[][] queries) {
      int[] preXOR = new int[arr.length + 1];
      for (int i = 1; i < preXOR.length; ++i) {
         preXOR[i] = preXOR[i - 1] ^ arr[i - 1];
      }
      int[] res = new int[queries.length];
      for (int i = 0; i < queries.length; ++i) {
         res[i] = preXOR[queries[i][1] + 1] ^ preXOR[queries[i][0]];
      }
      return res;

   }

   // 1371. 每个元音包含偶数次的最长子字符串 (Find the Longest Substring Containing Vowels in Even
   // Counts)
   public int findTheLongestSubstring(String s) {
      int[] status = new int[1 << 5];
      Arrays.fill(status, Integer.MAX_VALUE);
      int res = 0;
      status[0] = -1;
      int cur = 0;
      char[] chars = s.toCharArray();
      for (int i = 0; i < chars.length; ++i) {
         if (chars[i] == 'a') {
            cur ^= 1 << 0;
         } else if (chars[i] == 'e') {
            cur ^= 1 << 1;
         } else if (chars[i] == 'i') {
            cur ^= 1 << 2;
         } else if (chars[i] == 'o') {
            cur ^= 1 << 3;
         } else if (chars[i] == 'u') {
            cur ^= 1 << 4;
         }
         if (status[cur] == Integer.MAX_VALUE) {
            status[cur] = i;
         } else {
            res = Math.max(res, i - status[cur]);
         }
      }
      return res;

   }

   // 1413. 逐步求和得到正数的最小值 (Minimum Value to Get Positive Step by Step Sum)
   public int minStartValue(int[] nums) {
      int min = nums[0];
      for (int i = 1; i < nums.length; ++i) {
         nums[i] += nums[i - 1];
         min = Math.min(min, nums[i]);
      }
      return Math.max(1, 1 - min);

   }

   // 1480. 一维数组的动态和 (Running Sum of 1d Array)
   public int[] runningSum(int[] nums) {
      for (int i = 1; i < nums.length; ++i) {
         nums[i] += nums[i - 1];
      }
      return nums;

   }

   // 1524. 和为奇数的子数组数目 (Number of Sub-arrays With Odd Sum) --前缀和
   public int numOfSubarrays(int[] arr) {
      int odd = 0;
      int even = 1;
      int pre = 0;
      final int MOD = (int) (1e9 + 7);
      int res = 0;
      for (int num : arr) {
         pre = (pre + num) % 2;
         if (pre == 0) {
            res = (res + odd) % MOD;
            ++even;
         } else {
            res = (res + even) % MOD;
            ++odd;
         }
      }
      return res;

   }

   // 1546. 和为目标值且不重叠的非空子数组的最大数目 (Maximum Number of Non-Overlapping Subarrays
   // With Sum Equals Target)
   public int maxNonOverlapping(int[] nums, int target) {
      int res = 0;
      Set<Integer> set = new HashSet<>();
      set.add(0);
      int preSum = 0;
      for (int num : nums) {
         preSum += num;
         if (set.contains(preSum - target)) {
            preSum = 0;
            set.clear();
            set.add(0);
            ++res;
         } else {
            set.add(preSum);
         }
      }
      return res;

   }

   // 1588. 所有奇数长度子数组的和 (Sum of All Odd Length Subarrays)
   public int sumOddLengthSubarrays(int[] arr) {
      int res = 0;
      for (int i = 0; i < arr.length; ++i) {
         int leftOddCount = (i + 1) / 2;
         int leftEvenCount = i / 2 + 1;
         int rightOddCount = (arr.length - i) / 2;
         int rightEvenCount = (arr.length - i + 1) / 2;
         res += (leftOddCount * rightOddCount + leftEvenCount * rightEvenCount) * arr[i];
      }
      return res;

   }

   // 1685. 有序数组中差绝对值之和 (Sum of Absolute Differences in a Sorted Array)
   public int[] getSumAbsoluteDifferences(int[] nums) {
      int n = nums.length;
      int[] preSum = new int[n + 1];
      for (int i = 1; i < n + 1; ++i) {
         preSum[i] += nums[i - 1] + preSum[i - 1];
      }
      int[] res = new int[n];
      for (int i = 0; i < n; ++i) {
         res[i] = (i * nums[i] - preSum[i]) + (preSum[n] - preSum[i + 1] - (n - i - 1) * nums[i]);
      }
      return res;

   }

   // 1732. 找到最高海拔 (Find the Highest Altitude)
   public int largestAltitude(int[] gain) {
      int res = 0;
      int cur = 0;
      for (int num : gain) {
         cur += num;
         res = Math.max(cur, res);
      }
      return res;

   }

   // 1738. 找出第 K 大的异或坐标值 (Find Kth Largest XOR Coordinate Value) --二维前缀和+排序
   // 还需掌握二维前缀和 + 快速选择算法
   public int kthLargestValue(int[][] matrix, int k) {
      List<Integer> list = new ArrayList<>();
      for (int i = 0; i < matrix.length; ++i) {
         for (int j = 0; j < matrix[0].length; ++j) {
            if (i == 0 && j > 0) {
               matrix[i][j] ^= matrix[i][j - 1];
            } else if (j == 0 && i > 0) {
               matrix[i][j] ^= matrix[i - 1][j];
            } else if (i > 0 && j > 0) {
               matrix[i][j] ^= matrix[i][j - 1] ^ matrix[i - 1][j] ^ matrix[i - 1][j - 1];
            }
            list.add(matrix[i][j]);
         }
      }
      Collections.sort(list);
      return list.get(list.size() - k);

   }

   // 1829. 每个查询的最大异或值 (Maximum XOR for Each Query)
   public int[] getMaximumXor(int[] nums, int maximumBit) {
      int n = nums.length;
      int[] res = new int[n];
      int xor = 0;
      for (int num : nums) {
         xor ^= num;
      }
      int mask = (1 << maximumBit) - 1;
      for (int i = 0; i < n; ++i) {
         res[i] = mask ^ xor;
         xor ^= nums[n - i - 1];
      }
      return res;

   }

   // 806. 写字符串需要的行数 (Number of Lines To Write String)
   public int[] numberOfLines(int[] widths, String s) {
      int level = 1;
      int cur = 0;
      for (int i = 0; i < s.length(); ++i) {
         int count = widths[s.charAt(i) - 'a'];
         if (cur + count <= 100) {
            cur += count;
         } else {
            ++level;
            cur = count;
         }
      }
      return new int[] { level, cur };

   }

   // 2042. 检查句子中的数字是否递增 (Check if Numbers Are Ascending in a Sentence)
   public boolean areNumbersAscending(String s) {
      int n = s.length();
      int i = 0;
      int pre = -1;
      while (i < n) {
         if (!Character.isDigit(s.charAt(i))) {
            ++i;
            continue;
         }
         int num = 0;
         while (i < n && Character.isDigit(s.charAt(i))) {
            num = num * 10 + s.charAt(i) - '0';
            ++i;
         }
         if (pre >= num) {
            return false;
         }
         pre = num;
      }
      return true;

   }

   // 1838. 最高频元素的频数 (Frequency of the Most Frequent Element) --双指针+前缀和
   public int maxFrequency(int[] nums, int k) {
      Arrays.sort(nums);
      int res = 1;
      int left = 0;
      int right = 1;
      long sum = 0;
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

   // 1893. 检查是否区域内所有整数都被覆盖 (Check if All the Integers in a Range Are Covered)
   public boolean isCovered(int[][] ranges, int left, int right) {
      Arrays.sort(ranges, (o1, o2) -> o1[0] - o2[0]);
      for (int[] range : ranges) {
         if (range[0] <= left && left <= range[1]) {
            left = range[1] + 1;
         }
      }
      return left > right;

   }

   // 1893. 检查是否区域内所有整数都被覆盖 (Check if All the Integers in a Range Are Covered)
   public boolean isCovered2(int[][] ranges, int left, int right) {
      Arrays.sort(ranges, new Comparator<int[]>() {

         @Override
         public int compare(int[] o1, int[] o2) {
            return Integer.compare(o1[0], o2[0]);
         }

      });
      int i = 0;
      int n = ranges.length;
      while (i < n) {
         int l = ranges[i][0];
         int r = ranges[i][1];
         int j = i;
         while (j < n && ranges[j][0] <= r + 1) {
            r = Math.max(r, ranges[j][1]);
            ++j;
         }
         if (l <= left && right <= r) {
            return true;
         }
         i = j;
      }
      return false;

   }

   // 1893. 检查是否区域内所有整数都被覆盖 (Check if All the Integers in a Range Are Covered)
   // --差分数组
   public boolean isCovered3(int[][] ranges, int left, int right) {
      int[] diff = new int[52];
      for (int[] range : ranges) {
         ++diff[range[0]];
         --diff[range[1] + 1];
      }
      int preSum = 0;
      for (int i = 1; i <= 50; ++i) {
         preSum += diff[i];
         if (left <= i && i <= right && preSum <= 0) {
            return false;
         }
      }
      return true;

   }

   // 1893. 检查是否区域内所有整数都被覆盖 (Check if All the Integers in a Range Are Covered)
   // --位运算
   public boolean isCovered4(int[][] ranges, int left, int right) {
      long rangeNum = 0;
      for (int[] range : ranges) {
         rangeNum |= ((1L << (range[1] + 1)) - 1) ^ ((1L << range[0]) - 1);
      }
      long leftToRightNum = ((1L << (right + 1)) - 1) ^ ((1L << left) - 1);
      return (rangeNum & leftToRightNum) == leftToRightNum;

   }

   // 1894. 找到需要补充粉笔的学生编号 (Find the Student that Will Replace the Chalk)
   public int chalkReplacer(int[] chalk, int k) {
      long sum = 0;
      for (int c : chalk) {
         sum += c;
      }
      long remain = k % sum;
      for (int i = 0; i < chalk.length; ++i) {
         if (remain < chalk[i]) {
            return i;
         }
         remain -= chalk[i];
      }
      return -1;

   }

   // 1915. 最美子字符串的数目 (Number of Wonderful Substrings)
   public long wonderfulSubstrings(String word) {
      int[] state = new int[1 << 10];
      state[0] = 1;
      long res = 0;
      int cur = 0;
      for (char c : word.toCharArray()) {
         cur ^= 1 << (c - 'a');
         res += state[cur];
         for (int i = 0; i < 10; ++i) {
            res += state[cur ^ (1 << i)];
         }
         ++state[cur];
      }
      return res;

   }

   // 2024. 考试的最大困扰度 (Maximize the Confusion of an Exam)
   public int maxConsecutiveAnswers(String answerKey, int k) {
      int tCnt = 0;
      int fCnt = 0;
      int res = 0;
      int j = 0;
      char[] chars = answerKey.toCharArray();
      for (int i = 0; i < chars.length; ++i) {
         if (chars[i] == 'T') {
            ++tCnt;
         } else {
            ++fCnt;
         }
         while (tCnt > k && fCnt > k) {
            if (chars[j] == 'T') {
               --tCnt;
            } else {
               --fCnt;
            }
            ++j;
         }
         res = Math.max(res, i - j + 1);
      }
      return res;

   }

   // 383. 赎金信 (Ransom Note)
   public boolean canConstruct(String ransomNote, String magazine) {
      int[] count = new int[26];
      for (char c : magazine.toCharArray()) {
         ++count[c - 'a'];
      }
      for (char c : ransomNote.toCharArray()) {
         if (--count[c - 'a'] < 0) {
            return false;
         }
      }
      return true;
   }

   // 2085. 统计出现过一次的公共字符串 (Count Common Words With One Occurrence)
   public int countWords(String[] words1, String[] words2) {
      int res = 0;
      Map<String, Integer> counts1 = new HashMap<>();
      for (String word : words1) {
         counts1.merge(word, 1, Integer::sum);
      }
      Map<String, Integer> counts2 = new HashMap<>();
      for (String word : words2) {
         counts2.merge(word, 1, Integer::sum);
      }
      for (Map.Entry<String, Integer> entry : counts1.entrySet()) {
         if (entry.getValue() == 1 && counts2.getOrDefault(entry.getKey(), 0) == 1) {
            ++res;
         }
      }
      return res;

   }

   // 面试题 17.11. 单词距离 (Find Closest LCCI)
   public int findClosest(String[] words, String word1, String word2) {
      int index1 = -1;
      int index2 = -1;
      int res = Integer.MAX_VALUE;
      for (int i = 0; i < words.length; ++i) {
         if (words[i].equals(word1)) {
            if (index2 != -1) {
               res = Math.min(res, i - index2);
            }
            index1 = i;
         } else if (words[i].equals(word2)) {
            if (index1 != -1) {
               res = Math.min(res, i - index1);
            }
            index2 = i;
         }
      }
      return res;

   }

   // 面试题 17.05. 字母与数字 (Find Longest Subarray LCCI) --前缀和
   public String[] findLongestSubarray(String[] array) {
      int count = 0;
      int index = -1;
      Map<Integer, Integer> firstIndex = new HashMap<>();
      firstIndex.put(0, -1);
      int cur = 0;
      for (int i = 0; i < array.length; ++i) {
         cur += Character.isDigit(array[i].charAt(0)) ? 1 : -1;
         if (!firstIndex.containsKey(cur)) {
            firstIndex.put(cur, i);
         } else if (i - firstIndex.get(cur) > count) {
            count = i - firstIndex.get(cur);
            index = firstIndex.get(cur) + 1;
         }
      }
      if (index == -1) {
         return new String[] {};
      }
      return Arrays.copyOfRange(array, index, index + count);

   }

   // 523. 连续的子数组和 (Continuous Subarray Sum) --前缀和
   public boolean checkSubarraySum(int[] nums, int k) {
      Map<Integer, Integer> map = new HashMap<>();
      map.put(0, -1);
      int preSum = 0;
      for (int i = 0; i < nums.length; ++i) {
         preSum = (preSum + nums[i]) % k;
         if (!map.containsKey(preSum)) {
            map.put(preSum, i);
         } else if (i - map.get(preSum) >= 2) {
            return true;
         }
      }
      return false;

   }

   // 1010. 总持续时间可被 60 整除的歌曲 (Pairs of Songs With Total Durations Divisible by 60)
   public int numPairsDivisibleBy60(int[] time) {
      int[] counts = new int[60];
      for (int i = 0; i < time.length; ++i) {
         time[i] %= 60;
         ++counts[time[i]];
      }
      int res = 0;
      for (int i = 0; i < time.length; ++i) {
         if (time[i] > 0 && time[i] < 30) {
            res += counts[60 - time[i]];
         }
      }
      res += counts[0] * (counts[0] - 1) / 2;
      res += counts[30] * (counts[30] - 1) / 2;
      return res;

   }

   // 1010. 总持续时间可被 60 整除的歌曲 (Pairs of Songs With Total Durations Divisible by 60)
   public int numPairsDivisibleBy60_2(int[] time) {
      int[] counts = new int[60];
      int res = 0;
      for (int i = 0; i < time.length; ++i) {
         int t = time[i] % 60;
         res += counts[t];
         int remain = t == 0 ? 0 : 60 - t;
         ++counts[remain];
      }
      return res;
   }

   // 1590. 使数组和能被 P 整除 (Make Sum Divisible by P) --前缀和
   public int minSubarray(int[] nums, int p) {
      int n = nums.length;
      int res = n;
      int m = 0;
      for (int x : nums) {
         m += x;
         m %= p;
      }
      if (m == 0) {
         return 0;
      }
      Map<Integer, Integer> map = new HashMap<>();
      map.put(0, -1);
      int pre = 0;
      for (int i = 0; i < n; ++i) {
         pre += nums[i];
         pre %= p;
         res = Math.min(res, i - map.getOrDefault(((pre - m) % p + p) % p, -n));
         map.put(pre, i);
      }
      return res < n ? res : -1;
   }

   // 1984. 学生分数的最小差值 (Minimum Difference Between Highest and Lowest of K Scores)
   public int minimumDifference(int[] nums, int k) {
      Arrays.sort(nums);
      int res = Integer.MAX_VALUE;
      for (int i = 0; i + k - 1 < nums.length; ++i) {
         res = Math.min(res, nums[i + k - 1] - nums[i]);
      }
      return res;
   }

   // 1805. 字符串中不同整数的数目 (Number of Different Integers in a String)
   public int numDifferentIntegers(String word) {
      int i = 0;
      int n = word.length();
      Set<String> set = new HashSet<>();
      while (i < n) {
         if (i < n && Character.isLetter(word.charAt(i))) {
            ++i;
            continue;
         }
         int j = i;
         while (j < n && word.charAt(j) == '0') {
            ++j;
         }
         i = j;
         while (j < n && Character.isDigit(word.charAt(j))) {
            ++j;
         }
         if (i == j) {
            set.add("0");
         } else {
            set.add(word.substring(i, j));
         }
         i = j;
      }
      return set.size();

   }

   // 860. 柠檬水找零 (Lemonade Change)
   public boolean lemonadeChange(int[] bills) {
      int count5 = 0;
      int count10 = 0;
      for (int bill : bills) {
         switch (bill) {
            case 5:
               ++count5;
               break;
            case 10:
               if (count5 == 0) {
                  return false;
               }
               --count5;
               ++count10;
               break;
            case 20:
               if (count10 >= 1 && count5 >= 1) {
                  --count10;
                  --count5;
               } else if (count5 >= 3) {
                  count5 -= 3;
               } else {
                  return false;
               }
               break;
         }
      }
      return true;

   }

   // 1839. 所有元音按顺序排布的最长子字符串 (Longest Substring Of All Vowels in Order) --双指针+滑动窗口
   public int longestBeautifulSubstring(String word) {
      char[] chars = word.toCharArray();
      int left = 0;
      int right = 0;
      int res = 0;
      while (left < chars.length) {
         while (left < chars.length && chars[left] != 'a') {
            ++left;
         }
         int mask = 0;
         mask |= 1 << 0;
         right = left + 1;

         while (right < chars.length && chars[right - 1] <= chars[right]) {
            if (chars[right] == 'a') {
               mask |= 1 << 0;
            } else if (chars[right] == 'e') {
               mask |= 1 << 1;
            } else if (chars[right] == 'i') {
               mask |= 1 << 2;
            } else if (chars[right] == 'o') {
               mask |= 1 << 3;
            } else if (chars[right] == 'u') {
               mask |= 1 << 4;
            }
            ++right;
         }
         if (mask == 0b11111) {
            res = Math.max(res, right - left);
         }
         left = right;
      }
      return res;

   }

   // 3. 无重复字符的最长子串 (Longest Substring Without Repeating Characters)
   // 剑指 Offer 48. 最长不含重复字符的子字符串
   // 剑指 Offer II 016. 不含重复字符的最长子字符串
   public int lengthOfLongestSubstring(String s) {
      int[] counts = new int[128];
      int res = 0;
      int left = 0;
      int right = 0;
      char[] arr = s.toCharArray();
      int n = s.length();
      while (right < n) {
         ++counts[arr[right]];
         while (counts[arr[right]] > 1) {
            --counts[arr[left++]];
         }
         res = Math.max(res, right - left + 1);
         ++right;
      }
      return res;
   }

   // 76. 最小覆盖子串 (Minimum Window Substring) --滑动窗口
   // 剑指 Offer II 017. 含有所有字符的最短字符串
   public String minWindow(String s, String t) {
      int[] cnt = new int[128];
      int less = 0;
      for (char c : t.toCharArray()) {
         if (--cnt[c] == -1) {
            --less;
         }
      }
      int resLeft = -1;
      int resRight = -1;
      int left = 0;
      for (int right = 0; right < s.length(); ++right) {
         if (++cnt[s.charAt(right)] == 0) {
            ++less;
         }
         while (less == 0) {
            if (resLeft == -1 || right - left < resRight - resLeft) {
               resLeft = left;
               resRight = right;
            }
            if (--cnt[s.charAt(left)] == -1) {
               --less;
            }
            ++left;
         }
      }
      return resRight == -1 ? "" : s.substring(resLeft, resRight + 1);

   }

   // 209. 长度最小的子数组 (Minimum Size Subarray Sum) --O(n) 滑动窗口
   // 剑指 Offer II 008. 和大于等于 target 的最短子数组 --O(n)
   public int minSubArrayLen(int target, int[] nums) {
      int n = nums.length;
      int res = n + 1;
      int sum = 0;
      int i = 0;
      int j = 0;
      while (j < n) {
         sum += nums[j];
         while (sum >= target) {
            res = Math.min(res, j - i + 1);
            sum -= nums[i++];
         }
         ++j;
      }
      return res == n + 1 ? 0 : res;

   }

   // 209. 长度最小的子数组 (Minimum Size Subarray Sum) --O(nlog(n)) 二分查找
   // 剑指 Offer II 008. 和大于等于 target 的最短子数组 --O(nlog(n))
   public int minSubArrayLen2(int target, int[] nums) {
      int n = nums.length;
      int[] prefix = new int[n + 1];
      for (int i = 1; i < n + 1; ++i) {
         prefix[i] = prefix[i - 1] + nums[i - 1];
      }
      int res = n + 1;
      for (int i = 1; i < n + 1; ++i) {
         int find = prefix[i] - target;
         int index = binarySearch209(prefix, find);
         if (index != -1) {
            res = Math.min(res, i - index);
         }
      }
      return res == n + 1 ? 0 : res;

   }

   private int binarySearch209(int[] prefix, int target) {
      int n = prefix.length;
      int res = -1;
      int left = 0;
      int right = n - 1;
      while (left <= right) {
         int mid = left + ((right - left) >>> 1);
         if (prefix[mid] <= target) {
            res = mid;
            left = mid + 1;
         } else {
            right = mid - 1;
         }
      }
      return res;

   }

   // 219. 存在重复元素 II (Contains Duplicate II) --哈希表
   public boolean containsNearbyDuplicate(int[] nums, int k) {
      Map<Integer, Integer> map = new HashMap<>();
      for (int i = 0; i < nums.length; ++i) {
         if (map.containsKey(nums[i])) {
            if (i - map.get(nums[i]) <= k) {
               return true;
            }
         }
         map.put(nums[i], i);
      }
      return false;

   }

   // 424. 替换后的最长重复字符 (Longest Repeating Character Replacement)
   public int characterReplacement(String s, int k) {
      int n = s.length();
      int res = 0;
      int mx = 0;
      int left = 0;
      int[] cnts = new int[26];
      for (int right = 0; right < n; ++right) {
         mx = Math.max(mx, ++cnts[s.charAt(right) - 'A']);
         while (right - left + 1 - mx > k) {
            --cnts[s.charAt(left++) - 'A'];
         }
         res = Math.max(res, right - left + 1);
      }
      return res;

   }

   // 438. 找到字符串中所有字母异位词 (Find All Anagrams in a String) --滑动窗口
   // 剑指 Offer II 015. 字符串中的所有变位词
   public List<Integer> findAnagrams(String s, String p) {
      List<Integer> res = new ArrayList<>();
      if (p.length() > s.length()) {
         return res;
      }
      int[] needs = new int[26];
      for (char c : p.toCharArray()) {
         ++needs[c - 'a'];
      }
      int[] gives = new int[26];
      char[] chars = s.toCharArray();
      int count = p.length();
      int i = 0;
      while (i < count) {
         ++gives[chars[i++] - 'a'];
      }
      if (Arrays.equals(needs, gives)) {
         res.add(0);
      }
      while (i < chars.length) {
         --gives[chars[i - count] - 'a'];
         ++gives[chars[i] - 'a'];
         if (Arrays.equals(needs, gives)) {
            res.add(i - count + 1);
         }
         ++i;

      }
      return res;

   }

   // 438. 找到字符串中所有字母异位词 (Find All Anagrams in a String) --优化的滑动窗口
   // 剑指 Offer II 015. 字符串中的所有变位词
   public List<Integer> findAnagrams2(String s, String p) {
      List<Integer> res = new ArrayList<>();
      if (p.length() > s.length()) {
         return res;
      }
      int[] counts = new int[26];
      for (int i = 0; i < p.length(); ++i) {
         ++counts[s.charAt(i) - 'a'];
         --counts[p.charAt(i) - 'a'];
      }
      int diff = 0;
      for (int i = 0; i < counts.length; ++i) {
         if (counts[i] != 0) {
            ++diff;
         }
      }

      if (diff == 0) {
         res.add(0);
      }
      for (int i = p.length(); i < s.length(); ++i) {

         if (counts[s.charAt(i - p.length()) - 'a'] == 1) {
            --diff;
         } else if (counts[s.charAt(i - p.length()) - 'a'] == 0) {
            ++diff;
         }
         --counts[s.charAt(i - p.length()) - 'a'];

         if (counts[s.charAt(i) - 'a'] == -1) {
            --diff;
         } else if (counts[s.charAt(i) - 'a'] == 0) {
            ++diff;
         }
         ++counts[s.charAt(i) - 'a'];

         if (diff == 0) {
            res.add(i - p.length() + 1);
         }
      }
      return res;

   }

   // 567. 字符串的排列 (Permutation in String)
   // 剑指 Offer II 014. 字符串中的变位词 --滑动窗口
   public boolean checkInclusion(String s1, String s2) {
      int n1 = s1.length();
      int n2 = s2.length();
      if (n1 > n2) {
         return false;
      }
      int[] cnt1 = new int[26];
      for (char c : s1.toCharArray()) {
         ++cnt1[c - 'a'];
      }
      int[] cnt2 = new int[26];
      for (int i = 0; i < n2; ++i) {
         ++cnt2[s2.charAt(i) - 'a'];
         if (i >= n1) {
            --cnt2[s2.charAt(i - n1) - 'a'];
         }
         if (i >= n1 - 1 && Arrays.equals(cnt1, cnt2)) {
            return true;
         }
      }
      return false;
   }

   // 567. 字符串的排列 (Permutation in String)
   // 剑指 Offer II 014. 字符串中的变位词 --计数
   public boolean checkInclusion2(String s1, String s2) {
      if (s1.length() > s2.length()) {
         return false;
      }
      int diff = 0;
      int[] counts = new int[26];
      int n = s1.length();
      for (int i = 0; i < n; ++i) {
         --counts[s1.charAt(i) - 'a'];
         ++counts[s2.charAt(i) - 'a'];
      }
      for (int count : counts) {
         if (count != 0) {
            ++diff;
         }
      }
      if (diff == 0) {
         return true;
      }
      for (int i = n; i < s2.length(); ++i) {
         int x = s2.charAt(i - n) - 'a';
         int y = s2.charAt(i) - 'a';
         if (counts[x] == 0) {
            ++diff;
         }
         --counts[x];
         if (counts[x] == 0) {
            --diff;
         }
         if (counts[y] == 0) {
            ++diff;
         }
         ++counts[y];
         if (counts[y] == 0) {
            --diff;
         }
         if (diff == 0) {
            return true;
         }
      }
      return false;

   }

   // 567. 字符串的排列 (Permutation in String)
   // 剑指 Offer II 014. 字符串中的变位词 -- 双指针
   public boolean checkInclusion3(String s1, String s2) {
      if (s1.length() > s2.length()) {
         return false;
      }
      int[] counts = new int[26];
      for (char c : s1.toCharArray()) {
         --counts[c - 'a'];
      }
      int n = s1.length();
      int left = 0;
      for (int right = 0; right < s2.length(); ++right) {
         int x = s2.charAt(right) - 'a';
         ++counts[x];
         while (counts[x] > 0) {
            --counts[s2.charAt(left) - 'a'];
            ++left;
         }
         if (right - left + 1 == n) {
            return true;
         }
      }
      return false;
   }

   // 643. 子数组最大平均数 I (Maximum Average Subarray I)
   public double findMaxAverage(int[] nums, int k) {
      Double res = (double) Integer.MIN_VALUE;
      Double sum = 0D;
      for (int i = 0; i < nums.length; ++i) {
         sum += nums[i];
         if (i >= k) {
            sum -= nums[i - k];
         }
         if (i >= k - 1) {
            res = Math.max(res, sum / k);
         }
      }
      return res;

   }

   // 713. 乘积小于K的子数组 (Subarray Product Less Than K)
   // 剑指 Offer II 009. 乘积小于 K 的子数组 --双指针+滑动窗口
   public int numSubarrayProductLessThanK(int[] nums, int k) {
      int n = nums.length;
      int res = 0;
      int i = 0;
      int j = 0;
      int mul = 1;
      while (j < n) {
         mul *= nums[j];
         while (i <= j && mul >= k) {
            mul /= nums[i++];
         }
         res += j - i + 1;
         ++j;
      }
      return res;

   }

   // 984. 不含 AAA 或 BBB 的字符串 (String Without AAA or BBB)
   public String strWithout3a3b(int a, int b) {
      StringBuilder res = new StringBuilder();
      while (a > 0 && b > 0) {
         if (a > b) {
            res.append("aab");
            a -= 2;
            b -= 1;
         } else if (a < b) {
            res.append("bba");
            a -= 1;
            b -= 2;
         } else {
            res.append("ab");
            a -= 1;
            b -= 1;
         }
      }
      while (a-- > 0) {
         res.append('a');
      }
      while (b-- > 0) {
         res.append('b');
      }
      return res.toString();

   }

   // 1417. 重新格式化字符串 (Reformat The String)
   public String reformat(String s) {
      int digitCount = 0;
      int alphabetCount = 0;
      for (char c : s.toCharArray()) {
         if (Character.isDigit(c)) {
            ++digitCount;
         } else {
            ++alphabetCount;
         }
      }
      if (Math.abs(digitCount - alphabetCount) > 1) {
         return "";
      }
      char[] res = new char[s.length()];
      int i = 0;
      int j = 0;
      if (alphabetCount >= digitCount) {
         i = 0;
         j = 1;
      } else {
         i = 1;
         j = 0;
      }
      for (char c : s.toCharArray()) {
         if (Character.isLetter(c)) {
            res[i] = c;
            i += 2;
         } else {
            res[j] = c;
            j += 2;
         }
      }
      return String.valueOf(res);

   }

   // 1417. 重新格式化字符串 (Reformat The String)
   public String reformat2(String s) {
      List<Character> characters = new ArrayList<>();
      List<Character> integers = new ArrayList<>();
      StringBuilder res = new StringBuilder();
      for (char c : s.toCharArray()) {
         if (Character.isDigit(c)) {
            integers.add(c);
         } else {
            characters.add(c);
         }
      }
      if (Math.abs(characters.size() - integers.size()) > 1) {
         return "";
      }

      int i = 0;
      int j = 0;
      while (i < characters.size() && j < integers.size()) {
         res.append(characters.get(i++));
         res.append(integers.get(j++));
      }
      if (i < characters.size()) {
         res.append(characters.get(i));
      }
      if (j < integers.size()) {
         res.insert(0, String.valueOf(integers.get(j)));
      }
      return res.toString();
   }

   // 796. 旋转字符串 (Rotate String) --还需掌握 Rabin-Karp 字符串哈希 ； KMP 算法
   public boolean rotateString(String s, String goal) {
      return s.length() == goal.length() && (s + s).contains(goal);
   }

   // 面试题 17.18. 最短超串 (Shortest Supersequence LCCI)
   public int[] shortestSeq(int[] big, int[] small) {
      Map<Integer, Integer> map = new HashMap<>();
      for (int num : small) {
         map.put(num, -1);
      }
      int count = small.length;
      int[] res = new int[] { 0, big.length - 1 };
      for (int i = 0; i < big.length; ++i) {
         if (map.containsKey(big[i])) {
            if (map.get(big[i]) == -1) {
               --count;
            }
            map.put(big[i], i);
         }
         if (count <= 0) {
            int min = Collections.min(map.values());
            if (i - min < res[1] - res[0]) {
               res[1] = i;
               res[0] = min;
            }
         }
      }
      if (count > 0) {
         return new int[0];
      }
      return res;

   }

   // 面试题 17.18. 最短超串 (Shortest Supersequence LCCI)
   public int[] shortestSeq2(int[] big, int[] small) {
      if (small.length > big.length) {
         return new int[0];
      }
      Map<Integer, Integer> need = new HashMap<>();
      for (int num : small) {
         need.put(num, need.getOrDefault(num, 0) + 1);
      }
      int needCount = small.length;
      Map<Integer, Integer> give = new HashMap<>();
      int left = 0;
      int right = 0;
      int giveCount = 0;
      int minLength = big.length + 1;
      int[] res = new int[0];
      while (right < big.length) {
         if (need.containsKey(big[right])) {
            if (give.getOrDefault(big[right], 0) < need.get(big[right])) {
               ++giveCount;
            }
            give.put(big[right], give.getOrDefault(big[right], 0) + 1);
         }
         while (giveCount == needCount) {
            if (right - left + 1 < minLength) {
               minLength = right - left + 1;
               res = new int[] { left, right };
            }
            if (need.containsKey(big[left])) {
               if (give.get(big[left]) == need.get(big[left])) {
                  --giveCount;
               }
               give.put(big[left], give.get(big[left]) - 1);
            }
            ++left;
         }
         ++right;
      }
      return res;

   }

   // 1961. 检查字符串是否为数组前缀 (Check If String Is a Prefix of Array)
   public boolean isPrefixString(String s, String[] words) {
      int index = 0;
      char[] chars = s.toCharArray();
      for (String word : words) {
         for (char c : word.toCharArray()) {
            if (index < chars.length && chars[index] == c) {
               ++index;
            } else {
               return false;
            }
         }
         if (index == chars.length) {
            return true;
         }
      }
      return false;

   }

   // 1556. 千位分隔数 (Thousand Separator)
   public String thousandSeparator(int n) {
      int count = 0;
      StringBuilder res = new StringBuilder();
      do {
         res.append(n % 10);
         ++count;
         n /= 10;
         if (count % 3 == 0 && n != 0) {
            res.append('.');
         }
      } while (n != 0);
      return res.reverse().toString();

   }

   // 1909. 删除一个元素使数组严格递增 (Remove One Element to Make the Array Strictly
   // Increasing)
   public boolean canBeIncreasing(int[] nums) {
      for (int i = 1; i < nums.length; ++i) {
         if (nums[i - 1] >= nums[i]) {
            return check1909(i - 1, nums) || check1909(i, nums);
         }
      }
      return true;

   }

   private boolean check1909(int index, int[] nums) {
      for (int i = 1; i < index; ++i) {
         if (nums[i - 1] >= nums[i]) {
            return false;
         }
      }
      for (int i = index + 2; i < nums.length; ++i) {
         if (nums[i - 1] >= nums[i]) {
            return false;
         }
      }
      if (index - 1 >= 0 && index + 1 < nums.length) {
         return nums[index - 1] < nums[index + 1];
      }
      return true;
   }

   // 2027. 转换字符串的最少操作次数 (Minimum Moves to Convert String)
   public int minimumMoves(String s) {
      int res = 0;
      char[] chars = s.toCharArray();
      int i = 0;
      while (i < chars.length) {
         if (chars[i] == 'X') {
            ++res;
            i += 3;
         } else {
            ++i;
         }
      }
      return res;

   }

   // 1910. 删除一个字符串中所有出现的给定子字符串 (Remove All Occurrences of a Substring) --还需掌握KMP算法
   public String removeOccurrences(String s, String part) {
      StringBuilder res = new StringBuilder(s);
      int n = part.length();
      while (res.indexOf(part) != -1) {
         int start = res.indexOf(part);
         res.delete(start, start + n);
      }
      return res.toString();

   }

   // 1957. 删除字符使字符串变好 (Delete Characters to Make Fancy String)
   public String makeFancyString(String s) {
      StringBuilder res = new StringBuilder();
      for (char c : s.toCharArray()) {
         int n = res.length();
         if (n >= 2 && res.charAt(n - 1) == c && res.charAt(n - 2) == c) {
            continue;
         }
         res.append(c);
      }
      return res.toString();

   }

   // 1663. 具有给定数值的最小字符串 (Smallest String With A Given Numeric Value) --贪心
   public String getSmallestString(int n, int k) {
      char[] res = new char[n];
      // 初始都设置为'a'
      Arrays.fill(res, 'a');
      // 'z'的个数
      int countZ = (k - n) / 25;
      if (countZ > 0) {
         Arrays.fill(res, n - countZ, n, 'z');
      }
      // z之前的第i位应该放置的字母
      int i = n - countZ - 1;
      if (i >= 0) {
         int left = (k - n) % 25;
         res[i] = (char) ('a' + left);
      }
      return String.valueOf(res);
   }

   // 718. 最长重复子数组 (Maximum Length of Repeated Subarray)
   // --滑动窗口 还需掌握 “二分查找 + 哈希”
   public int findLength(int[] nums1, int[] nums2) {
      int m = nums1.length;
      int n = nums2.length;
      int res = 0;
      for (int i = 0; i < m; ++i) {
         int minLen = Math.min(m - i, n);
         res = Math.max(res, getMaxLength718(nums1, nums2, i, 0, minLen));
      }
      for (int i = 0; i < n; ++i) {
         int minLen = Math.min(m, n - i);
         res = Math.max(res, getMaxLength718(nums1, nums2, 0, i, minLen));
      }
      return res;

   }

   private int getMaxLength718(int[] nums1, int[] nums2, int add1, int add2, int minLen) {
      int max = 0;
      int count = 0;
      for (int i = 0; i < minLen; ++i) {
         if (nums1[add1 + i] != nums2[add2 + i]) {
            count = 0;
         } else {
            max = Math.max(max, ++count);
         }
      }
      return max;
   }

   // 718. 最长重复子数组 (Maximum Length of Repeated Subarray)
   private int n1_718;
   private int n2_718;
   private int[] nums1_718;
   private int[] nums2_718;
   private int[][] memo718;

   public int findLength2(int[] nums1, int[] nums2) {
      this.n1_718 = nums1.length;
      this.n2_718 = nums2.length;
      this.nums1_718 = nums1;
      this.nums2_718 = nums2;
      this.memo718 = new int[n1_718][n2_718];
      for (int i = 0; i < n1_718; ++i) {
         Arrays.fill(memo718[i], -1);
      }
      int res = 0;
      for (int i = 0; i < n1_718; ++i) {
         for (int j = 0; j < n2_718; ++j) {
            res = Math.max(res, dfs718(i, j));
         }
      }
      return res;

   }

   private int dfs718(int i, int j) {
      if (i == n1_718 || j == n2_718) {
         return 0;
      }
      if (memo718[i][j] != -1) {
         return memo718[i][j];
      }
      if (nums1_718[i] != nums2_718[j]) {
         return memo718[i][j] = 0;
      }
      return memo718[i][j] = dfs718(i + 1, j + 1) + 1;
   }

   // 978. 最长湍流子数组 (Longest Turbulent Subarray) --滑动窗口
   public int maxTurbulenceSize(int[] arr) {
      int res = 1;
      int left = 0;
      int right = 0;
      while (right < arr.length - 1) {
         if (left == right) {
            if (arr[left] == arr[left + 1]) {
               ++left;
            }
            ++right;
         } else {
            if (arr[right - 1] < arr[right] && arr[right] > arr[right + 1]) {
               ++right;
            } else if (arr[right - 1] > arr[right] && arr[right] < arr[right + 1]) {
               ++right;
            } else {
               left = right;
            }
         }
         res = Math.max(res, right - left + 1);
      }
      return res;

   }

   // 1031. 两个非重叠子数组的最大和 (Maximum Sum of Two Non-Overlapping Subarrays)
   public int maxSumTwoNoOverlap(int[] nums, int firstLen, int secondLen) {
      return Math.max(f1031(nums, firstLen, secondLen), f1031(nums, secondLen, firstLen));
   }

   private int f1031(int[] nums, int len1, int len2) {
      int n = nums.length;
      int max = 0;
      int maxSum1 = 0;
      int sum1 = 0;
      int sum2 = 0;
      for (int i = 0; i < len1; ++i) {
         sum1 += nums[i];
      }
      for (int i = len1; i < len1 + len2; ++i) {
         sum2 += nums[i];
      }
      maxSum1 = Math.max(maxSum1, sum1);
      max = Math.max(max, maxSum1 + sum2);
      for (int i = len1 + len2; i < n; ++i) {
         sum2 -= nums[i - len2];
         sum2 += nums[i];
         sum1 -= nums[i - len1 - len2];
         sum1 += nums[i - len2];
         maxSum1 = Math.max(maxSum1, sum1);
         max = Math.max(max, maxSum1 + sum2);
      }
      return max;
   }

   // 1052. 爱生气的书店老板 (Grumpy Bookstore Owner)
   public int maxSatisfied(int[] customers, int[] grumpy, int minutes) {
      int s1 = 0;
      int s2 = 0;
      int s = 0;
      int n = customers.length;
      for (int i = 0; i < n; ++i) {
         if (grumpy[i] == 0) {
            s1 += customers[i];
         } else {
            s += customers[i];
         }
         if (i >= minutes && grumpy[i - minutes] == 1) {
            s -= customers[i - minutes];
         }
         if (i >= minutes - 1) {
            s2 = Math.max(s2, s);
         }
      }
      return s1 + s2;

   }

   // 1208. 尽可能使字符串相等 (Get Equal Substrings Within Budget) --滑动窗口 还需掌握二分查找法
   public int equalSubstring(String s, String t, int maxCost) {
      int[] diff = new int[s.length()];
      for (int i = 0; i < s.length(); ++i) {
         diff[i] = Math.abs(s.charAt(i) - t.charAt(i));
      }
      int left = 0;
      int right = 0;
      int res = 0;
      int prefix = 0;
      while (right < diff.length) {
         prefix += diff[right];
         while (prefix > maxCost) {
            prefix -= diff[left++];
         }
         res = Math.max(res, right - left + 1);
         ++right;
      }
      return res;

   }

   // 1343. 大小为 K 且平均值大于等于阈值的子数组数目 (Number of Sub-arrays of Size K and Average
   // Greater than or Equal to Threshold)
   public int numOfSubarrays(int[] arr, int k, int threshold) {
      int res = 0;
      int sum = 0;
      for (int i = 0; i < arr.length; ++i) {
         sum += arr[i];
         if (i >= k) {
            sum -= arr[i - k];
         }
         if (i >= k - 1 && sum / k >= threshold) {
            ++res;
         }
      }
      return res;

   }

   // 1358. 包含所有三种字符的子字符串数目 (Number of Substrings Containing All Three Characters)
   // --滑动窗口
   public int numberOfSubstrings(String s) {
      int left = 0;
      int right = 0;
      int res = 0;
      int[] counts = new int[3];
      while (right < s.length()) {
         ++counts[s.charAt(right) - 'a'];
         while (counts[0] > 0 && counts[1] > 0 && counts[2] > 0) {
            res += s.length() - right;
            --counts[s.charAt(left++) - 'a'];
         }
         ++right;
      }
      return res;

   }

   // 1423. 可获得的最大点数 (Maximum Points You Can Obtain from Cards) --滑动窗口
   public int maxScore(int[] cardPoints, int k) {
      int res = Integer.MAX_VALUE;
      int sum = 0;
      int s = 0;
      int n = cardPoints.length;
      for (int i = 0; i < n; ++i) {
         sum += cardPoints[i];
         s += cardPoints[i];
         if (i >= n - k) {
            s -= cardPoints[i - (n - k)];
         }
         if (i >= n - k - 1) {
            res = Math.min(res, s);
         }
      }
      return sum - res;

   }

   // 1456. 定长子串中元音的最大数目 (Maximum Number of Vowels in a Substring of Given Length)
   // --滑动窗口
   public int maxVowels(String s, int k) {
      int res = 0;
      int cur = 0;
      int mask = 0;
      for (char c : "aeiou".toCharArray()) {
         mask |= 1 << (c - 'a');
      }
      char[] chars = s.toCharArray();
      int n = chars.length;
      for (int i = 0; i < n; ++i) {
         if (((mask >> (chars[i] - 'a')) & 1) == 1) {
            ++cur;
         }
         if (i >= k - 1) {
            if (i >= k && ((mask >> (chars[i - k] - 'a')) & 1) == 1) {
               --cur;
            }
            res = Math.max(res, cur);
         }
      }
      return res;

   }

   // 1493. 删掉一个元素以后全为 1 的最长子数组 (Longest Subarray of 1's After Deleting One
   // Element)
   public int longestSubarray(int[] nums) {
      int left = 0;
      int right = 0;
      int res = 0;
      int countZero = 0;
      while (right < nums.length) {
         if (nums[right] == 0) {
            ++countZero;
         }
         while (countZero > 1) {
            if (nums[left++] == 0) {
               --countZero;
            }
         }
         res = Math.max(res, right - left);
         ++right;
      }

      return res;
   }

   // 1493. 删掉一个元素以后全为 1 的最长子数组 (Longest Subarray of 1's After Deleting One
   // Element)
   public int longestSubarray2(int[] nums) {
      int a = 0;
      int b = 0;
      int n = nums.length;
      int res = 0;
      for (int num : nums) {
         if (num == 1) {
            ++a;
            ++b;
            res = Math.max(res, a);
         } else {
            a = b;
            b = 0;
         }
      }
      return Math.min(n - 1, res);

   }

   // 1658. 将 x 减到 0 的最小操作数 (Minimum Operations to Reduce X to Zero)
   public int minOperations(int[] nums, int x) {
      int n = nums.length;
      int sum = 0;
      for (int i = 0; i < n; ++i) {
         sum += nums[i];
      }
      int target = sum - x;
      if (target < 0) {
         return -1;
      }
      if (target == 0) {
         return n;
      }
      int res = Integer.MAX_VALUE;
      int i = 0;
      int j = 0;
      int windowSum = 0;
      while (j < n) {
         windowSum += nums[j];
         while (windowSum > target) {
            windowSum -= nums[i++];
         }
         if (windowSum == target) {
            res = Math.min(res, n - (j - i + 1));
         }
         ++j;
      }
      return res == Integer.MAX_VALUE ? -1 : res;

   }

   // 1695. 删除子数组的最大得分 (Maximum Erasure Value)
   public int maximumUniqueSubarray(int[] nums) {
      int[] cnts = new int[Arrays.stream(nums).max().getAsInt() + 1];
      int j = 0;
      int res = 0;
      int s = 0;
      for (int x : nums) {
         ++cnts[x];
         s += x;
         while (cnts[x] > 1) {
            --cnts[nums[j]];
            s -= nums[j];
            ++j;
         }
         res = Math.max(res, s);
      }
      return res;

   }

   // 1876.长度为三且各字符不同的子字符串 (Substrings of Size Three with Distinct Characters)
   public int countGoodSubstrings(String s) {
      int i = 2;
      int res = 0;
      while (i < s.length()) {
         if (s.charAt(i - 2) != s.charAt(i - 1) && s.charAt(i - 1) != s.charAt(i) && s.charAt(i - 2) != s.charAt(i)) {
            ++res;
         }
         ++i;
      }
      return res;

   }

   // 2091. 从数组中移除最大值和最小值 (Removing Minimum and Maximum From Array) --贪心
   public int minimumDeletions(int[] nums) {
      int min = Integer.MAX_VALUE;
      int max = Integer.MIN_VALUE;
      int minIndex = 0;
      int maxIndex = nums.length - 1;
      for (int i = 0; i < nums.length; ++i) {
         if (nums[i] > max) {
            max = nums[i];
            maxIndex = i;
         }
         if (nums[i] < min) {
            min = nums[i];
            minIndex = i;
         }
      }
      // 从左边删
      int count1 = Math.max(minIndex, maxIndex) + 1;
      // 从右边删
      int count2 = nums.length - Math.min(minIndex, maxIndex);
      // 从两侧删
      int count3 = nums.length - (Math.abs(maxIndex - minIndex) - 1);

      return Math.min(count1, Math.min(count2, count3));

   }

   // 2105. 给植物浇水 II (Watering Plants II)
   public int minimumRefill(int[] plants, int capacityA, int capacityB) {
      int res = 0;
      int leftA = capacityA;
      int leftB = capacityB;
      int i = 0;
      int j = plants.length - 1;
      while (i < j) {
         if (leftA < plants[i]) {
            ++res;
            leftA = capacityA;
         }
         leftA -= plants[i++];

         if (leftB < plants[j]) {
            ++res;
            leftB = capacityB;
         }
         leftB -= plants[j--];
      }

      if (i == j) {
         if (Math.max(leftA, leftB) < plants[i]) {
            ++res;
         }
      }
      return res;

   }

   // LCR 166. 珠宝的最高价值
   private int m_offer47;
   private int n_offer47;
   private int[][] grid_offer47;
   private int[][] memo_offer47;

   public int jewelleryValue(int[][] grid) {
      this.m_offer47 = grid.length;
      this.n_offer47 = grid[0].length;
      this.grid_offer47 = grid;
      this.memo_offer47 = new int[m_offer47][n_offer47];
      return dfs_offer47(0, 0);

   }

   private int dfs_offer47(int i, int j) {
      if (i == m_offer47 || j == n_offer47) {
         return 0;
      }
      if (memo_offer47[i][j] != 0) {
         return memo_offer47[i][j];
      }
      return memo_offer47[i][j] = Math.max(dfs_offer47(i + 1, j), dfs_offer47(i, j + 1)) + grid_offer47[i][j];
   }

   // LCR 166. 珠宝的最高价值
   public int maxValue2(int[][] grid) {
      int m = grid.length;
      int n = grid[0].length;
      for (int i = 0; i < m; ++i) {
         for (int j = 0; j < n; ++j) {
            if (i == 0 && j > 0) {
               grid[i][j] += grid[i][j - 1];
            } else if (j == 0 && i > 0) {
               grid[i][j] += grid[i - 1][j];
            } else if (i > 0 && j > 0) {
               grid[i][j] += Math.max(grid[i - 1][j], grid[i][j - 1]);
            }
         }
      }
      return grid[m - 1][n - 1];

   }

   // 2094. 找出 3 位偶数 (Finding 3-Digit Even Numbers)
   private int[] cnts2094;
   private Set<Integer> set2094;
   private int cur2094;

   public int[] findEvenNumbers(int[] digits) {
      cnts2094 = new int[10];
      for (int d : digits) {
         ++cnts2094[d];
      }
      cur2094 = 0;
      set2094 = new HashSet<>();
      dfs2094(0);
      int[] res = new int[set2094.size()];
      int i = 0;
      for (int x : set2094) {
         res[i++] = x;
      }
      Arrays.sort(res);
      return res;

   }

   private void dfs2094(int i) {
      if (i == 3) {
         set2094.add(cur2094);
         return;
      }
      for (int j = 0; j < 10; ++j) {
         if (i == 0 && j == 0 || cnts2094[j] == 0 || i == 2 && j % 2 == 1) {
            continue;
         }
         cur2094 = cur2094 * 10 + j;
         --cnts2094[j];
         dfs2094(i + 1);
         ++cnts2094[j];
         cur2094 /= 10;
      }
   }

   // 2092. 找出知晓秘密的所有专家 (Find All People With Secret) --dfs
   public List<Integer> findAllPeople2(int n, int[][] meetings, int firstPerson) {
      boolean[] knows = new boolean[n];
      knows[0] = true;
      knows[firstPerson] = true;
      Arrays.sort(meetings, new Comparator<int[]>() {

         @Override
         public int compare(int[] o1, int[] o2) {
            return o1[2] - o2[2];
         }

      });

      for (int i = 0; i < meetings.length; ++i) {
         int time = meetings[i][2];
         int j = i;
         while (j + 1 < meetings.length && meetings[j + 1][2] == time) {
            ++j;
         }
         Map<Integer, List<Integer>> g = new HashMap<>();
         Set<Integer> vis = new HashSet<>();
         for (int k = i; k <= j; ++k) {
            g.computeIfAbsent(meetings[k][0], o -> new ArrayList<>()).add(meetings[k][1]);
            g.computeIfAbsent(meetings[k][1], o -> new ArrayList<>()).add(meetings[k][0]);
         }
         for (int x : g.keySet()) {
            if (knows[x] && !vis.contains(x)) {
               dfs2092(x, knows, vis, g);
            }
         }
         i = j;
      }

      List<Integer> res = new ArrayList<>();
      for (int i = 0; i < knows.length; ++i) {
         if (knows[i]) {
            res.add(i);
         }
      }
      return res;
   }

   private void dfs2092(int x, boolean[] knows, Set<Integer> vis, Map<Integer, List<Integer>> g) {
      vis.add(x);
      knows[x] = true;
      for (int y : g.getOrDefault(x, new ArrayList<>())) {
         if (!vis.contains(y)) {
            dfs2092(y, knows, vis, g);
         }
      }
   }

   // 2092. 找出知晓秘密的所有专家 (Find All People With Secret) --并查集
   public List<Integer> findAllPeople(int n, int[][] meetings, int firstPerson) {
      UnionFind2092 unionFind = new UnionFind2092(n);
      unionFind.union(0, firstPerson);
      Arrays.sort(meetings, new Comparator<int[]>() {

         @Override
         public int compare(int[] o1, int[] o2) {
            return o1[2] - o2[2];
         }

      });

      int left = 0;
      int right = 0;
      while (right < meetings.length) {
         int time = meetings[right][2];
         while (right + 1 < meetings.length && time == meetings[right + 1][2]) {
            ++right;
         }
         for (int i = left; i <= right; ++i) {
            unionFind.union(meetings[i][0], meetings[i][1]);
         }
         for (int i = left; i <= right; ++i) {
            if (!unionFind.isConnected(meetings[i][0], 0)) {
               unionFind.isolate(meetings[i][0]);
               unionFind.isolate(meetings[i][1]);
            }
         }
         ++right;
         left = right;
      }
      List<Integer> res = new ArrayList<>();
      for (int i = 0; i < n; ++i) {
         if (unionFind.isConnected(0, i)) {
            res.add(i);
         }
      }
      return res;

   }

   public class UnionFind2092 {
      private int[] rank;
      private int[] parent;

      public UnionFind2092(int n) {
         rank = new int[n];
         Arrays.fill(rank, 1);
         parent = new int[n];
         for (int i = 0; i < n; ++i) {
            parent[i] = i;
         }
      }

      public int getRoot(int p) {
         if (p == parent[p]) {
            return p;
         }
         return parent[p] = getRoot(parent[p]);
      }

      public boolean isConnected(int p1, int p2) {
         return getRoot(p1) == getRoot(p2);
      }

      public void union(int p1, int p2) {
         int root1 = getRoot(p1);
         int root2 = getRoot(p2);
         if (root1 == root2) {
            return;
         }
         if (rank[root1] < rank[root2]) {
            parent[root1] = root2;
         } else {
            parent[root2] = root1;
            if (rank[root1] == rank[root2]) {
               ++rank[root1];
            }
         }
      }

      // 孤立节点
      public void isolate(int p) {
         if (parent[p] != p) {
            parent[p] = p;
            rank[p] = 1;
         }

      }

   }

   // 2076. 处理含限制条件的好友请求 (Process Restricted Friend Requests)
   public boolean[] friendRequests(int n, int[][] restrictions, int[][] requests) {
      boolean[] res = new boolean[requests.length];
      UnionFind2076 union = new UnionFind2076(n);
      for (int i = 0; i < requests.length; ++i) {
         int x = union.getRoot(requests[i][0]);
         int y = union.getRoot(requests[i][1]);
         if (x == y) {
            res[i] = true;
         } else {
            boolean flag = true;
            for (int[] restriction : restrictions) {
               int u = union.getRoot(restriction[0]);
               int v = union.getRoot(restriction[1]);
               if ((x == u && y == v) || (x == v && y == u)) {
                  flag = false;
                  break;
               }
            }
            if (flag) {
               res[i] = true;
               union.union(x, y);
            } else {
               res[i] = false;
            }
         }
      }
      return res;

   }

   public class UnionFind2076 {
      private int[] parent;
      private int[] rank;

      public UnionFind2076(int n) {
         parent = new int[n];
         for (int i = 0; i < n; ++i) {
            parent[i] = i;
         }
         rank = new int[n];
         Arrays.fill(rank, 1);
      }

      public int getRoot(int p) {
         if (parent[p] == p) {
            return p;
         }
         return parent[p] = getRoot(parent[p]);
      }

      public boolean isConnected(int p1, int p2) {
         return getRoot(p1) == getRoot(p2);
      }

      public void union(int p1, int p2) {
         int root1 = getRoot(p1);
         int root2 = getRoot(p2);
         if (root1 == root2) {
            return;
         }
         if (rank[root1] > rank[root2]) {
            parent[root2] = root1;
         } else {
            parent[root1] = root2;
            if (rank[root1] == rank[root2]) {
               ++rank[root2];
            }
         }
      }

   }

   // 面试题 17.14. Smallest K LCCI --还需掌握 堆；快排思想
   public int[] smallestK(int[] arr, int k) {
      Arrays.sort(arr);
      return Arrays.copyOfRange(arr, 0, k);

   }

   // 324. 摆动排序 II (Wiggle Sort II) --排序 时间O(nlog(n)) 空间O(n)
   // 还需掌握时间O(n) 空间O(1)的快速选择算法
   public void wiggleSort(int[] nums) {
      int[] copy = nums.clone();
      Arrays.sort(copy);
      int j = copy.length - 1;
      for (int i = 1; i < nums.length; i += 2) {
         nums[i] = copy[j--];
      }
      for (int i = 0; i < nums.length; i += 2) {
         nums[i] = copy[j--];
      }

   }

   // 324. 摆动排序 II (Wiggle Sort II) --桶排序 时间O(n) 空间O(n)
   // 还需掌握时间O(n) 空间O(1)的快速选择算法
   public void wiggleSort2(int[] nums) {
      int[] counts = new int[5001];
      for (int num : nums) {
         ++counts[num];
      }
      int j = 5000;
      for (int i = 1; i < nums.length; i += 2) {
         while (counts[j] == 0) {
            --j;
         }
         nums[i] = j;
         --counts[j];
      }
      for (int i = 0; i < nums.length; i += 2) {
         while (counts[j] == 0) {
            --j;
         }
         nums[i] = j;
         --counts[j];
      }
   }

   // 2103. 环和杆 (Rings and Rods)
   public int countPoints(String rings) {
      int[] mask = new int[3];
      for (int i = 0; i < rings.length(); i += 2) {
         if (rings.charAt(i) == 'R') {
            mask[0] |= 1 << (rings.charAt(i + 1) - '0');
         } else if (rings.charAt(i) == 'G') {
            mask[1] |= 1 << (rings.charAt(i + 1) - '0');
         } else {
            mask[2] |= 1 << (rings.charAt(i + 1) - '0');
         }
      }
      return Integer.bitCount(mask[0] & mask[1] & mask[2]);

   }

   // LCP 12. 小张刷题计划 --二分查找
   public int minTime(int[] time, int m) {
      int left = 0;
      int right = Integer.MAX_VALUE;
      int res = 0;
      while (left <= right) {
         int mid = left + ((right - left) >>> 1);
         if (checkLCP12(time, mid) <= m) {
            res = mid;
            right = mid - 1;
         } else {
            left = mid + 1;
         }
      }
      return res;

   }

   private int checkLCP12(int[] time, int t) {
      boolean help = true;
      int max = -1;
      int days = 1;
      int cur = t;
      for (int i = 0; i < time.length; ++i) {
         max = Math.max(max, time[i]);
         if (cur >= time[i]) {
            cur -= time[i];
         } else if (help) {
            help = false;
            cur += max;
            --i;
         } else {
            ++days;
            cur = t;
            --i;
            max = -1;
            help = true;
         }
      }
      return days;
   }

   // 794. 有效的井字游戏 (Valid Tic-Tac-Toe State)
   public boolean validTicTacToe(String[] board) {
      // 第0行表示X 第1行表示O
      // 0-2列表示行个数 3-5列表示列个数 6列表示正对角线个数 7表示负对角线个数
      int[][] counts = new int[2][8];
      int xCount = 0;
      int oCount = 0;
      for (int i = 0; i < board.length; ++i) {
         for (int j = 0; j < board[i].length(); ++j) {
            char c = board[i].charAt(j);
            if (c == 'X') {
               ++xCount;
               ++counts[0][i];
               ++counts[0][j + 3];
               if (i == j) {
                  ++counts[0][6];
               }
               if (i + j == 2) {
                  ++counts[0][7];
               }
            } else if (c == 'O') {
               ++oCount;
               ++counts[1][i];
               ++counts[1][j + 3];
               if (i == j) {
                  ++counts[1][6];
               }
               if (i + j == 2) {
                  ++counts[1][7];
               }
            }
         }
      }
      // X与O个数只能向相等 或 X比O多一（因为X先手）
      // 这是合法的必要条件
      if (xCount - oCount > 1 || xCount - oCount < 0) {
         return false;
      }
      int xMax = Arrays.stream(counts[0]).max().getAsInt();
      int oMax = Arrays.stream(counts[1]).max().getAsInt();
      // X赢了、但O与X数量相等 不合法
      // O赢了、但X比O数量还多 不合法
      if ((xMax == 3 && xCount == oCount) || (oMax == 3 && xCount > oCount)) {
         return false;
      }
      return true;

   }

   // 799. 香槟塔 (Champagne Tower)
   public double champagneTower(int poured, int query_row, int query_glass) {
      double[][] dp = new double[query_row + 2][query_row + 2];
      dp[0][0] = poured;
      for (int i = 0; i <= query_row; ++i) {
         for (int j = 0; j <= query_glass; ++j) {
            double q = Math.max(0d, dp[i][j] - 1) / 2;
            dp[i + 1][j] += q;
            dp[i + 1][j + 1] += q;
         }
      }
      return Math.min(1d, dp[query_row][query_glass]);

   }

   // 1438. 绝对差不超过限制的最长连续子数组 (Longest Continuous Subarray With Absolute Diff Less
   // Than or Equal to Limit
   public int longestSubarray(int[] nums, int limit) {
      int res = 0;
      TreeMap<Integer, Integer> map = new TreeMap<>();
      int i = 0;
      int j = 0;
      int n = nums.length;
      while (i < n) {
         map.merge(nums[i], 1, Integer::sum);
         while (map.lastKey() - map.firstKey() > limit) {
            map.merge(nums[j], -1, Integer::sum);
            if (map.get(nums[j]) == 0) {
               map.remove(nums[j]);
            }
            ++j;
         }
         res = Math.max(res, i - j + 1);
         ++i;
      }
      return res;
   }

   // 1438. 绝对差不超过限制的最长连续子数组 (Longest Continuous Subarray With Absolute Diff Less
   // Than or Equal to Limit
   public int longestSubarray2(int[] nums, int limit) {
      Deque<Integer> max = new LinkedList<>();
      Deque<Integer> min = new LinkedList<>();
      int left = 0;
      int right = 0;
      int res = 0;
      while (right < nums.length) {
         while (!max.isEmpty() && max.peekLast() < nums[right]) {
            max.pollLast();
         }
         max.offerLast(nums[right]);

         while (!min.isEmpty() && min.peekLast() > nums[right]) {
            min.pollLast();
         }
         min.offerLast(nums[right]);

         while (!max.isEmpty() && !min.isEmpty() && max.peekFirst() - min.peekFirst() > limit) {
            if (nums[left] == max.peekFirst()) {
               max.pollFirst();
            }
            if (nums[left] == min.peekFirst()) {
               min.pollFirst();
            }
            ++left;
         }
         res = Math.max(res, right - left + 1);
         ++right;
      }
      return res;

   }

   // 220. 存在重复元素 III (Contains Duplicate III)
   // 剑指 Offer II 057. 值和下标之差都在给定的范围内 --还需掌握 ：桶
   public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
      TreeSet<Long> set = new TreeSet<>();
      for (int i = 0; i < nums.length; ++i) {
         Long ceiling = set.ceiling((long) nums[i] - (long) t);
         if (ceiling != null && ceiling <= (long) nums[i] + (long) t) {
            return true;
         }
         set.add((long) nums[i]);
         if (i >= k) {
            set.remove((long) nums[i - k]);
         }
      }
      return false;
   }

   // 632. 最小区间 (Smallest Range Covering Elements from K Lists)
   public int[] smallestRange(List<List<Integer>> nums) {
      List<int[]> list = new ArrayList<>();
      for (int i = 0; i < nums.size(); ++i) {
         for (int num : nums.get(i)) {
            list.add(new int[] { num, i });
         }
      }
      Collections.sort(list, (o1, o2) -> o1[0] - o2[0]);
      int[] count = new int[nums.size()];
      int left = 0;
      int right = 0;
      int k = 0;
      int[] res = new int[2];
      while (right < list.size()) {
         if (count[list.get(right)[1]]++ == 0) {
            ++k;
         }
         while (k == count.length) {
            if ((res[0] == 0 && res[1] == 0) || (res[1] - res[0] > list.get(right)[0] - list.get(left)[0])) {
               res[0] = list.get(left)[0];
               res[1] = list.get(right)[0];
            }
            if (count[list.get(left)[1]] == 1) {
               --k;
            }
            --count[list.get(left++)[1]];
         }
         ++right;
      }
      return res;

   }

   // 2108. 找出数组中的第一个回文字符串 (Find First Palindromic String in the Array)
   public String firstPalindrome(String[] words) {
      for (String word : words) {
         if (checkPalindrome(word)) {
            return word;
         }
      }
      return "";

   }

   private boolean checkPalindrome(String word) {
      char[] chars = word.toCharArray();
      int left = 0;
      int right = chars.length - 1;
      while (left < right) {
         if (chars[left] != chars[right]) {
            return false;
         }
         ++left;
         --right;
      }
      return true;
   }

   // 1374. 生成每种字符都是奇数个的字符串 (Generate a String With Characters That Have Odd
   // Counts)
   public String generateTheString(int n) {
      char[] res = new char[n];
      Arrays.fill(res, 'a');
      if (n % 2 == 0) {
         res[n - 1] = 'b';
      }
      return String.valueOf(res);

   }

   // 2109. 向字符串添加空格 (Adding Spaces to a String)
   public String addSpaces(String s, int[] spaces) {
      int n = s.length();
      StringBuilder res = new StringBuilder();
      int j = 0;
      for (int i = 0; i < n; ++i) {
         if (j < spaces.length && i == spaces[j]) {
            res.append(" ");
            ++j;
         }
         res.append(s.charAt(i));
      }
      return res.toString();

   }

   // 1408. 数组中的字符串匹配 (String Matching in an Array)
   public List<String> stringMatching(String[] words) {
      StringBuilder builder = new StringBuilder();
      for (String word : words) {
         builder.append(word).append("/");
      }

      List<String> res = new ArrayList<>();
      for (String word : words) {
         if (builder.lastIndexOf(word) != builder.indexOf(word)) {
            res.add(word);
         }
      }
      return res;

   }

   // 1408. 数组中的字符串匹配 (String Matching in an Array)
   public List<String> stringMatching2(String[] words) {
      List<String> res = new ArrayList<>();
      for (int i = 0; i < words.length; ++i) {
         for (int j = 0; j < words.length; ++j) {
            if (i == j) {
               continue;
            }
            if (words[i].contains(words[j]) && !res.contains(words[j])) {
               res.add(words[j]);
            }
         }
      }
      return res;

   }

   // 824. 山羊拉丁文 (Goat Latin)
   public String toGoatLatin(String sentence) {
      StringBuilder res = new StringBuilder();
      int count = 1;
      for (String s : sentence.split("\\s+")) {
         char firstChar = s.charAt(0);
         if (isVowel824(firstChar)) {
            res.append(s);
         } else {
            res.append(s.substring(1));
            res.append(s.substring(0, 1));
         }
         res.append("ma");
         for (int i = 0; i < count; ++i) {
            res.append("a");
         }
         ++count;
         res.append(" ");
      }
      return res.substring(0, res.length() - 1);

   }

   private boolean isVowel824(char c) {
      char lowerC = Character.toLowerCase(c);
      if (lowerC == 'a' || lowerC == 'e' || lowerC == 'i' || lowerC == 'o' || lowerC == 'u') {
         return true;
      }
      return false;
   }

   // 187. 重复的DNA序列 (Repeated DNA Sequences)
   public List<String> findRepeatedDnaSequences(String s) {
      final int L = 10;
      Map<String, Integer> map = new HashMap<>();
      List<String> res = new ArrayList<>();
      for (int i = 0; i + L - 1 < s.length(); ++i) {
         String cur = s.substring(i, i + L);
         map.put(cur, map.getOrDefault(cur, 0) + 1);
         if (map.get(cur) == 2) {
            res.add(cur);
         }
      }
      return res;

   }

   // 187. 重复的DNA序列 (Repeated DNA Sequences)
   public List<String> findRepeatedDnaSequences2(String s) {
      final int L = 10;
      List<String> res = new ArrayList<>();
      char[] chars = s.toCharArray();
      if (chars.length <= L) {
         return res;
      }

      Map<Character, Integer> dic = new HashMap<>();
      dic.put('A', 0b00);
      dic.put('C', 0b01);
      dic.put('T', 0b10);
      dic.put('G', 0b11);
      int cur = 0;
      Map<Integer, Integer> map = new HashMap<>();
      for (int i = 0; i < L; ++i) {
         cur = (cur << 2) | dic.get(chars[i]);
      }
      map.put(cur, map.getOrDefault(cur, 0) + 1);
      for (int i = L; i < chars.length; ++i) {
         cur = ((cur << 2) | dic.get(chars[i])) & ((1 << L * 2) - 1);
         map.put(cur, map.getOrDefault(cur, 0) + 1);
         if (map.get(cur) == 2) {
            res.add(String.valueOf(Arrays.copyOfRange(chars, i - L + 1, i + 1)));
         }
      }
      return res;

   }

   // 904. 水果成篮 (Fruit Into Baskets)
   public int totalFruit(int[] fruits) {
      int res = 0;
      int j = 0;
      int n = fruits.length;
      Map<Integer, Integer> map = new HashMap<>();
      for (int i = 0; i < n; ++i) {
         map.merge(fruits[i], 1, Integer::sum);
         while (map.size() > 2) {
            map.merge(fruits[j], -1, Integer::sum);
            if (map.get(fruits[j]) == 0) {
               map.remove(fruits[j]);
            }
            ++j;
         }
         res = Math.max(res, i - j + 1);
      }
      return res;

   }

   // 2104. 子数组范围和 (Sum of Subarray Ranges) --O(n^2)
   public long subArrayRanges(int[] nums) {
      long res = 0L;
      for (int i = 0; i < nums.length; ++i) {
         int min = nums[i];
         int max = nums[i];
         for (int j = i + 1; j < nums.length; ++j) {
            min = Math.min(min, nums[j]);
            max = Math.max(max, nums[j]);
            res += max - min;
         }
      }
      return res;

   }

   // 2104. 子数组范围和 (Sum of Subarray Ranges) --O(n) 单调栈
   public long subArrayRanges2(int[] nums) {
      int n = nums.length;
      int[] leftMin = new int[n];
      Arrays.fill(leftMin, -1);
      int[] leftMax = new int[n];
      Arrays.fill(leftMax, -1);
      int[] rightMin = new int[n];
      Arrays.fill(rightMin, n);
      int[] rightMax = new int[n];
      Arrays.fill(rightMax, n);
      Stack<Integer> stMin = new Stack<>();
      Stack<Integer> stMax = new Stack<>();
      for (int i = 0; i < n; ++i) {
         while (!stMax.isEmpty() && nums[stMax.peek()] <= nums[i]) {
            rightMax[stMax.pop()] = i;
         }
         if (!stMax.isEmpty()) {
            leftMax[i] = stMax.peek();
         }

         while (!stMin.isEmpty() && nums[stMin.peek()] >= nums[i]) {
            rightMin[stMin.pop()] = i;
         }
         if (!stMin.isEmpty()) {
            leftMin[i] = stMin.peek();
         }
         stMax.push(i);
         stMin.push(i);
      }
      long res = 0L;
      for (int i = 0; i < n; ++i) {
         res += nums[i] * (((long) rightMax[i] - i) * (i - leftMax[i]) - ((long) rightMin[i] - i) * (i - leftMin[i]));
      }
      return res;
   }

   // 419. 甲板上的战舰 (Battleships in a Board)
   public int countBattleships(char[][] board) {
      int m = board.length;
      int n = board[0].length;
      int res = 0;
      for (int i = 0; i < m; ++i) {
         for (int j = 0; j < n; ++j) {
            if (board[i][j] == 'X') {
               if (i > 0 && board[i - 1][j] == 'X') {
                  continue;
               }
               if (j > 0 && board[i][j - 1] == 'X') {
                  continue;
               }
               ++res;
            }
         }
      }
      return res;

   }

   // 419. 甲板上的战舰 (Battleships in a Board)
   public int countBattleships2(char[][] board) {
      int m = board.length;
      int n = board[0].length;
      UnionFind419 union = new UnionFind419(m * n);
      for (int i = 0; i < m; ++i) {
         for (int j = 0; j < n; ++j) {
            if (board[i][j] == 'X') {
               if (i > 0 && board[i - 1][j] == 'X') {
                  union.union(getIndex419(i, j, n), getIndex419(i - 1, j, n));
               }
               if (j > 0 && board[i][j - 1] == 'X') {
                  union.union(getIndex419(i, j, n), getIndex419(i, j - 1, n));
               }
               if (i + 1 < m && board[i + 1][j] == 'X') {
                  union.union(getIndex419(i, j, n), getIndex419(i + 1, j, n));
               }
               if (j + 1 < n && board[i][j + 1] == 'X') {
                  union.union(getIndex419(i, j, n), getIndex419(i, j + 1, n));
               }
            }
         }
      }
      Set<Integer> set = new HashSet<>();
      for (int i = 0; i < m; ++i) {
         for (int j = 0; j < n; ++j) {
            if (board[i][j] == 'X') {
               int root = union.getRoot(getIndex419(i, j, n));
               set.add(root);
            }
         }
      }
      return set.size();

   }

   private int getIndex419(int i, int j, int n) {
      return i * n + j;
   }

   public class UnionFind419 {
      private int[] parent;
      private int[] rank;

      public UnionFind419(int n) {
         parent = new int[n];
         for (int i = 0; i < n; ++i) {
            parent[i] = i;
         }
         rank = new int[n];
         Arrays.fill(rank, 1);
      }

      public int getRoot(int p) {
         if (parent[p] == p) {
            return p;
         }
         return parent[p] = getRoot(parent[p]);
      }

      public boolean isConnected(int p1, int p2) {
         return getRoot(p1) == getRoot(p2);
      }

      public void union(int p1, int p2) {
         int root1 = getRoot(p1);
         int root2 = getRoot(p2);
         if (root1 == root2) {
            return;
         }
         if (rank[root1] < rank[root2]) {
            parent[root1] = root2;
         } else {
            parent[root2] = root1;
            if (rank[root1] == rank[root2]) {
               ++rank[root1];
            }
         }
      }

   }

   // 面试题 03.06. 动物收容所 (Animal Shelter LCCI)
   public class AnimalShelf {
      Queue<int[]> queueCat;
      Queue<int[]> queueDog;

      public AnimalShelf() {
         queueCat = new LinkedList<>();
         queueDog = new LinkedList<>();

      }

      public void enqueue(int[] animal) {
         // cat
         if (animal[1] == 0) {
            queueCat.offer(animal);
         } else {
            queueDog.offer(animal);
         }

      }

      public int[] dequeueAny() {
         if (queueCat.isEmpty() && queueDog.isEmpty()) {
            return new int[] { -1, -1 };
         }
         if (queueCat.isEmpty() && !queueDog.isEmpty()) {
            return queueDog.poll();
         }
         if (!queueCat.isEmpty() && queueDog.isEmpty()) {
            return queueCat.poll();
         }
         int[] headCat = queueCat.peek();
         int[] headDog = queueDog.peek();
         if (headCat[0] < headDog[0]) {
            return queueCat.poll();
         } else {
            return queueDog.poll();
         }

      }

      public int[] dequeueDog() {
         if (queueDog.isEmpty()) {
            return new int[] { -1, -1 };
         }
         return queueDog.poll();

      }

      public int[] dequeueCat() {
         if (queueCat.isEmpty()) {
            return new int[] { -1, -1 };
         }
         return queueCat.poll();

      }
   }

   // 1078. Bigram 分词 (Occurrences After Bigram)
   public String[] findOcurrences(String text, String first, String second) {
      String[] texts = text.split("\\s+");
      List<String> res = new ArrayList<>();
      for (int i = 2; i < texts.length; ++i) {
         if (texts[i - 2].equals(first) && texts[i - 1].equals(second)) {
            res.add(texts[i]);
         }
      }
      return res.toArray(new String[res.size()]);

   }

   // 937. 重新排列日志文件 (Reorder Data in Log Files)
   public String[] reorderLogFiles(String[] logs) {
      Arrays.sort(logs, new Comparator<String>() {

         @Override
         public int compare(String o1, String o2) {
            int index1 = o1.indexOf(" ");
            int index2 = o2.indexOf(" ");
            boolean isDigit1 = Character.isDigit(o1.charAt(index1 + 1));
            boolean isDigit2 = Character.isDigit(o2.charAt(index2 + 1));
            if (!isDigit1 && !isDigit2) {
               if (o1.substring(index1 + 1).equals(o2.substring(index2 + 1))) {
                  return o1.compareTo(o2);
               }
               return o1.substring(index1 + 1).compareTo(o2.substring(index2 + 1));
            }
            return isDigit1 ? (isDigit2 ? 0 : 1) : -1;
         }

      });
      return logs;

   }

   // 1995. 统计特殊四元组 (Count Special Quadruplets) --O(n^3)
   public int countQuadruplets(int[] nums) {
      int[] counts = new int[101];
      int res = 0;
      for (int c = nums.length - 2; c >= 2; --c) {
         ++counts[nums[c + 1]];
         for (int a = 0; a < c; ++a) {
            for (int b = a + 1; b < c; ++b) {
               if (nums[a] + nums[b] + nums[c] < counts.length) {
                  res += counts[nums[a] + nums[b] + nums[c]];
               }
            }
         }
      }
      return res;

   }

   // 1995. 统计特殊四元组 (Count Special Quadruplets) --O(n^2)
   public int countQuadruplets2(int[] nums) {
      Map<Integer, Integer> map = new HashMap<>();
      int res = 0;
      for (int b = nums.length - 3; b >= 1; --b) {
         for (int d = b + 2; d < nums.length; ++d) {
            map.put(nums[d] - nums[b + 1], map.getOrDefault(nums[d] - nums[b + 1], 0) + 1);
         }
         for (int a = 0; a < b; ++a) {
            res += map.getOrDefault(nums[a] + nums[b], 0);
         }
      }
      return res;

   }

   // 911. 在线选举 (Online Election)
   class TopVotedCandidate {
      private Map<Integer, Integer> map;
      private int[] times;
      private List<Integer> tops;

      public TopVotedCandidate(int[] persons, int[] times) {
         this.times = times;
         map = new HashMap<>();
         map.put(-1, -1);
         int top = -1;
         tops = new ArrayList<>();
         for (int i = 0; i < persons.length; ++i) {
            int p = persons[i];
            map.merge(p, 1, Integer::sum);
            if (map.get(p) >= map.get(top)) {
               top = p;
            }
            tops.add(top);
         }

      }

      public int q(int t) {
         int left = 0;
         int right = times.length - 1;
         while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (times[mid] <= t) {
               left = mid + 1;
            } else {
               right = mid - 1;
            }
         }
         return tops.get(left - 1);

      }
   }

   // 134. 加油站 (Gas Station)
   public int canCompleteCircuit(int[] gas, int[] cost) {
      int min = Integer.MAX_VALUE;
      int minIndex = 0;
      int cur = 0;
      for (int i = 0; i < cost.length; ++i) {
         cur += gas[i] - cost[i];
         if (cur < min) {
            min = cur;
            minIndex = i;
         }
      }
      return cur < 0 ? -1 : (minIndex + 1) % cost.length;

   }

   // 1696. 跳跃游戏 VI (Jump Game VI) --单调队列
   public int maxResult(int[] nums, int k) {
      int n = nums.length;
      int[] dp = new int[n];
      dp[0] = nums[0];
      Deque<Integer> q = new ArrayDeque<>();
      q.offerLast(0);
      for (int i = 1; i < n; ++i) {
         while (!q.isEmpty() && i - q.peekFirst() > k) {
            q.pollFirst();
         }
         if (!q.isEmpty()) {
            dp[i] = dp[q.peekFirst()] + nums[i];
         }
         while (!q.isEmpty() && dp[q.peekLast()] <= dp[i]) {
            q.pollLast();
         }
         q.offerLast(i);
      }
      return dp[n - 1];

   }

   // 1499. 满足不等式的最大值 (Max Value of Equation) --单调队列
   public int findMaxValueOfEquation(int[][] points, int k) {
      Deque<int[]> q = new ArrayDeque<>();
      int res = Integer.MIN_VALUE;
      for (int[] p : points) {
         while (!q.isEmpty() && p[0] - q.peekFirst()[0] > k) {
            q.pollFirst();
         }
         if (!q.isEmpty()) {
            res = Math.max(res, p[0] + p[1] + q.peekFirst()[1]);
         }
         while (!q.isEmpty() && p[1] - p[0] >= q.peekLast()[1]) {
            q.pollLast();
         }
         q.offer(new int[] { p[0], p[1] - p[0] });
      }
      return res;

   }

   // 225. 用队列实现栈 (Implement Stack using Queues) --两个队列
   class MyStack {
      private Queue<Integer> queue1;
      private Queue<Integer> queue2;

      public MyStack() {
         queue1 = new LinkedList<>();
         queue2 = new LinkedList<>();
      }

      public void push(int x) {
         queue2.offer(x);
         while (!queue1.isEmpty()) {
            queue2.offer(queue1.poll());
         }
         Queue<Integer> temp = queue2;
         queue2 = queue1;
         queue1 = temp;
      }

      public int pop() {
         return queue1.poll();

      }

      public int top() {
         return queue1.peek();

      }

      public boolean empty() {
         return queue1.isEmpty();
      }
   }

   // 225. 用队列实现栈 (225. Implement Stack using Queues) --一个队列
   class MyStack2 {
      private Queue<Integer> queue;

      public MyStack2() {
         queue = new LinkedList<>();

      }

      public void push(int x) {
         int size = queue.size();
         queue.offer(x);
         while (size-- > 0) {
            queue.offer(queue.poll());
         }
      }

      public int pop() {
         return queue.poll();

      }

      public int top() {
         return queue.peek();

      }

      public boolean empty() {
         return queue.isEmpty();

      }
   }

   // 232. 用栈实现队列 (Implement Queue using Stacks)
   // 面试题 03.04. 化栈为队
   // 剑指 Offer 09. 用两个栈实现队列
   class MyQueue {
      private Stack<Integer> stack1;
      private Stack<Integer> stack2;

      public MyQueue() {
         stack1 = new Stack<>();
         stack2 = new Stack<>();

      }

      public void push(int x) {
         stack1.push(x);

      }

      public int pop() {
         if (stack2.isEmpty()) {
            while (!stack1.isEmpty()) {
               stack2.push(stack1.pop());
            }
         }
         return stack2.pop();

      }

      public int peek() {
         if (stack2.isEmpty()) {
            while (!stack1.isEmpty()) {
               stack2.push(stack1.pop());
            }
         }
         return stack2.peek();

      }

      public boolean empty() {
         return stack1.isEmpty() && stack2.isEmpty();
      }
   }

   // 239. 滑动窗口最大值 (Sliding Window Maximum)
   // LCR 183. 望远镜中最高的海拔
   public int[] maxSlidingWindow(int[] nums, int k) {
      List<Integer> res = new ArrayList<>();
      Deque<Integer> deque = new LinkedList<>();
      for (int i = 0; i < nums.length; ++i) {
         while (!deque.isEmpty() && i - deque.peekFirst() >= k) {
            deque.pollFirst();
         }
         while (!deque.isEmpty() && nums[deque.peekLast()] <= nums[i]) {
            deque.pollLast();
         }
         deque.offerLast(i);
         if (i >= k - 1) {
            res.add(nums[deque.peekFirst()]);
         }
      }
      return res.stream().mapToInt(o -> o).toArray();

   }

   // 387. 字符串中的第一个唯一字符 (First Unique Character in a String) --哈希表
   public int firstUniqChar(String s) {
      char[] chars = s.toCharArray();
      int[] counts = new int[26];
      for (int i = 0; i < chars.length; ++i) {
         ++counts[chars[i] - 'a'];
      }
      for (int i = 0; i < chars.length; ++i) {
         if (counts[chars[i] - 'a'] == 1) {
            return i;
         }
      }
      return -1;

   }

   // 387. 字符串中的第一个唯一字符 (First Unique Character in a String) --队列
   public int firstUniqChar2(String s) {
      char[] chars = s.toCharArray();
      Map<Character, Integer> map = new HashMap<>();
      Queue<Pair387> queue = new LinkedList<>();
      for (int i = 0; i < chars.length; ++i) {
         if (!map.containsKey(chars[i])) {
            map.put(chars[i], i);
            queue.offer(new Pair387(chars[i], i));
         } else {
            map.put(chars[i], -1);
            while (!queue.isEmpty() && map.get(queue.peek().c) == -1) {
               queue.poll();
            }
         }
      }
      return queue.isEmpty() ? -1 : queue.peek().pos;

   }

   public class Pair387 {
      int pos;
      char c;

      public Pair387(char c, int pos) {
         this.c = c;
         this.pos = pos;
      }
   }

   // 剑指 Offer 50. 第一个只出现一次的字符
   public char firstUniqChar50(String s) {
      int[] counts = new int[26];
      for (char c : s.toCharArray()) {
         ++counts[c - 'a'];
      }
      for (char c : s.toCharArray()) {
         if (counts[c - 'a'] == 1) {
            return c;
         }
      }
      return ' ';
   }

   // 剑指 Offer 50. 第一个只出现一次的字符
   public char firstUniqChar50_2(String s) {
      // character // position
      Map<Character, Integer> map = new HashMap<>();
      Queue<Pair_Offer_30> queue = new LinkedList<>();
      for (int i = 0; i < s.length(); ++i) {
         if (!map.containsKey(s.charAt(i))) {
            map.put(s.charAt(i), i);
            queue.offer(new Pair_Offer_30(i, s.charAt(i)));
         } else {
            map.put(s.charAt(i), -1);
            while (!queue.isEmpty() && map.get(queue.peek().c) == -1) {
               queue.poll();
            }
         }
      }
      return queue.isEmpty() ? ' ' : queue.peek().c;

   }

   public class Pair_Offer_30 {
      public int pos;
      public char c;

      public Pair_Offer_30(int pos, char c) {
         this.pos = pos;
         this.c = c;
      }

   }

   // 649. Dota2 参议院 (Dota2 Senate)
   public String predictPartyVictory(String senate) {
      Queue<Integer> queueR = new LinkedList<>();
      Queue<Integer> queueD = new LinkedList<>();
      char[] chars = senate.toCharArray();
      for (int i = 0; i < chars.length; ++i) {
         if (chars[i] == 'R') {
            queueR.offer(i);
         } else {
            queueD.offer(i);
         }
      }
      while (!queueD.isEmpty() && !queueR.isEmpty()) {
         int peekR = queueR.poll();
         int peekD = queueD.poll();
         if (peekR < peekD) {
            queueR.offer(peekR + chars.length);
         } else {
            queueD.offer(peekD + chars.length);
         }
      }
      return queueD.isEmpty() ? "Radiant" : "Dire";

   }

   // 641. 设计循环双端队列 (Design Circular Deque) --数组
   class MyCircularDeque {
      private int[] arr;
      private int size;
      private int capacity;
      private int head;
      private int tail;

      public MyCircularDeque(int k) {
         this.capacity = k;
         this.arr = new int[k];
         this.size = 0;
         this.head = 0;
         this.tail = 0;
      }

      public boolean insertFront(int value) {
         if (isFull()) {
            return false;
         }
         head = (head - 1 + capacity) % capacity;
         arr[head] = value;
         ++size;
         return true;

      }

      public boolean insertLast(int value) {
         if (isFull()) {
            return false;
         }
         arr[tail] = value;
         tail = (tail + 1) % capacity;
         ++size;
         return true;
      }

      public boolean deleteFront() {
         if (isEmpty()) {
            return false;
         }
         head = (head + 1) % capacity;
         --size;
         return true;
      }

      public boolean deleteLast() {
         if (isEmpty()) {
            return false;
         }
         tail = (tail - 1 + capacity) % capacity;
         --size;
         return true;
      }

      public int getFront() {
         if (isEmpty()) {
            return -1;
         }
         return arr[head];

      }

      public int getRear() {
         if (isEmpty()) {
            return -1;
         }
         int index = (tail - 1 + capacity) % capacity;
         return arr[index];
      }

      public boolean isEmpty() {
         return size == 0;

      }

      public boolean isFull() {
         return size == capacity;

      }
   }

   // 641. 设计循环双端队列 (Design Circular Deque) --双向链表
   class MyCircularDeque2 {
      private class InnerNode {
         int val;
         InnerNode prev;
         InnerNode next;

         InnerNode(int val) {
            this.val = val;
         }
      }

      private InnerNode head;
      private InnerNode tail;
      private int capacity;
      private int curSize;

      public MyCircularDeque2(int k) {
         capacity = k;
      }

      public boolean insertFront(int value) {
         if (isFull()) {
            return false;
         }
         if (isEmpty()) {
            head = new InnerNode(value);
            tail = head;
         } else {
            InnerNode node = new InnerNode(value);
            head.prev = node;
            node.next = head;
            head = node;
         }
         ++curSize;
         return true;
      }

      public boolean insertLast(int value) {
         if (isFull()) {
            return false;
         }
         if (isEmpty()) {
            head = new InnerNode(value);
            tail = head;
         } else {
            InnerNode node = new InnerNode(value);
            tail.next = node;
            node.prev = tail;
            tail = node;
         }
         ++curSize;
         return true;

      }

      public boolean deleteFront() {
         if (isEmpty()) {
            return false;
         }
         head = head.next;
         if (head != null) {
            head.prev = null;
         }
         --curSize;
         return true;
      }

      public boolean deleteLast() {
         if (isEmpty()) {
            return false;
         }
         tail = tail.prev;
         if (tail != null) {
            tail.next = null;
         }
         --curSize;
         return true;
      }

      public int getFront() {
         if (isEmpty()) {
            return -1;
         }
         return head.val;
      }

      public int getRear() {
         if (isEmpty()) {
            return -1;
         }
         return tail.val;
      }

      public boolean isEmpty() {
         return curSize == 0;

      }

      public boolean isFull() {
         return curSize == capacity;
      }
   }

   // 622. 设计循环队列 (Design Circular Queue) --数组
   // --follow up：线程安全？
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

   // 622. 设计循环队列 (Design Circular Queue) --单链表
   class MyCircularQueue2 {
      class Node {
         int val;
         Node next;

         Node(int val) {
            this.val = val;
         }
      }

      private int size;
      private int capacity;
      private Node head;
      private Node tail;

      public MyCircularQueue2(int k) {
         this.capacity = k;
         this.size = 0;

      }

      public boolean enQueue(int value) {
         if (isFull()) {
            return false;
         }
         if (isEmpty()) {
            head = new Node(value);
            tail = head;
         } else {
            Node added = new Node(value);
            tail.next = added;
            tail = added;
         }
         ++size;
         return true;
      }

      public boolean deQueue() {
         if (isEmpty()) {
            return false;
         }
         head = head.next;
         --size;
         return true;

      }

      public int Front() {
         if (isEmpty()) {
            return -1;
         }
         return head.val;

      }

      public int Rear() {
         if (isEmpty()) {
            return -1;
         }
         return tail.val;
      }

      public boolean isEmpty() {
         return size == 0;
      }

      public boolean isFull() {
         return size == capacity;
      }
   }

   // 862. 和至少为 K 的最短子数组 (Shortest Subarray with Sum at Least K) --单调队列
   public int shortestSubarray(int[] nums, int k) {
      int n = nums.length;
      long[] pre = new long[n + 1];
      for (int i = 0; i < n; ++i) {
         pre[i + 1] = pre[i] + nums[i];
      }
      int res = Integer.MAX_VALUE;
      Deque<Integer> q = new ArrayDeque<>();
      for (int i = 0; i < n + 1; ++i) {
         while (!q.isEmpty() && pre[i] - pre[q.peekFirst()] >= k) {
            res = Math.min(res, i - q.pollFirst());
         }
         while (!q.isEmpty() && pre[q.peekLast()] >= pre[i]) {
            q.pollLast();
         }
         q.offer(i);
      }
      return res == Integer.MAX_VALUE ? -1 : res;

   }

   // 918. 环形子数组的最大和 (Maximum Sum Circular Subarray) -- 前缀和 + 单调队列
   public int maxSubarraySumCircular(int[] nums) {
      int n = nums.length;
      int[] s = new int[n * 2 + 1];
      for (int i = 0; i < n * 2; ++i) {
         s[i + 1] = s[i] + nums[i % n];
      }
      int res = Integer.MIN_VALUE;
      Deque<Integer> q = new ArrayDeque<>();
      // i 表示 前i个元素
      for (int i = 0; i < n * 2 + 1; ++i) {
         // i - q.peekFirst()表示前i个元素 - 前q.peekFirst()个元素
         // i - q.peekFirst() 最大是n，即最多可选n个元素，即全选
         while (!q.isEmpty() && i - q.peekFirst() > n) {
            q.pollFirst();
         }
         if (!q.isEmpty()) {
            res = Math.max(res, s[i] - s[q.peekFirst()]);
         }
         while (!q.isEmpty() && s[q.peekLast()] >= s[i]) {
            q.pollLast();
         }
         q.offer(i);
      }
      return res;

   }

   // 918. 环形子数组的最大和 (Maximum Sum Circular Subarray) -- 分类讨论
   public int maxSubarraySumCircular2(int[] nums) {
      int s = 0;
      // 最小前缀和，可以为空
      int min_pre = 0;
      // 最大前缀和，不可以为空
      int max_pre = Integer.MIN_VALUE / 2;
      // 最小子数组和
      int min_s = Integer.MAX_VALUE / 2;
      int res = Integer.MIN_VALUE / 2;
      for (int x : nums) {
         s += x;
         res = Math.max(res, s - min_pre);
         min_s = Math.min(min_s, s - max_pre);
         min_pre = Math.min(min_pre, s);
         max_pre = Math.max(max_pre, s);
      }
      return Math.max(res, s - min_s);
   }

   // 933. 最近的请求次数 (Number of Recent Calls)
   // 剑指 Offer II 042. 最近请求次数
   class RecentCounter {
      private Queue<Integer> queue;

      public RecentCounter() {
         queue = new LinkedList<>();
      }

      public int ping(int t) {
         queue.offer(t);
         while (t - queue.peek() > 3000) {
            queue.poll();
         }
         return queue.size();
      }
   }

   // 362. 敲击计数器 (Design Hit Counter) --plus
   class HitCounter {
      private Queue<Integer> queue;

      public HitCounter() {
         queue = new LinkedList<>();

      }

      public void hit(int timestamp) {
         queue.offer(timestamp);

      }

      public int getHits(int timestamp) {
         while (!queue.isEmpty() && timestamp - queue.peek() >= 300) {
            queue.poll();
         }
         return queue.size();
      }
   }

   // 1425. 带限制的子序列和 (Constrained Subsequence Sum) --动态规划 + 单调队列
   public int constrainedSubsetSum(int[] nums, int k) {
      int[] dp = new int[nums.length];
      dp[0] = nums[0];
      Deque<Integer> deque = new LinkedList<>();
      deque.offerLast(0);
      for (int i = 1; i < nums.length; ++i) {
         while (!deque.isEmpty() && i - deque.peekFirst() > k) {
            deque.pollFirst();
         }
         dp[i] = Math.max(0, dp[deque.peekFirst()]) + nums[i];
         while (!deque.isEmpty() && dp[i] >= dp[deque.peekLast()]) {
            deque.pollLast();
         }
         deque.offerLast(i);
      }
      return Arrays.stream(dp).max().getAsInt();

   }

   // 2120. 执行所有后缀指令 (Execution of All Suffix Instructions Staying in a Grid)
   public int[] executeInstructions(int n, int[] startPos, String s) {
      int[] res = new int[s.length()];
      for (int i = 0; i < s.length(); ++i) {
         res[i] = getCounts2120(s.substring(i), n, startPos);
      }
      return res;

   }

   private int getCounts2120(String s, int n, int[] startPos) {
      int count = 0;
      int curX = startPos[0];
      int curY = startPos[1];
      for (char c : s.toCharArray()) {
         switch (c) {
            case 'U':
               if (--curX < 0) {
                  return count;
               }
               ++count;
               break;
            case 'D':
               if (++curX == n) {
                  return count;
               }
               ++count;
               break;
            case 'L':
               if (--curY < 0) {
                  return count;
               }
               ++count;
               break;
            case 'R':
               if (++curY == n) {
                  return count;
               }
               ++count;
               break;
         }

      }
      return count;
   }

   // 2124. 检查是否所有 A 都在 B 之前 (Check if All A's Appears Before All B's)
   public boolean checkString(String s) {
      boolean flag = false;
      for (char c : s.toCharArray()) {
         if (c == 'b') {
            flag = true;
         } else if (flag) {
            return false;
         }
      }
      return true;
   }

   // 2125. 银行中的激光束数量 (Number of Laser Beams in a Bank)
   public int numberOfBeams(String[] bank) {
      int pre = 0;
      int res = 0;
      for (String s : bank) {
         int cnt1 = 0;
         for (char c : s.toCharArray()) {
            if (c == '1') {
               ++cnt1;
            }
         }
         if (cnt1 != 0) {
            res += cnt1 * pre;
            pre = cnt1;
         }
      }
      return res;

   }

   // 2063. 所有子字符串中的元音 (Vowels of All Substrings)
   public long countVowels(String word) {
      long res = 0L;
      char[] chars = word.toCharArray();
      for (int i = 0; i < chars.length; ++i) {
         if (chars[i] == 'a' || chars[i] == 'e' || chars[i] == 'i' || chars[i] == 'o' || chars[i] == 'u') {
            res += (long) (i + 1) * (chars.length - i);
         }
      }
      return res;
   }

   // 2086. 从房屋收集雨水需要的最少水桶数 (Minimum Number of Buckets Required to Collect
   // Rainwater from Houses)
   public int minimumBuckets(String street) {
      int res = 0;
      char[] chars = street.toCharArray();
      for (int i = 0; i < chars.length; ++i) {
         if (chars[i] == 'H') {
            if (i + 1 < chars.length && chars[i + 1] == '.') {
               ++res;
               i += 2;
            } else if (i - 1 >= 0 && chars[i - 1] == '.') {
               ++res;
            } else {
               return -1;
            }
         }
      }
      return res;

   }

   // 2114. 句子中的最多单词数 (Maximum Number of Words Found in Sentences)
   public int mostWordsFound(String[] sentences) {
      int res = 0;
      for (String sentence : sentences) {
         int cur = 1;
         for (char c : sentence.toCharArray()) {
            if (c == ' ') {
               ++cur;
            }
         }
         res = Math.max(res, cur);
      }
      return res;

   }

   // 1980. 找出不同的二进制字符串 (Find Unique Binary String)
   public String findDifferentBinaryString(String[] nums) {
      StringBuilder res = new StringBuilder();
      for (int i = 0; i < nums.length; ++i) {
         res.append((nums[i].charAt(i) - '0') ^ 1);
      }
      return res.toString();

   }

   // 2002. 两个回文子序列长度的最大乘积 (Maximum Product of the Length of Two Palindromic
   // Subsequences)
   public int maxProduct2(String s) {
      int n = s.length();
      boolean[] arr = new boolean[1 << n];
      for (int i = 1; i < 1 << n; ++i) {
         int cnt = Integer.bitCount(i);
         int lead = 32 - Integer.numberOfLeadingZeros(i) - 1;
         int trail = Integer.numberOfTrailingZeros(i);
         if (s.charAt(lead) == s.charAt(trail) && (cnt <= 2 || arr[(i ^ (1 << trail) ^ (1 << lead))])) {
            arr[i] = true;
         }
      }
      int res = 1;
      int u = (1 << n) - 1;
      int m = u - 1;
      while (m > 0) {
         int cnt = Integer.bitCount(m);
         if (arr[m] && res < cnt * (n - cnt)) {
            int j = u ^ m;
            int c = j;
            while (c > 0) {
               if (arr[c]) {
                  res = Math.max(res, cnt * Integer.bitCount(c));
               }
               c = (c - 1) & j;
            }
         }
         --m;
      }
      return res;

   }

   // 1255. 得分最高的单词集合 (Maximum Score Words Formed by Letters)
   public int maxScoreWords(String[] words, char[] letters, int[] score) {
      int[] counts = new int[26];
      for (char c : letters) {
         ++counts[c - 'a'];
      }
      Map<String, Bean1255> map = new HashMap<>();
      List<String> wordStrings = new ArrayList<>();
      search: for (String word : words) {
         int[] curCounts = new int[26];
         int curScore = 0;
         for (char c : word.toCharArray()) {
            ++curCounts[c - 'a'];
            if (curCounts[c - 'a'] > counts[c - 'a']) {
               continue search;
            }
            curScore += score[c - 'a'];
         }
         map.put(word, new Bean1255(curScore, curCounts));
         wordStrings.add(word);
      }
      int res = 0;
      int n = wordStrings.size();
      search: for (int i = 0; i < (1 << n); ++i) {
         int mask = i;
         int index = 0;
         int[] curCounts = new int[26];
         int curScore = 0;
         while (mask > 0) {
            if ((mask & 1) == 1) {
               String curWord = wordStrings.get(index);
               int[] cnts = map.get(curWord).wordCounts;
               for (int j = 0; j < 26; ++j) {
                  curCounts[j] += cnts[j];
                  if (curCounts[j] > counts[j]) {
                     continue search;
                  }
               }
               curScore += map.get(curWord).score;
            }
            ++index;
            mask >>= 1;
         }
         res = Math.max(res, curScore);
      }
      return res;

   }

   public class Bean1255 {
      int score;
      int[] wordCounts;

      Bean1255(int score, int[] wordCounts) {
         this.score = score;
         this.wordCounts = wordCounts;

      }

   }

   // 1601. 最多可达成的换楼请求数目 (Maximum Number of Achievable Transfer Requests) --状态压缩
   public int maximumRequests(int n, int[][] requests) {
      int res = 0;
      int[] out = new int[n];
      int[] in = new int[n];
      for (int[] request : requests) {
         if (request[0] != request[1]) {
            ++out[request[0]];
            ++in[request[1]];
         }
         // 将自己搬出，再将自己搬入
         else {
            ++res;
         }
      }

      List<int[]> actualRequests = new ArrayList<>();
      for (int[] request : requests) {
         int a = request[0];
         int b = request[1];
         // 有人进 有人出
         if (a != b && out[a] != 0 && in[a] != 0 && out[b] != 0 && in[b] != 0) {
            actualRequests.add(request);
         }
      }
      int[][] ac = actualRequests.toArray(new int[actualRequests.size()][]);

      int max = 0;
      for (int i = 0; i < (1 << ac.length); ++i) {
         max = Math.max(max, getCount1601(ac, i, n));
      }
      return max + res;

   }

   private int getCount1601(int[][] requests, int status, int n) {
      int[] out = new int[n];
      int[] in = new int[n];
      int max = 0;
      int index = 0;
      while (status != 0) {
         if (status % 2 == 1) {
            ++out[requests[index][0]];
            ++in[requests[index][1]];
            ++max;
         }
         status >>= 1;
         ++index;
      }
      if (!Arrays.equals(out, in)) {
         return -1;
      }
      return max;
   }

   // 2038. 如果相邻两个颜色均相同则删除当前颜色 (Remove Colored Pieces if Both Neighbors are the
   // Same Color)
   public boolean winnerOfGame(String colors) {
      char[] chars = colors.toCharArray();
      int a = 0;
      int b = 0;
      int cur = 1;
      for (int i = 1; i < chars.length; ++i) {
         if (chars[i] == chars[i - 1]) {
            if (++cur > 2) {
               if (chars[i] == 'A') {
                  ++a;
               } else {
                  ++b;
               }
            }
         } else {
            cur = 1;
         }
      }
      return a > b;
   }

   // 2075. 解码斜向换位密码 (Decode the Slanted Ciphertext)
   public String decodeCiphertext(String encodedText, int rows) {
      StringBuilder res = new StringBuilder();
      char[] chars = encodedText.toCharArray();
      int cols = chars.length / rows;
      for (int i = 0; i < cols; ++i) {
         int r = 0;
         int c = i;
         while (r < rows && c < cols) {
            res.append(chars[r * cols + c]);
            ++r;
            ++c;
         }
      }
      while (res.length() - 1 >= 0 && res.charAt(res.length() - 1) == ' ') {
         res.deleteCharAt(res.length() - 1);
      }
      return res.toString();

   }

   // 31. 下一个排列 (Next Permutation)
   public void nextPermutation(int[] nums) {
      int n = nums.length;
      int i = n - 2;
      while (i >= 0) {
         if (nums[i] < nums[i + 1]) {
            break;
         }
         --i;
      }
      if (i < 0) {
         Arrays.sort(nums);
         return;
      }
      int j = n - 1;
      while (i < j) {
         if (nums[i] < nums[j]) {
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
            Arrays.sort(nums, i + 1, n);
            break;
         }
         --j;
      }

   }

   // 剑指 Offer 38. 字符串的排列 --方法：31.下一个排列 还需掌握：回溯法
   public String[] permutation(String s) {
      List<String> list = new ArrayList<>();
      char[] chars = s.toCharArray();
      Arrays.sort(chars);
      do {
         list.add(String.valueOf(chars));
      } while (hasNextPermutation(chars));

      return list.toArray(new String[list.size()]);

   }

   private boolean hasNextPermutation(char[] chars) {
      int i = chars.length - 2;
      while (i >= 0) {
         if (chars[i] < chars[i + 1]) {
            break;
         }
         --i;
      }
      if (i < 0) {
         return false;
      }
      int j = chars.length - 1;
      while (i < j) {
         if (chars[i] < chars[j]) {
            swapChars(chars, i, j);
            break;
         }
         --j;
      }
      Arrays.sort(chars, i + 1, chars.length);
      return true;
   }

   private void swapChars(char[] chars, int i, int j) {
      char temp = chars[i];
      chars[i] = chars[j];
      chars[j] = temp;
   }

   // 剑指 Offer 46. 把数字翻译成字符串
   public int translateNum(int num) {
      String str = String.valueOf(num);
      int p = 0;
      int q = 0;
      int r = 1;
      for (int i = 0; i < str.length(); ++i) {
         p = q;
         q = r;
         r = 0;
         r += q;
         if (i == 0) {
            continue;
         }
         String cur = str.substring(i - 1, i + 1);
         if (cur.compareTo("10") >= 0 && cur.compareTo("25") <= 0) {
            r += p;
         }
      }
      return r;

   }

   // 剑指 Offer 46. 把数字翻译成字符串
   private int n_offer_46;
   private char[] arr_offer_46;
   private int[] memo_offer_46;

   public int translateNum2(int num) {
      this.arr_offer_46 = String.valueOf(num).toCharArray();
      this.n_offer_46 = arr_offer_46.length;
      this.memo_offer_46 = new int[n_offer_46];
      return dfs_offer_46(0);
   }

   private int dfs_offer_46(int i) {
      if (i == n_offer_46) {
         return 1;
      }
      if (memo_offer_46[i] != 0) {
         return memo_offer_46[i];
      }
      int res = dfs_offer_46(i + 1);
      if (arr_offer_46[i] == '1' && i + 1 < n_offer_46) {
         res += dfs_offer_46(i + 2);
      } else if (arr_offer_46[i] == '2' && i + 1 < n_offer_46 && arr_offer_46[i + 1] <= '5'
            && arr_offer_46[i + 1] >= '0') {
         res += dfs_offer_46(i + 2);
      }
      return memo_offer_46[i] = res;
   }

   // 266. 回文排列 (Palindrome Permutation) --Plus
   // Given a string s, return true if a permutation of the string could form a
   // palindrome.
   // 给定一个字符串，判断该字符串中是否可以通过重新排列组合，形成一个回文字符串。
   public boolean canPermutePalindrome(String s) {
      int[] counts = new int[26];
      for (char c : s.toCharArray()) {
         ++counts[c - 'a'];
      }
      int oddCount = 0;
      for (int count : counts) {
         if (((count & 1) == 1) && (++oddCount > 1)) {
            return false;
         }
      }
      return true;

   }

   // 1239. 串联字符串的最大长度 (Maximum Length of a Concatenated String with Unique
   // Characters)
   private List<Integer> a1239;

   private record Group1239(int i, int j) {
   }

   private int n1239;
   private Map<Group1239, Integer> memo1239;

   public int maxLength(List<String> arr) {
      this.a1239 = new ArrayList<>();
      search: for (String s : arr) {
         int m = 0;
         for (char c : s.toCharArray()) {
            if ((m >> (c - 'a') & 1) != 0) {
               continue search;
            }
            m |= 1 << (c - 'a');
         }
         this.a1239.add(m);
      }
      this.n1239 = a1239.size();
      this.memo1239 = new HashMap<>();
      return dfs1239(0, 0);

   }

   private int dfs1239(int i, int j) {
      if (i == n1239) {
         return 0;
      }
      // 当数据范围较小时，去掉记忆化，速度更快 （也就是回溯）
      Group1239 key = new Group1239(i, j);
      if (memo1239.get(key) != null) {
         return memo1239.get(key);
      }
      int res = dfs1239(i + 1, j);
      if ((j & a1239.get(i)) == 0) {
         res = Math.max(res, dfs1239(i + 1, j | a1239.get(i)) + Integer.bitCount(a1239.get(i)));
      }
      memo1239.put(key, res);
      return res;
   }

   // 1647. 字符频次唯一的最小删除次数 (Minimum Deletions to Make Character Frequencies Unique)
   // --哈希
   public int minDeletions(String s) {
      int[] counts = new int[26];
      for (char c : s.toCharArray()) {
         ++counts[c - 'a'];
      }
      Set<Integer> set = new HashSet<>();
      int res = 0;
      for (int count : counts) {
         int cur = count;
         while (set.contains(cur)) {
            ++res;
            --cur;
         }
         if (cur != 0) {
            set.add(cur);
         }
      }
      return res;

   }

   // 1647. 字符频次唯一的最小删除次数 (Minimum Deletions to Make Character Frequencies Unique)
   // --排序
   public int minDeletions2(String s) {
      int[] counts = new int[26];
      for (char c : s.toCharArray()) {
         ++counts[c - 'a'];
      }
      int res = 0;
      Arrays.sort(counts);
      for (int i = counts.length - 2; i >= 0; --i) {
         if (counts[i] >= counts[i + 1]) {
            int temp = counts[i];
            counts[i] = counts[i + 1] == 0 ? 0 : counts[i + 1] - 1;
            res += temp - counts[i];
         }
      }
      return res;

   }

   // 剑指 Offer 59 - II. 队列的最大值
   class MaxQueue {
      private Queue<Integer> mQueue;
      private Deque<Integer> mDeque;

      public MaxQueue() {
         mQueue = new LinkedList<>();
         mDeque = new LinkedList<>();
      }

      public int max_value() {
         if (mDeque.isEmpty()) {
            return -1;
         }
         return mDeque.peekFirst();
      }

      public void push_back(int value) {
         while (!mDeque.isEmpty() && mDeque.peekLast() < value) {
            mDeque.pollLast();
         }
         mDeque.offerLast(value);
         mQueue.offer(value);

      }

      public int pop_front() {
         if (mQueue.isEmpty()) {
            return -1;
         }
         int res = mQueue.poll();
         if (mDeque.peekFirst() == res) {
            mDeque.pollFirst();
         }
         return res;

      }
   }

   // 2073. 买票需要的时间 (Time Needed to Buy Tickets)
   public int timeRequiredToBuy(int[] tickets, int k) {
      int res = 0;
      for (int i = 0; i < tickets.length; ++i) {
         if (i <= k) {
            res += Math.min(tickets[i], tickets[k]);
         } else {
            res += Math.min(tickets[i], tickets[k] - 1);
         }
      }
      return res;

   }

   // 1823. 找出游戏的获胜者 (Find the Winner of the Circular Game)
   public int findTheWinner(int n, int k) {
      List<Integer> list = new LinkedList<>();
      for (int i = 0; i < n; ++i) {
         list.add(i + 1);
      }
      int index = 0;
      while (list.size() > 1) {
         index = (index + k - 1) % list.size();
         list.remove(index);
      }
      return list.get(0);

   }

   // 1700. 无法吃午餐的学生数量 (Number of Students Unable to Eat Lunch)
   public int countStudents(int[] students, int[] sandwiches) {
      int[] stuCounts = new int[2];
      for (int student : students) {
         ++stuCounts[student];
      }
      for (int sandwich : sandwiches) {
         if (stuCounts[sandwich] == 0) {
            break;
         }
         --stuCounts[sandwich];
      }
      return stuCounts[0] + stuCounts[1];

   }

   // 1291. 顺次数 (Sequential Digits) --枚举
   public List<Integer> sequentialDigits(int low, int high) {
      List<Integer> res = new ArrayList<>();
      for (int i = 1; i <= 9; ++i) {
         int num = i;
         for (int j = i + 1; j <= 9; ++j) {
            num = num * 10 + j;
            if (low <= num && num <= high) {
               res.add(num);
            }
         }
      }
      Collections.sort(res);
      return res;
   }

   // 1958. 检查操作是否合法 (Check if Move is Legal)
   public boolean checkMove(char[][] board, int rMove, int cMove, char color) {
      for (int dx = -1; dx <= 1; ++dx) {
         for (int dy = -1; dy <= 1; ++dy) {
            if (check1958(board, rMove, cMove, dx, dy, color)) {
               return true;
            }
         }
      }
      return false;
   }

   private boolean check1958(char[][] board, int rMove, int cMove, int dx, int dy, char color) {
      boolean flag = false;
      while (0 <= rMove + dx && rMove + dx < 8 && 0 <= cMove + dy && cMove + dy < 8) {
         rMove += dx;
         cMove += dy;
         if (board[rMove][cMove] == '.') {
            return false;
         }
         if (board[rMove][cMove] == color) {
            return flag;
         }
         flag = true;
      }
      return false;
   }

   // 2048. 下一个更大的数值平衡数 (Next Greater Numerically Balanced Number)
   public int nextBeautifulNumber(int n) {
      search: for (int num = n + 1; num <= 1224444; ++num) {
         int[] counts = new int[10];
         int cur = num;
         while (cur != 0) {
            ++counts[cur % 10];
            cur /= 10;
         }
         cur = num;
         while (cur != 0) {
            if (cur % 10 != counts[cur % 10]) {
               continue search;
            }
            cur /= 10;
         }
         return num;
      }
      return -1;

   }

   // 2133. 检查是否每一行每一列都包含全部整数 (Check if Every Row and Column Contains All Numbers)
   public boolean checkValid(int[][] matrix) {
      int n = matrix.length;
      Set<Integer> set1 = new HashSet<>();
      Set<Integer> set2 = new HashSet<>();
      for (int i = 0; i < n; ++i) {
         for (int j = 0; j < n; ++j) {
            if (!set1.add(matrix[i][j])) {
               return false;
            }
            if (!set2.add(matrix[j][i])) {
               return false;
            }
         }
         set1.clear();
         set2.clear();
      }
      return true;

   }

   // 820. 单词的压缩编码 (Short Encoding of Words)
   // 剑指 Offer II 065. 最短的单词编码 --还需掌握字典树
   public int minimumLengthEncoding(String[] words) {
      Set<String> set = new HashSet<>(Arrays.asList(words));
      for (String word : words) {
         for (int i = 1; i < word.length(); ++i) {
            set.remove(word.substring(i));
         }
      }
      int res = 0;
      for (String s : set) {
         res += s.length() + 1;
      }
      return res;

   }

   // 2047. 句子中的有效单词数 (Number of Valid Words in a Sentence)
   public int countValidWords(String sentence) {
      String[] strs = sentence.trim().split("\\s+");
      int res = 0;
      for (String s : strs) {
         if (check2047(s)) {
            ++res;
         }
      }
      return res;

   }

   private boolean check2047(String s) {
      int hyphenCount = 0;
      int punctuationCount = 0;
      for (int i = 0; i < s.length(); ++i) {
         char c = s.charAt(i);
         if (Character.isDigit(c)) {
            return false;
         }
         if (c == '-') {
            if (++hyphenCount > 1 || i == 0 || i == s.length() - 1 || !Character.isLetter(s.charAt(i - 1))
                  || !Character.isLetter(s.charAt(i + 1))) {
               return false;
            }
         }
         if (c == '!' || c == '.' || c == ',') {
            if (++punctuationCount > 1 || i != s.length() - 1) {
               return false;
            }
         }
      }
      return true;
   }

   // 2129. 将标题首字母大写 (Capitalize the Title)
   public String capitalizeTitle(String title) {
      char[] chars = title.toCharArray();
      for (int i = 0; i < chars.length; ++i) {
         chars[i] = Character.toLowerCase(chars[i]);
      }
      int left = 0;
      int right = 0;
      while (right < chars.length) {
         while (right < chars.length && chars[right] != ' ') {
            ++right;
         }
         if (right - left > 2) {
            chars[left] = Character.toUpperCase(chars[left]);
         }
         left = right + 1;
         ++right;
      }
      return String.valueOf(chars);

   }

   // 859. 亲密字符串 (Buddy Strings)
   public boolean buddyStrings(String s, String goal) {
      if (s.length() != goal.length()) {
         return false;
      }
      if (s.equals(goal)) {
         int[] counts = new int[26];
         for (char c : s.toCharArray()) {
            if (++counts[c - 'a'] > 1) {
               return true;
            }
         }
         return false;
      }
      int firstIndex = -1;
      int secondIndex = -1;
      for (int i = 0; i < s.length(); ++i) {
         if (s.charAt(i) != goal.charAt(i)) {
            if (firstIndex == -1) {
               firstIndex = i;
            } else if (secondIndex == -1) {
               secondIndex = i;
            } else {
               return false;
            }

         }
      }
      return secondIndex != -1 && s.charAt(firstIndex) == goal.charAt(secondIndex)
            && s.charAt(secondIndex) == goal.charAt(firstIndex);

   }

   // 2016. 增量元素之间的最大差值 (Maximum Difference Between Increasing Elements)
   public int maximumDifference(int[] nums) {
      int res = -1;
      int min = nums[0];
      for (int i = 1; i < nums.length; ++i) {
         if (nums[i] > min) {
            res = Math.max(res, nums[i] - min);
         } else {
            min = nums[i];
         }
      }
      return res;
   }

   // 2138. 将字符串拆分为若干长度为 k 的组 (Divide a String Into Groups of Size k)
   public String[] divideString(String s, int k, char fill) {
      List<String> res = new ArrayList<>();
      for (int i = 0; i < s.length(); i += k) {
         StringBuilder sb = new StringBuilder(s.substring(i, Math.min(i + k, s.length())));
         while (sb.length() < k) {
            sb.append(fill);
         }
         res.add(sb.toString());
      }
      return res.toArray(new String[0]);

   }

   // 2139. 得到目标值的最少行动次数 (Minimum Moves to Reach Target Score)
   public int minMoves(int target, int maxDoubles) {
      int res = 0;
      while (target > 1 && maxDoubles-- > 0) {
         ++res;
         if (target % 2 == 1) {
            ++res;
         }
         target /= 2;
      }
      return res + target - 1;

   }

   // 2140. 解决智力问题 (Solving Questions With Brainpower)
   private long[] memo2140;
   private int[][] questions2140;

   public long mostPoints(int[][] questions) {
      int n = questions.length;
      this.questions2140 = questions;
      memo2140 = new long[n];
      return dfs2140(0);

   }

   private long dfs2140(int i) {
      if (i >= questions2140.length) {
         return 0L;
      }
      if (memo2140[i] != 0L) {
         return memo2140[i];
      }
      return memo2140[i] = Math.max(questions2140[i][0] + dfs2140(i + questions2140[i][1] + 1), dfs2140(i + 1));
   }

   // 2140. 解决智力问题 (Solving Questions With Brainpower) --倒序 填表法
   public long mostPoints2(int[][] questions) {
      long[] dp = new long[questions.length + 1];
      for (int i = questions.length - 1; i >= 0; --i) {
         int j = i + questions[i][1] + 1;
         dp[i] = Math.max(dp[i + 1], questions[i][0] + (j < questions.length ? dp[j] : 0));
      }
      return dp[0];

   }

   // 2141. 同时运行 N 台电脑的最长时间 (Maximum Running Time of N Computers)
   public long maxRunTime(int n, int[] batteries) {
      long res = 0L;
      long left = 0L;
      long sumB = 0L;
      for (int battery : batteries) {
         sumB += battery;
      }
      long right = sumB / n;
      while (left <= right) {
         long mid = left + ((right - left) >> 1);
         long sum = 0L;
         for (int battery : batteries) {
            sum += Math.min(battery, mid);
         }
         if (mid * n <= sum) {
            res = mid;
            left = mid + 1;
         } else {
            right = mid - 1;
         }
      }
      return res;

   }

   // LCP 33. 蓄水
   public int storeWater(int[] bucket, int[] vat) {
      int n = vat.length;
      int max = Arrays.stream(vat).max().getAsInt();
      if (max == 0) {
         return 0;
      }
      int res = Integer.MAX_VALUE;
      for (int k = 1; k <= max && k < res; ++k) {
         int t = 0;
         for (int i = 0; i < n; ++i) {
            t += Math.max(0, (vat[i] + k - 1) / k - bucket[i]);
         }
         res = Math.min(res, k + t);
      }
      return res;

   }

   // LCP 30. 魔塔游戏
   public int magicTower(int[] nums) {
      long s = 1L;
      for (int num : nums) {
         s += num;
      }
      if (s <= 0) {
         return -1;
      }
      int res = 0;
      Queue<Integer> q = new PriorityQueue<>();
      long hp = 1L;
      for (int num : nums) {
         if (num < 0) {
            q.offer(num);
         }
         hp += num;
         if (hp < 1) {
            hp -= q.poll();
            ++res;
         }
      }
      return res;

   }

   // 2126. 摧毁小行星 (Destroying Asteroids)
   public boolean asteroidsDestroyed(int mass, int[] asteroids) {
      Arrays.sort(asteroids);
      long curMass = mass;
      for (int asteroid : asteroids) {
         if (curMass < asteroid) {
            return false;
         }
         curMass += asteroid;
      }
      return true;

   }

   // 2018. 判断单词是否能放入填字游戏内 (Check if Word Can Be Placed In Crossword)
   public boolean placeWordInCrossword(char[][] board, String word) {
      int m = board.length;
      int n = board[0].length;
      int wordLen = word.length();
      for (int i = 0; i < m; ++i) {
         for (int j = 0; j < n; ++j) {
            if (board[i][j] == '#') {
               continue;
            }
            int j0 = j;
            boolean flag1 = true;
            boolean flag2 = true;
            while (j < n && board[i][j] != '#') {
               if (j - j0 + 1 > wordLen || (board[i][j] != ' ' && board[i][j] != word.charAt(j - j0))) {
                  flag1 = false;
               }
               if (j - j0 + 1 > wordLen || (board[i][j] != ' ' && board[i][j] != word.charAt(wordLen - 1 - (j - j0)))) {
                  flag2 = false;
               }
               ++j;
            }

            if ((flag1 || flag2) && (j - j0 == wordLen)) {
               return true;
            }
         }
      }
      for (int j = 0; j < n; ++j) {
         for (int i = 0; i < m; ++i) {
            if (board[i][j] == '#') {
               continue;
            }
            int i0 = i;
            boolean flag1 = true;
            boolean flag2 = true;
            while (i < m && board[i][j] != '#') {
               if (i - i0 + 1 > wordLen || (board[i][j] != ' ' && board[i][j] != word.charAt(i - i0))) {
                  flag1 = false;
               }
               if (i - i0 + 1 > wordLen || (board[i][j] != ' ' && board[i][j] != word.charAt(wordLen - 1 - (i - i0)))) {
                  flag2 = false;
               }
               ++i;
            }
            if ((flag1 || flag2) && (i - i0 == wordLen)) {
               return true;
            }
         }
      }
      return false;

   }

   // 1620. 网络信号最好的坐标 (Coordinate With Maximum Network Quality)
   public int[] bestCoordinate(int[][] towers, int radius) {
      int[] res = new int[] { 0, 0 };
      int max = 0;
      for (int i = 0; i <= 50; ++i) {
         for (int j = 0; j <= 50; ++j) {
            int cur = 0;
            for (int[] tower : towers) {
               int dis = (tower[0] - i) * (tower[0] - i) + (tower[1] - j) * (tower[1] - j);
               if (dis <= radius * radius) {
                  cur += tower[2] / (1 + Math.sqrt(dis));
               }
            }
            if (cur > max) {
               max = cur;
               res = new int[] { i, j };
            }
         }
      }
      return res;

   }

   // 949. 给定数字能组成的最大时间 (Largest Time for Given Digits)
   public String largestTimeFromDigits(int[] arr) {
      Arrays.sort(arr);
      int res = -1;
      do {
         int hour = arr[0] * 10 + arr[1];
         int minute = arr[2] * 10 + arr[3];
         if (hour >= 24) {
            break;
         }
         if (hour <= 23 && minute <= 59) {
            res = Math.max(res, hour * 60 + minute);
         }
      } while (hasNextLargerPermutation(arr));
      return res >= 0 ? String.format("%02d:%02d", res / 60, res % 60) : "";

   }

   private boolean hasNextLargerPermutation(int[] arr) {
      int i = arr.length - 2;
      while (i >= 0) {
         if (arr[i] < arr[i + 1]) {
            break;
         }
         --i;
      }
      if (i < 0) {
         return false;
      }
      int j = arr.length - 1;
      while (i < j) {
         if (arr[i] < arr[j]) {
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            break;
         }
         --j;
      }
      Arrays.sort(arr, i + 1, arr.length);
      return true;
   }

   // 5971. 打折购买糖果的最小开销 (Minimum Cost of Buying Candies With Discount)
   public int minimumCost(int[] cost) {
      int res = 0;
      Arrays.sort(cost);
      int index = cost.length - 1;
      while (index >= 0) {
         res += cost[index];
         --index;
         if (index >= 0) {
            res += cost[index];
         }
         index -= 2;
      }
      return res;

   }

   // 2145. 统计隐藏数组数目 (Count the Hidden Sequences)
   public int numberOfArrays(int[] differences, int lower, int upper) {
      int l = lower;
      int r = upper;
      for (int d : differences) {
         l += d;
         r += d;
         if (l > upper || r < lower) {
            return 0;
         }
         l = Math.max(l, lower);
         r = Math.min(r, upper);
      }
      return r - l + 1;
   }

   // 2147. 分隔长廊的方案数 (Number of Ways to Divide a Long Corridor)
   public int numberOfWays(String corridor) {
      char[] arr = corridor.toCharArray();
      long res = 1L;
      final int MOD = (int) (1e9 + 7);
      int cnt = 0;
      for (char c : arr) {
         if (c == 'S') {
            ++cnt;
         }
      }
      if (cnt == 0 || cnt % 2 == 1) {
         return 0;
      }
      cnt = 0;
      int p1 = -1;
      int p2 = -1;
      for (int i = 0; i < arr.length; ++i) {
         if (arr[i] == 'S') {
            p1 = p2;
            p2 = i;
            ++cnt;
            if (cnt % 2 == 1 && p1 != -1 && p2 != -1) {
               res = (res * (p2 - p1)) % MOD;
            }
         }
      }
      return (int) res;

   }

   // 5989. 元素计数 (Count Elements With Strictly Smaller and Greater Elements)
   public int countElements(int[] nums) {
      int max = Arrays.stream(nums).max().getAsInt();
      int min = Arrays.stream(nums).min().getAsInt();
      int res = 0;
      for (int num : nums) {
         if (num != max && num != min) {
            ++res;
         }
      }
      return res;

   }

   // 5991. 按符号重排数组 (Rearrange Array Elements by Sign)
   public int[] rearrangeArray(int[] nums) {
      int[] res = new int[nums.length];
      int positiveIndex = 0;
      int negativeIndex = 1;
      for (int num : nums) {
         if (num > 0) {
            res[positiveIndex] = num;
            positiveIndex += 2;
         } else {
            res[negativeIndex] = num;
            negativeIndex += 2;
         }
      }
      return res;

   }

   // 5990. 找出数组中的所有孤独数字 (Find All Lonely Numbers in the Array)
   public List<Integer> findLonely(int[] nums) {
      Map<Integer, Integer> map = new HashMap<>();
      for (int num : nums) {
         map.put(num, map.getOrDefault(num, 0) + 1);
      }
      List<Integer> res = new ArrayList<>();
      for (int key : map.keySet()) {
         if (map.get(key) == 1 && map.getOrDefault(key - 1, 0) == 0 && map.getOrDefault(key + 1, 0) == 0) {
            res.add(key);
         }
      }
      return res;

   }

   // 5992. 基于陈述统计最多好人数 (Maximum Good People Based on Statements)
   public int maximumGood(int[][] statements) {
      int n = statements.length;
      int res = 0;
      search: for (int m = 0; m < (1 << n); ++m) {
         for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
               if (statements[i][j] == 0 && ((m >> i) & 1) == 1 && ((m >> j) & 1) == 1) {
                  continue search;
               }
               if (statements[i][j] == 1 && ((m >> i) & 1) == 1 && ((m >> j) & 1) == 0) {
                  continue search;
               }
            }
         }
         res = Math.max(res, Integer.bitCount(m));

      }
      return res;
   }

   // 2081. k 镜像数字的和 (Sum of k-Mirror Numbers)
   public long kMirror(int k, int n) {
      long res = 0L;
      int count = 0;
      int left = 1;
      while (count < n) {
         int right = left * 10;
         for (int i = 0; i < 2; ++i) {
            for (int num = left; num < right && count < n; ++num) {
               long combined = num;
               int x = (i == 0 ? num / 10 : num);
               while (x != 0) {
                  combined = combined * 10 + x % 10;
                  x /= 10;
               }
               if (judge2081(combined, k)) {
                  ++count;
                  res += combined;
               }
            }
         }
         left = right;
      }
      return res;

   }

   private boolean judge2081(long combined, int k) {
      StringBuilder builder = new StringBuilder();
      while (combined != 0) {
         builder.append(combined % k);
         combined /= k;
      }
      int left = 0;
      int right = builder.length() - 1;
      while (left < right) {
         if (builder.charAt(left++) != builder.charAt(right--)) {
            return false;
         }
      }
      return true;
   }

   // 2122. 还原原数组 (Recover the Original Array)
   public int[] recoverArray(int[] nums) {
      Arrays.sort(nums);
      for (int i = 1; i < nums.length; ++i) {
         if (nums[i] == nums[0] || (nums[i] - nums[0]) % 2 == 1) {
            continue;
         }
         boolean[] visited = new boolean[nums.length];
         visited[0] = true;
         visited[i] = true;
         int[] res = new int[nums.length / 2];
         int k = (nums[i] - nums[0]) / 2;
         res[0] = nums[0] + k;
         int index = 1;
         int left = 0;
         int right = i;
         for (int j = 0; j <= nums.length / 2 - 2; ++j) {
            while (left < nums.length && visited[left]) {
               ++left;
            }
            while (right < nums.length && (visited[right] || nums[right] - nums[left] != 2 * k)) {
               ++right;
            }
            if (right == nums.length) {
               continue;
            }
            visited[left] = visited[right] = true;
            res[index++] = nums[left] + k;
         }
         if (index == nums.length / 2) {
            return res;
         }
      }
      return null;

   }

   // 406. 根据身高重建队列 (Queue Reconstruction by Height)
   public int[][] reconstructQueue(int[][] people) {
      Arrays.sort(people, new Comparator<int[]>() {

         @Override
         public int compare(int[] o1, int[] o2) {
            if (o1[0] == o2[0]) {
               return o1[1] - o2[1];
            }
            return o2[0] - o1[0];

         }
      });
      List<int[]> res = new ArrayList<>();
      for (int[] p : people) {
         res.add(p[1], p);
      }
      return res.toArray(new int[res.size()][]);

   }

   // 939. 最小面积矩形 (Minimum Area Rectangle)
   public int minAreaRect(int[][] points) {
      int res = Integer.MAX_VALUE;
      Set<Integer> set = new HashSet<>();
      for (int[] point : points) {
         set.add(point[0] * 40001 + point[1]);
      }
      for (int i = 0; i < points.length; ++i) {
         for (int j = i + 1; j < points.length; ++j) {
            if (points[i][0] != points[j][0] && points[i][1] != points[j][1]) {
               if (set.contains(points[i][0] * 40001 + points[j][1])
                     && set.contains(points[j][0] * 40001 + points[i][1])) {
                  res = Math.min(res, Math.abs(points[i][0] - points[j][0]) * Math.abs(points[i][1] - points[j][1]));
               }
            }
         }
      }
      return res == Integer.MAX_VALUE ? 0 : res;

   }

   // 1529. 最少的后缀翻转次数 (Minimum Suffix Flips)
   public int minFlips(String target) {
      int res = target.charAt(0) - '0';
      for (int i = 1; i < target.length(); ++i) {
         if (target.charAt(i) != target.charAt(i - 1)) {
            ++res;
         }
      }
      return res;
   }

   // 1029. 两地调度 (Two City Scheduling)
   public int twoCitySchedCost(int[][] costs) {
      Arrays.sort(costs, new Comparator<int[]>() {

         @Override
         public int compare(int[] o1, int[] o2) {
            return Integer.compare(o1[0] - o1[1], o2[0] - o2[1]);
         }

      });

      int res = 0;
      int n = costs.length / 2;
      for (int i = 0; i < n; ++i) {
         res += costs[i][0] + costs[i + n][1];
      }
      return res;

   }

   // 1824. 最少侧跳次数 (Minimum Sideway Jumps)
   public int minSideJumps(int[] obstacles) {
      int[] dp = new int[3];
      int pre0 = 1;
      int pre1 = 0;
      int pre2 = 1;
      for (int i = 1; i < obstacles.length; ++i) {
         Arrays.fill(dp, Integer.MAX_VALUE - 1);
         if (obstacles[i] != 1) {
            dp[0] = pre0;
         }
         if (obstacles[i] != 2) {
            dp[1] = pre1;
         }
         if (obstacles[i] != 3) {
            dp[2] = pre2;
         }
         if (obstacles[i] != 1) {
            dp[0] = Math.min(dp[0], Math.min(dp[1], dp[2]) + 1);
         }
         if (obstacles[i] != 2) {
            dp[1] = Math.min(dp[1], Math.min(dp[0], dp[2]) + 1);
         }
         if (obstacles[i] != 3) {
            dp[2] = Math.min(dp[2], Math.min(dp[0], dp[1]) + 1);
         }
         pre0 = dp[0];
         pre1 = dp[1];
         pre2 = dp[2];
      }
      return Arrays.stream(dp).min().getAsInt();

   }

   // 1824. 最少侧跳次数 (Minimum Sideway Jumps) -- 0-1 bfs
   public int minSideJumps2(int[] obstacles) {
      int n = obstacles.length;
      int[][] dis = new int[n][3];
      for (int i = 0; i < n; ++i) {
         Arrays.fill(dis[i], n);
      }
      dis[0][1] = 0;
      Deque<int[]> deque = new LinkedList<>();
      deque.offer(new int[] { 0, 1 });
      while (!deque.isEmpty()) {
         int[] cur = deque.pollFirst();
         int i = cur[0];
         int j = cur[1];
         int d = dis[i][j];
         if (i == n - 1) {
            return d;
         }
         if (obstacles[i + 1] != j + 1 && d < dis[i + 1][j]) {
            dis[i + 1][j] = d;
            deque.offerFirst(new int[] { i + 1, j });
         }
         for (int k : new int[] { (j + 1) % 3, (j + 2) % 3 }) {
            if (obstacles[i] != k + 1 && d + 1 < dis[i][k]) {
               dis[i][k] = d + 1;
               deque.offerLast(new int[] { i, k });
            }
         }
      }
      return -1;

   }

   // 1899. 合并若干三元组以形成目标三元组 (Merge Triplets to Form Target Triplet)
   public boolean mergeTriplets(int[][] triplets, int[] target) {
      int[] expected = new int[3];
      for (int[] triplet : triplets) {
         if (triplet[0] <= target[0] && triplet[1] <= target[1] && triplet[2] <= target[2]) {
            expected[0] = Math.max(expected[0], triplet[0]);
            expected[1] = Math.max(expected[1], triplet[1]);
            expected[2] = Math.max(expected[2], triplet[2]);
         }
      }
      return Arrays.equals(expected, target);
   }

   // 1846. 减小和重新排列数组后的最大元素 (Maximum Element After Decreasing and Rearranging)
   public int maximumElementAfterDecrementingAndRearranging(int[] arr) {
      Arrays.sort(arr);
      arr[0] = 1;
      for (int i = 1; i < arr.length; ++i) {
         arr[i] = Math.min(arr[i], arr[i - 1] + 1);
      }
      return arr[arr.length - 1];

   }

   // 1558. 得到目标数组的最少函数调用次数 (Minimum Numbers of Function Calls to Make Target
   // Array)
   public int minOperations(int[] nums) {
      int res = 0;
      int max = 0;
      for (int num : nums) {
         res += Integer.bitCount(num);
         max = Math.max(max, num);
      }
      if (max != 0) {
         res += Math.log(max) / Math.log(2);
      }
      return res;

   }

   // 1727. 重新排列后的最大子矩阵 (Largest Submatrix With Rearrangements)
   public int largestSubmatrix(int[][] matrix) {
      int m = matrix.length;
      int n = matrix[0].length;
      int res = 0;
      for (int i = 1; i < m; ++i) {
         for (int j = 0; j < n; ++j) {
            if (matrix[i][j] == 1) {
               matrix[i][j] += matrix[i - 1][j];
            }
         }
      }
      for (int i = 0; i < m; ++i) {
         Arrays.sort(matrix[i]);
         for (int j = n - 1; j >= 0; --j) {
            res = Math.max(res, matrix[i][j] * (n - j));
         }
      }
      return res;

   }

   // 646. 最长数对链 (Maximum Length of Pair Chain)
   public int findLongestChain(int[][] pairs) {
      Arrays.sort(pairs, new Comparator<int[]>() {

         @Override
         public int compare(int[] o1, int[] o2) {
            return o1[1] - o2[1];
         }
      });
      int pre = Integer.MIN_VALUE;
      int res = 0;
      for (int[] pair : pairs) {
         if (pre < pair[0]) {
            ++res;
            pre = pair[1];
         }
      }
      return res;

   }

   // 1090. 受标签影响的最大值 (Largest Values From Labels)
   public int largestValsFromLabels(int[] values, int[] labels, int numWanted, int useLimit) {
      int n = values.length;
      Integer[] ids = IntStream.range(0, n).boxed().toArray(Integer[]::new);
      Arrays.sort(ids, new Comparator<Integer>() {

         @Override
         public int compare(Integer o1, Integer o2) {
            return Integer.compare(values[o2], values[o1]);
         }

      });
      Map<Integer, Integer> lab = new HashMap<>();
      int res = 0;
      for (int id : ids) {
         int labelCnts = lab.getOrDefault(labels[id], 0);
         if (labelCnts < useLimit) {
            lab.merge(labels[id], 1, Integer::sum);
            res += values[id];
            --numWanted;
         }
         if (numWanted == 0) {
            break;
         }
      }
      return res;

   }

   // 1005. K 次取反后最大化的数组和 (Maximize Sum Of Array After K Negations)
   public int largestSumAfterKNegations(int[] nums, int k) {
      Arrays.sort(nums);
      for (int i = 0; i < nums.length; ++i) {
         if (nums[i] < 0 && k > 0) {
            nums[i] = -nums[i];
            --k;
         } else {
            break;
         }
      }
      int res = 0;
      int min = nums[0];
      for (int i = 0; i < nums.length; ++i) {
         if (nums[i] < min) {
            min = nums[i];
         }
         res += nums[i];
      }

      if (k % 2 == 1) {
         res -= min * 2;
      }
      return res;
   }

   // 680. 验证回文字符串 Ⅱ (Valid Palindrome II)
   // 剑指 Offer II 019. 最多删除一个字符得到回文
   public boolean validPalindrome(String s) {
      char[] chars = s.toCharArray();
      int left = 0;
      int right = chars.length - 1;
      while (left < right) {
         if (chars[left] == chars[right]) {
            ++left;
            --right;
         } else {
            return judge680(chars, left + 1, right) || judge680(chars, left, right - 1);
         }
      }
      return true;

   }

   private boolean judge680(char[] chars, int left, int right) {
      while (left < right) {
         if (chars[left] != chars[right]) {
            return false;
         }
         ++left;
         --right;
      }
      return true;
   }

   // 1996. 游戏中弱角色的数量 (The Number of Weak Characters in the Game) --排序+贪心
   public int numberOfWeakCharacters(int[][] properties) {
      Arrays.sort(properties, new Comparator<int[]>() {

         @Override
         public int compare(int[] o1, int[] o2) {
            if (o1[0] == o2[0]) {
               return o1[1] - o2[1];
            }
            return o2[0] - o1[0];
         }
      });

      int max = 0;
      int res = 0;
      for (int[] property : properties) {
         if (property[1] < max) {
            ++res;
         } else {
            max = property[1];
         }
      }
      return res;

   }

   // 1996. 游戏中弱角色的数量 (The Number of Weak Characters in the Game) --排序+单调栈
   public int numberOfWeakCharacters2(int[][] properties) {
      Arrays.sort(properties, new Comparator<int[]>() {

         @Override
         public int compare(int[] o1, int[] o2) {
            if (o1[0] == o2[0]) {
               return o2[1] - o1[1];
            }
            return o1[0] - o2[0];
         }
      });
      int res = 0;
      Stack<Integer> stack = new Stack<>();
      for (int i = 0; i < properties.length; ++i) {
         while (!stack.isEmpty() && properties[i][1] > stack.peek()) {
            ++res;
            stack.pop();
         }
         stack.push(properties[i][1]);
      }
      return res;
   }

   // 1536. 排布二进制网格的最少交换次数 (Minimum Swaps to Arrange a Binary Grid)
   public int minSwaps(int[][] grid) {
      int n = grid.length;
      List<Integer> list = new ArrayList<>();
      for (int i = 0; i < n; ++i) {
         int count = 0;
         for (int j = n - 1; j >= 0; --j) {
            if (grid[i][j] != 0) {
               break;
            }
            ++count;
         }
         list.add(count);
      }
      int res = 0;
      search: for (int i = 0; i < n; ++i) {
         int threshold = n - i - 1;
         for (int j = 0; j < list.size(); ++j) {
            if (list.get(j) >= threshold) {
               res += j;
               list.remove(j);
               continue search;
            }
         }
         return -1;
      }
      return res;
   }

   // 1328. 破坏回文串 (Break a Palindrome)
   public String breakPalindrome(String palindrome) {
      char[] chars = palindrome.toCharArray();
      if (chars.length == 1) {
         return "";
      }
      int left = 0;
      int right = chars.length - 1;
      while (left < right && chars[left] == 'a') {
         ++left;
         --right;
      }
      if (left < right) {
         chars[left] = 'a';
      } else {
         chars[chars.length - 1] = 'b';
      }
      return String.valueOf(chars);

   }

   // 1147. 段式回文 (Longest Chunked Palindrome Decomposition)
   private char[] chars1147;
   private int n1147;

   public int longestDecomposition(String text) {
      this.chars1147 = text.toCharArray();
      this.n1147 = chars1147.length;
      return dfs1147(0, n1147 - 1);

   }

   private int dfs1147(int i, int j) {
      if (i > j) {
         return 0;
      }
      int left = 0;
      int right = 0;
      int base = 31;
      int mul = 1;
      int mod = (int) (1e9 + 7);
      while (i < j) {
         left = (int) (((long) left * base + chars1147[i]) % mod);
         right = (int) ((right + (long) mul * chars1147[j]) % mod);
         if (left == right) {
            return dfs1147(i + 1, j - 1) + 2;
         }
         mul = (int) (((long) mul * base) % mod);
         ++i;
         --j;
      }
      return 1;
   }

   // 2087. 网格图中机器人回家的最小代价 (Minimum Cost Homecoming of a Robot in a Grid)
   public int minCost(int[] startPos, int[] homePos, int[] rowCosts, int[] colCosts) {
      int res = 0;
      int minRow = Math.min(startPos[0], homePos[0]);
      int maxRow = Math.max(startPos[0], homePos[0]);
      int minCol = Math.min(startPos[1], homePos[1]);
      int maxCol = Math.max(startPos[1], homePos[1]);
      for (int i = minRow; i <= maxRow; ++i) {
         res += rowCosts[i];
      }
      for (int i = minCol; i <= maxCol; ++i) {
         res += colCosts[i];
      }
      res -= rowCosts[startPos[0]];
      res -= colCosts[startPos[1]];
      return res;

   }

   // 2154. 将找到的值乘以 2 (Keep Multiplying Found Values by Two)
   public int findFinalValue(int[] nums, int original) {
      int mask = 0;
      for (int x : nums) {
         int m = x % original;
         int d = x / original;
         if (m == 0 && (d & (d - 1)) == 0) {
            mask |= d;
         }
      }
      mask = ~mask;
      return original * (mask & -mask);
   }

   // 5981. 分组得分最高的所有下标
   public List<Integer> maxScoreIndices(int[] nums) {
      int n = nums.length;
      int[] left = new int[n + 1];
      for (int i = 1; i < n + 1; ++i) {
         left[i] = (nums[i - 1] == 0 ? 1 : 0) + left[i - 1];
      }
      int[] right = new int[n + 1];
      for (int i = n - 1; i >= 0; --i) {
         right[i] = nums[i] + right[i + 1];
      }
      List<Integer> res = new ArrayList<>();
      int max = 0;
      for (int i = 0; i < n + 1; ++i) {
         int sum = left[i] + right[i];
         if (sum > max) {
            max = sum;
            res.clear();
            res.add(i);
         } else if (sum == max) {
            res.add(i);
         }
      }
      return res;

   }

   // 1024. 视频拼接 (Video Stitching) --动态规划 还需掌握贪心
   public int videoStitching(int[][] clips, int time) {
      int[] dp = new int[time + 1];
      Arrays.fill(dp, Integer.MAX_VALUE - 1);
      dp[0] = 0;
      for (int i = 1; i < dp.length; ++i) {
         for (int[] clip : clips) {
            if (clip[0] < i && i <= clip[1]) {
               dp[i] = Math.min(dp[i], dp[clip[0]] + 1);
            }
         }
      }
      return dp[time] == Integer.MAX_VALUE - 1 ? -1 : dp[time];

   }

   // 1024. 视频拼接 (Video Stitching) --排序+贪心
   public int videoStitching2(int[][] clips, int time) {
      Arrays.sort(clips, new Comparator<int[]>() {

         @Override
         public int compare(int[] o1, int[] o2) {
            if (o1[0] == o2[0]) {
               return o2[1] - o1[1];
            }
            return o1[0] - o2[0];
         }

      });

      int res = 0;
      int index = 0;
      int curEnd = 0;
      int nextEnd = 0;
      while (index < clips.length && clips[index][0] <= curEnd) {
         while (index < clips.length && clips[index][0] <= curEnd) {
            nextEnd = Math.max(nextEnd, clips[index][1]);
            ++index;
         }
         ++res;
         curEnd = nextEnd;
         if (curEnd >= time) {
            return res;
         }
      }
      return -1;

   }

   // 1024. 视频拼接 (Video Stitching) --贪心
   public int videoStitching3(int[][] clips, int time) {
      int[] rightMost = new int[time + 1];
      for (int[] clip : clips) {
         if (clip[0] <= time) {
            rightMost[clip[0]] = Math.max(rightMost[clip[0]], clip[1]);
         }
      }
      int res = 0;
      int right = 0;
      int cur = 0;
      for (int i = 0; i <= time; ++i) {
         right = Math.max(right, rightMost[i]);
         if (cur >= time) {
            return res;
         }
         if (cur == i) {
            ++res;
            cur = right;
         }
      }
      return -1;

   }

   // 1262. 可被三整除的最大和 (Greatest Sum Divisible by Three) --贪心
   public int maxSumDivThree(int[] nums) {
      int[][] a = new int[2][2];
      for (int i = 0; i < 2; ++i) {
         Arrays.fill(a[i], Integer.MAX_VALUE / 2);
      }
      int s = 0;
      for (int x : nums) {
         s += x;
         int m = x % 3;
         if (m == 0) {
            continue;
         }
         --m;
         if (x < a[m][0]) {
            a[m][1] = a[m][0];
            a[m][0] = x;
         } else if (x < a[m][1]) {
            a[m][1] = x;
         }
      }
      int m = s % 3;
      if (m == 0) {
         return s;
      }
      --m;
      return Math.max(0, Math.max(s - a[m][0], s - a[m ^ 1][0] - a[m ^ 1][1]));

   }

   // 1262. 可被三整除的最大和 (Greatest Sum Divisible by Three) --记忆化搜索
   private int[] nums1262;
   private int[][] memo1262;

   public int maxSumDivThree2(int[] nums) {
      int n = nums.length;
      this.nums1262 = nums;
      this.memo1262 = new int[n][3];
      return dfs1262(n - 1, 0);
   }

   private int dfs1262(int i, int j) {
      if (i < 0) {
         return j == 0 ? 0 : Integer.MIN_VALUE / 2;
      }
      if (memo1262[i][j] != 0) {
         return memo1262[i][j];
      }
      return memo1262[i][j] = Math.max(dfs1262(i - 1, (nums1262[i] + j) % 3) + nums1262[i], dfs1262(i - 1, j));
   }

   // 1262. 可被三整除的最大和 (Greatest Sum Divisible by Three) --递推
   public int maxSumDivThree3(int[] nums) {
      int[] reminder = new int[3];
      for (int num : nums) {
         int a = reminder[0] + num;
         int b = reminder[1] + num;
         int c = reminder[2] + num;
         reminder[a % 3] = Math.max(reminder[a % 3], a);
         reminder[b % 3] = Math.max(reminder[b % 3], b);
         reminder[c % 3] = Math.max(reminder[c % 3], c);
      }
      return reminder[0];

   }

   // 1702. 修改后的最大二进制字符串 (Maximum Binary String After Change)
   public String maximumBinaryString(String binary) {
      char[] chars = binary.toCharArray();
      int left = 0;
      int right = 0;
      boolean flag = true;
      for (char c : chars) {
         if (flag) {
            if (c - '0' == 1) {
               ++left;
            } else {
               flag = false;
            }
         } else {
            if (c - '0' == 1) {
               ++right;
            }
         }
      }
      if (left + right != chars.length) {
         int reverseIndex = chars.length - right - 1;
         Arrays.fill(chars, '1');
         chars[reverseIndex] = '0';
      }
      return String.valueOf(chars);

   }

   // 1665. 完成所有任务的最少初始能量 (Minimum Initial Energy to Finish Tasks)
   public int minimumEffort(int[][] tasks) {
      Arrays.sort(tasks, new Comparator<int[]>() {

         @Override
         public int compare(int[] o1, int[] o2) {
            return (o1[1] - o1[0]) - (o2[1] - o2[0]);
         }

      });
      int res = 0;
      for (int[] task : tasks) {
         res = Math.max(res + task[0], task[1]);
      }
      return res;

   }

   // 1798. 你能构造出连续值的最大数目 (Maximum Number of Consecutive Values You Can Make)
   public int getMaximumConsecutive(int[] coins) {
      Arrays.sort(coins);
      int res = 1;
      for (int coin : coins) {
         if (coin > res) {
            break;
         }
         res += coin;
      }
      return res;

   }

   // 659. 分割数组为连续子序列 (Split Array into Consecutive Subsequences)
   public boolean isPossible(int[] nums) {
      Map<Integer, Integer> counts = new HashMap<>();
      Map<Integer, Integer> tails = new HashMap<>();
      for (int num : nums) {
         counts.put(num, counts.getOrDefault(num, 0) + 1);
      }
      for (int num : nums) {
         if (counts.getOrDefault(num, 0) == 0) {
            continue;
         }
         if (tails.getOrDefault(num - 1, 0) > 0) {
            tails.put(num - 1, tails.getOrDefault(num - 1, 0) - 1);
            tails.put(num, tails.getOrDefault(num, 0) + 1);
            counts.put(num, counts.getOrDefault(num, 0) - 1);
         } else if (counts.getOrDefault(num + 1, 0) > 0 && counts.getOrDefault(num + 2, 0) > 0) {
            counts.put(num, counts.getOrDefault(num, 0) - 1);
            counts.put(num + 1, counts.getOrDefault(num + 1, 0) - 1);
            counts.put(num + 2, counts.getOrDefault(num + 2, 0) - 1);
            tails.put(num + 2, tails.getOrDefault(num + 2, 0) + 1);
         } else {
            return false;
         }
      }
      return true;

   }

   class TreeNode {
      private int val;
      private TreeNode left;
      private TreeNode right;

      public TreeNode() {

      }

      public TreeNode(int val) {
         this.val = val;
      }

      public TreeNode(int val, TreeNode left, TreeNode right) {
         this.val = val;
         this.left = left;
         this.right = right;
      }

   }

   // 173. 二叉搜索树迭代器 (Binary Search Tree Iterator)
   // 剑指 Offer II 055. 二叉搜索树迭代器 --递归 实现二叉树中序遍历
   class BSTIterator {
      private int index;
      private List<Integer> list;

      public BSTIterator(TreeNode root) {
         this.index = 0;
         this.list = new ArrayList<>();
         inorderTraversal(root, list);
      }

      private void inorderTraversal(TreeNode root, List<Integer> list) {
         if (root == null) {
            return;
         }
         inorderTraversal(root.left, list);
         list.add(root.val);
         inorderTraversal(root.right, list);
      }

      public int next() {
         return list.get(index++);
      }

      public boolean hasNext() {
         return index < list.size();
      }
   }

   // 173. 二叉搜索树迭代器 (Binary Search Tree Iterator)
   // 剑指 Offer II 055. 二叉搜索树迭代器 --stack栈 实现二叉树中序遍历
   class BSTIterator2 {
      private Stack<TreeNode> stack;
      private TreeNode curNode;

      public BSTIterator2(TreeNode root) {
         this.curNode = root;
         this.stack = new Stack<>();
      }

      public int next() {
         while (curNode != null) {
            stack.push(curNode);
            curNode = curNode.left;
         }
         curNode = stack.pop();
         int res = curNode.val;
         curNode = curNode.right;
         return res;

      }

      public boolean hasNext() {
         return !stack.isEmpty() || curNode != null;
      }
   }

   // 144. 二叉树的前序遍历 (Binary Tree Preorder Traversal) -- 递归
   public List<Integer> preorderTraversal(TreeNode root) {
      List<Integer> res = new ArrayList<>();
      preorderTraversal144(root, res);
      return res;
   }

   private void preorderTraversal144(TreeNode root, List<Integer> res) {
      if (root == null) {
         return;
      }
      res.add(root.val);
      preorderTraversal144(root.left, res);
      preorderTraversal144(root.right, res);
   }

   // 94. 二叉树的中序遍历 (Binary Tree Inorder Traversal) --递归
   public List<Integer> inorderTraversal(TreeNode root) {
      List<Integer> res = new ArrayList<>();
      inorderTraversal94(root, res);
      return res;

   }

   private void inorderTraversal94(TreeNode root, List<Integer> res) {
      if (root == null) {
         return;
      }
      inorderTraversal94(root.left, res);
      res.add(root.val);
      inorderTraversal94(root.right, res);
   }

   // 145. 二叉树的后序遍历 (Binary Tree Postorder Traversal) --递归
   public List<Integer> postorderTraversal(TreeNode root) {
      List<Integer> res = new ArrayList<>();
      postorderTraversal145(root, res);
      return res;

   }

   private void postorderTraversal145(TreeNode root, List<Integer> res) {
      if (root == null) {
         return;
      }
      postorderTraversal145(root.left, res);
      postorderTraversal145(root.right, res);
      res.add(root.val);
   }

   // 144. 二叉树的前序遍历 (Binary Tree Preorder Traversal) --栈+迭代
   public List<Integer> preorderTraversal2(TreeNode root) {
      List<Integer> res = new ArrayList<>();
      if (root == null) {
         return res;
      }
      Stack<TreeNode> stack = new Stack<>();
      stack.add(root);
      while (!stack.isEmpty()) {
         root = stack.pop();
         res.add(root.val);
         if (root.right != null) {
            stack.push(root.right);
         }
         if (root.left != null) {
            stack.push(root.left);
         }
      }
      return res;
   }

   // 94. 二叉树的中序遍历 (Binary Tree Inorder Traversal) --迭代+栈 空间-O(n)
   public List<Integer> inorderTraversal2(TreeNode root) {
      Stack<TreeNode> stack = new Stack<>();
      List<Integer> res = new ArrayList<>();
      while (!stack.isEmpty() || root != null) {
         while (root != null) {
            stack.push(root);
            root = root.left;
         }
         if (!stack.isEmpty()) {
            root = stack.pop();
            res.add(root.val);
            root = root.right;
         }
      }
      return res;

   }

   // 145. 二叉树的后序遍历 (Binary Tree Postorder Traversal) --两个栈(本质用的是前序遍历的思路)
   public List<Integer> postorderTraversal2(TreeNode root) {
      List<Integer> res = new ArrayList<>();
      if (root == null) {
         return res;
      }
      Stack<TreeNode> stack1 = new Stack<>();
      Stack<TreeNode> stack2 = new Stack<>();
      stack1.push(root);
      while (!stack1.isEmpty()) {
         root = stack1.pop();
         stack2.push(root);
         if (root.left != null) {
            stack1.push(root.left);
         }
         if (root.right != null) {
            stack1.push(root.right);
         }
      }
      while (!stack2.isEmpty()) {
         res.add(stack2.pop().val);
      }
      return res;

   }

   // 145. 二叉树的后序遍历 (Binary Tree Postorder Traversal) --一个栈
   public List<Integer> postorderTraversal3(TreeNode root) {
      List<Integer> res = new ArrayList<>();
      if (root == null) {
         return res;
      }
      TreeNode cur = root;
      Stack<TreeNode> stack = new Stack<>();
      stack.push(root);
      while (!stack.isEmpty()) {
         TreeNode peek = stack.peek();
         if (peek.left != null && peek.left != cur && peek.right != cur) {
            stack.push(peek.left);
         } else if (peek.right != null && peek.right != cur) {
            stack.push(peek.right);
         } else {
            res.add(stack.pop().val);
            cur = peek;
         }
      }
      return res;

   }

   // 94. 二叉树的中序遍历 (Binary Tree Inorder Traversal) --莫里斯迭代 空间-O(1)
   public List<Integer> inorderTraversal3(TreeNode root) {
      List<Integer> res = new ArrayList<>();
      TreeNode pre = null;
      while (root != null) {
         if (root.left != null) {
            pre = root.left;
            while (pre.right != null && pre.right != root) {
               pre = pre.right;
            }
            if (pre.right == null) {
               pre.right = root;
               root = root.left;
            } else {
               res.add(root.val);
               pre.right = null;
               root = root.right;
            }
         } else {
            res.add(root.val);
            root = root.right;
         }
      }
      return res;
   }

   // 538. 把二叉搜索树转换为累加树 (Convert BST to Greater Tree)
   // 1038. 把二叉搜索树转换为累加树 (Binary Search Tree to Greater Sum Tree)
   // 剑指 Offer II 054. 所有大于等于节点的值之和 --栈+反向中序遍历
   public TreeNode convertBST(TreeNode root) {
      TreeNode head = root;
      int sum = 0;
      Stack<TreeNode> stack = new Stack<>();
      while (!stack.isEmpty() || head != null) {
         while (head != null) {
            stack.push(head);
            head = head.right;
         }
         if (!stack.isEmpty()) {
            head = stack.pop();
            sum += head.val;
            head.val = sum;
            head = head.left;
         }
      }
      return root;

   }

   // 538. 把二叉搜索树转换为累加树 (Convert BST to Greater Tree)
   // 1038. 把二叉搜索树转换为累加树 (Binary Search Tree to Greater Sum Tree)
   // 剑指 Offer II 054. 所有大于等于节点的值之和 --递归+反向中序遍历
   private int sum;

   public TreeNode convertBST2(TreeNode root) {
      TreeNode head = root;
      inorderTraversal538(head);
      return root;
   }

   private void inorderTraversal538(TreeNode head) {
      if (head == null) {
         return;
      }
      inorderTraversal538(head.right);
      sum += head.val;
      head.val = sum;
      inorderTraversal538(head.left);
   }

   // 538. 把二叉搜索树转换为累加树 (Convert BST to Greater Tree)
   // 1038. 把二叉搜索树转换为累加树 (Binary Search Tree to Greater Sum Tree)
   // 剑指 Offer II 054. 所有大于等于节点的值之和 --莫里斯迭代+反向中序遍历
   public TreeNode bstToGst(TreeNode root) {
      TreeNode head = root;
      TreeNode pre = null;
      int sum = 0;
      while (head != null) {
         if (head.right != null) {
            pre = head.right;
            while (pre.left != null && pre.left != head) {
               pre = pre.left;
            }
            if (pre.left == null) {
               pre.left = head;
               head = head.right;
            } else {
               sum += head.val;
               head.val = sum;
               pre.left = null;
               head = head.left;
            }

         } else {
            sum += head.val;
            head.val = sum;
            head = head.left;
         }
      }
      return root;

   }

   // 938. 二叉搜索树的范围和 (Range Sum of BST) --递归+分情况
   public int rangeSumBST(TreeNode root, int low, int high) {
      if (root == null) {
         return 0;
      }
      if (root.val > high) {
         return rangeSumBST(root.left, low, high);
      }
      if (root.val < low) {
         return rangeSumBST(root.right, low, high);
      }
      return root.val + rangeSumBST(root.left, low, high) + rangeSumBST(root.right, low, high);
   }

   // 938. 二叉搜索树的范围和 (Range Sum of BST) --bfs
   public int rangeSumBST2(TreeNode root, int low, int high) {
      Queue<TreeNode> q = new LinkedList<>();
      q.offer(root);
      int res = 0;
      while (!q.isEmpty()) {
         TreeNode node = q.poll();
         if (node == null) {
            continue;
         }
         if (high < node.val) {
            q.offer(node.left);
         } else if (low > node.val) {
            q.offer(node.right);
         } else {
            res += node.val;
            q.offer(node.left);
            q.offer(node.right);
         }
      }
      return res;

   }

   // 108. 将有序数组转换为二叉搜索树 (Convert Sorted Array to Binary Search Tree)
   // 面试题 04.02. 最小高度树 (Minimum Height Tree LCCI) --递归
   public TreeNode sortedArrayToBST(int[] nums) {
      return dfs108(0, nums.length - 1, nums);
   }

   private TreeNode dfs108(int i, int j, int[] nums) {
      if (i > j) {
         return null;
      }
      int mid = i + ((j - i) >> 1);
      TreeNode node = new TreeNode(nums[mid]);
      node.left = dfs108(i, mid - 1, nums);
      node.right = dfs108(mid + 1, j, nums);
      return node;
   }

   // 700. 二叉搜索树中的搜索 (Search in a Binary Search Tree) --迭代
   public TreeNode searchBST(TreeNode root, int val) {
      while (root != null) {
         if (root.val == val) {
            return root;
         } else if (root.val > val) {
            root = root.left;
         } else if (root.val < val) {
            root = root.right;
         }
      }
      return null;
   }

   // 109. 有序链表转换二叉搜索树 (Convert Sorted List to Binary Search Tree) --分治+递归
   public TreeNode sortedListToBST(ListNode head) {
      return build109(head, null);
   }

   private TreeNode build109(ListNode left, ListNode right) {
      if (left == right) {
         return null;
      }
      ListNode mid = getMiddle(left, right);
      TreeNode node = new TreeNode(mid.val);
      node.left = build109(left, mid);
      node.right = build109(mid.next, right);
      return node;
   }

   private ListNode getMiddle(ListNode left, ListNode right) {
      ListNode fast = left;
      ListNode slow = left;
      while (fast != right && fast.next != right) {
         fast = fast.next.next;
         slow = slow.next;
      }
      return slow;
   }

   // 109. 有序链表转换二叉搜索树 (Convert Sorted List to Binary Search Tree) --中序遍历
   private ListNode globalHead;

   public TreeNode sortedListToBST2(ListNode head) {
      globalHead = head;
      ListNode cur = head;
      int count = getCount109(cur);
      return build109(0, count - 1);

   }

   private TreeNode build109(int left, int right) {
      if (left > right) {
         return null;
      }
      int mid = left + ((right - left) >>> 1);
      TreeNode node = new TreeNode();
      node.left = build109(left, mid - 1);
      node.val = globalHead.val;
      globalHead = globalHead.next;
      node.right = build109(mid + 1, right);
      return node;
   }

   private int getCount109(ListNode cur) {
      int count = 0;
      while (cur != null) {
         ++count;
         cur = cur.next;
      }
      return count;
   }

   // 剑指 Offer 54. 二叉搜索树的第k大节点 -- 递归+逆中序
   private int k54;
   private int res54;

   public int kthLargest(TreeNode root, int k) {
      this.k54 = k;
      inorderTraversalOffer54(root);
      return res54;
   }

   private void inorderTraversalOffer54(TreeNode root) {
      if (root == null || k54 == 0) {
         return;
      }
      inorderTraversalOffer54(root.right);
      if (--k54 == 0) {
         res54 = root.val;
         return;
      }
      inorderTraversalOffer54(root.left);
   }

   // 剑指 Offer 54. 二叉搜索树的第k大节点 -- 栈+逆中序
   public int kthLargest2(TreeNode root, int k) {
      Stack<TreeNode> stack = new Stack<>();
      while (root != null || !stack.isEmpty()) {
         while (root != null) {
            stack.push(root);
            root = root.right;
         }
         root = stack.pop();
         if (--k == 0) {
            return root.val;
         }
         root = root.left;
      }
      return -1;
   }

   // 剑指 Offer 54. 二叉搜索树的第k大节点 -- 莫里斯迭代+逆中序
   public int kthLargest3(TreeNode root, int k) {
      TreeNode pre = null;
      while (root != null) {
         if (root.right != null) {
            pre = root.right;
            while (pre.left != null && pre.left != root) {
               pre = pre.left;
            }
            if (pre.left == null) {
               pre.left = root;
               root = root.right;
            } else {
               if (--k == 0) {
                  return root.val;
               }
               pre.left = null;
               root = root.left;
            }

         } else {
            if (--k == 0) {
               return root.val;
            }
            root = root.left;
         }
      }
      return -1;
   }

   // 230. 二叉搜索树中第K小的元素 (Kth Smallest Element in a BST) --递归+中序
   private int k230;
   private int res230;

   public int kthSmallest(TreeNode root, int k) {
      this.k230 = k;
      inorderTraversal230(root);
      return res230;

   }

   private void inorderTraversal230(TreeNode root) {
      if (root == null || k230 == 0) {
         return;
      }
      inorderTraversal230(root.left);
      if (--k230 == 0) {
         res230 = root.val;
         return;
      }
      inorderTraversal230(root.right);
   }

   // 230. 二叉搜索树中第K小的元素 (Kth Smallest Element in a BST) --栈+中序
   public int kthSmallest2(TreeNode root, int k) {
      Stack<TreeNode> stack = new Stack<>();
      while (root != null || !stack.isEmpty()) {
         while (root != null) {
            stack.push(root);
            root = root.left;
         }
         root = stack.pop();
         if (--k == 0) {
            return root.val;
         }
         root = root.right;
      }
      return -1;

   }

   // 230. 二叉搜索树中第K小的元素 (Kth Smallest Element in a BST) --莫里斯迭代+中序
   public int kthSmallest3(TreeNode root, int k) {
      TreeNode pre = null;
      while (root != null) {
         if (root.left != null) {
            pre = root.left;
            while (pre.right != null && pre.right != root) {
               pre = pre.right;
            }
            if (pre.right == null) {
               pre.right = root;
               root = root.left;
            } else {
               if (--k == 0) {
                  return root.val;
               }
               pre.right = null;
               root = root.right;
            }
         } else {
            if (--k == 0) {
               return root.val;
            }
            root = root.right;
         }
      }
      return -1;
   }

   // 897. 递增顺序搜索树 (Increasing Order Search Tree)
   // 剑指 Offer II 052. 展平二叉搜索树 --莫里斯迭代 + 中序遍历 + 新创建一棵树
   public TreeNode increasingBST(TreeNode root) {
      TreeNode dummy = new TreeNode(-1, null, root);
      TreeNode cur = dummy;
      TreeNode pre = null;
      while (root != null) {
         if (root.left != null) {
            pre = root.left;
            while (pre.right != null && pre.right != root) {
               pre = pre.right;
            }
            if (pre.right == null) {
               pre.right = root;
               root = root.left;
            } else {
               cur.right = new TreeNode(root.val);
               cur = cur.right;
               pre.right = null;
               root = root.right;
            }
         } else {
            cur.right = new TreeNode(root.val);
            cur = cur.right;
            root = root.right;
         }
      }
      return dummy.right;

   }

   // 面试题 17.12. BiNode
   // 897. 递增顺序搜索树 (Increasing Order Search Tree)
   // 剑指 Offer II 052. 展平二叉搜索树 --莫里斯迭代 + 中序遍历 + 直接本地修改
   public TreeNode increasingBST2(TreeNode root) {
      TreeNode dummy = new TreeNode(-1, null, root);
      TreeNode cur = dummy;
      TreeNode pre = null;
      while (root != null) {
         if (root.left != null) {
            pre = root.left;
            while (pre.right != null && pre.right != root) {
               pre = pre.right;
            }
            if (pre.right == null) {
               pre.right = root;
               root = root.left;
            } else {
               pre.right = null;
               cur.right = root;
               root.left = null;
               cur = cur.right;
               root = root.right;
            }
         } else {
            cur.right = root;
            root.left = null;
            cur = cur.right;
            root = root.right;
         }
      }
      return dummy.right;

   }

   // 面试题 17.12. BiNode
   // 98. 验证二叉搜索树 (Validate Binary Search Tree) --递归
   // 面试题 04.05. Legal Binary Search Tree LCCI
   public boolean isValidBST(TreeNode root) {
      return isValid98(root, Long.MIN_VALUE, Long.MAX_VALUE);
   }

   private boolean isValid98(TreeNode root, long minValue, long maxValue) {
      if (root == null) {
         return true;
      }
      if (root.val <= minValue || root.val >= maxValue) {
         return false;
      }
      return isValid98(root.left, minValue, root.val) && isValid98(root.right, root.val, maxValue);
   }

   // 面试题 17.12. BiNode
   // 98. 验证二叉搜索树 (Validate Binary Search Tree) --莫里斯迭代 + 中序遍历
   // 面试题 04.05. Legal Binary Search Tree LCCI
   public boolean isValidBST2(TreeNode root) {
      TreeNode predecessor = null;
      Integer pre = null;
      while (root != null) {
         if (root.left != null) {
            predecessor = root.left;
            while (predecessor.right != null && predecessor.right != root) {
               predecessor = predecessor.right;
            }
            if (predecessor.right == null) {
               predecessor.right = root;
               root = root.left;
            } else {
               predecessor.right = null;
               if (pre != null && root.val <= pre) {
                  return false;
               }
               pre = root.val;
               root = root.right;
            }
         } else {
            if (pre != null && root.val <= pre) {
               return false;
            }
            pre = root.val;
            root = root.right;
         }
      }
      return true;
   }

   // 面试题 17.12. BiNode
   // 98. 验证二叉搜索树 (Validate Binary Search Tree) --dfs 隐式栈
   // 面试题 04.05. Legal Binary Search Tree LCCI
   private Integer pre98;
   private boolean flag98 = true;

   public boolean isValidBST3(TreeNode root) {
      dfs98(root);
      return flag98;

   }

   private void dfs98(TreeNode root) {
      if (root == null) {
         return;
      }
      dfs98(root.left);
      if (pre98 == null || root.val > pre98) {
         pre98 = root.val;
      } else {
         flag98 = false;
         return;
      }
      dfs98(root.right);
   }

   // 面试题 17.12. BiNode
   // 98. 验证二叉搜索树 (Validate Binary Search Tree) --dfs 显式栈
   // 面试题 04.05. Legal Binary Search Tree LCCI
   public boolean isValidBST4(TreeNode root) {
      Stack<TreeNode> stack = new Stack<>();
      Integer pre = null;
      while (!stack.isEmpty() || root != null) {
         while (root != null) {
            stack.push(root);
            root = root.left;
         }
         if (!stack.isEmpty()) {
            TreeNode cur = stack.pop();
            if (pre != null && cur.val <= pre) {
               return false;
            }
            pre = cur.val;
            root = cur.right;
         }
      }
      return true;
   }

   // 235. 二叉搜索树的最近公共祖先 (Lowest Common Ancestor of a Binary Search Tree)
   // 剑指 Offer 68 - I. 二叉搜索树的最近公共祖先
   public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
      if (root.val >= p.val && root.val <= q.val || root.val <= p.val && root.val >= q.val) {
         return root;
      }
      if (p.val < root.val) {
         return lowestCommonAncestor(root.left, p, q);
      }
      return lowestCommonAncestor(root.right, p, q);

   }

   // 501. 二叉搜索树中的众数 (Find Mode in Binary Search Tree)
   public int[] findMode(TreeNode root) {
      TreeNode cur = root;
      int max = getMaxFrequent(cur);
      cur = root;
      return getResult501(cur, max);

   }

   private int[] getResult501(TreeNode node, int max) {
      List<Integer> list = new ArrayList<>();
      TreeNode pre = null;
      int count = 1;
      int preVal = Integer.MIN_VALUE;
      while (node != null) {
         if (node.left != null) {
            pre = node.left;
            while (pre.right != null && pre.right != node) {
               pre = pre.right;
            }
            if (pre.right == null) {
               pre.right = node;
               node = node.left;
            } else {
               pre.right = null;
               if (node.val > preVal) {
                  count = 1;
                  preVal = node.val;
               } else {
                  ++count;
               }
               if (count == max) {
                  list.add(node.val);
               }
               node = node.right;
            }

         } else {
            if (node.val > preVal) {
               count = 1;
               preVal = node.val;
            } else {
               ++count;
            }
            if (count == max) {
               list.add(node.val);
            }
            node = node.right;
         }
      }
      int[] res = new int[list.size()];
      for (int i = 0; i < res.length; ++i) {
         res[i] = list.get(i);
      }
      return res;
   }

   private int getMaxFrequent(TreeNode node) {
      TreeNode pre = null;
      int max = 1;
      int count = 1;
      int preVal = Integer.MIN_VALUE;
      while (node != null) {
         if (node.left != null) {
            pre = node.left;
            while (pre.right != null && pre.right != node) {
               pre = pre.right;
            }
            if (pre.right == null) {
               pre.right = node;
               node = node.left;
            } else {
               pre.right = null;
               if (node.val > preVal) {
                  count = 1;
                  preVal = node.val;
               } else {
                  max = Math.max(max, ++count);
               }
               node = node.right;
            }
         } else {
            if (node.val > preVal) {
               count = 1;
               preVal = node.val;
            } else {
               max = Math.max(max, ++count);
            }
            node = node.right;
         }
      }
      return max;
   }

   // 285
   // 面试题 04.06. 后继者 (Successor LCCI)
   // 剑指 Offer II 053. 二叉搜索树中的中序后继
   public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
      boolean flag = false;
      TreeNode pre = null;
      while (root != null) {
         if (root.left != null) {
            pre = root.left;
            while (pre.right != null && pre.right != root) {
               pre = pre.right;
            }
            if (pre.right == null) {
               pre.right = root;
               root = root.left;
            } else {
               pre.right = null;
               if (flag) {
                  return root;
               }
               if (root == p) {
                  flag = true;
               }
               root = root.right;
            }
         } else {
            if (flag) {
               return root;
            }
            if (root == p) {
               flag = true;
            }
            root = root.right;
         }
      }
      return null;

   }

   // 285
   // 面试题 04.06. 后继者 (Successor LCCI)
   // 剑指 Offer II 053. 二叉搜索树中的中序后继
   public TreeNode inorderSuccessor2(TreeNode root, TreeNode p) {
      TreeNode res = null;
      while (root != null) {
         if (root.val <= p.val) {
            root = root.right;
         } else {
            res = root;
            root = root.left;
         }
      }
      return res;
   }

   // 530. 二叉搜索树的最小绝对差 (Minimum Absolute Difference in BST) --莫里斯中序遍历
   // 783. 二叉搜索树节点最小距离 (Minimum Distance Between BST Nodes)
   public int getMinimumDifference(TreeNode root) {
      TreeNode pre = null;
      int preVal = -1;
      int res = Integer.MAX_VALUE;
      while (root != null) {
         if (root.left != null) {
            pre = root.left;
            while (pre.right != null && pre.right != root) {
               pre = pre.right;
            }
            if (pre.right == null) {
               pre.right = root;
               root = root.left;
            } else {
               pre.right = null;
               if (preVal != -1) {
                  res = Math.min(res, root.val - preVal);
               }
               preVal = root.val;
               root = root.right;
            }
         } else {
            if (preVal != -1) {
               res = Math.min(res, root.val - preVal);
            }
            preVal = root.val;
            root = root.right;
         }
      }
      return res;

   }

   // 653. 两数之和 IV - 输入 BST (Two Sum IV - Input is a BST)
   // 剑指 Offer II 056. 二叉搜索树中两个节点之和
   public boolean findTarget(TreeNode root, int k) {
      Set<Integer> set = new HashSet<>();
      TreeNode pre = null;
      while (root != null) {
         if (root.left != null) {
            pre = root.left;
            while (pre.right != null && pre.right != root) {
               pre = pre.right;
            }
            if (pre.right == null) {
               pre.right = root;
               root = root.left;
            } else {
               pre.right = null;
               int remain = k - root.val;
               if (set.contains(remain)) {
                  return true;
               }
               set.add(root.val);
               root = root.right;
            }
         } else {
            int remain = k - root.val;
            if (set.contains(remain)) {
               return true;
            }
            set.add(root.val);
            root = root.right;
         }
      }
      return false;
   }

   // 653. 两数之和 IV - 输入 BST (Two Sum IV - Input is a BST) --bfs
   // 剑指 Offer II 056. 二叉搜索树中两个节点之和
   public boolean findTarget2(TreeNode root, int k) {
      Set<Integer> set = new HashSet<>();
      Queue<TreeNode> queue = new LinkedList<>();
      queue.offer(root);
      while (!queue.isEmpty()) {
         TreeNode node = queue.poll();
         int remain = k - node.val;
         if (set.contains(remain)) {
            return true;
         }
         set.add(node.val);
         if (node.left != null) {
            queue.offer(node.left);
         }
         if (node.right != null) {
            queue.offer(node.right);
         }
      }
      return false;

   }

   public class Node {
      public int val;
      public Node left;
      public Node right;
      public Node next;

      public Node() {

      }

      public Node(int _val) {
         val = _val;
      }

      public Node(int _val, Node _left, Node _right) {
         val = _val;
         left = _left;
         right = _right;
      }

      public Node(int _val, Node _left, Node _right, Node _next) {
         val = _val;
         left = _left;
         right = _right;
         next = _next;
      }

   }

   // 剑指 Offer 36. 二叉搜索树与双向链表
   public Node treeToDoublyList(Node root) {
      Node dummy = new Node(-1);
      Node preNode = null;
      Node pre = null;
      while (root != null) {
         if (root.left != null) {
            pre = root.left;
            while (pre.right != null && pre.right != root) {
               pre = pre.right;
            }
            if (pre.right == null) {
               pre.right = root;
               root = root.left;
            } else {
               pre.right = null;
               if (preNode == null) {
                  dummy.right = root;
               } else {
                  preNode.right = root;
                  root.left = preNode;
               }
               preNode = root;
               root = root.right;
            }
         } else {
            if (preNode == null) {
               dummy.right = root;
            } else {
               preNode.right = root;
               root.left = preNode;
            }
            preNode = root;
            root = root.right;
         }
      }
      if (preNode != null) {
         preNode.right = dummy.right;
         dummy.right.left = preNode;
      }
      return dummy.right;

   }

   // 5984. 拆分数位后四位数字的最小和 (Minimum Sum of Four Digit Number After Splitting Digits)
   public int minimumSum(int num) {
      int[] arr = new int[4];
      int index = 0;
      while (num != 0) {
         arr[index++] = num % 10;
         num /= 10;
      }
      Arrays.sort(arr);
      return arr[0] * 10 + arr[3] + arr[1] * 10 + arr[2];

   }

   // 5985. 根据给定数字划分数组 (Partition Array According to Given Pivot)
   public int[] pivotArray(int[] nums, int pivot) {
      int index = 0;
      int actIndex = 0;
      int[] res = new int[nums.length];
      while (index < nums.length) {
         if (nums[index] < pivot) {
            res[actIndex++] = nums[index];
         }
         ++index;
      }
      index = 0;
      while (index < nums.length) {
         if (nums[index] == pivot) {
            res[actIndex++] = nums[index];
         }
         ++index;
      }
      index = 0;
      while (index < nums.length) {
         if (nums[index] > pivot) {
            res[actIndex++] = nums[index];
         }
         ++index;
      }
      return res;

   }

   // 2162. 设置时间的最少代价 (Minimum Cost to Set Cooking Time)
   public int minCostSetTime(int startAt, int moveCost, int pushCost, int targetSeconds) {
      int res = Integer.MAX_VALUE;
      int min = targetSeconds / 60;
      int sec = targetSeconds % 60;
      if (min >= 100) {
         --min;
         sec += 60;
      }
      res = Math.min(res, getMinTime2162(min, sec, startAt, moveCost, pushCost));
      if (min > 0 && sec <= 39) {
         --min;
         sec += 60;
         res = Math.min(res, getMinTime2162(min, sec, startAt, moveCost, pushCost));
      }
      return res;

   }

   private int getMinTime2162(int min, int sec, int startAt, int moveCost, int pushCost) {
      int res = 0;
      StringBuilder builder = new StringBuilder();
      if (min != 0) {
         builder.append(min);
      }
      if (min != 0 && sec < 10) {
         builder.append(0);
      }
      builder.append(sec);
      for (int i = 0; i < builder.length(); ++i) {
         if (i == 0) {
            if (startAt != builder.charAt(0) - '0') {
               res += moveCost;
            }
            res += pushCost;
         } else {
            if (builder.charAt(i - 1) != builder.charAt(i)) {
               res += moveCost;
            }
            res += pushCost;
         }
      }
      return res;

   }

   // 2164. 对奇偶下标分别排序 (Sort Even and Odd Indices Independently)
   public int[] sortEvenOdd(int[] nums) {
      List<Integer>[] list = new ArrayList[2];
      Arrays.setAll(list, k -> new ArrayList<>());
      for (int i = 0; i < nums.length; ++i) {
         list[i % 2].add(nums[i]);
      }
      Collections.sort(list[0]);
      Collections.sort(list[1], new Comparator<Integer>() {

         @Override
         public int compare(Integer o1, Integer o2) {
            return Integer.compare(o2, o1);
         }

      });
      int[] res = new int[nums.length];
      int j = 0;
      int[] p = new int[2];
      for (int i = 0; i < nums.length; ++i) {
         res[j++] = list[i % 2].get(p[i % 2]++);
      }
      return res;

   }

   // 2165. 重排数字的最小值 (Smallest Value of the Rearranged Number)
   public long smallestNumber(long num) {
      boolean isPositive = true;
      if (num < 0) {
         isPositive = false;
         num = -num;
      }

      long res = 0;
      char[] chars = String.valueOf(num).toCharArray();
      Arrays.sort(chars);
      int index = chars.length - 1;
      if (!isPositive) {
         while (index >= 0) {
            res = res * 10 + chars[index--] - '0';
         }
         return -res;
      }
      res = 0;
      index = 0;
      while (index < chars.length) {
         if (chars[index] != '0') {
            char temp = chars[index];
            chars[index] = chars[0];
            chars[0] = temp;
            break;
         }
         ++index;
      }
      return Long.parseLong(String.valueOf(chars));

   }

   // 6002. 设计位集 (Design Bitset)
   class Bitset {
      private int[] arr;
      private int oneCount;
      private int reverse;

      public Bitset(int size) {
         arr = new int[size];
      }

      public void fix(int idx) {
         if ((reverse ^ arr[idx]) == 0) {
            arr[idx] ^= 1;
            ++oneCount;
         }

      }

      public void unfix(int idx) {
         if ((reverse ^ arr[idx]) == 1) {
            arr[idx] ^= 1;
            --oneCount;
         }

      }

      public void flip() {
         reverse ^= 1;
         oneCount = arr.length - oneCount;
      }

      public boolean all() {
         return oneCount == arr.length;

      }

      public boolean one() {
         return oneCount > 0;

      }

      public int count() {
         return oneCount;

      }

      public String toString() {
         StringBuilder res = new StringBuilder();
         for (int num : arr) {
            res.append(num ^ reverse);
         }
         return res.toString();
      }
   }

   // 1382. 将二叉搜索树变平衡 (Balance a Binary Search Tree)
   public TreeNode balanceBST(TreeNode root) {
      List<Integer> a = new ArrayList<>();
      dfs1382(root, a);
      return makeTree1382(0, a.size() - 1, a);

   }

   private TreeNode makeTree1382(int i, int j, List<Integer> a) {
      if (i > j) {
         return null;
      }
      int mid = i + ((j - i) >> 1);
      TreeNode node = new TreeNode(a.get(mid));
      node.left = makeTree1382(i, mid - 1, a);
      node.right = makeTree1382(mid + 1, j, a);
      return node;
   }

   private void dfs1382(TreeNode root, List<Integer> a) {
      if (root == null) {
         return;
      }
      dfs1382(root.left, a);
      a.add(root.val);
      dfs1382(root.right, a);
   }

   // 面试题 04.03. List of Depth LCCI --bfs
   public ListNode[] listOfDepth(TreeNode tree) {
      List<ListNode> list = new ArrayList<>();
      Queue<TreeNode> queue = new LinkedList<>();
      queue.offer(tree);
      while (!queue.isEmpty()) {
         int size = queue.size();
         ListNode pre = null;
         ListNode head = null;
         for (int i = 0; i < size; ++i) {
            TreeNode treeNode = queue.poll();
            ListNode cur = new ListNode(treeNode.val);
            if (pre == null) {
               head = cur;
            } else {
               pre.next = cur;
            }
            pre = cur;
            if (treeNode.left != null) {
               queue.offer(treeNode.left);
            }
            if (treeNode.right != null) {
               queue.offer(treeNode.right);
            }
         }
         list.add(head);
      }
      ListNode[] res = new ListNode[list.size()];
      for (int i = 0; i < res.length; ++i) {
         res[i] = list.get(i);
      }
      return res;
   }

   // 2130. 链表最大孪生和 (Maximum Twin Sum of a Linked List)
   public int pairSum(ListNode head) {
      ListNode middle = getMiddle2130(head);
      ListNode reversed = getReversed2130(middle);
      int res = 0;
      while (reversed != null) {
         res = Math.max(res, reversed.val + head.val);
         head = head.next;
         reversed = reversed.next;
      }
      return res;

   }

   private ListNode getReversed2130(ListNode head) {
      ListNode pre = null;
      while (head != null) {
         ListNode temp = head.next;
         head.next = pre;
         pre = head;
         head = temp;
      }
      return pre;
   }

   private ListNode getMiddle2130(ListNode head) {
      ListNode slow = head;
      ListNode fast = head;
      while (fast != null && fast.next != null) {
         slow = slow.next;
         fast = fast.next.next;
      }
      return slow;
   }

   // 116. 填充每个节点的下一个右侧节点指针 (Populating Next Right Pointers in Each Node) --bfs
   // --还需掌握空间O(1)的解法
   public Node connect(Node root) {
      if (root == null) {
         return null;
      }
      Queue<Node> deque = new LinkedList<>();
      deque.offer(root);
      while (!deque.isEmpty()) {
         int size = deque.size();
         for (int i = 0; i < size; ++i) {
            Node node = deque.poll();
            if (i < size - 1) {
               node.next = deque.peek();
            }
            if (node.left != null) {
               deque.offer(node.left);
            }
            if (node.right != null) {
               deque.offer(node.right);
            }
         }
      }
      return root;

   }

   // 117. 填充每个节点的下一个右侧节点指针 II (Populating Next Right Pointers in Each Node II)
   // --bfs
   public Node connect117(Node root) {
      if (root == null) {
         return root;
      }
      Deque<Node> deque = new LinkedList<>();
      deque.offerLast(root);
      while (!deque.isEmpty()) {
         int size = deque.size();
         for (int i = 0; i < size; ++i) {
            Node cur = deque.pollFirst();
            if (i < size - 1) {
               cur.next = deque.peekFirst();
            }
            if (cur.left != null) {
               deque.offerLast(cur.left);
            }
            if (cur.right != null) {
               deque.offerLast(cur.right);
            }
         }
      }
      return root;
   }

   // 117. 填充每个节点的下一个右侧节点指针 II (Populating Next Right Pointers in Each Node II)
   // --dfs
   private List<Node> list117;

   public Node connect117_2(Node root) {
      this.list117 = new ArrayList<>();
      dfs117(root, 0);
      return root;
   }

   private void dfs117(Node root, int depth) {
      if (root == null) {
         return;
      }
      if (depth == list117.size()) {
         list117.add(root);
      } else {
         list117.get(depth).next = root;
         list117.set(depth, root);
      }
      dfs117(root.left, depth + 1);
      dfs117(root.right, depth + 1);
   }

   // 701. 二叉搜索树中的插入操作 (Insert into a Binary Search Tree)
   public TreeNode insertIntoBST(TreeNode root, int val) {
      TreeNode added = new TreeNode(val);
      TreeNode cur = root;
      while (cur != null) {
         if (cur.val < val) {
            if (cur.right != null) {
               cur = cur.right;
            } else {
               cur.right = added;
               return root;
            }
         } else {
            if (cur.left != null) {
               cur = cur.left;
            } else {
               cur.left = added;
               return root;
            }
         }
      }
      return added;

   }

   // 99. 恢复二叉搜索树 (Recover Binary Search Tree) --中序遍历 + 隐式栈
   private TreeNode x99;
   private TreeNode y99;
   private TreeNode pre99;

   public void recoverTree(TreeNode root) {
      TreeNode cur = root;
      dfs99(cur);
      swap99(x99, y99);

   }

   private void dfs99(TreeNode root) {
      if (root == null) {
         return;
      }
      dfs99(root.left);
      if (pre99 != null && pre99.val > root.val) {
         x99 = root;
         if (y99 == null) {
            y99 = pre99;
         }
      }
      pre99 = root;
      dfs99(root.right);
   }

   // 99. 恢复二叉搜索树 (Recover Binary Search Tree) --中序遍历 + 显式栈
   public void recoverTree2(TreeNode root) {
      Stack<TreeNode> stack = new Stack<>();
      TreeNode x = null;
      TreeNode y = null;
      TreeNode pre = null;
      TreeNode cur = root;
      while (cur != null || !stack.isEmpty()) {
         while (cur != null) {
            stack.push(cur);
            cur = cur.left;
         }
         if (!stack.isEmpty()) {
            cur = stack.pop();
            if (pre != null && pre.val > cur.val) {
               x = cur;
               if (y == null) {
                  y = pre;
               }
            }
            pre = cur;
            cur = cur.right;
         }
      }
      swap99(x, y);

   }

   // 99. 恢复二叉搜索树 (Recover Binary Search Tree) --中序遍历 + 莫里斯
   public void recoverTree3(TreeNode root) {
      TreeNode x = null;
      TreeNode y = null;
      TreeNode predecessor = null;
      TreeNode pre = null;
      TreeNode cur = root;
      while (cur != null) {
         if (cur.left != null) {
            predecessor = cur.left;
            while (predecessor.right != null && predecessor.right != cur) {
               predecessor = predecessor.right;
            }
            if (predecessor.right == null) {
               predecessor.right = cur;
               cur = cur.left;
            } else {
               predecessor.right = null;
               if (pre != null && pre.val > cur.val) {
                  x = cur;
                  if (y == null) {
                     y = pre;
                  }
               }
               pre = cur;
               cur = cur.right;
            }
         } else {
            if (pre != null && pre.val > cur.val) {
               x = cur;
               if (y == null) {
                  y = pre;
               }
            }
            pre = cur;
            cur = cur.right;
         }
      }
      swap99(x, y);
   }

   private void swap99(TreeNode x, TreeNode y) {
      int temp = x.val;
      x.val = y.val;
      y.val = temp;
   }

   // 669. 修剪二叉搜索树 (Trim a Binary Search Tree) --递归
   public TreeNode trimBST(TreeNode root, int low, int high) {
      if (root == null) {
         return null;
      }
      if (root.val > high) {
         return trimBST(root.left, low, high);
      }
      if (root.val < low) {
         return trimBST(root.right, low, high);
      }
      root.left = trimBST(root.left, low, high);
      root.right = trimBST(root.right, low, high);
      return root;

   }

   // 669. 修剪二叉搜索树 (Trim a Binary Search Tree) --迭代
   public TreeNode trimBST2(TreeNode root, int low, int high) {
      while (root != null && (root.val < low || root.val > high)) {
         if (root.val < low) {
            root = root.right;
         } else if (root.val > high) {
            root = root.left;
         }
      }
      if (root == null) {
         return null;
      }
      TreeNode node = root;
      while (node.left != null) {
         if (node.left.val < low) {
            node.left = node.left.right;
         } else {
            node = node.left;
         }
      }
      node = root;
      while (node.right != null) {
         if (node.right.val > high) {
            node.right = node.right.left;
         } else {
            node = node.right;
         }
      }
      return root;

   }

   // 703. 数据流中的第 K 大元素 (Kth Largest Element in a Stream)
   // 剑指 Offer II 059. 数据流的第 K 大数值
   class KthLargest {
      private PriorityQueue<Integer> priorityQueue;
      private int k;

      public KthLargest(int k, int[] nums) {
         this.k = k;
         this.priorityQueue = new PriorityQueue<>();
         for (int num : nums) {
            priorityQueue.offer(num);
         }

      }

      public int add(int val) {
         priorityQueue.offer(val);
         while (priorityQueue.size() > k) {
            priorityQueue.poll();
         }
         return priorityQueue.peek();
      }
   }

   // 450. 删除二叉搜索树中的节点 (Delete Node in a BST)
   public TreeNode deleteNode(TreeNode root, int key) {
      if (root == null) {
         return null;
      }
      if (root.val < key) {
         root.right = deleteNode(root.right, key);
      } else if (root.val > key) {
         root.left = deleteNode(root.left, key);
      } else {
         if (root.left == null) {
            root = root.right;
         } else if (root.right == null) {
            root = root.left;
         } else {
            TreeNode node = root.right;
            while (node.left != null) {
               node = node.left;
            }
            node.left = root.left;
            root = root.right;
         }
      }
      return root;

   }

   // 1405. 最长快乐字符串 (Longest Happy String)
   public String longestDiverseString(int a, int b, int c) {
      Bean1405[] arr = new Bean1405[3];
      arr[0] = new Bean1405('a', a);
      arr[1] = new Bean1405('b', b);
      arr[2] = new Bean1405('c', c);
      StringBuilder res = new StringBuilder();
      search: while (true) {
         Arrays.sort(arr, (o1, o2) -> o2.count - o1.count);
         if (res.length() >= 2) {
            for (Bean1405 bean : arr) {
               char last = res.charAt(res.length() - 1);
               char secondLast = res.charAt(res.length() - 2);
               if (bean.count > 0 && (last != bean.c || secondLast != bean.c)) {
                  res.append(bean.c);
                  --bean.count;
                  continue search;
               }
            }
            return res.toString();
         } else {
            res.append(arr[0].c);
            arr[0].count -= 1;
         }
      }

   }

   public class Bean1405 {
      public char c;
      public int count;

      public Bean1405(char c, int count) {
         this.c = c;
         this.count = count;
      }
   }

   // LCP 44. 开幕式焰火
   public int numColor(TreeNode root) {
      Set<Integer> set = new HashSet<>();
      TreeNode pre = null;
      while (root != null) {
         if (root.left != null) {
            pre = root.left;
            while (pre.right != null && pre.right != root) {
               pre = pre.right;
            }
            if (pre.right == null) {
               pre.right = root;
               root = root.left;
            } else {
               pre.right = null;
               set.add(root.val);
               root = root.right;
            }
         } else {
            set.add(root.val);
            root = root.right;
         }
      }
      return set.size();

   }

   // LCP 44. 开幕式焰火 --bfs
   public int numColor2(TreeNode root) {
      Set<Integer> set = new HashSet<>();
      Queue<TreeNode> queue = new LinkedList<>();
      queue.offer(root);
      while (!queue.isEmpty()) {
         TreeNode node = queue.poll();
         set.add(node.val);
         if (node.left != null) {
            queue.offer(node.left);
         }
         if (node.right != null) {
            queue.offer(node.right);
         }
      }
      return set.size();

   }

   // 101. 对称二叉树 (Symmetric Tree) --dfs
   // LCR 145. 判断对称二叉树
   public boolean isSymmetric(TreeNode root) {
      if (root == null) {
         return true;
      }
      return dfsOffer28(root.left, root.right);

   }

   private boolean dfsOffer28(TreeNode L, TreeNode R) {
      if (L == null && R == null) {
         return true;
      }
      if (L == null || R == null || L.val != R.val) {
         return false;
      }
      return dfsOffer28(L.left, R.right) && dfsOffer28(L.right, R.left);
   }

   // 101. 对称二叉树 (Symmetric Tree) --dfs 隐式栈
   // LCR 145. 判断对称二叉树
   public boolean isSymmetric2(TreeNode root) {
      Stack<TreeNode> stackLeft = new Stack<>();
      TreeNode rootLeft = root;
      Stack<TreeNode> stackRight = new Stack<>();
      TreeNode rootRight = root;
      while ((!stackLeft.isEmpty() || !stackRight.isEmpty()) || rootLeft != null || rootRight != null) {
         while (rootLeft != null || rootRight != null) {
            if (rootLeft == null || rootRight == null) {
               return false;
            }
            stackLeft.push(rootLeft);
            rootLeft = rootLeft.left;
            stackRight.push(rootRight);
            rootRight = rootRight.right;
         }
         rootLeft = stackLeft.pop();
         rootRight = stackRight.pop();
         if (rootLeft.val != rootRight.val) {
            return false;
         }
         rootLeft = rootLeft.right;
         rootRight = rootRight.left;
      }
      return true;

   }

   // 101. 对称二叉树 (Symmetric Tree) --bfs
   // LCR 145. 判断对称二叉树
   public boolean isSymmetric3(TreeNode root) {
      Queue<TreeNode> queue = new LinkedList<>();
      queue.offer(root);
      queue.offer(root);
      while (!queue.isEmpty()) {
         TreeNode node1 = queue.poll();
         TreeNode node2 = queue.poll();
         if (node1 == null && node2 == null) {
            continue;
         }
         if ((node1 == null || node2 == null) || (node1.val != node2.val)) {
            return false;
         }
         queue.offer(node1.left);
         queue.offer(node2.right);

         queue.offer(node1.right);
         queue.offer(node2.left);
      }
      return true;

   }

   // 1305. 两棵二叉搜索树中的所有元素 (All Elements in Two Binary Search Trees) --二叉树中序遍历 + 堆排序
   public List<Integer> getAllElements(TreeNode root1, TreeNode root2) {
      List<Integer> list1 = getInorderTraversal1305(root1);
      List<Integer> list2 = getInorderTraversal1305(root2);

      List<Integer> res = new ArrayList<>();
      int i = 0;
      int j = 0;
      while (i < list1.size() || j < list2.size()) {
         if (i < list1.size() && (j == list2.size() || list1.get(i) <= list2.get(j))) {
            res.add(list1.get(i++));
         } else {
            res.add(list2.get(j++));
         }
      }
      return res;

   }

   private List<Integer> getInorderTraversal1305(TreeNode root) {
      List<Integer> res = new ArrayList<>();
      TreeNode pre = null;
      while (root != null) {
         if (root.left != null) {
            pre = root.left;
            while (pre.right != null && pre.right != root) {
               pre = pre.right;
            }
            if (pre.right == null) {
               pre.right = root;
               root = root.left;
            } else {
               pre.right = null;
               res.add(root.val);
               root = root.right;
            }
         } else {
            res.add(root.val);
            root = root.right;
         }
      }
      return res;
   }

   // 剑指 Offer 33. 二叉搜索树的后序遍历序列
   public boolean verifyPostorder(int[] postorder) {
      return checkOffer33(postorder, 0, postorder.length - 1);
   }

   private boolean checkOffer33(int[] postorder, int i, int j) {
      if (i >= j) {
         return true;
      }
      int p = i;
      while (p < j && postorder[p] < postorder[j]) {
         ++p;
      }
      int m = p;
      while (p < j && postorder[p] > postorder[j]) {
         ++p;
      }
      return p == j && checkOffer33(postorder, i, m - 1) && checkOffer33(postorder, m, j - 1);

   }

   // 剑指 Offer 33. 二叉搜索树的后序遍历序列 --单调栈
   public boolean verifyPostorder2(int[] postorder) {
      int root = Integer.MAX_VALUE;
      Stack<Integer> stack = new Stack<>();
      for (int i = postorder.length - 1; i >= 0; --i) {
         if (postorder[i] > root) {
            return false;
         }
         while (!stack.isEmpty() && postorder[i] < stack.peek()) {
            root = stack.pop();
         }
         stack.push(postorder[i]);
      }
      return true;

   }

   // 1008. 前序遍历构造二叉搜索树 (Construct Binary Search Tree from Preorder Traversal)
   public TreeNode bstFromPreorder(int[] preorder) {
      return construct1008(preorder, 0, preorder.length - 1);
   }

   private TreeNode construct1008(int[] preorder, int left, int right) {
      if (left > right) {
         return null;
      }
      TreeNode root = new TreeNode(preorder[left]);
      if (left == right) {
         return root;
      }
      int index = left;
      while (index <= right) {
         if (preorder[index] > preorder[left]) {
            break;
         }
         ++index;
      }
      root.left = construct1008(preorder, left + 1, index - 1);
      root.right = construct1008(preorder, index, right);
      return root;
   }

   // 96. 不同的二叉搜索树 (Unique Binary Search Trees)
   public int numTrees(int n) {
      int[] G = new int[n + 1];
      G[0] = 1;
      G[1] = 1;
      for (int i = 2; i <= n; ++i) {
         for (int j = 1; j <= i; ++j) {
            G[i] += G[j - 1] * G[i - j];
         }
      }
      return G[n];

   }

   // 654. 最大二叉树 (Maximum Binary Tree) --递归 --还需掌握单调栈
   public TreeNode constructMaximumBinaryTree(int[] nums) {
      return construct654(nums, 0, nums.length - 1);

   }

   private TreeNode construct654(int[] nums, int left, int right) {
      if (left > right) {
         return null;
      }
      int index = -1;
      int max = -1;
      for (int i = left; i <= right; ++i) {
         if (nums[i] > max) {
            max = nums[i];
            index = i;
         }
      }
      TreeNode root = new TreeNode(nums[index]);
      root.left = construct654(nums, left, index - 1);
      root.right = construct654(nums, index + 1, right);
      return root;
   }

   // 95. 不同的二叉搜索树 II (Unique Binary Search Trees II) --递归、回溯
   public List<TreeNode> generateTrees(int n) {
      return getTrees(1, n);

   }

   private List<TreeNode> getTrees(int left, int right) {
      List<TreeNode> res = new ArrayList<>();
      if (left > right) {
         res.add(null);
         return res;
      }
      for (int i = left; i <= right; ++i) {
         List<TreeNode> leftTrees = getTrees(left, i - 1);
         List<TreeNode> rightTrees = getTrees(i + 1, right);
         for (TreeNode leftTree : leftTrees) {
            for (TreeNode rightTree : rightTrees) {
               TreeNode root = new TreeNode(i);
               root.left = leftTree;
               root.right = rightTree;
               res.add(root);
            }
         }
      }
      return res;
   }

   // 100. 相同的树 (Same Tree) --dfs
   public boolean isSameTree(TreeNode p, TreeNode q) {
      if (p == null && q == null) {
         return true;
      }
      if (p == null || q == null || p.val != q.val) {
         return false;
      }
      return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
   }

   // 100. 相同的树 (Same Tree) --bfs
   public boolean isSameTree2(TreeNode p, TreeNode q) {
      if (p == null && q == null) {
         return true;
      }
      if (p == null || q == null) {
         return false;
      }
      Queue<TreeNode> queue = new LinkedList<>();
      queue.offer(p);
      queue.offer(q);
      while (!queue.isEmpty()) {
         TreeNode nodeP = queue.poll();
         TreeNode nodeQ = queue.poll();
         if (nodeP.val != nodeQ.val) {
            return false;
         }
         if ((nodeP.left == null && nodeQ.left != null) || (nodeP.left != null && nodeQ.left == null)) {
            return false;
         }
         if (nodeP.left != null && nodeQ.left != null) {
            queue.offer(nodeP.left);
            queue.offer(nodeQ.left);
         }

         if ((nodeP.right == null && nodeQ.right != null) || (nodeP.right != null && nodeQ.right == null)) {
            return false;
         }
         if (nodeP.right != null && nodeQ.right != null) {
            queue.offer(nodeP.right);
            queue.offer(nodeQ.right);
         }
      }
      return true;

   }

   // 102. 二叉树的层序遍历 (Binary Tree Level Order Traversal) --bfs
   // 剑指 Offer 32 - II. 从上到下打印二叉树 II
   public List<List<Integer>> levelOrder(TreeNode root) {
      List<List<Integer>> res = new ArrayList<>();
      if (root == null) {
         return res;
      }
      Queue<TreeNode> queue = new LinkedList<>();
      queue.offer(root);
      while (!queue.isEmpty()) {
         int size = queue.size();
         List<Integer> list = new ArrayList<>();
         for (int i = 0; i < size; ++i) {
            TreeNode node = queue.poll();
            list.add(node.val);
            if (node.left != null) {
               queue.offer(node.left);
            }
            if (node.right != null) {
               queue.offer(node.right);
            }
         }
         res.add(list);
      }
      return res;

   }

   // 102. 二叉树的层序遍历 (Binary Tree Level Order Traversal) --dfs
   private List<List<Integer>> res102;

   public List<List<Integer>> levelOrder2(TreeNode root) {
      this.res102 = new ArrayList<>();
      dfs102(root, 0);
      return res102;

   }

   private void dfs102(TreeNode root, int d) {
      if (root == null) {
         return;
      }
      if (res102.size() == d) {
         res102.add(new ArrayList<>());
      }
      res102.get(d).add(root.val);
      dfs102(root.left, d + 1);
      dfs102(root.right, d + 1);
   }

   // 404. 左叶子之和 (Sum of Left Leaves) --bfs
   public int sumOfLeftLeaves(TreeNode root) {
      int res = 0;
      Queue<TreeNode> queue = new LinkedList<>();
      queue.offer(root);
      while (!queue.isEmpty()) {
         TreeNode node = queue.poll();
         if (node.left != null) {
            if (isLeafNode(node.left)) {
               res += node.left.val;
            }
            queue.offer(node.left);
         }
         if (node.right != null) {
            queue.offer(node.right);
         }
      }
      return res;
   }

   private boolean isLeafNode(TreeNode node) {
      return node != null && node.left == null && node.right == null;
   }

   // 404. 左叶子之和 (Sum of Left Leaves) --dfs
   public int sumOfLeftLeaves2(TreeNode root) {
      if (root == null) {
         return 0;
      }
      return sumOfLeftLeaves2(root.left) + sumOfLeftLeaves2(root.right) + (isLeafNode(root.left) ? root.left.val : 0);
   }

   // 226. 翻转二叉树 (Invert Binary Tree) --dfs
   // 剑指 Offer 27. 二叉树的镜像 --dfs
   public TreeNode invertTree(TreeNode root) {
      if (root == null) {
         return null;
      }
      TreeNode left = invertTree(root.left);
      TreeNode right = invertTree(root.right);
      root.left = right;
      root.right = left;
      return root;
   }

   // 226. 翻转二叉树 (Invert Binary Tree) --bfs
   // 剑指 Offer 27. 二叉树的镜像 --bfs
   public TreeNode invertTree2(TreeNode root) {
      if (root == null) {
         return root;
      }
      Queue<TreeNode> queue = new LinkedList<>();
      queue.offer(root);
      while (!queue.isEmpty()) {
         TreeNode cur = queue.poll();
         TreeNode temp = cur.left;
         cur.left = cur.right;
         cur.right = temp;
         if (cur.left != null) {
            queue.offer(cur.left);
         }
         if (cur.right != null) {
            queue.offer(cur.right);
         }
      }
      return root;

   }

   // 107. 二叉树的层序遍历 II (Binary Tree Level Order Traversal II) --bfs
   public List<List<Integer>> levelOrderBottom(TreeNode root) {
      List<List<Integer>> res = new LinkedList<>();
      if (root == null) {
         return res;
      }
      Queue<TreeNode> queue = new LinkedList<>();
      queue.offer(root);
      while (!queue.isEmpty()) {
         int size = queue.size();
         List<Integer> list = new LinkedList<>();
         for (int i = 0; i < size; ++i) {
            TreeNode node = queue.poll();
            list.add(node.val);
            if (node.left != null) {
               queue.offer(node.left);
            }
            if (node.right != null) {
               queue.offer(node.right);
            }
         }
         res.add(0, list);
      }
      return res;

   }

}