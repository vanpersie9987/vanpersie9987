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
         return prefixMap.getOrDefault(prefix, 0);
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

   // 2022. 将一维数组转变成二维数组 (Convert 1D Array Into 2D Array)
   public int[][] construct2DArray(int[] original, int m, int n) {
      if ((m * n) != original.length) {
         return new int[0][];
      }
      int[][] res = new int[m][n];
      int index = 0;
      for (int i = 0; i < m; ++i) {
         for (int j = 0; j < n; ++j) {
            res[i][j] = original[index++];
         }
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
   public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
      ListNode dummy = new ListNode(0);
      ListNode cur = dummy;
      while (l1 != null && l2 != null) {
         if (l1.val < l2.val) {
            cur.next = l1;
            l1 = l1.next;
         } else {
            cur.next = l2;
            l2 = l2.next;
         }
         cur = cur.next;
      }
      if (l1 != null) {
         cur.next = l1;
      }
      if (l2 != null) {
         cur.next = l2;
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
      while (cur.next != null && cur.next.next != null) {
         if (cur.next.val == cur.next.next.val) {
            int x = cur.next.val;
            while (cur.next != null && cur.next.val == x) {
               cur.next = cur.next.next;
            }
         } else {
            cur = cur.next;
         }
      }
      return dummy.next;

   }

   // 83. 删除排序链表中的重复元素 (Remove Duplicates from Sorted List)
   public ListNode deleteDuplicates83(ListNode head) {
      ListNode cur = head;
      while (cur != null && cur.next != null) {
         if (cur.val == cur.next.val) {
            cur.next = cur.next.next;
         } else {
            cur = cur.next;
         }
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
      ListNode l1 = head;
      ListNode middle = getMiddle(head);
      ListNode l2 = reverseNode(middle.next);
      middle.next = null;
      getMixed(l1, l2);

   }

   private void getMixed(ListNode l1, ListNode l2) {
      while (l1 != null && l2 != null) {
         ListNode temp1 = l1.next;
         ListNode temp2 = l2.next;

         l1.next = l2;
         l1 = temp1;

         l2.next = l1;
         l2 = temp2;
      }

   }

   private ListNode reverseNode(ListNode head) {
      ListNode pre = null;
      ListNode cur = head;
      while (cur != null) {
         ListNode temp = cur.next;
         cur.next = pre;

         pre = cur;
         cur = temp;

      }
      return pre;

   }

   private ListNode getMiddle(ListNode head) {
      ListNode slow = head;
      ListNode fast = head;
      while (fast != null && fast.next != null) {
         slow = slow.next;
         fast = fast.next.next;
      }
      return slow;

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

   // 148. 排序链表 (Sort List) -- 还需掌握空间复杂度为O(1)的归并排序、分治算法
   // 剑指 Offer II 077. 链表排序
   public ListNode sortList(ListNode head) {
      List<ListNode> list = new ArrayList<>();
      ListNode cur = head;
      while (cur != null) {
         list.add(cur);
         cur = cur.next;
      }
      Collections.sort(list, (o1, o2) -> o1.val - o2.val);

      ListNode dummy = new ListNode(0);
      cur = dummy;
      for (int i = 0; i < list.size(); ++i) {
         ListNode added = list.get(i);
         cur.next = added;
         cur = cur.next;
      }
      cur.next = null;
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
      l1 = reverseLinkedList(l1);
      l2 = reverseLinkedList(l2);
      ListNode res = add445(l1, l2);
      return reverseLinkedList(res);

   }

   private ListNode add445(ListNode l1, ListNode l2) {
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

   private ListNode reverseLinkedList(ListNode head) {
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
      Set<Integer> set = new HashSet<>();
      for (int num : nums) {
         set.add(num);
      }
      int res = 0;
      while (head != null) {
         while (head != null && !set.contains(head.val)) {
            head = head.next;
         }
         if (head == null) {
            break;
         }
         while (head != null && set.contains(head.val)) {
            head = head.next;
         }
         ++res;
      }
      return res;
   }

   // 707. 设计链表 (Design Linked List) --单链表
   class MyLinkedList {
      class Node {
         int val;
         Node next;

         Node() {

         }

         Node(int val, Node next) {
            this.val = val;
            this.next = next;
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
         Node cur = head;
         for (int i = 0; i < index; ++i) {
            cur = cur.next;
         }
         Node added = new Node(val, cur.next);
         cur.next = added;
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
         Node prev;
         Node next;
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
         // 从后往前
         if (2 * index >= size) {
            Node cur = tail;
            for (int i = 0; i < size - index; ++i) {
               cur = cur.prev;
            }
            return cur.val;
         } else {
            Node cur = head;
            for (int i = 0; i <= index; ++i) {
               cur = cur.next;
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
         if (2 * index >= size) {
            Node cur = tail;
            for (int i = 0; i < size - index; ++i) {
               cur = cur.prev;
            }
            Node added = new Node(val);
            added.prev = cur.prev;
            cur.prev = added;
            added.prev.next = added;
            added.next = cur;
         } else {
            Node cur = head;
            for (int i = 0; i < index; ++i) {
               cur = cur.next;
            }
            Node added = new Node(val);
            added.next = cur.next;
            cur.next = added;
            added.next.prev = added;
            added.prev = cur;
         }
         ++size;

      }

      public void deleteAtIndex(int index) {
         if (index < 0 || index >= size) {
            return;
         }
         if (2 * index >= size) {
            Node cur = tail;
            for (int i = 0; i < size - index - 1; ++i) {
               cur = cur.prev;
            }
            cur.prev = cur.prev.prev;
            cur.prev.next = cur;
         } else {
            Node cur = head;
            for (int i = 0; i < index; ++i) {
               cur = cur.next;
            }
            cur.next = cur.next.next;
            cur.next.prev = cur;
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
      ListNode dummy = new ListNode(0, list1);
      ListNode cur = dummy;
      for (int i = 0; i < a; ++i) {
         cur = cur.next;
      }
      ListNode node1 = cur;
      int diff = b - a + 1;
      for (int i = 0; i < diff; ++i) {
         cur = cur.next;
      }
      node1.next = list2;
      ListNode head2 = list2;
      while (head2.next != null) {
         head2 = head2.next;
      }
      head2.next = cur.next;
      return dummy.next;

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

   // 238. 除自身以外数组的乘积 (Product of Array Except Self) --前缀积
   // 剑指 Offer 66. 构建乘积数组
   public int[] constructArr(int[] a) {
      int k = 1;
      int[] res = new int[a.length];
      for (int i = 0; i < a.length; ++i) {
         res[i] = k;
         k *= a[i];
      }
      k = 1;
      for (int i = a.length - 1; i >= 0; --i) {
         res[i] *= k;
         k *= a[i];
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
      int preSum = 0;
      int res = 0;
      for (int i = 0; i < nums.length; ++i) {
         preSum += nums[i];
         res += map.getOrDefault(preSum - k, 0);
         map.put(preSum, map.getOrDefault(preSum, 0) + 1);
      }
      return res;

   }

   // 724. 寻找数组的中心下标 (Find Pivot Index) --前缀和
   // 1991. 找到数组的中间位置 (Find the Middle Index in Array)
   // 剑指 Offer II 012. 左右两边子数组的和相等
   public int pivotIndex(int[] nums) {
      int leftSum = 0;
      int sum = 0;
      for (int i = 0; i < nums.length; ++i) {
         sum += nums[i];
      }
      for (int i = 0; i < nums.length; ++i) {
         if (leftSum == sum - nums[i] - leftSum) {
            return i;
         }
         leftSum += nums[i];
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

   // 1004.最大连续1的个数 III (Max Consecutive Ones III) --双指针 + 前缀和 + 滑动窗口
   public int longestOnes(int[] nums, int k) {
      int zeroCount = 0;
      int left = 0;
      int right = 0;
      int res = 0;
      while (right < nums.length) {
         if (nums[right] == 0) {
            ++zeroCount;
         }
         while (zeroCount > k) {
            if (nums[left++] == 0) {
               --zeroCount;
            }
         }
         res = Math.max(res, right - left + 1);
         ++right;
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
      StringBuilder numberBuilder = new StringBuilder();
      for (char c : number.toCharArray()) {
         if (Character.isDigit(c)) {
            numberBuilder.append(c);
         }
      }
      StringBuilder res = new StringBuilder();
      if (numberBuilder.length() % 3 == 1) {
         int i = 0;
         while (i < numberBuilder.length() - 4) {
            res.append(numberBuilder.substring(i, i + 3)).append('-');
            i += 3;
         }
         while (i < numberBuilder.length() - 2) {
            res.append(numberBuilder.substring(i, i + 2)).append('-');
            i += 2;
         }
         res.append(numberBuilder.substring(i));
      } else if (numberBuilder.length() % 3 == 2) {
         int i = 0;
         while (i < numberBuilder.length() - 2) {
            res.append(numberBuilder.substring(i, i + 3)).append('-');
            i += 3;
         }
         res.append(numberBuilder.substring(i));
      } else {
         int i = 0;
         while (i < numberBuilder.length() - 3) {
            res.append(numberBuilder.substring(i, i + 3)).append('-');
            i += 3;
         }
         res.append(numberBuilder.substring(i));
      }
      return res.toString();

   }

   // 944. 删列造序 (Delete Columns to Make Sorted)
   public int minDeletionSize(String[] strs) {
      int res = 0;
      int n = strs[0].length();
      for (int i = 0; i < n; ++i) {
         res += judge944(strs, i);
      }
      return res;

   }

   private int judge944(String[] strs, int pos) {
      char pre = 'a';
      for (String str : strs) {
         if (str.charAt(pos) < pre) {
            return 1;
         }
         pre = str.charAt(pos);
      }
      return 0;
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

   // 1292. 元素和小于等于阈值的正方形的最大边长 --二分查找
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

   private int getRect(int[][] preSum, int startI, int startJ, int endI, int endJ) {
      return preSum[endI][endJ] - preSum[endI][startJ - 1] - preSum[startI - 1][endJ] + preSum[startI - 1][startJ - 1];
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
               int curSum = preSum[i + k][j + k] - preSum[i + k][j] - preSum[i][j + k]
                     + preSum[i][j];
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
      final int MOD = 1000000007;
      int odd = 0;
      int even = 1;
      int preSum = 0;
      int res = 0;
      for (int i = 0; i < arr.length; ++i) {
         preSum += arr[i];
         if (preSum % 2 == 1) {
            res = (res + even) % MOD;
            ++odd;
         } else {
            res = (res + odd) % MOD;
            ++even;
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
      int lines = 1;
      int cur = 0;
      for (char c : s.toCharArray()) {
         if (cur + widths[c - 'a'] <= 100) {
            cur += widths[c - 'a'];
         } else {
            cur = widths[c - 'a'];
            ++lines;
         }
      }
      return new int[] { lines, cur };

   }

   // 2042. 检查句子中的数字是否递增 (Check if Numbers Are Ascending in a Sentence)
   public boolean areNumbersAscending(String s) {
      String preNum = "0";
      char[] chars = s.toCharArray();
      for (int i = 0; i < chars.length; ++i) {
         if (Character.isDigit(chars[i])) {
            StringBuilder num = new StringBuilder();
            num.append(chars[i]);
            while (i + 1 < chars.length && Character.isDigit(chars[i + 1])) {
               num.append(chars[i + 1]);
               ++i;
            }
            if (num.length() < preNum.length()
                  || (num.length() == preNum.length() && num.toString().compareTo(preNum) <= 0)) {
               return false;
            }
            preNum = num.toString();
         }
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
      Set<Integer> set = new HashSet<>();
      for (int i = left; i <= right; ++i) {
         set.add(i);
      }
      for (int[] range : ranges) {
         if (range[1] < left || range[0] > right) {
            continue;
         }
         int min = Math.max(Math.min(left, range[1]), range[0]);
         int max = Math.min(Math.max(right, range[0]), range[1]);
         for (int i = min; i <= max; ++i) {
            set.remove(i);
         }
      }
      return set.isEmpty();
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

   // 2024. 考试的最大困扰度 (Maximize the Confusion of an Exam) --前缀和 + 滑动窗口
   public int maxConsecutiveAnswers(String answerKey, int k) {
      int tCount = 0;
      int fCount = 0;
      int res = 0;
      int left = 0;
      int right = 0;
      char[] chars = answerKey.toCharArray();
      while (right < chars.length) {
         if (chars[right] == 'T') {
            ++tCount;
         } else {
            ++fCount;
         }

         while (tCount > k && fCount > k) {
            if (chars[left++] == 'T') {
               --tCount;
            } else {
               --fCount;
            }
         }
         res = Math.max(res, right - left + 1);
         ++right;
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
         counts1.put(word, counts1.getOrDefault(word, 0) + 1);
      }
      Map<String, Integer> counts2 = new HashMap<>();
      for (String word : words2) {
         counts2.put(word, counts2.getOrDefault(word, 0) + 1);
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
      int mod = 0;
      for (int num : nums) {
         mod = (mod + num) % p;
      }
      if (mod == 0) {
         return 0;
      }
      Map<Integer, Integer> map = new HashMap<>();
      map.put(0, -1);
      int preSum = 0;
      int res = nums.length;
      for (int i = 0; i < nums.length; ++i) {
         preSum = (preSum + nums[i]) % p;
         int target = (preSum - mod + p) % p;
         if (map.containsKey(target)) {
            res = Math.min(res, i - map.get(target));
         }
         map.put(preSum, i);
      }
      return res == nums.length ? -1 : res;

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
      Set<String> set = new HashSet<>();
      char[] chars = word.toCharArray();
      StringBuilder builder = new StringBuilder();
      for (int i = 0; i < chars.length; ++i) {
         if (Character.isDigit(chars[i])) {
            while (i < chars.length && chars[i] == '0') {
               ++i;
            }
            while (i < chars.length && Character.isDigit(chars[i])) {
               builder.append(chars[i]);
               ++i;
            }
            if (builder.length() == 0) {
               set.add("0");
            } else {
               set.add(builder.toString());
            }
            builder.setLength(0);
         }
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

   // 3. 无重复字符的最长子串 (Longest Substring Without Repeating Characters) --滑动窗口
   // 剑指 Offer 48. 最长不含重复字符的子字符串
   // 剑指 Offer II 016. 不含重复字符的最长子字符串
   public int lengthOfLongestSubstring(String s) {
      Set<Character> set = new HashSet<>();
      int res = 0;
      char[] chars = s.toCharArray();
      int left = 0;
      int right = 0;
      while (right < chars.length) {
         if (set.add(chars[right])) {
            res = Math.max(res, right - left + 1);
            ++right;
         } else {
            set.remove(chars[left++]);
         }
      }
      return res;
   }

   // 76. 最小覆盖子串 (Minimum Window Substring) --滑动窗口
   // 剑指 Offer II 017. 含有所有字符的最短字符串
   public String minWindow(String s, String t) {
      if (t.length() > s.length()) {
         return "";
      }
      int need = 0;
      Map<Character, Integer> needMap = new HashMap<>();
      for (char c : t.toCharArray()) {
         needMap.put(c, needMap.getOrDefault(c, 0) + 1);
         ++need;
      }
      Map<Character, Integer> giveMap = new HashMap<>();
      int give = 0;
      int left = 0;
      int right = 0;
      int minLength = Integer.MAX_VALUE;
      String res = "";
      char[] chars = s.toCharArray();
      while (right < chars.length) {
         if (needMap.containsKey(chars[right])) {
            int needCount = needMap.get(chars[right]);
            int giveCount = giveMap.getOrDefault(chars[right], 0);
            if (giveCount < needCount) {
               ++give;
            }
            giveMap.put(chars[right], giveMap.getOrDefault(chars[right], 0) + 1);
            while (give == need) {
               if (right - left + 1 < minLength) {
                  res = s.substring(left, right + 1);
                  minLength = right - left + 1;
               }
               if (giveMap.containsKey(chars[left])) {
                  giveCount = giveMap.get(chars[left]);
                  needCount = needMap.get(chars[left]);
                  if (giveCount == needCount) {
                     --give;
                  }
                  giveMap.put(chars[left], giveMap.get(chars[left]) - 1);
               }
               ++left;
            }
         }
         ++right;
      }
      return minLength == Integer.MAX_VALUE ? "" : res;

   }

   // 76. 最小覆盖子串 (Minimum Window Substring) --滑动窗口
   // 剑指 Offer II 017. 含有所有字符的最短字符串
   public String minWindow2(String s, String t) {
      int[] need = new int[128];
      for (char c : t.toCharArray()) {
         ++need[c];
      }
      int needCount = t.length();
      int[] give = new int[128];
      int giveCount = 0;
      int left = 0;
      int right = 0;
      int minLength = s.length() + 1;
      String res = "";
      while (right < s.length()) {
         char c = s.charAt(right);
         if (need[c] > 0) {
            if (give[c] < need[c]) {
               ++giveCount;
            }
            ++give[c];
         }
         while (giveCount == needCount) {
            if (right - left + 1 < minLength) {
               minLength = right - left + 1;
               res = s.substring(left, left + minLength);
            }
            c = s.charAt(left);
            if (need[c] > 0) {
               if (give[c] == need[c]) {
                  --giveCount;
               }
               --give[c];
            }
            ++left;
         }
         ++right;
      }
      return res;

   }

   // 209. 长度最小的子数组 (Minimum Size Subarray Sum) --O(n) 滑动窗口
   // 剑指 Offer II 008. 和大于等于 target 的最短子数组 --O(n)
   public int minSubArrayLen(int target, int[] nums) {
      int left = 0;
      int right = 0;
      int preSum = 0;
      int res = Integer.MAX_VALUE;
      while (right < nums.length) {
         preSum += nums[right];
         while (preSum >= target) {
            res = Math.min(res, right - left + 1);
            preSum -= nums[left++];
         }
         ++right;
      }
      return res == Integer.MAX_VALUE ? 0 : res;

   }

   // 209. 长度最小的子数组 (Minimum Size Subarray Sum) --O(nlog(n)) 还需要掌握二分查找法
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

   // 395. 至少有 K 个重复字符的最长子串
   // 395. Longest Substring with At Least K Repeating Characters --分治 还需掌握滑动窗口
   public int longestSubstring(String s, int k) {
      if (s.length() < k) {
         return 0;
      }
      Map<Character, Integer> map = new HashMap<>();
      for (char c : s.toCharArray()) {
         map.put(c, map.getOrDefault(c, 0) + 1);
      }
      for (Map.Entry<Character, Integer> entry : map.entrySet()) {
         if (entry.getValue() < k) {
            int max = 0;
            for (String sub : s.split(String.valueOf(entry.getKey()))) {
               max = Math.max(max, longestSubstring(sub, k));
            }
            return max;
         }
      }
      return s.length();

   }

   // 424. 替换后的最长重复字符 (Longest Repeating Character Replacement)
   public int characterReplacement(String s, int k) {
      int[] counts = new int[26];
      int left = 0;
      int right = 0;
      int max = 0;
      char[] chars = s.toCharArray();
      int res = 0;
      while (right < chars.length) {
         max = Math.max(max, ++counts[chars[right] - 'A']);
         while (right - left + 1 - max > k) {
            --counts[chars[left++] - 'A'];
         }
         res = Math.max(res, right - left + 1);
         ++right;
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

   // 567. 字符串的排列 (Permutation in String)
   // 剑指 Offer II 014. 字符串中的变位词 --滑动窗口
   public boolean checkInclusion(String s1, String s2) {
      if (s1.length() > s2.length()) {
         return false;
      }
      int[] needs = new int[26];
      for (char c : s1.toCharArray()) {
         ++needs[c - 'a'];
      }
      int count = s1.length();
      int i = 0;
      int[] give = new int[26];
      char[] chars = s2.toCharArray();
      while (i < count) {
         ++give[chars[i++] - 'a'];
      }
      if (Arrays.equals(needs, give)) {
         return true;
      }
      while (i < chars.length) {
         --give[chars[i - count] - 'a'];
         ++give[chars[i] - 'a'];
         if (Arrays.equals(needs, give)) {
            return true;
         }
         ++i;
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
      int max = Integer.MIN_VALUE;
      int i = 0;
      int sum = 0;
      while (i < k) {
         sum += nums[i++];
      }
      max = Math.max(max, sum);
      while (i < nums.length) {
         sum -= nums[i - k];
         sum += nums[i];
         max = Math.max(max, sum);
         ++i;
      }
      return (double) max / k;

   }

   // 713. 乘积小于K的子数组 (Subarray Product Less Than K)
   // 剑指 Offer II 009. 乘积小于 K 的子数组 --双指针+滑动窗口
   public int numSubarrayProductLessThanK(int[] nums, int k) {
      int product = 1;
      int left = 0;
      int right = 0;
      int res = 0;
      while (right < nums.length) {
         product *= nums[right];
         while (left <= right && product >= k) {
            product /= nums[left++];
         }
         res += right - left + 1;
         ++right;
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
      int i = digitCount >= alphabetCount ? 0 : 1;
      int j = i == 0 ? 1 : 0;
      for (char c : s.toCharArray()) {
         if (Character.isDigit(c)) {
            res[i] = c;
            i += 2;
         } else {
            res[j] = c;
            j += 2;
         }
      }
      return String.valueOf(res);

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
   // --滑动窗口 还需掌握 “动态规划”、“二分查找 + 哈希”
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
      return Math.max(getMax1031(nums, firstLen, secondLen), getMax1031(nums, secondLen, firstLen));

   }

   private int getMax1031(int[] nums, int leftLen, int rightLen) {
      int[] prefix1 = new int[nums.length];
      int max = 0;
      int cur = 0;
      int i = 0;
      while (i < leftLen) {
         cur += nums[i++];
      }
      max = Math.max(max, cur);
      prefix1[i - 1] = max;
      while (i < nums.length) {
         cur -= nums[i - leftLen];
         cur += nums[i];
         max = Math.max(max, cur);
         prefix1[i] = max;
         ++i;
      }

      int[] prefix2 = new int[nums.length];
      max = 0;
      cur = 0;
      i = nums.length - 1;
      while (i >= nums.length - rightLen) {
         cur += nums[i--];
      }
      max = Math.max(max, cur);
      prefix2[i + 1] = cur;
      while (i >= 0) {
         cur -= nums[i + rightLen];
         cur += nums[i];
         max = Math.max(max, cur);
         prefix2[i] = max;
         --i;
      }
      int res = 0;
      for (int j = leftLen; j <= nums.length - rightLen; ++j) {
         res = Math.max(res, prefix1[j - 1] + prefix2[j]);

      }
      return res;
   }

   // 1052. 爱生气的书店老板 (Grumpy Bookstore Owner)
   public int maxSatisfied(int[] customers, int[] grumpy, int minutes) {
      int satisfy = 0;
      for (int i = 0; i < customers.length; ++i) {
         if (grumpy[i] == 0) {
            satisfy += customers[i];
         }
      }
      int index = 0;
      int cur = 0;
      int res = 0;
      while (index < minutes) {
         if (grumpy[index] == 1) {
            cur += customers[index];
         }
         ++index;
      }
      res = cur;
      while (index < customers.length) {
         if (grumpy[index] == 1) {
            cur += customers[index];
         }
         if (grumpy[index - minutes] == 1) {
            cur -= customers[index - minutes];
         }
         res = Math.max(res, cur);
         ++index;
      }
      return res + satisfy;

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

   // 1343. 大小为 K 且平均值大于等于阈值的子数组数目
   // 1343. Number of Sub-arrays of Size K and Average Greater than or Equal to
   // Threshold
   public int numOfSubarrays(int[] arr, int k, int threshold) {
      int sum = k * threshold;
      int cur = 0;
      for (int i = 0; i < k; ++i) {
         cur += arr[i];
      }
      int res = 0;
      if (cur >= sum) {
         ++res;
      }
      for (int i = k; i < arr.length; ++i) {
         cur -= arr[i - k];
         cur += arr[i];
         if (cur >= sum) {
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
      int window = cardPoints.length - k;
      int cur = 0;
      int min = Integer.MAX_VALUE;
      int sum = 0;
      for (int i = 0; i < window; ++i) {
         cur += cardPoints[i];
         sum += cardPoints[i];
      }
      min = Math.min(min, cur);
      for (int i = window; i < cardPoints.length; ++i) {
         sum += cardPoints[i];
         cur -= cardPoints[i - window];
         cur += cardPoints[i];
         min = Math.min(min, cur);
      }
      return sum - min;

   }

   // 1456. 定长子串中元音的最大数目 (Maximum Number of Vowels in a Substring of Given Length)
   // --滑动窗口
   public int maxVowels(String s, int k) {
      int res = 0;
      int cur = 0;
      char[] chars = s.toCharArray();
      for (int i = 0; i < k; ++i) {
         char c = chars[i];
         if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
            ++cur;
         }
      }
      res = Math.max(res, cur);
      for (int i = k; i < chars.length; ++i) {
         char c = chars[i];
         if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
            ++cur;
         }
         c = chars[i - k];
         if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
            --cur;
         }
         res = Math.max(res, cur);
      }
      return res;

   }

   // 1493. 删掉一个元素以后全为 1 的最长子数组 (Longest Subarray of 1's After Deleting One
   // Element) -- 滑动窗口 还需掌握动态规划
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

   // 1658. 将 x 减到 0 的最小操作数 (Minimum Operations to Reduce X to Zero)
   // --滑动窗口 + 双指针 + 前缀和
   public int minOperations(int[] nums, int x) {
      int target = Arrays.stream(nums).sum() - x;
      int left = 0;
      int right = 0;
      int maxLen = -1;
      int prefix = 0;
      while (right < nums.length) {
         prefix += nums[right];
         while (left <= right && prefix > target) {
            prefix -= nums[left++];
         }
         if (prefix == target) {
            maxLen = Math.max(maxLen, right - left + 1);
         }
         ++right;
      }
      return maxLen == -1 ? -1 : nums.length - maxLen;

   }

   // 1695. 删除子数组的最大得分 (Maximum Erasure Value)
   public int maximumUniqueSubarray(int[] nums) {
      Set<Integer> set = new HashSet<>();
      int left = 0;
      int right = 0;
      int prefix = 0;
      int max = 0;
      while (right < nums.length) {
         if (set.add(nums[right])) {
            prefix += nums[right];
            max = Math.max(max, prefix);
            ++right;
         } else {
            prefix -= nums[left];
            set.remove(nums[left]);
            ++left;
         }
      }
      return max;

   }

   // 1876.长度为三且各字符不同的子字符串 (Substrings of Size Three with Distinct Characters)
   public int countGoodSubstrings(String s) {
      int i = 2;
      int res = 0;
      while (i < s.length()) {
         if (s.charAt(i - 2) != s.charAt(i - 1) && s.charAt(i - 1) != s.charAt(i)
               && s.charAt(i - 2) != s.charAt(i)) {
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
      int remainA = capacityA;
      int remainB = capacityB;
      int i = 0;
      int j = plants.length - 1;
      while (i < j) {
         if (remainA >= plants[i]) {
            remainA -= plants[i];
         } else {
            ++res;
            remainA = capacityA - plants[i];
         }
         ++i;

         if (remainB >= plants[j]) {
            remainB -= plants[j];
         } else {
            ++res;
            remainB = capacityB - plants[j];
         }
         --j;
      }
      if (i == j && remainA < plants[i] && remainB < plants[i]) {
         ++res;
      }
      return res;

   }

   // 剑指 Offer 47. 礼物的最大价值
   public int maxValue(int[][] grid) {
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
   public int[] findEvenNumbers(int[] digits) {
      List<Integer> list = new ArrayList<>();
      int[] counts = new int[10];
      for (int digit : digits) {
         ++counts[digit];
      }
      int[] give = new int[10];
      search: for (int num = 100; num < 1000; num += 2) {
         Arrays.fill(give, 0);
         int cur = num;
         while (cur > 0) {
            int mod = cur % 10;
            ++give[mod];
            if (give[mod] > counts[mod]) {
               continue search;
            }
            cur /= 10;

         }
         list.add(num);
      }
      int[] res = new int[list.size()];
      for (int i = 0; i < list.size(); ++i) {
         res[i] = list.get(i);
      }
      return res;

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

   // 2103. 环和杆 (Rings and Rods) --哈希表+位运算+状态压缩
   public int countPoints(String rings) {
      int[] mask = new int[10];
      char[] chars = rings.toCharArray();
      for (int i = 0; i < chars.length; i += 2) {
         int index = chars[i + 1] - '0';
         if (chars[i] == 'R') {
            mask[index] |= 1 << 0;
         } else if (chars[i] == 'G') {
            mask[index] |= 1 << 1;
         } else if (chars[i] == 'B') {
            mask[index] |= 1 << 2;
         }
      }
      int res = 0;
      for (int num : mask) {
         if (num == 0b111) {
            ++res;
         }
      }
      return res;

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
      double[][] dp = new double[102][102];
      dp[0][0] = poured;
      for (int r = 0; r <= query_row; ++r) {
         for (int c = 0; c <= r; ++c) {
            double q = (dp[r][c] - 1.0) / 2;
            if (q > 0) {
               dp[r + 1][c] += q;
               dp[r + 1][c + 1] += q;
            }
         }
      }
      return Math.min(1.0d, dp[query_row][query_glass]);

   }

   // 1438. 绝对差不超过限制的最长连续子数组 (Longest Continuous Subarray With Absolute Diff Less
   // Than or Equal to Limit
   // --双指针+滑动窗口+TreeMap --还需要了解TreeMap的原理及更多用法
   public int longestSubarray(int[] nums, int limit) {
      TreeMap<Integer, Integer> map = new TreeMap<>();
      int left = 0;
      int right = 0;
      int res = 0;
      while (right < nums.length) {
         map.put(nums[right], map.getOrDefault(nums[right], 0) + 1);
         while (map.lastKey() - map.firstKey() > limit) {
            map.put(nums[left], map.get(nums[left]) - 1);
            if (map.get(nums[left]) == 0) {
               map.remove(nums[left]);
            }
            ++left;
         }
         res = Math.max(res, right - left + 1);
         ++right;
      }
      return res;

   }

   // 1438. 绝对差不超过限制的最长连续子数组 (Longest Continuous Subarray With Absolute Diff Less
   // Than or Equal to Limit
   // --双指针+滑动窗口+TreeMap --还需要了解双端队列的原理及更多用法
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
      StringBuilder res = new StringBuilder();
      char[] chars = s.toCharArray();
      int indexSpace = 0;
      int indexS = 0;
      while (indexSpace < spaces.length) {
         if (indexS == spaces[indexSpace]) {
            res.append(" ");
            ++indexSpace;
         } else {
            res.append(chars[indexS++]);
         }
      }
      res.append(Arrays.copyOfRange(chars, indexS, chars.length));

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
      int left = 0;
      int right = 0;
      Map<Integer, Integer> map = new HashMap<>();
      while (right < fruits.length) {
         map.put(fruits[right], map.getOrDefault(fruits[right], 0) + 1);
         while (map.keySet().size() > 2) {
            map.put(fruits[left], map.get(fruits[left]) - 1);
            if (map.get(fruits[left]) == 0) {
               map.remove(fruits[left]);
            }
            ++left;
         }
         res = Math.max(res, right - left + 1);
         ++right;
      }
      return res;

   }

   // 2104. 子数组范围和 (Sum of Subarray Ranges) --O(n^2) 还需掌握O(n)的解法
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
            map.put(p, map.getOrDefault(p, 0) + 1);
            if (map.get(p) >= map.get(top)) {
               top = p;
            }
            tops.add(top);
         }

      }

      public int q(int t) {
         int left = 0;
         int right = times.length - 1;
         int res = 0;
         while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (times[mid] <= t) {
               res = mid;
               left = mid + 1;
            } else if (times[mid] > t) {
               right = mid - 1;
            }
         }
         return tops.get(res);

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

   // 1696. 跳跃游戏 VI (Jump Game VI) --优先队列
   public int maxResult(int[] nums, int k) {
      Queue<int[]> priorityQueue = new PriorityQueue<>(new Comparator<int[]>() {

         @Override
         public int compare(int[] o1, int[] o2) {
            if (o1[0] != o2[0]) {
               return o2[0] - o1[0];
            }
            return o2[1] - o1[1];
         }

      });
      priorityQueue.offer(new int[] { nums[0], 0 });
      int res = priorityQueue.peek()[0];
      for (int i = 1; i < nums.length; ++i) {
         while (!priorityQueue.isEmpty() && i - priorityQueue.peek()[1] > k) {
            priorityQueue.poll();
         }
         res = priorityQueue.peek()[0] + nums[i];
         priorityQueue.offer(new int[] { res, i });

      }
      return res;

   }

   // 1696. 跳跃游戏 VI (Jump Game VI) --优先队列
   public int maxResult2(int[] nums, int k) {
      Deque<Integer> deque = new LinkedList<>();
      int[] dp = new int[nums.length];
      dp[0] = nums[0];
      deque.offerLast(0);
      for (int i = 1; i < nums.length; ++i) {
         while (!deque.isEmpty() && i - deque.peekFirst() > k) {
            deque.pollFirst();
         }
         dp[i] = dp[deque.peekFirst()] + nums[i];
         while (!deque.isEmpty() && dp[i] >= dp[deque.peekLast()]) {
            deque.pollLast();
         }
         deque.offerLast(i);
      }
      return dp[dp.length - 1];

   }

   // 1499. 满足不等式的最大值 (Max Value of Equation) --单调队列
   public int findMaxValueOfEquation(int[][] points, int k) {
      Deque<Integer> deque = new LinkedList<>();
      deque.offerLast(0);
      int res = Integer.MIN_VALUE;
      for (int i = 1; i < points.length; ++i) {
         while (!deque.isEmpty() && points[i][0] - points[deque.peekFirst()][0] > k) {
            deque.pollFirst();
         }
         if (!deque.isEmpty()) {
            res = Math.max(res,
                  points[i][0] + points[i][1] + points[deque.peekFirst()][1] - points[deque.peekFirst()][0]);
         }

         while (!deque.isEmpty()
               && points[deque.peekLast()][1] - points[deque.peekLast()][0] <= points[i][1] - points[i][0]) {
            deque.pollLast();
         }
         deque.offerLast(i);

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
   // 剑指 Offer 59 - I. 滑动窗口的最大值
   public int[] maxSlidingWindow(int[] nums, int k) {
      if (nums.length == 0) {
         return new int[0];
      }
      int[] res = new int[nums.length - k + 1];
      Deque<Integer> deque = new LinkedList<>();
      for (int i = 0; i < k; ++i) {
         while (!deque.isEmpty() && nums[deque.peekLast()] <= nums[i]) {
            deque.pollLast();
         }
         deque.offerLast(i);
      }
      res[0] = nums[deque.peekFirst()];
      for (int i = k; i < nums.length; ++i) {
         while (!deque.isEmpty() && nums[deque.peekLast()] <= nums[i]) {
            deque.pollLast();
         }
         deque.offerLast(i);
         while (!deque.isEmpty() && i - deque.peekFirst() >= k) {
            deque.pollFirst();
         }
         res[i - k + 1] = nums[deque.peekFirst()];
      }
      return res;

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

   // 641. 设计循环双端队列 (Design Circular Deque)
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

   // 862. 和至少为 K 的最短子数组 (Shortest Subarray with Sum at Least K)
   public int shortestSubarray(int[] nums, int k) {
      long[] preSum = new long[nums.length + 1];
      for (int i = 1; i < preSum.length; ++i) {
         preSum[i] += preSum[i - 1] + nums[i - 1];
         // if (nums[i - 1] >= k) {
         // return 1;
         // }
      }
      int res = Integer.MAX_VALUE;
      Deque<Integer> deque = new LinkedList<>();
      for (int i = 0; i < preSum.length; ++i) {
         while (!deque.isEmpty() && preSum[i] <= preSum[deque.peekLast()]) {
            deque.pollLast();
         }
         while (!deque.isEmpty() && preSum[i] - preSum[deque.peekFirst()] >= k) {
            res = Math.min(res, i - deque.pollFirst());
         }
         deque.offerLast(i);
      }
      return res == Integer.MAX_VALUE ? -1 : res;

   }

   // 918. 环形子数组的最大和 (Maximum Sum Circular Subarray) --动态规划
   public int maxSubarraySumCircular(int[] A) {
      int max = A[0];
      int pre = 0;
      for (int i = 0; i < A.length; ++i) {
         pre = Math.max(A[i], pre + A[i]);
         max = Math.max(pre, max);
      }

      int min = 0;
      pre = 0;
      for (int i = 1; i < A.length - 1; ++i) {
         pre = Math.min(A[i], pre + A[i]);
         min = Math.min(pre, min);
      }
      return Math.max(Arrays.stream(A).sum() - min, max);

   }

   // 918. 环形子数组的最大和 (Maximum Sum Circular Subarray) -- 前缀和 + 单调队列
   public int maxSubarraySumCircular2(int[] nums) {
      int[] preSum = new int[nums.length * 2 + 1];
      for (int i = 1; i < preSum.length; ++i) {
         preSum[i] = preSum[i - 1] + nums[(i - 1) % nums.length];
      }
      Deque<Integer> deque = new LinkedList<>();
      int res = Integer.MIN_VALUE;
      for (int i = 0; i < preSum.length; ++i) {
         while (!deque.isEmpty() && i - deque.peekFirst() > nums.length) {
            deque.pollFirst();
         }
         if (!deque.isEmpty()) {
            res = Math.max(res, preSum[i] - preSum[deque.peekFirst()]);
         }
         while (!deque.isEmpty() && preSum[i] <= preSum[deque.peekLast()]) {
            deque.pollLast();
         }
         deque.offerLast(i);
      }
      return res;

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
      List<Integer> oneCount = new ArrayList<>();
      for (String s : bank) {
         int one = 0;
         for (char c : s.toCharArray()) {
            if (c == '1') {
               ++one;
            }
         }
         if (one != 0) {
            oneCount.add(one);
         }
      }
      int res = 0;
      for (int i = 1; i < oneCount.size(); ++i) {
         res += oneCount.get(i) * oneCount.get(i - 1);
      }
      return res;
   }

   // 2125. 银行中的激光束数量 (Number of Laser Beams in a Bank)
   public int numberOfBeams2(String[] bank) {
      int last = 0;
      int res = 0;
      for (String s : bank) {
         int count = 0;
         for (char c : s.toCharArray()) {
            if (c == '1') {
               ++count;
            }
         }
         if (count != 0) {
            res += count * last;
            last = count;
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
   // Subsequences) --状态压缩
   public int maxProduct(String s) {
      int res = 0;
      List<Bean2002> list = new ArrayList<>();
      char[] chars = s.toCharArray();
      for (int i = 1; i < (1 << chars.length); ++i) {
         if (checkPalindromic(chars, i)) {
            list.add(new Bean2002(Integer.bitCount(i), i));
         }
      }
      for (int i = 0; i < list.size(); ++i) {
         for (int j = i + 1; j < list.size(); ++j) {
            if ((list.get(i).mask & list.get(j).mask) == 0) {
               res = Math.max(res, list.get(i).oneCount * list.get(j).oneCount);
            }
         }
      }
      return res;

   }

   private boolean checkPalindromic(char[] chars, int mask) {
      int left = 0;
      int right = chars.length - 1;
      while (left < right) {
         while (left < right && (mask >> left & 1) == 0) {
            ++left;
         }
         while (left < right && (mask >> right & 1) == 0) {
            --right;
         }
         if (chars[left++] != chars[right--]) {
            return false;
         }
      }
      return true;
   }

   public class Bean2002 {
      public int oneCount;
      public int mask;

      public Bean2002(int oneCount, int mask) {
         this.oneCount = oneCount;
         this.mask = mask;
      }

   }

   // 1255. 得分最高的单词集合 (Maximum Score Words Formed by Letters) --状态压缩
   public int maxScoreWords(String[] words, char[] letters, int[] score) {
      int[] counts = new int[26];
      for (char c : letters) {
         ++counts[c - 'a'];
      }
      // 记录不合法的单个词
      int illegalStatus = 0;
      for (int i = 0; i < words.length; ++i) {
         if (!isLegal(words[i], counts.clone())) {
            illegalStatus |= (1 << i);
         }
      }
      int res = 0;
      for (int i = 1; i < (1 << words.length); ++i) {
         // 若枚举值中存在不合法的单个词，则跳过
         if ((illegalStatus & i) == 0) {
            res = Math.max(res, check1255(i, words, counts.clone(), score));
         }
      }
      return res;

   }

   private int check1255(int mask, String[] words, int[] counts, int[] score) {
      int sum = 0;
      int index = 0;
      while (mask != 0) {
         if (mask % 2 == 1) {
            if (!isLegal(words[index], counts)) {
               return -1;
            }
            sum += getSum(words[index], score);
         }
         ++index;
         mask /= 2;
      }
      return sum;
   }

   private int getSum(String s, int[] score) {
      int sum = 0;
      for (char c : s.toCharArray()) {
         sum += score[c - 'a'];
      }
      return sum;
   }

   private boolean isLegal(String s, int[] counts) {
      for (char c : s.toCharArray()) {
         if (--counts[c - 'a'] < 0) {
            return false;
         }
      }
      return true;
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
      int i = nums.length - 2;
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
      int j = nums.length - 1;
      while (i < j) {
         if (nums[j] > nums[i]) {
            int temp = nums[j];
            nums[j] = nums[i];
            nums[i] = temp;
            break;
         }
         --j;
      }
      Arrays.sort(nums, i + 1, nums.length);
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

   // 剑指 Offer 46. 把数字翻译成字符串 --动态规划，类似 青蛙跳台、打家劫舍
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
   // Characters) --状态压缩 还需掌握回溯法
   public int maxLength(List<String> arr) {
      int res = 0;
      List<Integer> list = getList1239(arr);
      for (int status = 0; status < (1 << list.size()); ++status) {
         res = Math.max(res, getCount1239(status, list));
      }
      return res;

   }

   private int getCount1239(int status, List<Integer> list) {
      int mask = 0;
      int index = 0;
      while (status != 0) {
         if ((status & 1) == 1) {
            if ((mask & list.get(index)) != 0) {
               return -1;
            }
            mask |= list.get(index);
         }
         ++index;
         status >>= 1;
      }
      return Integer.bitCount(mask);
   }

   private List<Integer> getList1239(List<String> arr) {
      List<Integer> list = new ArrayList<>();
      search: for (String s : arr) {
         int mask = 0;
         for (char c : s.toCharArray()) {
            if ((mask & (1 << (c - 'a'))) != 0) {
               continue search;
            }
            mask |= 1 << (c - 'a');
         }
         list.add(mask);
      }
      return list;
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

   // 剑指 Offer II 041. 滑动窗口的平均值
   // 346
   class MovingAverage {
      private Queue<Integer> mQueue;
      private int mSum;
      private int mSize;

      /** Initialize your data structure here. */
      public MovingAverage(int size) {
         this.mSize = size;
         this.mQueue = new LinkedList<>();

      }

      public double next(int val) {
         mQueue.offer(val);
         mSum += val;
         while (mQueue.size() > mSize) {
            mSum -= mQueue.poll();
         }
         return (double) mSum / mQueue.size();
      }
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
   public int timeRequiredToBuy2(int[] tickets, int k) {
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
      int[] dx = { -1, -1, -1, 1, 1, 1, 0, 0 };
      int[] dy = { 1, 0, -1, 1, 0, -1, 1, -1 };
      for (int i = 0; i < 8; ++i) {
         if (check1958(board, rMove, cMove, dx[i], dy[i], color)) {
            return true;
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
         if (board[rMove][cMove] != color) {
            flag = true;
         } else {
            return flag;
         }
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
            if (++hyphenCount > 1 || i == 0 || i == s.length() - 1 || !Character.isLetter(s.charAt(i - 1)) || !Character
                  .isLetter(s.charAt(i + 1))) {
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
      int min = Integer.MAX_VALUE;
      for (int num : nums) {
         if (num < min) {
            min = num;
         } else if (num - min > 0) {
            res = Math.max(res, num - min);
         }
      }
      return res;

   }

   // 2138. 将字符串拆分为若干长度为 k 的组 (Divide a String Into Groups of Size k)
   public String[] divideString(String s, int k, char fill) {
      StringBuilder builder = new StringBuilder(s);
      while (builder.length() % k != 0) {
         builder.append(fill);
      }
      int n = builder.length() / k;
      String[] res = new String[n];
      int index = 0;
      for (int i = 0; i < n; ++i) {
         res[i] = builder.substring(index, index + k);
         index += k;
      }
      return res;

   }

   // 2139. 得到目标值的最少行动次数 (Minimum Moves to Reach Target Score)
   public int minMoves(int target, int maxDoubles) {
      int res = 0;
      while (target != 1) {
         if ((target & 1) != 0) {
            --target;
            ++res;
         } else if (maxDoubles-- > 0) {
            target >>>= 1;
            ++res;
         } else {
            res += target - 1;
            break;
         }
      }
      return res;

   }

   // 2140. 解决智力问题 (Solving Questions With Brainpower) --倒序 填表法
   public long mostPoints(int[][] questions) {
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
      int res = Integer.MAX_VALUE;
      int max = 0;
      for (int v : vat) {
         max = Math.max(max, v);
      }
      if (max == 0) {
         return 0;
      }
      for (int i = 1; i <= max; ++i) {
         if (i >= res) {
            break;
         }
         // 倒水的次数
         int cur = i;
         for (int j = 0; j < vat.length; ++j) {
            // 每个桶最少的增加容量次数
            cur += Math.max(0, vat[j] / i + (vat[j] % i == 0 ? 0 : 1) - bucket[j]);
         }
         res = Math.min(res, cur);
      }
      return res;

   }

   // LCP 30. 魔塔游戏
   public int magicTower(int[] nums) {
      long sum = Arrays.stream(nums).sum();
      if (sum < 0) {
         return -1;
      }
      int res = 0;
      PriorityQueue<Integer> priorityQueue = new PriorityQueue<>();
      long blood = 1L;
      for (int i = 0; i < nums.length; ++i) {
         if (nums[i] < 0) {
            priorityQueue.offer(nums[i]);
            if (nums[i] + blood <= 0) {
               ++res;
               blood -= priorityQueue.poll();
            }

         }
         blood += nums[i];
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
      int[] res = new int[2];
      int max = 0;
      for (int i = 0; i < 51; ++i) {
         for (int j = 0; j < 51; ++j) {
            int cur = 0;
            int x1 = i;
            int y1 = j;
            for (int[] tower : towers) {
               int x2 = tower[0];
               int y2 = tower[1];
               double distance = Math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
               if (distance <= radius) {
                  cur += tower[2] / (1 + distance);
               }
            }
            if (cur > max) {
               max = cur;
               res[0] = i;
               res[1] = j;
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

   // 5972. 统计隐藏数组数目 (Count the Hidden Sequences)
   public int numberOfArrays(int[] differences, int lower, int upper) {
      long min = 0;
      long max = 0;

      long cur = 0;
      for (int diff : differences) {
         cur += diff;
         min = Math.min(min, cur);
         max = Math.max(max, cur);
      }
      return (int) Math.max(0, (upper - lower) - (max - min) + 1);
   }

   // 5974. 分隔长廊的方案数 (Number of Ways to Divide a Long Corridor)
   public int numberOfWays(String corridor) {
      final int MOD = 1000000007;
      long res = 1L;
      char[] chars = corridor.toCharArray();
      int count = 0;
      int pre = -1;
      for (int i = 0; i < chars.length; ++i) {
         if (chars[i] == 'S') {
            if (++count >= 3 && count % 2 == 1) {
               res = res * (i - pre) % MOD;
            }
            pre = i;
         }
      }
      return count == 0 || count % 2 == 1 ? 0 : (int) res;

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
         if (map.get(key) == 1 && map.getOrDefault(key - 1, 0) == 0
               && map.getOrDefault(key + 1, 0) == 0) {
            res.add(key);
         }
      }
      return res;

   }

   // 5992. 基于陈述统计最多好人数 (Maximum Good People Based on Statements)
   public int maximumGood(int[][] statements) {
      int res = 0;
      int n = statements.length;
      for (int status = 0; status < (1 << n); ++status) {
         if (isLegal5992(status, statements)) {
            res = Math.max(res, Integer.bitCount(status));
         }
      }
      return res;

   }

   private boolean isLegal5992(int status, int[][] statements) {
      for (int i = 0; i < statements.length; ++i) {
         for (int j = 0; j < statements.length; ++j) {
            // i是好人 i认为j是坏人 但j是好人
            if (statements[i][j] == 0 && ((status >> i) & 1) == 1 && ((status >> j) & 1) == 1) {
               return false;
            }
            // i是好人 i认为j是好人 但j是坏人
            else if (statements[i][j] == 1 && ((status >> i) & 1) == 1 && ((status >> j) & 1) == 0) {
               return false;
            }
         }
      }
      return true;
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

   // 1403. 非递增顺序的最小子序列 (Minimum Subsequence in Non-Increasing Order)
   public List<Integer> minSubsequence(int[] nums) {
      List<Integer> res = new ArrayList<>();
      int sum = Arrays.stream(nums).sum();
      Arrays.sort(nums);
      int curSum = 0;
      for (int i = nums.length - 1; i >= 0; --i) {
         curSum += nums[i];
         res.add(nums[i]);
         if (curSum > sum - curSum) {
            return res;
         }
      }
      return res;

   }

   // 1029. 两地调度 (Two City Scheduling)
   public int twoCitySchedCost(int[][] costs) {
      Arrays.sort(costs, new Comparator<int[]>() {

         @Override
         public int compare(int[] o1, int[] o2) {
            return o1[0] - o1[1] - (o2[0] - o2[1]);
         }
      });
      int res = 0;
      for (int i = 0; i < costs.length / 2; ++i) {
         res += costs[i][0] + costs[i + costs.length / 2][1];
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

   // 1578. 使绳子变成彩色的最短时间 (Minimum Time to Make Rope Colorful)
   public int minCost(String colors, int[] neededTime) {
      int res = 0;
      char[] colorsChars = colors.toCharArray();
      int index = 0;
      while (index < colorsChars.length - 1) {
         if (colorsChars[index] == colorsChars[index + 1]) {
            int curSum = neededTime[index];
            int curMax = neededTime[index];
            while (index < colorsChars.length - 1 && colorsChars[index] == colorsChars[index + 1]) {
               curSum += neededTime[index + 1];
               curMax = Math.max(curMax, neededTime[index + 1]);
               ++index;
            }
            res += curSum - curMax;
         }
         ++index;
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
   public int largestValsFromLabels(int[] values, int[] labels, int num_wanted, int use_limit) {
      int n = values.length;
      int[][] set = new int[n][2];
      for (int i = 0; i < n; ++i) {
         set[i][0] = values[i];
         set[i][1] = labels[i];
      }
      Arrays.sort(set, new Comparator<int[]>() {

         @Override
         public int compare(int[] o1, int[] o2) {
            return o2[0] - o1[0];
         }
      });
      int res = 0;
      int count = 0;
      Map<Integer, Integer> map = new HashMap<>();
      for (int i = 0; i < n; ++i) {
         int label = set[i][1];
         map.put(label, map.getOrDefault(label, 0) + 1);
         if (map.get(label) <= use_limit) {
            res += set[i][0];
            if (++count == num_wanted) {
               return res;
            }
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

   // 1147. 段式回文 (Longest Chunked Palindrome Decomposition) --递归+贪心
   public int longestDecomposition(String text) {
      for (int i = 0; i < text.length() / 2; ++i) {
         if (text.substring(0, i + 1).equals(text.subSequence(text.length() - i - 1, text.length()))) {
            return 2 + longestDecomposition(text.substring(i + 1, text.length() - i - 1));
         }
      }
      return text.length() > 0 ? 1 : 0;

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

   // 5993. 将找到的值乘以 2
   public int findFinalValue(int[] nums, int original) {
      Set<Integer> set = new HashSet<>();
      for (int num : nums) {
         set.add(num);
      }
      int cur = original;
      while (set.contains(cur)) {
         cur <<= 1;
      }
      return cur;
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

   // 1262. 可被三整除的最大和 (Greatest Sum Divisible by Three)
   public int maxSumDivThree(int[] nums) {
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
      int res = 0;
      for (int coin : coins) {
         if (coin > res + 1) {
            break;
         }
         res += coin;
      }
      return res + 1;

   }

}