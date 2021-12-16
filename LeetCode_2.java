import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

public class LeetCode_2 {

   public static void main(final String[] args) {
      // String s = "4w31am0ets6sl5go5ufytjtjpb7b0sxqbee2blg9ss";
      // int res = numDifferentIntegers(s);
      // int res = maxSumTwoNoOverlap(new int[] { 1, 0, 3 }, 1, 2);

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
      int count = small.length;
      int left = 0;
      int right = 0;
      Map<Integer, Integer> map = new HashMap<>();
      int minCount = Integer.MAX_VALUE;
      for (int num : small) {
         map.put(num, map.getOrDefault(num, 0) + 1);
      }
      int[] res = new int[] {};
      while (right < big.length) {
         if (map.containsKey(big[right])) {
            if (map.getOrDefault(big[right], 0) > 0) {
               --count;
            }
            map.put(big[right], map.getOrDefault(big[right], 0) - 1);
         }
         while (count == 0) {
            if (right - left + 1 < minCount) {
               minCount = right - left + 1;
               res = new int[] { left, right };
            }
            if (map.containsKey(big[left])) {
               map.put(big[left], map.getOrDefault(big[left], 0) + 1);
               if (map.get(big[left]) > 0) {
                  ++count;
               }
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

   // 1442. 形成两个异或相等数组的三元组数目 (Count Triplets That Can Form Two Arrays of Equal XOR)
   // public int countTriplets(int[] arr) {
   // int count = 0;
   // for (int i = 0; i < arr.length; ++i) {
   // int xor = 0;
   // for (int k = i; k < arr.length; ++k) {
   // xor ^= arr[k];
   // if (xor == 0) {
   // count += k - i;
   // }
   // }
   // }
   // return count;

   // }

   // 1074. 元素和为目标值的子矩阵数量 (Number of Submatrices That Sum to Target)
   // public int numSubmatrixSumTarget(int[][] matrix, int target) {

   // }
}