import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

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

}