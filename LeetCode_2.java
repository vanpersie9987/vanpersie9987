import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
         arr[num] |= 0b001;
      }
      for (int num : nums2) {
         arr[num] |= 0b010;
      }
      for (int num : nums3) {
         arr[num] |= 0b100;
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
}