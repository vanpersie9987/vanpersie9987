public class Leetcode_6 {
    public static void main(String[] args) {

    }

    public class ListNode {
        public int val;
        public ListNode next;

        public ListNode(int val) {
            this.val = val;
        }

        public ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }

    }

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    // 1151. 最少交换次数来组合所有的 1 (Minimum Swaps to Group All 1's Together) --plus
    public int minSwaps(int[] data) {
        int n = data.length;
        int k = 0;
        for (int d : data) {
            k += d;
        }
        int cur = 0;
        for (int i = 0; i < k; ++i) {
            if (data[i] == 0) {
                ++cur;
            }
        }
        int res = cur;
        for (int i = k; i < n; ++i) {
            if (data[i] == 0) {
                ++cur;
            }
            if (data[i - k] == 0) {
                --cur;
            }
            res = Math.min(res, cur);
        }
        return res;

    }

    // 1648. 销售价值减少的颜色球 (Sell Diminishing-Valued Colored Balls)
    // public int maxProfit(int[] inventory, int orders) {

    // }

    // 1562. 查找大小为 M 的最新分组 (Find Latest Group of Size M)
    // public int findLatestStep(int[] arr, int m) {

    // }

    // 813. 最大平均值和的分组 (Largest Sum of Averages)
    // public double largestSumOfAverages(int[] nums, int k) {

    // }

    // 2484. 统计回文子序列数目 (Count Palindromic Subsequences)
    // public int countPalindromes(String s) {

    // }

    // 2488. 统计中位数为 K 的子数组 (Count Subarrays With Median K)
    // public int countSubarrays(int[] nums, int k) {

    // }

    // 2467. 树上最大得分和路径 (Most Profitable Path in a Tree)
    // public int mostProfitablePath(int[][] edges, int bob, int[] amount) {

    // }

    // 2466. 统计构造好字符串的方案数 (Count Ways To Build Good Strings)
    // public int countGoodStrings(int low, int high, int zero, int one) {

    // }

}
