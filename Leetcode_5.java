import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

public class Leetcode_5 {
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

    // 1451. 重新排列句子中的单词 (Rearrange Words in a Sentence)
    public String arrangeWords(String text) {
        StringBuilder builder = new StringBuilder(text);
        builder.replace(0, 1, String.valueOf(Character.toLowerCase(builder.charAt(0))));
        List<String> list = new ArrayList<>();
        int left = 0;
        int right = 0;
        while (right < builder.length()) {
            while (right < builder.length() && Character.isLetter(builder.charAt(right))) {
                ++right;
            }
            list.add(builder.substring(left, right));
            left = right + 1;
            right = left;
        }
        Collections.sort(list, new Comparator<String>() {

            @Override
            public int compare(String o1, String o2) {
                return o1.length() - o2.length();
            }

        });

        StringBuilder res = new StringBuilder();
        for (String s : list) {
            if (!res.isEmpty()) {
                res.append(" ");
            }
            res.append(s);
        }
        return res.replace(0, 1, String.valueOf(Character.toUpperCase(res.charAt(0)))).toString();
    }

    // 1845. 座位预约管理系统 (Seat Reservation Manager) --优先队列
    class SeatManager {
        private PriorityQueue<Integer> queue;

        public SeatManager(int n) {
            queue = new PriorityQueue<>();
            for (int i = 1; i <= n; ++i) {
                queue.offer(i);
            }
        }

        public int reserve() {
            return queue.poll();

        }

        public void unreserve(int seatNumber) {
            queue.offer(seatNumber);
        }
    }

    // 1749. 任意子数组和的绝对值的最大值 (Maximum Absolute Sum of Any Subarray)
    public int maxAbsoluteSum(int[] nums) {
        int prefix = 0;
        int min = 0;
        int max = 0;
        int res = 0;
        for (int num : nums) {
            prefix += num;
            res = Math.max(Math.abs(prefix - min), res);
            res = Math.max(Math.abs(prefix - max), res);
            min = Math.min(min, prefix);
            max = Math.max(max, prefix);
        }
        return res;

    }

    // 816. 模糊坐标 (Ambiguous Coordinates)
    public List<String> ambiguousCoordinates(String s) {
        int n = s.length();
        List<String> res = new ArrayList<>();
        for (int i = 2; i < n - 1; ++i) {
            for (String left : makeString816(s, 1, i)) {
                for (String right : makeString816(s, i, n - 1)) {
                    res.add("(" + left + ", " + right + ")");
                }
            }
        }
        return res;
    }

    private List<String> makeString816(String s, int i, int j) {
        List<String> list = new ArrayList<>();
        for (int d = 1; d <= j - i; ++d) {
            String left = s.substring(i, i + d);
            String right = s.substring(i + d, j);
            if ((!left.startsWith("0") || "0".equals(left)) && !right.endsWith("0")) {
                list.add(left + (d < j - i ? "." : "") + right);
            }
        }
        return list;
    }

    // 373. 查找和最小的 K 对数字 (Find K Pairs with Smallest Sums) --优先队列 + BFS 与786原理相同
    // （还需掌握二分查找）
    // 剑指 Offer II 061. 和最小的 k 个数对
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        int m = nums1.length;
        int n = nums2.length;
        PriorityQueue<int[]> priorityQueue = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return nums1[o1[0]] + nums2[o1[1]] - (nums1[o2[0]] + nums2[o2[1]]);
            }

        });

        List<List<Integer>> res = new ArrayList<>();

        Set<String> visited = new HashSet<>();

        priorityQueue.offer(new int[] { 0, 0 });
        visited.add(0 + "_" + 0);
        while (k-- > 0 && !priorityQueue.isEmpty()) {
            int[] cur = priorityQueue.poll();
            int index1 = cur[0];
            int index2 = cur[1];
            res.add(List.of(nums1[index1], nums2[index2]));

            if (index1 + 1 < m && visited.add((index1 + 1) + "_" + index2)) {
                priorityQueue.offer(new int[] { index1 + 1, index2 });
            }

            if (index2 + 1 < n && visited.add(index1 + "_" + (index2 + 1))) {
                priorityQueue.offer(new int[] { index1, index2 + 1 });
            }
        }
        return res;

    }

    // 813. 最大平均值和的分组 (Largest Sum of Averages) --dp
    public double largestSumOfAverages(int[] nums, int k) {
        // dp[i][j] : 前i个元素，分成j份，可得到的最大平均值
        int n = nums.length;
        double[][] dp = new double[n + 1][k + 1];
        double[] prefix = new double[n + 1];
        for (int i = 1; i < n + 1; ++i) {
            prefix[i] = prefix[i - 1] + nums[i - 1];
        }

        for (int i = 1; i < n + 1; ++i) {
            dp[i][1] = prefix[i] / i;
            for (int kk = 2; kk <= k && kk <= i; ++kk) {
                for (int j = 1; j < i; ++j) {
                    dp[i][kk] = Math.max(dp[i][kk], dp[j][kk - 1] + (prefix[i] - prefix[j]) / (i - j));
                }
            }
        }
        return dp[n][k];

    }

    // 563. 二叉树的坡度 (Binary Tree Tilt) --dfs
    private int res563;

    public int findTilt(TreeNode root) {
        dfs563(root);
        return res563;
    }

    private int dfs563(TreeNode node) {
        if (node == null) {
            return 0;
        }
        int left = dfs563(node.left);
        int right = dfs563(node.right);
        res563 += Math.abs(left - right);
        return left + right + node.val;
    }

    // 2054. 两个最好的不重叠活动 (Two Best Non-Overlapping Events) --优先队列
    public int maxTwoEvents(int[][] events) {
        Arrays.sort(events, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }

        });

        PriorityQueue<int[]> priorityQueue = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }

        });
        int max = 0;
        int res = 0;

        for (int[] event : events) {
            while (!priorityQueue.isEmpty() && priorityQueue.peek()[0] < event[0]) {
                int[] cur = priorityQueue.poll();
                max = Math.max(max, cur[1]);
            }
            res = Math.max(res, max + event[2]);
            priorityQueue.offer(new int[] { event[1], event[2] });
        }
        return res;
    }

    public int maxProduct(int[] nums) {
        int max = -1;
        int secondMax = -1;
        for (int num : nums) {
            num -= 1;
            if (num >= max) {
                secondMax = max;
                max = num;
            } else if (num >= secondMax) {
                secondMax = num;
            }
        }
        return max * secondMax;

    }

    // 437. 路径总和 III (Path Sum III)
    // 剑指 Offer II 050. 向下的路径节点之和
    // 面试题 04.12. 求和路径 (Paths with Sum LCCI)
    // public int pathSum(TreeNode root, int targetSum) {

    // }
}
