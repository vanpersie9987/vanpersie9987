import java.math.BigInteger;
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
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@SuppressWarnings("unchecked")
public class Leetcode_11 {
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

    class Node {
        public int val;
        public Node left;
        public Node right;
        public Node parent;
    }

    class Node2 {
        public int val;
        public Node2 prev;
        public Node2 next;
    }

    // 3919. 在下标间移动的最小代价 (Minimum Cost to Move Between Indices)
    public int[] minCost(int[] nums, int[][] queries) {
        int n = nums.length;
        int[] sumL = new int[n]; // sumL[i] 等于从 i 移动到 0 的代价和
        int[] sumR = new int[n]; // sumR[i] 等于从 0 移动到 i 的代价和
        for (int i = 1, cost; i < n; i++) {
            // 往左走 i -> i-1
            if (i < n - 1 && nums[i] - nums[i - 1] > nums[i + 1] - nums[i]) { // closest(i) = i+1
                cost = nums[i] - nums[i - 1]; // 只能用方式一往左走
            } else {
                cost = 1;
            }
            sumL[i] = sumL[i - 1] + cost;

            // 往右走 i-1 -> i
            if (i > 1 && nums[i - 1] - nums[i - 2] <= nums[i] - nums[i - 1]) { // closest(i-1) = i-2
                cost = nums[i] - nums[i - 1]; // 只能用方式一往右走
            } else {
                cost = 1;
            }
            sumR[i] = sumR[i - 1] + cost;
        }

        int[] ans = new int[queries.length];
        for (int i = 0; i < queries.length; i++) {
            int l = queries[i][0];
            int r = queries[i][1];
            if (l < r) {
                // cost(0 -> r) - cost(0 -> l) = cost(l -> r)
                ans[i] = sumR[r] - sumR[l];
            } else {
                // cost(l -> 0) - cost(r -> 0) = cost(l -> r)
                ans[i] = sumL[l] - sumL[r];
            }
        }
        return ans;

    }

}
