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
import java.util.Random;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Leetcode_7 {
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

    // 1363. 形成三的最大倍数 (Largest Multiple of Three)
    // public String largestMultipleOfThree(int[] digits) {

    // }

    // 2402. 会议室 III (Meeting Rooms III)
    // public int mostBooked(int n, int[][] meetings) {

    // }

    // 638. 大礼包 (Shopping Offers)
    // public int shoppingOffers(List<Integer> price, List<List<Integer>> special,
    // List<Integer> needs) {

    // }

    // 1316. 不同的循环子字符串 (Distinct Echo Substrings)
    // public int distinctEchoSubstrings(String text) {

    // }

    // 1473. 粉刷房子 III (Paint House III)
    // private int[][][] memo;
    // private int m1473;
    // private int n1473;
    // private int target1473;
    // private int[] houses1473;
    // private int[][] cost1473;

    // public int minCost(int[] houses, int[][] cost, int m, int n, int target) {
    // // memo[i][j][k] 将[0,i]房子涂色 第i个房子被涂成第j种颜色 且它属于第k个街区的 最小花销
    // memo = new int[m][n][target];
    // m1473 = m;
    // n1473 = n;
    // target1473 = target;
    // cost1473 = cost;
    // houses1473 = houses;
    // int res = dfs(0, n + 1, target);
    // return res == (int) 1e8 ? -1 : res;

    // }

    // private int dfs(int i, int lastColor, int kinds) {
    // if (i == m1473 || kinds < 0 || kinds > m1473 - i) {
    // if (i == m1473 && kinds == 0) {
    // return 0;
    // }
    // return (int) 1e8;
    // }

    // if (memo[i][lastColor][target1473] != 0) {
    // return memo[i][lastColor][target1473];
    // }
    // int min = (int) 1e8;
    // if (houses1473[i] != 0) {
    // min = Math.min(min, dfs(i + 1, houses1473[i], kinds + (lastColor !=
    // houses1473[i] ? -1 : 0)));
    // return memo[i][lastColor][target1473] = min;
    // }
    // for (int color = 1; color <= n1473; ++color) {
    // min = Math.min(min, cost1473[i][color - 1] + dfs(i + 1, color, kinds +
    // (lastColor != color ? -1 : 0)));
    // }
    // return memo[i][lastColor][target1473] = min;
    // }

    // 1671. 得到山形数组的最少删除次数 (Minimum Number of Removals to Make Mountain Array)
    // public int minimumMountainRemovals(int[] nums) {
    // }
}
