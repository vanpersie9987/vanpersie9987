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

    // 1105. 填充书架 (Filling Bookcase Shelves)
    private int[] memo1105;
    private int n1105;
    private int[][] books1105;
    private int shelfWidth1105;

    public int minHeightShelves(int[][] books, int shelfWidth) {
        this.n1105 = books.length;
        this.books1105 = books;
        this.shelfWidth1105 = shelfWidth;
        memo1105 = new int[n1105];

        Arrays.fill(memo1105, -1);

        return dfs1105(0);

    }

    private int dfs1105(int i) {
        if (i == n1105) {
            return 0;
        }
        if (memo1105[i] != -1) {
            return memo1105[i];
        }
        int min = (int) 1e7;
        int j = i;
        int thick = 0;
        int maxHeight = 0;
        while (j < n1105 && thick + books1105[j][0] <= shelfWidth1105) {
            thick += books1105[j][0];
            maxHeight = Math.max(maxHeight, books1105[j][1]);
            min = Math.min(min, dfs1105(j + 1) + maxHeight);
            ++j;
        }
        return memo1105[i] = min;
    }

    // 1993. 树上的操作 (Operations on Tree)
    class LockingTree {
        private int n;
        private List<Integer>[] g;
        private int[] lockStatus;

        public LockingTree(int[] parent) {
            this.n = parent.length;
            this.g = new ArrayList[n];
            this.lockStatus = new int[n];
            for (int i = 0; i < n; ++i) {
                g[i] = new ArrayList<>();
            }
            for (int i = 0; i < n; ++i) {
                if (parent[i] != -1) {
                    g[parent[i]].add(i);
                }
                lockStatus[i] = -1;
            }
        }

        public boolean lock(int num, int user) {
            if (lockStatus[num] != -1) {
                return false;
            }
            lockStatus[num] = user;
            return true;
        }

        public boolean unlock(int num, int user) {
            if (lockStatus[num] != user) {
                return false;
            }
            lockStatus[num] = -1;
            return true;
        }

        public boolean upgrade(int num, int user) {
            if (lockStatus[num] != -1) {
                return false;
            }
            // num的祖先节点是否都未上锁
            if (!dfs(0, num)) {
                return false;
            }
            // num是否至少有一个上锁的子孙节点，并将所有上锁节点解锁
            if (dfs2(num)) {
                lockStatus[num] = user;
                return true;
            }
            return false;

        }

        private boolean dfs(int x, int num) {
            if (lockStatus[x] != -1) {
                return false;
            }
            if (x == num) {
                return true;
            }
            for (int y : g[x]) {
                if (dfs(y, num)) {
                    return true;
                }
            }
            return false;
        }

        private boolean dfs2(int x) {
            boolean flag = false;
            for (int y : g[x]) {
                if (dfs2(y)) {
                    flag = true;
                }
            }
            if (lockStatus[x] != -1) {
                flag = true;
                lockStatus[x] = -1;
            }
            return flag;
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

    // 1937. 扣分后的最大得分 (Maximum Number of Points with Cost)
    // private int m;
    // private int n;
    // private int[][] points;
    // private long[][] memo;

    // public long maxPoints(int[][] points) {
    // this.m = points.length;
    // this.n = points[0].length;
    // this.points = points;
    // this.memo = new long[m][n];
    // for (int i = 0; i < m; ++i) {
    // Arrays.fill(memo[i], Long.MIN_VALUE);
    // }
    // long res = 0l;
    // for (int j = 0; j < n; ++j) {
    // res = Math.max(res, dfs(1, j) + points[0][j]);
    // }
    // return res;
    // }

    // private long dfs(int row, int lastCol) {
    // if (row == m) {
    // return 0;
    // }
    // if (memo[row][lastCol] != Long.MIN_VALUE) {
    // return memo[row][lastCol];
    // }
    // long res = Long.MIN_VALUE;
    // for (int j = 0; j < n; ++j) {
    // res = Math.max(res, dfs(row + 1, j) + points[row][j] - Math.abs(lastCol -
    // j));
    // }
    // return memo[row][lastCol] = res;
    // }

    // 1771. 由子序列构造的最长回文串的长度 (Maximize Palindrome Length From Subsequences)
    // private int m;
    // private int n;
    // private String s;
    // private int[][] memo;

    // public int longestPalindrome(String word1, String word2) {
    // this.m = word1.length();
    // this.n = word2.length();
    // this.s = word1 + word2;
    // this.memo = new int[m + n][m + n];
    // for (int i = 0; i < m + n; ++i) {
    // Arrays.fill(memo[i], -1);
    // }
    // int res = dfs(false, 0, m + n - 1);
    // return res > 1 ? res : 0;

    // }

    // private int dfs(boolean b, int i, int j) {
    // if (i > j) {
    // return 0;
    // }
    // if (i == j) {
    // return 1;
    // }
    // if (memo[i][j] != -1) {
    // return memo[i][j];
    // }
    // if (i >= m || j < m) {
    // if (!b) {
    // return 0;
    // }
    // }
    // if (s.charAt(i) == s.charAt(j)) {
    // return memo[i][j] = dfs(true, i + 1, j - 1) + 2;
    // }
    // return memo[i][j] = Math.max(dfs(b, i + 1, j), dfs(b, i, j - 1));
    // }

    
}
