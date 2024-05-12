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

public class Leetcode_9 {
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

    // 3008. 找出数组中的美丽下标 II (Find Beautiful Indices in the Given Array II)
    public List<Integer> beautifulIndices(String s, String a, String b, int k) {
        List<Integer> aList = zAlgorithm3008(a, s);
        List<Integer> bList = zAlgorithm3008(b, s);
        List<Integer> res = new ArrayList<>();
        int j = 0;
        for (int x : aList) {
            while (j < bList.size() && x - bList.get(j) > k) {
                ++j;
            }
            if (j < bList.size() && Math.abs(x - bList.get(j)) <= k) {
                res.add(x);
            }
        }
        return res;

    }

    private List<Integer> zAlgorithm3008(String t, String s) {
        String ss = t + s;
        int n = ss.length();
        int[] z = new int[n];
        int left = 0;
        int right = 0;
        List<Integer> res = new ArrayList<>();
        for (int i = 1; i < n; ++i) {
            if (i <= right) {
                z[i] = Math.min(z[i - left], right - i + 1);
            }
            while (i + z[i] < n && ss.charAt(z[i]) == ss.charAt(i + z[i])) {
                left = i;
                right = i + z[i];
                ++z[i];
            }
            if (i >= t.length() && z[i] >= t.length()) {
                res.add(i - t.length());
            }
        }
        return res;
    }

    public boolean satisfiesConditions(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        for (int j = 1; j < n; ++j) {
            if (grid[0][j] == grid[0][j - 1]) {
                return false;
            }
        }
        for (int i = 1; i < m; ++i) {
            if (!Arrays.equals(grid[0], grid[i])) {
                return false;
            }
        }
        return true;
    }

    public int maxPointsInsideSquare(int[][] points, String s) {
        int n = points.length;
        int[][] arr = new int[n][2];
        for (int i = 0; i < n; ++i) {
            arr[i][0] = Math.max(Math.abs(points[i][0]), Math.abs(points[i][1]));
            arr[i][1] = s.charAt(i) - 'a';
        }
        Arrays.sort(arr, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });

        int res = 0;
        int i = 0;
        int m = 0;
        while (i < n) {
            int j = i;
            int c = arr[i][0];
            while (j < n && c == arr[j][0]) {
                if ((m >> arr[j][1] & 1) == 1) {
                    return res;
                }
                m |= 1 << arr[j][1];
                ++j;
            }
            res += j - i;
            i = j;
        }
        return res;
    }

    private int n;
    private int[] memo;
    private String s;

    public int minimumSubstringsInPartition(String s) {
        this.n = s.length();
        this.memo = new int[n];
        Arrays.fill(memo, -1);
        this.s = s;
        return dfs(0);
    }

    private int dfs(int i) {
        if (i == n) {
            return 0;
        }
        if (memo[i] != -1) {
            return memo[i];
        }
        int res = Integer.MAX_VALUE;
        int[] cnts = new int[26];
        search: for (int j = i; j < n; ++j) {
            ++cnts[s.charAt(j) - 'a'];
            int d = 0;
            for (int k = 0; k < 26; ++k) {
                if (cnts[k] == 0) {
                    continue;
                }
                if (d == 0) {
                    d = cnts[k];
                } else if (d != cnts[k]) {
                    continue search;
                }
            }
            res = Math.min(res, dfs(j + 1) + 1);
        }
        return memo[i] = res;
    }
}
