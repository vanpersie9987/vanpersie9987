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

    // 3006. 找出数组中的美丽下标 I (Find Beautiful Indices in the Given Array I)
    // 3008. 找出数组中的美丽下标 II (Find Beautiful Indices in the Given Array II) --z函数
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

    // 3142. 判断矩阵是否满足条件 (Check if Grid Satisfies Conditions)
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

    // 3143. 正方形中的最多点数 (Maximum Points Inside the Square)
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

    // 3144. 分割字符频率相等的最少子字符串 (Minimum Substring Partition of Equal Character
    // Frequency)
    private int n3144;
    private int[] memo3144;
    private String s3144;

    public int minimumSubstringsInPartition(String s) {
        this.n3144 = s.length();
        this.memo3144 = new int[n3144];
        Arrays.fill(memo3144, -1);
        this.s3144 = s;
        return dfs3144(0);
    }

    private int dfs3144(int i) {
        if (i == n3144) {
            return 0;
        }
        if (memo3144[i] != -1) {
            return memo3144[i];
        }
        int res = Integer.MAX_VALUE;
        int[] cnts = new int[26];
        int c = 0;
        search: for (int j = i; j < n3144; ++j) {
            c += ++cnts[s3144.charAt(j) - 'a'] == 1 ? 1 : 0;
            if ((j - i + 1) % c != 0) {
                continue;
            }
            for (int k = 0; k < 26; ++k) {
                if (cnts[k] > 0 && cnts[k] != (j - i + 1) / c) {
                    continue search;
                }
            }
            res = Math.min(res, dfs3144(j + 1) + 1);
        }
        return memo3144[i] = res;
    }

    // 3146. 两个字符串的排列差 (Permutation Difference between Two Strings)
    public int findPermutationDifference(String s, String t) {
        int[] pos = new int[26];
        for (int i = 0; i < s.length(); ++i) {
            pos[s.charAt(i) - 'a'] = i;
        }
        int res = 0;
        for (int i = 0; i < t.length(); ++i) {
            res += Math.abs(i - pos[t.charAt(i) - 'a']);
        }
        return res;

    }

    private int[] energy;
    private int k;
    private int[] memo;

    public int maximumEnergy(int[] energy, int k) {
        int n = energy.length;
        this.energy = energy;
        this.k = k;
        this.memo = new int[n];
        int res = Integer.MIN_VALUE;
        for (int i = n - 1; i > n - k - 1; --i) {
            res = Math.max(res, dfs(i - k) + energy[i]);
        }
        return res;
    }

    private int dfs(int i) {
        if (i < 0) {
            return 0;
        }
        if (memo[i] != 0) {
            return memo[i];
        }
        return memo[i] = Math.max(dfs(i - k) + energy[i], 0);
    }

    public int maxScore(List<List<Integer>> grid) {
        int m = grid.size();
        int n = grid.get(0).size();
        int res = Integer.MIN_VALUE;
        int[][] pre = new int[m + 1][n + 1];
        for (int i = 0; i < m + 1; ++i) {
            Arrays.fill(pre[i], Integer.MAX_VALUE);
        }
        for (int i = 1; i < m + 1; ++i) {
            for (int j = 1; j < n + 1; ++j) {
                int min = Math.min(pre[i - 1][j], pre[i][j - 1]);
                // 若 grid 中存在负数，就会越界，应修改初值
                res = Math.max(res, grid.get(i - 1).get(j - 1) - min);
                pre[i][j] = Math.min(min, grid.get(i - 1).get(j - 1));
            }
        }
        return res;

    }

}
