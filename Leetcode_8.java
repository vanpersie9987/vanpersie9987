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

@SuppressWarnings("unchecked")
public class Leetcode_8 {
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

    // 面试题 17.25. 单词矩阵 (Word Rectangle LCCI)
    private List<String> res17_25;
    private Trie17_25 root17_25;
    private TreeMap<Integer, Set<String>> treeMap17_25;
    private int maxArea17_25;

    public String[] maxRectangle(String[] words) {
        this.res17_25 = new ArrayList<>();
        this.root17_25 = new Trie17_25();
        this.treeMap17_25 = new TreeMap<>(new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(o2, o1);
            }

        });
        for (String word : words) {
            root17_25.insert(word);
            treeMap17_25.computeIfAbsent(word.length(), k -> new HashSet<>()).add(word);
        }
        for (Map.Entry<Integer, Set<String>> entry : treeMap17_25.entrySet()) {
            int side = Math.min(entry.getValue().size(), entry.getKey());
            if (side * side <= maxArea17_25) {
                break;
            }
            dfs17_25(entry.getValue(), new ArrayList<>(), entry.getKey());
        }
        return res17_25.toArray(new String[0]);

    }

    private void dfs17_25(Set<String> set, List<String> path, int len) {
        if (path.size() > len) {
            return;
        }
        for (String word : set) {
            path.add(word);
            boolean[] check = root17_25.check(path);
            if (check[0]) {
                if (check[1] && path.size() * len > maxArea17_25) {
                    maxArea17_25 = path.size() * len;
                    res17_25 = new ArrayList<>(path);
                }
                dfs17_25(set, path, len);
            }
            path.remove(path.size() - 1);
        }

    }

    public class Trie17_25 {
        private Trie17_25[] children;
        private boolean isWord;

        public Trie17_25() {
            children = new Trie17_25[26];
        }

        public void insert(String s) {
            Trie17_25 node = this;
            for (char c : s.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new Trie17_25();
                }
                node = node.children[index];
            }
            node.isWord = true;
        }

        public boolean[] check(List<String> path) {
            int m = path.size();
            int n = path.get(0).length();
            boolean[] res = new boolean[2];
            Arrays.fill(res, true);
            for (int j = 0; j < n; ++j) {
                Trie17_25 node = root17_25;
                for (int i = 0; i < m; ++i) {
                    int index = path.get(i).charAt(j) - 'a';
                    if (node.children[index] == null) {
                        Arrays.fill(res, false);
                        return res;
                    }
                    node = node.children[index];
                }
                if (!node.isWord) {
                    res[1] = false;
                }
            }
            return res;
        }
    }

    // 2484. 统计回文子序列数目 (Count Palindromic Subsequences)
    public int countPalindromes(String s) {
        int n = s.length();
        int[] suf = new int[10];
        int[][] suf2 = new int[10][10];
        char[] arr = s.toCharArray();
        for (int i = n - 1; i >= 0; --i) {
            int d = arr[i] - '0';
            for (int j = 0; j < 10; ++j) {
                suf2[d][j] += suf[j];
            }
            ++suf[d];
        }
        final int MOD = (int) (1e9 + 7);
        int[] pre = new int[10];
        int[][] pre2 = new int[10][10];
        int res = 0;
        for (char d : s.toCharArray()) {
            d -= '0';
            --suf[d];
            for (int j = 0; j < 10; ++j) {
                suf2[d][j] -= suf[j];
            }
            for (int j = 0; j < 10; ++j) {
                for (int k = 0; k < 10; ++k) {
                    res = (int) ((res + (long) pre2[j][k] * suf2[j][k]) % MOD);
                }
            }
            for (int j = 0; j < 10; ++j) {
                pre2[d][j] += pre[j];
            }
            ++pre[d];
        }
        return res;

    }

    // 1691. 堆叠长方体的最大高度 (Maximum Height by Stacking Cuboids)
    private int n1691;
    private int[][] cuboids1691;
    private int[] memo1691;

    public int maxHeight(int[][] cuboids) {
        this.n1691 = cuboids.length;
        for (int[] c : cuboids) {
            Arrays.sort(c);
        }
        Arrays.sort(cuboids, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o2[0] + o2[1] + o2[2], o1[0] + o1[1] + o1[2]);
            }

        });
        this.cuboids1691 = cuboids;
        this.memo1691 = new int[n1691];
        int res = 0;
        for (int i = 0; i < n1691; ++i) {
            res = Math.max(res, dfs1691(i) + cuboids[i][2]);
        }
        return res;

    }

    private int dfs1691(int i) {
        if (i == n1691) {
            return 0;
        }
        if (memo1691[i] != 0) {
            return memo1691[i];
        }
        int max = 0;
        for (int j = i + 1; j < n1691; ++j) {
            if (check1691(cuboids1691[i], cuboids1691[j])) {
                max = Math.max(max, dfs1691(j) + cuboids1691[j][2]);
            }
        }
        return memo1691[i] = max;
    }

    private boolean check1691(int[] a, int[] b) {
        return b[0] <= a[0] && b[1] <= a[1] && b[2] <= a[2];
    }

    // 1771. 由子序列构造的最长回文串的长度 (Maximize Palindrome Length From Subsequences)
    private int m1771;
    private char[] arr1771;
    private int[][][] memo1771;

    public int longestPalindrome(String word1, String word2) {
        this.m1771 = word1.length();
        int n = word2.length();
        this.arr1771 = (word1 + word2).toCharArray();
        this.memo1771 = new int[m1771 + n][m1771 + n][2];
        for (int i = 0; i < m1771 + n; ++i) {
            for (int j = 0; j < m1771 + n; ++j) {
                Arrays.fill(memo1771[i][j], -1);
            }
        }
        int res = dfs1771(0, m1771 + n - 1, 0);
        return res > 1 ? res : 0;

    }

    private int dfs1771(int i, int j, int b) {
        if (i > j) {
            return 0;
        }
        if (i == j) {
            return 1;
        }
        if (!(i < m1771 && j >= m1771)) {
            if (b == 0) {
                return 0;
            }
        }
        if (memo1771[i][j][b] != -1) {
            return memo1771[i][j][b];
        }
        if (arr1771[i] == arr1771[j]) {
            return memo1771[i][j][b] = dfs1771(i + 1, j - 1, 1) + 2;
        }
        return memo1771[i][j][b] = Math.max(dfs1771(i + 1, j, b), dfs1771(i, j - 1, b));
    }

    // 1682. 最长回文子序列 II --plus 未提交
    private int[][][] memo1682;
    private char[] arr1682;

    public int longestPalindromeSubseq(String s) {
        int n = s.length();
        this.arr1682 = s.toCharArray();
        this.memo1682 = new int[n][n][27];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                Arrays.fill(memo1682[i][j], -1);
            }
        }
        return dfs1682(0, n - 1, 26);
    }

    private int dfs1682(int i, int j, int k) {
        if (i >= j) {
            return 0;
        }
        if (memo1682[i][j][k] != -1) {
            return memo1682[i][j][k];
        }
        if (arr1682[i] == arr1682[j]) {
            // 不选
            int max = dfs1682(i + 1, j - 1, k);
            // 选
            if (arr1682[i] - 'a' != k) {
                max = Math.max(max, dfs1682(i + 1, j - 1, arr1682[i] - 'a') + 2);
            }
            return memo1682[i][j][k] = max;
        }
        return memo1682[i][j][k] = Math.max(dfs1682(i, j - 1, k), dfs1682(i + 1, j, k));
    }

    // 1349. 参加考试的最大学生数 (Maximum Students Taking Exam)
    private int u1349;
    private int[] rows1349;
    private int[][] memo1349;

    public int maxStudents(char[][] seats) {
        int m = seats.length;
        int n = seats[0].length;
        this.rows1349 = new int[m];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (seats[i][j] == '#') {
                    rows1349[i] |= 1 << j;
                }
            }
        }
        this.u1349 = (1 << n) - 1;
        this.memo1349 = new int[m][1 << n];
        for (int i = 0; i < m; ++i) {
            Arrays.fill(memo1349[i], -1);
        }
        return dfs1349(m - 1, 0);
    }

    private int dfs1349(int i, int j) {
        if (i < 0) {
            return 0;
        }
        if (memo1349[i][j] != -1) {
            return memo1349[i][j];
        }
        int mask = ((j << 1) | (j >> 1) | rows1349[i]) & u1349;
        int c = mask ^ u1349;
        // 不选
        int max = dfs1349(i - 1, 0);
        // 选
        for (int k = c; k > 0; k = (k - 1) & c) {
            if ((k & (k >> 1)) == 0) {
                max = Math.max(max, dfs1349(i - 1, k) + Integer.bitCount(k));
            }
        }
        return memo1349[i][j] = max;
    }

    // 1681. 最小不兼容性 (Minimum Incompatibility)
    private int[] map1681;
    private int n1681;
    private int[][] memo1681;
    private int u1681;

    public int minimumIncompatibility(int[] nums, int k) {
        this.n1681 = nums.length;
        int[] cnts = new int[n1681 + 1];
        for (int num : nums) {
            ++cnts[num];
            if (cnts[num] > k) {
                return -1;
            }
        }
        if (n1681 == k) {
            return 0;
        }
        this.map1681 = new int[1 << n1681];
        Arrays.fill(map1681, -1);
        search: for (int i = 1; i < (1 << n1681); ++i) {
            if (Integer.bitCount(i) == n1681 / k) {
                int mask = 0;
                int j = i;
                int min = Integer.MAX_VALUE;
                int max = Integer.MIN_VALUE;
                while (j != 0) {
                    int index = Integer.numberOfTrailingZeros(j);
                    int num = nums[index];
                    if (((mask >> num) & 1) != 0) {
                        continue search;
                    }
                    mask |= 1 << num;
                    min = Math.min(min, num);
                    max = Math.max(max, num);
                    j &= j - 1;
                }
                map1681[i] = max - min;
            }
        }
        this.memo1681 = new int[k][1 << n1681];
        for (int i = 0; i < k; ++i) {
            Arrays.fill(memo1681[i], -1);
        }
        this.u1681 = (1 << n1681) - 1;
        return dfs1681(0, 0);

    }

    private int dfs1681(int i, int mask) {
        if (mask == u1681) {
            return 0;
        }
        if (memo1681[i][mask] != -1) {
            return memo1681[i][mask];
        }
        int c = u1681 ^ mask;
        int min = (int) 1e5;
        for (int j = c; j > 0; j = (j - 1) & c) {
            if (map1681[j] != -1) {
                min = Math.min(min, dfs1681(i + 1, mask | j) + map1681[j]);
            }
        }
        return memo1681[i][mask] = min;
    }

    // 1655. 分配重复整数 (Distribute Repeating Integers)
    private int n1655;
    private int[] sum1655;
    private int[] cnts1655;
    private int[][] memo1655;
    private int u1655;

    public boolean canDistribute(int[] nums, int[] quantity) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.merge(num, 1, Integer::sum);
        }
        this.n1655 = map.size();
        this.cnts1655 = new int[n1655];
        int id = 0;
        for (int c : map.values()) {
            cnts1655[id++] = c;
        }
        int m = quantity.length;
        this.sum1655 = new int[1 << m];
        for (int i = 1; i < 1 << m; ++i) {
            int bit = Integer.numberOfTrailingZeros(i);
            int y = i ^ (1 << bit);
            sum1655[i] = sum1655[y] + quantity[bit];
        }
        this.memo1655 = new int[n1655][1 << m];
        this.u1655 = (1 << m) - 1;
        return dfs1655(0, 0);

    }

    private boolean dfs1655(int i, int mask) {
        if (mask == u1655) {
            return true;
        }
        if (i == n1655) {
            return false;
        }
        if (memo1655[i][mask] != 0) {
            return memo1655[i][mask] > 0;
        }
        // 不选
        if (dfs1655(i + 1, mask)) {
            memo1655[i][mask] = 1;
            return true;
        }
        int c = u1655 ^ mask;
        for (int j = c; j > 0; j = (j - 1) & c) {
            if (cnts1655[i] >= sum1655[j] && dfs1655(i + 1, mask | j)) {
                // 选
                memo1655[i][mask] = 1;
                return true;
            }
        }
        memo1655[i][mask] = -1;
        return false;
    }

    // 1879. 两个数组最小的异或值之和 (Minimum XOR Sum of Two Arrays)
    private int n1879;
    private int[] nums1_1879;
    private int[] nums2_1879;
    private int[][] memo1879;

    public int minimumXORSum(int[] nums1, int[] nums2) {
        this.n1879 = nums1.length;
        this.nums1_1879 = nums1;
        this.nums2_1879 = nums2;
        this.memo1879 = new int[n1879][1 << n1879];
        for (int i = 0; i < n1879; ++i) {
            Arrays.fill(memo1879[i], -1);
        }
        return dfs1879(0, 0);

    }

    private int dfs1879(int i, int mask) {
        if (i == n1879) {
            return 0;
        }
        if (memo1879[i][mask] != -1) {
            return memo1879[i][mask];
        }
        int min = (int) 1e9;
        int c = ((1 << n1879) - 1) ^ mask;
        while (c != 0) {
            int bit = Integer.numberOfTrailingZeros(c);
            min = Math.min(min, dfs1879(i + 1, mask | (1 << bit)) + (nums1_1879[i] ^ nums2_1879[bit]));
            c &= c - 1;
        }
        return memo1879[i][mask] = min;
    }

    // 2744. 最大字符串配对数目 (Find Maximum Number of String Pairs)
    public int maximumNumberOfStringPairs(String[] words) {
        int res = 0;
        Set<Integer> set = new HashSet<>();
        for (String w : words) {
            int h = (w.charAt(1) << 8) + w.charAt(0);
            if (set.contains(h)) {
                ++res;
            }
            set.add((w.charAt(0) << 8) + w.charAt(1));
        }
        return res;
    }

    // 6895. 构造最长的新字符串 (Construct the Longest New String)
    public int longestString(int x, int y, int z) {
        return (Math.min(x, y) * 2 + (x != y ? 1 : 0) + z) * 2;
    }

    // 6898. 字符串连接删减字母 (Decremental String Concatenation)
    private int[][][] memo6898;
    private int n6898;
    private String[] words6898;

    public int minimizeConcatenatedLength(String[] words) {
        this.n6898 = words.length;
        this.words6898 = words;
        this.memo6898 = new int[n6898][26][26];
        return dfs6898(0, 0, 0);

    }

    private int dfs6898(int i, int start, int end) {
        if (i == n6898) {
            return 0;
        }
        int len = words6898[i].length();
        int left = words6898[i].charAt(0) - 'a';
        int right = words6898[i].charAt(len - 1) - 'a';
        if (i == 0) {
            return dfs6898(i + 1, left, right) + len;
        }
        if (memo6898[i][start][end] != 0) {
            return memo6898[i][start][end];
        }
        return memo6898[i][start][end] = Math.min(dfs6898(i + 1, start, right) + (left == end ? -1 : 0),
                dfs6898(i + 1, left, end) + (right == start ? -1 : 0)) + len;
    }

    // 6468. 统计没有收到请求的服务器数目 (Count Zero Request Servers)
    public int[] countServers(int n, int[][] logs, int x, int[] queries) {
        int m = queries.length;
        Integer[] ids = IntStream.range(0, m).boxed().toArray(Integer[]::new);
        Arrays.sort(ids, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(queries[o1], queries[o2]);
            }

        });
        Arrays.sort(logs, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[1], o2[1]);
            }

        });
        int[] res = new int[m];
        int[] cnts = new int[n + 1];
        int i = 0;
        int j = 0;
        int cur = 0;
        for (int id : ids) {
            int r = queries[id];
            while (j < logs.length && logs[j][1] <= r) {
                ++cnts[logs[j][0]];
                if (cnts[logs[j][0]] == 1) {
                    ++cur;
                }
                ++j;
            }
            int l = queries[id] - x;
            while (i < logs.length && logs[i][1] < l) {
                --cnts[logs[i][0]];
                if (cnts[logs[i][0]] == 0) {
                    --cur;
                }
                ++i;
            }
            res[id] = n - cur;
        }
        return res;

    }

    // 6466. 美丽下标对的数目 (Number of Beautiful Pairs)
    public int countBeautifulPairs(int[] nums) {
        int res = 0;
        int[] cnts = new int[10];
        for (int num : nums) {
            int b = num % 10;
            for (int i = 1; i < 10; ++i) {
                if (gcd6466(i, b) == 1) {
                    res += cnts[i];
                }
            }
            while (num >= 10) {
                num /= 10;
            }
            ++cnts[num];
        }
        return res;
    }

    private int gcd6466(int a, int b) {
        return b == 0 ? a : gcd6466(b, a % b);
    }

    // 6471. 得到整数零需要执行的最少操作数 (Minimum Operations to Make the Integer Zero)
    public int makeTheIntegerZero(int num1, int num2) {
        for (int i = 1; i <= 60; ++i) {
            long x = (long) num1 - (long) i * num2;
            if (x <= 0) {
                continue;
            }
            if (Long.bitCount(x) <= i && i <= x) {
                return i;
            }
        }
        return -1;

    }

    // 6910. 将数组划分成若干好子数组的方式 (Ways to Split Array Into Good Subarrays)
    public int numberOfGoodSubarraySplits(int[] nums) {
        int n = nums.length;
        int res = 1;
        int pre = -1;
        final int MOD = (int) (1e9 + 7);
        for (int i = 0; i < n; ++i) {
            if (nums[i] == 1) {
                if (pre >= 0) {
                    res = (int) ((long) res * (i - pre) % MOD);
                }
                pre = i;
            }
        }
        return pre < 0 ? 0 : res;
    }

    // 2172. 数组的最大与和 (Maximum AND Sum of Array)
    private int[][] memo2172;
    private int[] nums2172;
    private int n2172;
    private int u2172;

    public int maximumANDSum(int[] nums, int numSlots) {
        this.n2172 = nums.length;
        // 看成有 numSlots * 2 个篮子，每个篮子可放最多1个数
        this.u2172 = (1 << (numSlots * 2)) - 1;
        this.nums2172 = nums;
        this.memo2172 = new int[n2172][1 << (numSlots * 2)];
        for (int i = 0; i < n2172; ++i) {
            Arrays.fill(memo2172[i], Integer.MIN_VALUE);
        }
        return dfs2172(0, 0);

    }

    private int dfs2172(int i, int mask) {
        if (i == n2172) {
            return 0;
        }
        if (n2172 - i > Integer.bitCount(u2172 ^ mask)) {
            return (int) -1e5;
        }
        if (memo2172[i][mask] != Integer.MIN_VALUE) {
            return memo2172[i][mask];
        }
        int max = 0;
        int c = u2172 ^ mask;
        while (c > 0) {
            int bit = Integer.numberOfTrailingZeros(c);
            max = Math.max(max, dfs2172(i + 1, mask | (1 << bit)) + ((bit / 2 + 1) & nums2172[i]));
            c &= c - 1;
        }
        return memo2172[i][mask] = max;
    }

    // 730. 统计不同回文子序列 (Count Different Palindromic Subsequences)
    private int n730;
    private char[] arr730;
    private int[][][] memo730;

    public int countPalindromicSubsequences(String s) {
        this.n730 = s.length();
        this.arr730 = s.toCharArray();
        this.memo730 = new int[4][n730][n730];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < n730; ++j) {
                Arrays.fill(memo730[i][j], -1);
            }
        }
        final int MOD = (int) (1e9 + 7);
        int res = 0;
        for (int i = 0; i < 4; ++i) {
            res = (res + dfs730(i, 0, n730 - 1)) % MOD;
        }
        return res;
    }

    private int dfs730(int c, int l, int r) {
        if (l > r) {
            return 0;
        }
        if (l == r) {
            return arr730[l] - 'a' == c ? 1 : 0;
        }
        if (memo730[c][l][r] != -1) {
            return memo730[c][l][r];
        }
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        if (arr730[l] - 'a' == c && arr730[r] - 'a' == c) {
            res = (res + 2) % MOD;
            for (int i = 0; i < 4; ++i) {
                res = (res + dfs730(i, l + 1, r - 1)) % MOD;
            }
        } else if (arr730[l] - 'a' == c) {
            res = (res + dfs730(c, l, r - 1) + 1) % MOD;
        } else if (arr730[r] - 'a' == c) {
            res = (res + dfs730(c, l + 1, r) + 1) % MOD;
        } else {
            res = (res + dfs730(c, l + 1, r - 1)) % MOD;
        }
        return memo730[c][l][r] = res;
    }

    // 2751. 机器人碰撞 (Robot Collisions)
    public List<Integer> survivedRobotsHealths(int[] positions, int[] healths, String directions) {
        int n = positions.length;
        int[][] arr = new int[n][4];
        for (int i = 0; i < n; ++i) {
            // 编号，位置，健康度，方向
            arr[i] = new int[] { i, positions[i], healths[i], directions.charAt(i) == 'L' ? 0 : 1 };
        }
        Arrays.sort(arr, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[1], o2[1]);
            }

        });
        Stack<int[]> stack = new Stack<>();
        search: for (int[] a : arr) {
            // 向右
            if (a[3] == 1) {
                stack.push(a);
                continue;
            }
            // 当前向左 栈不为空 栈顶向右
            while (!stack.isEmpty() && stack.peek()[3] == 1) {
                int[] cur = stack.pop();
                if (cur[2] == a[2]) {
                    continue search;
                }
                if (cur[2] > a[2]) {
                    --cur[2];
                    stack.push(cur);
                    continue search;
                }
                --a[2];
            }
            stack.push(a);
        }
        List<int[]> list = new ArrayList<>(stack);
        Collections.sort(list, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });
        List<Integer> res = new ArrayList<>();
        for (int[] a : list) {
            res.add(a[2]);
        }
        return res;
    }

    // 1186. 删除一次得到子数组最大和 (Maximum Subarray Sum with One Deletion)
    private int n1186;
    private int[] arr1186;
    private int[][] memo1186;

    public int maximumSum(int[] arr) {
        this.n1186 = arr.length;
        this.arr1186 = arr;
        this.memo1186 = new int[n1186][2];
        for (int i = 0; i < n1186; ++i) {
            Arrays.fill(memo1186[i], Integer.MIN_VALUE);
        }
        int res = Integer.MIN_VALUE;
        for (int i = 0; i < n1186; ++i) {
            res = Math.max(res, Math.max(dfs1186(i, 1), dfs1186(i, 0)));
        }
        return res;
    }

    private int dfs1186(int i, int j) {
        if (i < 0) {
            return Integer.MIN_VALUE / 2;
        }
        if (memo1186[i][j] != Integer.MIN_VALUE) {
            return memo1186[i][j];
        }
        if (j == 0) {
            return memo1186[i][j] = Math.max(dfs1186(i - 1, 0) + arr1186[i], arr1186[i]);
        }
        return memo1186[i][j] = Math.max(dfs1186(i - 1, 1) + arr1186[i], dfs1186(i - 1, 0));
    }

    // 1240. 铺瓷砖 (Tiling a Rectangle with the Fewest Squares)
    private int n1240;
    private int m1240;
    private int[] filled1240;
    private int res1240;

    public int tilingRectangle(int n, int m) {
        this.n1240 = n;
        this.m1240 = m;
        this.filled1240 = new int[n];
        this.res1240 = n * m;
        dfs1240(0, 0, 0);
        return res1240;
    }

    private void dfs1240(int i, int j, int k) {
        if (i == n1240) {
            res1240 = k;
            return;
        }
        if (j == m1240) {
            dfs1240(i + 1, 0, k);
            return;
        }
        if (((filled1240[i] >> j) & 1) == 1) {
            dfs1240(i, j + 1, k);
        } else if (k + 1 < res1240) {
            int r = 0;
            int c = 0;
            for (int x = i; x < n1240; ++x) {
                if (((filled1240[x] >> j) & 1) == 1) {
                    break;
                }
                ++r;
            }
            for (int y = j; y < m1240; ++y) {
                if (((filled1240[i] >> y) & 1) == 1) {
                    break;
                }
                ++c;
            }
            int s = Math.min(r, c);
            int mask = ((1 << (j + s)) - 1) ^ ((1 << j) - 1);
            for (int x = i; x < i + s; ++x) {
                filled1240[x] |= mask;
            }
            for (int w = s; w > 0; --w) {
                dfs1240(i, j + w, k + 1);
                for (int x = 0; x < w; ++x) {
                    filled1240[i + w - 1] ^= 1 << (j + x);
                    if (x < w - 1) {
                        filled1240[i + x] ^= 1 << (j + w - 1);
                    }
                }
            }
        }
    }

    // 960. 删列造序 III (Delete Columns to Make Sorted III)
    private int n960;
    private int m960;
    private String[] strs960;
    private int[][] memo960;

    public int minDeletionSize(String[] strs) {
        this.n960 = strs.length;
        this.m960 = strs[0].length();
        this.strs960 = strs;
        this.memo960 = new int[m960 + 1][m960 + 1];
        for (int i = 0; i < m960 + 1; ++i) {
            Arrays.fill(memo960[i], -1);
        }
        return m960 - dfs960(1, 0);
    }

    private int dfs960(int i, int j) {
        if (i == m960 + 1) {
            return 0;
        }
        if (memo960[i][j] != -1) {
            return memo960[i][j];
        }
        // 不选
        int max = dfs960(i + 1, j);
        // 选
        for (int k = 0; j != 0 && k < n960; ++k) {
            if (strs960[k].charAt(i - 1) < strs960[k].charAt(j - 1)) {
                return memo960[i][j] = max;
            }
        }
        return memo960[i][j] = Math.max(max, dfs960(i + 1, i) + 1);
    }

    // 630. 课程表 III (Course Schedule III)
    public int scheduleCourse(int[][] courses) {
        Arrays.sort(courses, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[1], o2[1]);
            }

        });
        Queue<Integer> q = new PriorityQueue<>(new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(o2, o1);
            }

        });
        int day = 0;
        for (int[] c : courses) {
            int t = c[0];
            int d = c[1];
            if (day + t <= d) {
                q.offer(t);
                day += t;
            } else if (!q.isEmpty() && q.peek() > t) {
                day -= q.poll();
                day += t;
                q.offer(t);
            }
        }
        return q.size();

    }

    // 679. 24 点游戏 (24 Game)
    public boolean judgePoint24(int[] cards) {
        double[] arr = new double[4];
        for (int i = 0; i < 4; ++i) {
            arr[i] = cards[i];
        }
        return dfs679(arr);
    }

    private boolean dfs679(double[] arr) {
        if (arr.length == 1) {
            // arr[0] == 24
            return Math.abs(arr[0] - 24D) <= 1e-6;
        }
        for (int i = 0; i < arr.length; ++i) {
            for (int j = i + 1; j < arr.length; ++j) {
                double a = arr[i];
                double b = arr[j];
                double[] nArr = new double[arr.length - 1];
                int id = 0;
                for (int k = 0; k < arr.length; ++k) {
                    if (i != k && j != k) {
                        nArr[id++] = arr[k];
                    }
                }
                nArr[id] = a + b;
                if (dfs679(nArr)) {
                    return true;
                }

                nArr[id] = a - b;
                if (dfs679(nArr)) {
                    return true;
                }

                nArr[id] = b - a;
                if (dfs679(nArr)) {
                    return true;
                }

                nArr[id] = a * b;
                if (dfs679(nArr)) {
                    return true;
                }

                // b != 0
                if (b > 1e-6) {
                    nArr[id] = a / b;
                    if (dfs679(nArr)) {
                        return true;
                    }
                }

                // a != 0
                if (a > 1e-6) {
                    nArr[id] = b / a;
                    if (dfs679(nArr)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    // 1178. 猜字谜 (Number of Valid Words for Each Puzzle)
    public List<Integer> findNumOfValidWords(String[] words, String[] puzzles) {
        Map<Integer, Integer> cnts = new HashMap<>();
        for (String word : words) {
            int m = getMask1178(word);
            cnts.merge(m, 1, Integer::sum);
        }
        List<Integer> res = new ArrayList<>();
        for (String p : puzzles) {
            int cur = 0;
            int first = p.charAt(0) - 'a';
            int m = getMask1178(p);
            for (int i = m; i > 0; i = (i - 1) & m) {
                if (((i >> first) & 1) == 1) {
                    cur += cnts.getOrDefault(i, 0);
                }
            }
            res.add(cur);
        }
        return res;

    }

    private int getMask1178(String s) {
        int m = 0;
        for (char c : s.toCharArray()) {
            m |= 1 << (c - 'a');
        }
        return m;
    }

    // 1959. K 次调整数组大小浪费的最小总空间 (Minimum Total Space Wasted With K Resizing
    // Operations)
    private int k1959;
    private int n1959;
    private int[][] arr1959;
    private int[][] memo1959;

    public int minSpaceWastedKResizing(int[] nums, int k) {
        this.k1959 = k;
        this.n1959 = nums.length;
        this.arr1959 = new int[n1959][n1959];
        int sum = 0;
        for (int i = 0; i < n1959; ++i) {
            int max = nums[i];
            sum += nums[i];
            for (int j = i; j < n1959; ++j) {
                max = Math.max(max, nums[j]);
                arr1959[i][j] = max * (j - i + 1);
            }
        }
        this.memo1959 = new int[n1959][k];
        for (int i = 0; i < n1959; ++i) {
            Arrays.fill(memo1959[i], -1);
        }
        return dfs1959(0, 0) - sum;

    }

    private int dfs1959(int i, int j) {
        if (i == n1959) {
            return 0;
        }
        if (j == k1959) {
            return arr1959[i][n1959 - 1];
        }
        if (memo1959[i][j] != -1) {
            return memo1959[i][j];
        }
        int min = Integer.MAX_VALUE;
        for (int x = i; x < n1959; ++x) {
            min = Math.min(min, dfs1959(x + 1, j + 1) + arr1959[i][x]);
        }
        return memo1959[i][j] = min;
    }

    // 2193. 得到回文串的最少操作次数 (Minimum Number of Moves to Make Palindrome)
    public int minMovesToMakePalindrome(String s) {
        int n = s.length();
        char[] arr = s.toCharArray();
        int res = 0;
        for (int i = 0, j = n - 1; i < j; ++i) {
            int k = j;
            while (i < k) {
                if (arr[i] == arr[k]) {
                    break;
                }
                --k;
            }
            if (i != k) {
                while (k + 1 <= j) {
                    swap2193(arr, k, k + 1);
                    ++res;
                    ++k;
                }
                --j;
            } else {
                res += n / 2 - i;
            }
        }
        return res;

    }

    private void swap2193(char[] arr, int i, int j) {
        char tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    // 2760. 最长奇偶子数组 (Longest Even Odd Subarray With Threshold)
    public int longestAlternatingSubarray(int[] nums, int threshold) {
        int res = 0;
        int i = 0;
        int n = nums.length;
        while (i < n) {
            int j = i + 1;
            if (nums[i] % 2 == 0 && nums[i] <= threshold) {
                while (j < n && nums[j] % 2 != nums[j - 1] % 2 && nums[j] <= threshold) {
                    ++j;
                }
                res = Math.max(res, j - i);
            }
            i = j;
        }
        return res;

    }

    // 6916. 和等于目标值的质数对 (Prime Pairs With Target Sum)
    public List<List<Integer>> findPrimePairs(int n) {
        List<List<Integer>> res = new ArrayList<>();
        if (n <= 3) {
            return res;
        }
        if (n == 4) {
            res.add(List.of(2, 2));
        }
        boolean[] isPrime = new boolean[n + 1];
        Arrays.fill(isPrime, true);
        for (int i = 2; i < n + 1; ++i) {
            if (isPrime[i]) {
                for (int j = i + i; j < n + 1; j += i) {
                    isPrime[j] = false;
                }
            }
        }

        if (n % 2 == 1) {
            if (isPrime[n - 2]) {
                res.add(List.of(2, n - 2));
            }
            return res;
        }
        for (int i = 3; i < n + 1; i += 2) {
            if (i > n - i) {
                break;
            }
            if (isPrime[i] && isPrime[n - i]) {
                res.add(List.of(i, n - i));
            }
        }
        return res;

    }

    // 6911. 不间断子数组 (Continuous Subarrays)
    public long continuousSubarrays(int[] nums) {
        long res = 0L;
        int n = nums.length;
        TreeMap<Integer, Integer> map = new TreeMap<>();
        int i = 0;
        int j = 0;
        while (j < n) {
            map.merge(nums[j], 1, Integer::sum);
            while (map.lastKey() - map.firstKey() > 2) {
                map.merge(nums[i], -1, Integer::sum);
                if (map.getOrDefault(nums[i], 0) == 0) {
                    map.remove(nums[i]);
                }
                ++i;
            }
            res += j - i + 1;
            ++j;
        }
        return res;

    }

    // 1307. 口算难题 (Verbal Arithmetic Puzzle)
    private int used1307;
    private String result1307;
    private String[] words1307;
    private boolean flag1307;
    private int[] nums1307;

    public boolean isSolvable(String[] words, String result) {
        this.words1307 = words;
        this.result1307 = result;
        this.nums1307 = new int[26];
        Arrays.fill(nums1307, -1);
        dfs_result_1307(0, 0);
        return flag1307;

    }

    private void dfs_result_1307(int i, int sum) {
        if (i == result1307.length()) {
            dfs_words_1307(0, 0, 0, sum);
            return;
        }
        if (nums1307[result1307.charAt(i) - 'A'] != -1) {
            dfs_result_1307(i + 1, sum * 10 + nums1307[result1307.charAt(i) - 'A']);
        } else {
            for (int j = 0; j < 10; ++j) {
                if (i == 0 && j == 0 && result1307.length() > 1) {
                    continue;
                }
                if (((used1307 >> j) & 1) != 0) {
                    continue;
                }
                nums1307[result1307.charAt(i) - 'A'] = j;
                used1307 ^= 1 << j;
                dfs_result_1307(i + 1, sum * 10 + j);
                if (flag1307) {
                    return;
                }
                used1307 ^= 1 << j;
                nums1307[result1307.charAt(i) - 'A'] = -1;
            }
        }
    }

    private void dfs_words_1307(int i, int j, int curSum, int sum) {
        if (i == words1307.length) {
            if (curSum == sum) {
                flag1307 = true;
            }
            return;
        }
        if (j == words1307[i].length()) {
            dfs_words_1307(i + 1, 0, curSum, sum);
            return;
        }
        if (nums1307[words1307[i].charAt(j) - 'A'] != -1) {
            int cur = curSum
                    + nums1307[words1307[i].charAt(j) - 'A'] * (int) Math.pow(10, words1307[i].length() - j - 1);
            if (cur <= sum) {
                dfs_words_1307(i, j + 1, cur, sum);
            }
            return;
        }
        for (int k = 0; k < 10; ++k) {
            if (j == 0 && k == 0 && words1307[i].length() > 1) {
                continue;
            }
            if (((used1307 >> k) & 1) != 0) {
                continue;
            }
            int cur = curSum + k * (int) Math.pow(10, words1307[i].length() - j - 1);
            if (cur > sum) {
                break;
            }
            nums1307[words1307[i].charAt(j) - 'A'] = k;
            used1307 ^= 1 << k;
            dfs_words_1307(i, j + 1, cur, sum);
            if (flag1307) {
                return;
            }
            used1307 ^= 1 << k;
            nums1307[words1307[i].charAt(j) - 'A'] = -1;
        }
    }

    // 2763. 所有子数组中不平衡数字之和 (Sum of Imbalance Numbers of All Subarrays)
    public int sumImbalanceNumbers(int[] nums) {
        int res = 0;
        int n = nums.length;
        boolean[] vis = new boolean[n + 2];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(vis, false);
            int cnts = 0;
            vis[nums[i]] = true;
            for (int j = i + 1; j < n; ++j) {
                if (vis[nums[j]]) {
                    res += cnts;
                    continue;
                }
                if (vis[nums[j] + 1] && vis[nums[j] - 1]) {
                    --cnts;
                } else if (!vis[nums[j] + 1] && !vis[nums[j] - 1]) {
                    ++cnts;
                }
                res += cnts;
                vis[nums[j]] = true;
            }
        }
        return res;

    }

    // 1096. 花括号展开 II (Brace Expansion II)
    private TreeSet<String> set1096;

    public List<String> braceExpansionII(String expression) {
        this.set1096 = new TreeSet<>();
        dfs1096(expression);
        return new ArrayList<>(set1096);

    }

    private void dfs1096(String s) {
        int j = s.indexOf("}");
        if (j == -1) {
            set1096.add(s);
            return;
        }
        int i = s.lastIndexOf("{", j);
        String a = s.substring(0, i);
        String c = s.substring(j + 1);
        for (String b : s.substring(i + 1, j).split(",")) {
            dfs1096(a + b + c);
        }
    }

    // 1755. 最接近目标值的子序列和 (Closest Subsequence Sum)
    private int index1755;

    public int minAbsDifference(int[] nums, int goal) {
        int n = nums.length;
        int[] left = new int[1 << (n / 2)];
        dfs1755(0, Arrays.copyOfRange(nums, 0, n / 2), 0, left);
        int[] right = new int[1 << (n - n / 2)];
        index1755 = 0;
        dfs1755(0, Arrays.copyOfRange(nums, n / 2, n), 0, right);
        Arrays.sort(left);
        Arrays.sort(right);
        int l = 0;
        int r = right.length - 1;
        int res = Integer.MAX_VALUE;
        while (l < left.length && r >= 0) {
            int cur = left[l] + right[r];
            res = Math.min(res, Math.abs(cur - goal));
            if (cur > goal) {
                --r;
            } else {
                ++l;
            }
        }
        return res;

    }

    private void dfs1755(int i, int[] nums, int sum, int[] arr) {
        if (i == nums.length) {
            arr[index1755++] = sum;
            return;
        }
        // 不选
        dfs1755(i + 1, nums, sum, arr);
        // 选
        dfs1755(i + 1, nums, sum + nums[i], arr);
    }

    // 2163. 删除元素后和的最小差值 (Minimum Difference in Sums After Removal of Elements)
    public long minimumDifference(int[] nums) {
        int n = nums.length;
        Queue<Integer> q = new PriorityQueue<>(new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(o2, o1);
            }

        });
        long sum = 0L;
        long[] preMin = new long[n];
        for (int i = 0; i < n; ++i) {
            q.offer(nums[i]);
            sum += nums[i];
            if (q.size() > n / 3) {
                sum -= q.poll();
            }
            preMin[i] = sum;
        }
        sum = 0L;
        long res = Long.MAX_VALUE;
        q = new PriorityQueue<>();
        for (int i = n - 1; i >= n / 3; --i) {
            q.offer(nums[i]);
            sum += nums[i];
            if (q.size() > n / 3) {
                sum -= q.poll();
            }
            if (i <= n / 3 * 2) {
                res = Math.min(res, preMin[i - 1] - sum);
            }
        }
        return res;

    }

    // 2242. 节点序列的最大得分 (Maximum Score of a Node Sequence)
    public int maximumScore(int[] scores, int[][] edges) {
        int n = scores.length;
        List<int[]>[] g = new ArrayList[n];
        Arrays.setAll(g, k -> new ArrayList<>());
        for (int[] e : edges) {
            int x = e[0];
            int y = e[1];
            g[x].add(new int[] { scores[y], y });
            g[y].add(new int[] { scores[x], x });
        }
        for (int i = 0; i < n; ++i) {
            if (g[i].size() > 3) {
                g[i].sort(new Comparator<int[]>() {

                    @Override
                    public int compare(int[] o1, int[] o2) {
                        return Integer.compare(o2[0], o1[0]);
                    }

                });
                g[i] = new ArrayList<>(g[i].subList(0, 3));
            }
        }
        int res = -1;
        for (int[] e : edges) {
            int x = e[0];
            int y = e[1];
            for (int[] i : g[x]) {
                int a = i[1];
                for (int[] j : g[y]) {
                    int b = j[1];
                    if (y != a && x != b && a != b) {
                        res = Math.max(res, i[0] + scores[x] + scores[y] + j[0]);
                    }
                }
            }
        }
        return res;

    }

    // 2681. 英雄的力量 (Power of Heroes)
    public int sumOfPower(int[] nums) {
        Arrays.sort(nums);
        long res = 0L;
        long s = 0L;
        final int MOD = (int) (1e9 + 7);
        for (long x : nums) {
            res = (res + ((x * x) % MOD) * (x + s)) % MOD;
            s = (s * 2 + x) % MOD;
        }
        return (int) res;

    }

    // 2616. 最小化数对的最大差值 (Minimize the Maximum Difference of Pairs)
    private int[] nums2616;
    private int[] memo2616;
    private int n2616;
    private int diff2616;

    public int minimizeMax(int[] nums, int p) {
        Arrays.sort(nums);
        this.n2616 = nums.length;
        this.nums2616 = nums;
        this.memo2616 = new int[n2616];
        int left = 0;
        int right = nums[n2616 - 1] - nums[0];
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (check2616(mid) >= p) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;

    }

    private int check2616(int diff) {
        this.diff2616 = diff;
        Arrays.fill(memo2616, -1);
        return dfs2616(0);
    }

    private int dfs2616(int i) {
        if (i >= n2616) {
            return 0;
        }
        if (memo2616[i] != -1) {
            return memo2616[i];
        }
        // 不选
        int max = dfs2616(i + 1);
        // 选
        if (i + 1 < n2616 && nums2616[i + 1] - nums2616[i] <= diff2616) {
            max = Math.max(max, dfs2616(i + 2) + 1);
        }
        return memo2616[i] = max;
    }

    // 2765. 最长交替子数组 (Longest Alternating Subarray)
    public int alternatingSubarray(int[] nums) {
        int n = nums.length;
        int i = 0;
        int res = -1;
        while (i < n) {
            int j = i + 1;
            int d = 1;
            while (j < n && nums[j] - nums[j - 1] == d) {
                ++j;
                d = -d;
            }
            if (j - i > 1) {
                res = Math.max(res, j - i);
                i = j - 1;
            } else {
                i = j;
            }
        }
        return res;

    }

    // 6469. 重新放置石块 (Relocate Marbles)
    public List<Integer> relocateMarbles(int[] nums, int[] moveFrom, int[] moveTo) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        for (int i = 0; i < moveFrom.length; ++i) {
            set.remove(moveFrom[i]);
            set.add(moveTo[i]);
        }
        List<Integer> res = new ArrayList<>(set);
        Collections.sort(res);
        return res;

    }

    // 2767. 将字符串分割为最少的美丽子字符串 (Partition String Into Minimum Beautiful Substrings)
    private String s2767;
    private int[] memo2767;
    private int n2767;

    public int minimumBeautifulSubstrings(String s) {
        if (s.charAt(0) == '0') {
            return -1;
        }
        this.s2767 = s;
        this.n2767 = s.length();
        this.memo2767 = new int[n2767];
        Arrays.fill(memo2767, n2767 + 1);
        int res = dfs2767(0);
        return res == n2767 + 1 ? -1 : res;

    }

    private int dfs2767(int i) {
        if (i == n2767) {
            return 0;
        }
        if (s2767.charAt(i) == '0') {
            return n2767 + 1;
        }
        if (memo2767[i] != n2767 + 1) {
            return memo2767[i];
        }
        int min = n2767 + 1;
        long sum = 0L;
        for (int j = i; j < n2767; ++j) {
            sum = (sum << 1) | (s2767.charAt(j) - '0');
            if (check2767(sum)) {
                min = Math.min(min, dfs2767(j + 1) + 1);
            }
        }
        return memo2767[i] = min;
    }

    private boolean check2767(long sum) {
        while (sum != 1L) {
            if (sum % 5 != 0) {
                return false;
            }
            sum /= 5;
        }
        return true;
    }

    // 6928. 黑格子的数目 (Number of Black Blocks)
    public long[] countBlackBlocks(int m, int n, int[][] coordinates) {
        long[] res = new long[5];
        long M = (long) 1e6;
        Set<Long> set = new HashSet<>();
        for (int[] c : coordinates) {
            int x = c[0];
            int y = c[1];
            long s = x * M + y;
            set.add(s);
        }
        Set<Long> vis = new HashSet<>();
        for (int[] c : coordinates) {
            int x = c[0];
            int y = c[1];
            // 右下
            if (x > 0 && y > 0) {
                int cnt = 1;
                if (set.contains((x - 1) * M + y)) {
                    ++cnt;
                }
                if (set.contains(x * M + y - 1)) {
                    ++cnt;
                }
                if (set.contains((x - 1) * M + y - 1)) {
                    ++cnt;
                }
                if (vis.add((x - 1) * M + y - 1)) {
                    ++res[cnt];
                }
            }
            // 左下
            if (x > 0 && y + 1 < n) {
                int cnt = 1;
                if (set.contains((x - 1) * M + y)) {
                    ++cnt;
                }
                if (set.contains(x * M + y + 1)) {
                    ++cnt;
                }
                if (set.contains((x - 1) * M + y + 1)) {
                    ++cnt;
                }
                if (vis.add((x - 1) * M + y)) {
                    ++res[cnt];
                }
            }
            // 右上
            if (x + 1 < m && y > 0) {
                int cnt = 1;
                if (set.contains(x * M + y - 1)) {
                    ++cnt;
                }
                if (set.contains((x + 1) * M + y - 1)) {
                    ++cnt;
                }
                if (set.contains((x + 1) * M + y)) {
                    ++cnt;
                }
                if (vis.add(x * M + y - 1)) {
                    ++res[cnt];
                }
            }

            // 右下
            if (x + 1 < m && y + 1 < n) {
                int cnt = 1;
                if (set.contains(x * M + y + 1)) {
                    ++cnt;
                }
                if (set.contains((x + 1) * M + y + 1)) {
                    ++cnt;
                }
                if (set.contains((x + 1) * M + y)) {
                    ++cnt;
                }
                if (vis.add(x * M + y)) {
                    ++res[cnt];
                }
            }
        }
        res[0] = (long) (m - 1) * (n - 1) - vis.size();
        return res;

    }

    // 6451. 找出最大的可达成数字 (Find the Maximum Achievable Number)
    public int theMaximumAchievableX(int num, int t) {
        return num + t + t;
    }

    // 2770. 达到末尾下标所需的最大跳跃次数 (Maximum Number of Jumps to Reach the Last Index)
    private int[] memo2770;
    private int[] nums2770;
    private int target2770;
    private int n2770;

    public int maximumJumps(int[] nums, int target) {
        this.n2770 = nums.length;
        this.memo2770 = new int[n2770];
        Arrays.fill(memo2770, -1);
        this.nums2770 = nums;
        this.target2770 = target;
        int res = dfs2770(0);
        return res >= 0 ? res : -1;

    }

    private int dfs2770(int i) {
        if (i == n2770 - 1) {
            return 0;
        }
        if (memo2770[i] != -1) {
            return memo2770[i];
        }
        int max = Integer.MIN_VALUE;
        for (int j = i + 1; j < n2770; ++j) {
            if (Math.abs(nums2770[j] - nums2770[i]) <= target2770) {
                max = Math.max(max, dfs2770(j) + 1);
            }
        }
        return memo2770[i] = max;
    }
    // 2771. 构造最长非递减子数组 (Longest Non-decreasing Subarray From Two Arrays)
    private int n2771;
    private int[][] nums2771;
    private int[][] memo2771;

    public int maxNonDecreasingLength(int[] nums1, int[] nums2) {
        this.n2771 = nums1.length;
        this.nums2771 = new int[2][n2771];
        nums2771[0] = nums1;
        nums2771[1] = nums2;
        this.memo2771 = new int[2][n2771];
        int res = 0;
        for (int i = 0; i < n2771; ++i) {
            res = Math.max(res, dfs2771(0, i));
            res = Math.max(res, dfs2771(1, i));
        }
        return res;

    }

    private int dfs2771(int i, int j) {
        if (j == 0) {
            return 1;
        }
        if (memo2771[i][j] != 0) {
            return memo2771[i][j];
        }
        int max = 1;
        if (nums2771[0][j - 1] <= nums2771[i][j]) {
            max = dfs2771(0, j - 1) + 1;
        }
        if (nums2771[1][j - 1] <= nums2771[i][j]) {
            max = Math.max(max, dfs2771(1, j - 1) + 1);
        }
        return memo2771[i][j] = max;
    }

    // 979. 在二叉树中分配硬币 (Distribute Coins in Binary Tree)
    private int res979;

    public int distributeCoins(TreeNode root) {
        dfs979(root);
        return res979;
    }

    private int dfs979(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int l = dfs979(root.left);
        int r = dfs979(root.right);
        res979 += Math.abs(l) + Math.abs(r);
        return l + r + root.val - 1;
    }

    // 2778. 特殊元素平方和 (Sum of Squares of Special Elements)
    public int sumOfSquares(int[] nums) {
        int res = 0;
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            if (n % (i + 1) == 0) {
                res += nums[i] * nums[i];
            }
        }
        return res;

    }

    // 2779. 数组的最大美丽值 (Maximum Beauty of an Array After Applying Operation)
    public int maximumBeauty(int[] nums, int k) {
        TreeMap<Integer, Integer> map = new TreeMap<>();
        for (int num : nums) {
            int a = num - k;
            int b = num + k + 1;
            map.merge(a, -1, Integer::sum);
            map.merge(b, 1, Integer::sum);
        }
        int res = 0;
        int cur = 0;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            cur += entry.getValue();
            res = Math.max(res, Math.abs(cur));
        }
        return res;

    }

    // 2779. 数组的最大美丽值 (Maximum Beauty of an Array After Applying Operation)
    public int maximumBeauty2(int[] nums, int k) {
        int res = 0;
        Arrays.sort(nums);
        int left = 0;
        int right = 0;
        int n = nums.length;
        while (right < n) {
            while (nums[right] - nums[left] > 2 * k) {
                ++left;
            }
            res = Math.max(res, right - left + 1);
            ++right;
        }
        return res;

    }

    // 2780. 合法分割的最小下标 (Minimum Index of a Valid Split)
    public int minimumIndex(List<Integer> nums) {
        int n = nums.size();
        Map<Integer, Integer> cnts = new HashMap<>();
        for (int num : nums) {
            cnts.merge(num, 1, Integer::sum);
        }
        Map<Integer, Integer> cur = new HashMap<>();
        for (int i = 0; i < n - 1; ++i) {
            cur.merge(nums.get(i), 1, Integer::sum);
            cnts.merge(nums.get(i), -1, Integer::sum);
            if (cur.get(nums.get(i)) * 2 > i + 1 && cnts.get(nums.get(i)) * 2 > n - i - 1) {
                return i;
            }
        }
        return -1;

    }

    // 2781. 最长合法子字符串的长度 (Length of the Longest Valid Substring)
    public int longestValidSubstring(String word, List<String> forbidden) {
        Set<String> s = new HashSet<>(forbidden);
        int n = word.length();
        int l = 0;
        int r = 0;
        int res = 0;
        while (r < n) {
            int k = r;
            while (l <= k && k > r - 10) {
                if (s.contains(word.substring(k, r + 1))) {
                    l = k + 1;
                    break;
                }
                --k;
            }
            res = Math.max(res, r - l + 1);
            ++r;
        }
        return res;

    }

    // 656. 金币路径 --plus 未提交
    private List<Integer> res656;
    private int[] A656;
    private int B656;
    private int[] memo656;
    private int n656;

    public List<Integer> cheapestJump(int[] A, int B) {
        this.n656 = A.length;
        if (A[n656 - 1] == -1) {
            return List.of();
        }
        this.res656 = new ArrayList<>();
        this.A656 = A;
        this.B656 = B;
        this.memo656 = new int[n656];
        Arrays.fill(memo656, (int) 1e6);
        int min = dfs656(0);
        res656.add(0, 1);
        return min < (int) 1e6 ? res656 : List.of();

    }

    private int dfs656(int i) {
        if (i == n656 - 1) {
            // return 0;
            return A656[n656 - 1];
        }
        if (memo656[i] != (int) 1e6) {
            return memo656[i];
        }
        int min = (int) 1e6;
        int k = -1;
        for (int j = i + 1; j <= Math.min(i + B656, n656 - 1); ++j) {
            if (A656[j] == -1) {
                continue;
            }
            int cur = dfs656(j);
            if (cur < min) {
                min = cur;
                k = j;
            }
        }
        if (min < (int) 1e6) {
            res656.add(0, k + 1);
        }
        return memo656[i] = min + A656[i];
    }

    // 1928. 规定时间内到达终点的最小花费 (Minimum Cost to Reach Destination in Time)
    public int minCost(int maxTime, int[][] edges, int[] passingFees) {
        int n = passingFees.length;
        List<int[]>[] g = new ArrayList[n];
        Arrays.setAll(g, k -> new ArrayList<>());
        for (int[] e : edges) {
            int x = e[0];
            int y = e[1];
            int t = e[2];
            g[x].add(new int[] { y, t });
            g[y].add(new int[] { x, t });
        }

        // [x, t, c]
        Queue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[2], o2[2]);
            }

        });

        int[][] dis = new int[n][maxTime + 1];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(dis[i], Integer.MAX_VALUE);
        }
        dis[0][0] = passingFees[0];
        q.offer(new int[] { 0, 0, passingFees[0] });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int t = cur[1];
            int c = cur[2];
            if (x == n - 1) {
                return c;
            }
            for (int[] nei : g[x]) {
                int y = nei[0];
                int dt = nei[1];
                int nt = dt + t;
                if (nt > maxTime) {
                    continue;
                }
                int nc = passingFees[y] + c;
                if (nc < dis[y][nt]) {
                    dis[y][nt] = nc;
                    q.offer(new int[] { y, nt, nc });
                }
            }
        }
        return -1;

    }

    // 1388. 3n 块披萨 (Pizza With 3n Slices)
    private int n1388;
    private int[] nums1388;
    private int m1388;
    private int[][] memo1388;

    public int maxSizeSlices(int[] slices) {
        return Math.max(check1388(Arrays.copyOfRange(slices, 0, slices.length - 1)),
                check1388(Arrays.copyOfRange(slices, 1, slices.length)));
    }

    private int check1388(int[] nums) {
        this.nums1388 = nums;
        this.n1388 = nums.length;
        this.m1388 = (n1388 + 1) / 3;
        this.memo1388 = new int[n1388][m1388];
        return dfs1388(0, 0);
    }

    private int dfs1388(int i, int j) {
        if (j == m1388) {
            return 0;
        }
        if (i >= n1388) {
            return (int) -1e9;
        }
        if (memo1388[i][j] != 0) {
            return memo1388[i][j];
        }
        return memo1388[i][j] = Math.max(dfs1388(i + 1, j), dfs1388(i + 2, j + 1) + nums1388[i]);
    }

    // 6930. 检查数组是否是好的 (Check if Array is Good)
    public boolean isGood(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        for (int i = 0; i < n - 1; ++i) {
            if (nums[i] != i + 1) {
                return false;
            }
        }
        return nums[n - 1] == n - 1;

    }

    // 6926. 将字符串中的元音字母排序 (Sort Vowels in a String)
    public String sortVowels(String s) {
        List<Character> list = new ArrayList<>();
        char[] arr = s.toCharArray();
        String v = "aeiou";
        int m = 0;
        for (char c : v.toCharArray()) {
            m |= 1 << (c - 'a');
        }
        for (char c : s.toCharArray()) {
            if ((m & (1 << ((c & 31) - 1))) != 0) {
                list.add(c);
            }
        }
        Collections.sort(list);
        int j = 0;
        for (int i = 0; i < arr.length; ++i) {
            if ((m & (1 << ((arr[i] & 31) - 1))) != 0) {
                arr[i] = list.get(j++);
            }
        }
        return String.valueOf(arr);

    }

    // 6931. 访问数组中的位置使分数最大 (Visit Array Positions to Maximize Score)
    private long[] memo6931;
    private List<Long> list6931;
    private int x6931;
    private int len6931;

    public long maxScore(int[] nums, int x) {
        this.list6931 = new ArrayList<>();
        this.x6931 = x;
        int n = nums.length;
        int i = 0;
        while (i < n) {
            long cur = nums[i];
            long sum = 0L;
            int j = i;
            while (j < n && nums[j] % 2 == cur % 2) {
                sum += nums[j];
                ++j;
            }
            list6931.add(sum);
            i = j;
        }
        this.len6931 = list6931.size();
        this.memo6931 = new long[len6931];
        Arrays.fill(memo6922, (long) 1e12);
        return dfs6931(0);

    }

    private long dfs6931(int i) {
        if (i >= len6931) {
            return 0L;
        }
        if (memo6931[i] != (long) 1e12) {
            return memo6931[i];
        }
        return memo6931[i] = Math.max(dfs6931(i + 1) - x6931, dfs6931(i + 2)) + list6931.get(i);
    }

    // 6922. 将一个数字表示成幂的和的方案数 (Ways to Express an Integer as Sum of Powers)
    private int[][] memo6922;
    private int n6922;
    private int x6922;

    public int numberOfWays(int n, int x) {
        this.n6922 = n;
        this.x6922 = x;
        this.memo6922 = new int[n + 1][n + 1];
        for (int i = 0; i < n + 1; ++i) {
            Arrays.fill(memo6922[i], -1);
        }
        return dfs6922(1, 0);
    }

    private int dfs6922(int i, int sum) {
        if (sum == n6922) {
            return 1;
        }
        if (sum > n6922) {
            return 0;
        }
        if (Math.pow(i, x6922) > n6922) {
            return 0;
        }
        if (memo6922[i][sum] != -1) {
            return memo6922[i][sum];
        }
        final int M = (int) (1e9 + 7);
        return memo6922[i][sum] = (dfs6922(i + 1, sum) + dfs6922(i + 1, sum + (int) Math.pow(i, x6922))) % M;
    }

    // 2788. 按分隔符拆分字符串 (Split Strings by Separator)
    public List<String> splitWordsBySeparator(List<String> words, char separator) {
        List<String> res = new ArrayList<>();
        for (String s : words) {
            String[] split = s.split("\\" + separator);
            for (String sub : split) {
                if (!sub.isEmpty()) {
                    res.add(sub);
                }
            }
        }
        return res;

    }

    // 2789. 合并后数组中的最大元素 (Largest Element in an Array after Merge Operations)
    public long maxArrayValue(int[] nums) {
        int n = nums.length;
        long res = 0L;
        long cur = 0L;
        for (int i = n - 1; i >= 0; --i) {
            if (cur < nums[i]) {
                cur = 0L;
            }
            cur += nums[i];
            res = Math.max(res, cur);
        }
        return res;

    }

    // 6942. 树中可以形成回文的路径数 (Count Paths That Can Form a Palindrome in a Tree)
    private Map<Integer, List<int[]>> g6942;
    private Map<Integer, Integer> cnts6942;
    private long res6942;

    public long countPalindromePaths(List<Integer> parent, String s) {
        int n = parent.size();
        g6942 = new HashMap<>();
        for (int i = 1; i < n; ++i) {
            g6942.computeIfAbsent(parent.get(i), k -> new ArrayList<>()).add(new int[] { i, 1 << (s.charAt(i) - 'a') });
        }
        this.cnts6942 = new HashMap<>();
        dfs6942(0, 0);
        return res6942;

    }

    private void dfs6942(int x, int xor) {
        res6942 += cnts6942.getOrDefault(xor, 0);
        for (int i = 0; i < 26; ++i) {
            res6942 += cnts6942.getOrDefault(xor ^ (1 << i), 0);
        }
        cnts6942.merge(xor, 1, Integer::sum);
        for (int[] nei : g6942.getOrDefault(x, new ArrayList<>())) {
            int y = nei[0];
            int m = nei[1];
            dfs6942(y, xor ^ m);
        }
    }

    // 771. 宝石与石头 (Jewels and Stones)
    public int numJewelsInStones(String jewels, String stones) {
        int[] m = new int[2];
        for (char c : jewels.toCharArray()) {
            m[(c >> 5) & 1] |= 1 << ((c & 31) - 1);
        }
        int res = 0;
        for (char c : stones.toCharArray()) {
            res += m[(c >> 5) & 1] >> ((c & 31) - 1) & 1;
        }
        return res;
    }

    // 6917.满足目标工作时长的员工数目 (Number of Employees Who Met the Target)
    public int numberOfEmployeesWhoMetTarget(int[] hours, int target) {
        int res = 0;
        for (int h : hours) {
            if (h >= target) {
                ++res;
            }
        }
        return res;

    }

    // 2799. 统计完全子数组的数目 (Count Complete Subarrays in an Array)
    public int countCompleteSubarrays(int[] nums) {
        int[] cnts = new int[2001];
        int s = 0;
        for (int num : nums) {
            if (++cnts[num] == 1) {
                ++s;
            }
        }
        Arrays.fill(cnts, 0);
        int i = 0;
        int j = 0;
        int n = nums.length;
        int cur = 0;
        int res = 0;
        while (i < n) {
            if (++cnts[nums[i]] == 1) {
                ++cur;
            }
            while (cur == s) {
                res += n - i;
                if (--cnts[nums[j++]] == 0) {
                    --cur;
                }
                ++j;
            }
            ++i;
        }
        return res;

    }

    // 6918. 包含三个字符串的最短字符串 (Shortest String That Contains Three Strings)
    public String minimumString(String a, String b, String c) {
        String[] arr = new String[3];
        arr[0] = a;
        arr[1] = b;
        arr[2] = c;
        String res = "";
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    if (i != j && i != k && j != k) {
                        String cur = join6918(join6918(arr[i], arr[j]), arr[k]);
                        if (res.isEmpty()) {
                            res = cur;
                        } else if (cur.length() < res.length()) {
                            res = cur;
                        } else if (cur.length() == res.length() && cur.compareTo(res) < 0) {
                            res = cur;
                        }
                    }
                }
            }
        }
        return res;

    }

    private String join6918(String a, String b) {
        if (a.contains(b)) {
            return a;
        }
        for (int i = 0; i < a.length(); ++i) {
            if (b.startsWith(a.substring(i))) {
                return a.substring(0, i) + b;
            }
        }
        return a + b;
    }

    // 6957. 统计范围内的步进数字数目 (Count Stepping Numbers in Range)
    public int countSteppingNumbers(String low, String high) {
        final int MOD = (int) (1e9 + 7);
        return ((cal6957(high.toCharArray()) - cal6957(low.toCharArray()) + (isValid6957(low.toCharArray()) ? 1 : 0))
                % MOD + MOD) % MOD;
    }

    private boolean isValid6957(char[] arr) {
        for (int i = 1; i < arr.length; ++i) {
            if (Math.abs(arr[i] - arr[i - 1]) != 1) {
                return false;
            }
        }
        return true;
    }

    private char[] arr6957;
    private int[][] memo6957;
    private int n6957;

    private int cal6957(char[] arr) {
        this.arr6957 = arr;
        this.n6957 = arr.length;
        this.memo6957 = new int[n6957][10];
        for (int i = 0; i < n6957; ++i) {
            Arrays.fill(memo6957[i], -1);
        }
        return dfs6957(0, 0, true, false);
    }

    private int dfs6957(int i, int pre, boolean isLimit, boolean isNum) {
        if (i == n6957) {
            return isNum ? 1 : 0;
        }
        if (!isLimit && isNum && memo6957[i][pre] != -1) {
            return memo6957[i][pre];
        }
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        if (!isNum) {
            res = dfs6957(i + 1, pre, false, false);
        }
        int up = isLimit ? arr6957[i] - '0' : 9;
        for (int j = isNum ? 0 : 1; j <= up; ++j) {
            if (!isNum || Math.abs(j - pre) == 1) {
                res = (res + dfs6957(i + 1, j, isLimit && j == up, true)) % MOD;
            }
        }
        if (!isLimit && isNum) {
            memo6957[i][pre] = res;
        }
        return res;
    }

    // 2810. 故障键盘 (Faulty Keyboard)
    public String finalString(String s) {
        StringBuilder res = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (c == 'i') {
                res.reverse();
            } else {
                res.append(c);
            }
        }
        return res.toString();
    }

    // 2810. 故障键盘 (Faulty Keyboard)
    public String finalString2(String s) {
        boolean tail = true;
        StringBuilder res = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (c == 'i') {
                tail = !tail;
            } else if (tail) {
                res.append(c);
            } else {
                res.insert(0, c);
            }
        }
        return tail ? res.toString() : res.reverse().toString();
    }

    // 2811. 判断是否能拆分数组 (Check if it is Possible to Split Array)
    private int[] pre2811;
    private int n2811;
    private int m2811;
    private int[][] memo2811;

    public boolean canSplitArray(List<Integer> nums, int m) {
        this.n2811 = nums.size();
        if (n2811 <= 2) {
            return true;
        }
        this.pre2811 = new int[n2811 + 1];
        for (int i = 0; i < n2811; ++i) {
            pre2811[i + 1] = pre2811[i] + nums.get(i);
        }
        this.m2811 = m;
        this.memo2811 = new int[n2811][n2811];
        return dfs2811(0, n2811 - 1);

    }

    private boolean dfs2811(int i, int j) {
        if (i == j) {
            return true;
        }
        if (pre2811[j + 1] - pre2811[i] < m2811) {
            return false;
        }
        if (memo2811[i][j] != 0) {
            return memo2811[i][j] > 0;
        }
        for (int k = i; k < j; ++k) {
            if (dfs2811(i, k) && dfs2811(k + 1, j)) {
                memo2811[i][j] = 1;
                return true;
            }
        }
        memo2811[i][j] = -1;
        return false;
    }

    // 2811. 判断是否能拆分数组 (Check if it is Possible to Split Array)
    public boolean canSplitArray2(List<Integer> nums, int m) {
        int n = nums.size();
        if (n <= 2) {
            return true;
        }
        for (int i = 1; i < n; ++i) {
            if (nums.get(i - 1) + nums.get(i) >= m) {
                return true;
            }
        }
        return false;

    }

    // 2812. 找出最安全路径 (Find the Safest Path in a Grid)
    public int maximumSafenessFactor(List<List<Integer>> grid) {
        int n = grid.size();
        int[][] g = new int[n][n];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(g[i], -1);
        }
        Queue<int[]> q = new LinkedList<>();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid.get(i).get(j) == 1) {
                    g[i][j] = 0;
                    q.offer(new int[] { i, j });
                }
            }
        }
        int[][] dirs = { { 0, -1 }, { 1, 0 }, { -1, 0 }, { 0, 1 } };
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                int[] cur = q.poll();
                int x = cur[0];
                int y = cur[1];
                for (int[] d : dirs) {
                    int nx = x + d[0];
                    int ny = y + d[1];
                    if (nx >= 0 && nx < n && ny >= 0 && ny < n && g[nx][ny] == -1) {
                        g[nx][ny] = g[x][y] + 1;
                        q.offer(new int[] { nx, ny });
                    }
                }
            }
        }
        Queue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o2[2], o1[2]);
            }

        });

        boolean[][] vis = new boolean[n][n];
        pq.offer(new int[] { 0, 0, g[0][0] });
        while (!pq.isEmpty()) {
            int[] cur = pq.poll();
            int x = cur[0];
            int y = cur[1];
            int d = cur[2];
            if (vis[x][y]) {
                continue;
            }
            vis[x][y] = true;
            if (x == n - 1 && y == n - 1) {
                return d;
            }
            for (int[] dir : dirs) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                    pq.offer(new int[] { nx, ny, Math.min(d, g[nx][ny]) });
                }
            }
        }
        return -1;

    }

    // 2812. 找出最安全路径 (Find the Safest Path in a Grid)
    private int[][] g2812;
    private int n2812;

    public int maximumSafenessFactor2(List<List<Integer>> grid) {
        this.n2812 = grid.size();
        if (n2812 == 1) {
            return 0;
        }
        this.g2812 = new int[n2812][n2812];
        Queue<int[]> q = new LinkedList<>();
        for (int i = 0; i < n2812; ++i) {
            for (int j = 0; j < n2812; ++j) {
                if (grid.get(i).get(j) == 1) {
                    g2812[i][j] = 0;
                    q.offer(new int[] { i, j });
                } else {
                    g2812[i][j] = -1;
                }
            }
        }
        int[][] dirs = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                int[] cur = q.poll();
                int x = cur[0];
                int y = cur[1];
                for (int[] d : dirs) {
                    int nx = x + d[0];
                    int ny = y + d[1];
                    if (nx >= 0 && nx < n2812 && ny >= 0 && ny < n2812 && g2812[nx][ny] == -1) {
                        g2812[nx][ny] = g2812[x][y] + 1;
                        q.offer(new int[] { nx, ny });
                    }
                }
            }
        }
        int left = 0;
        int right = n2812;
        int res = 0;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (check2812(mid)) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    private boolean check2812(int target) {
        Union2812 union = new Union2812(n2812 * n2812);
        for (int i = 0; i < n2812; ++i) {
            for (int j = 0; j < n2812; ++j) {
                if (g2812[i][j] >= target) {
                    if (i + 1 < n2812 && g2812[i + 1][j] >= target) {
                        union.union(getIndex2812(i, j), getIndex2812(i + 1, j));
                    }
                    if (j + 1 < n2812 && g2812[i][j + 1] >= target) {
                        union.union(getIndex2812(i, j), getIndex2812(i, j + 1));
                    }
                }
            }
        }
        return union.isConnected(getIndex2812(0, 0), getIndex2812(n2812 - 1, n2812 - 1));
    }

    private int getIndex2812(int i, int j) {
        return i * n2812 + j;
    }

    public class Union2812 {
        private int[] rank;
        private int[] parent;

        public Union2812(int n) {
            this.rank = new int[n];
            this.parent = new int[n];
            for (int i = 0; i < n; ++i) {
                rank[i] = 1;
                parent[i] = i;
            }
        }

        public int getRoot(int p) {
            if (parent[p] == p) {
                return p;
            }
            return parent[p] = getRoot(parent[p]);
        }

        public boolean isConnected(int p1, int p2) {
            return getRoot(p1) == getRoot(p2);
        }

        public void union(int p1, int p2) {
            int root1 = getRoot(p1);
            int root2 = getRoot(p2);
            if (root1 == root2) {
                return;
            }
            if (rank[root1] < rank[root2]) {
                parent[root1] = root2;
            } else {
                parent[root2] = root1;
                if (rank[root1] == rank[root2]) {
                    ++rank[root1];
                }
            }
        }

    }

    // 2812. 找出最安全路径 (Find the Safest Path in a Grid)
    public int maximumSafenessFactor3(List<List<Integer>> grid) {
        this.n2812 = grid.size();
        List<List<int[]>> groups = new ArrayList<>();
        int[][] g = new int[n2812][n2812];
        Queue<int[]> q = new LinkedList<>();
        for (int i = 0; i < n2812; ++i) {
            for (int j = 0; j < n2812; ++j) {
                if (grid.get(i).get(j) == 1) {
                    q.offer(new int[] { i, j });
                    g[i][j] = 0;
                } else {
                    g[i][j] = -1;
                }
            }
        }
        groups.add(new ArrayList<>(q));
        int[][] dirs = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                int[] cur = q.poll();
                int x = cur[0];
                int y = cur[1];
                for (int[] d : dirs) {
                    int nx = x + d[0];
                    int ny = y + d[1];
                    if (nx >= 0 && nx < n2812 && ny >= 0 && ny < n2812 && g[nx][ny] == -1) {
                        q.offer(new int[] { nx, ny });
                        g[nx][ny] = g[x][y] + 1;
                    }
                }
            }
            groups.add(new ArrayList<>(q));
        }
        Union2812 union = new Union2812(n2812 * n2812);
        for (int i = groups.size() - 2; i >= 0; --i) {
            for (int[] cur : groups.get(i)) {
                int x = cur[0];
                int y = cur[1];
                for (int[] d : dirs) {
                    int nx = x + d[0];
                    int ny = y + d[1];
                    if (nx >= 0 && nx < n2812 && ny >= 0 && ny < n2812 && g[nx][ny] >= i) {
                        union.union(getIndex2812(x, y), getIndex2812(nx, ny));
                    }
                }
            }
            if (union.isConnected(getIndex2812(0, 0), getIndex2812(n2812 - 1, n2812 - 1))) {
                return i;
            }
        }
        return 0;
    }

    // 344. 反转字符串 (Reverse String)
    public void reverseString(char[] s) {
        int i = 0;
        int j = s.length - 1;
        while (i < j) {
            char t = s[i];
            s[i] = s[j];
            s[j] = t;
            ++i;
            --j;
        }
    }

    // 2806. 取整购买后的账户余额 (Account Balance After Rounded Purchase)
    public int accountBalanceAfterPurchase(int purchaseAmount) {
        return 100 - ((purchaseAmount + 5) / 10) * 10;
    }

    // 2807. 在链表中插入最大公约数 (Insert Greatest Common Divisors in Linked List)
    public ListNode insertGreatestCommonDivisors(ListNode head) {
        ListNode cur = head;
        while (cur != null && cur.next != null) {
            ListNode insert = new ListNode(gcd2807(cur.val, cur.next.val));
            insert.next = cur.next;
            cur.next = insert;
            cur = insert.next;
        }
        return head;

    }

    private int gcd2807(int a, int b) {
        return b == 0 ? a : gcd2807(b, a % b);
    }

    // 2808. 使循环数组所有元素相等的最少秒数 (Minimum Seconds to Equalize a Circular Array)
    public int minimumSeconds(List<Integer> nums) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        int n = nums.size();
        for (int i = 0; i < n; ++i) {
            map.computeIfAbsent(nums.get(i), k -> new ArrayList<>()).add(i);
        }
        int res = Integer.MAX_VALUE;
        for (List<Integer> items : map.values()) {
            int cur = 0;
            for (int i = 1; i < items.size(); ++i) {
                cur = Math.max(cur, (items.get(i) - items.get(i - 1)) / 2);
            }
            cur = Math.max(cur, (n - items.get(items.size() - 1) + items.get(0)) / 2);
            res = Math.min(res, cur);
        }
        return res;

    }

    // 6939. 数组中的最大数对和 (Max Pair Sum in an Array)
    public int maxSum(int[] nums) {
        int res = -1;
        int[] maxBit = new int[10];
        Arrays.fill(maxBit, Integer.MIN_VALUE);
        for (int num : nums) {
            int bit = getMaxBit6939(num);
            res = Math.max(res, num + maxBit[bit]);
            maxBit[bit] = Math.max(maxBit[bit], num);
        }
        return res;
    }

    private int getMaxBit6939(int num) {
        int max = 0;
        while (num != 0) {
            max = Math.max(max, num % 10);
            num /= 10;
        }
        return max;
    }

    // 6914. 翻倍以链表形式表示的数字 (Double a Number Represented as a Linked List)
    public ListNode doubleIt(ListNode head) {
        head = reverseList6914(head);
        int carry = 0;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode cur = head;
        ListNode res = dummy;
        while (cur != null || carry != 0) {
            int val = carry;
            if (cur != null) {
                val += cur.val * 2;
            }
            res.next = new ListNode(val % 10);
            carry = val / 10;
            res = res.next;
            if (cur != null) {
                cur = cur.next;
            }
        }
        return reverseList6914(dummy.next);

    }

    private ListNode reverseList6914(ListNode head) {
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

    // 7022. 限制条件下元素之间的最小绝对差 (Minimum Absolute Difference Between Elements With
    // Constraint)
    public int minAbsoluteDifference(List<Integer> nums, int x) {
        int n = nums.size();
        TreeSet<Integer> set = new TreeSet<>();
        set.add(Integer.MAX_VALUE);
        set.add(Integer.MIN_VALUE / 2);
        int res = (int) 1e9;
        for (int i = x; i < n; ++i) {
            set.add(nums.get(i - x));
            res = Math.min(res, nums.get(i) - set.floor(nums.get(i)));
            res = Math.min(res, set.ceiling(nums.get(i)) - nums.get(i));
        }
        return res;

    }

    // 2818. 操作使得分最大 (Apply Operations to Maximize Score)
    public int maximumScore(List<Integer> nums, int k) {
        int n = nums.size();
        int[] omega = new int[(int) 1e5 + 1];
        for (int i = 2; i < omega.length; ++i) {
            if (omega[i] == 0) {
                for (int j = i; j < omega.length; j += i) {
                    ++omega[j];
                }
            }
        }
        int[] left = new int[n];
        Arrays.fill(left, -1);
        int[] right = new int[n];
        Arrays.fill(right, n);
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < n; ++i) {
            while (!stack.isEmpty() && omega[nums.get(stack.peek())] < omega[nums.get(i)]) {
                right[stack.pop()] = i;
            }
            if (!stack.isEmpty()) {
                left[i] = stack.peek();
            }
            stack.push(i);
        }
        Integer[] ids = IntStream.range(0, n).boxed().toArray(Integer[]::new);
        Arrays.sort(ids, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(nums.get(o2), nums.get(o1));
            }

        });
        final int MOD = (int) (1e9 + 7);
        int res = 1;
        for (int id : ids) {
            int x = nums.get(id);
            int total = (right[id] - id) * (id - left[id]);
            if (total >= k) {
                res = (int) (((long) res * pow2818(x, k)) % MOD);
                break;
            }
            res = (int) (((long) res * pow2818(x, total)) % MOD);
            k -= total;
        }
        return res;

    }

    private int pow2818(int a, int b) {
        if (b == 0) {
            return 1;
        }
        final int MOD = (int) (1e9 + 7);
        int res = pow2818(a, b >> 1);
        res = (int) (((long) res * res) % MOD);
        if ((b & 1) == 1) {
            res = (int) (((long) res * a) % MOD);
        }
        return res;
    }

    // 6954. 统计和小于目标的下标对数目 (Count Pairs Whose Sum is Less than Target)
    public int countPairs(List<Integer> nums, int target) {
        Collections.sort(nums);
        int res = 0;
        int i = 0;
        int j = nums.size() - 1;
        while (i < j) {
            if (nums.get(i) + nums.get(j) >= target) {
                --j;
            } else {
                res += j - i;
                ++i;
            }
        }
        return res;
    }

    // 2825. 循环增长使字符串子序列等于另一个字符串 (Make String a Subsequence Using Cyclic Increments)
    public boolean canMakeSubsequence(String str1, String str2) {
        int n1 = str1.length();
        int n2 = str2.length();
        int i = 0;
        int j = 0;
        while (i < n1 && j < n2) {
            if (str1.charAt(i) == str2.charAt(j)
                    || ((char) ((str1.charAt(i) - 'a' + 1) % 26) + 'a') == str2.charAt(j)) {
                ++j;
            }
            ++i;
        }
        return j == n2;

    }

    // 6941. 将三个组排序 (Sorting Three Groups)
    private int n6941;
    private List<Integer> nums6941;
    private int[][] memo6941;

    public int minimumOperations(List<Integer> nums) {
        this.n6941 = nums.size();
        this.nums6941 = nums;
        this.memo6941 = new int[n6941][4];
        for (int i = 0; i < n6941; ++i) {
            Arrays.fill(memo6941[i], -1);
        }
        return dfs6941(0, 1);

    }

    private int dfs6941(int i, int j) {
        if (i == n6941) {
            return 0;
        }
        if (memo6941[i][j] != -1) {
            return memo6941[i][j];
        }
        int res = Integer.MAX_VALUE;
        for (int k = j; k <= 3; ++k) {
            if (nums6941.get(i) != k) {
                // 改
                res = Math.min(res, dfs6941(i + 1, k) + 1);
            } else {
                // 不改
                res = Math.min(res, dfs6941(i + 1, k));
            }
        }
        return memo6941[i][j] = res;
    }

    // 8013. 范围中美丽整数的数目 (Number of Beautiful Integers in the Range)
    private int k8013;

    public int numberOfBeautifulIntegers(int low, int high, int k) {
        this.k8013 = k;
        return check8013(high) - check8013(low - 1);

    }

    private char[] arr8013;
    private int[][][][] memo8013;
    private int n8013;

    private int check8013(int num) {
        arr8013 = String.valueOf(num).toCharArray();
        this.n8013 = arr8013.length;
        this.memo8013 = new int[n8013][n8013][n8013][20];
        for (int i = 0; i < n8013; ++i) {
            for (int j = 0; j < n8013; ++j) {
                for (int k = 0; k < n8013; ++k) {
                    Arrays.fill(memo8013[i][j][k], -1);
                }
            }
        }
        return dfs8013(0, 0, 0, 0, true, false);
    }

    private int dfs8013(int i, int even, int odd, int m, boolean isLimit, boolean isNum) {
        if (i == n8013) {
            return (isNum && even == odd && m % k8013 == 0) ? 1 : 0;
        }
        if (!isLimit && isNum && memo8013[i][even][odd][m] != -1) {
            return memo8013[i][even][odd][m];
        }
        int res = 0;
        if (!isNum) {
            res = dfs8013(i + 1, even, odd, m, false, false);
        }
        int up = isLimit ? arr8013[i] - '0' : 9;
        for (int j = isNum ? 0 : 1; j <= up; ++j) {
            res += dfs8013(i + 1, even + (j % 2 == 0 ? 1 : 0),
                    odd + (j % 2 == 1 ? 1 : 0), (m * 10 + j) % k8013, isLimit && j == up, true);
        }
        if (!isLimit && isNum) {
            memo8013[i][even][odd][m] = res;
        }
        return res;
    }

    // 7004. 判别首字母缩略词 (Check if a String Is an Acronym of Words)
    public boolean isAcronym(List<String> words, String s) {
        int n1 = words.size();
        int n2 = s.length();
        if (n1 != n2) {
            return false;
        }
        for (int i = 0; i < n1; ++i) {
            if (s.charAt(i) != words.get(i).charAt(0)) {
                return false;
            }
        }
        return true;

    }

    // 6450. k-avoiding 数组的最小总和 (Determine the Minimum Sum of a k-avoiding Array)
    public int minimumSum(int n, int k) {
        if (n == 1) {
            return 1;
        }
        int cnt = 0;
        int res = 0;
        for (int i = 1; i <= k / 2 && cnt < n; ++i) {
            res += i;
            ++cnt;
        }
        while (cnt++ < n) {
            res += k++;
        }
        return res;

    }

    // 7006. 销售利润最大化 (Maximize the Profit as the Salesman)
    private Map<Integer, List<int[]>> map7006;
    private int[] memo7006;

    public int maximizeTheProfit(int n, List<List<Integer>> offers) {
        this.map7006 = new HashMap<>();
        for (List<Integer> offer : offers) {
            map7006.computeIfAbsent(offer.get(1), k -> new ArrayList<>()).add(new int[] { offer.get(0), offer.get(2) });
        }
        this.memo7006 = new int[n];
        Arrays.fill(memo7006, -1);
        return dfs7006(n - 1);

    }

    private int dfs7006(int i) {
        if (i < 0) {
            return 0;
        }
        if (memo7006[i] != -1) {
            return memo7006[i];
        }
        // 不卖
        int res = dfs7006(i - 1);
        // 卖
        for (int[] j : map7006.getOrDefault(i, new ArrayList<>())) {
            res = Math.max(res, dfs7006(j[0] - 1) + j[1]);
        }
        return memo7006[i] = res;
    }

    // 6467. 找出最长等值子数组 (Find the Longest Equal Subarray)
    public int longestEqualSubarray(List<Integer> nums, int k) {
        int n = nums.size();
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            map.computeIfAbsent(nums.get(i), o -> new ArrayList<>()).add(i);
        }
        int res = 1;
        for (List<Integer> list : map.values()) {
            int curK = 0;
            int i = 0;
            int j = 0;
            int m = list.size();
            while (j < m) {
                if (j > i) {
                    curK += list.get(j) - list.get(j - 1) - 1;
                }
                while (curK > k) {
                    curK -= list.get(i + 1) - list.get(i) - 1;
                    ++i;
                }
                res = Math.max(res, list.get(j) - list.get(i) + 1 - curK);
                ++j;
            }
        }
        return res;

    }

    // 2833. 距离原点最远的点 (Furthest Point From Origin)
    public int furthestDistanceFromOrigin(String moves) {
        int l = 0;
        int r = 0;
        int line = 0;
        for (char c : moves.toCharArray()) {
            if (c == 'L') {
                ++l;
            } else if (c == 'R') {
                ++r;
            } else {
                ++line;
            }
        }
        return Math.abs(r - l) + line;
    }

    // 2834. 找出美丽数组的最小和 (Find the Minimum Possible Sum of a Beautiful Array)
    public long minimumPossibleSum(int n, int target) {
        long res = 0L;
        int cnt = 0;
        for (int i = 1; i <= target / 2 && cnt < n; ++i) {
            res += i;
            ++cnt;
        }
        while (cnt++ < n) {
            res += target++;
        }
        return res;

    }

    // 1654. 到家的最少跳跃次数 (Minimum Jumps to Reach Home)
    public int minimumJumps(int[] forbidden, int a, int b, int x) {
        Set<Integer> blocked = Arrays.stream(forbidden).boxed().collect(Collectors.toSet());
        boolean[][] vis = new boolean[6000][2];
        int res = 0;
        Queue<int[]> q = new LinkedList<>();
        q.offer(new int[] { 0, 0 });
        vis[0][0] = true;
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                int[] cur = q.poll();
                int node = cur[0];
                int leftSteps = cur[1];
                if (node == x) {
                    return res;
                }
                // 右
                if (node + a < vis.length && !vis[node + a][0] && !blocked.contains(node + a)) {
                    vis[node + a][0] = true;
                    q.offer(new int[] { node + a, 0 });
                }
                // 左
                if (leftSteps == 0) {
                    if (node - b >= 0 && !vis[node - b][1] && !blocked.contains(node - b)) {
                        vis[node - b][1] = true;
                        q.offer(new int[] { node - b, 1 });
                    }
                }
            }
            ++res;
        }
        return -1;

    }

    /**
     * 问题：
     * https://atcoder.jp/contests/abc233/tasks/abc233_d
     * 
     * 输入 n(1≤n≤2e5) k(-1e15≤k≤1e15) 和长为 n 的数组 a(-1e9≤a[i]≤1e9)。
     * 输出元素和等于 k 的连续子数组个数。
     * 
     * 如果你觉得本题太简单，请思考这个问题：
     * 所有元素和等于 k 的连续子数组的长度之和。
     */
    /**
     * 解答：
     * 用前缀和思考：
     * sum[R] - sum[L] = k
     * 枚举 R，问题变成有多少个 sum[L]，也就是 sum[R]-k 的个数。
     * 这可以用哈希表统计，代码如下。
     * 
     * https://atcoder.jp/contests/abc233/submissions/44903090
     * 
     * 关于思考题的提示：
     * 举例：
     * (R-L1) + (R-L2) + (R-L3)
     * = 3*R - (L1+L2+L3)
     * 所以除了维护前缀和的出现次数，还需要维护下标之和。
     */
    public int contests_abc233(int[] nums, int k) {
        Map<Integer, int[]> map = new HashMap<>();
        map.put(0, new int[] { 1, 0 });
        int pre = 0;
        int n = nums.length;
        int res = 0;
        for (int i = 0; i < n; ++i) {
            pre += nums[i];
            int[] cur = map.getOrDefault(pre - k, new int[] { 0, 0 });
            res += cur[0] * (i + 1) - cur[1];
            int[] arr = map.getOrDefault(pre, new int[] { 0, 0 });
            ++arr[0];
            arr[1] += i;
            map.put(pre, arr);
        }
        return res;
    }

    // 2839. 判断通过操作能否让字符串相等 I (Check if Strings Can be Made Equal With Operations I)
    // 2840. 判断通过操作能否让字符串相等 II (Check if Strings Can be Made Equal With Operations
    // II)
    public boolean checkStrings(String s1, String s2) {
        int[] cnt = new int[26];
        for (int i = 0; i < s1.length(); i += 2) {
            ++cnt[s1.charAt(i) - 'a'];
            --cnt[s2.charAt(i) - 'a'];
        }
        for (int c : cnt) {
            if (c != 0) {
                return false;
            }
        }
        for (int i = 1; i < s1.length(); i += 2) {
            ++cnt[s1.charAt(i) - 'a'];
            --cnt[s2.charAt(i) - 'a'];
        }
        for (int c : cnt) {
            if (c != 0) {
                return false;
            }
        }
        return true;

    }

    // 2841. 几乎唯一子数组的最大和 (Maximum Sum of Almost Unique Subarray)
    public long maxSum(List<Integer> nums, int m, int k) {
        long res = 0L;
        long cur = 0L;
        int kinds = 0;
        int n = nums.size();
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            cur += nums.get(i);
            map.merge(nums.get(i), 1, Integer::sum);
            if (map.get(nums.get(i)) == 1) {
                ++kinds;
            }
            if (i >= k) {
                cur -= nums.get(i - k);
                map.merge(nums.get(i - k), -1, Integer::sum);
                if (map.get(nums.get(i - k)) == 0) {
                    --kinds;
                }
            }
            if (i >= k - 1) {
                if (kinds >= m) {
                    res = Math.max(res, cur);
                }
            }
        }
        return res;
    }

    // 2843. 统计对称整数的数目 (Count Symmetric Integers)
    public int countSymmetricIntegers(int low, int high) {
        int res = 0;
        for (int i = low; i <= high; ++i) {
            if (check2843(i)) {
                ++res;
            }
        }
        return res;

    }

    private boolean check2843(int num) {
        String s = String.valueOf(num);
        int n = s.length();
        if (n % 2 == 1) {
            return false;
        }
        int sum = 0;
        for (int i = 0; i < n / 2; ++i) {
            sum += s.charAt(i) - '0';
        }
        for (int i = n / 2; i < n; ++i) {
            sum -= s.charAt(i) - '0';
        }
        return sum == 0;
    }

    // 2844. 生成特殊数字的最少操作 (Minimum Operations to Make a Special Number)
    private int n2844;
    private String num2844;

    public int minimumOperations(String num) {
        this.n2844 = num.length();
        this.num2844 = num;
        int res = n2844;
        if (num.contains("0")) {
            res = n2844 - 1;
        }
        res = Math.min(res, check2844("00"));
        res = Math.min(res, check2844("25"));
        res = Math.min(res, check2844("50"));
        res = Math.min(res, check2844("75"));
        return res;
    }

    private int check2844(String s) {
        int i = num2844.lastIndexOf("" + s.charAt(1));
        if (i < 0) {
            return n2844;
        }
        int j = num2844.substring(0, i).lastIndexOf("" + s.charAt(0));
        if (j < 0) {
            return n2844;
        }
        return n2844 - j - 2;
    }

    // 2845. 统计趣味子数组的数目 (Count of Interesting Subarrays)
    public long countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        long res = 0L;
        int pre = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for (int num : nums) {
            pre += num % modulo == k ? 1 : 0;
            pre %= modulo;
            int curMod = ((pre - k) % modulo + modulo) % modulo;
            res += map.getOrDefault(curMod, 0);
            map.merge(pre, 1, Integer::sum);
        }
        return res;
    }

    // 8029. 与车相交的点 (Points That Intersect With Cars)
    public int numberOfPoints(List<List<Integer>> nums) {
        int[] d = new int[102];
        for (List<Integer> num : nums) {
            int a = num.get(0);
            int b = num.get(1);
            ++d[a];
            --d[b + 1];
        }
        int res = 0;
        int cur = 0;
        for (int i : d) {
            cur += i;
            if (cur > 0) {
                ++res;
            }
        }
        return res;

    }

    // 8049. 判断能否在给定时间到达单元格 (Determine if a Cell Is Reachable at a Given Time)
    public boolean isReachableAtTime(int sx, int sy, int fx, int fy, int t) {
        int max = Math.max(Math.abs(sx - fx), Math.abs(sy - fy));
        if (max == 0 && t == 1) {
            return false;
        }
        return max <= t;

    }

    // 2850. 将石头分散到网格图的最少移动次数 (Minimum Moves to Spread Stones Over Grid)
    private List<int[]> give2850;
    private List<int[]> need2850;
    private int[][] memo2850;
    private int u2850;
    private int[][] grid2850;

    public int minimumMoves(int[][] grid) {
        this.grid2850 = grid;
        this.give2850 = new ArrayList<>();
        this.need2850 = new ArrayList<>();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (grid[i][j] != 1) {
                    if (grid[i][j] > 1) {
                        give2850.add(new int[] { i, j });
                    } else {
                        need2850.add(new int[] { i, j });
                    }
                }
            }
        }
        int m = give2850.size();
        int n = need2850.size();
        this.memo2850 = new int[m][1 << n];
        this.u2850 = (1 << n) - 1;
        for (int i = 0; i < m; ++i) {
            Arrays.fill(memo2850[i], -1);
        }
        return dfs2850(0, 0);

    }

    private int dfs2850(int i, int j) {
        if (j == u2850) {
            return 0;
        }
        if (memo2850[i][j] != -1) {
            return memo2850[i][j];
        }
        int min = (int) 1e9;
        int[] p = give2850.get(i);
        int val = grid2850[p[0]][p[1]] - 1;
        int candidate = j ^ u2850;
        for (int c = candidate; c > 0; c = (c - 1) & candidate) {
            if (Integer.bitCount(c) == val) {
                int copy = c;
                int dis = 0;
                while (copy != 0) {
                    int index = Integer.numberOfTrailingZeros(copy);
                    int[] p2 = need2850.get(index);
                    dis += Math.abs(p2[0] - p[0]) + Math.abs(p2[1] - p[1]);
                    copy &= copy - 1;
                }
                min = Math.min(min, dfs2850(i + 1, j | c) + dis);
            }
        }
        return memo2850[i][j] = min;
    }

    // 8039. 使数组成为递增数组的最少右移次数 (Minimum Right Shifts to Sort the Array)
    public int minimumRightShifts(List<Integer> nums) {
        int n = nums.size();
        int cnt = 0;
        int j = -1;
        for (int i = 1; i < n; ++i) {
            if (nums.get(i - 1) > nums.get(i)) {
                j = i;
                ++cnt;
            }
        }
        if (cnt == 0) {
            return 0;
        }
        if (nums.get(0) < nums.get(n - 1)) {
            ++cnt;
        }
        if (cnt > 1) {
            return -1;
        }
        return n - j;

    }

    // 2856. 删除数对后的最小数组长度 (Minimum Array Length After Pair Removals)
    public int minLengthAfterRemovals(List<Integer> nums) {
        Map<Integer, Integer> cnts = new HashMap<>();
        int max = 0;
        for (int num : nums) {
            cnts.merge(num, 1, Integer::sum);
            max = Math.max(max, cnts.get(num));
        }
        int n = nums.size();
        if (max >= n - max) {
            return max - (n - max);
        }
        return n & 1;

    }

    // 2858. 可以到达每一个节点的最少边反转次数 (Minimum Edge Reversals So Every Node Is Reachable)
    private List<int[]>[] g2858;
    private int[] res2858;

    public int[] minEdgeReversals(int n, int[][] edges) {
        this.g2858 = new ArrayList[n];
        Arrays.setAll(g2858, k -> new ArrayList<>());
        for (int[] e : edges) {
            g2858[e[0]].add(new int[] { e[1], 0 });
            g2858[e[1]].add(new int[] { e[0], 1 });
        }
        this.res2858 = new int[n];
        res2858[0] = dfs2858(0, -1);
        reRoot2858(0, -1);
        return res2858;

    }

    private void reRoot2858(int x, int fa) {
        for (int[] nei : g2858[x]) {
            int y = nei[0];
            int c = nei[1];
            if (y != fa) {
                res2858[y] = res2858[x] + (-2) * c + 1;
                reRoot2858(y, x);
            }
        }
    }

    private int dfs2858(int x, int fa) {
        int cnt = 0;
        for (int[] nei : g2858[x]) {
            int y = nei[0];
            int c = nei[1];
            if (y != fa) {
                cnt += dfs2858(y, x) + c;
            }
        }
        return cnt;
    }


    // 6988. 统计距离为 k 的点对 (Count Pairs of Points With Distance k)
    public int countPairs2(List<List<Integer>> coordinates, int k) {
        int res = 0;
        long M = (long) 1e7;
        Map<Long, Integer> map = new HashMap<>();
        for (List<Integer> c : coordinates) {
            int x = c.get(0);
            int y = c.get(1);
            for (int i = 0; i <= k; ++i) {
                int tx = x ^ i;
                int ty = y ^ (k - i);
                res += map.getOrDefault(tx * M + ty, 0);
            }
            map.merge(x * M + y, 1, Integer::sum);
        }
        return res;

    }

    // 100031. 计算 K 置位下标对应元素的和 (Sum of Values at Indices With K Set Bits)
    public int sumIndicesWithKSetBits(List<Integer> nums, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); ++i) {
            if (Integer.bitCount(i) == k) {
                res += nums.get(i);
            }
        }
        return res;
    }

    // 100033. 最大合金数 (Maximum Number of Alloys)
    public int maxNumberOfAlloys(int n, int k, int budget, List<List<Integer>> composition, List<Integer> stock,
            List<Integer> cost) {
        int res = 0;
        for (List<Integer> c : composition) {
            res = Math.max(res, getMax(c, stock, cost, budget));
        }
        return res;

    }

    private int getMax(List<Integer> list, List<Integer> stock, List<Integer> cost, int budget) {
        int left = 0;
        int right = Integer.MAX_VALUE;
        int res = 0;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (check(mid, list, stock, cost, budget)) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    private boolean check(int target, List<Integer> composition, List<Integer> stock, List<Integer> cost, int budget) {
        int n = stock.size();
        long cur = 0L;
        for (int i = 0; i < n; ++i) {
            long items = (long) target * composition.get(i);
            if (items <= stock.get(i)) {
                continue;
            }
            items -= stock.get(i);
            cur += items * cost.get(i);
            if (cur > budget) {
                return false;
            }
        }
        return true;
    }

    // 100040. 让所有学生保持开心的分组方法数 (Happy Students)
    public int countWays(List<Integer> nums) {
        Collections.sort(nums);
        int n = nums.size();
        int res = nums.get(0) > 0 ? 1 : 0;
        for (int i = 0; i < n - 1; ++i) {
            if (nums.get(i) < i + 1 && i + 1 < nums.get(i + 1)) {
                ++res;
            }
        }
        return res + 1;

    }

    // 8048. 最大二进制奇数 (Maximum Odd Binary Number)
    public String maximumOddBinaryNumber(String s) {
        int cnt1 = 0;
        int n = s.length();
        for (int c : s.toCharArray()) {
            cnt1 += c - '0';
        }
        char[] res = new char[n];
        Arrays.fill(res, '0');
        res[n - 1] = '1';
        Arrays.fill(res, 0, cnt1 - 1, '1');
        return String.valueOf(res);

    }

    // 100047. 统计树中的合法路径数目 (Count Valid Paths in a Tree)
    private List<Integer>[] g100047;
    private boolean[] isPrime100047;
    private List<Integer> nodes100047;

    public long countPaths(int n, int[][] edges) {
        this.g100047 = new ArrayList[n + 1];
        Arrays.setAll(g100047, k -> new ArrayList<>());
        for (int[] e : edges) {
            int a = e[0];
            int b = e[1];
            g100047[a].add(b);
            g100047[b].add(a);
        }
        this.isPrime100047 = new boolean[n + 1];
        Arrays.fill(isPrime100047, true);
        isPrime100047[1] = false;
        for (int i = 2; i * i < n + 1; ++i) {
            if (isPrime100047[i]) {
                for (int j = i * i; j < n + 1; j += i) {
                    isPrime100047[j] = false;
                }
            }
        }
        int[] size = new int[n + 1];
        long res = 0L;
        for (int i = 2; i <= n; ++i) {
            if (isPrime100047[i]) {
                long s = 0L;
                for (int y : g100047[i]) {
                    if (isPrime100047[y]) {
                        continue;
                    }
                    if (size[y] == 0) {
                        nodes100047 = new ArrayList<>();
                        dfs100047(y, -1);
                        for (int z : nodes100047) {
                            size[z] = nodes100047.size();
                        }
                    }
                    res += s * size[y];
                    s += size[y];
                }
                res += s;
            }
        }
        return res;
    }

    private void dfs100047(int x, int fa) {
        nodes100047.add(x);
        for (int y : g100047[x]) {
            if (y != fa && !isPrime100047[y]) {
                dfs100047(y, x);
            }
        }
    }

    // 2865. 美丽塔 I (Beautiful Towers I)
    // 2866. 美丽塔 II (Beautiful Towers II)
    public long maximumSumOfHeights(List<Integer> maxHeights) {
        int n = maxHeights.size();
        Stack<Integer> stack = new Stack<>();
        long[] right = new long[n + 1];
        long sum = 0L;
        stack.push(n);
        for (int i = n - 1; i >= 0; --i) {
            while (stack.size() > 1 && maxHeights.get(i) <= maxHeights.get(stack.peek())) {
                int j = stack.pop();
                sum -= (long) (stack.peek() - j) * maxHeights.get(j);
            }
            sum += (long) (stack.peek() - i) * maxHeights.get(i);
            right[i] = sum;
            stack.push(i);
        }
        long res = sum;
        sum = 0L;
        stack.clear();
        stack.push(-1);
        long[] left = new long[n];
        for (int i = 0; i < n; ++i) {
            while (stack.size() > 1 && maxHeights.get(i) <= maxHeights.get(stack.peek())) {
                int j = stack.pop();
                sum -= (long) (j - stack.peek()) * maxHeights.get(j);
            }
            sum += (long) (i - stack.peek()) * maxHeights.get(i);
            left[i] = sum;
            stack.push(i);
        }
        for (int i = 0; i < n; ++i) {
            res = Math.max(res, left[i] + right[i + 1]);
        }
        return res;

    }

    // 8038. 收集元素的最少操作次数 (Minimum Operations to Collect Elements)
    public int minOperations(List<Integer> nums, int k) {
        long m = 0L;
        long u = (1L << k) - 1;
        for (int i = nums.size() - 1; i >= 0; --i) {
            m |= 1L << (nums.get(i) - 1);
            if ((m & u) == u) {
                return nums.size() - i;
            }
        }
        return 0;

    }

    // 100032. 使数组为空的最少操作次数 (Minimum Number of Operations to Make Array Empty)
    public int minOperations(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.merge(num, 1, Integer::sum);
        }
        int res = 0;
        for (int c : map.values()) {
            if (c == 1) {
                return -1;
            }
            res += (c - 1) / 3 + 1;
        }
        return res;

    }

    // 100019. 将数组分割成最多数目的子数组 (Split Array Into Maximum Number of Subarrays)
    public int maxSubarrays(int[] nums) {
        int res = 0;
        int cur = -1;
        for (int num : nums) {
            cur &= num;
            if (cur == 0) {
                cur = -1;
                ++res;
            }
        }
        return Math.max(1, res);
    }

    // 8051. 可以被 K 整除连通块的最大数目 (Maximum Number of K-Divisible Components)
    private List<Integer>[] g8051;
    private int res8051;
    private int[] values8051;
    private int k8051;

    public int maxKDivisibleComponents(int n, int[][] edges, int[] values, int k) {
        this.g8051 = new ArrayList[n];
        Arrays.setAll(g8051, o -> new ArrayList<>());
        for (int[] e : edges) {
            g8051[e[0]].add(e[1]);
            g8051[e[1]].add(e[0]);
        }
        this.values8051 = values;
        this.k8051 = k;
        dfs8051(0, -1);
        return res8051;

    }

    private int dfs8051(int x, int fa) {
        int sum = values8051[x];
        for (int y : g8051[x]) {
            if (y != fa) {
                sum += dfs8051(y, x);
                sum %= k8051;
            }
        }
        if (sum % k8051 == 0) {
            ++res8051;
        }
        return sum % k8051;
    }

    // 2873. 有序三元组中的最大值 I (Maximum Value of an Ordered Triplet I)
    // 2874. 有序三元组中的最大值 II (Maximum Value of an Ordered Triplet II)
    public long maximumTripletValue(int[] nums) {
        long res = 0L;
        long diff = 0L;
        int preMax = 0;
        for (int num : nums) {
            res = Math.max(res, diff * num);
            diff = Math.max(diff, preMax - num);
            preMax = Math.max(preMax, num);
        }
        return res;

    }

    // 2875. 无限数组的最短子数组 (Minimum Size Subarray in Infinite Array)
    public int minSizeSubarray(int[] nums, int target) {
        int n = nums.length;
        long sum = 0L;
        for (int num : nums) {
            sum += num;
        }
        long cnt = target / sum;
        long res = cnt * n;
        target %= sum;
        if (target == 0) {
            return (int) res;
        }
        int[] arr = new int[n * 2];
        for (int i = 0; i < n; ++i) {
            arr[i] = nums[i];
            arr[i + n] = nums[i];
        }
        long cur = 0L;
        int i = 0;
        int j = 0;
        int d = n + 1;
        while (j < n * 2) {
            cur += arr[j];
            while (cur > target) {
                cur -= arr[i];
                ++i;
            }
            if (cur == target) {
                d = Math.min(d, j - i + 1);
            }
            ++j;
        }
        if (d == n + 1) {
            return -1;
        }
        return (int) (res + d);

    }

    // 2876. 有向图访问计数 (Count Visited Nodes in a Directed Graph)
    private List<Integer>[] rg2876;
    private int[] res2876;

    public int[] countVisitedNodes(List<Integer> edges) {
        int n = edges.size();
        int[] g = new int[n];
        Arrays.fill(g, -1);
        int[] deg = new int[n];
        for (int i = 0; i < n; ++i) {
            g[i] = edges.get(i);
            ++deg[edges.get(i)];
        }
        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < n; ++i) {
            if (deg[i] == 0) {
                q.offer(i);
            }
        }
        while (!q.isEmpty()) {
            int x = q.poll();
            int y = g[x];
            --deg[y];
            if (deg[y] == 0) {
                q.offer(y);
            }
        }
        this.res2876 = new int[n];
        for (int i = 0; i < n; ++i) {
            if (deg[i] != 0 && res2876[i] == 0) {
                int cnt = 1;
                int x = i;
                while (g[x] != i) {
                    ++cnt;
                    x = g[x];
                }
                x = i;
                res2876[x] = cnt;
                while (g[x] != i) {
                    x = g[x];
                    res2876[x] = cnt;
                }
            }
        }
        this.rg2876 = new ArrayList[n];
        Arrays.setAll(rg2876, k -> new ArrayList<>());
        for (int i = 0; i < n; ++i) {
            if (deg[i] != 0 && deg[edges.get(i)] != 0) {
                continue;
            }
            rg2876[edges.get(i)].add(i);
        }
        for (int i = 0; i < n; ++i) {
            if (deg[i] != 0) {
                rdfs2876(i, res2876[i]);
            }
        }
        return res2876;
    }

    private void rdfs2876(int x, int d) {
        res2876[x] = d;
        for (int y : rg2876[x]) {
            rdfs2876(y, d + 1);
        }
    }

    // 2572. 无平方子集计数 (Count the Number of Square-Free Subsets)
    private int[] masks2572;
    private int[][] memo2572;
    private int[] cnts2572;

    public int squareFreeSubsets(int[] nums) {
        int[] primes = { 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29 };
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < primes.length; ++i) {
            map.put(primes[i], i);
        }
        this.masks2572 = new int[31];
        for (int i = 1; i <= 30; ++i) {
            if (map.containsKey(i)) {
                masks2572[i] |= 1 << map.get(i);
            } else if (i % 4 == 0 || i % 9 == 0 || i % 25 == 0) {
                masks2572[i] = -1;
            } else {
                for (int j = 2; j <= i; ++j) {
                    if (map.containsKey(j) && i % j == 0) {
                        masks2572[i] |= 1 << map.get(j);
                    }
                }
            }
        }
        this.memo2572 = new int[31][1 << primes.length];
        for (int i = 0; i < 31; ++i) {
            Arrays.fill(memo2572[i], -1);
        }
        this.cnts2572 = new int[31];
        for (int num : nums) {
            ++cnts2572[num];
        }
        return dfs2572(1, 0);
    }

    private int dfs2572(int i, int j) {
        if (i == 31) {
            return j > 0 ? 1 : 0;
        }
        if (memo2572[i][j] != -1) {
            return memo2572[i][j];
        }
        int res = dfs2572(i + 1, j);
        final int MOD = (int) (1e9 + 7);
        if (i == 1) {
            res += (long) ((pow2572(2, cnts2572[i]) - 1 + MOD) % MOD) * dfs2572(i + 1, j | 1) % MOD;
        } else if (masks2572[i] != -1 && (masks2572[i] & j) == 0) {
            res += (long) cnts2572[i] * dfs2572(i + 1, j | masks2572[i]) % MOD;
        }
        return memo2572[i][j] = res % MOD;
    }

    private int pow2572(int a, int b) {
        if (b == 0) {
            return 1;
        }
        int res = pow2572(a, b >> 1);
        final int MOD = (int) (1e9 + 7);
        res = (int) ((long) res * res % MOD);
        if ((b & 1) == 1) {
            res = (int) ((long) res * a % MOD);
        }
        return res;
    }

    // 2035. 将数组分成两个数组并最小化数组和的差 (Partition Array Into Two Arrays to Minimize Sum
    // Difference) --折半搜索
    public int minimumDifference2035(int[] nums) {
        int n = nums.length / 2;
        int[] arr = Arrays.copyOfRange(nums, 0, n);
        Map<Integer, TreeSet<Integer>> map = new HashMap<>();
        for (int i = 0; i < 1 << n; ++i) {
            int cnt = 0;
            int sum = 0;
            for (int j = 0; j < n; ++j) {
                if (((i >> j) & 1) == 0) {
                    ++cnt;
                    sum += arr[j];
                } else {
                    sum -= arr[j];
                }
            }
            map.computeIfAbsent(cnt, k -> new TreeSet<>()).add(sum);
        }
        int res = Integer.MAX_VALUE;
        arr = Arrays.copyOfRange(nums, n, n * 2);
        for (int i = 0; i < 1 << n; ++i) {
            int cnt = 0;
            int sum = 0;
            for (int j = 0; j < n; ++j) {
                if (((i >> j) & 1) == 0) {
                    ++cnt;
                    sum += arr[j];
                } else {
                    sum -= arr[j];
                }
            }
            TreeSet<Integer> set = map.getOrDefault(n - cnt, new TreeSet<>());
            Integer ceiling = set.ceiling(-sum);
            if (ceiling != null) {
                res = Math.min(res, Math.abs(ceiling + sum));
            }
            Integer floor = set.floor(-sum);
            if (floor != null) {
                res = Math.min(res, Math.abs(floor + sum));
            }
            if (res == 0) {
                break;
            }
        }
        return res;

    }

    // 1994. 好子集的数目 (The Number of Good Subsets)
    private int[][] memo1994;
    private int[] masks1994;
    private int[] cnts1994;

    public int numberOfGoodSubsets(int[] nums) {
        int[] primes = { 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29 };
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < primes.length; ++i) {
            map.put(primes[i], i);
        }
        this.masks1994 = new int[31];
        for (int i = 1; i < 31; ++i) {
            if (map.containsKey(i)) {
                masks1994[i] = 1 << map.get(i);
            } else if (i % 4 == 0 || i % 9 == 0 || i % 25 == 0) {
                masks1994[i] = -1;
            } else {
                for (int j = 2; j <= i; ++j) {
                    if (map.containsKey(j) && i % j == 0) {
                        masks1994[i] |= 1 << map.get(j);
                    }
                }
            }
        }
        this.cnts1994 = new int[31];
        for (int num : nums) {
            ++cnts1994[num];
        }
        this.memo1994 = new int[31][1 << primes.length];
        for (int i = 0; i < 31; ++i) {
            Arrays.fill(memo1994[i], -1);
        }
        return dfs1994(1, 0);

    }

    private int dfs1994(int i, int j) {
        if (i == 31) {
            return j > 1 ? 1 : 0;
        }
        if (memo1994[i][j] != -1) {
            return memo1994[i][j];
        }
        // 不选
        int res = dfs1994(i + 1, j);
        final int MOD = (int) (1e9 + 7);
        // 选
        if (i == 1) {
            int c = power1994(2, cnts1994[i]) - 1;
            res += (long) c * dfs1994(i + 1, j | 1) % MOD;
        } else if (masks1994[i] != -1 && (masks1994[i] & j) == 0) {
            res += (long) cnts1994[i] * dfs1994(i + 1, j | masks1994[i]) % MOD;
        }
        return memo1994[i][j] = res % MOD;
    }

    private int power1994(int a, int b) {
        if (b == 0) {
            return 1;
        }
        int res = power1994(a, b >> 1);
        final int MOD = (int) (1e9 + 7);
        res = (int) ((long) res * res % MOD);
        if (b % 2 == 1) {
            res = (int) ((long) res * a % MOD);
        }
        return res;
    }

    // 805. 数组的均值分割 (Split Array With Same Average)
    public boolean splitArraySameAverage(int[] nums) {
        int n = nums.length;
        if (n == 1) {
            return false;
        }
        int total = Arrays.stream(nums).sum();
        Map<Integer, Set<Integer>> map = new HashMap<>();
        map.computeIfAbsent(0, k -> new HashSet<>()).add(0);
        int halfLen = n / 2;
        int[] sum = new int[1 << halfLen];
        for (int i = 1; i < 1 << halfLen; ++i) {
            int index = Integer.numberOfTrailingZeros(i);
            sum[i] = sum[i ^ (1 << index)] + nums[index];
            map.computeIfAbsent(Integer.bitCount(i), k -> new HashSet<>()).add(sum[i]);
        }
        sum = new int[1 << (n - halfLen)];
        for (int i = 1; i < 1 << (n - halfLen); ++i) {
            int index = Integer.numberOfTrailingZeros(i);
            sum[i] = sum[i ^ (1 << index)] + nums[index + halfLen];
            int cnt = Integer.bitCount(i);
            for (int j = 0; j <= halfLen; ++j) {
                Set<Integer> set = map.getOrDefault(j, new HashSet<>());
                // set + sum[i] |||| total - (set + sum[i])
                // ------------- ==== ----------------------
                // j + cnt |||| n - (j + cnt)
                // ==> (j + cnt) * (total - sum[i]) - (j + cnt) * SET == (n - j - cnt) * SET +
                // (n - j - cnt) * sum[i]
                // ==> (j + cnt) * (total - sum[i]) - (n - j - cnt) * sum[i] == n * SET
                // ==> j * total + cnt * total - n * sum[i] == n * SET
                if (j + cnt < n && (j * total + cnt * total - n * sum[i]) % n == 0
                        && set.contains((j * total + cnt * total - n * sum[i]) / n)) {
                    return true;
                }
            }
        }
        return false;

    }

    // 2894. 分类求和并作差 (Divisible and Non-divisible Sums Difference)
    public int differenceOfSums(int n, int m) {
        return (1 + n) * n / 2 - m * (1 + n / m) * (n / m);
    }

    // 2895. 最小处理时间 (Minimum Processing Time)
    public int minProcessingTime(List<Integer> processorTime, List<Integer> tasks) {
        Collections.sort(processorTime, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(o2, o1);
            }

        });
        Collections.sort(tasks);
        int res = 0;
        int n = tasks.size();
        for (int i = 0; i < n; ++i) {
            res = Math.max(res, tasks.get(i) + processorTime.get(i / 4));
        }
        return res;

    }

    // 2896. 执行操作使两个字符串相等 (Apply Operations to Make Two Strings Equal)
    private int n2896;
    private List<Integer> list2896;
    private int[] memo2896;
    private int x2896;

    public int minOperations(String s1, String s2, int x) {
        this.list2896 = new ArrayList<>();
        for (int i = 0; i < s1.length(); ++i) {
            if (s1.charAt(i) != s2.charAt(i)) {
                list2896.add(i);
            }
        }
        this.n2896 = list2896.size();
        if (n2896 == 0) {
            return 0;
        }
        if (n2896 % 2 == 1) {
            return -1;
        }
        this.memo2896 = new int[n2896];
        Arrays.fill(memo2896, -1);
        this.x2896 = x;
        return dfs2896(0) / 2;
    }

    private int dfs2896(int i) {
        if (i == n2896) {
            return 0;
        }
        if (i == n2896 - 1) {
            return x2896;
        }
        if (memo2896[i] != -1) {
            return memo2896[i];
        }
        return memo2896[i] = Math.min(dfs2896(i + 1) + x2896,
                dfs2896(i + 2) + (list2896.get(i + 1) - list2896.get(i)) * 2);
    }

    // 2897. 对数组执行操作使平方和最大 (Apply Operations on Array to Maximize Sum of Squares)
    public int maxSum(List<Integer> nums, int k) {
        int[] cnt = new int[31];
        for (int num : nums) {
            for (int i = 0; i < 31; ++i) {
                cnt[i] += (num >> i) & 1;
            }
        }
        final long MOD = (long) (1e9 + 7);
        long res = 0L;
        while (k-- > 0) {
            long x = 0L;
            for (int i = 0; i < 31; ++i) {
                if (cnt[i]-- > 0) {
                    x |= 1 << i;
                }
            }
            res += x * x;
            res %= MOD;
        }
        return (int) res;

    }

    // 368. 最大整除子集 (Largest Divisible Subset)
    private int[] memo368;
    private int n368;
    private int[] nums368;

    public List<Integer> largestDivisibleSubset(int[] nums) {
        Arrays.sort(nums);
        this.n368 = nums.length;
        this.memo368 = new int[n368];
        this.nums368 = nums;
        for (int i = 0; i < n368; ++i) {
            dfs368(i);
        }
        int max = Arrays.stream(memo368).max().getAsInt();
        int maxVal = 0;
        for (int i = n368 - 1; i >= 0; --i) {
            if (memo368[i] > max) {
                maxVal = nums[i];
            }

        }
        List<Integer> res = new ArrayList<>();
        for (int i = n368 - 1; i >= 0 && max > 0; --i) {
            if (memo368[i] == max && maxVal % nums[i] == 0) {
                res.add(nums[i]);
                maxVal = nums[i];
                --max;
            }
        }
        return res;
    }

    private int dfs368(int i) {
        if (memo368[i] != 0) {
            return memo368[i];
        }
        int res = 0;
        for (int j = 0; j < i; ++j) {
            if (nums368[i] % nums368[j] == 0) {
                res = Math.max(res, dfs368(j));
            }
        }
        return memo368[i] = res + 1;
    }

    // 100078. 最长相邻不相等子序列 I (Longest Unequal Adjacent Groups Subsequence I)
    public List<String> getWordsInLongestSubsequence(int n, String[] words, int[] groups) {
        List<String> res = new ArrayList<>();
        int pre = -1;
        for (int i = 0; i < n; ++i) {
            if (res.isEmpty() || (groups[i] ^ pre) == 1) {
                res.add(words[i]);
                pre = groups[i];
            }
        }
        return res;

    }

    // 100101. 找出满足差值条件的下标 II (Find Indices With Index and Value Difference II)
    // 100096. 找出满足差值条件的下标 I (Find Indices With Index and Value Difference I)
    public int[] findIndices(int[] nums, int indexDifference, int valueDifference) {
        int n = nums.length;
        int minIndex = 0;
        int maxIndex = 0;
        for (int i = indexDifference; i < n; ++i) {
            int j = i - indexDifference;
            if (nums[j] > nums[maxIndex]) {
                maxIndex = j;
            } else if (nums[j] < nums[minIndex]) {
                minIndex = j;
            }
            if (nums[i] - nums[minIndex] >= valueDifference) {
                return new int[] { minIndex, i };
            }
            if (nums[maxIndex] - nums[i] >= valueDifference) {
                return new int[] { maxIndex, i };
            }
        }
        return new int[] { -1, -1 };

    }

    // 100084. 最短且字典序最小的美丽子字符串 (Shortest and Lexicographically Smallest Beautiful
    // String)
    public String shortestBeautifulSubstring(String s, int k) {
        int n = s.length();
        String res = "";
        int i = 0;
        int j = 0;
        int cnt = 0;
        int min = Integer.MAX_VALUE;
        while (j < n) {
            cnt += s.charAt(j) - '0';
            while (cnt > k || i <= j && s.charAt(i) == '0') {
                cnt -= s.charAt(i) - '0';
                ++i;
            }
            if (cnt == k) {
                if (j - i + 1 < min) {
                    min = j - i + 1;
                    res = s.substring(i, j + 1);
                } else if (j - i + 1 == min && s.substring(i, j + 1).compareTo(res) < 0) {
                    res = s.substring(i, j + 1);
                }
            }
            ++j;
        }
        return res;

    }

    // 8026. 构造乘积矩阵 (Construct Product Matrix)
    public int[][] constructProductMatrix(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        long suf = 1L;
        final int MOD = 12345;
        int[][] p = new int[m][n];
        for (int i = m - 1; i >= 0; --i) {
            for (int j = n - 1; j >= 0; --j) {
                p[i][j] = (int) suf;
                suf *= grid[i][j];
                suf %= MOD;
            }
        }

        long pre = 1L;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                p[i][j] = (int) (p[i][j] * pre % MOD);
                pre *= grid[i][j];
                pre %= MOD;
            }
        }

        return p;

    }

    // 100095. 上一个遍历的整数 (Last Visited Integers)
    public List<Integer> lastVisitedIntegers(List<String> words) {
        List<Integer> res = new ArrayList<>();
        List<Integer> nums = new ArrayList<>();
        int k = 0;
        for (String s : words) {
            if ("prev".equals(s)) {
                ++k;
                if (k <= nums.size()) {
                    res.add(nums.get(nums.size() - k - 1));
                } else {
                    res.add(-1);
                }
            } else {
                k = 0;
                nums.add(Integer.parseInt(s));
            }
        }
        return res;

    }

    // LCP 13. 寻宝
    private int[][] memo_LCP_13;
    private int u_LCP_13;
    // 终点到各机关的最短距离
    private int[] targetToM_LCP_13;
    // minDis[i][j] 机关i到石堆j的最短距离
    private int[][] minDis_LCP_13;
    // 各机关的位置
    private List<int[]> mPos_LCP_13;
    // 各石堆的位置
    private List<int[]> oPos_LCP_13;

    public int minimalSteps(String[] maze) {
        int m = maze.length;
        int n = maze[0].length();
        // 起点
        int[] s = null;
        // 终点
        int[] t = null;
        // 机关的数量
        int mCnt = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (maze[i].charAt(j) == 'M') {
                    ++mCnt;
                } else if (maze[i].charAt(j) == 'S') {
                    s = new int[] { i, j };
                } else if (maze[i].charAt(j) == 'T') {
                    t = new int[] { i, j };
                }
            }
        }
        // 机关的位置（所有机关的位置都必须可达）
        this.mPos_LCP_13 = new ArrayList<>();
        // 所有可达的石堆位置
        this.oPos_LCP_13 = new ArrayList<>();
        // 终点必须可达
        boolean[][] vis = new boolean[m][n];
        vis[s[0]][s[1]] = true;
        Queue<int[]> q = new LinkedList<>();
        q.offer(new int[] { s[0], s[1] });
        int[][] dirs = { { 0, 1 }, { 0, -1 }, { -1, 0 }, { 1, 0 } };
        int level = 0;
        int disStartToTarget = 0;
        while (!q.isEmpty()) {
            ++level;
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                int[] cur = q.poll();
                int x = cur[0];
                int y = cur[1];
                for (int[] d : dirs) {
                    int nx = x + d[0];
                    int ny = y + d[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && maze[nx].charAt(ny) != '#' &&
                            !vis[nx][ny]) {
                        vis[nx][ny] = true;
                        if (maze[nx].charAt(ny) == 'M') {
                            mPos_LCP_13.add(new int[] { nx, ny, level });
                        } else if (maze[nx].charAt(ny) == 'O') {
                            oPos_LCP_13.add(new int[] { nx, ny, level });
                        } else if (maze[nx].charAt(ny) == 'T') {
                            disStartToTarget = level;
                        }
                        q.offer(new int[] { nx, ny });
                    }
                }

            }
        }

        // 终点不可达
        if (!vis[t[0]][t[1]]) {
            return -1;
        }
        // 存在不可达的机关
        if (mCnt != mPos_LCP_13.size()) {
            return -1;
        }
        // 没有机关
        if (mCnt == 0) {
            return disStartToTarget;
        }
        // 有机关、所有石堆均不可达
        if (oPos_LCP_13.isEmpty()) {
            return -1;
        }

        // 石堆位置 --> 编号
        Map<Integer, Integer> oPosToIndex = new HashMap<>();
        for (int i = 0; i < oPos_LCP_13.size(); ++i) {
            oPosToIndex.put(oPos_LCP_13.get(i)[0] * n + oPos_LCP_13.get(i)[1], i);
        }
        // 机关位置 --> 编号
        Map<Integer, Integer> mPosToIndex = new HashMap<>();
        for (int i = 0; i < mPos_LCP_13.size(); ++i) {
            mPosToIndex.put(mPos_LCP_13.get(i)[0] * n + mPos_LCP_13.get(i)[1], i);
        }
        // 机关均可达、有可达石堆、终点可达
        this.minDis_LCP_13 = new int[mPos_LCP_13.size()][oPos_LCP_13.size()];
        for (int i = 0; i < mPos_LCP_13.size(); ++i) {
            int mX = mPos_LCP_13.get(i)[0];
            int mY = mPos_LCP_13.get(i)[1];
            vis = new boolean[m][n];
            vis[mX][mY] = true;
            q.offer(new int[] { mX, mY, 0 });
            while (!q.isEmpty()) {
                int[] cur = q.poll();
                int x = cur[0];
                int y = cur[1];
                int dis = cur[2];
                for (int[] d : dirs) {
                    int nx = x + d[0];
                    int ny = y + d[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && maze[nx].charAt(ny) != '#' &&
                            !vis[nx][ny]) {
                        vis[nx][ny] = true;
                        if (maze[nx].charAt(ny) == 'O') {
                            int index = oPosToIndex.get(nx * n + ny);
                            minDis_LCP_13[i][index] = dis + 1;
                        }
                        q.offer(new int[] { nx, ny, dis + 1 });
                    }
                }
            }
        }

        // 终点到各机关的最短距离
        this.targetToM_LCP_13 = new int[mPos_LCP_13.size()];
        q.offer(new int[] { t[0], t[1], 0 });
        vis = new boolean[m][n];
        vis[t[0]][t[1]] = true;
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int y = cur[1];
            int dis = cur[2];
            for (int[] d : dirs) {
                int nx = x + d[0];
                int ny = y + d[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && maze[nx].charAt(ny) != '#' &&
                        !vis[nx][ny]) {
                    vis[nx][ny] = true;
                    if (maze[nx].charAt(ny) == 'M') {
                        int index = mPosToIndex.get(nx * n + ny);
                        targetToM_LCP_13[index] = dis + 1;
                    }
                    q.offer(new int[] { nx, ny, dis + 1 });
                }
            }
        }
        int M = mPos_LCP_13.size();
        this.memo_LCP_13 = new int[1 << M][M];
        for (int i = 0; i < 1 << M; ++i) {
            Arrays.fill(memo_LCP_13[i], -1);
        }
        this.u_LCP_13 = (1 << M) - 1;
        return dfs_LCP_13(0, 0);

    }

    private int dfs_LCP_13(int i, int j) {
        if (i == u_LCP_13) {
            // 最后一个落石的机关到target的距离
            return targetToM_LCP_13[j];
        }
        if (memo_LCP_13[i][j] != -1) {
            return memo_LCP_13[i][j];
        }
        // 起点
        if (i == 0) {
            int res = Integer.MAX_VALUE;
            for (int k = 0; k < mPos_LCP_13.size(); ++k) {
                int min = Integer.MAX_VALUE;
                for (int o = 0; o < oPos_LCP_13.size(); ++o) {
                    min = Math.min(min, minDis_LCP_13[k][o] + oPos_LCP_13.get(o)[2]);
                }
                res = Math.min(res, dfs_LCP_13(1 << k, k) + min);
            }
            return memo_LCP_13[i][j] = res;
        }
        int res = Integer.MAX_VALUE;
        int c = i ^ u_LCP_13;
        while (c != 0) {
            int mIndex = Integer.numberOfTrailingZeros(c);
            int min = Integer.MAX_VALUE;
            for (int x = 0; x < oPos_LCP_13.size(); ++x) {
                min = Math.min(min, minDis_LCP_13[mIndex][x] + minDis_LCP_13[j][x]);
            }
            res = Math.min(res, dfs_LCP_13(i | (1 << mIndex), mIndex) + min);
            c &= c - 1;
        }
        return memo_LCP_13[i][j] = res;
    }

    // 2908. 元素和最小的山形三元组 I (Minimum Sum of Mountain Triplets I)
    // 2909. 元素和最小的山形三元组 II (Minimum Sum of Mountain Triplets II)
    public int minimumSum(int[] nums) {
        int res = Integer.MAX_VALUE;
        int n = nums.length;
        int[] pre = new int[n];
        Arrays.fill(pre, Integer.MAX_VALUE);
        pre[0] = nums[0];
        for (int i = 1; i < n; ++i) {
            pre[i] = Math.min(pre[i - 1], nums[i]);
        }
        int suf = nums[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            suf = Math.min(suf, nums[i]);
            if (nums[i] > pre[i] && nums[i] > suf) {
                res = Math.min(res, nums[i] + pre[i] + suf);
            }
        }
        return res == Integer.MAX_VALUE ? -1 : res;
    }

    // 2911. 得到 K 个半回文串的最少修改次数 (Minimum Changes to Make K Semi-palindromes)
    private int[][] memo2911;
    private int[][] modify2911;
    private int n2911;
    private int k2911;

    public int minimumChanges(String s, int k) {
        this.n2911 = s.length();
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 1; i <= n2911; ++i) {
            for (int j = i * 2; j <= n2911; j += i) {
                map.computeIfAbsent(j, o -> new ArrayList<>()).add(i);
            }
        }
        this.modify2911 = new int[n2911][n2911];
        for (int i = 0; i < n2911; ++i) {
            for (int j = i + 1; j < n2911; ++j) {
                int len = j - i + 1;
                String sub = s.substring(i, j + 1);
                int res = Integer.MAX_VALUE;
                for (int d : map.getOrDefault(len, new ArrayList<>())) {
                    int sum = 0;
                    for (int c = 0; c < d; ++c) {
                        int x = c;
                        int y = len - d + c;
                        while (x < y) {
                            if (sub.charAt(x) != sub.charAt(y)) {
                                ++sum;
                            }
                            x += d;
                            y -= d;
                        }
                    }
                    res = Math.min(res, sum);
                }
                modify2911[i][j] = res;
            }
        }
        this.memo2911 = new int[n2911][k];
        for (int i = 0; i < n2911; ++i) {
            Arrays.fill(memo2911[i], -1);
        }
        this.k2911 = k;
        return dfs2911(0, 0);

    }

    private int dfs2911(int i, int j) {
        if (i == n2911) {
            return j == k2911 ? 0 : n2911;
        }
        if (j == k2911) {
            return n2911;
        }
        if ((n2911 - i) / 2 < k2911 - j) {
            return n2911;
        }
        if (memo2911[i][j] != -1) {
            return memo2911[i][j];
        }
        int res = n2911;
        for (int x = i + 1; x < n2911; ++x) {
            res = Math.min(res, dfs2911(x + 1, j + 1) + modify2911[i][x]);
        }
        return memo2911[i][j] = res;
    }

    // 100097. 合法分组的最少组数 (Minimum Number of Groups to Create a Valid Assignment)
    public int minGroupsForValidAssignment(int[] nums) {
        Map<Integer, Integer> cnt = new HashMap<>();
        for (int num : nums) {
            cnt.merge(num, 1, Integer::sum);
        }
        int min = Collections.min(cnt.values());
        search: for (int k = min; k > 0; --k) {
            int res = 0;
            for (int c : cnt.values()) {
                if (c / k < c % k) {
                    continue search;
                }
                res += (c + k) / (k + 1);
            }
            return res;
        }
        return -1;

    }

    // 100094. 子数组不同元素数目的平方和 I (Subarrays Distinct Element Sum of Squares I)
    public int sumCounts(List<Integer> nums) {
        final long MOD = (long) (1e9 + 7);
        long res = 0L;
        int n = nums.size();
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                Set<Integer> set = new HashSet<>();
                for (int k = i; k <= j; ++k) {
                    set.add(nums.get(k));
                }
                res += set.size() * set.size();
                res %= MOD;
            }
        }
        return (int) res;

    }
    
    // 100104. 使二进制字符串变美丽的最少修改次数 (Minimum Number of Changes to Make Binary String Beautiful)
    public int minChanges(String s) {
        int n = s.length();
        int res = 0;
        for (int i = 1; i < n; i += 2) {
            if (s.charAt(i) != s.charAt(i - 1)) {
                ++res;
            }
        }
        return res;

    }

    // 100042. 和为目标值的最长子序列的长度 (Length of the Longest Subsequence That Sums to
    // Target)
    private int[][] memo100042;
    private int n100042;
    private int target100042;
    private List<Integer> nums100042;

    public int lengthOfLongestSubsequence(List<Integer> nums, int target) {
        this.n100042 = nums.size();
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (sum < target) {
            return -1;
        }
        this.memo100042 = new int[n100042][target + 1];
        for (int i = 0; i < n100042; ++i) {
            Arrays.fill(memo100042[i], -1);
        }
        this.target100042 = target;
        this.nums100042 = nums;
        int res = dfs100042(0, 0);
        return res > 0 ? res : -1;

    }

    private int dfs100042(int i, int j) {
        if (j == target100042) {
            return 0;
        }
        if (j > target100042 || i == n100042) {
            return Integer.MIN_VALUE;
        }
        if (memo100042[i][j] != -1) {
            return memo100042[i][j];
        }
        return memo100042[i][j] = Math.max(dfs100042(i + 1, j), dfs100042(i + 1, j + nums100042.get(i)) + 1);
    }

    // 100111. 找出数组中的 K-or 值 (Find the K-or of an Array)
    public int findKOr(int[] nums, int k) {
        int[] cnts = new int[32];
        for (int num : nums) {
            for (int i = 0; i < 32; ++i) {
                cnts[i] += num >> i & 1;
            }
        }
        int res = 0;
        for (int i = 0; i < 32; ++i) {
            if (cnts[i] >= k) {
                res |= 1 << i;
            }
        }
        return res;

    }

    // 100102. 数组的最小相等和 (Minimum Equal Sum of Two Arrays After Replacing Zeros)
    public long minSum(int[] nums1, int[] nums2) {
        long sum1 = 0L;
        int cnt1 = 0;
        for (int num : nums1) {
            if (num == 0) {
                ++cnt1;
            }
            sum1 += num;
        }
        long sum2 = 0L;
        int cnt2 = 0;
        for (int num : nums2) {
            if (num == 0) {
                ++cnt2;
            }
            sum2 += num;
        }
        if (sum1 == sum2 && (cnt1 != 0 && cnt2 == 0 || cnt1 == 0 && cnt2 != 0)) {
            return -1;
        }
        if (sum1 > sum2) {
            long t = sum1;
            sum1 = sum2;
            sum2 = t;
            int tmp = cnt1;
            cnt1 = cnt2;
            cnt2 = tmp;
        }
        if (sum1 < sum2 && (cnt1 == 0 || cnt1 + sum1 > sum2 && cnt2 == 0)) {
            return -1;
        }
        return Math.max(sum1 + cnt1, sum2 + cnt2);

    }

    // 100107. 使数组变美的最小增量运算数 (Minimum Increment Operations to Make Array Beautiful)
    private long[][] memo100107;
    private int[] nums100107;
    private int k100107;
    private int n100107;

    public long minIncrementOperations(int[] nums, int k) {
        this.nums100107 = nums;
        this.k100107 = k;
        this.n100107 = nums.length;
        this.memo100107 = new long[n100107][3];
        for (int i = 0; i < n100107; ++i) {
            Arrays.fill(memo100107[i], -1);
        }
        return dfs100107(0, 0);

    }

    private long dfs100107(int i, int j) {
        if (i >= n100107) {
            return 0L;
        }
        if (memo100107[i][j] != -1L) {
            return memo100107[i][j];
        }
        long res = dfs100107(i + 1, 0) + Math.max(k100107 - nums100107[i], 0);
        if (j < 2) {
            res = Math.min(res, dfs100107(i + 1, j + 1));
        }
        return memo100107[i][j] = res;
    }

    // 2920. 收集所有金币可获得的最大积分 (Maximum Points After Collecting Coins From All Nodes)
    private List<Integer>[] g2920;
    private int[] coins2920;
    private int k2920;
    private int[][] memo2920;

    public int maximumPoints(int[][] edges, int[] coins, int k) {
        int n = coins.length;
        this.g2920 = new ArrayList[n];
        Arrays.setAll(g2920, o -> new ArrayList<>());
        for (int[] e : edges) {
            g2920[e[0]].add(e[1]);
            g2920[e[1]].add(e[0]);
        }
        this.coins2920 = coins;
        this.k2920 = k;
        this.memo2920 = new int[n][15];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(memo2920[i], (int) -10e9);
        }
        return dfs2920(0, 0, -1);

    }

    private int dfs2920(int i, int j, int fa) {
        if (memo2920[i][j] != (int) -10e9) {
            return memo2920[i][j];
        }
        int res1 = (coins2920[i] >> j) - k2920;
        int res2 = coins2920[i] >> (j + 1);
        for (int y : g2920[i]) {
            if (y != fa) {
                res1 += dfs2920(y, j, i);
                if (j < 14) {
                    res2 += dfs2920(y, j + 1, i);
                }
            }
        }
        return memo2920[i][j] = Math.max(res1, res2);
    }

    // 2003. 每棵子树内缺失的最小基因值 (Smallest Missing Genetic Value in Each Subtree)
    public int[] smallestMissingValueSubtree(int[] parents, int[] nums) {
        int n = parents.length;
        int[] res = new int[n];
        Arrays.fill(res, 1);
        int node = -1;
        for (int i = 0; i < n; ++i) {
            if (nums[i] == 1) {
                node = i;
            }
        }
        if (node < 0) {
            return res;
        }
        List<Integer>[] g = new ArrayList[n];
        Arrays.setAll(g, k -> new ArrayList<>());
        for (int i = 1; i < n; ++i) {
            g[parents[i]].add(i);
        }
        Set<Integer> vis = new HashSet<>();
        int mex = 2;
        while (node >= 0) {
            dfs2003(node, g, vis, nums);
            while (vis.contains(mex)) {
                ++mex;
            }
            res[node] = mex;
            node = parents[node];
        }
        return res;

    }
    
    private void dfs2003(int x, List<Integer>[] g, Set<Integer> vis, int[] nums) {
        vis.add(nums[x]); // 标记基因值
        for (int son : g[x]) {
            if (!vis.contains(nums[son])) {
                dfs2003(son, g, vis, nums);
            }
        }
    }

    // 100115. 找到冠军 I (Find Champion I)
    public int findChampion(int[][] grid) {
        int n = grid.length;
        int[] deg = new int[n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    if (grid[i][j] == 1) {
                        ++deg[j];
                    }
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            if (deg[i] == 0) {
                return i;
            }
        }
        return -1;
    }
    
    // 100116. 找到冠军 II (Find Champion II)
    public int findChampion(int n, int[][] edges) {
        int[] deg = new int[n];
        for (int[] e : edges) {
            ++deg[e[1]];
        }
        int res = -1;
        for (int i = 0; i < n; ++i) {
            if (deg[i] == 0) {
                if (res != -1) {
                    return -1;
                }
                res = i;
            }
        }
        return res;

    }

    // 100118. 在树上执行操作以后得到的最大分数 (Maximum Score After Applying Operations on a Tree)
    private List<Integer>[] g100118;
    private int n100118;
    private long[] s100118;
    private int[] values100118;

    public long maximumScoreAfterOperations(int[][] edges, int[] values) {
        this.n100118 = values.length;
        this.g100118 = new ArrayList[n100118];
        Arrays.setAll(g100118, k -> new ArrayList<>());
        for (int[] e : edges) {
            int u = e[0];
            int v = e[1];
            g100118[u].add(v);
            g100118[v].add(u);
        }
        g100118[0].add(-1);
        this.s100118 = new long[n100118];
        this.values100118 = values;
        dfs100118(0, -1);
        return dfs_score(0, -1);

    }

    private long dfs_score(int x, int fa) {
        // 叶子
        if (g100118[x].size() == 1) {
            return 0L;
        }
        // 不收集
        long res1 = s100118[x] - values100118[x];
        // 收集
        long res2 = values100118[x];
        for (int y : g100118[x]) {
            if (y != fa) {
                res2 += dfs_score(y, x);
            }
        }
        return Math.max(res1, res2);

    }

    private long dfs100118(int x, int fa) {
        s100118[x] = values100118[x];
        for (int y : g100118[x]) {
            if (y != fa) {
                s100118[x] += dfs100118(y, x);
            }
        }
        return s100118[x];
    }

    // 2928. 给小朋友们分糖果 I (Distribute Candies Among Children I)
    public long distributeCandies(int n, int limit) {
        long max = limit * 3L;
        if (n > max) {
            return 0L;
        }
        long res = 0L;
        int f = Math.min(n, limit);
        while (f >= 0) {
            long left = n - f;
            if (limit * 2L < left) {
                break;
            }
            if (left <= limit) {
                res += left + 1;
            } else {
                res += limit - (left - limit) + 1;
            }
            --f;
        }
        return res;

    }

    // 2930. 重新排列后包含指定子字符串的字符串数目 (Number of Strings Which Can Be Rearranged to Contain Substring)
    private int[][][][] memo2930;
    private int n2930;

    public int stringCount(int n) {
        this.memo2930 = new int[n][2][3][2];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 3; ++k) {
                    Arrays.fill(memo2930[i][j][k], -1);
                }
            }
        }
        this.n2930 = n;
        return dfs2930(0, 0, 0, 0);

    }

    private int dfs2930(int i, int j, int k, int l) {
        if (i == n2930) {
            return j >= 1 && k >= 2 && l >= 1 ? 1 : 0;
        }
        if (memo2930[i][j][k][l] != -1) {
            return memo2930[i][j][k][l];
        }
        final int MOD = (int) (1e9 + 7);
        return memo2930[i][j][k][l] = (int) (((long) dfs2930(i + 1, Math.min(j + 1, 1), k, l)
                + dfs2930(i + 1, j, Math.min(k + 1, 2), l)
                + dfs2930(i + 1, j, k, Math.min(l + 1, 1))
                + 23L * dfs2930(i + 1, j, k, l)) % MOD);
    }

    // 2931. 购买物品的最大开销 (Maximum Spending After Buying Items)
    public long maxSpending(int[][] values) {
        long res = 0L;
        int m = values.length;
        int n = values[0].length;
        int[] arr = new int[m * n];
        int index = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                arr[index++] = values[i][j];
            }
        }
        Arrays.sort(arr);
        for (int d = 1; d <= m * n; ++d) {
            res += (long) d * arr[d - 1];
        }
        return res;

    }

    // 2933. 高访问员工 (High-Access Employees)
    public List<String> findHighAccessEmployees(List<List<String>> access_times) {
        Map<String, List<String>> map = new HashMap<>();
        for (List<String> acc : access_times) {
            map.computeIfAbsent(acc.get(0), k -> new ArrayList<>()).add(acc.get(1));
        }
        List<String> res = new ArrayList<>();
        for (Map.Entry<String, List<String>> entry : map.entrySet()) {
            List<String> v = entry.getValue();
            Collections.sort(v);
            int n = v.size();
            for (int i = 2; i < n; ++i) {
                int cur = Integer.parseInt(v.get(i));
                int pre = Integer.parseInt(v.get(i - 2));
                if (cur - pre < 100) {
                    res.add(entry.getKey());
                    break;
                }
            }
        }
        return res;

    }

    // 2934. 最大化数组末位元素的最少操作次数 (Minimum Operations to Maximize Last Elements in Arrays)
    public int minOperations(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int res = 0;
        for (int i = n - 2; i >= 0; --i) {
            if (nums1[i] <= nums1[n - 1] && nums2[i] <= nums2[n - 1]) {
                continue;
            }
            if (nums1[i] <= nums2[n - 1] && nums2[i] <= nums1[n - 1]) {
                ++res;
            } else {
                return -1;
            }
        }
        int res2 = 1;
        for (int i = n - 2; i >= 0; --i) {
            if (nums1[i] <= nums2[n - 1] && nums2[i] <= nums1[n - 1]) {
                continue;
            }
            if (nums1[i] <= nums1[n - 1] && nums2[i] <= nums2[n - 1]) {
                ++res2;
            } else {
                return -1;
            }
        }
        return Math.min(res, res2);

    }

    // 100131. 使三个字符串相等 (Make Three Strings Equal)
    public int findMinimumOperations(String s1, String s2, String s3) {
        int n = Math.min(Math.min(s1.length(), s2.length()), s3.length());
        int i = 0;
        while (i < n) {
            if (s1.charAt(i) != s2.charAt(i) || s2.charAt(i) != s3.charAt(i)) {
                break;
            }
            ++i;
        }
        if (i == 0) {
            return -1;
        }
        return s1.length() - i + s2.length() - i + s3.length() - i;
    }

    // 2938. 区分黑球与白球 (Separate Black and White Balls)
    public long minimumSteps(String s) {
        long res = 0L;
        int cnt1 = 0;
        for (char c : s.toCharArray()) {
            if (c == '0') {
                res += cnt1;
            } else {
                ++cnt1;
            }
        }
        return res;
    }

    // 100110. 找到 Alice 和 Bob 可以相遇的建筑 (Find Building Where Alice and Bob Can Meet)
    public int[] leftmostBuildingQueries(int[] heights, int[][] queries) {
        int m = queries.length;
        int[] res = new int[m];
        Arrays.fill(res, -1);
        Map<Integer, List<int[]>> map = new HashMap<>();
        for (int i = 0; i < m; ++i) {
            int x = queries[i][0];
            int y = queries[i][1];
            if (x > y) {
                int t = x;
                x = y;
                y = t;
            }
            if (x == y || heights[x] < heights[y]) {
                res[i] = y;
            } else {
                map.computeIfAbsent(y, k -> new ArrayList<>()).add(new int[] { heights[x], i });
            }
        }
        Queue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });
        for (int i = 0; i < heights.length; ++i) {
            while (!pq.isEmpty() && pq.peek()[0] < heights[i]) {
                res[pq.poll()[1]] = i;
            }
            for (int[] p : map.getOrDefault(i, new ArrayList<>())) {
                pq.offer(p);
            }
        }
        return res;

    }

    // 100119. 最大异或乘积 (Maximum Xor Product)
    public int maximumXorProduct(long a, long b, int n) {
        if (a < b) {
            long t = a;
            a = b;
            b = t;
        }
        long mask = (1L << n) - 1;
        long ax = a & ~mask;
        long bx = b & ~mask;
        a &= mask;
        b &= mask;

        long diff = a ^ b;
        long one = mask ^ diff;
        ax |= one;
        bx |= one;
        if (diff > 0 && ax == bx) {
            long highestBit = 1L << (63 - Long.numberOfLeadingZeros(diff));
            ax |= highestBit;
            diff ^= highestBit;
        }
        bx |= diff;
        final int MOD = (int) (1e9 + 7);
        return (int) (ax % MOD * (bx % MOD) % MOD);

    }

    // 2946. 循环移位后的矩阵相似检查 (Matrix Similarity After Cyclic Shifts)
    public boolean areSimilar(int[][] mat, int k) {
        int m = mat.length;
        int n = mat[0].length;
        int[][] copy = new int[m][n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                copy[i][(j + k) % n] = mat[i][j];
            }
            if (!Arrays.equals(mat[i], copy[i])) {
                return false;
            }
        }
        return true;
    }

    // 2947. 统计美丽子字符串 I (Count Beautiful Substrings I)
    public int beautifulSubstrings(String s, int k) {
        int n = s.length();
        int res = 0;
        for (int i = 0; i < n; ++i) {
            int a = 0;
            int b = 0;
            for (int j = i; j < n; ++j) {
                if (check2947(s.charAt(j))) {
                    ++a;
                } else {
                    ++b;
                }
                if (a == b && (a * b) % k == 0) {
                    ++res;
                }
            }
        }
        return res;

    }

    private boolean check2947(char c) {
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
            return true;
        }
        return false;
    }

    // 2948. 交换得到字典序最小的数组 (Make Lexicographically Smallest Array by Swapping
    // Elements)
     public int[] lexicographicallySmallestArray(int[] nums, int limit) {
        List<int[]> list = new ArrayList<>();
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            list.add(new int[] { i, nums[i] });
        }
        Collections.sort(list, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[1], o2[1]);
            }
            
        });
        int[] res = new int[n];
        List<Integer> ids = new ArrayList<>();
        List<Integer> vals = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            ids.add(list.get(i)[0]);
            vals.add(list.get(i)[1]);
            if (i == n - 1 || list.get(i + 1)[1] - list.get(i)[1] > limit) {
                Collections.sort(ids);
                int m = ids.size();
                int j = 0;
                while (j < m) {
                    res[ids.get(j)] = vals.get(j);
                    ++j;
                }
                ids.clear();
                vals.clear();
            }
        }
        return res;

    }
    
    // 2942. 查找包含给定字符的单词 (Find Words Containing Character)
    public List<Integer> findWordsContaining(String[] words, char x) {
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < words.length; ++i) {
            if (words[i].contains(String.valueOf(x))) {
                res.add(i);
            }
        }
        return res;
    }

    // 2943. 最大化网格图中正方形空洞的面积 (Maximize Area of Square Hole in Grid)
    public int maximizeSquareHoleArea(int n, int m, int[] hBars, int[] vBars) {
        int s = Math.min(cal2943(hBars), cal2943(vBars));
        return s * s;
    }

    private int cal2943(int[] bars) {
        Arrays.sort(bars);
        int i = 0;
        int res = 0;
        int n = bars.length;
        while (i < n) {
            int j = i + 1;
            while (j < n && bars[j] - bars[j - 1] == 1) {
                ++j;
            }
            res = Math.max(res, j - i + 1);
            i = j;
        }
        return res;

    }

    // 2944. 购买水果需要的最少金币数 (Minimum Number of Coins for Fruits)
    private int n2944;
    private int[] prices2944;
    private int[][] memo2944;

    public int minimumCoins(int[] prices) {
        this.n2944 = prices.length;
        this.prices2944 = prices;
        this.memo2944 = new int[n2944][n2944];
        for (int i = 0; i < n2944; ++i) {
            Arrays.fill(memo2944[i], -1);
        }
        return dfs2944(0, 0);

    }

    private int dfs2944(int i, int j) {
        if (i == n2944) {
            return 0;
        }
        if (n2944 - i <= j) {
            return 0;
        }
        if (memo2944[i][j] != -1) {
            return memo2944[i][j];
        }
        int res = dfs2944(i + 1, i + 1) + prices2944[i];
        if (j > 0) {
            res = Math.min(res, dfs2944(i + 1, j - 1));
        }
        return memo2944[i][j] = res;
    }

    // 2951. 找出峰值 (Find the Peaks)
    public List<Integer> findPeaks(int[] mountain) {
        List<Integer> res = new ArrayList<>();
        for (int i = 1; i < mountain.length - 1; ++i) {
            if (mountain[i] > mountain[i - 1] && mountain[i] > mountain[i + 1]) {
                res.add(i);
            }
        }
        return res;

    }

    // 2952. 需要添加的硬币的最小数量 (Minimum Number of Coins to be Added)
    public int minimumAddedCoins(int[] coins, int target) {
        Arrays.sort(coins);
        int res = 0;
        int s = 1;
        int n = coins.length;
        int i = 0;
        while (s <= target) {
            if (i < n && coins[i] <= s) {
                s += coins[i];
                ++i;
            } else {
                s *= 2;
                res += 1;
            }
        }
        return res;
    }

    // 2953. 统计完全子字符串 (Count Complete Substrings)
    public int countCompleteSubstrings(String word, int k) {
        int n = word.length();
        int i = 0;
        int res = 0;
        while (i < n) {
            int j = i + 1;
            while (j < n && Math.abs(word.charAt(j) - word.charAt(j - 1)) <= 2) {
                ++j;
            }
            String s = word.substring(i, j);
            for (int c = 1; c <= 26; ++c) {
                int m = c * k;
                if (m > s.length()) {
                    break;
                }
                int[] cnt = new int[26];
                for (int x = 0; x < s.length(); ++x) {
                    ++cnt[s.charAt(x) - 'a'];
                    if (x >= m) {
                        --cnt[s.charAt(x - m) - 'a'];
                    }
                    if (x >= m - 1 && check2953(cnt, k)) {
                        ++res;
                    }
                }
            }
            i = j;
        }
        return res;

    }

    private boolean check2953(int[] cnt, int k) {
        for (int i = 0; i < 26; ++i) {
            if (cnt[i] != 0) {
                if (cnt[i] != k) {
                    return false;
                }
            }
        }
        return true;
    }

    // 100130. 找到两个数组中的公共元素 (Find Common Elements Between Two Arrays)
    public int[] findIntersectionValues(int[] nums1, int[] nums2) {
        return new int[] { check100130(nums1, nums2), check100130(nums2, nums1) };
    }

    private int check100130(int[] nums1, int[] nums2) {
        Set<Integer> set = Arrays.stream(nums2).boxed().collect(Collectors.toSet());
        int res = 0;
        for (int num : nums1) {
            if (set.contains(num)) {
                ++res;
            }
        }
        return res;
    }

    // 100152. 消除相邻近似相等字符 (Remove Adjacent Almost-Equal Characters)
    public int removeAlmostEqualCharacters(String word) {
        int n = word.length();
        int i = 0;
        int res = 0;
        while (i < n) {
            int j = i + 1;
            while (j < n && Math.abs(word.charAt(j) - word.charAt(j - 1)) <= 1) {
                ++j;
            }
            res += (j - i) / 2;
            i = j;
        }
        return res;

    }

    // 2958. 最多 K 个重复元素的最长子数组 (Length of Longest Subarray With at Most K Frequency)
    public int maxSubarrayLength(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        int res = 0;
        int n = nums.length;
        int i = 0;
        int j = 0;
        while (i < n) {
            map.merge(nums[i], 1, Integer::sum);
            while (map.get(nums[i]) > k) {
                map.merge(nums[j], -1, Integer::sum);
                ++j;
            }
            res = Math.max(res, i - j + 1);
            ++i;
        }
        return res;

    }

    // 100140. 关闭分部的可行集合数目 (Number of Possible Sets of Closing Branches)
    private int n100140;
    private int maxDistance100140;
    private List<int[]>[] g100140;

    public int numberOfSets(int n, int maxDistance, int[][] roads) {
        this.n100140 = n;
        this.maxDistance100140 = maxDistance;
        this.g100140 = new ArrayList[n];
        Arrays.setAll(g100140, k -> new ArrayList<>());
        for (int[] r : roads) {
            int u = r[0];
            int v = r[1];
            int w = r[2];
            g100140[u].add(new int[] { v, w });
            g100140[v].add(new int[] { u, w });
        }
        int res = 0;
        // i 删除的点
        for (int i = 0; i < (1 << n); ++i) {
            if (Integer.bitCount(i) >= n - 1 || check100140(i)) {
                ++res;
            }
        }
        return res;
    }

    private boolean check100140(int mask) {
        for (int i = 0; i < n100140; ++i) {
            if ((mask & (1 << i)) == 0) {
                if (!dijkstra100140(i, mask)) {
                    return false;
                }
            }
        }
        return true;
    }

    private boolean dijkstra100140(int start, int mask) {
        Queue<Integer> q = new PriorityQueue<>();
        int[] dis = new int[n100140];
        Arrays.fill(dis, Integer.MAX_VALUE);
        dis[start] = 0;
        q.offer(start);
        while (!q.isEmpty()) {
            int x = q.poll();
            for (int[] nei : g100140[x]) {
                int y = nei[0];
                int neiD = nei[1];
                if (((1 << y) & mask) == 0 && dis[x] + neiD < dis[y]) {
                    dis[y] = dis[x] + neiD;
                    q.offer(y);
                }
            }
        }
        for (int i = 0; i < n100140; ++i) {
            if ((mask & (1 << i)) == 0 && dis[i] > maxDistance100140) {
                return false;
            }
        }
        return true;
    }

    // 100143. 统计已测试设备 (Count Tested Devices After Test Operations)
    public int countTestedDevices(int[] batteryPercentages) {
        int cnt = 0;
        for (int b : batteryPercentages) {
            if (b - cnt > 0) {
                cnt += 1;
            }
        }
        return cnt;
    }

    // 100155. 双模幂运算 (Double Modular Exponentiation)
    public List<Integer> getGoodIndices(int[][] variables, int target) {
        List<Integer> res = new ArrayList<>();
        int n = variables.length;
        for (int i = 0; i < n; ++i) {
            int a = variables[i][0];
            int b = variables[i][1];
            int c = variables[i][2];
            int m = variables[i][3];
            int x = pow100155(a, b, 10);
            if (pow100155(x, c, m) == target) {
                res.add(i);
            }
        }
        return res;
    }

    public int pow100155(int a, int b, int m) {
        if (b == 0) {
            return 1;
        }
        int res = pow100155(a, b >> 1, m);
        res = (int) ((long) res * res % m);
        if ((b & 1) == 1) {
            res = (int) (((long) res * a) % m);
        }
        return res;
    }

    // 2962. 统计最大元素出现至少 K 次的子数组 (Count Subarrays Where Max Element Appears at
    // Least K Times)
    public long countSubarrays(int[] nums, int k) {
        int max = Arrays.stream(nums).max().getAsInt();
        int n = nums.length;
        long res = 0L;
        int i = 0;
        int j = 0;
        int cnt = 0;
        while (i < n) {
            if (nums[i] == max) {
                ++cnt;
            }
            while (cnt == k) {
                if (nums[j] == max) {
                    --cnt;
                }
                ++j;
            }
            res += j;
            ++i;
        }
        return res;
    }

    // 100136. 统计好分割方案的数目 (Count the Number of Good Partitions)
    public int numberOfGoodPartitions(int[] nums) {
        Map<Integer, Integer> first = new HashMap<>();
        Map<Integer, Integer> last = new HashMap<>();
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            first.putIfAbsent(nums[i], i);
            last.put(nums[i], i);
        }
        int j = 0;
        int cnt = 0;
        while (j < n) {
            int left = first.get(nums[j]);
            int right = last.get(nums[j]);
            int k = j;
            while (k < n && !(first.get(nums[k]) > right || last.get(nums[k]) < left)) {
                left = Math.min(left, first.get(nums[k]));
                right = Math.max(right, last.get(nums[k]));
                ++k;
            }
            cnt += 1;
            j = k;
        }
        return pow100136(2, cnt - 1);
    }

    private int pow100136(int a, int b) {
        if (b == 0) {
            return 1;
        }
        int res = pow100136(a, b >> 1);
        final int MOD = (int) (1e9 + 7);
        res = (int) ((long) res * res % MOD);
        if ((b & 1) == 1) {
            res = (int) ((long) res * a % MOD);
        }
        return res;
    }

    // 2454. 下一个更大元素 IV (Next Greater Element IV)
    public int[] secondGreaterElement(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        Arrays.fill(res, -1);
        Integer[] ids = IntStream.range(0, n).boxed().toArray(Integer[]::new);
        Arrays.sort(ids, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(nums[o2], nums[o1]);
            }

        });
        TreeSet<Integer> treeSet = new TreeSet<>();
        for (int id : ids) {
            Integer j = treeSet.higher(id);
            if (j != null) {
                Integer k = treeSet.higher(j);
                if (k != null) {
                    res[id] = nums[k];
                }
            }
            treeSet.add(id);
        }
        return res;
    }

    // 2132. 用邮票贴满网格图 (Stamping the Grid)
    public boolean possibleToStamp(int[][] grid, int stampHeight, int stampWidth) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] p = new int[m + 1][n + 1];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                p[i + 1][j + 1] = p[i][j + 1] + p[i + 1][j] - p[i][j] + grid[i][j];
            }
        }
        int[][] d = new int[m + 2][n + 2];
        for (int i2 = stampHeight; i2 <= m; ++i2) {
            for (int j2 = stampWidth; j2 <= n; ++j2) {
                int i1 = i2 - stampHeight + 1;
                int j1 = j2 - stampWidth + 1;
                if (p[i2][j2] - p[i2][j1 - 1] - p[i1 - 1][j2] + p[i1 - 1][j1 - 1] == 0) {
                    ++d[i1][j1];
                    --d[i1][j2 + 1];
                    --d[i2 + 1][j1];
                    ++d[i2 + 1][j2 + 1];
                }
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                d[i + 1][j + 1] += d[i][j + 1] + d[i + 1][j] - d[i][j];
                if (grid[i][j] == 0 && d[i + 1][j + 1] == 0) {
                    return false;
                }
            }
        }
        return true;

    }

    // 2965. 找出缺失和重复的数字 (Find Missing and Repeated Values)
    public int[] findMissingAndRepeatedValues(int[][] grid) {
        int n = grid.length;
        int[] cnt = new int[n * n + 1];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                ++cnt[grid[i][j]];
            }
        }
        int[] res = new int[2];
        for (int i = 1; i < n * n + 1; ++i) {
            if (cnt[i] == 2) {
                res[0] = i;
            } else if (cnt[i] == 0) {
                res[1] = i;
            }
        }
        return res;

    }

    // 2966. 划分数组并满足最大差限制 (Divide Array Into Arrays With Max Difference)
    public int[][] divideArray(int[] nums, int k) {
        int n = nums.length;
        int[][] res = new int[n / 3][3];
        Arrays.sort(nums);
        for (int i = 0; i < n; i += 3) {
            if (nums[i + 2] - nums[i] > k) {
                return new int[0][0];
            }
            res[i / 3] = new int[] { nums[i], nums[i + 1], nums[i + 2] };
        }
        return res;
    }

    // 2967. 使数组成为等数数组的最小代价 (Minimum Cost to Make Array Equalindromic)
    public long minimumCost(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        int m = nums[n / 2];
        return Math.min(check2967(nums, m, -1, 0), check2967(nums, m, 1, (int) 1e9));
    }

    private long check2967(int[] nums, int m, int step, int limit) {
        while (m != limit) {
            if (checkPalindromes2967(m)) {
                long res = 0L;
                for (int num : nums) {
                    res += (long) Math.abs(num - m);
                }
                return res;
            }
            m += step;
        }
        return Long.MAX_VALUE;

    }

    public boolean checkPalindromes2967(int num) {
        char[] arr = String.valueOf(num).toCharArray();
        int n = arr.length;
        int i = 0;
        int j = n - 1;
        while (i < j) {
            if (arr[i] != arr[j]) {
                return false;
            }
            ++i;
            --j;
        }
        return true;
    }

    // 2968. 执行操作使频率分数最大 (Apply Operations to Maximize Frequency Score)
    public int maxFrequencyScore(int[] nums, long k) {
        Arrays.sort(nums);
        int left = 1;
        int right = nums.length;
        int res = 1;
        int n = nums.length;
        long[] pre = new long[n + 1];
        for (int i = 0; i < n; ++i) {
            pre[i + 1] = pre[i] + nums[i];
        }
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (check2968(mid, nums, k, pre)) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;

    }

    private boolean check2968(int w, int[] nums, long k, long[] pre) {
        int n = nums.length;
        for (int i = w - 1; i < n; ++i) {
            int l = i - w + 1;
            int r = i;
            int mid = (r + l) / 2;
            if (pre[r + 1] - pre[mid] - (r - mid + 1) * nums[mid] + (mid - l + 1) * nums[mid]
                    - (pre[mid + 1] - pre[l]) <= k) {
                return true;
            }
        }
        return false;
    }

    // 2974. 最小数字游戏 (Minimum Number Game)
    public int[] numberGame(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        for (int i = 0; i < n; i += 2) {
            int temp = nums[i];
            nums[i] = nums[i + 1];
            nums[i + 1] = temp;
        }
        return nums;

    }

    // 2975. 移除栅栏得到的正方形田地的最大面积 (Maximum Square Area by Removing Fences From a Field)
    public int maximizeSquareArea(int m, int n, int[] hFences, int[] vFences) {
        Set<Integer> hSet = check2975(hFences, m);
        Set<Integer> vSet = check2975(vFences, n);
        long res = -1L;
        for (int h : hSet) {
            if (vSet.contains(h)) {
                res = Math.max(res, (long) h * h);
            }
        }
        final int MOD = (int) 1e9 + 7;
        return (int) (res % MOD);

    }

    private Set<Integer> check2975(int[] nums, int n) {
        List<Integer> list = new ArrayList<>();
        list.add(1);
        list.add(n);
        for (int num : nums) {
            list.add(num);
        }
        Collections.sort(list);
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < list.size(); ++i) {
            for (int j = i + 1; j < list.size(); ++j) {
                set.add(list.get(j) - list.get(i));
            }
        }
        return set;

    }

    // 2976. 转换字符串的最小成本 I (Minimum Cost to Convert String I)
    private List<int[]>[] g2976;

    public long minimumCost(String source, String target, char[] original, char[] changed, int[] cost) {
        this.g2976 = new ArrayList[26];
        Arrays.setAll(g2976, k -> new ArrayList<>());
        for (int i = 0; i < original.length; ++i) {
            g2976[original[i] - 'a'].add(new int[] { changed[i] - 'a', cost[i] });
        }
        int[][] dis = new int[26][26];
        for (int i = 0; i < 26; ++i) {
            dis[i] = dijkstra2976(i);
        }
        long res = 0L;
        for (int i = 0; i < source.length(); ++i) {
            int c1 = source.charAt(i) - 'a';
            int c2 = target.charAt(i) - 'a';
            if (dis[c1][c2] == Integer.MAX_VALUE) {
                return -1L;
            }
            res += dis[c1][c2];
        }
        return res;

    }

    private int[] dijkstra2976(int start) {
        int[] dis = new int[26];
        Arrays.fill(dis, Integer.MAX_VALUE);
        dis[start] = 0;
        Queue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[1], o2[1]);
            }

        });
        q.offer(new int[] { start, 0 });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int d = cur[1];
            for (int[] nei : g2976[x]) {
                int y = nei[0];
                int neiD = nei[1];
                if (d + neiD < dis[y]) {
                    dis[y] = d + neiD;
                    q.offer(new int[] { y, dis[y] });
                }
            }
        }
        return dis;
    }

    // 2970. 统计移除递增子数组的数目 I (Count the Number of Incremovable Subarrays I)
    // 2972. 统计移除递增子数组的数目 II (Count the Number of Incremovable Subarrays II)
    public long incremovableSubarrayCount(int[] nums) {
        long res = 0L;
        int n = nums.length;
        int i = 0;
        while (i < n - 1) {
            if (nums[i] >= nums[i + 1]) {
                break;
            }
            ++i;
        }
        // 整个数组单调递增
        if (i == n - 1) {
            return (1 + n) * n / 2;
        }
        res += i + 1;
        int j = n - 1;
        while (j >= 1) {
            if (nums[j - 1] >= nums[j]) {
                break;
            }
            --j;
        }
        res += n - j;
        res += 1;
        int x = 0;
        while (x <= i && j < n) {
            while (j < n && nums[x] >= nums[j]) {
                ++j;
            }
            res += n - j;
            ++x;
        }
        return res;

    }

    // 2971. 找到最大周长的多边形 (Find Polygon With the Largest Perimeter)
    public long largestPerimeter(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        long sum = 0L;
        for (int num : nums) {
            sum += num; 
        }
        for (int i = n - 1; i >= 2; --i) {
            if (sum - nums[i] > nums[i]) {
                return sum;
            }
            sum -= nums[i];
        }
        return -1L;
    }

    // 2973. 树中每个节点放置的金币数目 (Find Number of Coins to Place in Tree Nodes)
    private List<Integer>[] g2973;
    private int[] cost2973;
    private long[] res2973;

    public long[] placedCoins(int[][] edges, int[] cost) {
        int n = cost.length;
        this.g2973 = new ArrayList[n];
        this.cost2973 = cost;
        Arrays.setAll(g2973, k -> new ArrayList<>());
        for (int[] e : edges) {
            int u = e[0];
            int v = e[1];
            g2973[u].add(v);
            g2973[v].add(u);
        }
        this.res2973 = new long[n];
        dfs2973(0, -1);
        return res2973;
    }

    private List<Integer> dfs2973(int x, int fa) {
        List<Integer> list = new ArrayList<>();
        list.add(cost2973[x]);
        for (int y : g2973[x]) {
            if (y != fa) {
                list.addAll(dfs2973(y, x));
            }
        }
        Collections.sort(list);
        if (list.size() < 3) {
            res2973[x] = 1L;
        } else {
            res2973[x] = Math.max((long) list.get(0) * list.get(1) * list.get(list.size() - 1),
                    (long) list.get(list.size() - 1) * list.get(list.size() - 2) * list.get(list.size() - 3));
            res2973[x] = Math.max(0L, res2973[x]);
        }
        if (list.size() >= 5) {
            list = List.of(list.get(0), list.get(1), list.get(list.size() - 1), list.get(list.size() - 2),
                    list.get(list.size() - 3));
        }
        return list;
    }

    // 2977. 转换字符串的最小成本 II (Minimum Cost to Convert String II)
    private int id2977;
    private long[][] dic2977;
    private int m2977;
    private String source2977;
    private String target2977;
    private long[] memo2977;
    private int n2977;
    private Trie2977 trie2977;

    public long minimumCost(String source, String target, String[] original, String[] changed, int[] cost) {
        Set<String> set = new HashSet<>();
        for (int i = 0; i < original.length; ++i) {
            set.add(original[i]);
            set.add(changed[i]);
        }
        this.m2977 = set.size();
        this.dic2977 = new long[m2977][m2977];
        for (int i = 0; i < m2977; ++i) {
            Arrays.fill(dic2977[i], Long.MAX_VALUE / 2);
        }
        this.trie2977 = new Trie2977();
        for (int i = 0; i < original.length; ++i) {
            int oId = trie2977.insert(original[i]);
            int cId = trie2977.insert(changed[i]);
            dic2977[oId][oId] = 0;
            dic2977[cId][cId] = 0;
            dic2977[oId][cId] = Math.min(dic2977[oId][cId], cost[i]);
        }
        /**
         * Floyd
         */
        // for (int k = 0; k < m2977; ++k) {
        //     for (int i = 0; i < m2977; ++i) {
        //         if (dic2977[i][k] == Long.MAX_VALUE / 2) {
        //             continue;
        //         }
        //         for (int j = 0; j < m2977; ++j) {
        //             dic2977[i][j] = Math.min(dic2977[i][j], dic2977[i][k] + dic2977[k][j]);
        //         }
        //     }
        
        // }
        for (int i = 0; i < m2977; ++i) {
            dic2977[i] = dijkstra2977(i);
        }
        this.source2977 = source;
        this.target2977 = target;
        this.n2977 = source.length();
        this.memo2977 = new long[n2977];
        Arrays.fill(memo2977, -1L);
        long res = dfs2977(0);
        if (res < Long.MAX_VALUE / 2) {
            return res;
        }
        return -1L;
    }

    private long dfs2977(int i) {
        if (i == n2977) {
            return 0L;
        }
        if (memo2977[i] != -1L) {
            return memo2977[i];
        }
        long res = Long.MAX_VALUE / 2;
        if (source2977.charAt(i) == target2977.charAt(i)) {
            res = dfs2977(i + 1);
        }
        for (int[] j : trie2977.check(source2977.substring(i), target2977.substring(i))) {
            res = Math.min(res, dfs2977(i + j[0] + 1) + dic2977[j[1]][j[2]]);
        }
        return memo2977[i] = res;
    }

    private long[] dijkstra2977(int start) {
        long[] dis = new long[m2977];
        Arrays.fill(dis, Long.MAX_VALUE / 2);
        dis[start] = 0L;
        Queue<long[]> q = new PriorityQueue<>(new Comparator<long[]>() {

            @Override
            public int compare(long[] o1, long[] o2) {
                return Long.compare(o1[1], o2[1]);
            }

        });
        q.offer(new long[] { start, 0L });
        while (!q.isEmpty()) {
            long[] cur = q.poll();
            int x = (int) cur[0];
            long d = cur[1];
            for (int y = 0; y < m2977; ++y) {
                if (d + dic2977[x][y] < dis[y]) {
                    dis[y] = d + dic2977[x][y];
                    q.offer(new long[] { y, dis[y] });
                }
            }
        }
        return dis;
    }

    public class Trie2977 {
        private Trie2977[] children;
        private int i;

        public Trie2977() {
            this.children = new Trie2977[26];
            this.i = -1;
        }

        public int insert(String s) {
            Trie2977 node = this;
            for (char c : s.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new Trie2977();
                }
                node = node.children[index];
            }
            if (node.i == -1) {
                node.i = id2977++;
            }
            return node.i;
        }

        public List<int[]> check(String s, String t) {
            List<int[]> res = new ArrayList<>();
            Trie2977 node1 = this;
            Trie2977 node2 = this;
            for (int i = 0; i < s.length(); ++i) {
                int index1 = s.charAt(i) - 'a';
                int index2 = t.charAt(i) - 'a';
                if (node1.children[index1] == null || node2.children[index2] == null) {
                    break;
                }
                node1 = node1.children[index1];
                node2 = node2.children[index2];
                if (node1.i == -1 || node2.i == -1) {
                    continue;
                }
                if (dic2977[node1.i][node2.i] < Long.MAX_VALUE / 2) {
                    res.add(new int[] { i, node1.i, node2.i });
                }
            }
            return res;
        }

    }

    // 2772. 使数组中的所有元素都等于零 (Apply Operations to Make All Array Elements Equal to
    // Zero)
    public boolean checkArray(int[] nums, int k) {
        long sumD = 0L;
        int n = nums.length;
        long[] diff = new long[n + 1];
        for (int i = 0; i < n; ++i) {
            sumD += diff[i];
            int x = nums[i];
            x += sumD;
            if (x == 0) {
                continue;
            }
            if (x < 0 || i + k > n) {
                return false;
            }
            sumD -= x;
            diff[i + k] += x;
        }
        return true;
    }

    // 1316. 不同的循环子字符串 (Distinct Echo Substrings)
    public int distinctEchoSubstrings(String text) {
        final long BASE = 31L;
        final long MOD = (long) (1e9 + 7);
        int n = text.length();
        int[] pre = new int[n + 1];
        int[] mul = new int[n + 1];
        mul[0] = 1;
        for (int i = 0; i < n; ++i) {
            pre[i + 1] = (int) (((pre[i] * BASE % MOD + text.charAt(i) - 'a') + MOD) % MOD);
            mul[i + 1] = (int) (mul[i] * BASE % MOD);
        }
        int res = 0;
        Set<Integer>[] set = new HashSet[n];
        Arrays.setAll(set, k -> new HashSet<>());
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                int l = j - i;
                if (j + l <= n) {
                    int leftHash = hash1316(i, j - 1, pre, mul, MOD);
                    if (!set[l - 1].contains(leftHash) && leftHash == hash1316(j, j + l - 1, pre, mul, MOD)) {
                        ++res;
                        set[l - 1].add(leftHash);
                    }
                }
            }
        }
        return res;

    }

    private int hash1316(int i, int j, int[] pre, int[] mul, long MOD) {
        return (int) ((pre[j + 1] - (long) pre[i] * mul[j - i + 1] % MOD + MOD) % MOD);
    }

    // 1937. 扣分后的最大得分 (Maximum Number of Points with Cost)
    public long maxPoints(int[][] points) {
        int m = points.length;
        int n = points[0].length;
        long[] dp = new long[n];
        for (int i = 0; i < m; ++i) {
            long max = Long.MIN_VALUE;
            long[] mx = new long[n];
            for (int j = 0; j < n; ++j) {
                max = Math.max(max, dp[j] + j);
                mx[j] = Math.max(mx[j], max + points[i][j] - j);
            }
            max = Long.MIN_VALUE;
            for (int j = n - 1; j >= 0; --j) {
                max = Math.max(max, dp[j] - j);
                mx[j] = Math.max(mx[j], max + points[i][j] + j);
            }
            dp = mx;
        }
        long res = 0L;
        for (int j = 0; j < n; ++j) {
            res = Math.max(res, dp[j]);
        }
        return res;

    }

    // 2981. 找出出现至少三次的最长特殊子字符串 I (Find Longest Special Substring That Occurs Thrice
    // I)
    // 2982. 找出出现至少三次的最长特殊子字符串 II (Find Longest Special Substring That Occurs Thrice
    // II) --二分 python超时 Java通过
    public int maximumLength(String s) {
        int res = -1;
        int left = 1;
        int n = s.length();
        int right = n;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (check2982(mid, s)) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    private boolean check2982(int w, String s) {
        int[] cnts = new int[26];
        int m = 0;
        int n = s.length();
        int[] ret = new int[26];
        for (int i = 0; i < n; ++i) {
            int index = s.charAt(i) - 'a';
            ++cnts[index];
            m |= 1 << index;
            if (i >= w) {
                --cnts[s.charAt(i - w) - 'a'];
                if (cnts[s.charAt(i - w) - 'a'] == 0) {
                    m ^= 1 << (s.charAt(i - w) - 'a');
                }
            }
            if (i >= w - 1) {
                // (m & -m) == m
                // Integer.bitCount(m) == 1
                if ((m & (m - 1)) == 0) {
                    if (++ret[index] == 3) {
                         return true;
                    }
                }
            }
        }
        return false;
    }

    // 2981. 找出出现至少三次的最长特殊子字符串 I (Find Longest Special Substring That Occurs Thrice
    // I) --分类统计
    // 2982. 找出出现至少三次的最长特殊子字符串 II (Find Longest Special Substring That Occurs Thrice
    // II)
    public int maximumLength2(String s) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        int n = s.length();
        int cnt = 0;
        for (int i = 0; i < n; ++i) {
            ++cnt;
            if (i == n - 1 || s.charAt(i) != s.charAt(i + 1)) {
                map.computeIfAbsent(s.charAt(i) - 'a', k -> new ArrayList<>()).add(cnt);
                cnt = 0;
            }
        }
        int res = 0;
        for (List<Integer> list : map.values()) {
            list.addAll(List.of(0, 0));
            Collections.sort(list, new Comparator<Integer>() {

                @Override
                public int compare(Integer o1, Integer o2) {
                    return Integer.compare(o2, o1);
                }

            });
            res = Math.max(res, list.get(0) - 2);
            res = Math.max(res, Math.min(list.get(0) - 1, list.get(1)));
            res = Math.max(res, list.get(2));
        }
        return res == 0 ? -1 : res;

    }

    // 2980. 检查按位或是否存在尾随零 (Check if Bitwise OR Has Trailing Zeros)
    public boolean hasTrailingZeros(int[] nums) {
        int cnt = 0;
        for (int num : nums) {
            if (num % 2 == 0) {
                if (++cnt >= 2) {
                    return true;
                }
            }
        }
        return false;

    }

    // 2996. 大于等于顺序前缀和的最小缺失整数 (Smallest Missing Integer Greater Than Sequential
    // Prefix Sum)
    public int missingInteger(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        int s = nums[0];
        for (int i = 1; i < nums.length; ++i) {
            if (nums[i] - nums[i - 1] != 1) {
                break;
            }
            s += nums[i];
        }
        while (set.contains(s)) {
            ++s;
        }
        return s;

    }

    // 2997. 使数组异或和等于 K 的最少操作次数 (Minimum Number of Operations to Make Array XOR
    // Equal to K)
    public int minOperations(int[] nums, int k) {
        for (int num : nums) {
            k ^= num;
        }
        return Integer.bitCount(k);

    }

    // 2998. 使 X 和 Y 相等的最少操作次数 (Minimum Number of Operations to Make X and Y Equal)
    public int minimumOperationsToMakeEqual(int x, int y) {
        if (x <= y) {
            return y - x;
        }
        int res = x - y;
        Queue<Integer> q = new LinkedList<>();
        Set<Integer> set = new HashSet<>();
        set.add(x);
        q.offer(x);
        int step = 0;
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                int v = q.poll();
                if (v == y) {
                    return Math.min(res, step);
                }
                if (!set.contains(v + 1)) {
                    set.add(v + 1);
                    q.offer(v + 1);
                }
                if (v - 1 < y) {
                    res = Math.min(res, step + y - v + 2);
                } else if (!set.contains(v - 1)) {
                    set.add(v - 1);
                    q.offer(v - 1);
                }
                if (v % 5 == 0) {
                    if (v / 5 < y) {
                        res = Math.min(res, step + y - v / 5 + 1);
                    } else if (!set.contains(v / 5)) {
                        set.add(v / 5);
                        q.offer(v / 5);
                    }
                }
                if (v % 11 == 0) {
                    if (v / 11 < y) {
                        res = Math.min(res, step + y - v / 11 + 1);
                    } else if (!set.contains(v / 11)) {
                        set.add(v / 11);
                        q.offer(v / 11);
                    }
                }
            }
            ++step;
        }
        return -1;

    }

    // 2998. 使 X 和 Y 相等的最少操作次数 (Minimum Number of Operations to Make X and Y Equal)
    private int[] memo2998;

    public int minimumOperationsToMakeEqual2(int x, int y) {
        if (x <= y) {
            return y - x;
        }
        int d = x - y;
        this.memo2998 = new int[d + x + 1];
        Arrays.fill(memo2998, -1);
        return dfs2998(x, y);

    }

    private int dfs2998(int x, int y) {
        if (x <= y) {
            return y - x;
        }
        if (memo2998[x] != -1) {
            return memo2998[x];
        }
        int res = x - y;
        res = Math.min(res, dfs2998(x / 11, y) + x % 11 + 1);
        res = Math.min(res, dfs2998(x / 11 + 1, y) + 11 - x % 11 + 1);
        res = Math.min(res, dfs2998(x / 5, y) + x % 5 + 1);
        res = Math.min(res, dfs2998(x / 5 + 1, y) + 5 - x % 5 + 1);
        return memo2998[x] = res;
    }

    // 2999. 统计强大整数的数目 (Count the Number of Powerful Integers)
    private int limit2999;
    private String s2999;

    public long numberOfPowerfulInt(long start, long finish, int limit, String s) {
        this.limit2999 = limit;
        this.s2999 = s;
        return check2999(String.valueOf(finish)) - check2999(String.valueOf(start - 1));

    }

    private long[] memo2999;
    private String num2999;

    private long check2999(String num) {
        if (num.length() < s2999.length()) {
            return 0L;
        }
        this.memo2999 = new long[num.length()];
        this.num2999 = num;
        Arrays.fill(memo2999, -1L);
        return dfs2999(0, true, false);

    }

    // 可以去掉 isNum
    private long dfs2999(int i, boolean isLimit, boolean isNum) {
        if (num2999.length() - i == s2999.length()) {
            return isLimit ? (Long.parseLong(num2999.substring(i)) >= Long.parseLong(s2999) ? 1 : 0) : 1;
        }
        if (!isLimit && isNum && memo2999[i] != -1L) {
            return memo2999[i];
        }
        long res = 0L;
        if (!isNum) {
            res = dfs2999(i + 1, false, false);
        }
        int up = isLimit ? (num2999.charAt(i) - '0') : 9;
        for (int d = isNum ? 0 : 1; d <= Math.min(limit2999, up); ++d) {
            res += dfs2999(i + 1, isLimit && d == up, true);
        }
        if (!isLimit && isNum) {
            memo2999[i] = res;
        }
        return res;
    }

    // 3000. 对角线最长的矩形的面积 (Maximum Area of Longest Diagonal Rectangle)
    public int areaOfMaxDiagonal(int[][] dimensions) {
        int res = 0;
        int maxD = 0;
        for (int[] d : dimensions) {
            int curD = d[0] * d[0] + d[1] * d[1];
            if (curD > maxD) {
                maxD = curD;
                res = d[0] * d[1];
            } else if (curD == maxD) {
                res = Math.max(res, d[0] * d[1]);
            }
        }
        return res;

    }

    // 3001. 捕获黑皇后需要的最少移动次数 (Minimum Moves to Capture The Queen)
    public int minMovesToCaptureTheQueen(int a, int b, int c, int d, int e, int f) {
        if (check3001(a, b, c, d, e, f, 0, 1) || check3001(a, b, c, d, e, f, 0, -1) || check3001(a, b, c, d, e, f, 1, 0)
                || check3001(a, b, c, d, e, f, -1, 0)) {
            return 1;
        }
        if (check3001(c, d, a, b, e, f, 1, 1) || check3001(c, d, a, b, e, f, 1, -1) || check3001(c, d, a, b, e, f, -1, 1)
                || check3001(c, d, a, b, e, f, -1, -1)) {
            return 1;
        }
        return 2;

    }

    private boolean check3001(int a, int b, int c, int d, int e, int f, int dx, int dy) {
        while (e >= 1 && e <= 8 && f >= 1 && f <= 8) {
            if (e == c && f == d) {
                return false;
            } else if (e == a && f == b) {
                return true;
            }
            e += dx;
            f += dy;
        }
        return false;
    }

    // 3002. 移除后集合的最多元素数 (Maximum Size of a Set After Removals)
    public int maximumSetSize(int[] nums1, int[] nums2) {
        Set<Integer> set1 = Arrays.stream(nums1).boxed().collect(Collectors.toSet());
        Set<Integer> set2 = Arrays.stream(nums2).boxed().collect(Collectors.toSet());
        int n = nums1.length;
        int i = 0;
        while (i < n && set1.size() > n / 2) {
            if (set2.contains(nums1[i])) {
                set1.remove(nums1[i]);
            }
            ++i;
        }
        i = 0;
        while (i < n && set2.size() > n / 2) {
            if (set1.contains(nums2[i])) {
                set2.remove(nums2[i]);
            }
            ++i;
        }
        if (set1.size() <= n / 2 && set2.size() <= n / 2) {
            set1.addAll(set2);
            return set1.size();
        }
        return Math.min(set1.size(), n / 2) + Math.min(set2.size(), n / 2);

    }

    // 3003. 执行操作后的最大分割数量 (Maximize the Number of Partitions After Operations)
    private Map<Long, Integer> memo3003;
    private int n3003;
    private char[] arr3003;
    private int k3003;

    public int maxPartitionsAfterOperations(String s, int k) {
        this.memo3003 = new HashMap<>();
        this.n3003 = s.length();
        this.arr3003 = s.toCharArray();
        this.k3003 = k;
        return dfs3003(0, 0, 0);
    }

    private int dfs3003(int i, int mask, int changed) {
        if (i == n3003) {
            return 1;
        }
        long total = ((long) i << 30) | mask | (changed << 28);
        if (memo3003.containsKey(total)) {
            return memo3003.get(total);
        }
        int res = 0;
        // 不变
        int bits = mask | (1 << (arr3003[i] - 'a'));
        if (Integer.bitCount(bits) <= k3003) {
            res = Math.max(res, dfs3003(i + 1, bits, changed));
        } else {
            res = Math.max(res, dfs3003(i + 1, 1 << (arr3003[i] - 'a'), changed) + 1);
        }
        // 变
        if (changed == 0) {
            for (int j = 0; j < 26; ++j) {
                bits = mask | (1 << j);
                if (Integer.bitCount(bits) <= k3003) {
                    res = Math.max(res, dfs3003(i + 1, bits, 1));
                } else {
                    res = Math.max(res, dfs3003(i + 1, 1 << j, 1) + 1);
                }
            }
        }
        memo3003.put(total, res);
        return res;
    }

    // 3005. 最大频率元素计数 (Count Elements With Maximum Frequency)
    public int maxFrequencyElements(int[] nums) {
        int[] cnts = new int[101];
        int max = 0;
        int res = 0;
        for (int num : nums) {
            ++cnts[num];
            if (cnts[num] > max) {
                res = cnts[num];
                max = cnts[num];
            } else if (cnts[num] == max) {
                res += cnts[num];
            }
        }
        return res;

    }

    // 3006. 找出数组中的美丽下标 I (Find Beautiful Indices in the Given Array I)
    public List<Integer> beautifulIndices(String s, String a, String b, int k) {
        List<Integer> aIds = getIndices3006(s, a);
        List<Integer> bIds = getIndices3006(s, b);
        List<Integer> res = new ArrayList<>();
        int i = 0;
        int j = 0;
        while (i < aIds.size()) {
            while (j < bIds.size() && aIds.get(i) - bIds.get(j) > k) {
                ++j;
            }
            if (j < bIds.size() && Math.abs(aIds.get(i) - bIds.get(j)) <= k) {
                res.add(aIds.get(i));
            }
            ++i;
        }
        return res;

    }

    private List<Integer> getIndices3006(String s, String p) {
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i <= s.length() - p.length(); ++i) {
            if (s.substring(i, i + p.length()).equals(p)) {
                res.add(i);
            }
        }
        return res;
    }

    // 3007. 价值和小于等于 K 的最大数字 (Maximum Number That Sum of the Prices Is Less Than or
    // Equal to K)
    private int x3007;

    public long findMaximumNumber(long k, int x) {
        long left = 1L;
        long right = (long) 1e15;
        long res = 1L;
        this.x3007 = x;
        while (left <= right) {
            long mid = left + ((right - left) >> 1);
            if (check3007(mid) <= k) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;

    }

    private String s3007;
    private int n3007;
    private long[][] memo3007;

    private long check3007(long num) {
        this.s3007 = Long.toBinaryString(num);
        this.n3007 = s3007.length();
        this.memo3007 = new long[n3007][n3007];
        for (int i = 0; i < n3007; ++i) {
            Arrays.fill(memo3007[i], -1L);
        }
        return dfs3007(0, 0, true, false);
    }

    private long dfs3007(int i, int j, boolean isLimit, boolean isNum) {
        if (i == n3007) {
            return isNum ? j : 0;
        }
        if (!isLimit && isNum && memo3007[i][j] != -1L) {
            return memo3007[i][j];
        }

        long res = 0L;
        if (!isNum) {
            res = dfs3007(i + 1, j, false, false);
        }
        int up = isLimit ? (s3007.charAt(i) - '0') : 1;
        for (int d = isNum ? 0 : 1; d <= up; ++d) {
            res += dfs3007(i + 1, j + (((n3007 - i) % x3007 == 0 && d == 1) ? 1 : 0), up == d && isLimit, true);
        }
        if (!isLimit && isNum) {
            memo3007[i][j] = res;
        }
        return res;
    }

    // 2901. 最长相邻不相等子序列 II (Longest Unequal Adjacent Groups Subsequence II)
    public List<String> getWordsInLongestSubsequenceII(int n, String[] words, int[] groups) {
        List<String> res = new ArrayList<>();
        int[] f = new int[n];
        int[] fromIdx = new int[n];
        int mx = n - 1;
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                if (f[j] > f[i] && groups[i] != groups[j] && check2901(words[i], words[j])) {
                    f[i] = f[j];
                    fromIdx[i] = j;
                }
            }
            ++f[i];
            if (f[i] > f[mx]) {
                mx = i;
            }
        }
        int m = f[mx];
        for (int i = 0; i < m; ++i) {
            res.add(words[mx]);
            mx = fromIdx[mx];
        }
        return res;
    }

    private boolean check2901(String s1, String s2) {
        if (s1.length() != s2.length()) {
            return false;
        }
        int s = 0;
        for (int i = 0; i < s1.length(); ++i) {
            if (s1.charAt(i) != s2.charAt(i)) {
                if (++s > 1) {
                    return false;
                }
            }
        }
        return s == 1;
    }

    // 3010. 将数组分成最小总代价的子数组 I (Divide an Array Into Subarrays With Minimum Cost I)
    public int minimumCost3010(int[] nums) {
        Arrays.sort(nums, 1, nums.length);
        return nums[0] + nums[1] + nums[2];
    }

    // 3011. 判断一个数组是否可以变为有序 (Find if Array Can Be Sorted)
    public boolean canSortArray(int[] nums) {
        int n = nums.length;
        int i = 0;
        int pre = 0;
        while (i < n) {
            int bit = Integer.bitCount(nums[i]);
            int j = i;
            int min = nums[i];
            int max = nums[i];
            while (j < n && Integer.bitCount(nums[j]) == bit) {
                min = Math.min(min, nums[j]);
                max = Math.max(max, nums[j]);
                ++j;
            }
            if (min < pre) {
                return false;
            }
            pre = max;
            i = j;
        }
        return true;

    }

    // 3012. 通过操作使数组长度最小 (Minimize Length of Array Using Operations)
    public int minimumArrayLength(int[] nums) {
        int m = Integer.MAX_VALUE;
        for (int num : nums) {
            m = Math.min(m, num);
        }
        for (int num : nums) {
            if (num % m > 0) {
                return 1;
            }
        }
        int cnt = 0;
        for (int num : nums) {
            if (num == m) {
                ++cnt;
            }
        }
        return (cnt + 1) / 2;

    }

    // 3014. 输入单词需要的最少按键次数 I (Minimum Number of Pushes to Type Word I)
    public int minimumPushes(String word) {
        int res = 0;
        for (int i = 0; i < word.length(); ++i) {
            res += i / 8 + 1;
        }
        return res;
    }

    // 3015. 按距离统计房屋对数目 I (Count the Number of Houses at a Certain Distance I)
    private int[] res3015;
    private List<Integer>[] g3015;
    private int n3015;

    public int[] countOfPairs(int n, int x, int y) {
        this.g3015 = new ArrayList[n];
        this.n3015 = n;
        Arrays.setAll(g3015, k -> new ArrayList<>());
        for (int i = 2; i <= n; ++i) {
            g3015[i - 1].add(i - 2);
            g3015[i - 2].add(i - 1);
        }
        if (x != y) {
            g3015[x - 1].add(y - 1);
            g3015[y - 1].add(x - 1);
        }
        this.res3015 = new int[n];
        for (int i = 0; i < n; ++i) {
            for (int d : check3015(i)) {
                if (d - 1 >= 0) {
                    ++res3015[d - 1];
                }
            }
        }
        return res3015;

    }

    private int[] check3015(int start) {
        int[] dis = new int[n3015];
        Arrays.fill(dis, -1);
        Queue<Integer> q = new LinkedList<>();
        dis[start] = 0;
        q.add(start);
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                int x = q.poll();
                for (int y : g3015[x]) {
                    if (dis[y] == -1) {
                        dis[y] = dis[x] + 1;
                        q.offer(y);
                    }
                }
            }
        }
        return dis;
    }

    // 3016. 输入单词需要的最少按键次数 II (Minimum Number of Pushes to Type Word II)
    public int minimumPushes3016(String word) {
        int[] cnts = new int[26];
        for (char c : word.toCharArray()) {
            ++cnts[c - 'a'];
        }
        Arrays.sort(cnts);
        int res = 0;
        int cnt = 0;
        for (int i = 25; i >= 0; --i) {
            res += (cnt / 8 + 1) * cnts[i];
            ++cnt;
        }
        return res;

    }

    // 3019. 按键变更的次数 (Number of Changing Keys)
    public int countKeyChanges(String s) {
        int res = 0;
        for (int i = 1; i < s.length(); ++i) {
            res += (s.charAt(i) & 31) != (s.charAt(i - 1) & 31) ? 1 : 0;
        }
        return res;
    }

    // 3020. 子集中元素的最大数量 (Find the Maximum Number of Elements in Subset)
    public int maximumLength(int[] nums) {
        Map<Long, Integer> map = new HashMap<>();
        for (long num : nums) {
            map.merge(num, 1, Integer::sum);
        }
        int cnt1 = map.getOrDefault(1L, 0);
        int res = cnt1 - ((cnt1 % 2) ^ 1);
        map.remove(1L);
        for (long num : map.keySet()) {
            int k = 1;
            int cur = 0;
            while (map.getOrDefault((long) Math.pow(num, k), 0) >= 2) {
                cur += 2;
                k <<= 1;
            }
            if (map.getOrDefault((long) Math.pow(num, k), 0) >= 1) {
                res = Math.max(res, cur + 1);
            } else {
                res = Math.max(res, cur - 1);
            }
        }
        return res;
    }

    // 3021. Alice 和 Bob 玩鲜花游戏 (Alice and Bob Playing Flower Game)
    public long flowerGame(int n, int m) {
        long even1 = n / 2;
        long even2 = m / 2;
        return even1 * (m - even2) + (n - even1) * even2;
    }
    // 2862. 完全子集的最大元素和 (Maximum Element-Sum of a Complete Subset of Indices)
    public long maximumSum(List<Integer> nums) {
        int n = nums.size();
        long[] s = new long[n + 1];
        long res = 0L;
        for (int i = 0; i < n; ++i) {
            int cur = core2862(i + 1);
            s[cur] += nums.get(i);
            res = Math.max(res, s[cur]);
        }
        return res;

    }

    private int core2862(int n) {
        int res = 1;
        for (int i = 2; i < Math.sqrt(n) + 1; ++i) {
            int e = 0;
            while (n % i == 0) {
                e ^= 1;
                n /= i;
            }
            if (e > 0) {
                res *= i;
            }
        }
        if (n > 1) {
            res *= n;
        }
        return res;
    }

    // 292. Nim 游戏 (Nim Game)
    public boolean canWinNim(int n) {
        return n % 4 != 0;
    }

    // 3024. 三角形类型 II (Type of Triangle II)
    public String triangleType(int[] nums) {
        Arrays.sort(nums);
        if (nums[0] + nums[1] <= nums[2]) {
            return "none";
        }
        if (nums[0] == nums[1] && nums[1] == nums[2]) {
            return "equilateral";
        }
        if (nums[0] == nums[1] || nums[1] == nums[2]) {
            return "isosceles";
        }
        return "scalene";

    }

    // 3026. 最大好子数组和 (Maximum Good Subarray Sum)
    public long maximumSubarraySum(int[] nums, int k) {
        int n = nums.length;
        long[] pre = new long[n + 1];
        for (int i = 0; i < n; ++i) {
            pre[i + 1] = pre[i] + nums[i];
        }
        long res = (long) -1e15;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            if (map.containsKey(nums[i])) {
                res = Math.max(res, pre[i + 1] - pre[map.get(nums[i])]);
            }
            if (!map.containsKey(nums[i] + k) || pre[map.get(nums[i] + k)] > pre[i]) {
                map.put(nums[i] + k, i);
            }
            if (!map.containsKey(nums[i] - k) || pre[map.get(nums[i] - k)] > pre[i]) {
                map.put(nums[i] - k, i);
            }
        }
        if (res == (long) -1e15) {
            return 0;
        }
        return res;

    }

    // 3025. 人员站位的方案数 I (Find the Number of Ways to Place People I)
    // 3027. 人员站位的方案数 II (Find the Number of Ways to Place People II)
    public int numberOfPairs(int[][] points) {
        int n = points.length;
        Arrays.sort(points, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] == o2[0]) {
                    return Integer.compare(o2[1], o1[1]);
                }
                return Integer.compare(o1[0], o2[0]);
            }

        });
        int res = 0;
        for (int i = 0; i < n; ++i) {
            int y = points[i][1];
            int maxy = Integer.MIN_VALUE;
            for (int j = i + 1; j < n; ++j) {
                int y0 = points[j][1];
                if (maxy < y0 && y0 <= y) {
                    ++res;
                    maxy = y0;
                }
            }
        }
        return res;

    }

    // 2932. 找出强数对的最大异或值 I (Maximum Strong Pair XOR I)
    // 2935. 找出强数对的最大异或值 II (Maximum Strong Pair XOR II) --哈希表
    public int maximumStrongPairXor(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        int highestBit = 31 - Integer.numberOfLeadingZeros(nums[n - 1]);
        int res = 0;
        int mask = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = highestBit; i >= 0; --i) {
            map.clear();
            mask |= 1 << i;
            int newRes = res | (1 << i);
            for (int y : nums) {
                int mask_y = y & mask;
                if (map.containsKey(mask_y ^ newRes) && map.get(mask_y ^ newRes) * 2 >= y) {
                    res = newRes;
                    break;
                }
                map.put(mask_y, y);
            }
        }
        return res;

    }

    // 2932. 找出强数对的最大异或值 I (Maximum Strong Pair XOR I)
    // 2935. 找出强数对的最大异或值 II (Maximum Strong Pair XOR II) -- 0-1字典树
    private int highestBit2935;
    public int maximumStrongPairXor2(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        this.highestBit2935 = 31 - Integer.numberOfLeadingZeros(nums[n - 1]);
        Trie2935 trie = new Trie2935();
        int res = 0;
        int i = 0;
        int j = 0;
        while (j < n) {
            trie.insert(nums[j]);
            while (nums[i] * 2 < nums[j]) {
                trie.delete(nums[i]);
                ++i;
            }
            res = Math.max(res, trie.check(nums[j]));
            ++j;
        }
        return res;

    }

    public class Trie2935 {
        private Trie2935[] children;
        public int cnt;

        public Trie2935() {
            this.children = new Trie2935[2];
            this.cnt = 0;
        }

        public void insert(int num) {
            Trie2935 node = this;
            for (int i = highestBit2935; i >= 0; --i) {
                int index = (num >> i) & 1;
                if (node.children[index] == null) {
                    node.children[index] = new Trie2935();
                }
                node = node.children[index];
                ++node.cnt;
            }
        }

        public void delete(int num) {
            Trie2935 node = this;
            for (int i = highestBit2935; i >= 0; --i) {
                int index = (num >> i) & 1;
                node = node.children[index];
                --node.cnt;
            }
        }

        public int check(int num) {
            Trie2935 node = this;
            int res = 0;
            for (int i = highestBit2935; i >= 0; --i) {
                int index = (num >> i) & 1;
                if (node.children[index ^ 1] != null && node.children[index ^ 1].cnt > 0) {
                    index ^= 1;
                    res |= 1 << i;
                }
                node = node.children[index];
            }
            return res;
        }
    }

    // 1686. 石子游戏 VI (Stone Game VI)
    public int stoneGameVI(int[] aliceValues, int[] bobValues) {
        int n = aliceValues.length;
        Integer[] ids = IntStream.range(0, n).boxed().toArray(Integer[]::new);
        Arrays.sort(ids, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(aliceValues[o2] + bobValues[o2], aliceValues[o1] + bobValues[o1]);
            }

        });
        int x = 0;
        int y = 0;
        for (int i = 0; i < n; ++i) {
            if (i % 2 == 0) {
                x += aliceValues[ids[i]];
            } else {
                y += bobValues[ids[i]];
            }
        }
        return Integer.compare(x - y, 0);


    }

}