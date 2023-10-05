import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
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

    private int dfs1349(int i, int preMask) {
        if (i < 0) {
            return 0;
        }
        if (memo1349[i][preMask] != -1) {
            return memo1349[i][preMask];
        }
        int mask = ((preMask << 1) | (preMask >> 1) | rows1349[i]) & u1349;
        int c = mask ^ u1349;
        // 不选
        int max = dfs1349(i - 1, 0);
        // 选
        for (int j = c; j > 0; j = (j - 1) & c) {
            if ((j & (j >> 1)) == 0) {
                max = Math.max(max, dfs1349(i - 1, j) + Integer.bitCount(j));
            }
        }
        return memo1349[i][preMask] = max;
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

    // 6892. 最大字符串配对数目 (Find Maximum Number of String Pairs)
    public int maximumNumberOfStringPairs(String[] words) {
        int res = 0;
        Set<Integer> set = new HashSet<>();
        for (String w : words) {
            int h = (w.charAt(0) - 'a') * 26 + w.charAt(1) - 'a';
            if (set.contains(h)) {
                ++res;
            }
            set.add((w.charAt(1) - 'a') * 26 + w.charAt(0) - 'a');
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

    // 6909. 最长奇偶子数组 (Longest Even Odd Subarray With Threshold)
    public int longestAlternatingSubarray(int[] nums, int threshold) {
        int n = nums.length;
        int res = 0;
        int i = 0;
        int j = 0;
        while (j < n) {
            if (nums[j] > threshold) {
                ++j;
                i = j;
                continue;
            }
            if (nums[j] % 2 == 1) {
                ++j;
                i = j;
                continue;
            }
            while (j < n && nums[j] <= threshold && (i == j || nums[j] % 2 != nums[j - 1] % 2)) {
                ++j;
            }
            res = Math.max(res, j - i);
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

    // 6913. 最长交替子序列 (Longest Alternating Subarray)
    public int alternatingSubarray(int[] nums) {
        int res = -1;
        int n = nums.length;
        s: for (int i = 0; i < n; ++i) {
            int f = 1;
            for (int j = i + 1; j < n; ++j) {
                if (nums[j] - nums[j - 1] != f) {
                    continue s;
                }
                f = -f;
                res = Math.max(res, j - i + 1);
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

    // 6899. 达到末尾下标所需的最大跳跃次数 (Maximum Number of Jumps to Reach the Last Index)
    private int[] memo6899;
    private int[] nums6899;
    private int target6899;
    private int n6899;

    public int maximumJumps(int[] nums, int target) {
        this.n6899 = nums.length;
        this.memo6899 = new int[n6899];
        Arrays.fill(memo6899, (int) -1e9);
        this.nums6899 = nums;
        this.target6899 = target;
        int res = dfs6899(0);
        return res > 0 ? res : -1;

    }

    private int dfs6899(int i) {
        if (i == n6899 - 1) {
            return 0;
        }
        if (memo6899[i] != (int) -1e9) {
            return memo6899[i];
        }
        int max = (int) -1e9;
        for (int j = i + 1; j < n6899; ++j) {
            if (Math.abs(nums6899[j] - nums6899[i]) <= target6899) {
                max = Math.max(max, dfs6899(j) + 1);
            }
        }
        return memo6899[i] = max;
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

    // 6921. 按分隔符拆分字符串 (Split Strings by Separator)
    public List<String> splitWordsBySeparator(List<String> words, char separator) {
        List<String> res = new ArrayList<>();
        for (String w : words) {
            int n = w.length();
            int i = 0;
            int j = 0;
            while (j < n) {
                while (j < n && w.charAt(j) != separator) {
                    ++j;
                }
                if (i != j) {
                    res.add(w.substring(i, j));
                }
                ++j;
                i = j;
            }
        }
        return res;

    }

    // 6915. 合并后数组中的最大元素 (Largest Element in an Array after Merge Operations)
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

    // 6900. 统计完全子数组的数目 (Count Complete Subarrays in an Array)
    public int countCompleteSubarrays(int[] nums) {
        int n = nums.length;
        int res = 0;
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        int i = 0;
        int j = 0;
        Map<Integer, Integer> map = new HashMap<>();
        while (i < n) {
            while (j < n && map.size() != set.size()) {
                map.merge(nums[j], 1, Integer::sum);
                ++j;
            }
            if (map.size() != set.size()) {
                break;
            }
            res += n - j + 1;
            map.merge(nums[i], -1, Integer::sum);
            if (map.get(nums[i]) == 0) {
                map.remove(nums[i]);
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

    // 8014. 循环增长使字符串子序列等于另一个字符串 (Make String a Subsequence Using Cyclic Increments)
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
    public boolean canBeEqual(String s1, String s2) {
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
        long sum = 0L;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.size(); ++i) {
            sum += nums.get(i);
            map.merge(nums.get(i), 1, Integer::sum);
            if (i - k >= 0) {
                sum -= nums.get(i - k);
                map.merge(nums.get(i - k), -1, Integer::sum);
                if (map.get(nums.get(i - k)) == 0) {
                    map.remove(nums.get(i - k));
                }
            }
            if (map.size() >= m) {
                res = Math.max(res, sum);
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

    // 100041. 可以到达每一个节点的最少边反转次数 (Minimum Edge Reversals So Every Node Is Reachable)
    private List<Integer>[] g100041;
    private Set<Long> set100041;
    private long M100041;
    private int res0_100041;
    private int[] res100041;

    public int[] minEdgeReversals(int n, int[][] edges) {
        this.g100041 = new ArrayList[n];
        this.M100041 = (long) 1e6;

        Arrays.setAll(g100041, k -> new ArrayList<>());
        set100041 = new HashSet<>();
        for (int[] e : edges) {
            int u = e[0];
            int v = e[1];
            set100041.add(u * M100041 + v);
            g100041[u].add(v);
            g100041[v].add(u);
        }
        dfs100041(0, -1);
        this.res100041 = new int[n];
        reRoot100041(0, -1, res0_100041);
        return res100041;

    }

    private void reRoot100041(int x, int fa, int cur) {
        res100041[x] = cur;
        for (int y : g100041[x]) {
            int copy = cur;
            if (y != fa) {
                if (set100041.contains(x * M100041 + y)) {
                    ++copy;
                } else {
                    --copy;
                }
                reRoot100041(y, x, copy);
            }
        }
    }

    private void dfs100041(int x, int fa) {
        for (int y : g100041[x]) {
            if (y != fa) {
                if (!set100041.contains(x * M100041 + y)) {
                    ++res0_100041;
                }
                dfs100041(y, x);
            }
        }
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

    // 100049. 美丽塔 I (Beautiful Towers I)
    // 100048. 美丽塔 II (Beautiful Towers II)
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

    private int[][] memo;
    private int n;
    private int m;
    private int u;
    private String[] stickers;
    private String target;

    public int minStickers(String[] stickers, String target) {
        this.n = stickers.length;
        this.m = target.length();
        this.memo = new int[n][1 << m];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(memo[i], -1);
        }
        this.u = (1 << m) - 1;
        this.stickers = stickers;
        this.target = target;
        int res = dfs(0, 0);
        return res < m + 1 ? res : -1;
    }

    private int dfs(int i, int j) {
        if (j == u) {
            return 0;
        }
        if (i == n) {
            return m + 1;
        }
        if (memo[i][j] != -1) {
            return memo[i][j];
        }
        // 不选
        int res = dfs(i + 1, j);
        // 选
        int give = 0;
        int[] cnts = new int[26];
        for (char chr : stickers[i].toCharArray()) {
            give |= 1 << (chr - 'a');
            ++cnts[chr - 'a'];
        }
        int c = u ^ j;
        int need = 0;
        int[] needCnts = new int[26];
        while (c > 0) {
            int index = Integer.numberOfTrailingZeros(c);
            need |= 1 << (target.charAt(index) - 'a');
            ++needCnts[target.charAt(index) - 'a'];
            c &= c - 1;
        }
        if ((give & need) == 0) {
            return memo[i][j] = res;
        }
        int max = 0;
        for (int k = 0; k < 26; ++k) {
            if (cnts[k] != 0 && needCnts[k] != 0) {
                max = Math.max(max, (needCnts[k] + cnts[k] - 1) / cnts[k]);
            }
        }
        c = u ^ j;
        for (int cnt = 1; cnt <= max; ++cnt) {
            int[] copy = cnts.clone();
            for (int k = 0; k < m; ++k) {
                if (((c >> k) & 1) == 1 && copy[target.charAt(k) - 'a'] > 0) {
                    --copy[target.charAt(k) - 'a'];
                    c ^= 1 << k;
                }
            }
            res = Math.min(res, dfs(i + 1, c ^ u) + cnt);
        }
        return memo[i][j] = res;
    }
}