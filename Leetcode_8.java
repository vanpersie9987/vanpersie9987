import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.IntStream;

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
        for (int i = 1; i < (1 << m); ++i) {
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
        boolean res = dfs1655(i + 1, mask);
        int c = u1655 ^ mask;
        for (int j = c; j > 0; j = (j - 1) & c) {
            if (cnts1655[i] >= sum1655[j]) {
                // 选
                res = res || dfs1655(i + 1, mask | j);
            }
            if (res) {
                break;
            }
        }
        memo1655[i][mask] = res ? 1 : -1;
        return res;
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

    // 837. 新 21 点 (New 21 Game)
    // private int n;
    // private int k;
    // private int maxPts;
    // private double[] memo;

    // public double new21Game(int n, int k, int maxPts) {
    // this.n = n;
    // this.k = k;
    // this.maxPts = maxPts;
    // this.memo = new double[k];
    // Arrays.fill(memo, -1D);
    // return dfs(0);
    // }

    // private double dfs(int score) {
    // if (score >= k) {
    // return score <= n ? 1D : 0D;
    // }
    // if (memo[score] != -1D) {
    // return memo[score];
    // }
    // double sum = 0D;
    // int i = 1;
    // while (i <= maxPts && i + score < k) {
    // sum += dfs(score + i);
    // }
    // if (i <= maxPts) {
    // sum += n - (i + score) + 1;
    // }
    // return memo[score] = sum / maxPts;
    // }

}
