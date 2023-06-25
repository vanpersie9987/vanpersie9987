import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
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
    private int[][] memo1681;
    private int k1681;
    private int[] map1681;
    private int n1681;

    public int minimumIncompatibility(int[] nums, int k) {
        this.n1681 = nums.length;
        int[] cnts = new int[n1681 + 1];
        for (int num : nums) {
            ++cnts[num];
        }
        for (int c : cnts) {
            if (c > k) {
                return -1;
            }
        }
        this.k1681 = k;
        this.memo1681 = new int[k][1 << n1681];
        for (int i = 0; i < k; ++i) {
            Arrays.fill(memo1681[i], -1);
        }
        this.map1681 = new int[1 << n1681];
        Arrays.fill(map1681, -1);
        search: for (int i = 0; i < (1 << n1681); ++i) {
            if (Integer.bitCount(i) == n1681 / k) {
                int bit = 0;
                int tmp = i;
                while (tmp != 0) {
                    int b = Integer.numberOfTrailingZeros(tmp);
                    int num = nums[b];
                    if (((bit >> num) & 1) != 0) {
                        continue search;
                    }
                    bit |= 1 << num;
                    tmp &= tmp - 1;
                }
                int min = Integer.numberOfTrailingZeros(bit);
                int max = 31 - Integer.numberOfLeadingZeros(bit);
                map1681[i] = max - min;
            }
        }
        return dfs1681(0, 0);

    }

    private int dfs1681(int i, int mask) {
        if (i == k1681) {
            return 0;
        }
        if (memo1681[i][mask] != -1) {
            return memo1681[i][mask];
        }
        int res = 1000;
        int c = ((1 << n1681) - 1) ^ mask;
        for (int j = c; j > 0; j = (j - 1) & c) {
            if (Integer.bitCount(j) == n1681 / k1681 && map1681[j] != -1) {
                res = Math.min(res, dfs1681(i + 1, mask | j) + map1681[j]);
            }
        }
        return memo1681[i][mask] = res;
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
        if (i == 0) {
            return dfs6898(i + 1, words6898[i].charAt(0) - 'a', words6898[i].charAt(len - 1) - 'a') + len;
        }
        if (memo6898[i][start][end] != 0) {
            return memo6898[i][start][end];
        }
        int left = words6898[i].charAt(0) - 'a';
        int right = words6898[i].charAt(len - 1) - 'a';
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
}
