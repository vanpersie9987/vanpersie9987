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

@SuppressWarnings("unchecked")
public class Leetcode_10 {

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
    };

    // 3688. 偶数的按位或运算 (Bitwise OR of Even Numbers in an Array)
    public int evenNumberBitwiseORs(int[] nums) {
        int res = 0;
        for (int x : nums) {
            if ((x & 1) == 0) {
                res |= x;
            }
        }
        return res;

    }

    // 3689. 最大子数组总值 I (Maximum Total Subarray Value I)
    public long maxTotalValue(int[] nums, int k) {
        int max = Integer.MIN_VALUE;
        int min = Integer.MAX_VALUE;
        for (int x : nums) {
            max = Math.max(max, x);
            min = Math.min(min, x);
        }
        return (long) (max - min) * k;

    }

    // 3690. 拆分合并数组 (Split and Merge Array Transformation)
    public int minSplitMerge(int[] nums1, int[] nums2) {
        int n = nums1.length;
        List<Integer> nums2List = toList3690(nums2);
        Set<List<Integer>> vis = new HashSet<>();
        vis.add(toList3690(nums1));
        Queue<List<Integer>> q = new LinkedList<>();
        q.add(toList3690(nums1));
        int res = 0;
        while (!q.isEmpty()) {
            int size = q.size();
            for (int s = 0; s < size; ++s) {
                List<Integer> a = q.poll();
                if (a.equals(nums2List)) {
                    return res;
                }
                for (int l = 0; l < n; l++) {
                    for (int r = l + 1; r <= n; r++) {
                        List<Integer> sub = a.subList(l, r);
                        List<Integer> b = new ArrayList<>(a);
                        b.subList(l, r).clear(); // 从 b 中移除 sub
                        for (int i = 0; i <= b.size(); i++) {
                            List<Integer> c = new ArrayList<>(b);
                            c.addAll(i, sub);
                            if (vis.add(c)) { // c 不在 vis 中
                                q.add(c);
                            }
                        }
                    }
                }
            }
            ++res;
        }
        return -1;
    }

    private List<Integer> toList3690(int[] nums) {
        return Arrays.stream(nums).boxed().collect(Collectors.toList());
    }

    // 166. 分数到小数 (Fraction to Recurring Decimal)
    public String fractionToDecimal(int numerator, int denominator) {
        long a = numerator;
        long b = denominator;
        String sign = a * b < 0 ? "-" : "";
        a = Math.abs(a);
        b = Math.abs(b);
        long q = a / b;
        long r = a % b;
        if (r == 0) {
            return sign + q;
        }
        StringBuilder res = new StringBuilder(sign).append(q).append(".");
        Map<Long, Integer> map = new HashMap<>();
        map.put(r, res.length());
        while (r != 0) {
            r *= 10;
            q = r / b;
            r %= b;
            res.append(q);
            if (map.containsKey(r)) {
                int index = map.get(r);
                res.insert(index, "(");
                res.append(")");
                break;
            }
            map.put(r, res.length());
        }
        return res.toString();
    }

    // 976. 三角形的最大周长 (Largest Perimeter Triangle)
    public int largestPerimeter(int[] nums) {
        Arrays.sort(nums);
        for (int i = nums.length - 1; i >= 2; --i) {
            if (nums[i] < nums[i - 1] + nums[i - 2]) {
                return nums[i] + nums[i - 1] + nums[i - 2];
            }
        }
        return 0;
    }

    // 3692. 众数频率字符 (Majority Frequency Characters)
    public String majorityFrequencyGroup(String s) {
        int[] cnts = new int[26];
        int mxFreq = 0;
        for (char c : s.toCharArray()) {
            ++cnts[c - 'a'];
            mxFreq = Math.max(mxFreq, cnts[c - 'a']);
        }
        int[] freqCnts = new int[mxFreq + 1];
        int mxLen = 0;
        for (int i = 0; i < 26; ++i) {
            if (cnts[i] > 0) {
                freqCnts[cnts[i]] |= (1 << i);
                mxLen = Math.max(mxLen, Integer.bitCount(freqCnts[cnts[i]]));
            }
        }
        int mask = 0;
        for (int f = mxFreq; f >= 1; --f) {
            if (Integer.bitCount(freqCnts[f]) == mxLen) {
                mask = freqCnts[f];
                break;
            }
        }
        StringBuilder sb = new StringBuilder();
        while (mask != 0) {
            int lb = Integer.numberOfTrailingZeros(mask);
            sb.append((char) (lb + 'a'));
            mask &= (mask - 1);
        }
        return sb.toString();
    }

    // 3693. 爬楼梯 II (Climbing Stairs II)
    private int[] costs3693;
    private int n3693;
    private int[] memo3693;

    public int climbStairs(int n, int[] costs) {
        ++n;
        this.costs3693 = new int[n];
        for (int i = 0; i < n - 1; ++i) {
            this.costs3693[i + 1] = costs[i];
        }
        this.n3693 = n;
        this.memo3693 = new int[n];
        return dfs3693(0);
    }

    private int dfs3693(int i) {
        if (i == n3693 - 1) {
            return 0;
        }
        if (memo3693[i] != 0) {
            return memo3693[i];
        }
        int res = Integer.MAX_VALUE;
        for (int j = i + 1; j < Math.min(n3693, i + 4); ++j) {
            res = Math.min(res, dfs3693(j) + costs3693[j] + (j - i) * (j - i));
        }
        return memo3693[i] = res;
    }

    // 3694. 删除子字符串后不同的终点 (Distinct Points Reachable After Substring Removal)
    public int distinctPoints(String s, int k) {
        int n = s.length();
        long MUL = (long) 1e5;
        int r = 0;
        int c = 0;
        Set<Long> points = new HashSet<>();
        for (int i = 0; i < n; ++i) {
            char ch = s.charAt(i);
            if (ch == 'R') {
                ++c;
            } else if (ch == 'L') {
                --c;
            } else if (ch == 'U') {
                ++r;
            } else if (ch == 'D') {
                --r;
            }
            if (i >= k) {
                char preCh = s.charAt(i - k);
                if (preCh == 'R') {
                    --c;
                } else if (preCh == 'L') {
                    ++c;
                } else if (preCh == 'U') {
                    --r;
                } else if (preCh == 'D') {
                    ++r;
                }
            }
            if (i >= k - 1) {
                points.add(r * MUL + c);
            }
        }
        return points.size();

    }

    // 3695. 交换元素后的最大交替和 (Maximize Alternating Sum Using Swaps)
    public long maxAlternatingSum(int[] nums, int[][] swaps) {
        int n = nums.length;
        boolean[] vis = new boolean[n];
        Union3695 u = new Union3695(n);
        for (int[] s : swaps) {
            int i = s[0];
            int j = s[1];
            u.union(i, j);
            vis[i] = true;
            vis[j] = true;
        }
        long res = 0L;
        Map<Integer, List<Integer>> groups = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            if (!vis[i]) {
                if (i % 2 == 0) {
                    res += nums[i];
                } else {
                    res -= nums[i];
                }
            } else {
                int root = u.getRoot(i);
                groups.computeIfAbsent(root, k -> new ArrayList<>()).add(i);
            }
        }
        for (Map.Entry<Integer, List<Integer>> entry : groups.entrySet()) {
            int odd = 0;
            for (int i : entry.getValue()) {
                if ((i & 1) == 1) {
                    ++odd;
                }
            }
            Collections.sort(entry.getValue(), new Comparator<Integer>() {

                @Override
                public int compare(Integer o1, Integer o2) {
                    return Integer.compare(nums[o1], nums[o2]);
                }

            });
            for (int i = 0; i < entry.getValue().size(); ++i) {
                int idx = entry.getValue().get(i);
                if (i < odd) {
                    res -= nums[idx];
                } else {
                    res += nums[idx];
                }
            }
        }
        return res;

    }

    public class Union3695 {
        private int[] parent;
        private int[] rank;

        public Union3695(int n) {
            this.parent = new int[n];
            this.rank = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
                rank[i] = 1;
            }
        }

        public int getRoot(int x) {
            if (parent[x] == x) {
                return x;
            }
            return parent[x] = getRoot(parent[x]);
        }

        public boolean isConnected(int x, int y) {
            return getRoot(x) == getRoot(y);
        }

        public void union(int x, int y) {
            int rootX = getRoot(x);
            int rootY = getRoot(y);
            if (rootX == rootY) {
                return;
            }
            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else {
                parent[rootY] = rootX;
                if (rank[rootX] == rank[rootY]) {
                    ++rank[rootX];
                }
            }
        }
    }

    // 3697. 计算十进制表示 (Compute Decimal Representation)
    public int[] decimalRepresentation(int n) {
        List<Integer> res = new ArrayList<>();
        int p = 1;
        while (n != 0) {
            int m = (n % 10) * p;
            if (m != 0) {
                res.add(m);
            }
            n /= 10;
            p *= 10;
        }
        Collections.reverse(res);
        return res.stream().mapToInt(i -> i).toArray();

    }

    // 3698. 分割数组得到最小绝对差 (Split Array With Minimum Difference)
    public long splitArray(int[] nums) {
        int n = nums.length;
        long[] pre = new long[n];
        Arrays.fill(pre, -1L);
        pre[0] = nums[0];
        for (int i = 1; i < n; ++i) {
            if (nums[i] <= nums[i - 1]) {
                break;
            }
            pre[i] = pre[i - 1] + nums[i];
        }
        long res = Long.MAX_VALUE;
        long suf = 0L;
        for (int i = n - 1; i >= 0; --i) {
            if (i < n - 1 && nums[i] <= nums[i + 1]) {
                break;
            }
            suf += nums[i];
            if (i - 1 >= 0 && pre[i - 1] != -1L) {
                res = Math.min(res, Math.abs(pre[i - 1] - suf));
            }
        }
        return res == Long.MAX_VALUE ? -1L : res;

    }

    // 3699. 锯齿形数组的总数 I (Number of ZigZag Arrays I)
    public int zigZagArrays(int n, int l, int r) {
        int k = r - l + 1;
        // f0[j] 最后一个数填j，最后两个数是递增的 数组个数
        int[] f0 = new int[k];
        // f1[j] 最后一个数填j，最后两个数是递减的 数组个数
        int[] f1 = new int[k];
        Arrays.fill(f0, 1);
        Arrays.fill(f1, 1);
        int MOD = (int) (1e9 + 7);
        for (int i = 2; i <= n; ++i) {
            int[] pre0 = new int[k + 1];
            int[] pre1 = new int[k + 1];
            for (int j = 0; j < k; ++j) {
                pre0[j + 1] = pre0[j] + f0[j];
                pre0[j + 1] %= MOD;

                pre1[j + 1] = pre1[j] + f1[j];
                pre1[j + 1] %= MOD;
            }
            for (int j = 0; j < k; ++j) {
                f0[j] = pre1[k] - pre1[j + 1];
                f0[j] = (f0[j] + MOD) % MOD;

                f1[j] = pre0[j];
                f1[j] = (f1[j] + MOD) % MOD;
            }
        }
        int res = 0;
        for (int i = 0; i < k; ++i) {
            res += f0[i];
            res %= MOD;
            res += f1[i];
            res %= MOD;
        }
        return res;
    }

    // 3701. 计算交替和 (Compute Alternating Sum)
    public int alternatingSum(int[] nums) {
        int res = 0;
        for (int i = 0; i < nums.length; ++i) {
            res += ((i & 1) * (-2) + 1) * nums[i];
        }
        return res;

    }

    // 3702. 按位异或非零的最长子序列 (Longest Subsequence With Non-Zero Bitwise XOR)
    public int longestSubsequence(int[] nums) {
        int mx = 0;
        int xor = 0;
        int n = nums.length;
        for (int x : nums) {
            xor ^= x;
            mx = Math.max(mx, x);
        }
        if (xor != 0) {
            return n;
        }
        return mx == 0 ? 0 : n - 1;
    }

    // 3703. 移除K-平衡子字符串 (Remove K-Balanced Substrings)
    public String removeSubstring(String s, int k) {
        List<int[]> st = new ArrayList<>();
        for (char c : s.toCharArray()) {
            if (!st.isEmpty() && st.get(st.size() - 1)[0] == c) {
                st.get(st.size() - 1)[1] += 1;
            } else {
                st.add(new int[] { c, 1 });
            }
            if (st.size() >= 2 && st.get(st.size() - 1)[0] == ')' && st.get(st.size() - 1)[1] == k
                    && st.get(st.size() - 2)[1] >= k) {
                st.remove(st.size() - 1);
                int[] last = st.get(st.size() - 1);
                last[1] -= k;
                if (last[1] == 0) {
                    st.remove(st.size() - 1);
                }
            }
        }
        StringBuilder res = new StringBuilder();
        for (int[] ss : st) {
            while (ss[1]-- > 0) {
                res.append((char) ss[0]);
            }
        }
        return res.toString();

    }

    // 3310. 移除可疑的方法 (Remove Methods From Project)
    public List<Integer> remainingMethods(int n, int k, int[][] invocations) {
        List<Integer>[] g = new ArrayList[n];
        Arrays.setAll(g, o -> new ArrayList<>());
        for (int[] i : invocations) {
            g[i[0]].add(i[1]);
        }
        boolean[] s = new boolean[n];
        dfs3310(k, g, s);
        for (int[] i : invocations) {
            if (!s[i[0]] && s[i[1]]) {
                List<Integer> res = new ArrayList<>();
                for (int j = 0; j < n; ++j) {
                    res.add(j);
                }
                return res;
            }
        }
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            if (!s[i]) {
                res.add(i);
            }
        }
        return res;

    }

    private void dfs3310(int x, List<Integer>[] g, boolean[] s) {
        if (s[x]) {
            return;
        }
        s[x] = true;
        for (int y : g[x]) {
            dfs3310(y, g, s);
        }
    }

    // 3707. 相等子字符串分数 (Equal Score Substrings)
    public boolean scoreBalance(String s) {
        int sum = 0;
        for (char c : s.toCharArray()) {
            sum += c - 'a' + 1;
        }
        int pre = 0;
        for (char c : s.toCharArray()) {
            pre += c - 'a' + 1;
            if (pre * 2 == sum) {
                return true;
            }
            if (pre * 2 > sum) {
                return false;
            }
        }
        return false;
    }

    // 3708. 最长斐波那契子数组 (Longest Fibonacci Subarray)
    public int longestSubarray(int[] nums) {
        int res = 0;
        int n = nums.length;
        for (int i = 1; i < n;) {
            int pre = nums[i - 1];
            int cur = nums[i];
            int j = i + 1;
            while (j < n && pre + cur == nums[j]) {
                pre = cur;
                cur = nums[j];
                ++j;
            }
            res = Math.max(res, j - i + 1);
            i = j;
        }
        return res;
    }

    // 3709. 设计考试分数记录器 (Design Exam Scores Tracker)
    class ExamTracker {
        private TreeMap<Integer, Long> pre;

        public ExamTracker() {
            this.pre = new TreeMap<>();
            this.pre.put(0, 0L);
        }

        public void record(int time, int score) {
            long last = pre.lastEntry().getValue();
            pre.put(time, last + score);
        }

        public long totalScore(int startTime, int endTime) {
            Integer end = pre.floorKey(endTime);
            Integer start = pre.lowerKey(startTime);
            return pre.get(end) - pre.get(start);
        }
    }

    // 3709. 设计考试分数记录器 (Design Exam Scores Tracker)
    class ExamTracker2 {
        private List<long[]> pre;

        public ExamTracker2() {
            pre = new ArrayList<>();
            pre.add(new long[] { 0L, 0L });
        }

        public void record(int time, int score) {
            pre.add(new long[] { time, pre.get(pre.size() - 1)[1] + score });
        }

        public long totalScore(int startTime, int endTime) {
            long end = bisectRight(endTime + 1);
            long start = bisectRight(startTime);
            return end - start;
        }

        private long bisectRight(int target) {
            int left = 0;
            int right = pre.size() - 1;
            while (left <= right) {
                int mid = left + ((right - left) >> 1);
                if (pre.get(mid)[0] < target) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            return pre.get(left - 1)[1];
        }
    }

    // 3712. 出现次数能被 K 整除的元素总和 (Sum of Elements With Frequency Divisible by K)
    public int sumDivisibleByK(int[] nums, int k) {
        int[] cnts = new int[101];
        for (int x : nums) {
            ++cnts[x];
        }
        int res = 0;
        for (int i = 0; i < 101; ++i) {
            if (cnts[i] != 0 && cnts[i] % k == 0) {
                res += i * cnts[i];
            }
        }
        return res;
    }

    // 3713. 最长的平衡子串 I (Longest Balanced Substring I)
    public int longestBalanced(String s) {
        int n = s.length();
        // 枚举字符串长度 L
        for (int L = n; L >= 1; --L) {
            int[] cnts = new int[26];
            for (int i = 0; i < n; ++i) {
                ++cnts[s.charAt(i) - 'a'];
                if (i >= L) {
                    --cnts[s.charAt(i - L) - 'a'];
                }
                if (i >= L - 1 && check3713(cnts)) {
                    return L;
                }
            }
        }
        return -1;
    }

    private boolean check3713(int[] cnts) {
        int cnt = -1;
        for (int c : cnts) {
            if (c == 0) {
                continue;
            }
            if (cnt != -1 && cnt != c) {
                return false;
            }
            cnt = c;
        }
        return true;
    }

    // 3715. 完全平方数的祖先个数总和 (Sum of Perfect Square Ancestors)
    public long sumOfAncestors(int n, int[][] edges, int[] nums) {
        for (int i = 0; i < n; ++i) {
            int x = nums[i];
            int ret = x;
            for (int p = 2; p * p <= x; ++p) {
                int cnt = 0;
                while (x % p == 0) {
                    cnt ^= 1;
                    x /= p;
                    if (cnt == 0) {
                        ret /= p * p;
                    }
                }
            }
            nums[i] = ret;
        }
        List<Integer>[] g = new ArrayList[n];
        Arrays.setAll(g, k -> new ArrayList<>());
        for (int[] e : edges) {
            g[e[0]].add(e[1]);
            g[e[1]].add(e[0]);
        }
        Map<Integer, Integer> cnts = new HashMap<>();
        return dfs3715(0, -1, cnts, g, nums);
    }

    private long dfs3715(int x, int fa, Map<Integer, Integer> cnts, List<Integer>[] g, int[] nums) {
        long res = cnts.getOrDefault(nums[x], 0);
        cnts.merge(nums[x], 1, Integer::sum);
        for (int y : g[x]) {
            if (y != fa) {
                res += dfs3715(y, x, cnts, g, nums);
            }
        }
        cnts.merge(nums[x], -1, Integer::sum);
        return res;
    }

    // 3710. 最大划分因子 (Maximum Partition Factor)
    public int maxPartitionFactor(int[][] points) {
        int n = points.length;
        if (n == 2) {
            return 0;
        }
        int mx = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                mx = Math.max(mx, Math.abs(points[i][0] - points[j][0]) + Math.abs(points[i][1] - points[j][1]));
            }
        }
        int left = 0;
        int right = mx;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (check3710(mid, points)) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left - 1;

    }

    private boolean check3710(int low, int[][] points) {
        int n = points.length;
        int[] color = new int[n];
        for (int i = 0; i < n; ++i) {
            if (color[i] == 0) {
                if (!dfs3710(i, 1, low, points, color)) {
                    return false;
                }
            }
        }
        return true;
    }

    private boolean dfs3710(int x, int c, int low, int[][] points, int[] color) {
        color[x] = c;
        for (int y = 0; y < points.length; ++y) {
            if (x == y) {
                continue;
            }
            int d = Math.abs(points[x][0] - points[y][0]) + Math.abs(points[x][1] - points[y][1]);
            if (d >= low) {
                continue;
            }
            if (color[y] == c) {
                return false;
            }
            if (color[y] == 0 && !dfs3710(y, -c, low, points, color)) {
                return false;
            }
        }
        return true;
    }

    // 3718. 缺失的最小倍数 (Smallest Missing Multiple of K)
    public int missingMultiple(int[] nums, int k) {
        boolean[] s = new boolean[101];
        for (int x : nums) {
            s[x] = true;
        }
        int p = 1;
        while (p * k < 101 && s[p * k]) {
            ++p;
        }
        return p * k;
    }

    // 3720. 大于目标字符串的最小字典序排列 (Lexicographically Smallest Permutation Greater Than
    // Target)
    public String lexGreaterPermutation(String s, String target) {
        int[] cnts_s = new int[26];
        for (char c : s.toCharArray()) {
            ++cnts_s[c - 'a'];
        }
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < target.length(); ++i) {
            int idx = (int) target.charAt(i) - 'a';
            cnts_s[idx] -= 1;
            if (cnts_s[idx] >= 0 && check3720(cnts_s, target.substring(i + 1))) {
                res.append(target.charAt(i));
                continue;
            }
            cnts_s[idx] += 1;
            for (int j = idx + 1; j < 26; ++j) {
                if (cnts_s[j] > 0) {
                    cnts_s[j] -= 1;
                    res.append((char) (j + 'a'));
                    for (int k = 0; k < 26; ++k) {
                        while (cnts_s[k]-- > 0) {
                            res.append((char) (k + 'a'));
                        }
                    }
                    return res.toString();
                }
            }
            return "";
        }
        return res.toString();

    }

    private boolean check3720(int[] cnts, String target) {
        StringBuilder sb = new StringBuilder();
        for (int i = 25; i >= 0; --i) {
            for (int j = 0; j < cnts[i]; ++j) {
                sb.append((char) (i + 'a'));
            }
        }
        String s = sb.toString();
        return s.compareTo(target) > 0;
    }

    // 3719. 最长平衡子数组 I (Longest Balanced Subarray I)
    public int longestBalanced(int[] nums) {
        int n = nums.length;
        int res = 0;
        for (int i = 0; i < n && res < n - i; ++i) {
            Set<Integer>[] cnts = new HashSet[2];
            Arrays.setAll(cnts, o -> new HashSet<>());
            for (int j = i; j < n; ++j) {
                cnts[nums[j] % 2].add(nums[j]);
                if (cnts[0].size() == cnts[1].size()) {
                    res = Math.max(res, j - i + 1);
                }
            }
        }
        return res;
    }

    // 3578. 统计极差最大为 K 的分割方式数 (Count Partitions With Max-Min Difference at Most K)
    public int countPartitions(int[] nums, int k) {
        int n = nums.length;
        int MOD = (int) (1e9 + 7);
        int left = 0;
        Deque<Integer> qMin = new ArrayDeque<>();
        Deque<Integer> qMax = new ArrayDeque<>();
        int[] f = new int[n + 1];
        int sumF = 0;
        f[0] = 1;
        for (int i = 0; i < n; ++i) {
            sumF += f[i];
            sumF %= MOD;
            while (!qMin.isEmpty() && nums[qMin.peekLast()] >= nums[i]) {
                qMin.pollLast();
            }
            qMin.offerLast(i);
            while (!qMax.isEmpty() && nums[qMax.peekLast()] <= nums[i]) {
                qMax.pollLast();
            }
            qMax.offerLast(i);
            while (nums[qMax.peekFirst()] - nums[qMin.peekFirst()] > k) {
                sumF -= f[left];
                sumF = (sumF + MOD) % MOD;
                if (qMin.peekFirst() == left) {
                    qMin.pollFirst();
                }
                if (qMax.peekFirst() == left) {
                    qMax.pollFirst();
                }
                ++left;
            }
            f[i + 1] = sumF;
        }
        return f[n];

    }

    // 3722. 反转后字典序最小的字符串 (Lexicographically Smallest String After Reverse)
    public String lexSmallest(String s) {
        int n = s.length();
        String res = "";
        for (int i = 0; i < n; ++i) {
            String cur = reversed3722(s.substring(0, i)) + s.substring(i);
            if (res.equals("") || cur.compareTo(res) < 0) {
                res = cur;
            }
            cur = s.substring(0, i) + reversed3722(s.substring(i));
            if (res.equals("") || cur.compareTo(res) < 0) {
                res = cur;
            }
        }
        return res;

    }

    private String reversed3722(String s) {
        StringBuilder sb = new StringBuilder(s);
        return sb.reverse().toString();
    }

    // 3723. 数位平方和的最大值 (Maximize Sum of Squares of Digits)
    public String maxSumOfSquares(int num, int sum) {
        if (9 * num < sum) {
            return "";
        }
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < num; ++i) {
            int d = Math.min(9, sum);
            res.append((char) (d + '0'));
            sum -= d;
        }
        return res.toString();

    }

    // 3724. 转换数组的最少操作次数 (Minimum Operations to Transform Array)
    private int n3724;
    private int[] nums1_3724;
    private int[] nums2_3724;
    private long[][] memo3724;

    public long minOperations(int[] nums1, int[] nums2) {
        this.nums1_3724 = nums1;
        this.nums2_3724 = nums2;
        this.n3724 = nums1.length;
        this.memo3724 = new long[n3724][2];
        for (long[] row : memo3724) {
            Arrays.fill(row, -1L);
        }
        return dfs3724(0, 0);

    }

    private long dfs3724(int i, int j) {
        if (i == n3724) {
            return j == 0 ? Long.MAX_VALUE / 2 : 0L;
        }
        if (memo3724[i][j] != -1L) {
            return memo3724[i][j];
        }
        long res = dfs3724(i + 1, j) + Math.abs(nums1_3724[i] - nums2_3724[i]);
        if (j == 0) {
            int max = Math.max(Math.max(nums1_3724[i], nums2_3724[i]), nums2_3724[n3724]);
            int min = Math.min(Math.min(nums1_3724[i], nums2_3724[i]), nums2_3724[n3724]);
            res = Math.min(res, dfs3724(i + 1, 1) + max - min + 1);
        }
        return memo3724[i][j] = res;
    }

    // 3725. 统计每一行选择互质整数的方案数 (Count Ways to Choose Coprime Integers from Rows)
    private int[][] mat3725;
    private int[][] memo3725;

    public int countCoprime(int[][] mat) {
        this.mat3725 = mat;
        int m = mat.length;
        this.memo3725 = new int[m][151];
        for (int[] r : memo3725) {
            Arrays.fill(r, -1);
        }
        return dfs3725(m - 1, 0);

    }

    private int dfs3725(int i, int j) {
        if (i < 0) {
            return j == 1 ? 1 : 0;
        }
        if (memo3725[i][j] != -1) {
            return memo3725[i][j];
        }
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        for (int x : mat3725[i]) {
            res += dfs3725(i - 1, gcd3725(j, x));
            res %= MOD;
        }
        return memo3725[i][j] = res;
    }

    private int gcd3725(int a, int b) {
        return b == 0 ? a : gcd3725(b, a % b);
    }

    // 3726. 移除十进制表示中的所有零 (Remove Zeros in Decimal Representation)
    public long removeZeros(long n) {
        long res = 0L;
        long p = 1L;
        while (n != 0L) {
            int m = (int) (n % 10);
            n /= 10;
            if (m != 0) {
                res += p * m;
                p *= 10L;
            }
        }
        return res;

    }

    // 3727. 最大交替平方和 (Maximum Alternating Sum of Squares)
    public long maxAlternatingSum(int[] nums) {
        int n = nums.length;
        int[] a = new int[n];
        for (int i = 0; i < n; ++i) {
            a[i] = Math.abs(nums[i]);
        }
        Arrays.sort(a);
        long res = 0L;
        for (int i = 0; i < n; ++i) {
            res += (long) a[i] * a[i] * (i < n / 2 ? -1 : 1);
        }
        return res;

    }

    private record Group3728(int a, long b) {
    }

    // 3728. 边界与内部和相等的稳定子数组 (Stable Subarrays With Equal Boundary and Interior Sum)
    public long countStableSubarrays(int[] capacity) {
        long res = 0L;
        Map<Group3728, Integer> cnt = new HashMap<>();
        long s = capacity[0];
        for (int i = 1; i < capacity.length; ++i) {
            int last = capacity[i - 1];
            int x = capacity[i];
            res += cnt.getOrDefault(new Group3728(x, s), 0);
            cnt.merge(new Group3728(last, last + s), 1, Integer::sum);
            s += x;
        }
        return res;

    }

    // 1578. 使绳子变成彩色的最短时间 (Minimum Time to Make Rope Colorful)
    private int[][] memo1578;
    private char[] a1578;
    private int[] neededTime1578;

    public int minCost(String colors, int[] neededTime) {
        int n = neededTime.length;
        this.a1578 = colors.toCharArray();
        this.neededTime1578 = neededTime;
        this.memo1578 = new int[n][27];
        for (int[] row : memo1578) {
            Arrays.fill(row, -1);
        }
        return dfs1578(n - 1, 26);

    }

    private int dfs1578(int i, int j) {
        if (i < 0) {
            return 0;
        }
        if (memo1578[i][j] != -1) {
            return memo1578[i][j];
        }
        int res = dfs1578(i - 1, j) + neededTime1578[i];
        int c = a1578[i] - 'a';
        if (c != j) {
            res = Math.min(res, dfs1578(i - 1, c));
        }
        return memo1578[i][j] = res;
    }

    // 1578. 使绳子变成彩色的最短时间 (Minimum Time to Make Rope Colorful)
    public int minCost2(String colors, int[] neededTime) {
        int n = colors.length();
        int i = 0;
        int res = 0;
        while (i < n) {
            int max = 0;
            int sum = 0;
            int j = i;
            while (j < n && colors.charAt(i) == colors.charAt(j)) {
                max = Math.max(max, neededTime[j]);
                sum += neededTime[j];
                ++j;
            }
            res += sum - max;
            i = j;
        }
        return res;

    }

    // 3731. 找出缺失的元素 (Find Missing Elements)
    public List<Integer> findMissingElements(int[] nums) {
        Set<Integer> s = new HashSet<>();
        int max = 0;
        int min = Integer.MAX_VALUE;
        for (int x : nums) {
            s.add(x);
            max = Math.max(max, x);
            min = Math.min(min, x);
        }
        List<Integer> res = new ArrayList<>();
        for (int x = min + 1; x < max; ++x) {
            if (!s.contains(x)) {
                res.add(x);
            }
        }
        return res;

    }

    // 3732. 一次替换后的三元素最大乘积 (Maximum Product of Three Elements After One Replacement)
    public long maxProduct(int[] nums) {
        Arrays.sort(nums);
        long res1 = Math.abs((long) nums[0] * nums[1]);
        long res2 = Math.abs((long) nums[nums.length - 1] * nums[nums.length - 2]);
        long res3 = Math.abs((long) nums[0] * nums[nums.length - 1]);
        return Math.max(Math.max(res1, res2), res3) * (long) 1e5;

    }

    // 3733. 完成所有送货任务的最少时间 (Minimum Time to Complete All Deliveries)
    public long minimumTime(int[] d, int[] r) {
        long left = d[0] + d[1];
        long right = (long) 1e10;
        long lcm = lcm3733(r[0], r[1]);
        while (left <= right) {
            long mid = left + ((right - left) >> 1);
            if (check3733(mid, d, r, lcm)) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return right + 1;

    }

    private boolean check3733(long t, int[] d, int[] r, long lcm) {
        // 都不可以送货的小时数
        long neither = t / lcm;
        // 仅无人机1可以送的小时数
        long x1 = t / r[1] - neither;
        // 仅无人机2可以送的小时数
        long x2 = t / r[0] - neither;
        return Math.max(0, d[0] - x1) + Math.max(0, d[1] - x2) <= t - x1 - x2 - neither;
    }

    private long lcm3733(int a, int b) {
        return (long) a * b / gcd3733(a, b);
    }

    private long gcd3733(int a, int b) {
        return b == 0 ? a : gcd3733(b, a % b);
    }

    // 3318. 计算子数组的 x-sum I (Find X-Sum of All K-Long Subarrays I) --暴力
    public int[] findXSum(int[] nums, int k, int x) {
        int n = nums.length;
        int[] res = new int[n - k + 1];
        Map<Integer, Integer> cnts = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            cnts.merge(nums[i], 1, Integer::sum);
            if (i >= k) {
                cnts.merge(nums[i - k], -1, Integer::sum);
            }
            if (i >= k - 1) {
                TreeSet<int[]> treeSet = new TreeSet<>(new Comparator<>() {

                    @Override
                    public int compare(int[] o1, int[] o2) {
                        if (o1[0] == o2[0]) {
                            return Integer.compare(o2[1], o1[1]);
                        }
                        return Integer.compare(o2[0], o1[0]);
                    }

                });
                for (Map.Entry<Integer, Integer> entry : cnts.entrySet()) {
                    treeSet.add(new int[] { entry.getValue(), entry.getKey() });
                }
                int s = 0;
                int c = 0;
                for (int[] cur : treeSet) {
                    ++c;
                    if (c > x) {
                        break;
                    }
                    s += cur[0] * cur[1];
                }
                res[i - k + 1] = s;
            }
        }
        return res;

    }

    // 3734. 大于目标字符串的最小字典序回文排列 (Lexicographically Smallest Palindromic Permutation
    // Greater Than Target)
    public String lexPalindromicPermutation(String s, String target) {
        int n = s.length();
        int[] cnt = new int[26];
        for (char c : s.toCharArray()) {
            ++cnt[c - 'a'];
        }
        int midIdx = -1;
        int oddCnt = 0;
        for (int i = 0; i < 26; ++i) {
            if (cnt[i] % 2 != 0) {
                midIdx = i;
                if (++oddCnt > 1) {
                    return "";
                }
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 25; i >= 0; --i) {
            int c = cnt[i] / 2;
            while (c-- > 0) {
                sb.append((char) (i + 'a'));
            }
        }
        String max = sb.toString() + (n % 2 == 0 ? "" : (char) (midIdx + 'a')) + sb.reverse().toString();
        if (max.compareTo(target) <= 0) {
            return "";
        }
        char[] res = new char[n];
        if (n % 2 != 0) {
            res[n / 2] = (char) (midIdx + 'a');
            --cnt[midIdx];
        }
        for (int i = 0; i < 26; ++i) {
            cnt[i] >>= 1;
        }
        for (int i = 0; i < n / 2; ++i) {
            int tIdx = target.charAt(i) - 'a';
            if (cnt[tIdx] == 0 || !check3734(res, cnt.clone(), i, target)) {
                for (int j = tIdx + 1; j < 26; ++j) {
                    if (cnt[j] != 0) {
                        res[i] = res[n - i - 1] = (char) (j + 'a');
                        --cnt[j];
                        ++i;
                        break;
                    }
                }
                for (int j = 0; j < 26; ++j) {
                    while (cnt[j] != 0) {
                        res[i] = res[n - i - 1] = (char) (j + 'a');
                        --cnt[j];
                        ++i;
                    }
                }
                return String.valueOf(res);
            }
            --cnt[tIdx];
            res[i] = res[n - i - 1] = target.charAt(i);
        }
        return String.valueOf(res);

    }

    private boolean check3734(char[] res, int[] cnt, int i, String target) {
        int n = res.length;
        int tIdx = target.charAt(i) - 'a';
        --cnt[tIdx];
        res[i] = res[n - i - 1] = target.charAt(i);
        ++i;
        for (int j = 25; j >= 0; --j) {
            while (cnt[j]-- > 0) {
                res[i] = res[n - i - 1] = (char) (j + 'a');
                ++i;
            }
        }
        return String.valueOf(res).compareTo(target) > 0;
    }

    // 3736. 最小操作次数使数组元素相等 III (Minimum Moves to Equal Array Elements III)
    public int minMoves(int[] nums) {
        int n = nums.length;
        int s = 0;
        int mx = 0;
        for (int x : nums) {
            s += x;
            mx = Math.max(mx, x);
        }
        return n * mx - s;

    }

    // 3740. 三个相等元素之间的最小距离 I (Minimum Distance Between Three Equal Elements I)
    // 3741. 三个相等元素之间的最小距离 II (Minimum Distance Between Three Equal Elements II)
    public int minimumDistance(int[] nums) {
        int res = Integer.MAX_VALUE;
        int n = nums.length;
        Map<Integer, List<Integer>> g = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            g.computeIfAbsent(nums[i], k -> new ArrayList<>()).add(i);
            List<Integer> list = g.get(nums[i]);
            if (list.size() >= 3) {
                res = Math.min(res, list.get(list.size() - 1) - list.get(list.size() - 3));
            }
        }
        return res == Integer.MAX_VALUE ? -1 : res * 2;

    }

    // 3742. 网格中得分最大的路径 (Maximum Path Score in a Grid)
    private int[][] grid3742;
    private int[][][] memo3742;

    public int maxPathScore(int[][] grid, int k) {
        int m = grid.length;
        int n = grid[0].length;
        k = Math.min(k, m + n - 2);
        this.grid3742 = grid;
        this.memo3742 = new int[m][n][k + 1];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                Arrays.fill(memo3742[i][j], Integer.MAX_VALUE / 2);
            }
        }
        int res = dfs3742(m - 1, n - 1, k);
        return res < 0 ? -1 : res;
    }

    private int dfs3742(int i, int j, int k) {
        if (i < 0 || j < 0) {
            return Integer.MIN_VALUE;
        }
        int curK = k - (grid3742[i][j] == 0 ? 0 : 1);
        if (curK < 0) {
            return Integer.MIN_VALUE;
        }
        if (i == 0 && j == 0) {
            return grid3742[i][j];
        }
        if (memo3742[i][j][k] != Integer.MAX_VALUE / 2) {
            return memo3742[i][j][k];
        }
        return memo3742[i][j][k] = Math.max(dfs3742(i - 1, j, curK), dfs3742(i, j - 1, curK)) + grid3742[i][j];
    }

    // 3737. 统计主要元素子数组数目 I (Count Subarrays With Majority Element I)
    // 3739. 统计主要元素子数组数目 II (Count Subarrays With Majority Element II)
    public long countMajoritySubarrays(int[] nums, int target) {
        List<Integer> list = new ArrayList<>();
        list.add(0);
        int s = 0;
        long res = 0L;
        for (int x : nums) {
            s += x == target ? 1 : -1;
            int i = bisectLeft3739(list, s);
            res += i;
            list.add(i, s);
        }
        return res;

    }

    private int bisectLeft3739(List<Integer> list, int x) {
        int left = 0;
        int right = list.size() - 1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (list.get(mid) >= x) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return right + 1;
    }

    // 3738. 替换至多一个元素后最长非递减子数组 (Longest Non-Decreasing Subarray After Replacing at
    // Most One Element)
    public int longestSubarray3738(int[] nums) {
        int n = nums.length;
        if (n == 1) {
            return 1;
        }
        int res = 2;
        int[] suf = new int[n];
        suf[n - 1] = 1;
        for (int i = n - 2; i >= 1; --i) {
            if (nums[i] <= nums[i + 1]) {
                suf[i] = suf[i + 1] + 1;
                res = Math.max(res, suf[i] + 1);
            } else {
                suf[i] = 1;
            }
        }
        int pre = 1;
        for (int i = 1; i < n - 1; ++i) {
            if (nums[i - 1] <= nums[i + 1]) {
                res = Math.max(res, pre + 1 + suf[i + 1]);
            }
            if (nums[i - 1] <= nums[i]) {
                ++pre;
                res = Math.max(res, pre + 1);
            } else {
                pre = 1;
            }
        }
        return res;
    }

    // 3745. 三元素表达式的最大值 (Maximize Expression of Three Elements)
    public int maximizeExpressionOfThree(int[] nums) {
        int max1 = Integer.MIN_VALUE;
        int max2 = Integer.MIN_VALUE;
        int min1 = Integer.MAX_VALUE;
        for (int x : nums) {
            if (x >= max1) {
                max2 = max1;
                max1 = x;
            } else if (x > max2) {
                max2 = x;
            }
            if (x < min1) {
                min1 = x;
            }
        }
        return max1 + max2 - min1;

    }

    // 3746. 等量移除后的字符串最小长度 (Minimum String Length After Balanced Removals)
    public int minLengthAfterRemovals(String s) {
        int bCnt = 0;
        for (char c : s.toCharArray()) {
            bCnt += c - 'a';
        }
        return Math.abs(s.length() - bCnt * 2);
    }

    // 3747. 统计移除零后不同整数的数目 (Count Distinct Integers After Removing Zeros)
    private long[][] memo3747;
    private char[] a3747;
    private int n3747;

    public long countDistinct(long n) {
        this.a3747 = String.valueOf(n).toCharArray();
        this.n3747 = a3747.length;
        this.memo3747 = new long[this.n3747][2];
        return n - dfs3747(0, 0, true, false);
    }

    private long dfs3747(int i, int j, boolean isLimit, boolean isNum) {
        if (i == n3747) {
            return j == 1 && isNum ? 1 : 0;
        }
        if (!isLimit && isNum && memo3747[i][j] != 0L) {
            return memo3747[i][j];
        }
        long res = 0L;
        if (!isNum) {
            res += dfs3747(i + 1, j, false, false);
        }
        int up = isLimit ? a3747[i] - '0' : 9;
        for (int d = isNum ? 0 : 1; d <= up; ++d) {
            res += dfs3747(i + 1, (j == 1 || d == 0) ? 1 : 0, isLimit && up == d, true);
        }
        if (!isLimit && isNum) {
            memo3747[i][j] = res;
        }
        return res;
    }

    // 3748. 统计稳定子数组的数目 (Count Stable Subarrays)
    public long[] countStableSubarrays(int[] nums, int[][] queries) {
        int n = nums.length;
        List<Integer> left = new ArrayList<>();
        List<Long> s = new ArrayList<>();
        s.add(0L);
        int start = 0;
        for (int i = 0; i < n; ++i) {
            if (i == n - 1 || nums[i] > nums[i + 1]) {
                left.add(start);
                int m = i - start + 1;
                s.add(s.get(s.size() - 1) + (long) (1 + m) * m / 2);
                start = i + 1;
            }
        }
        long[] res = new long[queries.length];
        for (int x = 0; x < queries.length; ++x) {
            int l = queries[x][0];
            int r = queries[x][1];
            int i = bisectRight3748(left, l);
            int j = bisectRight3748(left, r) - 1;
            if (i > j) {
                int m = r - l + 1;
                res[x] = (long) m * (m + 1) / 2;
                continue;
            }
            int m1 = left.get(i) - l;
            int m2 = r - left.get(j) + 1;
            res[x] = (long) m1 * (m1 + 1) / 2 + s.get(j) - s.get(i) + (long) m2 * (m2 + 1) / 2;
        }
        return res;

    }

    private int bisectRight3748(List<Integer> list, int t) {
        int left = 0;
        int right = list.size() - 1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (list.get(mid) > t) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return right + 1;
    }

    // 3273. 对 Bob 造成的最少伤害 (Minimum Amount of Damage Dealt to Bob)
    public long minDamage(int power, int[] damage, int[] health) {
        int n = damage.length;
        int[][] a = new int[n][2];
        for (int i = 0; i < n; ++i) {
            a[i][0] = (health[i] + power - 1) / power;
            a[i][1] = damage[i];
        }
        Arrays.sort(a, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Double.compare((double) o1[0] / o1[1], (double) o2[0] / o2[1]);
            }
        });
        long res = 0L;
        long s = 0L;
        for (int i = 0; i < n; ++i) {
            s += a[i][0];
            res += s * a[i][1];
        }
        return res;

    }

    // 3334. 数组的最大因子得分 (Find the Maximum Factor Score of Array)
    public long maxScore(int[] nums) {
        int n = nums.length;
        long[] preLCM = new long[n + 1];
        preLCM[0] = 1L;
        long[] preGCD = new long[n + 1];
        for (int i = 0; i < n; ++i) {
            preGCD[i + 1] = gcd3334(preGCD[i], nums[i]);
            preLCM[i + 1] = lcm3334(preLCM[i], nums[i]);
        }
        long res = preLCM[n] * preGCD[n];
        long sufLCM = 1L;
        long sufGCD = 0L;
        for (int i = n - 1; i >= 0; --i) {
            res = Math.max(res, lcm3334(preLCM[i], sufLCM) * gcd3334(preGCD[i], sufGCD));
            sufGCD = gcd3334(sufGCD, nums[i]);
            sufLCM = lcm3334(sufLCM, nums[i]);
        }
        return res;
    }

    private long lcm3334(long a, long b) {
        return a / gcd3334(a, b) * b;
    }

    private long gcd3334(long a, long b) {
        return b == 0L ? a : gcd3334(b, a % b);
    }

    // 3433. 统计用户被提及情况 (Count Mentions Per User)
    public int[] countMentions(int n, List<List<String>> events) {
        Collections.sort(events, (o1, o2) -> {
            int t1 = Integer.parseInt(o1.get(1));
            int t2 = Integer.parseInt(o2.get(1));
            if (t1 == t2) {
                return o1.get(2).compareTo(o2.get(2));
            }
            return Integer.compare(t1, t2);
        });
        int[] res = new int[n];
        int[] nxtOffLine = new int[n];
        Arrays.fill(nxtOffLine, -100);
        int cntAll = 0;
        for (List<String> even : events) {
            String m = even.get(0);
            int t = Integer.parseInt(even.get(1));
            String ids = even.get(2);
            if ("MESSAGE".equals(m)) {
                if ("ALL".equals(ids)) {
                    ++cntAll;
                } else if ("HERE".equals(ids)) {
                    for (int i = 0; i < n; ++i) {
                        if (nxtOffLine[i] <= t && t < nxtOffLine[i] + 60) {
                            continue;
                        }
                        res[i] += 1;
                    }
                } else {
                    for (String id : ids.split(" ")) {
                        int idx = Integer.parseInt(id.substring(2));
                        res[idx] += 1;
                    }
                }
            } else {
                int idx = Integer.parseInt(ids);
                nxtOffLine[idx] = t;
            }
        }
        for (int i = 0; i < n; ++i) {
            res[i] += cntAll;
        }
        return res;

    }

    // 3387. 两天自由外汇交易后的最大货币数 (Maximize Amount After Two Days of Conversions)
    public double maxAmount(String initialCurrency, List<List<String>> pairs1, double[] rates1,
            List<List<String>> pairs2, double[] rates2) {
        Map<String, Double> m1 = getTrans3387(pairs1, rates1, initialCurrency);
        Map<String, Double> m2 = getTrans3387(pairs2, rates2, initialCurrency);
        Double res = 1.0D;
        for (Map.Entry<String, Double> entry : m2.entrySet()) {
            String currency = entry.getKey();
            Double money = entry.getValue();
            res = Math.max(res, m1.getOrDefault(currency, 0.0D) / money);
        }
        return res;

    }

    private Map<String, Double> getTrans3387(List<List<String>> pairs, double[] rates, String initialCurrency) {
        Map<String, List<String[]>> g = new HashMap<>();
        int n = pairs.size();
        for (int i = 0; i < n; ++i) {
            List<String> p = pairs.get(i);
            g.computeIfAbsent(p.get(0), k -> new ArrayList<>()).add(new String[] { p.get(1), rates[i] + "" });
            g.computeIfAbsent(p.get(1), k -> new ArrayList<>()).add(new String[] { p.get(0), (1.0D / rates[i]) + "" });
        }
        Map<String, Double> res = new HashMap<>();
        dfs3387(g, 1.0D, initialCurrency, res);
        return res;
    }

    private void dfs3387(Map<String, List<String[]>> g, double d, String currency, Map<String, Double> res) {
        res.put(currency, d);
        for (String[] nei : g.getOrDefault(currency, new ArrayList<>())) {
            if (res.containsKey(nei[0])) {
                continue;
            }
            String nextCurrency = nei[0];
            double rate = Double.parseDouble(nei[1]);
            dfs3387(g, d * rate, nextCurrency, res);
        }
    }

    // 3750. 最少反转次数得到翻转二进制字符串 (Minimum Number of Flips to Reverse Binary String)
    public int minimumFlips(int n) {
        int rev = Integer.reverse(n) >>> Integer.numberOfLeadingZeros(n);
        return Integer.bitCount(n ^ rev);
    }

    // 3751. 范围内总波动值 I (Total Waviness of Numbers in Range I)
    // 3753. 范围内总波动值 II (Total Waviness of Numbers in Range II)
    public long totalWaviness(long num1, long num2) {
        return cal3753(num2) - cal3753(num1 - 1);
    }

    private long[][][][] memo3753;
    private char[] a3753;
    private int n3753;

    private long cal3753(long x) {
        this.a3753 = String.valueOf(x).toCharArray();
        this.n3753 = a3753.length;
        if (n3753 < 3) {
            return 0L;
        }
        this.memo3753 = new long[n3753][11][11][n3753 - 1];
        for (long[][][] matrix : memo3753) {
            for (long[][] row : matrix) {
                for (long[] col : row) {
                    Arrays.fill(col, -1L);
                }
            }
        }
        return dfs3753(0, 10, 10, 0, true, false);
    }

    private long dfs3753(int i, int pre, int pre2, int j, boolean isLimit, boolean isNum) {
        if (i == n3753) {
            return j;
        }
        if (!isLimit && isNum && memo3753[i][pre][pre2][j] != -1L) {
            return memo3753[i][pre][pre2][j];
        }
        long res = 0L;
        if (!isNum) {
            res += dfs3753(i + 1, pre, pre2, j, false, false);
        }
        int up = isLimit ? a3753[i] - '0' : 9;
        for (int d = isNum ? 0 : 1; d <= up; ++d) {
            int add = pre != 10 && pre2 != 10 && ((pre - pre2) * (pre - d) > 0) ? 1 : 0;
            res += dfs3753(i + 1, d, pre, j + add, isLimit && d == up, true);
        }
        if (!isLimit && isNum) {
            memo3753[i][pre][pre2][j] = res;
        }
        return res;
    }

    // 3752. 字典序最小和为目标值且绝对值是排列的数组 (Lexicographically Smallest Negated Permutation
    // that Sums to Target)
    public int[] lexSmallestNegatedPerm(int n, long target) {
        long s = (long) n * (n + 1) / 2;
        if (Math.abs(target) > s || (s - target) % 2 != 0) {
            return new int[0];
        }
        long negS = (s - target) / 2;
        int[] res = new int[n];
        int left = 0;
        int right = n - 1;
        for (int x = n; x >= 1; --x) {
            if (negS >= x) {
                res[left++] = -x;
                negS -= x;
            } else {
                res[right--] = x;
            }
        }
        return res;
    }

    // 3754. 连接非零数字并乘以其数字和 I (Concatenate Non-Zero Digits and Multiply by Sum I)
    public long sumAndMultiply(int n) {
        long x = 0L;
        int s = 0;
        long p = 1L;
        while (n != 0) {
            int m = n % 10;
            if (m != 0) {
                x += p * m;
                s += m;
                p *= 10L;
            }
            n /= 10;
        }
        return x * s;

    }

    // 3756. 连接非零数字并乘以其数字和 II (Concatenate Non-Zero Digits and Multiply by Sum II)
    public int[] sumAndMultiply(String s, int[][] queries) {
        final int MOD = (int) 1e9 + 7;
        int n = s.length();
        int m = queries.length;
        int[] preS = new int[n + 1];
        long[] preX = new long[n + 1];
        int[] preCnt = new int[n + 1];
        int[] pow10 = new int[n + 1];
        pow10[0] = 1;
        for (int i = 0; i < s.length(); ++i) {
            preS[i + 1] = preS[i] + (s.charAt(i) - '0');
            pow10[i + 1] = (int) (((long) pow10[i] * 10) % MOD);
            if (s.charAt(i) != '0') {
                preX[i + 1] = (preX[i] * 10 + (s.charAt(i) - '0')) % MOD;
                preCnt[i + 1] = preCnt[i] + 1;
            } else {
                preX[i + 1] = preX[i];
                preCnt[i + 1] = preCnt[i];
            }
        }
        int[] res = new int[m];
        for (int i = 0; i < m; ++i) {
            int l = queries[i][0];
            int r = queries[i][1];
            int xVal = (preS[r + 1] - preS[l]) % MOD;
            long sVal = ((preX[r + 1] - (preX[l] * pow10[preCnt[r + 1] - preCnt[l]]) % MOD) + MOD) % MOD;
            res[i] = (int) ((xVal * sVal) % MOD);
        }
        return res;
    }

    // 3755. 最大平衡异或子数组的长度 (Find Maximum Balanced XOR Subarray Length)
    private record Group3755(int a, int b) {
    }

    public int maxBalancedSubarray(int[] nums) {
        Map<Group3755, Integer> pos = new HashMap<>();
        pos.put(new Group3755(0, 0), -1);
        int xor = 0;
        int diff = 0;
        int res = 0;
        for (int i = 0; i < nums.length; ++i) {
            xor ^= nums[i];
            diff += ((nums[i] & 1) == 0 ? 1 : -1);
            Group3755 key = new Group3755(xor, diff);
            if (pos.containsKey(key)) {
                res = Math.max(res, i - pos.get(key));
            } else {
                pos.put(key, i);
            }
        }
        return res;
    }

    // 3575. 最大好子树分数 (Maximum Good Subtree Score)
    private List<Integer>[] g3575;
    private int res3575;
    private int[] vals3575;
    private int[] masks3575;

    public int goodSubtreeSum(int[] vals, int[] par) {
        int n = vals.length;
        this.vals3575 = vals;
        this.masks3575 = new int[n];
        this.g3575 = new ArrayList[n];
        Arrays.setAll(g3575, k -> new ArrayList<>());
        for (int i = 0; i < n; ++i) {
            if (i > 0) {
                g3575[par[i]].add(i);
            }
            masks3575[i] = check3575(vals[i]);
        }
        dfsTree3575(0);
        return res3575;

    }

    private List<Integer>[] dfsTree3575(int x) {
        List<Integer>[] ret = new ArrayList[10];
        Arrays.setAll(ret, k -> new ArrayList<>());
        for (int y : g3575[x]) {
            List<Integer>[] list = dfsTree3575(y);
            for (int i = 0; i < 10; ++i) {
                ret[i].addAll(list[i]);
            }
        }
        for (int c = masks3575[x]; c != 0; c &= c - 1) {
            int lb = Integer.numberOfTrailingZeros(c);
            ret[lb].add(x);
        }
        final int MOD = (int) (1e9 + 7);
        res3575 += cal3575(ret);
        res3575 %= MOD;
        return ret;
    }

    private List<Integer>[] list3575;
    private int[][] memo3575;

    private int cal3575(List<Integer>[] list) {
        this.list3575 = list;
        this.memo3575 = new int[10][1 << 10];
        for (int[] r : memo3575) {
            Arrays.fill(r, -1);
        }
        return dfs3575(0, 0);
    }

    private int dfs3575(int i, int j) {
        if (i == 10 || j == (1 << 10) - 1) {
            return 0;
        }
        if (memo3575[i][j] != -1) {
            return memo3575[i][j];
        }
        // 不选
        int res = dfs3575(i + 1, j);
        // 选
        if (((j >> i) & 1) == 0) {
            for (int x : list3575[i]) {
                // mask 一定不是0
                int mask = masks3575[x];
                if ((mask & j) == 0) {
                    res = Math.max(res, dfs3575(i + 1, mask | j) + vals3575[x]);
                }
            }
        }
        return memo3575[i][j] = res;
    }

    private int check3575(int v) {
        int mask = 0;
        while (v != 0) {
            int m = v % 10;
            if (((mask >> m) & 1) != 0) {
                return 0;
            }
            mask |= 1 << m;
            v /= 10;
        }
        return mask;
    }

    // 3759. 统计合格元素的数目 (Count Elements With at Least K Greater Values)
    public int countElements(int[] nums, int k) {
        int n = nums.length;
        if (k == 0) {
            return n;
        }
        Arrays.sort(nums);
        int left = 0;
        int right = n - 1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (nums[mid] >= nums[n - k]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return right + 1;

    }

    // 3760. 不同首字母的子字符串数目 (Maximum Substrings With Distinct Start)
    public int maxDistinct(String s) {
        int m = 0;
        for (char c : s.toCharArray()) {
            m |= 1 << (c - 'a');
        }
        return Integer.bitCount(m);

    }

    // 3761. 镜像对之间最小绝对距离 (Minimum Absolute Distance Between Mirror Pairs)
    public int minMirrorPairDistance(int[] nums) {
        int res = Integer.MAX_VALUE;
        Map<Integer, Integer> d = new HashMap<>();
        for (int i = 0; i < nums.length; ++i) {
            res = Math.min(res, i - d.getOrDefault(nums[i], Integer.MIN_VALUE / 2));
            d.put(reverse3761(nums[i]), i);
        }
        return res < nums.length ? res : -1;
    }

    private int reverse3761(int x) {
        int res = 0;
        while (x != 0) {
            res = res * 10 + x % 10;
            x /= 10;
        }
        return res;
    }

    // 3765. 完全质数 (Complete Prime Number)
    public boolean completePrime(int num) {
        String s = String.valueOf(num);
        int n = s.length();
        for (int i = 1; i < n + 1; ++i) {
            if (!checkPrime3765(Integer.parseInt(s.substring(0, i)))) {
                return false;
            }
        }
        for (int i = n - 1; i >= 0; --i) {
            if (!checkPrime3765(Integer.parseInt(s.substring(i, n)))) {
                return false;
            }
        }
        return true;

    }

    private boolean checkPrime3765(int x) {
        for (int i = 2; i <= Math.sqrt(x); ++i) {
            if (x % i == 0) {
                return false;
            }
        }
        return x > 1;
    }

    // 3766. 将数字变成二进制回文数的最少操作 (Minimum Operations to Make Binary Palindrome)
    public int[] minOperations(int[] nums) {
        TreeSet<Integer> ts = new TreeSet<>();
        for (int i = 0; i <= 5050; ++i) {
            StringBuilder sb = new StringBuilder(Integer.toBinaryString(i));
            if (sb.toString().equals(reversed3766(sb.toString()))) {
                ts.add(i);
            }
        }

        int n = nums.length;
        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            res[i] = Math.min(ts.ceiling(nums[i]) - nums[i], nums[i] - ts.floor(nums[i]));
        }
        return res;
    }

    private String reversed3766(String s) {
        StringBuilder sb = new StringBuilder(s);
        return sb.reverse().toString();
    }

    public long maxPoints(int[] technique1, int[] technique2, int k) {
        int n = technique1.length;
        long res = 0L;
        List<Integer> d = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            res += technique1[i];
            if (technique2[i] - technique1[i] > 0) {
                d.add(technique2[i] - technique1[i]);
            }
        }
        Collections.sort(d, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(o2, o1);
            }

        });
        for (int i = 0; i < Math.min(n - k, d.size()); ++i) {
            res += d.get(i);
        }
        return res;

    }

    // 3769. 二进制反射排序 (Sort Integers by Binary Reflection)
    public int[] sortByReflection(int[] nums) {
        List<Integer> a = new ArrayList<>();
        for (int x : nums) {
            a.add(x);
        }
        Collections.sort(a, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                int x = Integer.reverse(o1) >>> Integer.numberOfLeadingZeros(o1);
                int y = Integer.reverse(o2) >>> Integer.numberOfLeadingZeros(o2);
                if (x == y) {
                    return Integer.compare(o1, o2);
                }
                return Integer.compare(x, y);
            }

        });
        return a.stream().mapToInt(o -> o).toArray();

    }

    // 3770. 可表示为连续质数和的最大质数 (Largest Prime from Consecutive Prime Sum)
    public int largestPrime(int n) {
        int MX = n + 1;
        boolean[] p = new boolean[MX];
        Arrays.fill(p, true);
        for (int i = 2; i < MX; ++i) {
            if (p[i]) {
                for (long j = (long) i * i; j < MX; j += i) {
                    p[(int) j] = false;
                }
            }
        }
        int res = 0;
        int s = 0;
        for (int i = 2; i < MX; ++i) {
            if (p[i]) {
                s += i;
                if (s > n) {
                    break;
                }
                if (p[s]) {
                    res = s;
                }
            }
        }
        return res;
    }

    // 3771. 探索地牢的得分 (Total Score of Dungeon Runs)
    public long totalScore(int hp, int[] damage, int[] requirement) {
        long res = 0L;
        int n = damage.length;
        long[] s = new long[n + 1];
        for (int i = 0; i < n; ++i) {
            s[i + 1] = s[i] + damage[i];
        }
        for (int i = 0; i < n; ++i) {
            long x = s[i + 1] + requirement[i] - hp;
            int j = binarySearch3771(s, x, i);
            res += i - j + 1;
        }
        return res;

    }

    private int binarySearch3771(long[] s, long x, int i) {
        int left = 0;
        int right = i;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (s[mid] >= x) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return right + 1;
    }

    // 3772. 子图的最大得分 (Maximum Subgraph Score in a Tree)
    private List<Integer>[] g3772;
    private int[] good3772;
    private int[] s3772;
    private int[] res3772;

    public int[] maxSubgraphScore(int n, int[][] edges, int[] good) {
        this.g3772 = new ArrayList[n];
        Arrays.setAll(g3772, k -> new ArrayList<>());
        for (int[] e : edges) {
            g3772[e[0]].add(e[1]);
            g3772[e[1]].add(e[0]);
        }
        this.good3772 = good;
        this.s3772 = new int[n];
        dfs3772(0, -1);
        this.res3772 = new int[n];
        reRoot3772(0, -1, 0);
        return res3772;
    }

    private void reRoot3772(int x, int fa, int faScore) {
        int score = s3772[x] + Math.max(0, faScore);
        res3772[x] = score;
        for (int y : g3772[x]) {
            if (y != fa) {
                reRoot3772(y, x, score - Math.max(0, s3772[y]));
            }
        }
    }

    private int dfs3772(int x, int fa) {
        for (int y : g3772[x]) {
            if (y != fa) {
                s3772[x] += Math.max(0, dfs3772(y, x));
            }
        }
        s3772[x] += good3772[x] == 1 ? 1 : -1;
        return s3772[x];
    }

    // 3774. 最大和最小 K 个元素的绝对差 (Absolute Difference Between Maximum and Minimum K
    // Elements)
    public int absDifference(int[] nums, int k) {
        int res = 0;
        int n = nums.length;
        Arrays.sort(nums);
        for (int i = 0; i < n; ++i) {
            if (i < k) {
                res -= nums[i];
            }
            if (i >= n - k) {
                res += nums[i];
            }
        }
        return res;

    }

    // 3775. 反转元音数相同的单词 (Reverse Words With Same Vowel Count)
    public String reverseWords(String s) {
        String[] split = s.split(" ");
        StringBuilder res = new StringBuilder();
        int mask = 0;
        for (char c : "aeiou".toCharArray()) {
            mask |= 1 << (c - 'a');
        }
        int cnt = -1;
        for (String word : split) {
            if (!res.isEmpty()) {
                res.append(" ");
            }
            if (cnt == -1) {
                cnt = check3775(word, mask);
                res.append(word);
            } else if (check3775(word, mask) == cnt) {
                StringBuilder rev = new StringBuilder(word).reverse();
                res.append(rev.toString());
            } else {
                res.append(word);
            }
        }
        return res.toString();
    }

    private int check3775(String word, int mask) {
        int cnt = 0;
        for (char c : word.toCharArray()) {
            if (((mask >> (c - 'a')) & 1) != 0) {
                ++cnt;
            }
        }
        return cnt;
    }

    // 3776. 使循环数组余额非负的最少移动次数 (Minimum Moves to Balance Circular Array)
    public long minMoves3776(int[] balance) {
        long s = 0L;
        int id = -1;
        int n = balance.length;
        for (int i = 0; i < n; ++i) {
            s += balance[i];
            if (balance[i] < 0) {
                id = i;
            }
        }
        if (s < 0) {
            return -1;
        }
        if (id < 0) {
            return 0L;
        }
        long res = 0L;
        int need = Math.abs(balance[id]);
        for (int d = 1; need > 0; ++d) {
            int cur = Math.min(balance[((id - d) % n + n) % n] + balance[(id + d) % n], need);
            need -= cur;
            res += (long) cur * d;
        }
        return res;

    }

    // 3779. 得到互不相同元素的最少操作次数 (Minimum Number of Operations to Have Distinct
    // Elements)
    public int minOperations3779(int[] nums) {
        int n = nums.length;
        int res = 0;
        Map<Integer, Integer> cnts = new HashMap<>();
        for (int x : nums) {
            cnts.merge(x, 1, Integer::sum);
        }
        for (int i = 0; i < n; i += 3) {
            if (cnts.size() == n - i) {
                break;
            }
            ++res;
            for (int j = i; j < Math.min(n, i + 3); ++j) {
                int x = nums[j];
                cnts.merge(x, -1, Integer::sum);
                if (cnts.get(x) == 0) {
                    cnts.remove(x);
                }
            }
        }
        return res;

    }

    // 3780. 能被 3 整除的三元组最大和 (Maximum Sum of Three Numbers Divisible by Three)
    public int maximumSum(int[] nums) {
        int[][] g = new int[3][3];
        for (int i = 0; i < 3; ++i) {
            Arrays.fill(g[i], 0);
        }
        for (int x : nums) {
            int m = x % 3;
            if (x >= g[m][0]) {
                g[m][2] = g[m][1];
                g[m][1] = g[m][0];
                g[m][0] = x;
            } else if (x >= g[m][1]) {
                g[m][2] = g[m][1];
                g[m][1] = x;
            } else if (x >= g[m][2]) {
                g[m][2] = x;
            }
        }
        int res = 0;
        for (int i = 0; i < 3; ++i) {
            if (g[i][2] > 0) {
                res = Math.max(res, g[i][0] + g[i][1] + g[i][2]);
            }
        }
        if (g[0][0] > 0 && g[1][0] > 0 && g[2][0] > 0) {
            res = Math.max(res, g[0][0] + g[1][0] + g[2][0]);
        }
        return res;

    }

    // 3783. 整数的镜像距离 (Mirror Distance of an Integer)
    public int mirrorDistance(int n) {
        return Math.abs(n - rev3783(n));
    }

    private int rev3783(int x) {
        int res = 0;
        while (x != 0) {
            res = res * 10 + x % 10;
            x /= 10;
        }
        return res;
    }

    // 3784. 使所有字符相等的最小删除代价 (Minimum Deletion Cost to Make All Characters Equal)
    public long minCost3784(String s, int[] cost) {
        long[] d = new long[26];
        long sum = 0L;
        long mx = 0L;
        for (int i = 0; i < s.length(); ++i) {
            int idx = s.charAt(i) - 'a';
            sum += cost[i];
            d[idx] += cost[i];
            mx = Math.max(mx, d[idx]);
        }
        return sum - mx;

    }

    // 3786. 树组的交互代价总和 (Total Sum of Interaction Cost in Tree Groups)
    private long res3776;

    public long interactionCosts(int n, int[][] edges, int[] group) {
        Map<Integer, Integer> gCnts = new HashMap<>();
        for (int g : group) {
            gCnts.merge(g, 1, Integer::sum);
        }
        List<Integer>[] g = new ArrayList[n];
        Arrays.setAll(g, k -> new ArrayList<>());
        for (int[] e : edges) {
            g[e[0]].add(e[1]);
            g[e[1]].add(e[0]);
        }
        for (Map.Entry<Integer, Integer> entry : gCnts.entrySet()) {
            int grp = entry.getKey();
            int cnt = entry.getValue();
            dfs3786(0, -1, g, group, grp, cnt);
        }
        return res3776;

    }

    private int dfs3786(int x, int fa, List<Integer>[] g, int[] group, int grp, int cnt) {
        int s = group[x] == grp ? 1 : 0;
        for (int y : g[x]) {
            if (y != fa) {
                int t = dfs3786(y, x, g, group, grp, cnt);
                res3776 += (long) t * (cnt - t);
                s += t;
            }
        }
        return s;
    }

    // 3781. 二进制交换后的最大分数 (Maximum Score After Binary Swaps)
    public long maximumScore(int[] nums, String s) {
        int n = nums.length;
        Queue<Integer> pq = new PriorityQueue<>();
        long res = 0L;
        for (int i = n - 1; i >= 0; --i) {
            pq.offer(nums[i]);
            res += nums[i];
            if (s.charAt(i) == '0') {
                res -= pq.poll();
            }
        }
        return res;

    }

    // 3785. 避免禁用值的最小交换次数 (Minimum Swaps to Avoid Forbidden Values)
    public int minSwaps(int[] nums, int[] forbidden) {
        int n = nums.length;
        Map<Integer, Integer> cnts = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            cnts.merge(nums[i], 1, Integer::sum);
            if (cnts.get(nums[i]) > n) {
                return -1;
            }

            cnts.merge(forbidden[i], 1, Integer::sum);
            if (cnts.get(forbidden[i]) > n) {
                return -1;
            }
        }
        Map<Integer, Integer> a = new HashMap<>();
        int k = 0;
        int mx = 0;
        for (int i = 0; i < n; ++i) {
            if (nums[i] == forbidden[i]) {
                a.merge(nums[i], 1, Integer::sum);
                ++k;
                mx = Math.max(mx, a.get(nums[i]));
            }
        }
        return Math.max(mx, (k + 1) / 2);

    }

    // 3782. 交替删除操作后最后剩下的整数 (Last Remaining Integer After Alternating Deletion
    // Operations)
    public long lastInteger(long n) {
        long start = 1L;
        long d = 1L;
        while (n > 1) {
            start += (n - 2 + n % 2) * d;
            d *= -2L;
            n = (n + 1) / 2;
        }
        return start;
    }

    // 3788. 分割的最大得分 (Maximum Score of a Split)
    public long maximumScore(int[] nums) {
        int n = nums.length;
        int[] sufMin = new int[n];
        sufMin[n - 1] = Integer.MAX_VALUE;
        for (int i = n - 2; i >= 0; --i) {
            sufMin[i] = Math.min(sufMin[i + 1], nums[i + 1]);
        }
        long preSum = 0L;
        long res = Long.MIN_VALUE;
        for (int i = 0; i < n - 1; ++i) {
            preSum += nums[i];
            res = Math.max(res, preSum - sufMin[i]);
        }
        return res;
    }

    // 3789. 采购的最小花费 (Minimum Cost to Acquire Required Items)
    public long minimumCost(int cost1, int cost2, int costBoth, int need1, int need2) {
        long res1 = (long) cost1 * need1 + (long) cost2 * need2;
        long res2 = (long) costBoth * Math.max(need1, need2);
        long res3 = 0L;
        int m = Math.min(need1, need2);
        res3 += (long) costBoth * m;
        need1 -= m;
        need2 -= m;
        res3 += (long) cost1 * need1 + (long) cost2 * need2;
        return Math.min(res1, Math.min(res2, res3));
    }

    // 3790. 最小全 1 倍数 (mallest All-Ones Multiple)
    public int minAllOneMultiple(int k) {
        if (k % 2 == 0 || k % 5 == 0) {
            return -1;
        }
        int mod = 0;
        for (int i = 1; i <= k; ++i) {
            mod = (mod * 10 + 1) % k;
            if (mod == 0) {
                return i;
            }
        }
        return -1;
    }

    // 3791. 给定范围内平衡整数的数目 (Number of Balanced Integers in a Range)
    public long countBalanced(long low, long high) {
        return cal3791(high) - cal3791(low - 1);
    }

    private long[][][] memo3791;
    private String s3791;
    private int n3791;

    private long cal3791(long x) {
        this.s3791 = String.valueOf(x);
        this.n3791 = s3791.length();
        this.memo3791 = new long[n3791][9 * ((n3791 + 1) / 2) + 1][9 * ((n3791 + 1) / 2) + 1];
        for (long[][] r1 : memo3791) {
            for (long[] r2 : r1) {
                Arrays.fill(r2, -1L);
            }
        }
        return dfs3791(0, 0, 0, true);
    }

    private long dfs3791(int i, int j, int k, boolean isLimit) {
        if (i == n3791) {
            return j == k ? 1L : 0L;
        }
        if (!isLimit && memo3791[i][j][k] != -1L) {
            return memo3791[i][j][k];
        }
        long res = 0L;
        int up = isLimit ? (s3791.charAt(i) - '0') : 9;
        for (int d = 0; d <= up; ++d) {
            res += dfs3791(i + 1, j + (i % 2 == 0 ? d : 0), k + (i % 2 == 1 ? d : 0), isLimit && d == up);
        }
        if (!isLimit) {
            memo3791[i][j][k] = res;
        }
        return res;
    }

    // 961. 在长度 2N 的数组中找出重复 N 次的元素 (N-Repeated Element in Size 2N Array)
    public int repeatedNTimes(int[] nums) {
        Set<Integer> s = new HashSet<>();
        for (int x : nums) {
            if (s.contains(x)) {
                return x;
            }
            s.add(x);
        }
        return -1;
    }

    // 3794. 反转字符串前缀 (Reverse String Prefix)
    public String reversePrefix(String s, int k) {
        return new StringBuilder(s.substring(0, k)).reverse().toString() + s.substring(k);
    }

    // 3795. 不同元素和至少为 K 的最短子数组长度 (Minimum Subarray Length With Distinct Sum At Least
    // K)
    public int minLength(int[] nums, int k) {
        Map<Integer, Integer> cnts = new HashMap<>();
        int n = nums.length;
        int res = Integer.MAX_VALUE;
        int s = 0;
        int left = 0;
        for (int right = 0; right < n; ++right) {
            cnts.merge(nums[right], 1, Integer::sum);
            if (cnts.get(nums[right]) == 1) {
                s += nums[right];
            }
            while (s >= k) {
                res = Math.min(res, right - left + 1);
                cnts.merge(nums[left], -1, Integer::sum);
                if (cnts.get(nums[left]) == 0) {
                    s -= nums[left];
                    cnts.remove(nums[left]);
                }
                ++left;
            }
        }
        return res < Integer.MAX_VALUE ? res : -1;

    }

    // 3797. 统计在矩形格子里移动的路径数目 (Count Routes to Climb a Rectangular Grid)
    public int numberOfRoutes(String[] grid, int d) {
        int m = grid.length;
        int n = grid[0].length();
        int[] f = new int[n];
        int[] pre = new int[n + 1];
        for (int j = 0; j < n; ++j) {
            pre[j + 1] = pre[j] + (grid[m - 1].charAt(j) == '.' ? 1 : 0);
        }
        for (int j = 0; j < n; ++j) {
            if (grid[m - 1].charAt(j) == '#') {
                continue;
            }
            f[j] = pre[Math.min(n, j + d + 1)] - pre[Math.max(0, j - d)];
        }
        int dis = (int) Math.sqrt(d * d - 1);
        final int MOD = (int) (1e9 + 7);
        for (int i = m - 2; i >= 0; --i) {
            int[] newF = new int[n];
            int[] cur = new int[n];
            pre = new int[n + 1];
            for (int j = 0; j < n; ++j) {
                pre[j + 1] = pre[j] + f[j];
                pre[j + 1] %= MOD;
            }
            for (int j = 0; j < n; ++j) {
                if (grid[i].charAt(j) == '#') {
                    continue;
                }
                cur[j] = pre[Math.min(n, j + dis + 1)] - pre[Math.max(0, j - dis)];
                cur[j] = (cur[j] % MOD + MOD) % MOD;
            }
            pre = new int[n + 1];
            for (int j = 0; j < n; ++j) {
                pre[j + 1] = pre[j] + cur[j];
                pre[j + 1] %= MOD;
            }
            for (int j = 0; j < n; ++j) {
                if (grid[i].charAt(j) == '#') {
                    continue;
                }
                newF[j] = pre[Math.min(n, j + d + 1)] - pre[Math.max(0, j - d)];
                newF[j] = (newF[j] % MOD + MOD) % MOD;
            }
            f = newF;
        }
        int res = 0;
        for (int x : f) {
            res += x;
            res %= MOD;
        }
        return res;

    }

    // 3798. 最大的偶数 (Largest Even Number)
    public String largestEven(String s) {
        for (int i = s.length() - 1; i >= 0; --i) {
            if (s.charAt(i) == '2') {
                return s.substring(0, i + 1);
            }
        }
        return "";
    }

    // 3799. 单词方块 II (Word Squares II)
    public List<List<String>> wordSquares(String[] words) {
        List<List<String>> res = new ArrayList<>();
        Arrays.sort(words);
        dfs3799(0, new ArrayList<>(), words, res);
        return res;
    }

    private void dfs3799(int m, List<String> a, String[] words, List<List<String>> res) {
        if (a.size() == 4) {
            res.add(new ArrayList<>(a));
            return;
        }
        for (int i = 0; i < words.length; ++i) {
            if (((m >> i) & 1) != 0) {
                continue;
            }
            if (a.size() == 1 && words[i].charAt(0) != a.get(0).charAt(0)) {
                continue;
            }
            if (a.size() == 2 && (words[i].charAt(0) != a.get(0).charAt(3))) {
                continue;
            }
            if (a.size() == 3
                    && (words[i].charAt(0) != a.get(1).charAt(3) || words[i].charAt(3) != a.get(2).charAt(3))) {
                continue;
            }
            a.add(words[i]);
            dfs3799(m | (1 << i), a, words, res);
            a.remove(a.size() - 1);
        }
    }

    // 3800. 使二进制字符串相等的最小成本 (Minimum Cost to Make Two Binary Strings Equal)
    public long minimumCost(String s, String t, int flipCost, int swapCost, int crossCost) {
        int x = 0;
        int y = 0;
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == t.charAt(i)) {
                continue;
            }
            if (s.charAt(i) == '0') {
                ++x;
            } else {
                ++y;
            }
        }
        if (x < y) {
            int temp = x;
            x = y;
            y = temp;
        }
        long res1 = (long) (x + y) * flipCost;
        long res2 = (x + y) % 2 == 0 ? 0L : flipCost;
        if ((x + y) % 2 != 0) {
            --x;
        }
        res2 += (long) swapCost * y;
        x -= y;
        res2 += Math.min((long) (x / 2) * (crossCost + swapCost), (long) flipCost * x);
        return Math.min(res1, res2);
    }

    // 3796. 找到带限制序列的最大值 (Find Maximum Value in a Constrained Sequence)
    public int findMaxVal(int n, int[][] restrictions, int[] diff) {
        int[] a = new int[n];
        for (int[] r : restrictions) {
            a[r[0]] = r[1];
        }
        for (int i = 1; i < n; ++i) {
            if (a[i] != 0) {
                a[i] = Math.min(a[i], a[i - 1] + diff[i - 1]);
            } else {
                a[i] = a[i - 1] + diff[i - 1];
            }
        }
        int res = a[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            a[i] = Math.min(a[i], a[i + 1] + diff[i]);
            res = Math.max(res, a[i]);
        }
        return res;

    }

    // 3801. 合并有序列表的最小成本 (Minimum Cost to Merge Sorted Lists)
    private long[] memo3801;
    private int[][] g3801;

    public long minMergeCost(int[][] lists) {
        int n = lists.length;
        this.g3801 = new int[1 << n][0];
        for (int i = 0; i < n; ++i) {
            int highest_bit = 1 << i;
            for (int s = 0; s < highest_bit; ++s) {
                g3801[s | highest_bit] = merge3801(g3801[s], lists[i]);
            }
        }
        this.memo3801 = new long[1 << n];
        Arrays.fill(memo3801, -1L);
        return dfs3801((1 << n) - 1);
    }

    private long dfs3801(int c) {
        if ((c & (c - 1)) == 0) {
            return 0;
        }
        if (memo3801[c] != -1L) {
            return memo3801[c];
        }
        long res = Long.MAX_VALUE;
        // 避免对称枚举
        for (int subA = c; subA > (c ^ (subA)); subA = (subA - 1) & c) {
            int subB = c ^ subA;
            if (subB != 0) {
                int lenA = g3801[subA].length;
                int lenB = g3801[subB].length;
                int medA = g3801[subA][(lenA - 1) / 2];
                int medB = g3801[subB][(lenB - 1) / 2];
                res = Math.min(res, dfs3801(subA) + dfs3801(subB) + lenA + lenB + Math.abs(medA - medB));
            }
        }
        return memo3801[c] = res;
    }

    // 3801. 合并有序列表的最小成本 (Minimum Cost to Merge Sorted Lists)
    public long minMergeCost2(int[][] lists) {
        int n = lists.length;
        int[][] g = new int[1 << n][0];
        for (int i = 0; i < n; ++i) {
            int highest_bit = 1 << i;
            for (int s = 0; s < highest_bit; ++s) {
                g[s | highest_bit] = merge3801(g[s], lists[i]);
            }
        }
        long[] f = new long[1 << n];
        for (int i = 1; i < 1 << n; ++i) {
            if ((i & (i - 1)) == 0) {
                continue;
            }
            f[i] = Long.MAX_VALUE;
            // 避免对称枚举
            for (int subA = i; subA > (i ^ subA); subA = (subA - 1) & i) {
                int subB = i ^ subA;
                if (subB != 0) {
                    int lenA = g[subA].length;
                    int lenB = g[subB].length;
                    int medA = g[subA][(lenA - 1) / 2];
                    int medB = g[subB][(lenB - 1) / 2];
                    f[i] = Math.min(f[i], f[subA] + f[subB] + lenA + lenB + Math.abs(medA - medB));
                }
            }
        }
        return f[(1 << n) - 1];
    }

    private int[] merge3801(int[] a, int[] b) {
        int i = 0;
        int j = 0;
        int m = a.length;
        int n = b.length;
        int[] res = new int[m + n];
        while (i < m && j < n) {
            if (a[i] < b[j]) {
                res[i + j] = a[i];
                ++i;
            } else {
                res[i + j] = b[j];
                ++j;
            }
        }
        while (i < m) {
            res[i + j] = a[i];
            ++i;
        }
        while (j < n) {
            res[i + j] = b[j];
            ++j;
        }
        return res;

    }

    // 3803. 统计残差前缀 (Count Residue Prefixes)
    public int residuePrefixes(String s) {
        int res = 0;
        int mask = 0;
        for (int i = 0; i < s.length() && Integer.bitCount(mask) < 3; ++i) {
            mask |= 1 << (s.charAt(i) - 'a');
            if (Integer.bitCount(mask) == (i + 1) % 3) {
                ++res;
            }
        }
        return res;

    }

    // 3804. 中心子数组的数量 (Number of Centered Subarrays)
    public int centeredSubarrays(int[] nums) {
        int res = 0;
        int n = nums.length;
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < n; ++i) {
            set.clear();
            int s = 0;
            for (int j = i; j < n; ++j) {
                s += nums[j];
                set.add(nums[j]);
                if (set.contains(s)) {
                    ++res;
                }
            }
        }
        return res;

    }

    // 3805. 统计凯撒加密对数目 (Count Caesar Cipher Pairs)
    public long countPairs(String[] words) {
        long res = 0L;
        Map<String, Integer> cnts = new HashMap<>();
        for (String s : words) {
            String transStr = trans3805(s);
            res += cnts.getOrDefault(transStr, 0);
            cnts.merge(transStr, 1, Integer::sum);
        }
        return res;

    }

    private String trans3805(String s) {
        int d = s.charAt(0) - 'a';
        if (d == 0) {
            return s;
        }
        d = 26 - d;
        char[] a = s.toCharArray();
        for (int i = 0; i < s.length(); ++i) {
            a[i] = (char) (((a[i] - 'a' + d) % 26) + 'a');
        }
        return String.valueOf(a);
    }

    // 3806. 增加操作后最大按位与的结果 (Maximum Bitwise AND After Increment Operations)
    public int maximumAND(int[] nums, int k, int m) {
        int n = nums.length;
        int res = 0;
        int[] ops = new int[n];
        int max = 0;
        for (int x : nums) {
            max = Math.max(max, x);
        }
        max += k;
        int high = 32 - Integer.numberOfLeadingZeros(max);
        search: for (int bit = high - 1; bit >= 0; --bit) {
            int target = res | (1 << bit);
            for (int i = 0; i < n; ++i) {
                int j = 32 - Integer.numberOfLeadingZeros(target & ~nums[i]);
                int mask = (1 << j) - 1;
                ops[i] = (target & mask) - (nums[i] & mask);
            }
            Arrays.sort(ops);
            int s = 0;
            for (int i = 0; i < m; ++i) {
                s += ops[i];
                if (s > k) {
                    continue search;
                }
            }
            res = target;
        }
        return res;

    }

    // 3453. 分割正方形 I (Separate Squares I)
    public double separateSquares(int[][] squares) {
        double left = Double.MAX_VALUE;
        double right = Double.MIN_VALUE;
        double total = 0D;
        for (int[] s : squares) {
            left = Math.min(left, s[1]);
            right = Math.max(right, s[1] + s[2]);
            total += (double) s[2] * s[2];
        }
        while (right - left > 1e-5) {
            double mid = left + (right - left) / 2D;
            if (check3453(mid, squares) * 2D >= total) {
                right = mid - 1e-6;
            } else {
                left = mid + 1e-6;
            }
        }
        return right + 1e-6;

    }

    private double check3453(double y, int[][] squares) {
        double res = 0D;
        for (int[] s : squares) {
            res += s[2] * Math.max(0D, Math.min(y, s[1] + s[2]) - s[1]);
        }
        return res;
    }

    // 3434. 子数组操作后的最大频率 (Maximum Frequency After Subarray Operation)
    private int[] nums3434;
    private int k3434;
    private int n3434;

    public int maxFrequency(int[] nums, int k) {
        this.nums3434 = nums;
        this.n3434 = nums.length;
        this.k3434 = k;
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for (int x : nums) {
            min = Math.min(min, x);
            max = Math.max(max, x);
        }
        int res = 0;
        for (int d = k - max; d <= k - min; ++d) {
            res = Math.max(res, cal3434(d));
        }
        return res;
    }

    private int d3434;
    private int[][] memo3434;

    private int cal3434(int d) {
        this.d3434 = d;
        this.memo3434 = new int[n3434][3];
        for (int[] r : memo3434) {
            Arrays.fill(r, -1);
        }
        return dfs3434(n3434 - 1, 0);
    }

    private int dfs3434(int i, int j) {
        if (i < 0) {
            return 0;
        }
        if (memo3434[i][j] != -1) {
            return memo3434[i][j];
        }
        int res = 0;
        // j == 0 之前未修改过
        if (j == 0) {
            // 不修改
            res = Math.max(res, dfs3434(i - 1, j) + (nums3434[i] == k3434 ? 1 : 0));
            // 修改
            res = Math.max(res, dfs3434(i - 1, j + 1) + (nums3434[i] + d3434 == k3434 ? 1 : 0));
        }
        // j == 1 已经在修改途中
        else if (j == 1) {
            // 仍然修改
            res = Math.max(res, dfs3434(i - 1, j) + (nums3434[i] + d3434 == k3434 ? 1 : 0));
            // 恢复为不修改
            res = Math.max(res, dfs3434(i - 1, j + 1) + (nums3434[i] == k3434 ? 1 : 0));
        }
        // j == 2 已恢复为不修改
        else {
            res = Math.max(res, dfs3434(i - 1, j) + (nums3434[i] == k3434 ? 1 : 0));
        }
        return memo3434[i][j] = res;
    }

    // 998. 最大二叉树 II (Maximum Binary Tree II)
    public TreeNode insertIntoMaxTree(TreeNode root, int val) {
        TreeNode fa = null;
        TreeNode x = root;
        while (x != null) {
            if (val > x.val) {
                if (fa == null) {
                    return new TreeNode(val, root, null);
                }
                fa.right = new TreeNode(val, x, null);
                return root;
            }
            fa = x;
            x = x.right;
        }
        fa.right = new TreeNode(val);
        return root;
    }

    // 829. 连续整数求和 (Consecutive Numbers Sum)
    public int consecutiveNumbersSum(int s) {
        // 枚举序列长度
        int n = 1;
        int res = 0;
        while (n * (n + 1) <= s * 2) {
            if (s * 2 > n * (n - 1) && (s * 2 - n * (n - 1)) % (n * 2) == 0) {
                ++res;
            }
            ++n;
        }
        return res;

    }

    // 3809. 最好可到达的塔 (Best Reachable Tower)
    public int[] bestTower(int[][] towers, int[] center, int radius) {
        int[] res = { -1, -1 };
        int maxQuality = -1;
        for (int[] tower : towers) {
            int dx = Math.abs(tower[0] - center[0]);
            int dy = Math.abs(tower[1] - center[1]);
            if (dx + dy <= radius) {
                if (tower[2] > maxQuality) {
                    maxQuality = tower[2];
                    res[0] = tower[0];
                    res[1] = tower[1];
                } else if (tower[2] == maxQuality && (tower[0] < res[0] || (tower[0] == res[0] && tower[1] < res[1]))) {
                    res[0] = tower[0];
                    res[1] = tower[1];
                }
            }
        }
        return res;

    }

    // 3810. 变成目标数组的最少操作次数 (Minimum Operations to Reach Target Array)
    public int minOperations3810(int[] nums, int[] target) {
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < target.length; ++i) {
            if (nums[i] != target[i]) {
                set.add(nums[i]);
            }
        }
        return set.size();

    }

    // 3811. 交替按位异或分割的数目 (Number of Alternating XOR Partitions)
    private Map<Long, Integer> memo3811;
    private int[] nums3811;
    private int n3811;
    private int[] target3811;

    public int alternatingXOR(int[] nums, int target1, int target2) {
        this.n3811 = nums.length;
        this.nums3811 = nums;
        this.target3811 = new int[] { target1, target2 };
        this.memo3811 = new HashMap<>();
        this.n3811 = nums.length;
        return dfs3811(0, 0, 0);

    }

    private int dfs3811(int i, int j, int k) {
        if (i == n3811) {
            return j == (1 << 18) - 1 ? 1 : 0;
        }
        long key = (((long) i) << 20) | (((long) j) << 1) | ((long) k);
        if (memo3811.containsKey(key)) {
            return memo3811.get(key);
        }
        final int MOD = (int) (1e9 + 7);
        int nj = j == (1 << 18) - 1 ? nums3811[i] : (j ^ nums3811[i]);
        int res = dfs3811(i + 1, nj, k);
        if (nj == target3811[k]) {
            res += dfs3811(i + 1, (1 << 18) - 1, k ^ 1);
            res %= MOD;
        }
        memo3811.put(key, res);
        return res;
    }

    // 3811. 交替按位异或分割的数目 (Number of Alternating XOR Partitions)
    public int alternatingXOR2(int[] nums, int target1, int target2) {
        Map<Integer, Integer> f1 = new HashMap<>();
        Map<Integer, Integer> f2 = new HashMap<>();
        f2.put(0, 1);
        final int MOD = (int) (1e9 + 7);
        int xor = 0;
        for (int i = 0; i < nums.length; ++i) {
            xor ^= nums[i];
            int last1 = f2.getOrDefault(xor ^ target1, 0);
            int last2 = f1.getOrDefault(xor ^ target2, 0);
            if (i == nums.length - 1) {
                return (last1 + last2) % MOD;
            }
            f1.put(xor, (f1.getOrDefault(xor, 0) + last1) % MOD);
            f2.put(xor, (f2.getOrDefault(xor, 0) + last2) % MOD);
        }
        return 0;

    }

    // 3812. 翻转树上最少边 (Minimum Edge Toggles on a Tree)
    private List<Integer> res3812;
    private List<int[]>[] g3812;
    private String start3812;
    private String target3812;

    public List<Integer> minimumFlips(int n, int[][] edges, String start, String target) {
        this.res3812 = new ArrayList<>();
        this.g3812 = new ArrayList[n];
        Arrays.setAll(g3812, k -> new ArrayList<>());
        for (int i = 0; i < edges.length; ++i) {
            g3812[edges[i][0]].add(new int[] { edges[i][1], i });
            g3812[edges[i][1]].add(new int[] { edges[i][0], i });
        }
        this.start3812 = start;
        this.target3812 = target;
        if (dfs3812(0, -1) != 0) {
            return List.of(-1);
        }
        Collections.sort(res3812);
        return res3812;
    }

    private int dfs3812(int x, int fa) {
        int cnt = Math.abs(start3812.charAt(x) - target3812.charAt(x));
        for (int[] neighbor : g3812[x]) {
            int y = neighbor[0];
            int id = neighbor[1];
            if (y != fa) {
                int cntY = dfs3812(y, x);
                if (cntY != 0) {
                    res3812.add(id);
                }
                cnt ^= cntY;
            }
        }
        return cnt;
    }

    // 3813. 元音辅音得分 (Vowel-Consonant Score)
    public int vowelConsonantScore(String s) {
        int v = 0;
        int c = 0;
        int u = 0;
        for (char chr : "aeiou".toCharArray()) {
            u |= 1 << (chr - 'a');
        }
        for (char chr : s.toCharArray()) {
            if (Character.isLetter(chr)) {
                if (((u >> (chr - 'a')) & 1) != 0) {
                    ++v;
                } else {
                    ++c;
                }
            }
        }
        return c != 0 ? v / c : 0;
    }

    // 3814. 预算下的最大总容量 (Maximum Capacity Within Budget)
    public int maxCapacity(int[] costs, int[] capacity, int budget) {
        int res = 0;
        // 仅一个机器
        for (int i = 0; i < costs.length; ++i) {
            if (costs[i] < budget) {
                res = Math.max(res, capacity[i]);
            }
        }
        if (res == 0) {
            return res;
        }
        // 两个cost相同的机器
        Map<Integer, List<Integer>> cnts = new HashMap<>();
        for (int i = 0; i < costs.length; ++i) {
            cnts.computeIfAbsent(costs[i], o -> new ArrayList<>()).add(capacity[i]);
        }
        for (Map.Entry<Integer, List<Integer>> entry : cnts.entrySet()) {
            int cost = entry.getKey();
            List<Integer> caps = entry.getValue();
            Collections.sort(caps, Collections.reverseOrder());
            if (caps.size() >= 2 && cost * 2 < budget) {
                res = Math.max(res, caps.get(0) + caps.get(1));
            }
        }
        // 不同cost的两个机器
        List<int[]> machines = new ArrayList<>();
        for (Map.Entry<Integer, List<Integer>> entry : cnts.entrySet()) {
            int cost = entry.getKey();
            List<Integer> caps = entry.getValue();
            if (cost >= budget) {
                continue;
            }
            machines.add(new int[] { cost, caps.get(0) });
        }
        machines.sort(Comparator.comparingInt(a -> a[0]));
        int[] preMax = new int[machines.size()];
        preMax[0] = machines.get(0)[1];
        for (int i = 1; i < machines.size(); ++i) {
            preMax[i] = Math.max(preMax[i - 1], machines.get(i)[1]);
        }
        for (int i = 1; i < machines.size(); ++i) {
            int cost = machines.get(i)[0];
            int cap = machines.get(i)[1];
            int remain = budget - cost;
            int left = 0;
            int right = i - 1;
            while (left <= right) {
                int mid = left + (right - left) / 2;
                if (machines.get(mid)[0] < remain) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            if (left - 1 >= 0) {
                res = Math.max(res, cap + preMax[left - 1]);
            }
        }
        return res;
    }

    // 3815. 设计拍卖系统 (Design Auction System)
    class AuctionSystem {
        // itemId -> (userId -> bidAmount)
        private Map<Integer, Map<Integer, Integer>> itemToUserToBids;
        // itemId -> (bidAmount -> TreeSet of userId)
        private Map<Integer, TreeMap<Integer, TreeSet<Integer>>> itemToBidsToUsers;

        public AuctionSystem() {
            this.itemToUserToBids = new HashMap<>();
            this.itemToBidsToUsers = new HashMap<>();
        }

        public void addBid(int userId, int itemId, int bidAmount) {
            Integer oldAmount = itemToUserToBids.computeIfAbsent(itemId, o -> new HashMap<>()).put(userId, bidAmount);
            if (oldAmount != null) {
                itemToBidsToUsers.get(itemId).get(oldAmount).remove(userId);
                if (itemToBidsToUsers.get(itemId).get(oldAmount).isEmpty()) {
                    itemToBidsToUsers.get(itemId).remove(oldAmount);
                }
            }
            itemToBidsToUsers.computeIfAbsent(itemId, o -> new TreeMap<>((o1, o2) -> Integer.compare(o2, o1)))
                    .computeIfAbsent(bidAmount, o -> new TreeSet<>((o1, o2) -> Integer.compare(o2, o1))).add(userId);
        }

        public void updateBid(int userId, int itemId, int newAmount) {
            int oldAmount = itemToUserToBids.get(itemId).put(userId, newAmount);
            itemToBidsToUsers.get(itemId).get(oldAmount).remove(userId);
            if (itemToBidsToUsers.get(itemId).get(oldAmount).isEmpty()) {
                itemToBidsToUsers.get(itemId).remove(oldAmount);
            }
            itemToBidsToUsers.get(itemId)
                    .computeIfAbsent(newAmount, o -> new TreeSet<>((o1, o2) -> Integer.compare(o2, o1))).add(userId);

        }

        public void removeBid(int userId, int itemId) {
            int oldAmount = itemToUserToBids.get(itemId).remove(userId);
            itemToBidsToUsers.get(itemId).get(oldAmount).remove(userId);
            if (itemToBidsToUsers.get(itemId).get(oldAmount).isEmpty()) {
                itemToBidsToUsers.get(itemId).remove(oldAmount);
            }
        }

        public int getHighestBidder(int itemId) {
            if (!itemToBidsToUsers.containsKey(itemId) || itemToBidsToUsers.get(itemId).isEmpty()) {
                return -1;
            }
            Map.Entry<Integer, TreeSet<Integer>> entry = itemToBidsToUsers.get(itemId).firstEntry();
            return entry.getValue().first();
        }
    }

    // 3816. 删除重复字符后的字典序最小字符串 (Lexicographically Smallest String After Deleting
    // Duplicate Characters)
    public String lexSmallestAfterDeletion(String s) {
        int[] cnts = new int[26];
        for (char c : s.toCharArray()) {
            ++cnts[c - 'a'];
        }
        Stack<Character> st = new Stack<>();
        for (char c : s.toCharArray()) {
            while (!st.isEmpty() && st.peek() > c && cnts[st.peek() - 'a'] > 1) {
                --cnts[st.pop() - 'a'];
            }
            st.push(c);
        }
        while (cnts[st.peek() - 'a'] > 1) {
            --cnts[st.pop() - 'a'];
        }
        StringBuilder sb = new StringBuilder();
        for (char c : st) {
            sb.append(c);
        }
        return sb.toString();
    }

    // 3507. 移除最小数对使数组有序 I (Minimum Pair Removal to Sort Array I)
    public int minimumPairRemoval(int[] nums) {
        int res = 0;
        while (!checkOrder3507(nums)) {
            ++res;
            nums = removeSmallestPairs3507(nums);
        }
        return res;

    }

    private int[] removeSmallestPairs3507(int[] nums) {
        int minPair = Integer.MAX_VALUE;
        int minId = -1;
        for (int i = 1; i < nums.length; ++i) {
            if (nums[i] + nums[i - 1] < minPair) {
                minPair = nums[i] + nums[i - 1];
                minId = i - 1;
            }
        }
        int[] res = new int[nums.length - 1];
        for (int i = 0; i < minId; ++i) {
            res[i] = nums[i];
        }
        res[minId] = minPair;
        for (int i = minId + 2; i < nums.length; ++i) {
            res[i - 1] = nums[i];
        }
        return res;
    }

    private boolean checkOrder3507(int[] nums) {
        for (int i = 1; i < nums.length; ++i) {
            if (nums[i] < nums[i - 1]) {
                return false;
            }
        }
        return true;
    }

    // 3818. 移除前缀使数组严格递增 (Minimum Prefix Removal to Make Array Strictly Increasing)
    public int minimumPrefixLength(int[] nums) {
        for (int i = nums.length - 2; i >= 0; --i) {
            if (nums[i] >= nums[i + 1]) {
                return i + 1;
            }
        }
        return 0;
    }

    // 3819. 非负元素轮替 (Rotate Non Negative Elements)
    public int[] rotateElements(int[] nums, int k) {
        if (k == 0) {
            return nums;
        }
        List<Integer> a = new ArrayList<>();
        for (int x : nums) {
            if (x >= 0) {
                a.add(x);
            }
        }
        if (a.size() == 0) {
            return nums;
        }
        k = k % a.size();
        Collections.rotate(a, -k);
        int j = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] < 0) {
                continue;
            }
            nums[i] = a.get(j++);
        }
        return nums;

    }

    // 3820. 树上的勾股距离节点 (Pythagorean Distance Nodes in a Tree)
    private List<Integer>[] g3820;
    private int n3820;

    public int specialNodes(int n, int[][] edges, int x, int y, int z) {
        this.g3820 = new ArrayList[n];
        this.n3820 = n;
        Arrays.setAll(g3820, k -> new ArrayList<>());
        for (int[] e : edges) {
            g3820[e[0]].add(e[1]);
            g3820[e[1]].add(e[0]);
        }
        int[] dx = cal3820(x);
        int[] dy = cal3820(y);
        int[] dz = cal3820(z);
        int res = 0;
        for (int i = 0; i < n; ++i) {
            long[] dists = new long[] { dx[i], dy[i], dz[i] };
            Arrays.sort(dists);
            if (dists[0] * dists[0] + dists[1] * dists[1] == dists[2] * dists[2]) {
                ++res;
            }
        }
        return res;

    }

    private int[] cal3820(int node) {
        int[] dis = new int[n3820];
        dfs3820(node, -1, 0, dis);
        return dis;
    }

    private void dfs3820(int x, int fa, int d, int[] dis) {
        dis[x] = d;
        for (int y : g3820[x]) {
            if (y != fa) {
                dfs3820(y, x, d + 1, dis);
            }
        }
    }

    // 3821. 二进制中恰好K个1的第N小整数 (Find Nth Smallest Integer With K One Bits)
    private int k3821;

    public long nthSmallest(long n, int k) {
        this.k3821 = k;
        long left = (1L << k) - 1;
        long right = 1L << 50;
        while (left <= right) {
            long mid = left + ((right - left) >> 1L);
            if (check3821(mid) >= n) {
                right = mid - 1L;
            } else {
                left = mid + 1L;
            }
        }
        return right + 1;
    }

    private long[][] memo3821;
    private int n3821;
    private String s3821;

    private long check3821(long x) {
        this.s3821 = Long.toBinaryString(x);
        this.n3821 = s3821.length();
        this.memo3821 = new long[n3821][k3821 + 1];
        for (long[] row : memo3821) {
            Arrays.fill(row, -1L);
        }
        return dfs3821(0, 0, true);
    }

    private long dfs3821(int i, int cnt, boolean isLimit) {
        if (cnt == k3821) {
            return 1L;
        }
        if (i == n3821 || cnt + (n3821 - i) < k3821) {
            return 0L;
        }
        if (!isLimit && memo3821[i][cnt] != -1L) {
            return memo3821[i][cnt];
        }
        long res = 0L;
        int up = isLimit ? (s3821.charAt(i) - '0') : 1;
        for (int d = 0; d <= up && cnt + d <= k3821; ++d) {
            res += dfs3821(i + 1, cnt + d, isLimit && d == up);
        }
        if (!isLimit) {
            memo3821[i][cnt] = res;
        }
        return res;
    }

    // 3821. 二进制中恰好K个1的第N小整数 (Find Nth Smallest Integer With K One Bits)
    public long nthSmallest2(long n, int k) {
        long[][] comb = new long[51][51];
        for (int i = 0; i < 51; ++i) {
            comb[i][0] = 1L;
            for (int j = 1; j <= i; ++j) {
                comb[i][j] = comb[i - 1][j - 1] + comb[i - 1][j];
            }
        }
        long res = 0L;
        for (int i = 49; i >= 0; --i) {
            long c = comb[i][k];
            if (n > c) {
                n -= c;
                res |= 1L << i;
                if (--k == 0) {
                    break;
                }
            }
        }
        return res;
    }

    // 156. 上下翻转二叉树 (Binary Tree Upside Down) --plus
    public TreeNode upsideDownBinaryTree(TreeNode root) {
        if (root == null || root.left == null) {
            return root;
        }
        TreeNode newRoot = upsideDownBinaryTree(root.left);
        root.left.left = root.right;
        root.left.right = root;
        root.left = null;
        root.right = null;
        return newRoot;
    }

    // 255. 验证二叉搜索树的前序遍历序列 (Verify Preorder Sequence in Binary Search Tree) --plus
    public boolean verifyPreorder(int[] preorder) {
        Stack<Integer> st = new Stack<>();
        int lowerBound = Integer.MIN_VALUE;
        for (int x : preorder) {
            if (x < lowerBound) {
                return false;
            }
            while (!st.isEmpty() && x > st.peek()) {
                lowerBound = st.pop();
            }
            st.push(x);
        }
        return true;

    }

    // 291. 单词规律 II (Word Pattern II) --plus
    private Map<Character, String> memoWordPatternMatch291;
    private Set<String> usedWordPatternMatch291;
    private String pattern291;
    private String s291;

    public boolean wordPatternMatch(String pattern, String s) {
        this.memoWordPatternMatch291 = new HashMap<>();
        this.usedWordPatternMatch291 = new HashSet<>();
        this.pattern291 = pattern;
        this.s291 = s;
        return dfs291(0, 0);
    }

    private boolean dfs291(int i, int j) {
        if (i == pattern291.length() || j == s291.length()) {
            return i == pattern291.length() && j == s291.length();
        }
        if (memoWordPatternMatch291.containsKey(pattern291.charAt(i))) {
            if (!s291.substring(j).startsWith(memoWordPatternMatch291.get(pattern291.charAt(i)))) {
                return false;
            }
            return dfs291(i + 1, j + memoWordPatternMatch291.get(pattern291.charAt(i)).length());
        }
        for (int l = 1; j + l <= s291.length(); ++l) {
            String sub = s291.substring(j, j + l);
            if (usedWordPatternMatch291.contains(sub)) {
                continue;
            }
            memoWordPatternMatch291.put(pattern291.charAt(i), sub);
            usedWordPatternMatch291.add(sub);
            if (dfs291(i + 1, j + l)) {
                return true;
            }
            memoWordPatternMatch291.remove(pattern291.charAt(i));
            usedWordPatternMatch291.remove(sub);
        }
        return false;
    }

    // 302. 包含全部黑色像素的最小矩形 (Smallest Rectangle Enclosing Black Pixels) --plus
    public int minArea(char[][] image, int x, int y) {
        int left = y;
        int right = y;
        int top = x;
        int bottom = x;
        int m = image.length;
        int n = image[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (image[i][j] == '1') {
                    left = Math.min(left, j);
                    right = Math.max(right, j);
                    top = Math.min(top, i);
                    bottom = Math.max(bottom, i);
                }
            }
        }
        return (right - left + 1) * (bottom - top + 1);

    }

    // 302. 包含全部黑色像素的最小矩形 (Smallest Rectangle Enclosing Black Pixels) --plus
    private int left302;
    private int right302;
    private int top302;
    private int bottom302;

    public int minArea2(char[][] image, int x, int y) {
        this.left302 = y;
        this.right302 = y;
        this.top302 = x;
        this.bottom302 = x;
        dfs302(image, x, y);
        return (right302 - left302 + 1) * (bottom302 - top302 + 1);
    }

    private void dfs302(char[][] image, int x, int y) {
        if (x < 0 || x >= image.length || y < 0 || y >= image[0].length || image[x][y] == '0') {
            return;
        }
        image[x][y] = '0';
        left302 = Math.min(left302, y);
        right302 = Math.max(right302, y);
        top302 = Math.min(top302, x);
        bottom302 = Math.max(bottom302, x);
        for (int[] dir : new int[][] { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 } }) {
            dfs302(image, x + dir[0], y + dir[1]);
        }
    }

    // 302. 包含全部黑色像素的最小矩形 (Smallest Rectangle Enclosing Black Pixels) --plus
    public int minArea3(char[][] image, int x, int y) {
        int left = binLeft302(image, 0, y);
        int right = binRight302(image, y, image[0].length - 1);
        int top = binTop302(image, 0, x);
        int bottom = binBottom302(image, x, image.length - 1);
        return (right - left + 1) * (bottom - top + 1);
    }

    private int binLeft302(char[][] image, int left, int right) {
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            boolean hasOne = false;
            for (int i = 0; i < image.length; ++i) {
                if (image[i][mid] == '1') {
                    hasOne = true;
                    break;
                }
            }
            if (hasOne) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return right + 1;
    }

    private int binRight302(char[][] image, int left, int right) {
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            boolean hasOne = false;
            for (int i = 0; i < image.length; ++i) {
                if (image[i][mid] == '1') {
                    hasOne = true;
                    break;
                }
            }
            if (hasOne) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left - 1;
    }

    private int binTop302(char[][] image, int left, int right) {
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            boolean hasOne = false;
            for (int j = 0; j < image[0].length; ++j) {
                if (image[mid][j] == '1') {
                    hasOne = true;
                    break;
                }
            }
            if (hasOne) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return right + 1;
    }

    private int binBottom302(char[][] image, int left, int right) {
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            boolean hasOne = false;
            for (int j = 0; j < image[0].length; ++j) {
                if (image[mid][j] == '1') {
                    hasOne = true;
                    break;
                }
            }
            if (hasOne) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left - 1;
    }

    // 3491. 电话号码前缀 (Phone Number Prefix) --plus
    public boolean phonePrefix(String[] numbers) {
        Arrays.sort(numbers, new Comparator<String>() {

            @Override
            public int compare(String o1, String o2) {
                return Integer.compare(o2.length(), o1.length());
            }

        });
        Trie3491 trie = new Trie3491();
        for (String num : numbers) {
            if (trie.insert(num)) {
                return false;
            }
        }
        return true;

    }

    public class Trie3491 {
        private Trie3491[] children;

        public Trie3491() {
            this.children = new Trie3491[10];
        }

        public boolean insert(String s) {
            Trie3491 node = this;
            boolean isPrefix = true;
            for (char c : s.toCharArray()) {
                int idx = c - '0';
                if (node.children[idx] == null) {
                    isPrefix = false;
                    node.children[idx] = new Trie3491();
                }
                node = node.children[idx];
            }
            return isPrefix;
        }
    }

    // 356. 直线镜像 (Line Reflection) --plus
    public boolean isReflected(int[][] points) {
        Map<Integer, Set<Integer>> map = new HashMap<>();
        for (int[] p : points) {
            map.computeIfAbsent(p[1], o -> new HashSet<>()).add(p[0]);
        }
        int mid_2 = Integer.MAX_VALUE;
        for (Set<Integer> s : map.values()) {
            int cur = check356(s);
            if (cur == Integer.MAX_VALUE) {
                return false;
            }
            if (mid_2 != Integer.MAX_VALUE && mid_2 != cur) {
                return false;
            }
            mid_2 = cur;
        }
        return true;

    }

    private int check356(Set<Integer> s) {
        List<Integer> list = new ArrayList<>(s);
        Collections.sort(list);
        int left = 0;
        int right = list.size() - 1;
        int mid_2 = list.get(left) + list.get(right);
        while (left < right) {
            if (list.get(left) + list.get(right) != mid_2) {
                return Integer.MAX_VALUE;
            }
            ++left;
            --right;
        }
        if (left == right) {
            return list.get(left) * 2 == mid_2 ? mid_2 : Integer.MAX_VALUE;
        }
        return mid_2;
    }

    // 1644. 二叉树的最近公共祖先 II (Lowest Common Ancestor of a Binary Tree II) --plus
    private TreeNode res1644;

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        dfs1644(root, p, q);
        return res1644;
    }

    private TreeNode dfs1644(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {
            return null;
        }
        if (root == p || root == q) {
            if (check1644(root.left, p, q) || check1644(root.right, p, q)) {
                res1644 = root;
                return root;
            }
            return root;
        }
        TreeNode left = dfs1644(root.left, p, q);
        TreeNode right = dfs1644(root.right, p, q);
        if (res1644 != null) {
            return res1644;
        }
        if (left != null && right != null) {
            res1644 = root;
            return res1644;
        }
        if (left != null) {
            return left;
        }
        return right;
    }

    private boolean check1644(TreeNode node, TreeNode p, TreeNode q) {
        if (node == null) {
            return false;
        }
        if (node == p || node == q) {
            return true;
        }
        return check1644(node.left, p, q) || check1644(node.right, p, q);
    }

    // 510. 二叉搜索树中的中序后继 II (Inorder Successor in BST II) --plus
    public Node inorderSuccessor(Node node) {
        if (node.right != null) {
            node = node.right;
            while (node.left != null) {
                node = node.left;
            }
            return node;
        }
        while (node.parent != null) {
            if (node.parent.left == node) {
                break;
            }
            node = node.parent;
        }
        return node.parent;
    }

    // 545. 二叉树的边界 (Boundary of Binary Tree) --plus
    public List<Integer> boundaryOfBinaryTree(TreeNode root) {
        // 左边界
        List<Integer> left = cal545(root.left, true);
        // 右边界
        List<Integer> right = cal545(root.right, false);
        // 叶子节点
        List<Integer> leaves = new ArrayList<>();
        dfsLeaves545(root, root, leaves);
        List<Integer> res = new ArrayList<>();
        res.add(root.val);
        res.addAll(left);
        res.addAll(leaves);
        Collections.reverse(right);
        res.addAll(right);
        return res;
    }

    private void dfsLeaves545(TreeNode root, TreeNode node, List<Integer> leaves) {
        if (node == null) {
            return;
        }
        if (node.left == null && node.right == null && node != root) {
            leaves.add(node.val);
            return;
        }
        dfsLeaves545(root, node.left, leaves);
        dfsLeaves545(root, node.right, leaves);
    }

    private List<Integer> cal545(TreeNode node, boolean isLeft) {
        List<Integer> res = new ArrayList<>();
        while (node != null) {
            // node 是叶子
            if (node.left == null && node.right == null) {
                break;
            }
            res.add(node.val);
            node = isLeft && node.left != null || !isLeft && node.right == null ? node.left : node.right;
        }
        return res;
    }

    // 800. 相似 RGB 颜色 (Similar RGB Color) --plus
    public String similarRGB(String color) {
        StringBuilder res = new StringBuilder();
        for (int i = 1; i < color.length() - 1; i += 2) {
            res.append(check800(color.substring(i, i + 2)));
        }
        return "#" + res.toString();

    }

    private String check800(String s) {
        int diff = Integer.MAX_VALUE;
        int val = Integer.parseInt(s, 16);
        int res = 0;
        int first = Integer.parseInt(s.substring(0, 1), 16);
        for (int i = Math.max(0, first - 1); i <= Math.min(15, first + 1); ++i) {
            int candidate = i * 16 + i;
            if (Math.abs(candidate - val) < diff) {
                diff = Math.abs(candidate - val);
                res = candidate;

            }
        }
        return String.format("%02x", res);
    }

    // 3817. 数字字符串中的好索引 (Good Indices in a Digit String) --plus
    public List<Integer> goodIndices(String s) {
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < s.length(); ++i) {
            int len = String.valueOf(i).length();
            // 最后一位不相等
            if (s.charAt(i) - '0' != i % 10) {
                continue;
            }
            if (Integer.parseInt(s.substring(Math.max(0, i - len + 1), i + 1)) == i) {
                res.add(i);
            }
        }
        return res;

    }

    // 3807. 修复边以遍历图的最小成本 (Minimum Cost to Repair Edges to Traverse a Graph) --plus
    public int minCost(int n, int[][] edges, int k) {
        List<int[]>[] g = new ArrayList[n];
        Arrays.setAll(g, o -> new ArrayList<>());
        int maxW = 0;
        for (int[] e : edges) {
            g[e[0]].add(new int[] { e[1], e[2] });
            g[e[1]].add(new int[] { e[0], e[2] });
            maxW = Math.max(maxW, e[2]);
        }
        if (!check3807(g, k, n, Integer.MAX_VALUE)) {
            return -1;
        }
        int left = 0;
        int right = maxW;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (check3807(g, k, n, mid)) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return right + 1;

    }

    private boolean check3807(List<int[]>[] g, int k, int n, int limit) {
        boolean[] vis = new boolean[n];
        vis[0] = true;
        Queue<int[]> q = new ArrayDeque<>();
        q.add(new int[] { 0, 0 });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int d = cur[1];
            if (x == n - 1) {
                return true;
            }
            for (int[] neighbor : g[x]) {
                int y = neighbor[0];
                int w = neighbor[1];
                if (!vis[y] && w <= limit && d + 1 <= k) {
                    vis[y] = true;
                    q.add(new int[] { y, d + 1 });
                }
            }
        }
        return false;
    }

    // 1794. 统计距离最小的子串对个数 (Count Pairs of Equal Substrings With Minimum Difference)
    // --plus
    public int countQuadruples(String firstString, String secondString) {
        int[] left = new int[26];
        Arrays.fill(left, -1);
        for (int i = 0; i < firstString.length(); ++i) {
            int id = firstString.charAt(i) - 'a';
            if (left[id] == -1) {
                left[id] = i;
            }
        }
        int[] right = new int[26];
        Arrays.fill(right, -1);
        for (int i = secondString.length() - 1; i >= 0; --i) {
            int id = secondString.charAt(i) - 'a';
            if (right[id] == -1) {
                right[id] = i;
            }
        }
        int minLen = Integer.MAX_VALUE;
        for (int i = 0; i < 26; ++i) {
            if (left[i] != -1 && right[i] != -1) {
                minLen = Math.min(minLen, left[i] - right[i]);
            }
        }
        int res = 0;
        for (int i = 0; i < firstString.length(); ++i) {
            int id = firstString.charAt(i) - 'a';
            if (right[id] != -1 && i - right[id] == minLen) {
                ++res;
            }
        }
        return res;

    }

    // 1788. 最大化花园的美观度 (Maximize the Beauty of the Garden) --plus
    public int maximumBeauty(int[] flowers) {
        int n = flowers.length;
        Map<Integer, Integer> left = new HashMap<>();
        Map<Integer, Integer> right = new HashMap<>();
        int[] pre = new int[n + 1];
        for (int i = 0; i < n; ++i) {
            pre[i + 1] = pre[i] + Math.max(0, flowers[i]);
            left.putIfAbsent(flowers[i], i);
            right.put(flowers[i], i);
        }
        int res = Integer.MIN_VALUE;
        for (int i = 0; i < n; ++i) {
            int l = left.get(flowers[i]);
            int r = right.get(flowers[i]);
            if (l == r) {
                continue;
            }
            int val = pre[r + 1] - pre[l] + Math.min(0, flowers[l]) * 2;
            res = Math.max(res, val);
        }
        return res;
    }

    // 3792. 递增乘积块之和 (Sum of Increasing Product Blocks) --plus
    public int sumOfBlocks(int n) {
        final int MOD = (int) (1e9 + 7);
        int res = 0;
        int cnt = 0;
        int x = 1;
        for (int i = 1; i <= n; ++i) {
            ++cnt;
            int val = 1;
            for (int j = x; j < x + cnt; ++j) {
                val = (int) ((long) val * j % MOD);
            }
            res = (res + val) % MOD;
            x += cnt;
        }
        return res;

    }

    // 3787. 查找树的直径端点 (Find Diameter Endpoints of a Tree) --plus
    /**
     * 1、找到树中一条直径的方法：对任意一个点 DFS 找到最远点，然后对这个点进行第二次相同的 DFS 找另一个最远点，得到的路径就是直径
     * 2、找树中所有直径端点的方法：稍作修改，对任意一个点 DFS 找到的所有最远点加入集合；从其中一个最远点进行第二次 DFS
     * 找到的所有最远点加入集合。可以证明，两个集合的并集即为答案。
     */
    private int maxDepth3787;
    private int node3787;

    public String findSpecialNodes(int n, int[][] edges) {
        List<Integer>[] g = new ArrayList[n];
        Arrays.setAll(g, o -> new ArrayList<>());
        for (int[] e : edges) {
            g[e[0]].add(e[1]);
            g[e[1]].add(e[0]);
        }
        Set<Integer> s1 = new HashSet<>();
        dfs3787(0, -1, g, 0, s1);
        Set<Integer> s2 = new HashSet<>();
        dfs3787(node3787, -1, g, 0, s2);
        s1.addAll(s2);
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < n; ++i) {
            if (s1.contains(i)) {
                res.append('1');
            } else {
                res.append('0');
            }
        }
        return res.toString();

    }

    private void dfs3787(int x, int fa, List<Integer>[] g, int d, Set<Integer> s) {
        if (d > maxDepth3787) {
            maxDepth3787 = d;
            s.clear();
            s.add(x);
            node3787 = x;
        } else if (d == maxDepth3787) {
            s.add(x);
        }
        for (int y : g[x]) {
            if (y != fa) {
                dfs3787(y, x, g, d + 1, s);
            }
        }
    }

    // 2036. 最大交替子数组和 (Maximum Alternating Subarray Sum) --plus
    private int[] nums2036;
    private int n2036;
    private long[][] memo2036;

    public long maximumAlternatingSubarraySum(int[] nums) {
        this.n2036 = nums.length;
        this.nums2036 = nums;
        this.memo2036 = new long[n2036][2];
        for (long[] row : memo2036) {
            Arrays.fill(row, Long.MIN_VALUE);
        }
        long res = Long.MIN_VALUE;
        for (int i = 0; i < n2036; ++i) {
            res = Math.max(res, dfs2036(i + 1, 1) + nums[i]);
        }
        return res;

    }

    private long dfs2036(int i, int j) {
        if (i == n2036) {
            return 0L;
        }
        if (memo2036[i][j] != Long.MIN_VALUE) {
            return memo2036[i][j];
        }
        return memo2036[i][j] = Math.max(0L, dfs2036(i + 1, j ^ 1) + (1 - j * 2) * nums2036[i]);
    }

    // 3822. 设计订单管理系统 (Design Order Management System) --plus
    class OrderManagementSystem {
        private record Order(String orderType, int orderPrice) {
        }

        private Map<Integer, Order> idToOrder;
        private Map<String, Map<Integer, Set<Integer>>> typeToPriceIds;

        public OrderManagementSystem() {
            this.idToOrder = new HashMap<>();
            this.typeToPriceIds = new HashMap<>();

        }

        public void addOrder(int orderId, String orderType, int price) {
            idToOrder.put(orderId, new Order(orderType, price));
            typeToPriceIds.computeIfAbsent(orderType, o -> new HashMap<>()).computeIfAbsent(price, o -> new HashSet<>())
                    .add(orderId);
        }

        public void modifyOrder(int orderId, int newPrice) {
            Order oldOrder = idToOrder.remove(orderId);
            String orderType = oldOrder.orderType;
            int oldPrice = oldOrder.orderPrice;
            idToOrder.put(orderId, new Order(orderType, newPrice));
            typeToPriceIds.get(orderType).get(oldPrice).remove(orderId);
            typeToPriceIds.get(orderType).computeIfAbsent(newPrice, o -> new HashSet<>()).add(orderId);
        }

        public void cancelOrder(int orderId) {
            Order oldOrder = idToOrder.remove(orderId);
            String orderType = oldOrder.orderType;
            int oldPrice = oldOrder.orderPrice;
            typeToPriceIds.get(orderType).get(oldPrice).remove(orderId);
        }

        public int[] getOrdersAtPrice(String orderType, int price) {
            Set<Integer> ids = typeToPriceIds.getOrDefault(orderType, new HashMap<>()).getOrDefault(price,
                    new HashSet<>());
            int[] res = new int[ids.size()];
            int i = 0;
            for (int id : ids) {
                res[i++] = id;
            }
            return res;
        }
    }

    // 3167. 字符串的更好压缩 (Better Compression of String) --plus
    public String betterCompression(String compressed) {
        int[] cnts = new int[26];
        int i = 0;
        while (i < compressed.length()) {
            char c = compressed.charAt(i++);
            int cnt = 0;
            while (i < compressed.length() && Character.isDigit(compressed.charAt(i))) {
                cnt = cnt * 10 + (compressed.charAt(i++) - '0');
            }
            cnts[c - 'a'] += cnt;
        }
        StringBuilder res = new StringBuilder();
        for (int j = 0; j < 26; ++j) {
            if (cnts[j] > 0) {
                res.append((char) (j + 'a'));
                res.append(cnts[j]);
            }
        }
        return res.toString();

    }

    // 3173. 相邻元素的按位或 (Bitwise OR of Adjacent Elements) --plus
    public int[] orArray(int[] nums) {
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < nums.length - 1; ++i) {
            res.add(nums[i] | nums[i + 1]);
        }
        return res.stream().mapToInt(Integer::intValue).toArray();

    }

    // 3062. 链表游戏的获胜者 (Winner of the Linked List Game) --plus
    public String gameResult(ListNode head) {
        int[] cnt = new int[2];
        while (head != null) {
            int even = head.val;
            int odd = head.next.val;
            if (even != odd) {
                ++cnt[even > odd ? 0 : 1];
            }
            head = head.next.next;
        }
        if (cnt[0] > cnt[1]) {
            return "Even";
        }
        if (cnt[0] < cnt[1]) {
            return "Odd";
        }
        return "Tie";

    }

    // 2743. 计算没有重复字符的子字符串数量 (Count Substrings Without Repeating Characters) --plus
    public int numberOfSpecialSubstrings(String s) {
        int[] cnts = new int[26];
        int res = 0;
        int j = 0;
        for (int i = 0; i < s.length(); ++i) {
            ++cnts[s.charAt(i) - 'a'];
            while (cnts[s.charAt(i) - 'a'] > 1) {
                --cnts[s.charAt(j++) - 'a'];
            }
            res += i - j + 1;
        }
        return res;

    }

    // 3773. 最大等长连续字符组 (Maximum Number of Equal Length Runs) --plus
    public int maxSameLengthRuns(String s) {
        int res = 0;
        int cnt = 0;
        Map<Integer, Integer> cnts = new HashMap<>();
        for (int i = 0; i < s.length(); ++i) {
            ++cnt;
            if (i == s.length() - 1 || s.charAt(i) != s.charAt(i + 1)) {
                cnts.merge(cnt, 1, Integer::sum);
                res = Math.max(res, cnts.get(cnt));
                cnt = 0;
            }
        }
        return res;
    }

    // 3063. 链表频率 (Linked List Frequency) --plus
    public ListNode frequenciesOfElements(ListNode head) {
        Map<Integer, Integer> freq = new HashMap<>();
        while (head != null) {
            freq.merge(head.val, 1, Integer::sum);
            head = head.next;
        }
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        for (int f : freq.values()) {
            cur.next = new ListNode(f);
            cur = cur.next;
        }
        return dummy.next;

    }

    // 3247. 奇数和子序列的数量 (Number of Subsequences with Odd Sum) --plus
    private int[][] memo3247;
    private int[] nums3247;

    public int subsequenceCount(int[] nums) {
        int n = nums.length;
        this.memo3247 = new int[n][2];
        this.nums3247 = nums;
        for (int[] row : memo3247) {
            Arrays.fill(row, -1);
        }
        return dfs3247(n - 1, 0);

    }

    private int dfs3247(int i, int j) {
        if (i < 0) {
            return j;
        }
        if (memo3247[i][j] != -1) {
            return memo3247[i][j];
        }
        final int MOD = (int) (1e9 + 7);
        return memo3247[i][j] = (dfs3247(i - 1, j) + dfs3247(i - 1, (j + nums3247[i]) & 1)) % MOD;
    }

    // 3247. 奇数和子序列的数量 (Number of Subsequences with Odd Sum) --plus
    public int subsequenceCount2(int[] nums) {
        int[] cnts = new int[2];
        for (int x : nums) {
            ++cnts[x & 1];
        }
        final int MOD = (int) (1e9 + 7);
        return cnts[1] == 0 ? 0 : (int) ((long) pow3247(2, cnts[0]) * pow3247(2, cnts[1] - 1) % MOD);
    }

    private int pow3247(int a, int b) {
        if (b == 0) {
            return 1;
        }
        final int MOD = (int) (1e9 + 7);
        int res = pow3247(a, b >> 1);
        res = (int) ((long) res * res % MOD);
        return (b & 1) == 1 ? (int) ((long) res * a % MOD) : res;
    }

    // 3763. 带阈值约束的最大总和 (Maximum Total Sum with Threshold Constraints) --plus
    public long maxSum(int[] nums, int[] threshold) {
        int n = nums.length;
        int[][] a = new int[n][2];
        for (int i = 0; i < n; ++i) {
            a[i][0] = nums[i];
            a[i][1] = threshold[i];
        }
        Arrays.sort(a, (o1, o2) -> Integer.compare(o1[1], o2[1]));
        long res = 0L;
        for (int step = 0; step < n; ++step) {
            if (a[step][1] > step + 1) {
                break;
            }
            res += a[step][0];
        }
        return res;

    }

    // 3758. 将数字词转换为数字 (Convert Number Words to Digits) --plus
    public String convertNumber(String s) {
        Map<String, Character> map = new HashMap<>() {
            {
                put("zero", '0');
                put("one", '1');
                put("two", '2');
                put("three", '3');
                put("four", '4');
                put("five", '5');
                put("six", '6');
                put("seven", '7');
                put("eight", '8');
                put("nine", '9');
            }
        };
        int i = 0;
        StringBuilder res = new StringBuilder();
        search: while (i < s.length()) {
            for (String key : map.keySet()) {
                if (s.startsWith(key, i)) {
                    res.append(map.get(key));
                    i += key.length();
                    continue search;
                }
            }
            ++i;
        }
        return res.toString();

    }

    // 3119. 最大数量的可修复坑洼 (Maximum Number of Potholes That Can Be Fixed) --plus
    public int maxPotholes(String road, int budget) {
        int n = road.length();
        int i = 0;
        Map<Integer, Integer> cnts = new HashMap<>();
        while (i < n) {
            if (road.charAt(i) == 'x') {
                int j = i;
                while (j < n && road.charAt(j) == 'x') {
                    ++j;
                }
                cnts.merge(j - i, 1, Integer::sum);
                i = j;
            } else {
                ++i;
            }
        }
        int res = 0;
        List<int[]> list = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry : cnts.entrySet()) {
            list.add(new int[] { entry.getKey(), entry.getValue() });
        }
        Collections.sort(list, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o2[0], o1[0]);
            }

        });
        for (int[] p : list) {
            int l = p[0];
            int parts = p[1];
            int completeParts = budget / (l + 1);
            res += l * Math.min(parts, completeParts);
            budget -= (l + 1) * Math.min(parts, completeParts);
            parts -= Math.min(parts, completeParts);
            if (parts > 0 && budget > 0) {
                res += budget - 1;
                break;
            }
        }
        return res;

    }

    // 3183. 达到总和的方法数量 (The Number of Ways to Make the Sum) --plus
    private int[][][] memo3183;
    private int[] a3183 = { 4, 2, 6 };
    private int n3183;

    public int numberOfWays(int n) {
        this.n3183 = n;
        this.memo3183 = new int[3][n][3];
        for (int[][] matrix : memo3183) {
            for (int[] row : matrix) {
                Arrays.fill(row, -1);
            }
        }
        return dfs3183(0, 0, 0);

    }

    private int dfs3183(int i, int j, int k) {
        if (j == n3183) {
            return 1;
        }
        if (j > n3183) {
            return 0;
        }
        if (i == 3) {
            return 1;
        }
        if (memo3183[i][j][k] != -1) {
            return memo3183[i][j][k];
        }
        int res = dfs3183(i + 1, j, k);
        final int MOD = (int) (1e9 + 7);
        if (i == 0) {
            if (k < 2) {
                res += dfs3183(i, j + a3183[i], k + 1);
            }
        } else {
            res += dfs3183(i, j + a3183[i], k);
        }
        return memo3183[i][j][k] = res % MOD;
    }

    // 3183. 达到总和的方法数量 (The Number of Ways to Make the Sum) --plus
    public int numberOfWays2(int n) {
        final int MOD = (int) (1e9 + 7);
        int[] f = new int[n + 1];
        f[0] = 1;
        for (int c : new int[] { 1, 2, 6 }) {
            for (int i = c; i <= n; ++i) {
                f[i] += f[i - c];
                f[i] %= MOD;
            }
        }
        int res = f[n];
        if (n >= 4) {
            res += f[n - 4];
            res %= MOD;
        }
        if (n >= 8) {
            res += f[n - 8];
            res %= MOD;
        }
        return res;
    }

    // 3778. 排除一个最大权重边的最小距离 (Minimum Distance Excluding One Maximum Weighted Edge)
    // --plus
    public long minCostExcludingMax(int n, int[][] edges) {
        List<int[]>[] g = new ArrayList[n];
        Arrays.setAll(g, o -> new ArrayList<>());
        for (int[] e : edges) {
            g[e[0]].add(new int[] { e[1], e[2] });
            g[e[1]].add(new int[] { e[0], e[2] });
        }
        long[][] dis = new long[n][2];
        for (long[] d : dis) {
            Arrays.fill(d, Long.MAX_VALUE);
        }
        dis[0][0] = 0L;
        Queue<long[]> q = new PriorityQueue<>(new Comparator<long[]>() {

            @Override
            public int compare(long[] o1, long[] o2) {
                if (o1[1] != o2[1]) {
                    return Long.compare(o2[1], o1[1]);
                }
                return Long.compare(o1[2], o2[2]);
            }

        });
        // (node, 0:该路径未去掉一个边；1:该路径已经去掉一个边, dis)
        q.offer(new long[] { 0L, 0L, 0L });
        while (!q.isEmpty()) {
            long[] cur = q.poll();
            int x = (int) cur[0];
            int isDelete = (int) cur[1];
            long d = cur[2];
            if (d > dis[x][isDelete]) {
                continue;
            }
            for (int[] neighbor : g[x]) {
                int y = neighbor[0];
                int w = neighbor[1];
                if (d + w < dis[y][isDelete]) {
                    dis[y][isDelete] = d + w;
                    q.offer(new long[] { y, isDelete, d + w });
                }
                if (isDelete == 0 && d < dis[y][1]) {
                    dis[y][1] = d;
                    q.offer(new long[] { y, 1, d });
                }
            }
        }
        return dis[n - 1][1];

    }

    // 2664. 巡逻的骑士 (The Knight’s Tour) --plus
    private int m2664;
    private int n2664;
    private int[][] res2664;
    private int[][] dirs2664 = { { 2, 1 }, { 1, 2 }, { -1, 2 }, { -2, 1 }, { -2, -1 }, { -1, -2 }, { 1, -2 },
            { 2, -1 } };

    public int[][] tourOfKnight(int m, int n, int r, int c) {
        this.m2664 = m;
        this.n2664 = n;
        this.res2664 = new int[m][n];
        for (int[] row : res2664) {
            Arrays.fill(row, -1);
        }
        dfs2664(r, c, 0);
        return res2664;
    }

    private boolean dfs2664(int x, int y, int step) {
        res2664[x][y] = step;
        if (step == m2664 * n2664 - 1) {
            return true;
        }
        for (int[] d : dirs2664) {
            int nx = x + d[0];
            int ny = y + d[1];
            if (nx >= 0 && nx < m2664 && ny >= 0 && ny < n2664 && res2664[nx][ny] == -1 && dfs2664(nx, ny, step + 1)) {
                return true;
            }
        }
        res2664[x][y] = -1;
        return false;
    }

    // 2852. 所有单元格的远离程度之和 (Sum of Remoteness of All Cells) --plus
    public long sumRemoteness(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        long sum = 0L;
        Union2852 union = new Union2852(m * n);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                sum += Math.max(0, grid[i][j]);
                if (i - 1 >= 0 && grid[i][j] > 0 && grid[i - 1][j] > 0) {
                    union.union(i * n + j, (i - 1) * n + j);
                }
                if (j - 1 >= 0 && grid[i][j] > 0 && grid[i][j - 1] > 0) {
                    union.union(i * n + j, i * n + j - 1);
                }
            }
        }
        Map<Integer, Long> rootToSum = new HashMap<>();
        Map<Integer, Integer> rootToCnt = new HashMap<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] > 0) {
                    int root = union.getRoot(i * n + j);
                    rootToSum.merge(root, (long) grid[i][j], Long::sum);
                    rootToCnt.merge(root, 1, Integer::sum);
                }
            }
        }
        long res = 0L;
        for (Map.Entry<Integer, Integer> entry : rootToCnt.entrySet()) {
            int root = entry.getKey();
            int size = entry.getValue();
            long curSum = rootToSum.get(root);
            res += (long) (sum - curSum) * size;
        }
        return res;

    }

    public class Union2852 {
        private int[] parent;
        private int[] rank;

        public Union2852(int n) {
            this.parent = new int[n];
            this.rank = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
                rank[i] = 1;
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
                ++rank[root1];
            }
        }
    }

    // 2847. 给定数字乘积的最小数字 (Smallest Number With Given Digit Product) --plus
    public String smallestNumber(long n) {
        if (n == 1L) {
            return "1";
        }
        int[] cnt = new int[10];
        for (int i = 9; i > 1; --i) {
            while (n % i == 0) {
                n /= i;
                ++cnt[i];
            }
        }
        if (n != 1L) {
            return "-1";
        }
        StringBuilder res = new StringBuilder();
        for (int i = 2; i <= 9; ++i) {
            while (cnt[i]-- > 0) {
                res.append(i);
            }
        }
        return res.toString();
    }

    // 3481. 应用替换 (Apply Substitutions) --plus
    public String applySubstitutions(List<List<String>> replacements, String text) {
        String[] map = new String[26];
        for (List<String> s : replacements) {
            int i = s.get(0).charAt(0) - 'A';
            map[i] = s.get(1);
        }
        return dfs(text, map);
    }

    private String dfs(String s, String[] map) {
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) != '%') {
                res.append(s.charAt(i));
                continue;
            }
            res.append(dfs(map[s.charAt(i + 1) - 'A'], map));
            i += 2;
        }
        return res.toString();
    }

    // 3696. 不同单词间的最大距离 I (Maximum Distance Between Unequal Words in Array I) --plus
    public int maxDistance(String[] words) {
        Map<String, int[]> map = new HashMap<>();
        for (int i = 0; i < words.length; ++i) {
            int[] v = map.getOrDefault(words[i], new int[] { Integer.MAX_VALUE, Integer.MIN_VALUE });
            v[0] = Math.min(v[0], i);
            v[1] = Math.max(v[1], i);
            map.put(words[i], v);
        }
        int res = 0;
        int min_id = Integer.MAX_VALUE;
        int max_id = Integer.MIN_VALUE;
        for (int[] v : map.values()) {
            int cur_min_id = v[0];
            int cur_max_id = v[1];
            if (min_id != Integer.MAX_VALUE) {
                res = Math.max(res, Math.abs(min_id - cur_max_id) + 1);
                res = Math.max(res, Math.abs(max_id - cur_min_id) + 1);
            }
            min_id = Math.min(min_id, cur_min_id);
            max_id = Math.max(max_id, cur_max_id);
        }
        return res;

    }
}
