import java.net.Inet4Address;
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

    // 100850. 反转后字典序最小的字符串
    public String lexSmallest(String s) {
        int n = s.length();
        String res = "";
        for (int i = 0; i < n; ++i) {
            String cur = reversed100850(s.substring(0, i)) + s.substring(i);
            if (res.equals("") || cur.compareTo(res) < 0) {
                res = cur;
            }
            cur = s.substring(0, i) + reversed100850(s.substring(i));
            if (res.equals("") || cur.compareTo(res) < 0) {
                res = cur;
            }
        }
        return res;

    }

    private String reversed100850(String s) {
        StringBuilder sb = new StringBuilder(s);
        return sb.reverse().toString();
    }

    public String maxSumOfSquares(int num, int sum) {
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < num; ++i) {
            for (int d = 9; d >= (i == 0 ? 1 : 0); --d) {
                if (0 <= sum - d && sum - d <= (num - i - 1) * 9) {
                    res.append((char) (d + '0'));
                    sum -= d;
                    break;
                }
            }
            if (sum == 0) {
                for (int j = i + 1; j < num; ++j) {
                    res.append('0');
                }
                break;
            }
        }
        return res.toString();

    }

    private int n;
    private int[] nums1;
    private int[] nums2;
    private long[][] memo;

    public long minOperations(int[] nums1, int[] nums2) {
        this.nums1 = nums1;
        this.nums2 = nums2;
        this.n = nums1.length;
        this.memo = new long[n][2];
        for (long[] row : memo) {
            Arrays.fill(row, -1L);
        }
        return dfs(0, 0);

    }

    private long dfs(int i, int j) {
        if (i == n) {
            return j == 0 ? Long.MAX_VALUE / 2 : 0L;
        }
        if (memo[i][j] != -1L) {
            return memo[i][j];
        }
        long res = dfs(i + 1, j) + Math.abs(nums1[i] - nums2[i]);
        if (j == 0) {
            int max = Math.max(Math.max(nums1[i], nums2[i]), nums2[n]);
            int min = Math.min(Math.min(nums1[i], nums2[i]), nums2[n]);
            res = Math.min(res, dfs(i + 1, 1) + max - min + 1);
        }
        return memo[i][j] = res;
    }
}
