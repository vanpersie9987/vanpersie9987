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
import java.util.stream.IntStream;

@SuppressWarnings("unchecked")
public class Leetcode_11 {
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
    }

    class Node2 {
        public int val;
        public Node2 prev;
        public Node2 next;
    }

    // 3919. 在下标间移动的最小代价 (Minimum Cost to Move Between Indices)
    public int[] minCost(int[] nums, int[][] queries) {
        int n = nums.length;
        int[] sumL = new int[n]; // sumL[i] 等于从 i 移动到 0 的代价和
        int[] sumR = new int[n]; // sumR[i] 等于从 0 移动到 i 的代价和
        for (int i = 1, cost; i < n; i++) {
            // 往左走 i -> i-1
            if (i < n - 1 && nums[i] - nums[i - 1] > nums[i + 1] - nums[i]) { // closest(i) = i+1
                cost = nums[i] - nums[i - 1]; // 只能用方式一往左走
            } else {
                cost = 1;
            }
            sumL[i] = sumL[i - 1] + cost;

            // 往右走 i-1 -> i
            if (i > 1 && nums[i - 1] - nums[i - 2] <= nums[i] - nums[i - 1]) { // closest(i-1) = i-2
                cost = nums[i] - nums[i - 1]; // 只能用方式一往右走
            } else {
                cost = 1;
            }
            sumR[i] = sumR[i - 1] + cost;
        }

        int[] ans = new int[queries.length];
        for (int i = 0; i < queries.length; i++) {
            int l = queries[i][0];
            int r = queries[i][1];
            if (l < r) {
                // cost(0 -> r) - cost(0 -> l) = cost(l -> r)
                ans[i] = sumR[r] - sumR[l];
            } else {
                // cost(l -> 0) - cost(r -> 0) = cost(l -> r)
                ans[i] = sumL[l] - sumL[r];
            }
        }
        return ans;

    }

    // 3921. 分数验证器 (Score Validator)
    public int[] scoreValidator(String[] events) {
        int score = 0;
        int counter = 0;
        for (String e : events) {
            if ("W".equals(e)) {
                if (++counter == 10) {
                    break;
                }
            } else {
                if (e.length() == 1) {
                    score += Integer.parseInt(e);
                } else {
                    ++score;
                }
            }
        }
        return new int[] { score, counter };

    }

    // 3922. 使二进制字符串连贯的最少翻转次数 (Minimum Flips to Make Binary String Coherent)
    public int minFlips(String s) {
        int cnt1 = 0;
        for (char c : s.toCharArray()) {
            cnt1 += c - '0';
        }
        int n = s.length();
        // 都是1或都是0
        int res = Math.min(cnt1, n - cnt1);
        // 既有1也有0的情况，若首尾都是1，则吧中间的都变成0
        // 否则需要整个字符串只保留1个1
        return Math.min(res, Math.max(0, cnt1 - 1 - (s.charAt(0) == '1' && s.charAt(n - 1) == '1' ? 1 : 0)));
    }

    // 3925. 连接逆序数组 (Concatenate Array With Reverse)
    public int[] concatWithReverse(int[] nums) {
        int n = nums.length;
        int[] res = new int[n * 2];
        for (int i = 0; i < n; ++i) {
            res[i] = nums[i];
            res[i + n] = nums[n - i - 1];
        }
        return res;

    }

    // 3926. 有效单词计数 (Count Valid Word Occurrences)
    public int[] countWordOccurrences(String[] chunks, String[] queries) {
        char[] s = String.join("", chunks).toCharArray();
        int n = s.length;
        Map<String, Integer> cnt = new HashMap<>();

        for (int i = 0; i < n; i++) {
            if (s[i] == ' ' || s[i] == '-') {
                continue;
            }
            int start = i;
            // 遇到 ' ' 或者 "--" 或者 "- " 时，跳出循环
            while (i < n && s[i] != ' ' && (s[i] != '-' || i < n - 1 && s[i + 1] != '-' && s[i + 1] != ' ')) {
                i++;
            }
            String word = new String(s, start, i - start);
            cnt.merge(word, 1, Integer::sum); // cnt[word]++
        }

        int[] ans = new int[queries.length];
        for (int i = 0; i < queries.length; i++) {
            ans[i] = cnt.getOrDefault(queries[i], 0);
        }
        return ans;
    }

    // 3927. 可整除替换后的数组最小元素和 (Minimize Array Sum Using Divisible Replacements)
    public long minArraySum(int[] nums) {
        long res = 0L;
        Map<Integer, Integer> cnts = new HashMap<>();
        for (int x : nums) {
            if (x == 1) {
                return nums.length;
            }
            cnts.merge(x, 1, Integer::sum);
        }
        int MX = (int) (1e5 + 1);
        boolean[] isPrime = new boolean[MX];
        Arrays.fill(isPrime, true);
        for (int i = 2; i < MX; ++i) {
            if (isPrime[i]) {
                for (long j = (long) i * i; j < MX; j += i) {
                    isPrime[(int) j] = false;
                }
            }
        }
        search: for (Map.Entry<Integer, Integer> entry : cnts.entrySet()) {
            int x = entry.getKey();
            int c = entry.getValue();
            if (isPrime[x]) {
                res += (long) x * c;
            } else {
                int min = Integer.MAX_VALUE;
                for (int i = 2; i <= Math.sqrt(x); ++i) {
                    if (x % i == 0) {
                        if (cnts.containsKey(i)) {
                            res += (long) i * c;
                            continue search;
                        }
                        if (cnts.containsKey(x / i)) {
                            min = Math.min(min, x / i);
                        }
                    }
                }
                if (min == Integer.MAX_VALUE) {
                    res += (long) x * c;
                } else {
                    res += (long) min * c;
                }
            }
        }
        return res;

    }

    // 3924. 有限重边的最小阈值路径 (Minimum Threshold Path With Limited Heavy Edges)
    public int minimumThreshold(int n, int[][] edges, int source, int target, int k) {
        if (source == target) {
            return 0;
        }
        List<int[]>[] g = new ArrayList[n];
        int mx = 0;
        Arrays.setAll(g, o -> new ArrayList<>());
        for (int[] e : edges) {
            g[e[0]].add(new int[] { e[1], e[2] });
            g[e[1]].add(new int[] { e[0], e[2] });
            mx = Math.max(mx, e[2]);
        }
        int left = 0;
        int right = mx;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (check3924(mid, n, source, target, k, g)) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return right + 1 > mx ? -1 : right + 1;
    }

    private boolean check3924(int limit, int n, int source, int target, int k, List<int[]>[] g) {
        int[] dis = new int[n];
        Arrays.fill(dis, k + 1);
        dis[source] = 0;
        Queue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });
        q.offer(new int[] { 0, source });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int d = cur[0];
            int x = cur[1];
            if (x == target) {
                return true;
            }
            for (int[] nxt : g[x]) {
                int y = nxt[0];
                int w = nxt[1];
                int dx = w > limit ? 1 : 0;
                if (d + dx < dis[y]) {
                    dis[y] = d + dx;
                    q.offer(new int[] { d + dx, y });
                }
            }
        }
        return false;
    }

    // 3923. 得到目标点的最少代数 (Minimum Generations to Target Point)
    private record Group3923(int x, int y, int z) {
    }

    public int minGenerations(int[][] points, int[] target) {
        Set<Group3923> cur = new HashSet<>();
        for (int[] p : points) {
            cur.add(new Group3923(p[0], p[1], p[2]));
        }
        Group3923 tar = new Group3923(target[0], target[1], target[2]);
        int res = 0;
        while (true) {
            if (cur.contains(tar)) {
                return res;
            }
            Set<Group3923> nxt = new HashSet<>();
            for (Group3923 g : cur) {
                nxt.add(g);
            }
            for (Group3923 g0 : cur) {
                for (Group3923 g1 : cur) {
                    nxt.add(new Group3923((g0.x + g1.x) / 2, (g0.y + g1.y) / 2, (g0.z + g1.z) / 2));
                }
            }
            if (cur.size() == nxt.size()) {
                return -1;
            }
            ++res;
            cur = nxt;
        }
    }

    // 3928. 购买苹果的最低成本 II (Minimum Cost to Buy Apples II)
    public int[] minCost(int n, int[] prices, int[][] roads) {
        List<int[]>[] g = new ArrayList[n];
        Arrays.setAll(g, o -> new ArrayList<>());
        for (int[] r : roads) {
            g[r[0]].add(new int[] { r[1], r[2], r[3] });
            g[r[1]].add(new int[] { r[0], r[2], r[3] });
        }
        // dis0[i][j] : 从i到j 不携带苹果，的最小值
        int[][] dis0 = new int[n][n];
        // dis1[i][j] : 从i到j 携带苹果，的最小值
        int[][] dis1 = new int[n][n];
        for (int i = 0; i < n; ++i) {
            dis0[i] = cal3928(i, g, false, prices);
            dis1[i] = cal3928(i, g, true, prices);
        }
        int[] res = new int[n];
        Arrays.fill(res, Integer.MAX_VALUE);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                res[i] = (int) Math.min(res[i], (long) dis0[i][j] + dis1[j][i] + prices[j]);
            }
        }
        return res;

    }

    private int[] cal3928(int start, List<int[]>[] g, boolean carryApple, int[] prices) {
        int[] dis = new int[g.length];
        Arrays.fill(dis, carryApple ? Integer.MAX_VALUE : prices[start]);
        dis[start] = 0;
        Queue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });
        q.offer(new int[] { 0, start });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int d = cur[0];
            int x = cur[1];
            if (d > dis[x]) {
                continue;
            }
            for (int[] nxt : g[x]) {
                int y = nxt[0];
                int cost = nxt[1];
                int tax = nxt[2];
                long dx = carryApple ? (long) cost * tax : cost;
                if (d + dx < dis[y]) {
                    dis[y] = (int) (d + dx);
                    q.offer(new int[] { (int) (d + dx), y });
                }
            }
        }
        return dis;
    }

    // 3931. 检查相邻数字差 (Check Adjacent Digit Differences)
    public boolean isAdjacentDiffAtMostTwo(String s) {
        for (int i = 1; i < s.length(); ++i) {
            if (Math.abs(s.charAt(i) - s.charAt(i - 1)) > 2) {
                return false;
            }
        }
        return true;

    }

    // 3932. 统计区间内的完全 K 次幂数量 (Count K-th Roots in a Range)
    public int countKthRoots(int l, int r, int k) {
        if (k == 1) {
            return r - l + 1;
        }
        int res = 0;
        int x = 0;
        while (true) {
            long y = (long) Math.pow(x, k);
            if (y > r) {
                break;
            }
            if (y >= l) {
                ++res;
            }
            ++x;
        }
        return res;

    }

    // 3933. 矩阵中的局部最大值 II (Largest Local Values in a Matrix II)
    public int countLocalMaximums(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        Map<Integer, List<int[]>> map = new HashMap<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                map.computeIfAbsent(matrix[i][j], o -> new ArrayList<>()).add(new int[] { i, j });
            }
        }
        map.remove(0);
        int res = 0;
        for (Map.Entry<Integer, List<int[]>> entry : map.entrySet()) {
            res += check3933(matrix, entry.getKey(), entry.getValue());
        }
        return res;

    }

    private int check3933(int[][] matrix, int x, List<int[]> list) {
        int m = matrix.length;
        int n = matrix[0].length;
        int[][] pre = new int[m + 1][n + 1];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                pre[i + 1][j + 1] = pre[i + 1][j] + pre[i][j + 1] - pre[i][j] + (matrix[i][j] > x ? 1 : 0);
            }
        }
        int res = 0;
        for (int[] p : list) {
            int i = p[0];
            int j = p[1];
            int x1 = Math.min(m, i + x + 1);
            int y1 = Math.min(n, j + x + 1);
            int x0 = Math.max(0, i - x);
            int y0 = Math.max(0, j - x);
            int cnt = pre[x1][y1] - pre[x1][y0] - pre[x0][y1] + pre[x0][y0];
            if (i - x >= 0 && j - x >= 0) {
                cnt -= matrix[i - x][j - x] > x ? 1 : 0;
            }
            if (i - x >= 0 && j + x < n) {
                cnt -= matrix[i - x][j + x] > x ? 1 : 0;
            }
            if (i + x < m && j - x >= 0) {
                cnt -= matrix[i + x][j - x] > x ? 1 : 0;
            }
            if (i + x < m && j + x < n) {
                cnt -= matrix[i + x][j + x] > x ? 1 : 0;
            }
            if (cnt == 0) {
                ++res;
            }

        }
        return res;
    }

    // 3934. 最短唯一子数组 (Smallest Unique Subarray) --滚动哈希
    public int smallestUniqueSubarray(int[] nums) {
        int n = nums.length;
        int left = 1;
        int right = n;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (hasUniqueSubarray3934(nums, mid)) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return right + 1;

    }

    public boolean hasUniqueSubarray3934(int[] arr, int k) {
        int n = arr.length;
        final long BASE = 131L;
        final long MOD = (1L << 61) - 1; // 梅森素数

        // 预计算 BASE^k mod MOD
        long power = modPow3934(BASE, k);

        // 计算第一个窗口的哈希值
        long h = 0;
        for (int i = 0; i < k; i++) {
            h = modMul3934(h, BASE);
            h = (h + arr[i]) % MOD;
        }

        Map<Long, Integer> count = new HashMap<>();
        count.put(h, 1);

        // 滑动窗口
        for (int i = 1; i <= n - k; i++) {
            // 滚出左边元素，滚入右边元素
            long left = modMul3934(arr[i - 1], power);
            h = modMul3934(h, BASE);
            h = ((h - left + arr[i + k - 1] + MOD) % MOD + MOD) % MOD;
            count.merge(h, 1, Integer::sum);
        }

        // 检查是否存在只出现一次的子数组
        for (int v : count.values()) {
            if (v == 1)
                return true;
        }
        return false;
    }

    /** 模乘：防止 long 溢出，使用 128 位分段乘法 */
    /** 梅森素数专用快速模乘，彻底安全 */
    private static final long MOD3934 = (1L << 61) - 1;

    private long modMul3934(long a, long b) {
        // Java 9+ 提供无符号128位乘法的高64位

        long hi = Math.multiplyHigh(a, b);
        long lo = a * b; // 低64位（自动截断）

        // 128位数 hi:lo 对 2^61-1 取模
        // 利用 2^61 ≡ 1，所以 x * 2^61k ≡ x
        // hi:lo = hi * 2^64 + lo
        // 2^64 = 2^3 * 2^61 ≡ 2^3 = 8
        // 所以 hi * 2^64 ≡ hi * 8
        // 最终 = (hi * 8 + lo) mod (2^61-1)
        //
        // 但 hi*8 + lo 仍可能 > 2^64，需再次折叠
        // hi < 2^64, hi*8 < 2^67, 取高3位折叠
        long result = (hi << 3 | lo >>> 61) + (lo & MOD3934);
        if (result >= MOD3934)
            result -= MOD3934;
        return result;
    }

    private long modPow3934(long base, long exp) {
        long result = 1L;
        base %= MOD3934;
        while (exp > 0) {
            if ((exp & 1) == 1)
                result = modMul3934(result, base);
            base = modMul3934(base, base);
            exp >>= 1;
        }
        return result;
    }

    // 3936. 将 0 移到末尾的最少交换次数 (Minimum Swaps to Move Zeros to End)
    public int minimumSwaps(int[] nums) {
        int n = nums.length;
        int left = 0;
        int right = n - 1;
        int res = 0;
        while (left < right) {
            while (left < right && nums[left] != 0) {
                ++left;
            }
            while (left < right && nums[right] == 0) {
                --right;
            }
            if (left < right) {
                ++left;
                --right;
                ++res;
            }
        }
        return res;

    }

    // 3937. 使数组变为模交替数组的最少操作次数 I (Minimum Operations to Make Array Modulo
    // Alternating I)
    public int minOperations(int[] nums, int k) {
        List<Integer> s0 = check3937(nums, 0, k);
        List<Integer> s1 = check3937(nums, 1, k);
        if (s0.get(1) != s1.get(1)) {
            return s0.get(0) + s1.get(0);
        }
        return Math.min(s0.get(0) + s1.get(2), s0.get(2) + s1.get(0));
    }

    private List<Integer> check3937(int[] nums, int start, int k) {
        int s0 = Integer.MAX_VALUE;
        int x0 = Integer.MAX_VALUE;
        int s1 = Integer.MAX_VALUE;
        for (int x = 0; x < k; ++x) {
            int s = 0;
            for (int i = start; i < nums.length; i += 2) {
                int v = nums[i] % k;
                int d = Math.abs(v - x);
                s += Math.min(d, k - d);
            }
            if (s <= s0) {
                s1 = s0;
                s0 = s;
                x0 = x;
            } else if (s <= s1) {
                s1 = s;
            }
        }
        return List.of(s0, x0, s1);
    }
}
