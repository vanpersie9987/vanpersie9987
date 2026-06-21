import java.math.BigInteger;
import java.sql.Time;
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

    // 3938. 矩阵中最大共享路径和 (Maximum Path Intersection Sum in a Grid)
    public int maxScore(int[][] grid) {
        return Math.max(check3938(grid), check3938(rotate3938(grid)));
    }

    private int[][] rotate3938(int[][] a) {
        int m = a.length;
        int n = a[0].length;
        int[][] res = new int[n][m];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                res[j][i] = a[i][j];
            }
        }
        return res;
    }

    private int check3938(int[][] a) {
        int m = a.length;
        int n = a[0].length;
        int res = Integer.MIN_VALUE;
        for (int i = 1; i < m - 1; ++i) {
            for (int j = 1; j < n - 1; ++j) {
                res = Math.max(res, a[i][j]);
            }
        }

        for (int[] row : a) {
            int s0 = 0;
            int s1 = 0;
            int pre = 0;
            int min_s1 = Integer.MAX_VALUE / 2;
            for (int x : row) {
                pre += x;
                res = Math.max(res, pre - min_s1);
                s1 = s0;
                s0 = pre;
                min_s1 = Math.min(min_s1, s1);
            }
        }
        return res;
    }

    // 3940. 限制有序数组中的元素出现次数 (Limit Occurrences in Sorted Array)
    public int[] limitOccurrences(int[] nums, int k) {
        int pre = -1;
        int cnt = 0;
        List<Integer> res = new ArrayList<>();
        for (int x : nums) {
            if (pre != x) {
                cnt = 0;
                pre = x;
            }
            if (cnt++ < k) {
                res.add(x);
            }
        }
        return res.stream().mapToInt(i -> i).toArray();

    }

    // 3941. 密码强度 (Password Strength)
    public int passwordStrength(String password) {
        Set<Character> s = new HashSet<>();
        for (char c : password.toCharArray()) {
            s.add(c);
        }
        int res = 0;
        for (char c : s) {
            if (c >= '0' && c <= '9') {
                res += 3;
            } else if (c >= 'a' && c <= 'z') {
                ++res;
            } else if (c >= 'A' && c <= 'Z') {
                res += 2;
            } else {
                res += 5;
            }
        }
        return res;

    }

    // 3945. 计算数字频率得分 (Digit Frequency Score)
    public int digitFrequencyScore(int n) {
        int[] cnts = new int[10];
        for (; n > 0; n /= 10) {
            ++cnts[n % 10];
        }
        int res = 0;
        for (int i = 0; i < 10; ++i) {
            res += i * cnts[i];
        }
        return res;

    }

    // 3946. 购买最多物品数目 I (Maximum Number of Items From Sale I)
    private int[][] memo3946;
    private int[][] items3946;
    private int[] cnts3946;

    public int maximumSaleItems(int[][] items, int budget) {
        int n = items.length;
        this.cnts3946 = new int[n];
        int minPrice = Integer.MAX_VALUE;
        for (int i = 0; i < n; ++i) {
            minPrice = Math.min(minPrice, items[i][1]);
            for (int j = 0; j < n; ++j) {
                if (i != j && items[j][0] % items[i][0] == 0) {
                    ++cnts3946[i];
                }
            }
        }
        this.memo3946 = new int[n][budget + 1];
        this.items3946 = items;
        for (int i = 0; i < n; ++i) {
            Arrays.fill(memo3946[i], -1);
        }
        int res = 0;
        for (int i = 1; i <= budget; ++i) {
            res = Math.max(res, dfs3946(n - 1, i) + (budget - i) / minPrice);
        }
        return res;

    }

    private int dfs3946(int i, int j) {
        if (i < 0) {
            return 0;
        }
        if (memo3946[i][j] != -1) {
            return memo3946[i][j];
        }
        int res = dfs3946(i - 1, j);
        if (j >= items3946[i][1]) {
            res = Math.max(res, dfs3946(i - 1, j - items3946[i][1]) + 1 + cnts3946[i]);
        }
        return memo3946[i][j] = res;
    }

    // 3950. 恰好一对连续置位 (Exactly One Consecutive Set Bits Pair)
    public boolean consecutiveSetBits(int n) {
        int s = n & (n >> 1);
        return Integer.bitCount(s) == 1;

    }

    // 3951. 维持亮度的最小总能量 (Minimum Energy to Maintain Brightness)
    public long minEnergy(int n, int brightness, int[][] intervals) {
        Arrays.sort(intervals, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });
        int i = 0;
        long cnt = 0L;
        while (i < intervals.length) {
            int left = intervals[i][0];
            int right = intervals[i][1];
            int j = i;
            while (j < intervals.length && right >= intervals[j][0]) {
                right = Math.max(right, intervals[j++][1]);
            }
            cnt += right - left + 1;
            i = j;
        }
        return (brightness + 3 - 1) / 3 * cnt;

    }

    // 3952. 下标覆盖处的最大总和 (Maximum Total Value of Covered Indices)
    private long[][] memo3952;
    private boolean[] pre3952;
    private String s3952;
    private int[] nums3952;

    public long maxTotal(int[] nums, String s) {
        int n = nums.length;
        this.s3952 = s;
        this.nums3952 = nums;
        this.pre3952 = new boolean[n + 1];
        for (int i = 0; i < n; ++i) {
            pre3952[i + 1] = pre3952[i] || (s.charAt(i) == '0');
        }
        this.memo3952 = new long[n][2];
        for (long[] r : memo3952) {
            Arrays.fill(r, Long.MAX_VALUE);
        }
        return dfs3952(n - 1, 0);

    }

    private long dfs3952(int i, int j) {
        if (i < 0) {
            return 0L;
        }
        if (!pre3952[i + 1] && j == 1) {
            return Long.MIN_VALUE / 2;
        }
        if (memo3952[i][j] != Long.MAX_VALUE) {
            return memo3952[i][j];
        }
        if (s3952.charAt(i) == '0') {
            return memo3952[i][j] = dfs3952(i - 1, 0) + (j == 1 ? nums3952[i] : 0);
        }
        if (j == 0) {
            return memo3952[i][j] = Math.max(dfs3952(i - 1, 0) + nums3952[i], dfs3952(i - 1, 1));
        }
        if (pre3952[i + 1]) {
            return memo3952[i][j] = dfs3952(i - 1, 1) + nums3952[i];
        }
        return memo3952[i][j] = Long.MIN_VALUE / 2;
    }

    // 3954. 区间内的兼容数字之和 I (Sum of Compatible Numbers in Range I)
    public int sumOfGoodIntegers(int n, int k) {
        int res = 0;
        for (int x = Math.max(0, n - k); x <= n + k; ++x) {
            if ((n & x) == 0) {
                res += x;
            }
        }
        return res;

    }

    // 3955. 成本限制的有效二进制字符串 (Valid Binary Strings With Cost Limit)
    private List<String> res3955;
    private int n3955;
    private int k3955;
    private StringBuilder path3955;

    public List<String> generateValidStrings(int n, int k) {
        this.res3955 = new ArrayList<>();
        this.n3955 = n;
        this.k3955 = k;
        this.path3955 = new StringBuilder();
        dfs3955(0);
        return res3955;
    }

    private void dfs3955(int c) {
        if (path3955.length() == n3955) {
            res3955.add(path3955.toString());
            return;
        }
        path3955.append('0');
        dfs3955(c);
        path3955.deleteCharAt(path3955.length() - 1);
        if ((path3955.length() == 0 || path3955.charAt(path3955.length() - 1) == '0')
                && c + path3955.length() <= k3955) {
            int nc = c + path3955.length();
            path3955.append('1');
            dfs3955(nc);
            path3955.deleteCharAt(path3955.length() - 1);
        }
    }

    // 3959. 判定好整数 (Check Good Integer)
    public boolean checkGoodInteger(int n) {
        int squareSum = 0;
        int digitSum = 0;
        while (n != 0) {
            int m = n % 10;
            squareSum += m * m;
            digitSum += m;
            n /= 10;
        }
        return squareSum - digitSum >= 50;

    }

    // 3960. 频率平衡子数组 (Frequency Balance Subarray)
    public int getLength(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        Map<Integer, Integer> cnts = new HashMap<>();
        int res = 0;
        for (int i = 0; i < nums.length; ++i) {
            map.clear();
            cnts.clear();
            for (int j = i; j < nums.length; ++j) {
                int c = map.getOrDefault(nums[j], 0);
                if (c != 0) {
                    cnts.merge(c, -1, Integer::sum);
                    if (cnts.get(c) == 0) {
                        cnts.remove(c);
                    }
                }
                map.merge(nums[j], 1, Integer::sum);
                cnts.merge(c + 1, 1, Integer::sum);
                if (map.size() == 1) {
                    res = Math.max(res, j - i + 1);
                } else if (cnts.size() == 2) {
                    int c1 = 0;
                    int c2 = 0;
                    for (int cnt : cnts.keySet()) {
                        if (c1 == 0) {
                            c1 = cnt;
                        } else if (c2 == 0) {
                            c2 = cnt;
                        }
                    }
                    if (c1 * 2 == c2 || c2 * 2 == c1) {
                        res = Math.max(res, j - i + 1);
                    }
                }
            }
        }
        return res;

    }

    // 3961. 设备评分的最大和 (Maximize Sum of Device Ratings)
    public long maxRatings(int[][] units) {
        int n = units[0].length;
        long res = 0L;
        if (n == 1) {
            for (int[] unit : units) {
                res += unit[0];
            }
            return res;
        }
        int mn1 = Integer.MAX_VALUE;
        int mn2 = Integer.MAX_VALUE;
        for (int[] unit : units) {
            int min1 = Integer.MAX_VALUE;
            int min2 = Integer.MAX_VALUE;
            for (int x : unit) {
                if (x <= min1) {
                    min2 = min1;
                    min1 = x;
                } else if (x <= min2) {
                    min2 = x;
                }
            }
            res += min2;
            mn2 = Math.min(min2, mn2);
            mn1 = Math.min(min1, mn1);
        }
        res += mn1 - mn2;
        return res;

    }

    // 3963. 构造恰好一条路径的网格 (Create Grid With Exactly One Path)
    public String[] createGrid(int m, int n) {
        String[] res = new String[m];
        for (int i = 0; i < m; ++i) {
            StringBuilder row = new StringBuilder();
            for (int j = 0; j < n; ++j) {
                if (i == 0 || j == n - 1) {
                    row.append(".");
                } else {
                    row.append("#");
                }
            }
            res[i] = row.toString();
        }
        return res;

    }

    // 3964. 照亮道路的最少灯泡数 (Minimum Lights to Illuminate a Road)
    public int minLights(int[] lights) {
        int n = lights.length;
        int[] diff = new int[n + 1];
        for (int i = 0; i < n; ++i) {
            if (lights[i] > 0) {
                ++diff[Math.max(0, i - lights[i])];
                --diff[Math.min(n, i + lights[i] + 1)];
            }
        }
        int res = 0;
        int s = 0;
        int cur = 0;
        for (int i = 0; i < n; ++i) {
            s += diff[i];
            if (s == 0) {
                ++cur;
            } else {
                res += (cur + 2) / 3;
                cur = 0;
            }
        }
        res += (cur + 2) / 3;
        return res;
    }

}
