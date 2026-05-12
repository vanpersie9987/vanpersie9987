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
        Arrays.fill(dis, Integer.MAX_VALUE);
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
                if (w > limit) {
                    if (d + 1 < dis[y] && d + 1 <= k) {
                        dis[y] = d + 1;
                        q.offer(new int[] { d + 1, y });
                    }
                } else {
                    if (d < dis[y]) {
                        dis[y] = d;
                        q.offer(new int[] { d, y });
                    }
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

}
