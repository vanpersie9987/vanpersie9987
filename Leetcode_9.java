import java.net.Inet4Address;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
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
        TreeMap<Integer, List<Integer>> map = new TreeMap<>();
        for (int i = 0; i < points.length; ++i) {
            map.computeIfAbsent(Math.max(Math.abs(points[i][0]), Math.abs(points[i][1])), k -> new ArrayList<>())
                    .add(s.charAt(i) - 'a');
        }
        int res = 0;
        int m = 0;
        for (Map.Entry<Integer, List<Integer>> entey : map.entrySet()) {
            int curM = 0;
            for (int v : entey.getValue()) {
                if (((curM >> v) & 1) != 0 || ((m >> v) & 1) != 0) {
                    return res;
                }
                curM |= 1 << v;
            }
            res += Integer.bitCount(curM);
            m |= curM;
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

    // 3147. 从魔法师身上吸取的最大能量 (Taking Maximum Energy From the Mystic Dungeon)
    public int maximumEnergy(int[] energy, int k) {
        int n = energy.length;
        int res = Integer.MIN_VALUE;
        for (int i = n - k; i < n; ++i) {
            int s = 0;
            for (int j = i; j >= 0; j -= k) {
                s += energy[j];
                res = Math.max(res, s);
            }
        }
        return res;
    }

    // 3148. 矩阵中的最大得分 (Maximum Difference Score in a Grid)
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

    // 3149. 找出分数最低的排列 (Find the Minimum Cost Array Permutation)
    private int n3149;
    private int[] nums3149;
    private int[][] memo3149;
    private int u3149;
    private int[] res3149;

    public int[] findPermutation(int[] nums) {
        this.n3149 = nums.length;
        this.nums3149 = nums;
        this.u3149 = (1 << n3149) - 1;
        this.memo3149 = new int[1 << n3149][n3149];
        for (int i = 0; i < 1 << n3149; ++i) {
            Arrays.fill(memo3149[i], -1);
        }
        dfs3149(1, 0);
        this.res3149 = new int[n3149];
        makeAns3149(1, 0);
        return res3149;
    }

    // score(perm) = |perm[0] - nums[perm[1]]| + |perm[1] - nums[perm[2]]| + ... +
    // |perm[n - 1] - nums[perm[0]]|
    private int dfs3149(int i, int j) {
        if (i == u3149) {
            return Math.abs(j - nums3149[0]);
        }
        if (memo3149[i][j] != -1) {
            return memo3149[i][j];
        }
        int min = Integer.MAX_VALUE;
        for (int c = u3149 ^ i; c != 0; c &= c - 1) {
            int lb = Integer.numberOfTrailingZeros(c);
            min = Math.min(min, dfs3149(i | (1 << lb), lb) + Math.abs(j - nums3149[lb]));
        }
        return memo3149[i][j] = min;
    }

    private void makeAns3149(int i, int j) {
        res3149[Integer.bitCount(i) - 1] = j;
        if (i == u3149) {
            return;
        }
        int finalAns = dfs3149(i, j);
        for (int c = u3149 ^ i; c != 0; c &= c - 1) {
            int lb = Integer.numberOfTrailingZeros(c);
            if (dfs3149(i | (1 << lb), lb) + Math.abs(j - nums3149[lb]) == finalAns) {
                makeAns3149(i | (1 << lb), lb);
                break;
            }
        }
    }

    // 368. 最大整除子集 (Largest Divisible Subset)
    private int[] nums368;
    private int n368;
    private int[] memo368;
    private List<Integer> res368;

    public List<Integer> largestDivisibleSubset(int[] nums) {
        Arrays.sort(nums);
        this.nums368 = nums;
        this.n368 = nums.length;
        this.memo368 = new int[n368];
        int mx = 0;
        int f = 0;
        for (int i = 0; i < n368; ++i) {
            int cur = dfs368(i);
            if (cur > mx) {
                mx = cur;
                f = i;
            }
        }
        this.res368 = new ArrayList<>();
        makeAns368(f);
        return res368;

    }

    private void makeAns368(int i) {
        res368.add(nums368[i]);
        int finalAns = dfs368(i);
        for (int j = 0; j < i; ++j) {
            if (nums368[i] % nums368[j] == 0 && dfs368(j) + 1 == finalAns) {
                makeAns368(j);
                break;
            }
        }
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

    // 943. 最短超级串 (Find the Shortest Superstring)
    private String[] words943;
    private int n943;
    private int[][] lcp943;
    private int[][] memo943;
    private StringBuilder res943;
    private int u943;

    public String shortestSuperstring(String[] words) {
        this.words943 = words;
        this.n943 = words.length;
        this.lcp943 = new int[n943][n943];
        for (int i = 0; i < n943; ++i) {
            for (int j = 0; j < n943; ++j) {
                lcp943[i][j] = cal943(words[i], words[j]);
            }
        }
        this.u943 = (1 << n943) - 1;
        this.memo943 = new int[1 << n943][n943];
        int min = Integer.MAX_VALUE;
        int f = 0;
        for (int i = 0; i < n943; ++i) {
            int cur = dfs943(1 << i, i) + words[i].length();
            if (cur < min) {
                min = cur;
                f = i;
            }
        }
        this.res943 = new StringBuilder();
        res943.append(words[f]);
        makeAns943(1 << f, f);
        return res943.toString();
    }

    private void makeAns943(int i, int j) {
        if (i == u943) {
            return;
        }
        int finalAns = dfs943(i, j);
        for (int c = i ^ u943; c != 0; c &= c - 1) {
            int lb = Integer.numberOfTrailingZeros(c);
            if (dfs943(i | (1 << lb), lb) + words943[lb].length() - lcp943[j][lb] == finalAns) {
                res943.append(words943[lb].substring(lcp943[j][lb]));
                makeAns943(i | (1 << lb), lb);
                break;
            }
        }
    }

    private int dfs943(int i, int j) {
        if (i == u943) {
            return 0;
        }
        if (memo943[i][j] != 0) {
            return memo943[i][j];
        }
        int res = Integer.MAX_VALUE;
        for (int c = i ^ u943; c != 0; c &= c - 1) {
            int lb = Integer.numberOfTrailingZeros(c);
            res = Math.min(res, dfs943(i | (1 << lb), lb) + words943[lb].length() - lcp943[j][lb]);
        }
        return memo943[i][j] = res;
    }

    private int cal943(String s, String t) {
        int n = Math.min(s.length(), t.length());
        for (int i = s.length() - n; i < s.length(); ++i) {
            if (t.startsWith(s.substring(i))) {
                return s.length() - i;
            }
        }
        return 0;
    }

    // 3151. 特殊数组 I (Special Array I)
    public boolean isArraySpecial(int[] nums) {
        int n = nums.length;
        for (int i = 1; i < n; ++i) {
            if (nums[i] % 2 == nums[i - 1] % 2) {
                return false;
            }
        }
        return true;

    }

    // 3152. 特殊数组 II (Special Array II)
    public boolean[] isArraySpecial(int[] nums, int[][] queries) {
        int n = nums.length;
        int m = queries.length;
        int[] pre = new int[n];
        for (int i = 1; i < n; ++i) {
            pre[i] = pre[i - 1] + ((nums[i] ^ nums[i - 1] ^ 1) & 1);
        }
        boolean[] res = new boolean[m];
        for (int i = 0; i < m; ++i) {
            res[i] = pre[queries[i][0]] == pre[queries[i][1]];
        }
        return res;
    }

    // 3153. 所有数对中数位不同之和 (Sum of Digit Differences of All Pairs)
    public long sumDigitDifferences(int[] nums) {
        int n = nums.length;
        long res = 0L;
        int bitLen = String.valueOf(nums[0]).length();
        for (int i = 0; i < bitLen; ++i) {
            int pow = (int) Math.pow(10, i);
            Map<Integer, Integer> cnts = new HashMap<>();
            for (int j = 0; j < n; ++j) {
                int b = nums[j] / pow % 10;
                res += j - cnts.getOrDefault(b, 0);
                cnts.merge(b, 1, Integer::sum);
            }
        }
        return res;
    }

    // 3154. 到达第 K 级台阶的方案数 (Find Number of Ways to Reach the K-th Stair)
    private int k3154;

    private Map<Long, Integer> memo3154;

    public int waysToReachStair(int k) {
        this.k3154 = k;
        this.memo3154 = new HashMap<>();
        return dfs3154(1L, 0, 1);

    }

    private int dfs3154(long i, int j, int last) {
        if (i > k3154 + 1) {
            return 0;
        }
        long m = (i << 10) | (j << 1) | last;
        if (memo3154.get(m) != null) {
            return memo3154.get(m);
        }
        int res = dfs3154(i + (1L << j), j + 1, 1);
        if (i > 0 && last == 1) {
            res += dfs3154(i - 1, j, 0);
        }
        res += i == k3154 ? 1 : 0;
        memo3154.put(m, res);
        return res;
    }

    // 3158. 求出出现两次数字的 XOR 值 (Find the XOR of Numbers Which Appear Twice)
    public int duplicateNumbersXOR(int[] nums) {
        long mask = 0L;
        int res = 0;
        for (int x : nums) {
            if ((mask >> x & 1) == 1) {
                res ^= x;
            } else {
                mask |= 1L << x;
            }
        }
        return res;

    }

    // 3159. 查询数组中元素的出现位置 (Find Occurrences of an Element in an Array)
    public int[] occurrencesOfElement(int[] nums, int[] queries, int x) {
        List<Integer> p = new ArrayList<>();
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] == x) {
                p.add(i);
            }
        }
        int[] res = new int[queries.length];
        for (int i = 0; i < queries.length; ++i) {
            res[i] = queries[i] <= p.size() ? p.get(queries[i] - 1) : -1;
        }
        return res;

    }

    // 3160. 所有球里面不同颜色的数目 (Find the Number of Distinct Colors Among the Balls)
    public int[] queryResults(int limit, int[][] queries) {
        Map<Integer, Integer> map = new HashMap<>();
        Map<Integer, Integer> kinds = new HashMap<>();
        int m = queries.length;
        int[] res = new int[m];
        for (int i = 0; i < m; ++i) {
            int id = queries[i][0];
            int color = queries[i][1];
            if (map.containsKey(id)) {
                int preColor = map.get(id);
                kinds.merge(preColor, -1, Integer::sum);
                if (kinds.get(preColor) == 0) {
                    kinds.remove(preColor);
                }
            }
            kinds.merge(color, 1, Integer::sum);
            map.put(id, color);
            res[i] = kinds.size();
        }
        return res;

    }

    // 3162. 优质数对的总数 I (Find the Number of Good Pairs I)
    // 3164. 优质数对的总数 II (Find the Number of Good Pairs II)
    public long numberOfPairs(int[] nums1, int[] nums2, int k) {
        Map<Integer, Integer> cnts = new HashMap<>();
        for (int x : nums1) {
            if (x % k != 0) {
                continue;
            }
            x /= k;
            for (int i = 1; i <= Math.sqrt(x); ++i) {
                if (x % i == 0) {
                    cnts.merge(i, 1, Integer::sum);
                    if (i * i < x) {
                        cnts.merge(x / i, 1, Integer::sum);
                    }
                }
            }
        }
        long res = 0L;
        for (int x : nums2) {
            res += cnts.getOrDefault(x, 0);
        }
        return res;

    }

    // 3163. 压缩字符串 III (String Compression III)
    public String compressedString(String word) {
        StringBuilder res = new StringBuilder();
        int i = 0;
        while (i < word.length()) {
            int j = i;
            while (j < word.length() && word.charAt(i) == word.charAt(j) && j - i + 1 <= 9) {
                ++j;
            }
            res.append(j - i).append(word.charAt(i));
            i = j;
        }
        return res.toString();

    }

    // 552. 学生出勤记录 II (Student Attendance Record II)
    private int n552;
    private int[][][] memo552;

    public int checkRecord(int n) {
        this.n552 = n;
        this.memo552 = new int[n][2][3];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < 2; ++j) {
                Arrays.fill(memo552[i][j], -1);
            }
        }
        return dfs552(0, 0, 0);
    }

    private int dfs552(int i, int j, int k) {
        if (j == 2 || k == 3) {
            return 0;
        }
        if (i == n552) {
            return 1;
        }
        if (memo552[i][j][k] != -1) {
            return memo552[i][j][k];
        }
        final int MOD = (int) (1e9 + 7);
        return memo552[i][j][k] = (int) (((long) dfs552(i + 1, j, 0) + dfs552(i + 1, j + 1, 0)
                + dfs552(i + 1, j, k + 1)) % MOD);

    }

    // 575. 分糖果 (Distribute Candies)
    public int distributeCandies(int[] candyType) {
        Set<Integer> s = Arrays.stream(candyType).boxed().collect(Collectors.toSet());
        return Math.min(candyType.length / 2, s.size());

    }

    // 3168. 候诊室中的最少椅子数 (Minimum Number of Chairs in a Waiting Room)
    public int minimumChairs(String s) {
        int res = 0;
        int cnt = 0;
        for (char c : s.toCharArray()) {
            cnt += c == 'E' ? 1 : -1;
            res = Math.max(res, cnt);
        }
        return res;

    }

    // 3169. 无需开会的工作日 (Count Days Without Meetings)
    public int countDays(int days, int[][] meetings) {
        List<int[]> list = new ArrayList<>();
        for (int[] m : meetings) {
            list.add(m);
        }
        list.add(new int[] { 0, 0 });
        list.add(new int[] { days + 1, days + 1 });
        Collections.sort(list, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });
        int res = 0;
        int i = 0;
        int n = list.size();
        while (i < n) {
            int right = list.get(i)[1];
            int j = i + 1;
            while (j < n && list.get(j)[0] <= right + 1) {
                right = Math.max(right, list.get(j)[1]);
                ++j;
            }
            if (j == n) {
                break;
            }
            res += list.get(j)[0] - right - 1;
            i = j;
        }
        return res;

    }

    // 3170. 删除星号以后字典序最小的字符串 (Lexicographically Minimum String After Removing Stars)
    public String clearStars(String s) {
        char[] arr = s.toCharArray();
        List<Integer>[] list = new ArrayList[26];
        Arrays.setAll(list, k -> new ArrayList<>());
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == '*') {
                for (int j = 0; j < 26; ++j) {
                    if (!list[j].isEmpty()) {
                        arr[list[j].remove(list[j].size() - 1)] = '*';
                        break;
                    }
                }
            } else {
                list[s.charAt(i) - 'a'].add(i);
            }
        }
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < arr.length; ++i) {
            if (arr[i] != '*') {
                res.append(arr[i]);
            }
        }
        return res.toString();

    }

    // 3171. 找到按位与最接近 K 的子数组 (Find Subarray With Bitwise AND Closest to K)
    public int minimumDifference(int[] nums, int k) {
        int res = Integer.MAX_VALUE;
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            int and = -1;
            for (int j = i; j >= 0 && i - j < 28; --j) {
                and &= nums[j];
                res = Math.min(res, Math.abs(k - and));
            }
        }
        return res;

    }

    // 3174. 清除数字 (Clear Digits)
    public String clearDigits(String s) {
        StringBuilder res = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (Character.isDigit(c)) {
                if (!res.isEmpty()) {
                    res.setLength(res.length() - 1);
                }
            } else {
                res.append(c);
            }
        }
        return res.toString();

    }

    // 3175. 找到连续赢 K 场比赛的第一位玩家 (Find The First Player to win K Games in a Row)
    public int findWinningPlayer(int[] skills, int k) {
        int max_i = 0;
        int win = -1;
        for (int i = 0; i < skills.length; ++i) {
            if (skills[i] > skills[max_i]) {
                max_i = i;
                win = 0;
            }
            ++win;
            if (win == k) {
                break;
            }
        }
        return max_i;
    }

    // 3176. 求出最长好子序列 I (Find the Maximum Length of a Good Subsequence I)
    private int[] nums3176;
    private int n3176;
    private int[][] memo3176;

    public int maximumLength(int[] nums, int k) {
        this.nums3176 = nums;
        this.n3176 = nums.length;
        this.memo3176 = new int[n3176][k + 1];
        for (int i = 0; i < n3176; ++i) {
            Arrays.fill(memo3176[i], -1);
        }
        int res = 0;
        for (int i = 0; i < n3176; ++i) {
            for (int j = 0; j <= k; ++j) {
                res = Math.max(res, dfs3176(i, j));
            }
        }
        return res;
    }

    private int dfs3176(int i, int j) {
        if (j < 0) {
            return -n3176 - 1;
        }
        if (memo3176[i][j] != -1) {
            return memo3176[i][j];
        }
        int res = 0;
        for (int x = i - 1; x >= 0; --x) {
            if (nums3176[x] != nums3176[i]) {
                res = Math.max(res, dfs3176(x, j - 1));
            } else {
                res = Math.max(res, dfs3176(x, j));
            }
        }
        return memo3176[i][j] = res + 1;
    }

    // 3178. 找出 K 秒后拿着球的孩子 (Find the Child Who Has the Ball After K Seconds)
    public int numberOfChild(int n, int k) {
        k %= (n - 1) << 1;
        if (k <= n - 1) {
            return k;
        }
        k -= n - 1;
        return n - 1 - k;
    }

    // 3184. 构成整天的下标对数目 I (Count Pairs That Form a Complete Day I)
    // 3185. 构成整天的下标对数目 II (Count Pairs That Form a Complete Day II)
    public long countCompleteDayPairs(int[] hours) {
        long res = 0L;
        int n = hours.length;
        int[] cnt = new int[24];
        for (int i = 0; i < n; ++i) {
            int h = hours[i] % 24;
            if (h != 0) {
                res += cnt[24 - h];
            }
            ++cnt[h];
        }
        return res + (long) cnt[0] * (cnt[0] - 1) / 2;

    }

    // 3186. 施咒的最大总伤害 (Maximum Total Damage With Spell Casting)
    private List<int[]> list3186;
    private int n3186;
    private long[] memo3186;

    public long maximumTotalDamage(int[] power) {
        Map<Integer, Integer> cnts = new HashMap<>();
        for (int p : power) {
            cnts.merge(p, 1, Integer::sum);
        }
        this.list3186 = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry : cnts.entrySet()) {
            list3186.add(new int[] { entry.getKey(), entry.getValue() });
        }
        Collections.sort(list3186, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });
        this.n3186 = list3186.size();
        this.memo3186 = new long[n3186];
        Arrays.fill(memo3186, -1L);
        return dfs3186(0);

    }

    private long dfs3186(int i) {
        if (i >= n3186) {
            return 0L;
        }
        if (memo3186[i] != -1) {
            return memo3186[i];
        }
        long res = dfs3186(i + 1);
        int j = i + 1;
        while (j < n3186 && list3186.get(j)[0] - list3186.get(i)[0] <= 2) {
            ++j;
        }
        res = Math.max(res, dfs3186(j) + (long) list3186.get(i)[0] * list3186.get(i)[1]);
        return memo3186[i] = res;
    }

    // 3190. 使所有元素都可以被 3 整除的最少操作数 (Find Minimum Operations to Make All Elements
    // Divisible by Three)
    public int minimumOperations(int[] nums) {
        int res = 0;
        for (int x : nums) {
            res += Math.min(x % 3, 1);
        }
        return res;
    }

    // 3191. 使二进制数组全部等于 1 的最少操作次数 I
    public int minOperations(int[] nums) {
        int res = 0;
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            if (nums[i] == 1) {
                continue;
            }
            if (i + 2 >= n) {
                return -1;
            }
            nums[i + 1] ^= 1;
            nums[i + 2] ^= 1;
            ++res;
        }
        return res;
    }

    // 3192. 使二进制数组全部等于 1 的最少操作次数 II (Minimum Operations to Make Binary Array
    // Elements Equal to One II)
    public int minOperationsII(int[] nums) {
        int res = 0;
        int cnt = 0;
        for (int x : nums) {
            x ^= cnt;
            if (x == 0) {
                ++res;
                cnt ^= 1;
            }
        }
        return res;

    }

    // 3194. 最小元素和最大元素的最小平均值 (Minimum Average of Smallest and Largest Elements)
    public double minimumAverage(int[] nums) {
        int min = 100;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length / 2; ++i) {
            min = Math.min(min, nums[i] + nums[nums.length - i - 1]);
        }
        return min / 2D;
    }

    // 3195. 包含所有 1 的最小矩形面积 I (Find the Minimum Area to Cover All Ones I)
    public int minimumArea(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int minRow = m + 1;
        int maxRow = -1;
        int minCol = n + 1;
        int maxCol = -1;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    minRow = Math.min(minRow, i);
                    maxRow = Math.max(maxRow, i);
                    minCol = Math.min(minCol, j);
                    maxCol = Math.max(maxCol, j);
                }
            }
        }
        return Math.max(0, maxRow - minRow + 1) * Math.max(0, maxCol - minCol + 1);

    }

    // 3196. 最大化子数组的总成本 (Maximize Total Cost of Alternating Subarrays)
    private int n3196;
    private int[] nums3196;
    private long[][] memo3196;

    public long maximumTotalCost(int[] nums) {
        this.nums3196 = nums;
        this.n3196 = nums.length;
        this.memo3196 = new long[n3196][2];
        for (int i = 0; i < n3196; ++i) {
            Arrays.fill(memo3196[i], (long) 1e15);
        }
        return dfs3196(0, 0);

    }

    private long dfs3196(int i, int j) {
        if (i == n3196) {
            return 0;
        }
        if (memo3196[i][j] != (long) 1e15) {
            return memo3196[i][j];
        }
        return memo3196[i][j] = Math.max(dfs3196(i + 1, 1) + nums3196[i],
                dfs3196(i + 1, j ^ 1) + (-2 * j + 1) * nums3196[i]);
    }

    // 3193. 统计逆序对的数目 (Count the Number of Inversions)
    private int[][] memo3193;
    private int[] req3193;
    private int max3193;

    public int numberOfPermutations(int n, int[][] requirements) {
        for (int[] r : requirements) {
            max3193 = Math.max(max3193, r[1]);
        }
        this.req3193 = new int[n];
        Arrays.fill(req3193, max3193 + 1);
        req3193[0] = 0;
        for (int[] r : requirements) {
            req3193[r[0]] = r[1];
        }
        if (req3193[0] != 0) {
            return 0;
        }
        this.memo3193 = new int[n][max3193 + 2];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(memo3193[i], -1);
        }
        return dfs3193(n - 1, req3193[n - 1]);

    }

    private int dfs3193(int i, int j) {
        if (i == 0) {
            return 1;
        }
        if (memo3193[i][j] != -1) {
            return memo3193[i][j];
        }
        int r = req3193[i - 1];
        if (r != max3193 + 1) {
            return memo3193[i][j] = (j >= r && j <= i + r) ? dfs3193(i - 1, r) : 0;
        }
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        for (int k = 0; k <= Math.min(i, j); ++k) {
            res += dfs3193(i - 1, j - k);
            res %= MOD;
        }
        return memo3193[i][j] = res;
    }

    // 3200. 三角形的最大高度 (Maximum Height of a Triangle)
    public int maxHeightOfTriangle(int red, int blue) {
        return Math.max(check3200(red, blue), check3200(blue, red));
    }

    private int check3200(int x, int y) {
        int i = 1;
        int[] left = new int[] { x, y };
        while (left[i % 2] >= i) {
            left[i % 2] -= i;
            ++i;
        }
        return i - 1;
    }

    // 3201. 找出有效子序列的最大长度 I (Find the Maximum Length of Valid Subsequence I)
    public int maximumLength(int[] nums) {
        int cntEven = 0;
        for (int x : nums) {
            if (x % 2 == 0) {
                ++cntEven;
            }
        }
        return Math.max(Math.max(cntEven, nums.length - cntEven), Math.max(check3201(nums, 0), check3201(nums, 1)));
    }

    private int check3201(int[] nums, int pre) {
        int res = 0;
        for (int x : nums) {
            if (x % 2 != pre % 2) {
                ++res;
                pre = x;
            }
        }
        return res;
    }

    // 3202. 找出有效子序列的最大长度 II (Find the Maximum Length of Valid Subsequence II)
    public int maximumLengthII(int[] nums, int k) {
        int res = 0;
        for (int m = 0; m < k; ++m) {
            int[] f = new int[k];
            for (int x : nums) {
                x %= k;
                f[x] = f[((m - x) % k + k) % k] + 1;
                res = Math.max(res, f[x]);
            }
        }
        return res;
    }

    // 3203. 合并两棵树后的最小直径 (Find Minimum Diameter After Merging Two Trees)
    public int minimumDiameterAfterMerge(int[][] edges1, int[][] edges2) {
        int d1 = check3203(edges1);
        int d2 = check3203(edges2);
        return Math.max(Math.max(d1, d2), ((d1 + 1) >> 1) + ((d2 + 1) >> 1) + 1);

    }

    private List<Integer>[] g3203;
    private int res3203;

    private int check3203(int[][] edges) {
        int n = edges.length + 1;
        this.g3203 = new ArrayList[n];
        Arrays.setAll(g3203, k -> new ArrayList<>());
        for (int[] e : edges) {
            g3203[e[0]].add(e[1]);
            g3203[e[1]].add(e[0]);
        }
        res3203 = 0;
        dfs3203(0, -1);
        return res3203 - 1;
    }

    private int dfs3203(int x, int fa) {
        int pre = 0;
        int mx = 0;
        for (int y : g3203[x]) {
            if (y != fa) {
                int cur = dfs3203(y, x);
                mx = Math.max(mx, pre + cur);
                pre = Math.max(pre, cur);
            }
        }
        res3203 = Math.max(res3203, mx + 1);
        return pre + 1;
    }

    // 3206. 交替组 I (Alternating Groups I)
    public int numberOfAlternatingGroups(int[] colors) {
        int n = colors.length;
        int res = 0;
        for (int i = 0; i < n; ++i) {
            if (colors[i] != colors[(i - 1 + n) % n] && colors[i] != colors[(i + 1) % n]) {
                ++res;
            }
        }
        return res;

    }

    // 3207. 与敌人战斗后的最大分数 (Maximum Points After Enemy Battles)
    public long maximumPoints(int[] enemyEnergies, int currentEnergy) {
        Arrays.sort(enemyEnergies);
        long res = 0L;
        int n = enemyEnergies.length;
        if (enemyEnergies[0] > currentEnergy) {
            return 0;
        }
        int i = 0;
        int j = n - 1;
        while (i <= j) {
            if (currentEnergy >= enemyEnergies[i]) {
                res += currentEnergy / enemyEnergies[i];
                currentEnergy %= enemyEnergies[i];
            } else {
                currentEnergy += enemyEnergies[j--];
            }
        }
        return res;
    }

    // 3208. 交替组 II (Alternating Groups II)
    public int numberOfAlternatingGroups(int[] colors, int k) {
        int n = colors.length;
        int res = 0;
        int d = 0;
        for (int i = 0; i < k - 1; ++i) {
            if (colors[i] != colors[i + 1]) {
                ++d;
            }
        }
        if (d == k - 1) {
            ++res;
        }
        for (int i = 1; i < n; ++i) {
            if (colors[i] != colors[i - 1]) {
                --d;
            }
            if (colors[(i + k - 2 + n) % n] != colors[(i + k - 1 + n) % n]) {
                ++d;
            }
            if (d == k - 1) {
                ++res;
            }
        }
        return res;

    }

    // 3210. 找出加密后的字符串 (Find the Encrypted String)
    public String getEncryptedString(String s, int k) {
        int n = s.length();
        k %= n;
        return s.substring(k) + s.substring(0, k);
    }

    // 3211. 生成不含相邻零的二进制字符串 (Generate Binary Strings Without Adjacent Zeros)
    private List<String> res3211;
    private StringBuilder builder3211;
    private int n3211;

    public List<String> validStrings(int n) {
        this.res3211 = new ArrayList<>();
        this.builder3211 = new StringBuilder();
        this.n3211 = n;
        dfs3211(0, 1);
        return res3211;

    }

    private void dfs3211(int i, int j) {
        if (i == n3211) {
            res3211.add(builder3211.toString());
            return;
        }
        builder3211.append('1');
        dfs3211(i + 1, 1);
        builder3211.deleteCharAt(builder3211.length() - 1);
        if (j == 1) {
            builder3211.append('0');
            dfs3211(i + 1, 0);
            builder3211.deleteCharAt(builder3211.length() - 1);
        }
    }

    // 3212. 统计 X 和 Y 频数相等的子矩阵数量 (Count Submatrices With Equal Frequency of X and Y)
    public int numberOfSubmatrices(char[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int res = 0;
        int[][] xs = new int[m + 1][n + 1];
        int[][] ys = new int[m + 1][n + 1];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                xs[i + 1][j + 1] = xs[i + 1][j] + xs[i][j + 1] - xs[i][j] + (grid[i][j] == 'X' ? 1 : 0);
                ys[i + 1][j + 1] = ys[i + 1][j] + ys[i][j + 1] - ys[i][j] + (grid[i][j] == 'Y' ? 1 : 0);
                if (xs[i + 1][j + 1] == ys[i + 1][j + 1] && xs[i + 1][j + 1] > 0) {
                    ++res;
                }
            }
        }
        return res;

    }

    // 3216. 交换后字典序最小的字符串 (Lexicographically Smallest String After a Swap)
    public String getSmallestString(String s) {
        int n = s.length();
        char[] arr = s.toCharArray();
        for (int i = 0; i < n - 1; ++i) {
            if ((arr[i] - 'a') % 2 == (arr[i + 1] - 'a') % 2 && arr[i] > arr[i + 1]) {
                char temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                break;
            }
        }
        return String.valueOf(arr);

    }

    // 3217. 从链表中移除在数组中存在的节点 (Delete Nodes From Linked List Present in Array)
    public ListNode modifiedList(int[] nums, ListNode head) {
        Set<Integer> s = new HashSet<>();
        for (int x : nums) {
            s.add(x);
        }
        ListNode dummy = new ListNode(0, head);
        ListNode cur = dummy;
        while (true) {
            while (head != null && s.contains(head.val)) {
                head = head.next;
            }
            cur.next = head;
            cur = cur.next;
            if (head == null) {
                break;
            }
            head = head.next;
        }
        return dummy.next;

    }

    // 3218. 切蛋糕的最小总开销 I (Minimum Cost for Cutting Cake I) --O((mn)^2)
    private int[] horizontalCut3218;
    private int[] verticalCut3218;
    private int[][][][] memo3218;

    public int minimumCost(int m, int n, int[] horizontalCut, int[] verticalCut) {
        this.horizontalCut3218 = horizontalCut;
        this.verticalCut3218 = verticalCut;
        this.memo3218 = new int[m][n][m][n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < m; ++k) {
                    Arrays.fill(memo3218[i][j][k], -1);
                }
            }
        }
        return dfs3218(0, 0, m - 1, n - 1);

    }

    private int dfs3218(int i0, int j0, int i1, int j1) {
        if (i0 == i1 && j0 == j1) {
            return 0;
        }
        if (memo3218[i0][j0][i1][j1] != -1) {
            return memo3218[i0][j0][i1][j1];
        }
        int res = Integer.MAX_VALUE;
        for (int i = i0; i < i1; ++i) {
            res = Math.min(res, dfs3218(i0, j0, i, j1) + dfs3218(i + 1, j0, i1, j1) + horizontalCut3218[i]);
        }
        for (int j = j0; j < j1; ++j) {
            res = Math.min(res, dfs3218(i0, j0, i1, j) + dfs3218(i0, j + 1, i1, j1) + verticalCut3218[j]);
        }
        return memo3218[i0][j0][i1][j1] = res;
    }

    // 3219. 切蛋糕的最小总开销 II (Minimum Cost for Cutting Cake II) --O(log(m) + log(n))
    public long minimumCostII(int m, int n, int[] horizontalCut, int[] verticalCut) {
        Arrays.sort(verticalCut);
        Arrays.sort(horizontalCut);
        long h = 1L;
        long v = 1L;
        long res = 0L;
        int i = horizontalCut.length - 1;
        int j = verticalCut.length - 1;
        while (i >= 0 || j >= 0) {
            if (i >= 0 && (j >= 0 && horizontalCut[i] > verticalCut[j] || j < 0)) {
                res += horizontalCut[i--] * v;
                ++h;
            } else {
                res += verticalCut[j--] * h;
                ++v;
            }
        }
        return res;

    }

    // 3222. 求出硬币游戏的赢家 (Find the Winning Player in Coin Game)
    public String losingPlayer(int x, int y) {
        return Math.min(x, y / 4) % 2 == 0 ? "Bob" : "Alice";
    }

    // 3223. 操作后字符串的最短长度 (Minimum Length of String After Operations)
    public int minimumLength(String s) {
        int[] cnt = new int[26];
        for (char c : s.toCharArray()) {
            ++cnt[c - 'a'];
        }
        int res = 0;
        for (int c : cnt) {
            res += (c - 1) % 2 + 1;
        }
        return res;
    }

    // 3224. 使差值相等的最少数组改动次数 (Minimum Array Changes to Make Differences Equal)
    public int minChanges(int[] nums, int k) {
        int[] diff = new int[k + 2];
        int n = nums.length;
        for (int i = 0; i < n / 2; ++i) {
            int p = nums[i];
            int q = nums[n - i - 1];
            if (p > q) {
                int temp = p;
                p = q;
                q = temp;
            }
            int mx = Math.max(q, k - p);
            ++diff[0];
            --diff[q - p];
            ++diff[q - p + 1];
            ++diff[mx + 1];
        }
        int res = diff[0];
        for (int i = 1; i < k + 1; ++i) {
            diff[i] += diff[i - 1];
            res = Math.min(res, diff[i]);
        }
        return res;

    }

    // 3226. 使两个整数相等的位更改次数 (Number of Bit Changes to Make Two Integers Equal)
    public int minChanges(int n, int k) {
        if ((n & k) != k) {
            return -1;
        }
        return Integer.bitCount(n ^ k);

    }

    // 3227. 字符串元音游戏 (Vowels Game in a String)
    public boolean doesAliceWin(String s) {
        for (char c : s.toCharArray()) {
            if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
                return true;
            }
        }
        return false;
    }

    // 3228. 将 1 移动到末尾的最大操作次数 (Maximum Number of Operations to Move Ones to the End)
    public int maxOperations(String s) {
        int cnt1 = 0;
        int res = 0;
        for (int i = 0; i < s.length() - 1; ++i) {
            if (s.charAt(i) == '1') {
                ++cnt1;
                if (s.charAt(i + 1) == '0') {
                    res += cnt1;
                }
            }
        }
        return res;

    }

    // 3232. 判断是否可以赢得数字游戏 (Find if Digit Game Can Be Won)
    public boolean canAliceWin(int[] nums) {
        int a = 0;
        int b = 0;
        for (int x : nums) {
            if (x < 10) {
                a += x;
            } else {
                b += x;
            }
        }
        return a != b;

    }

    // 3233. 统计不是特殊数字的数字数量 (Find the Count of Numbers Which Are Not Special)
    public int nonSpecialCount(int l, int r) {
        boolean[] isPrime = new boolean[(int) Math.sqrt(r) + 1];
        Arrays.fill(isPrime, true);
        isPrime[1] = false;
        for (int i = 2; i < (int) Math.sqrt(r) + 1; ++i) {
            if (isPrime[i]) {
                for (int j = i * i; j < (int) Math.sqrt(r) + 1; j += i) {
                    isPrime[j] = false;
                }
            }
        }
        return check3233(r, isPrime) - check3233(l - 1, isPrime);

    }

    private int check3233(int x, boolean[] isPrime) {
        int res = 0;
        for (int i = 2; i < (int) Math.sqrt(x) + 1; ++i) {
            if (isPrime[i] && i * i <= x) {
                ++res;
            }
        }
        return x - res;
    }

    // 3235. 判断矩形的两个角落是否可达 (Check if the Rectangle Corner Is Reachable)
    public boolean canReachCorner(int X, int Y, int[][] circles) {
        int n = circles.length;
        Union3235 union = new Union3235(n + 2);
        for (int i = 0; i < n; ++i) {
            // 圆与左边界或上边界相连
            if (circles[i][0] <= circles[i][2] || circles[i][1] + circles[i][2] >= Y) {
                union.union(i, n);
            }
            // 圆与右边界或下边界相连
            if (circles[i][0] + circles[i][2] >= X || circles[i][2] >= circles[i][1]) {
                union.union(i, n + 1);
            }
            for (int j = 0; j < i; ++j) {
                if ((circles[i][0] - circles[j][0]) * (circles[i][0] - circles[j][0]) + (circles[i][1] - circles[j][1])
                        * (circles[i][1] - circles[j][1]) <= (circles[i][2] + circles[j][2])
                                * (circles[i][2] + circles[j][2])) {
                    union.union(i, j);
                }
            }
            if (union.isConnected(n, n + 1)) {
                return false;
            }
        }
        return true;
    }

    public class Union3235 {
        private int[] rank;
        private int[] parent;

        public Union3235(int n) {
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
            int r1 = getRoot(p1);
            int r2 = getRoot(p2);
            if (r1 == r2) {
                return;
            }
            if (rank[r1] < rank[r2]) {
                parent[r1] = r2;
            } else {
                parent[r2] = r1;
                if (rank[r1] < rank[r2]) {
                    ++rank[r1];
                }
            }

        }

    }

}
