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
        // String[] s = { "SLR.L", ".LLXR" };
        // minMoves(s, 9);
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
        int l = String.valueOf(nums[0]).length();
        long res = 0L;
        for (int i = 0; i < l; ++i) {
            int s = 0;
            int[] cnts = new int[10];
            for (int j = 0; j < nums.length; ++j) {
                int b = nums[j] % 10;
                nums[j] /= 10;
                res += s - cnts[b];
                ++cnts[b];
                ++s;
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
        int n = arr.length;
        List<Integer>[] st = new ArrayList[26];
        Arrays.setAll(st, k -> new ArrayList<>());
        int bits = 0;
        for (int i = 0; i < n; ++i) {
            if (arr[i] != '*') {
                int idx = arr[i] - 'a';
                st[idx].add(i);
                bits |= 1 << idx;
            } else {
                int lb = Integer.numberOfTrailingZeros(bits);
                arr[st[lb].remove(st[lb].size() - 1)] = '*';
                if (st[lb].size() == 0) {
                    bits ^= 1 << lb;
                }
            }
        }
        StringBuilder res = new StringBuilder();
        for (char c : arr) {
            if (c != '*') {
                res.append(c);
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
            if (Character.isLetter(c)) {
                res.append(c);
            } else {
                res.setLength(res.length() - 1);
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

    // 3211. 生成不含相邻零的二进制字符串 (Generate Binary Strings Without Adjacent Zeros)
    public List<String> validStrings2(int n) {
        List<String> res = new ArrayList<>();
        int u = (1 << n) - 1;
        for (int i = 0; i < 1 << n; ++i) {
            int c = i ^ u;
            if ((c & (c >> 1)) == 0) {
                String s = Integer.toBinaryString(i);
                while (s.length() < n) {
                    s = "0" + s;
                }
                res.add(s);
            }
        }
        return res;
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

    public int winningPlayerCount(int n, int[][] pick) {
        int[][] cnt = new int[n][11];
        for (int[] p : pick) {
            ++cnt[p[0]][p[1]];
        }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < 11; ++j) {
                if (cnt[i][j] > i) {
                    ++res;
                    break;
                }
            }
        }
        return res;

    }

    // 3239. 最少翻转次数使二进制矩阵回文 I (Minimum Number of Flips to Make Binary Grid
    // Palindromic I)
    public int minFlips(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n / 2; ++j) {
                res += grid[i][j] != grid[i][n - j - 1] ? 1 : 0;
            }
        }
        int res2 = 0;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m / 2; ++i) {
                res2 += grid[i][j] != grid[m - i - 1][j] ? 1 : 0;
            }
        }
        return Math.min(res, res2);
    }

    // 3242. 设计相邻元素求和服务 (Design Neighbor Sum Service)
    class neighborSum {
        private int[][] grid;
        private Map<Integer, int[]> map;
        private int n;

        public neighborSum(int[][] grid) {
            this.grid = grid;
            this.map = new HashMap<>();
            this.n = grid.length;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    map.put(grid[i][j], new int[] { i, j });
                }
            }
        }

        public int adjacentSum(int value) {
            int[] p = map.get(value);
            int x = p[0];
            int y = p[1];
            int res = 0;
            for (int i = Math.max(0, x - 1); i <= Math.min(n - 1, x + 1); ++i) {
                for (int j = Math.max(0, y - 1); j <= Math.min(n - 1, y + 1); ++j) {
                    if (i == x || j == y) {
                        res += grid[i][j];
                    }
                }
            }
            return res - grid[x][y];
        }

        public int diagonalSum(int value) {
            int[] p = map.get(value);
            int x = p[0];
            int y = p[1];
            int res = 0;
            for (int i = Math.max(0, x - 1); i <= Math.min(n - 1, x + 1); ++i) {
                for (int j = Math.max(0, y - 1); j <= Math.min(n - 1, y + 1); ++j) {
                    if (i != x && j != y) {
                        res += grid[i][j];
                    }
                }
            }
            return res;
        }
    }

    // 3243. 新增道路查询后的最短距离 I (Shortest Distance After Road Addition Queries I)
    private List<Integer>[] g3243;
    private int n3243;
    private int[] memo3243;

    public int[] shortestDistanceAfterQueries(int n, int[][] queries) {
        this.n3243 = n;
        this.g3243 = new ArrayList[n];
        Arrays.setAll(g3243, k -> new ArrayList<>());
        for (int i = 0; i < n - 1; ++i) {
            g3243[i].add(i + 1);
        }
        this.memo3243 = new int[n];
        int[] res = new int[queries.length];
        for (int i = 0; i < queries.length; ++i) {
            g3243[queries[i][0]].add(queries[i][1]);
            Arrays.fill(memo3243, -1);
            res[i] = dfs3243(0);
        }
        return res;

    }

    private int dfs3243(int i) {
        if (i == n3243 - 1) {
            return 0;
        }
        if (memo3243[i] != -1) {
            return memo3243[i];
        }
        int res = Integer.MAX_VALUE;
        for (int j : g3243[i]) {
            res = Math.min(res, dfs3243(j) + 1);
        }
        return memo3243[i] = res;
    }

    // 3248. 矩阵中的蛇 (Snake in Matrix)
    public int finalPositionOfSnake(int n, List<String> commands) {
        int x = 0;
        int y = 0;
        for (String c : commands) {
            if ("UP".equals(c)) {
                --x;
            } else if ("RIGHT".equals(c)) {
                ++y;
            } else if ("DOWN".equals(c)) {
                ++x;
            } else {
                --y;
            }
        }
        return x * n + y;

    }

    // 3249. 统计好节点的数目 (Count the Number of Good Nodes)
    private int res3249;
    private List<Integer>[] g3249;

    public int countGoodNodes(int[][] edges) {
        int n = edges.length + 1;
        this.g3249 = new ArrayList[n];
        Arrays.setAll(g3249, k -> new ArrayList<>());
        for (int[] e : edges) {
            g3249[e[0]].add(e[1]);
            g3249[e[1]].add(e[0]);
        }
        dfs3249(0, -1);
        return res3249;
    }

    private int dfs3249(int x, int fa) {
        int pre = -1;
        boolean valid = true;
        int s = 1;
        for (int y : g3249[x]) {
            if (y != fa) {
                int c = dfs3249(y, x);
                if (pre != -1 && c != pre) {
                    valid = false;
                }
                pre = c;
                s += c;
            }
        }
        if (valid) {
            ++res3249;
        }
        return s + 1;
    }

    // 3250. 单调数组对的数目 I (Find the Count of Monotonic Pairs I)
    private int[] nums3250;
    private int n3250;
    private int[][][] memo3250;

    public int countOfPairs(int[] nums) {
        this.nums3250 = nums;
        this.n3250 = nums.length;
        this.memo3250 = new int[n3250][51][51];
        for (int i = 0; i < n3250; ++i) {
            for (int j = 0; j < 51; ++j) {
                Arrays.fill(memo3250[i][j], -1);
            }
        }
        return dfs3250(0, 0, 50);

    }

    private int dfs3250(int i, int j, int k) {
        if (i == n3250) {
            return 1;
        }
        if (memo3250[i][j][k] != -1) {
            return memo3250[i][j][k];
        }
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        for (int x = Math.max(j, nums3250[i] - k); x <= nums3250[i]; ++x) {
            res += dfs3250(i + 1, x, nums3250[i] - x);
            res %= MOD;
        }
        return memo3250[i][j][k] = res;

    }

    // 551. 学生出勤记录 I (Student Attendance Record I)
    public boolean checkRecord(String s) {
        int cntA = 0;
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == 'A') {
                if (++cntA > 1) {
                    return false;
                }
            }
            if (i > 0 && i < s.length() - 1 && s.charAt(i - 1) == s.charAt(i) && s.charAt(i) == s.charAt(i + 1)
                    && s.charAt(i) == 'L') {
                return false;
            }
        }
        return true;
    }

    // 3254. 长度为 K 的子数组的能量值 I (Find the Power of K-Size Subarrays I)
    // 3255. 长度为 K 的子数组的能量值 II (Find the Power of K-Size Subarrays II)
    public int[] resultsArray(int[] nums, int k) {
        int n = nums.length;
        int[] res = new int[n - k + 1];
        Arrays.fill(res, -1);
        int cnt = 1;
        for (int i = 0; i < n; ++i) {
            if (i > 0 && nums[i] - nums[i - 1] == 1) {
                ++cnt;
            }
            if (i >= k && nums[i - k + 1] - nums[i - k] == 1) {
                --cnt;
            }
            if (cnt == k) {
                res[i - k + 1] = nums[i];
            }
        }
        return res;

    }

    public long maximumValueSum(int[][] board) {
        int m = board.length;
        int n = board[0].length;
        long res = (long) -1e10;
        List<int[][]> list = new ArrayList<>();
        for (int i = 0; i < m; ++i) {
            int[][] item = new int[n][2];
            for (int j = 0; j < n; ++j) {
                item[j][0] = board[i][j];
                item[j][1] = j;
            }
            Arrays.sort(item, new Comparator<int[]>() {

                @Override
                public int compare(int[] o1, int[] o2) {
                    return Integer.compare(o2[0], o1[0]);
                }

            });
            int[][] item2 = new int[3][2];
            for (int k = 0; k < 3; ++k) {
                item2[k][0] = item[k][0];
                item2[k][1] = item[k][1];
            }
            list.add(item2);
        }
        for (int i = 0; i < m; ++i) {
            int[][] v1 = list.get(i);

            for (int j = i + 1; j < m; ++j) {
                int[][] v2 = list.get(j);

                for (int k = j + 1; k < m; ++k) {
                    int[][] v3 = list.get(k);

                    for (int x = 0; x < 3; ++x) {
                        int j1 = v1[x][1];
                        search: for (int y = 0; y < 3; ++y) {
                            int j2 = v2[y][1];
                            if (j1 == j2) {
                                continue;
                            }
                            for (int z = 0; z < 3; ++z) {
                                int j3 = v3[z][1];
                                if (j1 != j2 && j1 != j3 && j2 != j3) {
                                    res = Math.max(res, (long) v1[x][0] + v2[y][0] + v3[z][0]);
                                    continue search;
                                }
                            }
                        }
                    }
                }
            }
        }
        return res;
    }

    public int[] getFinalState(int[] nums, int k, int multiplier) {
        int n = nums.length;
        Queue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[1] == o2[1]) {
                    return Integer.compare(o1[0], o2[0]);
                }
                return Integer.compare(o1[1], o2[1]);
            }

        });
        for (int i = 0; i < n; ++i) {
            q.offer(new int[] { i, nums[i] });
        }
        while (k-- > 0) {
            int[] cur = q.poll();
            int i = cur[0];
            int x = cur[1];
            q.offer(new int[] { i, x * multiplier });
        }
        int[] res = new int[n];
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            res[cur[0]] = cur[1];
        }
        return res;

    }

    public int countPairs(int[] nums) {
        int res = 0;
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (nums[i] == nums[j]) {
                    ++res;
                    continue;
                }
                if (check_countPairs(nums[i], nums[j])) {
                    ++res;
                }
            }
        }
        return res;

    }

    private boolean check_countPairs(int x, int y) {
        StringBuilder s1 = new StringBuilder(String.valueOf(x));
        StringBuilder s2 = new StringBuilder(String.valueOf(y));
        int d = 0;
        int i = s1.length() - 1;
        int j = s2.length() - 1;
        int x1 = -1;
        int x2 = -1;
        int y1 = -1;
        int y2 = -1;
        while (i >= 0 || j >= 0) {
            if (i >= 0 && j >= 0) {
                if (s1.charAt(i) != s2.charAt(j)) {
                    ++d;
                    if (d >= 3) {
                        return false;
                    }
                    if (x1 == -1) {
                        x1 = s1.charAt(i) - '0';
                        y1 = s2.charAt(j) - '0';
                    } else {
                        x2 = s1.charAt(i) - '0';
                        y2 = s2.charAt(j) - '0';
                    }
                }
                --i;
                --j;
            } else {
                if (i >= 0) {
                    if (s1.charAt(i) != '0') {
                        ++d;
                        if (d >= 3) {
                            return false;
                        }
                        if (x1 == -1) {
                            x1 = s1.charAt(i) - '0';
                            y1 = 0;
                        } else {
                            x2 = s1.charAt(i) - '0';
                            y2 = 0;
                        }
                    }
                    --i;
                } else {
                    if (s2.charAt(j) != '0') {
                        ++d;
                        if (d >= 3) {
                            return false;
                        }
                        if (x1 == -1) {
                            x1 = 0;
                            y1 = s2.charAt(j) - '0';
                        } else {
                            x2 = 0;
                            y2 = s2.charAt(j) - '0';
                        }
                    }
                    --j;
                }
            }
        }
        return d == 0 || d == 2 && x2 == y1 && x1 == y2;
    }

    // 3270. 求出数字答案 (Find the Key of the Numbers)
    public int generateKey(int num1, int num2, int num3) {
        int res = 0;
        int p = 1;
        while (num1 > 0 && num2 > 0 && num3 > 0) {
            int d = 9;
            d = Math.min(d, num1 % 10);
            num1 /= 10;
            d = Math.min(d, num2 % 10);
            num2 /= 10;
            d = Math.min(d, num3 % 10);
            num3 /= 10;
            res += p * d;
            p *= 10;
        }
        return res;
    }

    // 3271. 哈希分割字符串 (Hash Divided String)
    public String stringHash(String s, int k) {
        int n = s.length();
        StringBuilder res = new StringBuilder();
        int hash = 0;
        for (int i = 0; i < n; ++i) {
            hash += s.charAt(i) - 'a';
            hash %= 26;
            if ((i + 1) % k == 0) {
                res.append((char) ('a' + hash));
                hash = 0;
            }
        }
        return res.toString();

    }

    // 3274. 检查棋盘方格颜色是否相同 (Check if Two Chessboard Squares Have the Same Color)
    public boolean checkTwoChessboards(String coordinate1, String coordinate2) {
        return Math.abs(coordinate1.charAt(0) - coordinate2.charAt(0))
                % 2 == Math.abs(coordinate1.charAt(1) - coordinate2.charAt(1)) % 2;
    }

    // 3275. 第 K 近障碍物查询 (K-th Nearest Obstacle Queries)
    public int[] resultsArray(int[][] queries, int k) {
        Queue<Integer> q = new PriorityQueue<>(new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(o2, o1);
            }

        });
        int[] res = new int[queries.length];
        for (int i = 0; i < queries.length; ++i) {
            q.offer(Math.abs(queries[i][0]) + Math.abs(queries[i][1]));
            if (q.size() > k) {
                q.poll();
            }
            res[i] = q.size() == k ? q.peek() : -1;
        }
        return res;
    }

    // 3276. 选择矩阵中单元格的最大得分 (Select Cells in Grid With Maximum Score)
    private List<int[]> list3276;
    private int l3276;
    private int[][] memo3276;
    private int u3276;

    public int maxScore3276(List<List<Integer>> grid) {
        int m = grid.size();
        int n = grid.get(0).size();
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                map.merge(grid.get(i).get(j), 1 << i, (a, b) -> a | b);
            }
        }
        List<int[]> list = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            list.add(new int[] { entry.getKey(), entry.getValue() });
        }
        this.list3276 = list;
        this.l3276 = list.size();
        this.memo3276 = new int[l3276][1 << m];
        for (int i = 0; i < l3276; ++i) {
            Arrays.fill(memo3276[i], -1);
        }
        this.u3276 = (1 << m) - 1;
        return dfs3276(0, 0);
    }

    private int dfs3276(int i, int j) {
        if (i == l3276) {
            return 0;
        }
        if (memo3276[i][j] != -1) {
            return memo3276[i][j];
        }
        int res = dfs3276(i + 1, j);
        for (int c = (u3276 ^ j) & list3276.get(i)[1]; c != 0; c &= c - 1) {
            res = Math.max(res, dfs3276(i + 1, j | (1 << Integer.numberOfTrailingZeros(c))) + list3276.get(i)[0]);
        }
        return memo3276[i][j] = res;
    }

    // 3280. 将日期转换为二进制表示 (Convert Date to Binary)
    public String convertDateToBinary(String date) {
        String[] arr = date.split("-");
        for (int i = 0; i < arr.length; ++i) {
            arr[i] = Integer.toBinaryString(Integer.parseInt(arr[i]));
        }
        return String.join("-", arr);

    }

    // 3281. 范围内整数的最大得分 (Maximize Score of Numbers in Ranges)
    public int maxPossibleScore(int[] start, int d) {
        Arrays.sort(start);
        int n = start.length;
        int left = 0;
        int right = start[n - 1] + d - start[0];
        int res = 0;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (binarySearch3281(mid, start, d)) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;

    }

    private boolean binarySearch3281(int target, int[] start, int d) {
        long left = start[0];
        for (int i = 0; i < start.length - 1; ++i) {
            if (left + target > start[i + 1] + d) {
                return false;
            }
            left = Math.max(left + target, start[i + 1]);
        }
        return true;
    }

    // 3282. 到达数组末尾的最大得分 (Reach End of Array With Max Score)
    public long findMaximumScore(List<Integer> nums) {
        long res = 0L;
        int mx = 0;
        for (int x : nums) {
            mx = Math.max(mx, x);
            res += mx;
        }
        return res - mx;
    }

    // 3283. 吃掉所有兵需要的最多移动次数 (Maximum Number of Moves to Kill All Pawns)
    private int[][] positions3283;
    private int[][] directions3283 = { { 1, -2 }, { 1, 2 }, { -1, -2 }, { -1, 2 }, { 2, -1 }, { 2, 1 }, { -2, 1 },
            { -2, -1 } };
    private int[][] distance3283;
    private int n3283;
    private int[][] memo3283;
    private int u3283;
    private Set<Integer> set3283;

    public int maxMoves(int kx, int ky, int[][] positions) {
        this.n3283 = positions.length;
        this.distance3283 = new int[n3283 + 1][2500];
        this.positions3283 = positions;
        this.set3283 = new HashSet<>();
        for (int i = 0; i < n3283; ++i) {
            set3283.add(positions[i][0] * 50 + positions[i][1]);
        }
        for (int i = 0; i < n3283; ++i) {
            cal3283(i, positions[i][0], positions[i][1]);
        }
        cal3283(n3283, kx, ky);
        this.u3283 = (1 << n3283) - 1;
        this.memo3283 = new int[51][u3283];
        for (int i = 0; i < 51; ++i) {
            Arrays.fill(memo3283[i], -1);
        }
        return dfs3283(n3283, 0);

    }

    private int dfs3283(int i, int mask) {
        if (mask == u3283) {
            return 0;
        }
        if (memo3283[i][mask] != -1) {
            return memo3283[i][mask];
        }
        int res = Integer.bitCount(mask) % 2 == 0 ? 0 : Integer.MAX_VALUE;
        for (int c = u3283 ^ mask; c != 0; c &= c - 1) {
            int lb = Integer.numberOfTrailingZeros(c);
            int add = distance3283[i][positions3283[lb][0] * 50 + positions3283[lb][1]];
            if (Integer.bitCount(mask) % 2 == 0) {
                res = Math.max(res, dfs3283(lb, mask | (1 << lb)) + add);
            } else {
                res = Math.min(res, dfs3283(lb, mask | (1 << lb)) + add);
            }
        }
        return memo3283[i][mask] = res;

    }

    private void cal3283(int p, int startX, int startY) {
        boolean[][] vis = new boolean[50][50];
        vis[startX][startY] = true;
        int step = 0;
        Queue<int[]> q = new LinkedList<>();
        q.add(new int[] { startX, startY });
        int cnt = p < n3283 ? 1 : 0;
        while (!q.isEmpty() && cnt < n3283) {
            int size = q.size();
            ++step;
            for (int i = 0; i < size; ++i) {
                int[] cur = q.poll();
                int x = cur[0];
                int y = cur[1];
                for (int[] d : directions3283) {
                    int dx = d[0];
                    int dy = d[1];
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < 50 && ny >= 0 && ny < 50 && !vis[nx][ny]) {
                        vis[nx][ny] = true;
                        distance3283[p][nx * 50 + ny] = step;
                        if (set3283.contains(nx * 50 + ny)) {
                            if (++cnt == n3283) {
                                return;
                            }
                        }
                        q.offer(new int[] { nx, ny });
                    }
                }
            }
        }
    }

    // 3285. 找到稳定山的下标 (Find Indices of Stable Mountains)
    public List<Integer> stableMountains(int[] height, int threshold) {
        int n = height.length;
        List<Integer> res = new ArrayList<>();
        for (int i = 1; i < n; ++i) {
            if (height[i - 1] > threshold) {
                res.add(i);
            }
        }
        return res;
    }

    // 3286. 穿越网格图的安全路径 (Find a Safe Walk Through a Grid) -- 0-1bfs
    public boolean findSafeWalk(List<List<Integer>> grid, int health) {
        int m = grid.size();
        int n = grid.get(0).size();

        Deque<int[]> q = new ArrayDeque<>();
        int[][] directions = { { 0, -1 }, { 0, 1 }, { 1, 0 }, { -1, 0 } };
        int[][] dis = new int[m][n];
        for (int i = 0; i < m; ++i) {
            Arrays.fill(dis[i], Integer.MAX_VALUE);
        }
        dis[0][0] = grid.get(0).get(0);
        q.offer(new int[] { 0, 0 });
        while (true) {
            int[] cur = q.pollFirst();
            int x = cur[0];
            int y = cur[1];
            if (dis[x][y] >= health) {
                return false;
            }
            if (x == m - 1 && y == n - 1) {
                return true;
            }
            for (int[] nxt : directions) {
                int dx = nxt[0];
                int dy = nxt[1];
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && dis[x][y] + grid.get(nx).get(ny) < dis[nx][ny]) {
                    dis[nx][ny] = dis[x][y] + grid.get(nx).get(ny);
                    if (grid.get(nx).get(ny) == 0) {
                        q.offerFirst(new int[] { nx, ny });
                    } else {
                        q.offerLast(new int[] { nx, ny });
                    }
                }
            }
        }

    }

    // 3289. 数字小镇中的捣蛋鬼 (The Two Sneaky Numbers of Digitville)
    public int[] getSneakyNumbers(int[] nums) {
        int n = nums.length - 2;
        int xor = n ^ (n + 1);
        for (int i = 0; i < n + 2; ++i) {
            xor ^= i ^ nums[i];
        }
        int lb = Integer.numberOfTrailingZeros(xor);
        int[] res = new int[2];
        for (int i = 0; i < n + 2; ++i) {
            if (i < n) {
                res[i >> lb & 1] ^= i;
            }
            res[nums[i] >> lb & 1] ^= nums[i];
        }
        return res;
    }

    // 3290. 最高乘法得分 (Maximum Multiplication Score)
    private long[][] memo3290;
    private int[] a3290;
    private int[] b3290;
    private int n3290;

    public long maxScore(int[] a, int[] b) {
        this.a3290 = a;
        this.b3290 = b;
        this.n3290 = b.length;
        this.memo3290 = new long[4][n3290];
        for (int i = 0; i < 4; ++i) {
            Arrays.fill(memo3290[i], (long) 1e11);
        }
        return dfs3290(0, 0);

    }

    private long dfs3290(int i, int j) {
        if (i == 4) {
            return 0L;
        }
        if (n3290 - j < 4 - i) {
            return (long) -1e12;
        }
        if (memo3290[i][j] != (long) 1e11) {
            return memo3290[i][j];
        }
        return memo3290[i][j] = Math.max(dfs3290(i, j + 1), dfs3290(i + 1, j + 1) + (long) a3290[i] * b3290[j]);
    }

    // 3291. 形成目标字符串需要的最少字符串数 I (Minimum Number of Valid Strings to Form Target I)
    private int[] memo3291;
    private int n3291;
    private Trie3291 trie3291;
    private String target3291;

    public int minValidStrings(String[] words, String target) {
        this.trie3291 = new Trie3291();
        for (String s : words) {
            trie3291.insert(s);
        }
        this.n3291 = target.length();
        this.target3291 = target;
        this.memo3291 = new int[n3291];
        Arrays.fill(memo3291, -1);
        int res = dfs3291(0);
        return res <= n3291 ? res : -1;
    }

    private int dfs3291(int i) {
        if (i == n3291) {
            return 0;
        }
        if (memo3291[i] != -1) {
            return memo3291[i];
        }
        int res = (int) 1e9;
        Trie3291 node = trie3291;
        for (int j = i; j < n3291; ++j) {
            if (node.children[target3291.charAt(j) - 'a'] == null) {
                break;
            }
            node = node.children[target3291.charAt(j) - 'a'];
            res = Math.min(res, dfs3291(j + 1) + 1);
        }
        return memo3291[i] = res;
    }

    public class Trie3291 {
        private Trie3291[] children;

        public Trie3291() {
            this.children = new Trie3291[26];
        }

        public void insert(String s) {
            Trie3291 node = this;
            for (char c : s.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new Trie3291();
                }
                node = node.children[index];
            }
        }

    }

    // 3295. 举报垃圾信息 (Report Spam Message)
    public boolean reportSpam(String[] message, String[] bannedWords) {
        Set<String> s = Arrays.stream(bannedWords).collect(Collectors.toSet());
        int cnt = 0;
        for (String m : message) {
            if (s.contains(m)) {
                if (++cnt >= 2) {
                    return true;
                }
            }
        }
        return false;

    }

    // 3296. 移山所需的最少秒数 (Minimum Number of Seconds to Make Mountain Height Zero)
    public long minNumberOfSeconds(int mountainHeight, int[] workerTimes) {
        long left = 1L;
        long right = (long) 1e16;
        long res = -1L;
        while (left <= right) {
            long mid = left + ((right - left) >> 1L);
            if (check_3296(mid, mountainHeight, workerTimes)) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    private boolean check_3296(long target, int mountainHeight, int[] workerTimes) {
        long h = 0L;
        for (int w : workerTimes) {
            h += check2_3296(target, w, mountainHeight);
            if (h >= mountainHeight) {
                return true;
            }
        }
        return false;
    }

    private long check2_3296(long target, int w, int mountainHeight) {
        long left = 0L;
        long right = mountainHeight;
        long res = 0L;
        while (left <= right) {
            long mid = left + ((right - left) >> 1);
            if (w * (long) (1 + mid) * mid / 2 <= target) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    // 3297. 统计重新排列后包含另一个字符串的子字符串数目 I (Count Substrings That Can Be Rearranged to
    // Contain a String I)
    // 3298. 统计重新排列后包含另一个字符串的子字符串数目 II (Count Substrings That Can Be Rearranged to
    // Contain a String II)
    public long validSubstringCount(String word1, String word2) {
        int n = word1.length();
        long res = 0L;
        int[] cnt = new int[26];
        int less = 0;
        for (char c : word2.toCharArray()) {
            if (--cnt[c - 'a'] == -1) {
                --less;
            }
        }
        int left = 0;
        for (int right = 0; right < n; ++right) {
            if (++cnt[word1.charAt(right) - 'a'] == 0) {
                ++less;
            }
            while (less == 0) {
                if (--cnt[word1.charAt(left) - 'a'] == -1) {
                    --less;
                }
                ++left;
            }
            res += left;
        }
        return res;

    }

    // 2306. 公司命名 (Naming a Company)
    public long distinctNames(String[] ideas) {
        long res = 0L;
        Set<String>[] set = new HashSet[26];
        Arrays.setAll(set, k -> new HashSet<>());
        for (String idea : ideas) {
            int index = idea.charAt(0) - 'a';
            set[index].add(idea.substring(1));
        }
        for (int i = 0; i < 26; ++i) {
            if (set[i].isEmpty()) {
                continue;
            }
            for (int j = i + 1; j < 26; ++j) {
                if (set[j].isEmpty()) {
                    continue;
                }
                int cnt = 0;
                for (String str : set[j]) {
                    if (set[i].contains(str)) {
                        ++cnt;
                    }
                }
                res += (long) (set[i].size() - cnt) * (set[j].size() - cnt);
            }
        }
        return res * 2L;

    }

    // 3300. 替换为数位和以后的最小元素 (Minimum Element After Replacement With Digit Sum)
    public int minElement(int[] nums) {
        int res = Integer.MAX_VALUE;
        for (int x : nums) {
            int cur = 0;
            while (x != 0) {
                cur += x % 10;
                x /= 10;
            }
            res = Math.min(res, cur);
        }
        return res;

    }

    // 3301. 高度互不相同的最大塔高和 (Maximize the Total Height of Unique Towers)
    public long maximumTotalSum(int[] maximumHeight) {
        Arrays.sort(maximumHeight);
        long res = 0L;
        int mx = Integer.MAX_VALUE;
        for (int i = maximumHeight.length - 1; i >= 0; --i) {
            mx = Math.min(mx - 1, maximumHeight[i]);
            if (mx == 0) {
                return -1L;
            }
            res += mx;
        }
        return res;

    }

    // 3304. 找出第 K 个字符 I (Find the K-th Character in String Game I)
    public char kthCharacter(int k) {
        String s = "a";
        while (s.length() < k) {
            StringBuilder builder = new StringBuilder(s);
            for (int i = 0; i < builder.length(); ++i) {
                builder.setCharAt(i, (char) (((builder.charAt(i) - 'a' + 1) % 26) + 'a'));
            }
            s += builder.toString();
        }
        return s.charAt(k - 1);

    }

    // 2565. 最少得分子序列 (Subsequence With the Minimum Score)
    public int minimumScore(String S, String T) {
        char[] s = S.toCharArray();
        char[] t = T.toCharArray();
        int n = s.length;
        int m = t.length;
        int[] suf = new int[n + 1];
        suf[n] = m;
        int j = m - 1;
        for (int i = n - 1; i >= 0; --i) {
            if (s[i] == t[j]) {
                --j;
            }
            if (j < 0) {
                return 0;
            }
            suf[i] = j + 1;
        }
        int res = suf[0];
        j = 0;
        for (int i = 0; i < n; ++i) {
            if (s[i] == t[j]) {
                ++j;
                res = Math.min(res, suf[i + 1] - j);
            }
        }
        return res;

    }

    // 3302. 字典序最小的合法序列 (Find the Lexicographically Smallest Valid Sequence)
    public int[] validSequence(String word1, String word2) {
        char[] s = word1.toCharArray();
        char[] t = word2.toCharArray();
        int n = s.length;
        int m = t.length;

        int[] suf = new int[n + 1];
        suf[n] = m;
        int j = m - 1;
        for (int i = n - 1; i >= 0; i--) {
            if (j >= 0 && s[i] == t[j]) {
                j--;
            }
            suf[i] = j + 1;
        }

        int[] ans = new int[m];
        boolean changed = false; // 是否修改过
        j = 0;
        for (int i = 0; i < n; i++) {
            if (s[i] == t[j] || !changed && suf[i + 1] <= j + 1) {
                if (s[i] != t[j]) {
                    changed = true;
                }
                ans[j++] = i;
                if (j == m) {
                    return ans;
                }
            }
        }
        return new int[] {};

    }

    // 3303. 第一个几乎相等子字符串的下标 (Find the Occurrence of First Almost Equal Substring)
    public int minStartingIndex(String s, String pattern) {
        int[] preZ = calcZ3303(pattern + s);
        int[] sufZ = calcZ3303(rev3303(pattern) + rev3303(s));
        // 可以不反转 sufZ，下面写 sufZ[sufZ.length - i]
        int n = s.length();
        int m = pattern.length();
        for (int i = m; i <= n; i++) {
            if (preZ[i] + sufZ[sufZ.length - i] >= m - 1) {
                return i - m;
            }
        }
        return -1;
    }

    private int[] calcZ3303(String S) {
        char[] s = S.toCharArray();
        int n = s.length;
        int[] z = new int[n];
        int boxL = 0;
        int boxR = 0; // z-box 左右边界
        for (int i = 1; i < n; i++) {
            if (i <= boxR) {
                z[i] = Math.min(z[i - boxL], boxR - i + 1);
            }
            while (i + z[i] < n && s[z[i]] == s[i + z[i]]) {
                boxL = i;
                boxR = i + z[i];
                z[i]++;
            }
        }
        return z;
    }

    private String rev3303(String s) {
        return new StringBuilder(s).reverse().toString();

    }

    // 3305. 元音辅音字符串计数 I (Count of Substrings Containing Every Vowel and K
    // Consonants I)
    // 3306. 元音辅音字符串计数 II (Count of Substrings Containing Every Vowel and K
    // Consonants II)
    public long countOfSubstrings(String word, int k) {
        return check3306(word, k) - check3306(word, k + 1);
    }

    private long check3306(String word, int k) {
        long res = 0L;
        Map<Character, Integer> cnt = new HashMap<>();
        int left = 0;
        int consonant = 0;
        for (int right = 0; right < word.length(); ++right) {
            if (isVowel3306(word.charAt(right))) {
                cnt.merge(word.charAt(right), 1, Integer::sum);
            } else {
                ++consonant;
            }
            while (cnt.size() == 5 && consonant >= k) {
                if (isVowel3306(word.charAt(left))) {
                    cnt.merge(word.charAt(left), -1, Integer::sum);
                    if (cnt.get(word.charAt(left)) == 0) {
                        cnt.remove(word.charAt(left));
                    }
                } else {
                    --consonant;
                }
                ++left;
            }
            res += left;
        }
        return res;
    }

    private boolean isVowel3306(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';

    }

    // 3307. 找出第 K 个字符 II (Find the K-th Character in String Game II)
    public char kthCharacter(long k, int[] operations) {
        int n = Math.min(operations.length, 60);
        return dfs3307(k, operations, n - 1);

    }

    private char dfs3307(long k, int[] operations, int i) {
        if (i < 0) {
            return 'a';
        }
        if (k <= 1L << i) {
            return dfs3307(k, operations, i - 1);
        }
        char res = dfs3307(k - (1L << i), operations, i - 1);
        res = (char) ('a' + (res - 'a' + operations[i]) % 26);
        return res;
    }

    // 3314. 构造最小位运算数组 I (Construct the Minimum Bitwise Array I)
    // 3315. 构造最小位运算数组 II (Construct the Minimum Bitwise Array II)
    public int[] minBitwiseArray(List<Integer> nums) {
        int[] res = new int[nums.size()];
        for (int i = 0; i < nums.size(); ++i) {
            if (nums.get(i) == 2) {
                res[i] = -1;
            } else {
                int t = ~nums.get(i);
                res[i] = nums.get(i) ^ ((t & -t) >> 1);
            }
        }
        return res;

    }

    // 3316. 从原字符串里进行删除操作的最多次数 (Find Maximum Removals From Source String)
    private String s3316;
    private String p3316;
    private int n3316;
    private int m3316;
    private boolean[] t3316;
    private int[][] memo3316;

    public int maxRemovals(String source, String pattern, int[] targetIndices) {
        this.s3316 = source;
        this.p3316 = pattern;
        this.n3316 = source.length();
        this.m3316 = pattern.length();
        this.t3316 = new boolean[n3316];
        for (int idx : targetIndices) {
            t3316[idx] = true;
        }
        this.memo3316 = new int[n3316][m3316];
        for (int i = 0; i < n3316; ++i) {
            Arrays.fill(memo3316[i], -1);
        }
        return dfs3316(0, 0);

    }

    private int dfs3316(int i, int j) {
        if (i == n3316 || j == m3316) {
            if (i == n3316 && j == m3316) {
                return 0;
            }
            if (j == m3316) {
                return dfs3316(i + 1, j) + (t3316[i] ? 1 : 0);
            }
            return Integer.MIN_VALUE;
        }
        if (memo3316[i][j] != -1) {
            return memo3316[i][j];
        }
        if (!t3316[i]) {
            return dfs3316(i + 1, j + (s3316.charAt(i) == p3316.charAt(j) ? 1 : 0));
        }
        int res = dfs3316(i + 1, j) + 1;
        if (s3316.charAt(i) == p3316.charAt(j)) {
            res = Math.max(res, dfs3316(i + 1, j + 1));
        }
        return memo3316[i][j] = res;
    }

    // 1884. 鸡蛋掉落-两枚鸡蛋 (Egg Drop With 2 Eggs and N Floors)
    private int[] memo1884;

    public int twoEggDrop(int n) {
        this.memo1884 = new int[n + 1];
        Arrays.fill(memo1884, -1);
        return dfs1884(n);

    }

    private int dfs1884(int i) {
        if (i == 0) {
            return 0;
        }
        if (memo1884[i] != -1) {
            return memo1884[i];
        }
        int res = Integer.MAX_VALUE;
        for (int j = 1; j <= i; ++j) {
            res = Math.min(res, Math.max(j, dfs1884(i - j) + 1));
        }
        return memo1884[i] = res;
    }

    // 3319. 第 K 大的完美二叉子树的大小 (K-th Largest Perfect Subtree Size in Binary Tree)
    private List<Integer> list3319;

    public int kthLargestPerfectSubtree(TreeNode root, int k) {
        this.list3319 = new ArrayList<>();
        dfs3319(root);
        if (list3319.size() < k) {
            return -1;
        }
        Collections.sort(list3319, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(o2, o1);
            }

        });
        return list3319.get(k - 1);
    }

    private int[] dfs3319(TreeNode root) {
        if (root == null) {
            return new int[] { 0, 0 };
        }
        int[] left = dfs3319(root.left);
        int[] right = dfs3319(root.right);
        if (left[0] == -1 || right[0] == -1 || left[1] != right[1]) {
            return new int[] { -1, -1 };
        }
        list3319.add(left[0] + right[0] + 1);
        return new int[] { left[0] + right[0] + 1, left[1] + 1 };

    }

    // 3320. 统计能获胜的出招序列数 (Count The Number of Winning Sequences)
    private int n3320;
    private String s3320;
    private int[][][] memo3320;

    public int countWinningSequences(String s) {
        this.n3320 = s.length();
        this.s3320 = s;
        this.memo3320 = new int[n3320][n3320 * 2][4];
        for (int i = 0; i < n3320; ++i) {
            for (int j = 0; j < n3320 * 2; ++j) {
                Arrays.fill(memo3320[i][j], -1);
            }
        }
        return dfs3320(0, 0, 3);

    }

    private int dfs3320(int i, int j, int k) {
        if (i == n3320) {
            return j > 0 ? 1 : 0;
        }
        if (n3320 - i + j <= 0) {
            return 0;
        }
        if (memo3320[i][j + n3320][k] != -1) {
            return memo3320[i][j + n3320][k];
        }
        int res = 0;
        int iVal = s3320.charAt(i) == 'F' ? 0 : (s3320.charAt(i) == 'E' ? 1 : 2);
        for (int x = 0; x <= 2; ++x) {
            if (x == k) {
                continue;
            }
            if (x == iVal) {
                res += dfs3320(i + 1, j, x);
            } else {
                res += dfs3320(i + 1, j + ((x + 1) % 3 == iVal ? 1 : -1), x);
            }
            final int MOD = (int) (1e9 + 7);
            res %= MOD;
        }
        return memo3320[i][j + n3320][k] = res;
    }

    // 910. 最小差值 II (Smallest Range II)
    public int smallestRangeII(int[] nums, int k) {
        Arrays.sort(nums);
        int n = nums.length;
        int res = nums[n - 1] - nums[0];
        for (int i = 1; i < n; ++i) {
            int mx = Math.max(nums[n - 1] - k, nums[i - 1] + k);
            int mi = Math.min(nums[0] + k, nums[i] - k);
            res = Math.min(res, mx - mi);
        }
        return res;

    }

    // 3324. 出现在屏幕上的字符串序列 (Find the Sequence of Strings Appeared on the Screen)
    public List<String> stringSequence(String target) {
        List<String> res = new ArrayList<>();
        for (char t : target.toCharArray()) {
            String pre = res.isEmpty() ? "" : res.get(res.size() - 1);
            for (char c = 'a'; c <= t; ++c) {
                res.add(pre + String.valueOf(c));
            }
        }
        return res;
    }

    // 3325. 字符至少出现 K 次的子字符串 I (Count Substrings With K-Frequency Characters I)
    public int numberOfSubstrings(String s, int k) {
        int res = 0;
        int[] cnt = new int[26];
        int left = 0;
        for (char c : s.toCharArray()) {
            ++cnt[c - 'a'];
            while (cnt[c - 'a'] >= k) {
                --cnt[s.charAt(left++) - 'a'];
            }
            res += left;
        }
        return res;
    }

    // 3326. 使数组非递减的最少除法操作次数 (Minimum Division Operations to Make Array Non
    // Decreasing)
    public int minOperations3326(int[] nums) {
        int res = 0;
        int max = Arrays.stream(nums).max().getAsInt();
        int[] p = new int[max + 1];
        Arrays.fill(p, -1);
        for (int i = 2; i <= max; ++i) {
            if (p[i] == -1) {
                for (long j = (long) i * i; j <= max; j += i) {
                    if (p[(int) j] == -1) {
                        p[(int) j] = i;
                    }
                }
            }
        }
        for (int i = nums.length - 2; i >= 0; --i) {
            int pre = nums[i];
            while (pre > nums[i + 1]) {
                pre = p[pre];
                ++res;
            }
            if (pre == -1) {
                return -1;
            }
            nums[i] = pre;
        }
        return res;

    }

    // 3180. 执行操作可获得的最大总奖励 I (Maximum Total Reward Using Operations I)
    private Map<Integer, Integer> memo3180;
    private List<Integer> list3180;
    private int n3180;

    public int maxTotalReward(int[] rewardValues) {
        this.memo3180 = new HashMap<>();
        Set<Integer> s = Arrays.stream(rewardValues).boxed().collect(Collectors.toSet());
        list3180 = new ArrayList<>(s);
        Collections.sort(list3180);
        this.n3180 = list3180.size();
        return dfs3180(0, 0);
    }

    private int dfs3180(int i, int j) {
        if (i == n3180) {
            return j;
        }
        int m = (j << 10) | i;
        if (memo3180.get(m) != null) {
            return memo3180.get(m);
        }
        int res = dfs3180(i + 1, j);
        if (list3180.get(i) > j) {
            res = Math.max(res, dfs3180(i + 1, j + list3180.get(i)));
        }
        memo3180.put(m, res);
        return res;
    }

    // 3330. 找到初始输入字符串 I (Find the Original Typed String I)
    public int possibleStringCount(String word) {
        int res = 1;
        int n = word.length();
        int i = 0;
        while (i < n) {
            int j = i;
            while (j < n && word.charAt(i) == word.charAt(j)) {
                ++j;
            }
            res += j - i - 1;
            i = j;
        }
        return res;

    }

    // 3331. 修改后子树的大小 (Find Subtree Sizes After Changes)
    private int n3331;
    private List<Integer>[] g3331;
    private List<Integer>[] st3331;
    private String s3331;
    private List<Integer>[] g2_3331;
    private int[] res3331;

    public int[] findSubtreeSizes(int[] parent, String s) {
        this.s3331 = s;
        this.n3331 = s.length();
        this.g3331 = new ArrayList[n3331];
        Arrays.setAll(g3331, k -> new ArrayList<>());
        for (int i = 1; i < n3331; ++i) {
            g3331[parent[i]].add(i);
        }
        this.st3331 = new ArrayList[26];
        Arrays.setAll(st3331, k -> new ArrayList<>());
        this.g2_3331 = new ArrayList[n3331];
        Arrays.setAll(g2_3331, k -> new ArrayList<>());
        dfs3331(0, -1);
        this.res3331 = new int[n3331];
        dfs2_3331(0);
        return res3331;
    }

    private int dfs2_3331(int x) {
        for (int y : g2_3331[x]) {
            res3331[x] += dfs2_3331(y);
        }
        return ++res3331[x];
    }

    private void dfs3331(int x, int fa) {
        int index = s3331.charAt(x) - 'a';
        if (!st3331[index].isEmpty()) {
            g2_3331[st3331[index].get(st3331[index].size() - 1)].add(x);
        } else if (fa != -1) {
            g2_3331[fa].add(x);
        }
        st3331[index].add(x);
        for (int y : g3331[x]) {
            dfs3331(y, x);
        }
        st3331[index].remove(st3331[index].size() - 1);
    }

    // 3332. 旅客可以得到的最多点数 (Maximum Points Tourist Can Earn)
    private int[][] memo3332;
    private int n3332;
    private int k3332;
    private int[][] stayScore3332;
    private int[][] travelScore3332;

    public int maxScore(int n, int k, int[][] stayScore, int[][] travelScore) {
        this.n3332 = n;
        this.k3332 = k;
        this.stayScore3332 = stayScore;
        this.travelScore3332 = travelScore;
        this.memo3332 = new int[n][k];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(memo3332[i], -1);
        }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            res = Math.max(res, dfs3332(i, 0));
        }
        return res;

    }

    private int dfs3332(int i, int j) {
        if (j == k3332) {
            return 0;
        }
        if (memo3332[i][j] != -1) {
            return memo3332[i][j];
        }
        int res = 0;
        for (int x = 0; x < n3332; ++x) {
            if (x == i) {
                res = Math.max(res, dfs3332(x, j + 1) + stayScore3332[j][x]);
            } else {
                res = Math.max(res, dfs3332(x, j + 1) + travelScore3332[i][x]);
            }
        }
        return memo3332[i][j] = res;
    }

    // 3259. 超级饮料的最大强化能量 (Maximum Energy Boost From Two Drinks)
    private int n3259;
    private int[] energyDrinkA3259;
    private int[] energyDrinkB3259;
    private long[][] memo3259;

    public long maxEnergyBoost(int[] energyDrinkA, int[] energyDrinkB) {
        this.n3259 = energyDrinkA.length;
        this.energyDrinkA3259 = energyDrinkA;
        this.energyDrinkB3259 = energyDrinkB;
        this.memo3259 = new long[n3259][2];
        return Math.max(dfs3259(0, 0), dfs3259(0, 1));

    }

    private long dfs3259(int i, int j) {
        if (i >= n3259) {
            return 0;
        }
        if (memo3259[i][j] != 0) {
            return memo3259[i][j];
        }
        return memo3259[i][j] = Math.max(dfs3259(i + 1, j), dfs3259(i + 2, j ^ 1))
                + (j == 0 ? energyDrinkA3259[i] : energyDrinkB3259[i]);
    }

    // 3340. 检查平衡字符串 (Check Balanced String)
    public boolean isBalanced(String num) {
        int[] s = new int[2];
        for (int i = 0; i < num.length(); ++i) {
            s[i & 1] += num.charAt(i) - '0';
        }
        return s[0] == s[1];

    }

    // 3341. 到达最后一个房间的最少时间 I (Find Minimum Time to Reach Last Room I)
    public int minTimeToReach(int[][] moveTime) {
        int m = moveTime.length;
        int n = moveTime[0].length;
        int[][] dis = new int[m][n];
        for (int i = 0; i < m; ++i) {
            Arrays.fill(dis[i], Integer.MAX_VALUE);
        }
        dis[0][0] = 0;
        Queue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[2], o2[2]);
            }

        });
        int[][] dirs = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        q.offer(new int[] { 0, 0, 0 });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int y = cur[1];
            int t = cur[2];
            if (t > dis[x][y]) {
                continue;
            }
            if (x == m - 1 && y == n - 1) {
                return t;
            }
            for (int[] d : dirs) {
                int nx = x + d[0];
                int ny = y + d[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    int dt = Math.max(0, moveTime[nx][ny] - t) + 1;
                    if (t + dt < dis[nx][ny]) {
                        dis[nx][ny] = t + dt;
                        q.offer(new int[] { nx, ny, t + dt });
                    }
                }
            }
        }
        return -1;
    }

    // 3342. 到达最后一个房间的最少时间 II (Find Minimum Time to Reach Last Room II)
    public int minTimeToReach3342(int[][] moveTime) {
        int m = moveTime.length;
        int n = moveTime[0].length;
        int[][] dis = new int[m][n];
        for (int i = 0; i < m; ++i) {
            Arrays.fill(dis[i], Integer.MAX_VALUE);
        }
        dis[0][0] = 0;
        Queue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[2], o2[2]);
            }

        });
        q.offer(new int[] { 0, 0, 0, 0 });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int y = cur[1];
            int t = cur[2];
            int xor = cur[3];
            if (t > dis[x][y]) {
                continue;
            }
            if (x == m - 1 && y == n - 1) {
                return t;
            }
            int[][] dirs = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
            for (int[] dir : dirs) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    int nd = Math.max(moveTime[nx][ny], t) + 1 + xor;
                    if (nd < dis[nx][ny]) {
                        dis[nx][ny] = nd;
                        q.offer(new int[] { nx, ny, nd, xor ^ 1 });
                    }
                }
            }
        }
        return -1;

    }

    // 3345. 最小可整除数位乘积 I (Smallest Divisible Digit Product I)
    public int smallestNumber(int n, int t) {
        while (true) {
            if (check3345(n, t)) {
                return n;
            }
            ++n;
        }
    }

    private boolean check3345(int n, int t) {
        int res = 1;
        while (n != 0) {
            res *= n % 10;
            n /= 10;
        }
        return res % t == 0;
    }

    // 3346. 执行操作后元素的最高频率 I (Maximum Frequency of an Element After Performing
    // Operations I)
    // 3347. 执行操作后元素的最高频率 II (Maximum Frequency of an Element After Performing
    // Operations II)
    public int maxFrequency(int[] nums, int k, int numOperations) {
        TreeMap<Integer, Integer> diff = new TreeMap<>();
        Map<Integer, Integer> cnt = new HashMap<>();
        for (int x : nums) {
            cnt.merge(x, 1, Integer::sum);
            diff.putIfAbsent(x, 0);
            diff.merge(x - k, 1, Integer::sum);
            diff.merge(x + k + 1, -1, Integer::sum);
        }
        int res = 0;
        int sumD = 0;
        for (Map.Entry<Integer, Integer> entry : diff.entrySet()) {
            sumD += entry.getValue();
            res = Math.max(res, Math.min(sumD, cnt.getOrDefault(entry.getKey(), 0) + numOperations));
        }
        return res;

    }

    // 3349. 检测相邻递增子数组 I (Adjacent Increasing Subarrays Detection I)
    public boolean hasIncreasingSubarrays(List<Integer> nums, int k) {
        int n = nums.size();
        int cnt = 1;
        boolean[] a = new boolean[n];
        for (int i = 0; i < n; ++i) {
            if (i > 0 && nums.get(i) > nums.get(i - 1)) {
                ++cnt;
            }
            if (i >= k && nums.get(i - k + 1) > nums.get(i - k)) {
                --cnt;
            }
            if (cnt >= k) {
                a[i] = true;
                if (i - k >= 0 && a[i - k]) {
                    return true;
                }
            }
        }
        return false;

    }

    // 3350. 检测相邻递增子数组 II (Adjacent Increasing Subarrays Detection II)
    public int maxIncreasingSubarrays(List<Integer> nums) {
        int n = nums.size();
        int left = 1;
        int right = n / 2;
        int res = 1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (hasIncreasingSubarrays(nums, mid)) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    // 3351. 好子序列的元素之和 (Sum of Good Subsequences)
    public int sumOfGoodSubsequences(int[] nums) {
        long res = 0;
        final int MOD = (int) (1e9 + 7);
        Map<Integer, Integer> f = new HashMap<>();
        Map<Integer, Integer> cnt = new HashMap<>();
        for (int x : nums) {
            long c = cnt.getOrDefault(x - 1, 0) + cnt.getOrDefault(x + 1, 0) + 1;
            f.put(x, (int) ((x * c + f.getOrDefault(x, 0) + f.getOrDefault(x - 1, 0) + f.getOrDefault(x + 1, 0))
                    % MOD));
            cnt.put(x, (int) ((cnt.getOrDefault(x, 0) + c) % MOD));
        }
        for (int x : f.values()) {
            res += x;
        }
        return (int) (res % MOD);
    }

    // 3352. 统计小于 N 的 K 可约简整数 (Count K-Reducible Numbers Less Than N)
    private int[][] memo3352;
    private int n3352;
    private String s3352;

    public int countKReducibleNumbers(String s, int k) {
        this.n3352 = s.length();
        this.s3352 = s;
        this.memo3352 = new int[n3352][n3352 + 1];
        for (int i = 0; i < n3352; ++i) {
            Arrays.fill(memo3352[i], -1);
        }
        int[] f = new int[n3352 + 1];
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        for (int i = 1; i < n3352 + 1; ++i) {
            if (i > 1) {
                f[i] = f[Integer.bitCount(i)] + 1;
            }
            if (f[i] < k) {
                res += dfs3352(0, i, true);
                res %= MOD;
            }
        }
        return res;

    }

    private int dfs3352(int i, int j, boolean isLimit) {
        if (i == n3352) {
            return isLimit || j > 0 ? 0 : 1;
        }
        if (!isLimit && memo3352[i][j] != -1) {
            return memo3352[i][j];
        }
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        int up = isLimit ? s3352.charAt(i) - '0' : 1;
        for (int d = 0; d <= up; ++d) {
            if (j - d < 0) {
                continue;
            }
            res += dfs3352(i + 1, j - d, isLimit && up == d);
            res %= MOD;
        }
        if (!isLimit) {
            memo3352[i][j] = res;
        }
        return res;
    }

    // 3258. 统计满足 K 约束的子字符串数量 I ( Count Substrings That Satisfy K-Constraint I)
    public int countKConstraintSubstrings(String s, int k) {
        int res = 0;
        int[] cnt = new int[2];
        int j = 0;
        for (int i = 0; i < s.length(); ++i) {
            ++cnt[s.charAt(i) - '0'];
            while (cnt[0] > k && cnt[1] > k) {
                --cnt[s.charAt(j++) - '0'];
            }
            res += i - j + 1;
        }
        return res;
    }

    // 3354. 使数组元素等于零 (Make Array Elements Equal to Zero)
    public int countValidSelections(int[] nums) {
        int n = nums.length;
        int[] right = new int[n + 1];
        for (int i = n - 1; i >= 0; --i) {
            right[i] = right[i + 1] + nums[i];
        }
        int res = 0;
        int left = 0;
        for (int i = 0; i < n; ++i) {
            if (nums[i] == 0) {
                res += Math.max(0, 2 - Math.abs(left - right[i + 1]));
            }
            left += nums[i];
        }
        return res;
    }

    // 3355. 零数组变换 I (Zero Array Transformation I)
    public boolean isZeroArray(int[] nums, int[][] queries) {
        int n = nums.length;
        int[] diff = new int[n + 1];
        for (int[] q : queries) {
            --diff[q[0]];
            ++diff[q[1] + 1];
        }
        int d = 0;
        for (int i = 0; i < n; ++i) {
            d += diff[i];
            if (d + nums[i] > 0) {
                return false;
            }
        }
        return true;

    }

    // 3356. 零数组变换 II (Zero Array Transformation II)
    public int minZeroArray(int[] nums, int[][] queries) {
        int left = 0;
        int right = queries.length;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (check3356(nums, mid, queries)) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;

    }

    private boolean check3356(int[] nums, int t, int[][] queries) {
        int n = nums.length;
        int[] diff = new int[n + 1];
        for (int i = 0; i < t; ++i) {
            diff[queries[i][0]] -= queries[i][2];
            diff[queries[i][1] + 1] += queries[i][2];
        }
        long d = 0L;
        for (int i = 0; i < n; ++i) {
            d += diff[i];
            if (d + nums[i] > 0) {
                return false;
            }
        }
        return true;
    }

    // 3360. 移除石头游戏 (Stone Removal Game)
    public boolean canAliceWin(int n) {
        int x = 10;
        while (n >= x) {
            n -= x--;
        }
        return (x & 1) > 0;
    }

    // 3361. 两个字符串的切换距离 (Shift Distance Between Two Strings)
    public long shiftDistance(String s, String t, int[] nextCost, int[] previousCost) {
        long res = 0L;
        long[] a = new long[27];
        long[] b = new long[27];
        for (int i = 1; i < 27; ++i) {
            a[i] = a[i - 1] + nextCost[i - 1];
            b[i] = b[i - 1] + previousCost[i - 1];
        }
        for (int i = 0; i < s.length(); ++i) {
            char x = s.charAt(i);
            char y = t.charAt(i);
            if (x < y) {
                res += Math.min(a[y - 'a'] - a[x - 'a'], b[26] - (b[y - 'a' + 1] - b[x - 'a' + 1]));
            } else {
                char tmp = x;
                x = y;
                y = tmp;
                res += Math.min(a[26] - (a[y - 'a'] - a[x - 'a']), b[y - 'a' + 1] - b[x - 'a' + 1]);
            }
        }
        return res;

    }

    // 3363. 最多可收集的水果数目 (Find the Maximum Number of Fruits Collected)
    private int n3363;
    private int[][] memo3363;
    private int[][] fruits3363;

    public int maxCollectedFruits(int[][] fruits) {
        this.n3363 = fruits.length;
        int res = 0;
        for (int i = 0; i < n3363; ++i) {
            res += fruits[i][i];
            fruits[i][i] = 0;
        }
        this.memo3363 = new int[n3363][n3363];
        for (int i = 0; i < n3363; ++i) {
            Arrays.fill(memo3363[i], -1);
        }
        this.fruits3363 = fruits;
        res += dfs3363(0, n3363 - 1);
        this.memo3363 = new int[n3363][n3363];
        for (int i = 0; i < n3363; ++i) {
            Arrays.fill(memo3363[i], -1);
        }
        res += dfs2_3363(n3363 - 1, 0);
        return res;
    }

    private int dfs3363(int i, int j) {
        if (i == n3363 - 1) {
            return j == n3363 - 1 ? 0 : Integer.MIN_VALUE;
        }
        if (i < (n3363 + 1) / 2 && i + j < n3363 - 1 || i >= (n3363 + 1) / 2 && i > j) {
            return Integer.MIN_VALUE;
        }
        if (memo3363[i][j] != -1) {
            return memo3363[i][j];
        }
        int mx = 0;
        for (int k = Math.max(0, j - 1); k < Math.min(n3363, j + 2); ++k) {
            mx = Math.max(mx, dfs3363(i + 1, k));
        }
        return memo3363[i][j] = mx + fruits3363[i][j];
    }

    private int dfs2_3363(int i, int j) {
        if (j == n3363 - 1) {
            return i == n3363 - 1 ? 0 : Integer.MIN_VALUE;
        }
        if (j < (n3363 + 1) / 2 && i + j < n3363 - 1 || j >= (n3363 + 1) / 2 && i < j) {
            return Integer.MIN_VALUE;
        }
        if (memo3363[i][j] != -1) {
            return memo3363[i][j];
        }
        int mx = 0;
        for (int k = Math.max(0, i - 1); k < Math.min(n3363, i + 2); ++k) {
            mx = Math.max(mx, dfs2_3363(k, j + 1));
        }
        return memo3363[i][j] = mx + fruits3363[i][j];
    }

    // 3364. 最小正和子数组 (Minimum Positive Sum Subarray )
    public int minimumSumSubarray(List<Integer> nums, int l, int r) {
        int n = nums.size();
        int res = Integer.MAX_VALUE;
        for (int i = 0; i < n; ++i) {
            int s = 0;
            for (int j = i; j < n; ++j) {
                s += nums.get(j);
                if (j - i + 1 <= r && j - i + 1 >= l) {
                    if (s > 0) {
                        res = Math.min(res, s);
                    }
                }
            }
        }
        return res == Integer.MAX_VALUE ? -1 : res;
    }

    // 3365. 重排子字符串以形成目标字符串 (Rearrange K Substrings to Form Target String)
    public boolean isPossibleToRearrange(String s, String t, int k) {
        int n = s.length();
        int l = n / k;
        Map<String, Integer> cnt = new HashMap<>();
        for (int i = l; i <= n; i += l) {
            cnt.merge(s.substring(i - l, i), 1, Integer::sum);
            cnt.merge(t.substring(i - l, i), -1, Integer::sum);
        }
        for (int c : cnt.values()) {
            if (c != 0) {
                return false;
            }
        }
        return true;

    }

    // 3366. 最小数组和 (Minimum Array Sum)
    private int[] nums3366;
    private int n3366;
    private int k3366;
    private int[][][] memo3366;

    public int minArraySum(int[] nums, int k, int op1, int op2) {
        this.nums3366 = nums;
        this.n3366 = nums.length;
        this.k3366 = k;
        this.memo3366 = new int[n3366][op1 + 1][op2 + 1];
        for (int i = 0; i < n3366; ++i) {
            for (int j = 0; j < op1 + 1; ++j) {
                Arrays.fill(memo3366[i][j], -1);
            }
        }
        return dfs3366(0, op1, op2);
    }

    private int dfs3366(int i, int op1, int op2) {
        if (i == n3366) {
            return 0;
        }
        if (memo3366[i][op1][op2] != -1) {
            return memo3366[i][op1][op2];
        }
        int res = dfs3366(i + 1, op1, op2) + nums3366[i];
        if (op1 > 0) {
            int add = (nums3366[i] + 1) / 2;
            res = Math.min(res, dfs3366(i + 1, op1 - 1, op2) + add);
            if (op2 > 0 && add >= k3366) {
                add -= k3366;
                res = Math.min(res, dfs3366(i + 1, op1 - 1, op2 - 1) + add);
            }
        }
        if (op2 > 0 && nums3366[i] >= k3366) {
            int add = nums3366[i] - k3366;
            res = Math.min(res, dfs3366(i + 1, op1, op2 - 1) + add);
            if (op1 > 0) {
                add = (add + 1) / 2;
                res = Math.min(res, dfs3366(i + 1, op1 - 1, op2 - 1) + add);
            }
        }
        return memo3366[i][op1][op2] = res;
    }

    // 3370. 仅含置位位的最小整数 (Smallest Number With All Set Bits)
    public int smallestNumber(int n) {
        return (1 << Integer.toBinaryString(n).length()) - 1;
    }

    // 3371. 识别数组中的最大异常值 (Identify the Largest Outlier in an Array)
    public int getLargestOutlier(int[] nums) {
        int s = 0;
        Map<Integer, Integer> cnt = new HashMap<>();
        for (int x : nums) {
            s += x;
            cnt.merge(x, 1, Integer::sum);
        }
        int res = Integer.MIN_VALUE;
        for (int x : nums) {
            cnt.merge(x, -1, Integer::sum);
            if ((s - x) % 2 == 0 && cnt.getOrDefault((s - x) / 2, 0) > 0) {
                res = Math.max(res, x);
            }
            cnt.merge(x, 1, Integer::sum);
        }
        return res;

    }

    // 3372. 连接两棵树后最大目标节点数目 I (Maximize the Number of Target Nodes After
    // Connecting Trees I)
    public int[] maxTargetNodes(int[][] edges1, int[][] edges2, int k) {
        List<Integer>[] g2 = buildTree3372(edges2);
        int mx = 0;
        for (int i = 0; i < g2.length; ++i) {
            mx = Math.max(mx, dfs3372(i, -1, g2, k - 1));
        }
        List<Integer>[] g1 = buildTree3372(edges1);
        int[] res = new int[edges1.length + 1];
        for (int i = 0; i < res.length; ++i) {
            res[i] = dfs3372(i, -1, g1, k) + mx;
        }
        return res;
    }

    private int dfs3372(int x, int fa, List<Integer>[] g, int k) {
        if (k < 0) {
            return 0;
        }
        int res = 0;
        for (int y : g[x]) {
            if (y != fa) {
                res += dfs3372(y, x, g, k - 1);
            }
        }
        return res + 1;
    }

    private List<Integer>[] buildTree3372(int[][] edges) {
        int n = edges.length + 1;
        List<Integer>[] g = new ArrayList[n];
        Arrays.setAll(g, k -> new ArrayList<>());
        for (int[] e : edges) {
            g[e[0]].add(e[1]);
            g[e[1]].add(e[0]);
        }
        return g;
    }

    // 3373. 连接两棵树后最大目标节点数目 II (Maximize the Number of Target Nodes After Connecting
    // Trees II)
    public int[] maxTargetNodes(int[][] edges1, int[][] edges2) {
        List<Integer>[] g2 = buildTree3373(edges2);
        int[] cnts2 = cal3373(g2);
        int mx = Math.max(cnts2[0], cnts2[1]);
        List<Integer>[] g1 = buildTree3373(edges1);
        int[] cnts1 = cal3373(g1);
        int[] res = new int[edges1.length + 1];
        tree3373(0, -1, 0, g1, cnts1, res, mx);
        return res;

    }

    private void tree3373(int x, int fa, int d, List<Integer>[] g, int[] cnts1, int[] res, int mx) {
        res[x] = cnts1[d] + mx;
        for (int y : g[x]) {
            if (y != fa) {
                tree3373(y, x, d ^ 1, g, cnts1, res, mx);
            }
        }
    }

    private int[] cal3373(List<Integer>[] g) {
        int[] cnt = new int[2];
        dfs3373(0, -1, 0, g, cnt);
        return cnt;
    }

    private void dfs3373(int x, int fa, int d, List<Integer>[] g, int[] cnt) {
        ++cnt[d];
        for (int y : g[x]) {
            if (y != fa) {
                dfs3373(y, x, d ^ 1, g, cnt);
            }
        }
    }

    private List<Integer>[] buildTree3373(int[][] edges) {
        int n = edges.length + 1;
        List<Integer>[] g = new ArrayList[n];
        Arrays.setAll(g, k -> new ArrayList<>());
        for (int[] e : edges) {
            g[e[0]].add(e[1]);
            g[e[1]].add(e[0]);
        }
        return g;
    }

    // 2056. 棋盘上有效移动组合的数目 (Number of Valid Move Combinations On Chessboard)
    public int countCombinations(String[] pieces, int[][] positions) {
        List<int[]> flat = List.of(new int[] { 1, 0 }, new int[] { 0, 1 }, new int[] { -1, 0 }, new int[] { 0, -1 });
        List<int[]> diagnal = List.of(new int[] { 1, 1 }, new int[] { -1, 1 }, new int[] { -1, -1 },
                new int[] { 1, -1 });
        Map<Character, List<int[]>> dic = new HashMap<>();
        dic.put('r', flat);
        List<int[]> all = new ArrayList<>();
        all.addAll(flat);
        all.addAll(diagnal);
        dic.put('q', all);
        dic.put('b', diagnal);
        int n = pieces.length;
        List<List<int[]>> allMoves = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            allMoves.add(generateAllMoves2056(dic.get(pieces[i].charAt(0)), positions[i][0] - 1, positions[i][1] - 1));
        }
        int[][] moves = new int[n][];
        return dfs2056(0, n, moves, allMoves);

    }

    private boolean check2056(int[] move0, int[] move1) {
        int x0 = move0[0];
        int y0 = move0[1];
        int dx0 = move0[2];
        int dy0 = move0[3];
        int step0 = move0[4];

        int x1 = move1[0];
        int y1 = move1[1];
        int dx1 = move1[2];
        int dy1 = move1[3];
        int step1 = move1[4];

        for (int i = 0; i < Math.max(step0, step1); ++i) {
            if (i < step0) {
                x0 += dx0;
                y0 += dy0;
            }
            if (i < step1) {
                x1 += dx1;
                y1 += dy1;
            }
            if (x0 == x1 && y0 == y1) {
                return false;
            }
        }
        return true;
    }

    private int dfs2056(int i, int n, int[][] moves, List<List<int[]>> allMoves) {
        if (i == n) {
            return 1;
        }
        int res = 0;
        search: for (int[] move0 : allMoves.get(i)) {
            for (int j = 0; j < i; ++j) {
                int[] move1 = moves[j];
                if (!check2056(move0, move1)) {
                    continue search;
                }
            }
            moves[i] = move0;
            res += dfs2056(i + 1, n, moves, allMoves);
        }
        return res;

    }

    private List<int[]> generateAllMoves2056(List<int[]> list, int x0, int y0) {
        final int SIZE = 8;
        List<int[]> items = new ArrayList<>();
        items.add(new int[] { x0, y0, 0, 0, 0 });
        for (int[] d : list) {
            int dx = d[0];
            int dy = d[1];
            int x = x0;
            int y = y0;
            int step = 0;
            while (x + dx >= 0 && x + dx < SIZE && y + dy >= 0 && y + dy < SIZE) {
                x += dx;
                y += dy;
                ++step;
                items.add(new int[] { x0, y0, dx, dy, step });
            }
        }
        return items;
    }

    // 3375. 使数组的值全部为 K 的最少操作次数 (Minimum Operations to Make Array Values Equal to K)
    public int minOperations(int[] nums, int k) {
        Set<Integer> s = new HashSet<>();
        for (int x : nums) {
            if (k > x) {
                return -1;
            }
            if (x > k) {
                s.add(x);
            }
        }
        return s.size();

    }

    // 3377. 使两个整数相等的数位操作 (Digit Operations to Make Two Integers Equal)
    public int minOperations(int n, int m) {
        int digit = String.valueOf(n).length();
        int SIZE = (int) Math.pow(10, digit);
        boolean[] prime = new boolean[SIZE];
        Arrays.fill(prime, true);
        prime[0] = prime[1] = false;
        for (int i = 2; i < SIZE; ++i) {
            if (prime[i]) {
                for (int j = i * i; j < SIZE; j += i) {
                    prime[j] = false;
                }
            }
        }
        if (prime[n] || prime[m]) {
            return -1;
        }
        int[] dis = new int[SIZE];
        Arrays.fill(dis, (int) 1e9);
        dis[n] = n;
        Queue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });
        q.offer(new int[] { n, n });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int s = cur[0];
            int x = cur[1];
            if (x == m) {
                return s;
            }
            if (s > dis[x]) {
                continue;
            }
            char[] arr = String.valueOf(x).toCharArray();
            for (int i = 0; i < arr.length; ++i) {
                if (arr[i] != '9') {
                    ++arr[i];
                    int y = Integer.parseInt(String.valueOf(arr));
                    if (!prime[y] && s + y < dis[y]) {
                        dis[y] = s + y;
                        q.offer(new int[] { s + y, y });
                    }
                    --arr[i];
                }
                if (arr[i] != '0') {
                    --arr[i];
                    int y = Integer.parseInt(String.valueOf(arr));
                    if (!prime[y] && s + y < dis[y]) {
                        dis[y] = s + y;
                        q.offer(new int[] { s + y, y });
                    }
                    ++arr[i];
                }
            }
        }
        return -1;

    }

    // 100489. 破解锁的最少时间 I (Minimum Time to Break Locks I)
    private int[] memo100489;
    private int u100489;
    private int K100489;
    private List<Integer> strength100489;

    public int findMinimumTime(List<Integer> strength, int K) {
        this.strength100489 = strength;
        int n = strength.size();
        this.u100489 = (1 << n) - 1;
        this.K100489 = K;
        this.memo100489 = new int[1 << n];
        Arrays.fill(memo100489, -1);
        return dfs100489(0);
    }

    private int dfs100489(int i) {
        if (i == u100489) {
            return 0;
        }
        if (memo100489[i] != -1) {
            return memo100489[i];
        }
        int j = 1 + Integer.bitCount(i) * K100489;
        int res = Integer.MAX_VALUE;
        for (int c = i ^ u100489; c != 0; c &= c - 1) {
            int lb = Integer.numberOfTrailingZeros(c);
            res = Math.min(res, dfs100489(i | (1 << lb)) + (strength100489.get(lb) - 1) / j + 1);
        }
        return memo100489[i] = res;
    }

    // 3379. 转换数组 (Transformed Array)
    public int[] constructTransformedArray(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            res[i] = nums[((i + nums[i]) % n + n) % n];
        }
        return res;

    }

    // 3381. 长度可被 K 整除的子数组的最大元素和 (Maximum Subarray Sum With Length Divisible by K)
    public long maxSubarraySum(int[] nums, int k) {
        int n = nums.length;
        long[] pre = new long[n + 1];
        for (int i = 1; i < n + 1; ++i) {
            pre[i] = pre[i - 1] + nums[i - 1];
        }
        long res = (long) -1e15;
        long[] modK = new long[k];
        Arrays.fill(modK, (long) 1e15);
        for (int i = 0; i < n + 1; ++i) {
            int j = i % k;
            res = Math.max(res, pre[i] - modK[j]);
            modK[j] = Math.min(modK[j], pre[i]);
        }
        return res;

    }

    // 3386. 按下时间最长的按钮 (Button with Longest Push Time)
    public int buttonWithLongestTime(int[][] events) {
        int d = events[0][1];
        int res = events[0][0];
        for (int i = 1; i < events.length; ++i) {
            if (events[i][1] - events[i - 1][1] > d) {
                res = events[i][0];
                d = events[i][1] - events[i - 1][1];
            } else if (events[i][1] - events[i - 1][1] == d && events[i][0] < res) {
                res = events[i][0];
            }
        }
        return res;

    }

    // 3392. 统计符合条件长度为 3 的子数组数目 (Count Subarrays of Length Three With a Condition)
    public int countSubarrays(int[] nums) {
        int res = 0;
        for (int i = 1; i < nums.length - 1; ++i) {
            if ((nums[i - 1] + nums[i + 1]) * 2 == nums[i]) {
                ++res;
            }
        }
        return res;

    }

    // 3393. 统计异或值为给定值的路径数目 (Count Paths With the Given XOR Value)
    private int[][][] memo3393;
    private int m3393;
    private int n3393;
    private int[][] grid3393;
    private int k3393;

    public int countPathsWithXorValue(int[][] grid, int k) {
        this.m3393 = grid.length;
        this.n3393 = grid[0].length;
        this.k3393 = k;
        this.grid3393 = grid;
        this.memo3393 = new int[m3393][n3393][17];
        for (int i = 0; i < m3393; ++i) {
            for (int j = 0; j < n3393; ++j) {
                Arrays.fill(memo3393[i][j], -1);
            }
        }
        return dfs3393(0, 0, 0);

    }

    private int dfs3393(int i, int j, int x) {
        if (i == m3393 || j == n3393) {
            return 0;
        }
        x ^= grid3393[i][j];
        if (i == m3393 - 1 && j == n3393 - 1) {
            return x == k3393 ? 1 : 0;
        }
        if (memo3393[i][j][x] != -1) {
            return memo3393[i][j][x];
        }
        final int MOD = (int) (1e9 + 7);
        return memo3393[i][j][x] = (dfs3393(i + 1, j, x) + dfs3393(i, j + 1, x)) % MOD;
    }

    // 3394. 判断网格图能否被切割成块 (Check if Grid can be Cut into Sections)
    public boolean checkValidCuts(int n, int[][] rectangles) {
        return check3394(rectangles, 0, 2) || check3394(rectangles, 1, 3);
    }

    private boolean check3394(int[][] rectangles, int i0, int i1) {
        Arrays.sort(rectangles, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[i0], o2[i0]);
            }

        });
        int res = 0;
        int i = 0;
        while (i < rectangles.length) {
            int x = rectangles[i][i1];
            int j = i + 1;
            while (j < rectangles.length && rectangles[j][i0] < x) {
                x = Math.max(x, rectangles[j][i1]);
                ++j;
            }
            i = j;
            ++res;
            if (res >= 3) {
                return true;
            }
        }
        return false;
    }

    // 3396. 使数组元素互不相同所需的最少操作次数 (Minimum Number of Operations to Make Elements in
    // Array Distinct)
    public int minimumOperations3396(int[] nums) {
        Map<Integer, Integer> cnt = new HashMap<>();
        for (int x : nums) {
            cnt.merge(x, 1, Integer::sum);
        }
        int res = 0;
        int n = nums.length;
        for (int i = 0; i < n - 2; i += 3) {
            if (n - i == cnt.size()) {
                break;
            }
            cnt.merge(nums[i], -1, Integer::sum);
            if (cnt.get(nums[i]) == 0) {
                cnt.remove(nums[i]);
            }
            cnt.merge(nums[i + 1], -1, Integer::sum);
            if (cnt.get(nums[i + 1]) == 0) {
                cnt.remove(nums[i + 1]);
            }
            cnt.merge(nums[i + 2], -1, Integer::sum);
            if (cnt.get(nums[i + 2]) == 0) {
                cnt.remove(nums[i + 2]);
            }
            ++res;
        }
        for (Map.Entry<Integer, Integer> entry : cnt.entrySet()) {
            if (entry.getValue() > 1) {
                ++res;
                break;
            }
        }
        return res;
    }

    // 3397. 执行操作后不同元素的最大数量 (Maximum Number of Distinct Elements After Operations)
    public int maxDistinctElements(int[] nums, int k) {
        Arrays.sort(nums);
        int res = 0;
        int pre = Integer.MIN_VALUE;
        for (int x : nums) {
            x = Math.min(Math.max(x - k, pre + 1), x + k);
            if (x > pre) {
                ++res;
                pre = x;
            }
        }
        return res;
    }

    // 3398. 字符相同的最短子字符串 I (Smallest Substring With Identical Characters I)
    // 3399. 字符相同的最短子字符串 II (Smallest Substring With Identical Characters II)
    public int minLength(String s, int numOps) {
        int left = 1;
        int right = s.length();
        char[] arr = s.toCharArray();
        int res = s.length();
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (check3398(arr, mid, numOps)) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    private boolean check3398(char[] arr, int target, int numOps) {
        int cnt = 0;
        if (target == 1) {
            for (int i = 0; i < arr.length; ++i) {
                cnt += ((arr[i] - '0') ^ i) & 1;
            }
            cnt = Math.min(arr.length - cnt, cnt);
            return cnt <= numOps;
        }
        int k = 0;
        for (int i = 0; i < arr.length; ++i) {
            ++k;
            if (i == arr.length - 1 || arr[i] != arr[i + 1]) {
                cnt += k / (target + 1);
                k = 0;
            }
        }
        return cnt <= numOps;
    }

    // 3402. 使每一列严格递增的最少操作次数 (Minimum Operations to Make Columns Strictly
    // Increasing)
    public int minimumOperations(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int res = 0;
        for (int j = 0; j < n; ++j) {
            for (int i = 1; i < m; ++i) {
                int add = Math.max(0, grid[i - 1][j] + 1 - grid[i][j]);
                res += add;
                grid[i][j] += add;
            }
        }
        return res;

    }

    // 3403. 从盒子中找出字典序最大的字符串 I (Find the Lexicographically Largest String From the
    // Box I)
    public String answerString(String word, int numFriends) {
        if (numFriends == 1) {
            return word;
        }
        int n = word.length();
        String res = "";
        for (int i = 0; i < n; ++i) {
            String s = word.substring(i, Math.min(n, n - numFriends + 1 + i));
            if (s.compareTo(res) > 0) {
                res = s;
            }
        }
        return res;
    }

    // 3417. 跳过交替单元格的之字形遍历 (Zigzag Grid Traversal With Skip)
    public List<Integer> zigzagTraversal(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        List<Integer> res = new ArrayList<>();
        int d = 0;
        for (int i = 0; i < m; ++i) {
            if (i % 2 == 0) {
                for (int j = 0; j < n; ++j) {
                    if (d == 0) {
                        res.add(grid[i][j]);
                    }
                    d ^= 1;
                }
            } else {
                for (int j = n - 1; j >= 0; --j) {
                    if (d == 0) {
                        res.add(grid[i][j]);
                    }
                    d ^= 1;
                }
            }
        }
        return res;

    }

    // 3418. 机器人可以获得的最大金币数 (Maximum Amount of Money Robot Can Earn)
    private int m3418;
    private int n3418;
    private int[][] coins3418;
    private int[][][] memo3418;

    public int maximumAmount(int[][] coins) {
        this.coins3418 = coins;
        this.m3418 = coins.length;
        this.n3418 = coins[0].length;
        this.memo3418 = new int[m3418][n3418][3];
        for (int i = 0; i < m3418; ++i) {
            for (int j = 0; j < n3418; ++j) {
                Arrays.fill(memo3418[i][j], (int) -1e9);
            }
        }
        return dfs3418(0, 0, 2);
    }

    private int dfs3418(int i, int j, int k) {
        if (!(i >= 0 && i < m3418 && j >= 0 && j < n3418)) {
            return (int) -1e9;
        }
        if (i == m3418 - 1 && j == n3418 - 1) {
            int res = coins3418[i][j];
            if (k > 0) {
                res = Math.max(res, 0);
            }
            return res;
        }
        if (memo3418[i][j][k] != (int) -1e9) {
            return memo3418[i][j][k];
        }
        int res = Math.max(dfs3418(i + 1, j, k), dfs3418(i, j + 1, k)) + coins3418[i][j];
        if (k > 0) {
            res = Math.max(res, dfs3418(i + 1, j, k - 1));
            res = Math.max(res, dfs3418(i, j + 1, k - 1));
        }
        return memo3418[i][j][k] = res;
    }

    // 3419. 图的最大边权的最小值 (Minimize the Maximum Edge Weight of Graph)
    public int minMaxWeight(int n, int[][] edges, int threshold) {
        int left = 0;
        int right = 0;
        for (int[] e : edges) {
            right = Math.max(right, e[2]);
        }
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (check3419(n, edges, mid)) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    private boolean check3419(int n, int[][] edges, int mx) {
        List<int[]>[] g = new ArrayList[n];
        Arrays.setAll(g, k -> new ArrayList<>());
        for (int[] e : edges) {
            g[e[1]].add(new int[] { e[0], e[2] });
        }
        boolean[] vis = new boolean[n];
        return dfs3419(0, vis, g, mx) == n;
    }

    private int dfs3419(int x, boolean[] vis, List<int[]>[] g, int mx) {
        int res = 1;
        vis[x] = true;
        for (int[] y : g[x]) {
            if (y[1] <= mx && !vis[y[0]]) {
                res += dfs3419(y[0], vis, g, mx);
            }
        }
        return res;
    }

    // 3423. 循环数组中相邻元素的最大差值 (Maximum Difference Between Adjacent Elements in a
    // Circular Array)
    public int maxAdjacentDistance(int[] nums) {
        int n = nums.length;
        int res = Math.abs(nums[0] - nums[n - 1]);
        for (int i = 1; i < n; ++i) {
            res = Math.max(res, Math.abs(nums[i] - nums[i - 1]));
        }
        return res;
    }

    // 3424. 将数组变相同的最小代价 (Minimum Cost to Make Arrays Identical)
    public long minCost(int[] arr, int[] brr, long k) {
        long res1 = 0L;
        for (int i = 0; i < arr.length; ++i) {
            res1 += Math.abs(arr[i] - brr[i]);
        }
        Arrays.sort(arr);
        Arrays.sort(brr);
        long res2 = 0L;
        for (int i = 0; i < arr.length; ++i) {
            res2 += Math.abs(arr[i] - brr[i]);
        }
        return Math.min(res1, res2 + k);

    }

    // 3427. 变长子数组求和 (Sum of Variable Length Subarrays)
    public int subarraySum(int[] nums) {
        int n = nums.length;
        int[] pre = new int[n + 1];
        for (int i = 0; i < n; ++i) {
            pre[i + 1] = pre[i] + nums[i];
        }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            int start = Math.max(0, i - nums[i]);
            res += pre[i + 1] - pre[start];
        }
        return res;
    }

    // 3429. 粉刷房子 IV (Paint House IV)
    private int n3429;
    private int[][] cost3429;
    private long[][][] memo3429;

    public long minCost(int n, int[][] cost) {
        this.n3429 = n;
        this.cost3429 = cost;
        this.memo3429 = new long[n][4][4];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < 4; ++j) {
                Arrays.fill(memo3429[i][j], -1L);
            }
        }
        return dfs3429(0, 3, 3);

    }

    private long dfs3429(int i, int j, int k) {
        if (i == n3429 / 2) {
            return 0L;
        }
        if (memo3429[i][j][k] != -1L) {
            return memo3429[i][j][k];
        }
        long res = Long.MAX_VALUE;
        for (int l = 0; l < 3; ++l) {
            for (int r = 0; r < 3; ++r) {
                if (l == r || l == j || r == k) {
                    continue;
                }
                res = Math.min(res, dfs3429(i + 1, l, r) + cost3429[i][l] + cost3429[n3429 - i - 1][r]);
            }
        }
        return memo3429[i][j][k] = res;
    }

    // 3428. 最多 K 个元素的子序列的最值之和 (Maximum and Minimum Sums of at Most Size K
    // Subsequences)
    private int[][] memo3428;

    public int minMaxSums(int[] nums, int k) {
        Arrays.sort(nums);
        int n = nums.length;
        this.memo3428 = new int[n][Math.min(n, k) + 1];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(memo3428[i], -1);
        }
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < Math.min(k, i + 1); ++j) {
                res += ((long) dfs3428(i, j) * (nums[i] + nums[n - i - 1])) % MOD;
                res %= MOD;
            }
        }
        return res;

    }

    private int dfs3428(int i, int j) {
        if (i == j || j == 0) {
            return 1;
        }
        if (memo3428[i][j] != -1) {
            return memo3428[i][j];
        }
        final int MOD = (int) (1e9 + 7);
        return memo3428[i][j] = (dfs3428(i - 1, j - 1) + dfs3428(i - 1, j)) % MOD;
    }

    // 3425. 最长特殊路径 (Longest Special Path)
    private int maxS3425 = -1;
    private int minCnt3425 = 0;
    private List<int[]>[] g3425;
    private int[] nums3425;
    private int n3425;
    private Map<Integer, Integer> lastDepth3425;
    private List<Integer> preSum3425;

    public int[] longestSpecialPath(int[][] edges, int[] nums) {
        this.n3425 = nums.length;
        this.nums3425 = nums;
        this.g3425 = new ArrayList[n3425];
        Arrays.setAll(g3425, k -> new ArrayList<>());
        for (int[] e : edges) {
            g3425[e[0]].add(new int[] { e[1], e[2] });
            g3425[e[1]].add(new int[] { e[0], e[2] });
        }
        this.lastDepth3425 = new HashMap<>();
        this.preSum3425 = new ArrayList<>();
        preSum3425.add(0);
        dfs3425(0, -1, 0);
        return new int[] { maxS3425, minCnt3425 };
    }

    private void dfs3425(int x, int fa, int topDepth) {
        int color = nums3425[x];
        int lastDep = lastDepth3425.getOrDefault(color, 0);
        topDepth = Math.max(topDepth, lastDep);

        int s = preSum3425.get(preSum3425.size() - 1) - preSum3425.get(topDepth);
        int cnt = preSum3425.size() - topDepth;
        if (s > maxS3425 || s == maxS3425 && cnt < minCnt3425) {
            maxS3425 = s;
            minCnt3425 = cnt;
        }
        lastDepth3425.put(color, preSum3425.size());
        for (int[] nxt : g3425[x]) {
            int y = nxt[0];
            int w = nxt[1];
            if (y != fa) {
                preSum3425.add(preSum3425.get(preSum3425.size() - 1) + w);
                dfs3425(y, x, topDepth);
                preSum3425.remove(preSum3425.size() - 1);
            }
        }
        lastDepth3425.put(color, lastDep);
    }

    // 541. 反转字符串 II (Reverse String II)
    public String reverseStr(String s, int k) {
        char[] arr = s.toCharArray();
        int n = arr.length;
        for (int i = 0; i < n; i += 2 * k) {
            int l = i;
            int r = Math.min(i + k - 1, n - 1);
            while (l < r) {
                char tmp = arr[l];
                arr[l] = arr[r];
                arr[r] = tmp;
                ++l;
                --r;
            }
        }
        return new String(arr);
    }

    // 3461. 判断操作后字符串中的数字是否相等 I (Check If Digits Are Equal in String After
    // Operations I)
    public boolean hasSameDigits(String s) {
        while (s.length() != 2) {
            char[] arr = s.toCharArray();
            for (int i = 0; i < arr.length - 1; ++i) {
                arr[i] = (char) ((arr[i] - '0' + arr[i + 1] - '0') % 10 + '0');
            }
            s = new String(arr, 0, arr.length - 1);
        }
        return s.charAt(0) == s.charAt(1);

    }

    // 3462. 提取至多 K 个元素的最大总和 (Maximum Sum With at Most K Elements)
    public long maxSum(int[][] grid, int[] limits, int k) {
        int m = grid.length;
        int n = grid[0].length;
        List<Integer> list = new ArrayList<>();
        long res = 0L;
        for (int i = 0; i < m; ++i) {
            Arrays.sort(grid[i]);
            for (int j = n - 1; j >= n - limits[i]; --j) {
                list.add(grid[i][j]);
            }
        }
        Collections.sort(list, Collections.reverseOrder());
        for (int i = 0; i < k; ++i) {
            res += list.get(i);
        }
        return res;
    }

    // 3477. 将水果放入篮子 II (Fruits Into Baskets II)
    public int numOfUnplacedFruits(int[] fruits, int[] baskets) {
        int n = fruits.length;
        int res = n;
        for (int x : fruits) {
            for (int i = 0; i < n; ++i) {
                if (x <= baskets[i]) {
                    baskets[i] = 0;
                    --res;
                    break;
                }
            }
        }
        return res;

    }

    // 3478. 选出和最大的 K 个元素 (Choose K Elements With Maximum Sum)
    public long[] findMaxSum(int[] nums1, int[] nums2, int k) {
        int n = nums1.length;
        int[][] a = new int[n][3];
        for (int i = 0; i < n; i++) {
            a[i] = new int[] { nums1[i], nums2[i], i };
        }
        Arrays.sort(a, (p, q) -> p[0] - q[0]);

        long[] ans = new long[n];
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        long s = 0;
        // 分组循环模板
        for (int i = 0; i < n;) {
            int start = i;
            int x = a[start][0];
            // 找到所有相同的 nums1[i]，这些数的答案都是一样的
            while (i < n && a[i][0] == x) {
                ans[a[i][2]] = s;
                i++;
            }
            // 把这些相同的 nums1[i] 对应的 nums2[i] 入堆
            for (int j = start; j < i; j++) {
                int y = a[j][1];
                s += y;
                pq.offer(y);
                if (pq.size() > k) {
                    s -= pq.poll();
                }
            }
        }
        return ans;
    }

    // 3471. 找出最大的几近缺失整数 (Find the Largest Almost Missing Integer)
    public int largestInteger(int[] nums, int k) {
        int n = nums.length;
        if (k == n) {
            return Arrays.stream(nums).max().getAsInt();
        }
        if (k == 1) {
            Map<Integer, Integer> cnt = new HashMap<>();
            for (int x : nums) {
                cnt.merge(x, 1, Integer::sum); // cnt[x]++
            }
            int ans = -1;
            for (var e : cnt.entrySet()) {
                if (e.getValue() == 1) {
                    ans = Math.max(ans, e.getKey());
                }
            }
            return ans;
        }
        // nums[0] 不能出现在其他地方，nums[n-1] 同理
        return Math.max(f3471(nums, 1, n, nums[0]), f3471(nums, 0, n - 1, nums[n - 1]));
    }

    private int f3471(int[] nums, int begin, int end, int x) {
        for (int i = begin; i < end; i++) {
            if (nums[i] == x) {
                return -1;
            }
        }
        return x;
    }

    // 3472. 至多 K 次操作后的最长回文子序列 (Longest Palindromic Subsequence After at Most K
    // Operations)
    private int n3472;
    private char[] arr3472;
    private int[][][] memo3472;

    public int longestPalindromicSubsequence(String s, int k) {
        this.n3472 = s.length();
        this.arr3472 = s.toCharArray();
        this.memo3472 = new int[n3472][n3472][k + 1];
        return dfs3472(0, n3472 - 1, k);

    }

    private int dfs3472(int i, int j, int k) {
        if (i >= j) {
            return i == j ? 1 : 0;
        }
        if (memo3472[i][j][k] != -1) {
            return memo3472[i][j][k];
        }
        int res = Math.max(dfs3472(i + 1, j, k), dfs3472(i, j - 1, k));
        int a = arr3472[i] - 'a';
        int b = arr3472[j] - 'a';
        int c = Math.min(Math.abs(a - b), 26 - Math.abs(a - b));
        if (c <= k) {
            res = Math.max(res, dfs3472(i + 1, j - 1, k - c) + 2);
        }
        return memo3472[i][j][k] = res;
    }

    // 3467. 将数组按照奇偶性转化 (Transform Array by Parity)
    public int[] transformArray(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            nums[i] &= 1;
        }
        Arrays.sort(nums);
        return nums;

    }

    // 3468. 可行数组的数目 (Find the Number of Copy Arrays)
    public int countArrays(int[] original, int[][] bounds) {
        int min = (int) -1e9;
        int max = (int) 1e9;
        for (int i = 0; i < original.length; ++i) {
            int l = bounds[i][0];
            int r = bounds[i][1];
            int x = original[i];
            if (x + max < l || x + min > r) {
                return 0;
            }
            min = Math.max(min, l - x);
            max = Math.min(max, r - x);
        }
        return max - min + 1;

    }

    // 3469. 移除所有数组元素的最小代价 (Find Minimum Cost to Remove Array Elements)
    private int n3469;
    private int[][] memo3469;
    private int[] nums3469;

    public int minCost(int[] nums) {
        this.nums3469 = nums;
        this.n3469 = nums.length;
        if (n3469 < 3) {
            return Arrays.stream(nums).max().getAsInt();
        }
        this.memo3469 = new int[n3469][n3469];
        return Math.min(dfs3469(0, 3) + Math.max(nums[1], nums[2]),
                Math.min(dfs3469(2, 3) + Math.max(nums[0], nums[1]), dfs3469(1, 3) + Math.max(nums[0], nums[2])));

    }

    private int dfs3469(int i, int j) {
        if (j >= n3469 - 1) {
            int res = nums3469[i];
            if (j < n3469) {
                res = Math.max(res, nums3469[j]);
            }
            return res;
        }
        if (memo3469[i][j] != 0) {
            return memo3469[i][j];
        }
        return memo3469[i][j] = Math.min(dfs3469(j + 1, j + 2) + Math.max(nums3469[i], nums3469[j]),
                Math.min(dfs3469(j, j + 2) + Math.max(nums3469[i], nums3469[j + 1]),
                        dfs3469(i, j + 2) + Math.max(nums3469[j], nums3469[j + 1])));
    }

    // 3456. 找出长度为 K 的特殊子字符串 (Find Special Substring of Length K)
    public boolean hasSpecialSubstring(String s, int k) {
        int[] cnt = new int[26];
        int n = s.length();
        for (int i = 0; i < s.length(); ++i) {
            ++cnt[s.charAt(i) - 'a'];
            if (i >= k) {
                --cnt[s.charAt(i - k) - 'a'];
            }
            if (i >= k - 1 && cnt[s.charAt(i) - 'a'] == k && (i == k - 1 || s.charAt(i - k) != s.charAt(i))
                    && (i == n - 1 || s.charAt(i + 1) != s.charAt(i))) {
                return true;
            }
        }
        return false;

    }

    // 3457. 吃披萨 (Eat Pizzas!)
    public long maxWeight(int[] pizzas) {
        Arrays.sort(pizzas);
        long res = 0L;
        int n = pizzas.length;
        int d = n / 4;
        int odd = (d + 1) / 2;
        for (int i = 0; i < odd; ++i) {
            res += pizzas[n - i - 1];
        }
        for (int i = 0; i < d / 2; ++i) {
            res += pizzas[n - 2 - odd - i * 2];
        }
        return res;

    }

    // 3432. 统计元素和差值为偶数的分区方案 (Count Partitions with Even Sum Difference)
    public int countPartitions(int[] nums) {
        int res = 0;
        int s = Arrays.stream(nums).sum();
        int pre = 0;
        for (int i = 0; i < nums.length - 1; ++i) {
            pre += nums[i];
            if ((pre - (s - pre)) % 2 == 0) {
                ++res;
            }
        }
        return res;

    }

    // 3483. 不同三位偶数的数目 (Unique 3-Digit Even Numbers)
    private int n3483;
    private int[] digits3483;
    private int vis3483;
    private int cur3483;
    private Set<Integer> set3483;

    public int totalNumbers(int[] digits) {
        this.digits3483 = digits;
        this.n3483 = digits.length;
        this.set3483 = new HashSet<>();
        dfs3483();
        return set3483.size();

    }

    private void dfs3483() {
        if (Integer.bitCount(vis3483) == 3) {
            if (cur3483 >= 100 && cur3483 <= 999 && cur3483 % 2 == 0) {
                set3483.add(cur3483);
            }
            return;
        }
        for (int i = 0; i < n3483; ++i) {
            if (((vis3483 >> i) & 1) == 0) {
                vis3483 ^= 1 << i;
                cur3483 = cur3483 * 10 + digits3483[i];
                dfs3483();
                cur3483 /= 10;
                vis3483 ^= 1 << i;
            }
        }
    }

    // 3484. 设计电子表格 (Design Spreadsheet)
    class Spreadsheet {
        private int[][] sheets;

        public Spreadsheet(int rows) {
            this.sheets = new int[rows + 1][26];
        }

        public void setCell(String cell, int value) {
            int[] rc = check(cell);
            sheets[rc[0]][rc[1]] = value;
        }

        public void resetCell(String cell) {
            int[] rc = check(cell);
            sheets[rc[0]][rc[1]] = 0;
        }

        public int getValue(String formula) {
            int plusPos = formula.indexOf("+");
            String x = formula.substring(1, plusPos);
            int a = 0;
            if (Character.isDigit(x.charAt(0))) {
                a = Integer.parseInt(x);
            } else {
                int[] rc = check(x);
                a = sheets[rc[0]][rc[1]];
            }

            String y = formula.substring(plusPos + 1);
            int b = 0;
            if (Character.isDigit(y.charAt(0))) {
                b = Integer.parseInt(y);
            } else {
                int[] rc = check(y);
                b = sheets[rc[0]][rc[1]];
            }
            return a + b;
        }

        private int[] check(String cell) {
            int r = Integer.parseInt(cell.substring(1));
            int c = cell.charAt(0) - 'A';
            return new int[] { r, c };
        }
    }

    // 3487. 删除后的最大子数组元素和 (Maximum Unique Subarray Sum After Deletion)
    public int maxSum(int[] nums) {
        Set<Integer> s1 = new HashSet<>();
        Set<Integer> s2 = new HashSet<>();
        for (int x : nums) {
            if (x >= 0) {
                s1.add(x);
            } else {
                s2.add(x);
            }
        }
        if (!s1.isEmpty()) {
            return s1.stream().mapToInt(Integer::intValue).sum();
        }
        return s2.stream().mapToInt(Integer::intValue).max().getAsInt();

    }

    // 3488. 距离最小相等元素查询 (Closest Equal Element Queries)
    public List<Integer> solveQueries(int[] nums, int[] queries) {
        List<Integer> res = new ArrayList<>();
        Map<Integer, List<Integer>> pos = new HashMap<>();
        for (int i = 0; i < nums.length; ++i) {
            pos.computeIfAbsent(nums[i], k -> new ArrayList<>()).add(i);
        }
        for (int q : queries) {
            List<Integer> list = pos.getOrDefault(nums[q], new ArrayList<>());
            if (list.size() <= 1) {
                res.add(-1);
                continue;
            }
            int p = binarySearch3488(list, q);
            int a = Math.abs(list.get((p - 1 + list.size()) % list.size()) - list.get(p));
            a = Math.min(a, Math.abs(nums.length - a));
            int b = Math.abs(list.get((p + 1) % list.size()) - list.get(p));
            b = Math.min(b, Math.abs(nums.length - b));
            res.add(Math.min(a, b));
        }
        return res;
    }

    private int binarySearch3488(List<Integer> list, int x) {
        int left = 0;
        int right = list.size() - 1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (list.get(mid) == x) {
                return mid;
            } else if (list.get(mid) < x) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }

    // 3489. 零数组变换 IV (Zero Array Transformation IV)
    private int[][] queries3489;
    private int[][] memo3489;
    private int pos3489;

    public int minZeroArray3489(int[] nums, int[][] queries) {
        int res = -1;
        this.queries3489 = queries;
        for (int i = 0; i < nums.length; ++i) {
            this.pos3489 = i;
            this.memo3489 = new int[queries.length][nums[i] + 1];
            res = Math.max(res, dfs3489(0, nums[i]));
            if (res == (int) 1e9) {
                return -1;
            }
        }
        return res;
    }

    private int dfs3489(int i, int j) {
        if (j == 0) {
            return i;
        }
        if (i == queries3489.length) {
            return (int) 1e9;
        }
        if (memo3489[i][j] != 0) {
            return memo3489[i][j];
        }
        // 不选
        int res = dfs3489(i + 1, j);
        // 选
        if (queries3489[i][0] <= pos3489 && pos3489 <= queries3489[i][1] && j >= queries3489[i][2]) {
            res = Math.min(res, dfs3489(i + 1, j - queries3489[i][2]));
        }
        return memo3489[i][j] = res;
    }

    // 3492. 船上可以装载的最大集装箱数量 (Maximum Containers on a Ship)
    public int maxContainers(int n, int w, int maxWeight) {
        maxWeight = Math.min(maxWeight, n * n * w);
        return maxWeight / w;

    }

    // 3493. 属性图 (Properties Graph)
    public int numberOfComponents(int[][] properties, int k) {
        int n = properties.length;
        Union3493 union = new Union3493(n);
        for (int i = 0; i < n; ++i) {
            Set<Integer> s1 = Arrays.stream(properties[i]).boxed().collect(Collectors.toSet());
            for (int j = i + 1; j < n; ++j) {
                Set<Integer> s2 = Arrays.stream(properties[j]).boxed().collect(Collectors.toSet());
                int cnt = 0;
                for (int x : s1) {
                    if (s2.contains(x)) {
                        ++cnt;
                    }
                }
                if (cnt >= k) {
                    union.union(i, j);
                }
            }
        }
        return union.getCount();
    }

    public class Union3493 {
        private int[] parent;
        private int[] rank;
        private int n;

        public Union3493(int n) {
            this.parent = new int[n];
            this.rank = new int[n];
            for (int i = 0; i < n; ++i) {
                this.parent[i] = i;
                this.rank[i] = 1;
            }
            this.n = n;
        }

        public int getRoot(int p) {
            if (p == parent[p]) {
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
                    rank[root1]++;
                }
            }
            --n;
        }

        public int getCount() {
            return n;
        }

    }

    // 3494. 酿造药水需要的最少总时间 (Find the Minimum Amount of Time to Brew Potions)
    public long minTime(int[] skill, int[] mana) {
        int n = skill.length;
        long[] pre = new long[n];
        for (int m : mana) {
            long s = 0L;
            for (int i = 0; i < skill.length; ++i) {
                s = Math.max(s, pre[i]);
                s += skill[i] * m;
            }
            pre[n - 1] = s;
            for (int i = n - 2; i >= 0; --i) {
                s -= skill[i + 1] * m;
                pre[i] = s;
            }
        }
        return pre[n - 1];

    }

    public int countPrefixes(String[] words, String s) {
        Trie trie = new Trie();
        for (String word : words) {
            trie.insert(word);
        }
        return trie.countPrefixes(s);
    }

    public class Trie {
        private Trie[] children;
        private int end;

        public Trie() {
            this.children = new Trie[26];
        }

        public void insert(String s) {
            Trie node = this;
            for (char c : s.toCharArray()) {
                int idx = c - 'a';
                if (node.children[idx] == null) {
                    node.children[idx] = new Trie();
                }
                node = node.children[idx];
            }
            ++node.end;
        }

        public int countPrefixes(String s) {
            Trie node = this;
            int cnt = 0;
            for (char c : s.toCharArray()) {
                int idx = c - 'a';
                if (node.children[idx] == null) {
                    break;
                }
                node = node.children[idx];
                cnt += node.end;
            }
            return cnt;
        }

    }

    // 3498. 字符串的反转度 (Reverse Degree of a String)
    public int reverseDegree(String s) {
        int res = 0;
        for (int i = 0; i < s.length(); ++i) {
            res += (i + 1) * (26 - (s.charAt(i) - 'a'));
        }
        return res;

    }

    // 3499. 操作后最大活跃区段数 I (Maximize Active Section with Trade I)
    public int maxActiveSectionsAfterTrade(String s) {
        int n = s.length();
        int pre = -1;
        int cnt1 = 0;
        int res = 0;
        int i = 0;
        while (i < n) {
            if (s.charAt(i) == '0') {
                int cnt = 0;
                while (i < n && s.charAt(i) == '0') {
                    ++cnt;
                    ++i;
                }
                if (pre != -1) {
                    res = Math.max(res, cnt + pre);
                }
                pre = cnt;
            } else {
                ++i;
                ++cnt1;
            }
        }
        return res + cnt1;
    }

    // 3502. 到达每个位置的最小费用 (Minimum Cost to Reach Every Position)
    public int[] minCosts(int[] cost) {
        int n = cost.length;
        int[] res = new int[n];
        res[0] = cost[0];
        for (int i = 1; i < n; ++i) {
            res[i] = Math.min(res[i - 1], cost[i]);
        }
        return res;
    }

    // 3503. 子字符串连接后的最长回文串 I (Longest Palindrome After Substring Concatenation I)
    // 3504. 子字符串连接后的最长回文串 II (Longest Palindrome After Substring Concatenation II)
    private int calc3503(String S, String T) {
        int ans = 0;
        char[] s = S.toCharArray();
        char[] t = T.toCharArray();
        int n = s.length;
        int m = t.length;
        int[] mx = new int[n + 1];
        int[][] f = new int[n + 1][m + 1];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (s[i] == t[j]) {
                    f[i + 1][j] = f[i][j + 1] + 1;
                    mx[i + 1] = Math.max(mx[i + 1], f[i + 1][j]);
                }
            }
            ans = Math.max(ans, mx[i + 1] * 2); // |x| = |y| 的情况
        }

        // 计算 |x| > |y| 的情况，中心扩展法
        for (int i = 0; i < 2 * n - 1; i++) {
            int l = i / 2, r = (i + 1) / 2;
            while (l >= 0 && r < n && s[l] == s[r]) {
                l--;
                r++;
            }
            if (l + 1 <= r - 1) { // s[l+1] 到 s[r-1] 是非空回文串
                ans = Math.max(ans, r - l - 1 + mx[l + 1] * 2);
            }
        }
        return ans;
    }

    public int longestPalindrome(String s, String t) {
        String revS = new StringBuilder(s).reverse().toString();
        String revT = new StringBuilder(t).reverse().toString();
        return Math.max(calc3503(s, t), calc3503(revT, revS));
    }

    // 3512. 使数组和能被 K 整除的最少操作次数 (Minimum Operations to Make Array Sum Divisible by
    // K)
    public int minOperations3512(int[] nums, int k) {
        int s = Arrays.stream(nums).sum();
        return s % k;
    }

    // 3513. 不同 XOR 三元组的数目 I (Number of Unique XOR Triplets I)
    public int uniqueXorTriplets(int[] nums) {
        int n = nums.length;
        return n <= 2 ? n : 1 << Integer.toBinaryString(n).length();
    }

    // 3514. 不同 XOR 三元组的数目 II (Number of Unique XOR Triplets II)
    public int uniqueXorTriplets3514(int[] nums) {
        int n = nums.length;
        int mx = Arrays.stream(nums).max().getAsInt();
        int u = 1 << (32 - Integer.numberOfLeadingZeros(mx));
        boolean[] has = new boolean[u];
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                has[nums[i] ^ nums[j]] = true;
            }
        }
        boolean[] has3 = new boolean[u];
        for (int xy = 0; xy < u; ++xy) {
            if (!has[xy]) {
                continue;
            }
            for (int z : nums) {
                has3[xy ^ z] = true;
            }
        }
        int res = 0;
        for (boolean b : has3) {
            if (b) {
                ++res;
            }
        }
        return res;

    }

    // 3516. 找到最近的人 (Find Closest Person)
    public int findClosest(int x, int y, int z) {
        return x == y || x - z == z - y ? 0 : Math.abs(x - z) < Math.abs(y - z) ? 1 : 2;
    }

    // 3517. 最小回文排列 I (Smallest Palindromic Rearrangement I)
    public String smallestPalindrome(String s) {
        int[] cnt = new int[26];
        for (char c : s.toCharArray()) {
            ++cnt[c - 'a'];
        }
        int n = s.length();
        char[] res = new char[n];
        int l = 0;
        int r = n - 1;
        for (int i = 0; i < 26; ++i) {
            while (cnt[i] > 1) {
                res[l++] = (char) (i + 'a');
                res[r--] = (char) (i + 'a');
                cnt[i] -= 2;
            }
            if (cnt[i] == 1) {
                res[n / 2] = (char) (i + 'a');
                cnt[i]--;
            }
        }
        return new String(res);

    }

    // 3519. 统计逐位非递减的整数 (Count Numbers with Non-Decreasing Digits)
    private int b3519;

    public int countNumbers(String l, String r, int b) {
        final int MOD = (int) (1e9 + 7);
        this.b3519 = b;
        return ((cal3519(r) - cal3519(l) + check3519(l)) % MOD + MOD) % MOD;
    }

    private char[] trans(String s) {
        return new BigInteger(s).toString(b3519).toCharArray();
    }

    private int check3519(String s) {
        char[] arr = trans(s);
        for (int i = 1; i < arr.length; ++i) {
            if (arr[i] < arr[i - 1]) {
                return 0;
            }
        }
        return 1;
    }

    private char[] arr3519;
    private int n3519;
    private int[][] memo3519;

    private int cal3519(String s) {
        this.arr3519 = trans(s);
        this.n3519 = arr3519.length;
        this.memo3519 = new int[n3519][b3519];
        for (int i = 0; i < n3519; ++i) {
            Arrays.fill(memo3519[i], -1);
        }
        return dfs3519(0, 0, true, false);
    }

    private int dfs3519(int i, int j, boolean isLimit, boolean isNum) {
        if (i == n3519) {
            return 1;
        }
        if (!isLimit && isNum && memo3519[i][j] != -1) {
            return memo3519[i][j];
        }
        int res = 0;
        if (!isNum) {
            res = dfs3519(i + 1, j, false, false);
        }
        final int MOD = (int) (1e9 + 7);
        int up = isLimit ? arr3519[i] - '0' : b3519 - 1;
        for (int d = Math.max(j, isNum ? 0 : 1); d <= up; ++d) {
            res += dfs3519(i + 1, d, isLimit && d == up, true);
            res %= MOD;
        }
        if (!isLimit && isNum) {
            memo3519[i][j] = res;
        }
        return res;
    }

    // 3522. 执行指令后的得分 (Calculate Score After Performing Instructions)
    public long calculateScore(String[] instructions, int[] values) {
        int n = instructions.length;
        boolean[] vis = new boolean[n];
        int i = 0;
        long res = 0L;
        while (i < n && i >= 0 && !vis[i]) {
            vis[i] = true;
            if ("add".equals(instructions[i])) {
                res += values[i];
                ++i;
            } else {
                i += values[i];
            }
        }
        return res;

    }

    // 3523. 非递减数组的最大长度 (Make Array Non-decreasing)
    public int maximumPossibleSize(int[] nums) {
        int res = 0;
        int mx = 0;
        for (int x : nums) {
            if (x >= mx) {
                mx = x;
                ++res;
            }
        }
        return res;

    }

    // 1399. 统计最大组的数目 (Count Largest Group)
    private String s1399;
    private int l1399;
    private int[][] memo1399;
    private int sum1399;

    public int countLargestGroup(int n) {
        this.s1399 = String.valueOf(n);
        this.l1399 = s1399.length();
        int mx = 0;
        int res = 0;
        for (int i = 1; i < l1399 * 9 + 1; ++i) {
            this.sum1399 = i;
            this.memo1399 = new int[l1399][i + 1];
            for (int j = 0; j < l1399; ++j) {
                Arrays.fill(memo1399[j], -1);
            }
            int cur = dfs1399(0, 0, true);
            if (cur > mx) {
                mx = cur;
                res = 1;
            } else if (cur == mx) {
                ++res;
            }
        }
        return res;
    }

    private int dfs1399(int i, int j, boolean isLimit) {
        if (i == l1399) {
            return j == sum1399 ? 1 : 0;
        }
        if (!isLimit && memo1399[i][j] != -1) {
            return memo1399[i][j];
        }
        int res = 0;
        int up = Math.min(sum1399 - j, isLimit ? s1399.charAt(i) - '0' : 9);
        for (int d = 0; d <= up; ++d) {
            res += dfs1399(i + 1, j + d, isLimit && d == up);
        }
        if (!isLimit) {
            memo1399[i][j] = res;
        }
        return res;
    }

    // 3527. 找到最常见的回答 (Find the Most Common Response)
    public String findCommonResponse(List<List<String>> responses) {
        int mx = 0;
        Map<String, Integer> cnts = new HashMap<>();
        for (List<String> response : responses) {
            Set<String> set = new HashSet<>(response);
            for (String s : set) {
                cnts.merge(s, 1, Integer::sum);
                mx = Math.max(mx, cnts.get(s));
            }
        }
        String res = "";
        for (Map.Entry<String, Integer> entry : cnts.entrySet()) {
            if (entry.getValue() == mx) {
                if (res.isEmpty() || entry.getKey().compareTo(res) < 0) {
                    res = entry.getKey();
                }
            }
        }
        return res;
    }

    // 3528. 单位转换 I (Unit Conversion I)
    public int[] baseUnitConversions(int[][] conversions) {
        int n = conversions.length + 1;
        List<int[]>[] g = new ArrayList[n];
        Arrays.setAll(g, k -> new ArrayList<>());
        for (int[] c : conversions) {
            g[c[0]].add(new int[] { c[1], c[2] });
        }
        int[] res = new int[n];
        dfs3528(0, 1, g, res);
        return res;

    }

    private void dfs3528(int x, int v, List<int[]>[] g, int[] res) {
        final int MOD = (int) (1e9 + 7);
        res[x] = v;
        for (int[] neighbor : g[x]) {
            int y = neighbor[0];
            int w = neighbor[1];
            dfs3528(y, (int) ((long) v * w % MOD), g, res);
        }
    }

    // 3529. 统计水平子串和垂直子串重叠格子的数目 (Count Cells in Overlapping Horizontal and Vertical
    // Substrings) --z函数 + 差分数组
    public int countCells(char[][] grid, String pattern) {
        int pLen = pattern.length();
        int m = grid.length;
        int n = grid[0].length;
        int[] diff1 = algorithmZ3529(pattern + check_3529(grid), pLen);
        int[] diff2 = algorithmZ3529(pattern + check2_3529(grid), pLen);
        int res = 0;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                if (diff2[j * m + i] > 0 && diff1[i * n + j] > 0) {
                    ++res;
                }
            }
        }
        return res;
    }

    private String check2_3529(char[][] grid) {
        StringBuilder sb = new StringBuilder();
        for (int j = 0; j < grid[0].length; ++j) {
            for (int i = 0; i < grid.length; ++i) {
                sb.append(grid[i][j]);
            }
        }
        return sb.toString();
    }

    private String check_3529(char[][] grid) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < grid.length; ++i) {
            for (int j = 0; j < grid[0].length; ++j) {
                sb.append(grid[i][j]);
            }
        }
        return sb.toString();
    }

    private int[] algorithmZ3529(String s, int pLen) {
        int left = 0;
        int right = 0;
        int[] z = new int[s.length()];
        int[] diff = new int[s.length() - pLen + 1];
        for (int i = 1; i < s.length(); ++i) {
            if (i <= right) {
                z[i] = Math.min(z[i - left], right - i + 1);
            }
            while (i + z[i] < s.length() && s.charAt(z[i]) == s.charAt(i + z[i])) {
                left = i;
                right = i + z[i];
                ++z[i];
            }
            if (Math.min(i, z[i]) >= pLen) {
                diff[i - pLen] += 1;
                diff[i] -= 1;
            }
        }
        for (int i = 1; i < diff.length; ++i) {
            diff[i] += diff[i - 1];
        }
        return diff;
    }

    // 3530. 有向无环图中合法拓扑排序的最大利润 (Maximum Profit from Valid Topological Order in DAG)
    private int[] memo3530;
    private int u3530;
    private int[] score3530;
    private int[] pre3530;

    public int maxProfit(int n, int[][] edges, int[] score) {
        if (edges.length == 0) {
            Arrays.sort(score);
            int res = 0;
            for (int i = 0; i < n; ++i) {
                res += score[i] * (i + 1);
            }
            return res;
        }
        this.pre3530 = new int[n];
        for (int[] edge : edges) {
            pre3530[edge[1]] |= 1 << edge[0];
        }
        this.memo3530 = new int[1 << n];
        this.u3530 = (1 << n) - 1;
        this.score3530 = score;
        return dfs3530(0);
    }

    private int dfs3530(int i) {
        if (i == u3530) {
            return 0;
        }
        if (memo3530[i] != 0) {
            return memo3530[i];
        }
        int res = 0;
        for (int c = u3530 ^ i; c != 0; c &= c - 1) {
            int lb = Integer.numberOfTrailingZeros(c);
            // lb未选择、且lb的直接祖先节点均已选择
            if ((pre3530[lb] | i) == i) {
                res = Math.max(res, dfs3530(i | (1 << lb)) + score3530[lb] * (Integer.bitCount(i) + 1));
            }
        }
        return memo3530[i] = res;
    }

    // 3531. 统计被覆盖的建筑 (Count Covered Buildings)
    public int countCoveredBuildings(int n, int[][] buildings) {
        int res = 0;
        int[][] col = new int[2][n + 1];
        Arrays.fill(col[0], Integer.MAX_VALUE);
        int[][] row = new int[2][n + 1];
        Arrays.fill(row[0], Integer.MAX_VALUE);
        for (int[] b : buildings) {
            col[0][b[0]] = Math.min(col[0][b[0]], b[1]);
            col[1][b[0]] = Math.max(col[1][b[0]], b[1]);
            row[0][b[1]] = Math.min(row[0][b[1]], b[0]);
            row[1][b[1]] = Math.max(row[1][b[1]], b[0]);
        }
        for (int[] b : buildings) {
            if (b[1] == col[0][b[0]] || b[1] == col[1][b[0]]) {
                continue;
            }
            if (b[0] == row[0][b[1]] || b[0] == row[1][b[1]]) {
                continue;
            }
            ++res;
        }
        return res;
    }

    // 3532. 针对图的路径存在性查询 I (Path Existence Queries in a Graph I)
    public boolean[] pathExistenceQueries(int n, int[] nums, int maxDiff, int[][] queries) {
        Integer[] idx = IntStream.range(0, n).boxed().toArray(Integer[]::new);
        Arrays.sort(idx, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(nums[o1], nums[o2]);
            }

        });
        Union3532 union = new Union3532(n);
        for (int i = 1; i < n; ++i) {
            if (nums[idx[i]] - nums[idx[i - 1]] <= maxDiff) {
                union.union(idx[i], idx[i - 1]);
            }
        }
        boolean[] res = new boolean[queries.length];
        for (int i = 0; i < queries.length; ++i) {
            res[i] = union.isConnected(queries[i][0], queries[i][1]);
        }
        return res;

    }

    public class Union3532 {
        private int[] rank;
        private int[] parent;

        public Union3532(int n) {
            this.rank = new int[n];
            Arrays.fill(rank, 1);
            this.parent = new int[n];
            for (int i = 0; i < n; ++i) {
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

    // 3533. 判断连接可整除性 (Concatenated Divisibility)
    private int[] nums3533;
    private int k3533;
    private int[][] memo3533;
    private int n3533;
    private int[] pow10_3533;
    private int[] res3533;

    public int[] concatenatedDivisibility(int[] nums, int k) {
        Arrays.sort(nums);
        this.nums3533 = nums;
        this.n3533 = nums.length;
        this.k3533 = k;
        this.memo3533 = new int[1 << n3533][k];
        this.pow10_3533 = new int[n3533];
        this.res3533 = new int[n3533];
        for (int i = 0; i < n3533; ++i) {
            pow10_3533[i] = (int) Math.pow(10, String.valueOf(nums[i]).length());
        }
        if (!dfs3533(0, 0)) {
            return new int[0];
        }
        for (int i = 0; i < (n3533 >> 1); ++i) {
            int tmp = nums[i];
            nums[i] = nums[n3533 - 1 - i];
            nums[n3533 - 1 - i] = tmp;
        }
        return res3533;

    }

    private boolean dfs3533(int i, int j) {
        if (i == (1 << n3533) - 1) {
            return j == 0;
        }
        if (memo3533[i][j] != 0) {
            return memo3533[i][j] > 0;
        }
        for (int id = 0; id < n3533; ++id) {
            if (((i >> id) & 1) == 0 && dfs3533(i | (1 << id), (j * pow10_3533[id] + nums3533[id]) % k3533)) {
                res3533[Integer.bitCount(i)] = nums3533[id];
                memo3533[i][j] = 1;
                return true;
            }
        }
        memo3533[i][j] = -1;
        return false;
    }

    // 3536. 两个数字的最大乘积 (Maximum Product of Two Digits)
    public int maxProduct(int n) {
        int[] cnts = new int[10];
        while (n != 0) {
            ++cnts[n % 10];
            n /= 10;
        }
        int[] ret = new int[2];
        int j = 0;
        for (int i = 9; i >= 0; --i) {
            while (cnts[i] > 0 && j < 2) {
                ret[j++] = i;
                --cnts[i];
            }
            if (j == 2) {
                break;
            }
        }
        return ret[0] * ret[1];

    }

    // 3537. 填充特殊网格 (Fill a Special Grid)
    private int cur3537;
    private int[][] res3537;

    public int[][] specialGrid(int n) {
        this.res3537 = new int[1 << n][1 << n];
        dfs3537(0, 0, (1 << n) - 1, (1 << n) - 1);
        return res3537;

    }

    private void dfs3537(int i0, int j0, int i1, int j1) {
        if (i0 == i1 && j0 == j1) {
            res3537[i0][j0] = cur3537++;
            return;
        }
        // 右上
        int newI0 = i0;
        int newJ0 = (j0 + j1) / 2 + 1;
        int newI1 = (i0 + i1) / 2;
        int newJ1 = j1;
        dfs3537(newI0, newJ0, newI1, newJ1);

        // 右下
        newI0 = (i0 + i1) / 2 + 1;
        newJ0 = (j0 + j1) / 2 + 1;
        newI1 = i1;
        newJ1 = j1;
        dfs3537(newI0, newJ0, newI1, newJ1);

        // 左下
        newI0 = (i0 + i1) / 2 + 1;
        newJ0 = j0;
        newI1 = i1;
        newJ1 = (j0 + j1) / 2;
        dfs3537(newI0, newJ0, newI1, newJ1);

        // 左上
        newI0 = i0;
        newJ0 = j0;
        newI1 = (i0 + i1) / 2;
        newJ1 = (j0 + j1) / 2;
        dfs3537(newI0, newJ0, newI1, newJ1);

    }

    // 3538. 合并得到最小旅行时间 (Merge Operations for Minimum Travel Time)
    private int[] pre3538;
    private int[] pos3538;
    private int n3538;
    private int[][][] memo3538;

    public int minTravelTime(int l, int n, int k, int[] position, int[] time) {
        this.pre3538 = new int[n + 1];
        for (int i = 0; i < n; ++i) {
            pre3538[i + 1] = pre3538[i] + time[i];
        }
        this.pos3538 = position;
        this.n3538 = n;
        this.memo3538 = new int[n - 1][n - 1][k + 1];
        for (int i = 0; i < n - 1; ++i) {
            for (int j = 0; j < n - 1; ++j) {
                Arrays.fill(memo3538[i][j], -1);
            }
        }
        return dfs3538(0, 0, k);

    }

    private int dfs3538(int i, int j, int leftK) {
        if (j == n3538 - 1) {
            return leftK == 0 ? 0 : (int) 1e9;
        }
        if (memo3538[i][j][leftK] != -1) {
            return memo3538[i][j][leftK];
        }
        int t = pre3538[j + 1] - pre3538[i];
        int res = (int) 1e9;
        for (int k = j + 1; k < Math.min(n3538, j + 2 + leftK); ++k) {
            res = Math.min(res, dfs3538(j + 1, k, leftK - (k - j - 1)) + (pos3538[k] - pos3538[j]) * t);
        }
        return memo3538[i][j][leftK] = res;
    }

    // 3335. 字符串转换后的长度 I (Total Characters in String After Transformations I)
    public int lengthAfterTransformations(String s, int t) {
        int[] cnts = new int[26];
        for (char c : s.toCharArray()) {
            ++cnts[c - 'a'];
        }
        final int MOD = (int) (1e9 + 7);
        int res = 0;
        for (int r = 0; r < t; ++r) {
            int[] nxt = new int[26];
            nxt[0] = cnts[25];
            nxt[1] = (cnts[0] + cnts[25]) % MOD;
            for (int i = 2; i < 26; ++i) {
                nxt[i] = cnts[i - 1];
            }
            cnts = nxt;
        }
        for (int c : cnts) {
            res += c;
            res %= MOD;
        }
        return res;

    }

    // 3541. 找到频率最高的元音和辅音 (Find Most Frequent Vowel and Consonant)
    public int maxFreqSum(String s) {
        int[] cnts = new int[26];
        int c0 = 0;
        int c1 = 0;
        for (char c : s.toCharArray()) {
            ++cnts[c - 'a'];
            if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
                c0 = Math.max(c0, cnts[c - 'a']);
            } else {
                c1 = Math.max(c1, cnts[c - 'a']);
            }
        }
        return c0 + c1;

    }

    // 3543. K 条边路径的最大边权和 (Maximum Weighted K-Edge Path) --dfs
    private List<int[]>[] g3543;
    private int k3543;
    private int t3543;
    private boolean[][][] memo3543;
    private int res3543;

    public int maxWeight(int n, int[][] edges, int k, int t) {
        this.g3543 = new ArrayList[n];
        this.k3543 = k;
        this.t3543 = t;
        this.memo3543 = new boolean[n][k][t];
        Arrays.setAll(g3543, o -> new ArrayList<>());
        for (int[] e : edges) {
            g3543[e[0]].add(new int[] { e[1], e[2] });
        }
        res3543 = -1;
        for (int i = 0; i < n; ++i) {
            dfs3543(i, 0, 0);
        }
        return res3543;
    }

    private void dfs3543(int i, int j, int w) {
        if (j == k3543) {
            res3543 = Math.max(res3543, w);
            return;
        }
        if (memo3543[i][j][w]) {
            return;
        }
        for (int[] nxt : g3543[i]) {
            int y = nxt[0];
            int nw = nxt[1];
            if (w + nw < t3543) {
                dfs3543(y, j + 1, w + nw);
            }
        }
        memo3543[i][j][w] = true;
    }

    // 3543. K 条边路径的最大边权和 (Maximum Weighted K-Edge Path) --拓扑排序
    public int maxWeight2(int n, int[][] edges, int k, int t) {
        List<int[]>[] g = new ArrayList[n];
        Arrays.setAll(g, o -> new ArrayList<>());
        for (int[] e : edges) {
            g[e[0]].add(new int[] { e[1], e[2] });
        }
        boolean[][][] vis = new boolean[n][k + 1][t];
        int res = -1;
        for (int i = 0; i < n; ++i) {
            res = Math.max(res, bfs3543(i, g, k, t, vis));
        }
        return res;
    }

    private int bfs3543(int start, List<int[]>[] g, int k, int t, boolean[][][] vis) {
        vis[start][0][0] = true;
        Queue<int[]> q = new LinkedList<>();
        q.offer(new int[] { start, 0, 0 });
        int res = -1;
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int s = cur[1];
            int w = cur[2];
            if (s == k) {
                res = Math.max(res, w);
                continue;
            }
            for (int[] nxt : g[x]) {
                int y = nxt[0];
                int wd = nxt[1];
                if (w + wd < t && !vis[y][s + 1][w + wd]) {
                    vis[y][s + 1][w + wd] = true;
                    q.offer(new int[] { y, s + 1, w + wd });
                }
            }
        }
        return res;
    }

    // 3542. 将所有元素变为 0 的最少操作次数 (Minimum Operations to Convert All Elements to Zero)
    public int minOperations3542(int[] nums) {
        Stack<Integer> st = new Stack<>();
        int res = 0;
        for (int x : nums) {
            while (!st.isEmpty() && st.peek() > x) {
                st.pop();
                ++res;
            }
            if (st.isEmpty() || st.peek() != x) {
                st.push(x);
            }
        }
        return res + st.size() - (st.get(0) == 0 ? 1 : 0);
    }

    // 3544. 子树反转和 (Subtree Inversion Sum)
    private List<Integer>[] g3544;
    private long[][][] memo3544;
    private int[] nums3544;
    private int k3544;

    public long subtreeInversionSum(int[][] edges, int[] nums, int k) {
        int n = nums.length;
        this.g3544 = new ArrayList[n];
        Arrays.setAll(g3544, o -> new ArrayList<>());
        for (int[] e : edges) {
            g3544[e[0]].add(e[1]);
            g3544[e[1]].add(e[0]);
        }
        this.k3544 = k;
        this.memo3544 = new long[n][k][2];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                Arrays.fill(memo3544[i][j], Long.MIN_VALUE);
            }
        }
        this.nums3544 = nums;
        return dfs3544(0, -1, 0, 1);

    }

    private long dfs3544(int x, int fa, int left, int p) {
        if (memo3544[x][left][p] != Long.MIN_VALUE) {
            return memo3544[x][left][p];
        }
        // 不反转
        long res = nums3544[x] * (p * 2 - 1);
        for (int y : g3544[x]) {
            if (y != fa) {
                res += dfs3544(y, x, Math.max(left - 1, 0), p);
            }

        }
        // 反转
        if (left == 0) {
            long s = nums3544[x] * -(p * 2 - 1);
            for (int y : g3544[x]) {
                if (y != fa) {
                    s += dfs3544(y, x, k3544 - 1, p ^ 1);
                }
            }
            if (s > res) {
                res = s;
            }
        }
        return memo3544[x][left][p] = res;
    }

    // 3545. 不同字符数量最多为 K 时的最少删除数 (Minimum Deletions for At Most K Distinct
    // Characters)
    public int minDeletion(String s, int k) {
        int[] cnts = new int[26];
        for (char c : s.toCharArray()) {
            ++cnts[c - 'a'];
        }
        Arrays.sort(cnts);
        int res = 0;
        for (int i = 25 - k; i >= 0; --i) {
            res += cnts[i];
        }
        return res;

    }

    // 3546. 等和矩阵分割 I (Equal Sum Grid Partition I)
    public boolean canPartitionGrid(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        long[] row = new long[m];
        long[] col = new long[n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                row[i] += grid[i][j];
                col[j] += grid[i][j];
            }
        }
        return check3546(row) || check3546(col);

    }

    private boolean check3546(long[] arr) {
        long pre = 0L;
        long suf = Arrays.stream(arr).sum();
        for (long x : arr) {
            pre += x;
            suf -= x;
            if (pre == suf) {
                return true;
            }
        }
        return false;
    }

    // 3547. 图中边值的最大和 (Maximum Sum of Edge Values in a Graph)
    public long maxScore(int n, int[][] edges) {
        long res = ((long) n * n * 2 + n * 5 - 6) * (n - 1) / 6;
        if (n == edges.length) {
            res += 2;
        }
        return res;
    }

    // 3550. 数位和等于下标的最小下标 (Smallest Index With Digit Sum Equal to Index)
    public int smallestIndex(int[] nums) {
        search: for (int i = 0; i < nums.length; ++i) {
            int s = 0;
            while (nums[i] != 0) {
                s += nums[i] % 10;
                nums[i] /= 10;
                if (s > i) {
                    continue search;
                }
            }
            if (s == i) {
                return i;
            }
        }
        return -1;

    }

    // 3551. 数位和排序需要的最小交换次数 (Minimum Swaps to Sort by Digit Sum)
    public int minSwaps(int[] nums) {
        int n = nums.length;
        int[][] arr = new int[n][2];
        for (int i = 0; i < n; ++i) {
            arr[i][0] = nums[i];
            arr[i][1] = i;
        }
        Arrays.sort(arr, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                int a1 = o1[0];
                int a2 = o2[0];
                int s1 = 0;
                while (a1 != 0) {
                    s1 += a1 % 10;
                    a1 /= 10;
                }
                int s2 = 0;
                while (a2 != 0) {
                    s2 += a2 % 10;
                    a2 /= 10;
                }
                if (s1 == s2) {
                    return Integer.compare(o1[0], o2[0]);
                }
                return Integer.compare(s1, s2);
            }

        });
        // System.out.println(Arrays.deepToString(arr));
        int res = n;
        boolean[] vis = new boolean[n];
        for (int i = 0; i < n; ++i) {
            if (!vis[i]) {
                --res;
                int j = i;
                while (!vis[j]) {
                    vis[j] = true;
                    j = arr[j][1];
                }
            }
        }
        return res;

    }

    // 3552. 网格传送门旅游 (Grid Teleportation Traversal)
    public int minMoves(String[] matrix) {
        int m = matrix.length;
        int n = matrix[0].length();
        if (matrix[m - 1].charAt(n - 1) == '#') {
            return -1;
        }
        int[][] dis = new int[m][n];
        for (int i = 0; i < m; ++i) {
            Arrays.fill(dis[i], Integer.MAX_VALUE);
        }
        dis[0][0] = 0;
        Map<Integer, List<int[]>> map = new HashMap<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (matrix[i].charAt(j) == '#' || matrix[i].charAt(j) == '.') {
                    continue;
                }
                int d = matrix[i].charAt(j) - 'A';
                map.computeIfAbsent(d, k -> new ArrayList<>()).add(new int[] { i, j });
            }
        }
        Deque<int[]> q = new LinkedList<>();
        int[][] dirs = { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 } };
        q.offer(new int[] { 0, 0, 0 });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int i = cur[0];
            int j = cur[1];
            int d = cur[2];
            if (i == m - 1 && j == n - 1) {
                return d;
            }
            for (int[] dir : dirs) {
                int ni = i + dir[0];
                int nj = j + dir[1];
                if (ni >= 0 && ni < m && nj >= 0 && nj < n && matrix[ni].charAt(nj) != '#') {
                    if (d + 1 < dis[ni][nj]) {
                        dis[ni][nj] = d + 1;
                        q.addLast(new int[] { ni, nj, d + 1 });
                    }
                }
            }
            if (Character.isUpperCase(matrix[i].charAt(j)) && map.containsKey(matrix[i].charAt(j) - 'A')) {
                List<int[]> nxt = map.get(matrix[i].charAt(j) - 'A');
                for (int[] pos : nxt) {
                    int ni = pos[0];
                    int nj = pos[1];
                    if (d < dis[ni][nj]) {
                        dis[ni][nj] = d;
                        q.addFirst(new int[] { ni, nj, d });
                    }
                }
                map.remove(matrix[i].charAt(j) - 'A');
            }

        }
        return -1;

    }

    // 3556. 最大质数子字符串之和 (Sum of Largest Prime Substrings)
    public long sumOfLargestPrimes(String s) {
        int n = s.length();
        TreeSet<Long> set = new TreeSet<>(new Comparator<Long>() {

            @Override
            public int compare(Long o1, Long o2) {
                return Long.compare(o2, o1);
            }

        });
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                long num = Long.parseLong(s.substring(i, j + 1));
                if (isPrime3556(num)) {
                    set.add(num);
                    if (set.size() > 3) {
                        set.pollLast();
                    }
                }
            }
        }
        return set.stream().mapToLong(Long::valueOf).sum();

    }

    private boolean isPrime3556(long x) {
        if (x < 2) {
            return false;
        }
        for (int i = 2; (long) i * i <= x; ++i) {
            if (x % i == 0) {
                return false;
            }
        }
        return true;
    }

    // 3557. 不相交子字符串的最大数量 (Find Maximum Number of Non Intersecting Substrings)
    public int maxSubstrings(String word) {
        int res = 0;
        Map<Character, Integer> pos = new HashMap<>();
        for (int i = 0; i < word.length(); ++i) {
            if (pos.containsKey(word.charAt(i))) {
                if (i - pos.get(word.charAt(i)) > 2) {
                    ++res;
                    pos.clear();
                }
            } else {
                pos.put(word.charAt(i), i);
            }
        }
        return res;

    }

    // 3558. 给边赋权值的方案数 I (Number of Ways to Assign Edge Weights I)
    private int mx3558;
    private int[][] memo3558;

    public int assignEdgeWeights(int[][] edges) {
        int n = edges.length + 1;
        List<Integer>[] g = new ArrayList[n];
        Arrays.setAll(g, k -> new ArrayList<>());
        for (int[] e : edges) {
            g[e[0] - 1].add(e[1] - 1);
            g[e[1] - 1].add(e[0] - 1);
        }
        maxDepth3558(0, -1, 0, g);
        this.memo3558 = new int[mx3558][2];
        for (int i = 0; i < mx3558; ++i) {
            Arrays.fill(memo3558[i], -1);
        }
        return dfs3558(0, 0);

    }

    private int dfs3558(int i, int j) {
        if (i == mx3558) {
            return j & 1;
        }
        if (memo3558[i][j] != -1) {
            return memo3558[i][j];
        }
        final int MOD = (int) (1e9 + 7);
        return memo3558[i][j] = (dfs3558(i + 1, j) + dfs3558(i + 1, j ^ 1)) % MOD;
    }

    private void maxDepth3558(int x, int fa, int d, List<Integer>[] g) {
        this.mx3558 = Math.max(mx3558, d);
        for (int y : g[x]) {
            if (y != fa) {
                maxDepth3558(y, x, d + 1, g);
            }
        }
    }

    // 3560. 木材运输的最小成本 (Find Minimum Log Transportation Cost)
    public long minCuttingCost(int n, int m, int k) {
        long res = 0L;
        if (n > k) {
            res += (long) k * (n - k);
        }
        if (m > k) {
            res += (long) k * (m - k);
        }
        return res;
    }

    // 3561. 移除相邻字符 (Resulting String After Adjacent Removals)
    public String resultingString(String s) {
        List<Character> st = new ArrayList<>();
        for (char c : s.toCharArray()) {
            if (!st.isEmpty() && isConsecutive3561(st.get(st.size() - 1), c)) {
                st.remove(st.size() - 1);
            } else {
                st.add(c);
            }
        }
        return st.stream().map(String::valueOf).collect(Collectors.joining());

    }

    private boolean isConsecutive3561(char a, char b) {
        int d = Math.abs(a - b);
        return d == 1 || d == 25;
    }

    // 3563. 移除相邻字符后字典序最小的字符串 (Lexicographically Smallest String After Adjacent
    // Removals)
    private char[] arr3563;
    private int n3563;
    private int[][] memo3563;
    private String[] memoDfs3563;

    public String lexicographicallySmallestString(String s) {
        this.n3563 = s.length();
        this.arr3563 = s.toCharArray();
        this.memo3563 = new int[n3563][n3563];
        this.memoDfs3563 = new String[n3563];
        return dfs3563(0);
    }

    private String dfs3563(int i) {
        if (i == n3563) {
            return "";
        }
        if (memoDfs3563[i] != null) {
            return memoDfs3563[i];
        }
        String res = String.valueOf(arr3563[i]) + dfs3563(i + 1);
        for (int j = i + 1; j < n3563; j += 2) {
            if (check3563(i, j)) {
                String t = dfs3563(j + 1);
                if (t.compareTo(res) < 0) {
                    res = t;
                }
            }
        }
        return memoDfs3563[i] = res;
    }

    private boolean check3563(int i, int j) {
        if (i > j) {
            return true;
        }
        if (memo3563[i][j] != 0) {
            return memo3563[i][j] > 0;
        }
        if (isConsecutive3563(i, j) && check3563(i + 1, j - 1)) {
            memo3563[i][j] = 1;
            return true;
        }
        for (int k = i + 1; k < j; k += 2) {
            if (check3563(i, k) && check3563(k + 1, j)) {
                memo3563[i][j] = 1;
                return true;
            }
        }
        memo3563[i][j] = -1;
        return false;
    }

    private boolean isConsecutive3563(int i, int j) {
        int d = Math.abs(arr3563[i] - arr3563[j]);
        return d == 1 || d == 25;
    }

    // 3566. 等积子集的划分方案 (Partition Array into Two Equal Product Subsets)
    public boolean checkEqualPartitions(int[] nums, long target) {
        long mul = 1L;
        for (int x : nums) {
            mul *= x;
        }
        if (mul != target * target) {
            return false;
        }
        int n = nums.length;
        return dfs3566(0, 1L, false, false, n, target, nums);

    }

    private boolean dfs3566(int i, long j, boolean k, boolean l, int n, long target, int[] nums) {
        if (i == n) {
            return k && l && j == target;
        }
        if (j > target) {
            return false;
        }
        return dfs3566(i + 1, j, true, l, n, target, nums) || dfs3566(i + 1, j * nums[i], k, true, n, target, nums);
    }

    // 3567. 子矩阵的最小绝对差 (Minimum Absolute Difference in Sliding Submatrix)
    public int[][] minAbsDiff(int[][] grid, int k) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] res = new int[m - k + 1][n - k + 1];
        for (int i = 0; i < m - k + 1; ++i) {
            for (int j = 0; j < n - k + 1; ++j) {
                List<Integer> list = new ArrayList<>();
                for (int ii = i; ii < i + k; ++ii) {
                    for (int jj = j; jj < j + k; ++jj) {
                        list.add(grid[ii][jj]);
                    }
                }
                Collections.sort(list);
                int min = Integer.MAX_VALUE;
                for (int l = 1; l < list.size(); ++l) {
                    if (list.get(l) > list.get(l - 1)) {
                        min = Math.min(min, list.get(l) - list.get(l - 1));
                    }
                }
                if (min != Integer.MAX_VALUE) {
                    res[i][j] = min;
                }
            }
        }
        return res;

    }

    // 3568. 清理教室的最少移动 (Minimum Moves to Clean the Classroom)
    public int minMoves(String[] classroom, int energy) {
        int m = classroom.length;
        int n = classroom[0].length();
        int[][] mask_clear = new int[m][n];
        int startX = -1;
        int startY = -1;
        int cnt = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (classroom[i].charAt(j) == 'S') {
                    startX = i;
                    startY = j;
                } else if (classroom[i].charAt(j) == 'L') {
                    mask_clear[i][j] = 1 << cnt;
                    ++cnt;
                }
            }
        }
        if (cnt == 0) {
            return 0;
        }
        int[][] dirs = { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 } };
        int u = (1 << cnt) - 1;
        int[][][] maxEnergy = new int[m][n][1 << cnt];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                Arrays.fill(maxEnergy[i][j], -1);
            }
        }
        maxEnergy[startX][startY][0] = energy;
        ;
        Queue<int[]> q = new LinkedList<>();
        int res = 0;
        // startX, startY, energy, mask
        q.offer(new int[] { startX, startY, energy, 0 });
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                int[] cur = q.poll();
                int x = cur[0];
                int y = cur[1];
                int e = cur[2];
                int mask = cur[3];
                if (mask == u) {
                    return res;
                }
                if (e == 0) {
                    continue;
                }
                for (int[] dir : dirs) {
                    int nx = x + dir[0];
                    int ny = y + dir[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && classroom[nx].charAt(ny) != 'X') {
                        int nmask = mask | mask_clear[nx][ny];
                        int ne = classroom[nx].charAt(ny) == 'R' ? energy : e - 1;
                        if (ne > maxEnergy[nx][ny][nmask]) {
                            maxEnergy[nx][ny][nmask] = ne;
                            q.offer(new int[] { nx, ny, ne, nmask });
                        }
                    }
                }
            }
            ++res;
        }
        return -1;

    }

    // 3565. 顺序网格路径覆盖 (Sequential Grid Path Cover) --plus
    private List<List<Integer>> resPath3565;
    private int[][] grid3565;
    private int m3565;
    private int n3565;

    public List<List<Integer>> findPath(int[][] grid, int k) {
        this.m3565 = grid.length;
        this.n3565 = grid[0].length;
        this.grid3565 = grid;
        resPath3565 = new ArrayList<>();
        for (int i = 0; i < m3565; ++i) {
            for (int j = 0; j < n3565; ++j) {
                if (grid[i][j] <= 1 && dfs3565(i, j, grid[i][j], 1L << (i * n3565 + j))) {
                    resPath3565.add(List.of(i, j));
                    Collections.reverse(resPath3565);
                    return resPath3565;
                }
                resPath3565.clear();
            }
        }
        return List.of();

    }

    private boolean dfs3565(int x, int y, int mx, long mask) {
        if (mask == (1L << (m3565 * n3565)) - 1) {
            return true;
        }
        int[][] dirs = { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 } };
        for (int[] dir : dirs) {
            int nx = x + dir[0];
            int ny = y + dir[1];
            if (nx >= 0 && nx < m3565 && ny >= 0 && ny < n3565 && ((mask >> (nx * n3565 + ny)) & 1) == 0) {
                if (grid3565[nx][ny] - mx == 1 || grid3565[nx][ny] == 0) {
                    if (dfs3565(nx, ny, Math.max(mx, grid3565[nx][ny]), mask | (1 << (nx * n3565 + ny)))) {
                        resPath3565.add(List.of(nx, ny));
                        return true;
                    }

                }
            }
        }
        return false;

    }

    // 3466. 最大硬币收集量 (Maximum Coin Collection) --plus
    private int n3466;
    private long[][][][] memo3466;
    private int[][] lane3466;

    public long maxCoins(int[] lane1, int[] lane2) {
        this.n3466 = lane1.length;
        this.lane3466 = new int[n3466][2];
        for (int i = 0; i < n3466; ++i) {
            lane3466[i][0] = lane1[i];
            lane3466[i][1] = lane2[i];
        }
        this.memo3466 = new long[n3466][2][3][2];
        for (int i = 0; i < n3466; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 3; ++k) {
                    Arrays.fill(memo3466[i][j][k], Long.MIN_VALUE);
                }
            }
        }
        return Math.max(dfs3466(0, 0, 0, 0), dfs3466(0, 1, 1, 0));

    }

    // dfs(i, j, k, l) 从索引i开始，当前在赛道j，已经切换了k次赛道，已经跑了至少1英里（l == True 表示至少跑了1英里， l == False表示还未跑）时，
    // 可获得的最大硬币数
    private long dfs3466(int i, int j, int k, int l) {
        if (i == n3466) {
            return l == 1 ? 0L : (long) -1e15;
        }
        if (memo3466[i][j][k][l] != Long.MIN_VALUE) {
            return memo3466[i][j][k][l];
        }
        long res = Long.MIN_VALUE;
        // 之前还未进入赛道，在第i位置，仍可以不进入赛道
        if (l == 0) {
            res = Math.max(res, dfs3466(i + 1, j, k, l));
        }
        // 不换赛道
        res = Math.max(res, Math.max(0, dfs3466(i + 1, j, k, 1)) + lane3466[i][j]);
        // 换赛道
        if (k < 2) {
            res = Math.max(res, Math.max(0, dfs3466(i + 1, j ^ 1, k + 1, 1)) + lane3466[i][j ^ 1]);
        }
        return memo3466[i][j][k][l] = res;
    }

    // 440. 字典序的第K小数字 (K-th Smallest in Lexicographical Order)
    public int findKthNumber(int n, int k) {
        int node = 1;
        --k;
        while (k > 0) {
            long size = cal440(node, n);
            if (size <= k) {
                ++node;
                k -= size;
            } else {
                node *= 10;
                --k;
            }
        }
        return node;
        
    }

    private long cal440(int node, int n) {
        long left = node;
        long right = node + 1;
        long size = 0L;
        while (left <= n) {
            size += Math.min(right - 1, n) - left + 1;
            left *= 10;
            right *= 10;
        }
        return size;
    }

    // 3442. 奇偶频次间的最大差值 I (Maximum Difference Between Even and Odd Frequency I)
    public int maxDifference(String s) {
        int[] cnts = new int[26];
        for (char c : s.toCharArray()) {
            ++cnts[c - 'a'];
        }
        int maxOdd = 0;
        int minEven = Integer.MAX_VALUE;
        for (int c : cnts) {
            if ((c & 1) == 1) {
                maxOdd = Math.max(maxOdd, c);
            } else if (c > 0) {
                minEven = Math.min(minEven, c);
            }
        }
        return maxOdd - minEven;

    }

    // 3572. 选择不同 X 值三元组使 Y 值之和最大 (Maximize Y‑Sum by Picking a Triplet of Distinct
    // X‑Values)
    public int maxSumDistinctTriplet(int[] x, int[] y) {
        int n = x.length;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            map.merge(x[i], y[i], Math::max);
        }
        int max1 = 0;
        int max2 = 0;
        int max3 = 0;
        for (int v : map.values()) {
            if (v >= max1) {
                max3 = max2;
                max2 = max1;
                max1 = v;
            } else if (v >= max2) {
                max3 = max2;
                max2 = v;
            } else if (v >= max3) {
                max3 = v;
            }
        }
        return max1 == 0 || max2 == 0 || max3 == 0 ? -1 : max1 + max2 + max3;
    }

    // 3573. 买卖股票的最佳时机 V (Best Time to Buy and Sell Stock V)
    private int n3573;
    private int[] prices3573;
    private int k3573;
    private long[][][] memo3573;

    public long maximumProfit(int[] prices, int k) {
        this.prices3573 = prices;
        this.k3573 = k;
        this.n3573 = prices.length;
        this.memo3573 = new long[n3573][3][k];
        return dfs3573(0, 0, 0);
    }

    private long dfs3573(int i, int j, int l) {
        if (i == n3573) {
            return j == 0 ? 0L : (long) -1e13;
        }
        if (l == k3573) {
            return 0L;
        }
        if (memo3573[i][j][l] != 0L) {
            return memo3573[i][j][l];
        }
        // 不操作
        long res = dfs3573(i + 1, j, l);
        if (j == 0) {
            // 买
            res = Math.max(res, dfs3573(i + 1, 1, l) - prices3573[i]);
            // 做空
            res = Math.max(res, dfs3573(i + 1, 2, l) + prices3573[i]);
        }
        // 已买入 未卖出
        else if (j == 1) {
            // 卖
            res = Math.max(res, dfs3573(i + 1, 0, l + 1) + prices3573[i]);
        }
        // 已做空
        else {
            // 赎回
            res = Math.max(res, dfs3573(i + 1, 0, l + 1) - prices3573[i]);
        }
        return memo3573[i][j][l] = res;
    }

    // 3576. 数组元素相等转换 (Transform Array to All Equal Elements)
    public boolean canMakeEqual(int[] nums, int k) {
        return check3576(nums, 1, k) || check3576(nums, -1, k);

    }

    private boolean check3576(int[] nums, int target, int k) {
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            if (nums[i] == target) {
                continue;
            }
            int j = i + 1;
            while (j < n && nums[j] == target) {
                ++j;
            }
            if (j == n) {
                return false;
            }
            if (k < j - i) {
                return false;
            }
            k -= j - i;
            i = j;
        }
        return true;
    }

    // 3577. 统计计算机解锁顺序排列数 (Count the Number of Computer Unlocking Permutations)
    public int countPermutations(int[] complexity) {
        int n = complexity.length;
        int res = 1;
        final int MOD = (int) (1e9 + 7);
        int min = complexity[0];
        for (int i = 1; i < n; ++i) {
            if (complexity[i] <= min) {
                return 0;
            }
            min = Math.min(min, complexity[i]);
            res = (int) (((long) res * i) % MOD);
        }
        return res;
    }

    // 3574. 最大子数组 GCD 分数 (Maximize Subarray GCD Score)
    public long maxGCDScore(int[] nums, int k) {
        long res = 0L;
        for (int i = 0; i < nums.length; ++i) {
            int lbMin = 33;
            int g = 0;
            int lbCnt = 0;
            for (int j = i; j >= 0; --j) {
                int x = nums[j];
                int lb = Integer.numberOfTrailingZeros(x);
                if (lb < lbMin) {
                    lbMin = lb;
                    lbCnt = 1;
                } else if (lb == lbMin) {
                    ++lbCnt;
                }
                g = gcd(g, x);
                int ng = g;
                if (lbCnt <= k) {
                    ng <<= 1;
                }
                res = Math.max(res, (long) ng * (i - j + 1));
            }
        }
        return res;

    }

    private int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

}
