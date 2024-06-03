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
        int n = points.length;
        int[][] arr = new int[n][2];
        for (int i = 0; i < n; ++i) {
            arr[i][0] = Math.max(Math.abs(points[i][0]), Math.abs(points[i][1]));
            arr[i][1] = s.charAt(i) - 'a';
        }
        Arrays.sort(arr, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });

        int res = 0;
        int i = 0;
        int m = 0;
        while (i < n) {
            int j = i;
            int c = arr[i][0];
            while (j < n && c == arr[j][0]) {
                if ((m >> arr[j][1] & 1) == 1) {
                    return res;
                }
                m |= 1 << arr[j][1];
                ++j;
            }
            res += j - i;
            i = j;
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



}
