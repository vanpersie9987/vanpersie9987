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
import java.util.Random;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Leetcode_7 {
    public static void main(String[] args) {
        // int[] arr = { 1, 4, 1, 3 };
        // int k = 2;

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

    // 1105. 填充书架 (Filling Bookcase Shelves)
    private int[] memo1105;
    private int n1105;
    private int[][] books1105;
    private int shelfWidth1105;

    public int minHeightShelves(int[][] books, int shelfWidth) {
        this.n1105 = books.length;
        this.books1105 = books;
        this.shelfWidth1105 = shelfWidth;
        memo1105 = new int[n1105];

        Arrays.fill(memo1105, -1);

        return dfs1105(0);

    }

    private int dfs1105(int i) {
        if (i == n1105) {
            return 0;
        }
        if (memo1105[i] != -1) {
            return memo1105[i];
        }
        int min = (int) 1e7;
        int j = i;
        int thick = 0;
        int maxHeight = 0;
        while (j < n1105 && thick + books1105[j][0] <= shelfWidth1105) {
            thick += books1105[j][0];
            maxHeight = Math.max(maxHeight, books1105[j][1]);
            min = Math.min(min, dfs1105(j + 1) + maxHeight);
            ++j;
        }
        return memo1105[i] = min;
    }

    // 1993. 树上的操作 (Operations on Tree)
    class LockingTree {
        private int n;
        private List<Integer>[] g;
        private int[] lockStatus;

        public LockingTree(int[] parent) {
            this.n = parent.length;
            this.g = new ArrayList[n];
            this.lockStatus = new int[n];
            for (int i = 0; i < n; ++i) {
                g[i] = new ArrayList<>();
            }
            for (int i = 0; i < n; ++i) {
                if (parent[i] != -1) {
                    g[parent[i]].add(i);
                }
                lockStatus[i] = -1;
            }
        }

        public boolean lock(int num, int user) {
            if (lockStatus[num] != -1) {
                return false;
            }
            lockStatus[num] = user;
            return true;
        }

        public boolean unlock(int num, int user) {
            if (lockStatus[num] != user) {
                return false;
            }
            lockStatus[num] = -1;
            return true;
        }

        public boolean upgrade(int num, int user) {
            if (lockStatus[num] != -1) {
                return false;
            }
            // num的祖先节点是否都未上锁
            if (!dfs(0, num)) {
                return false;
            }
            // num是否至少有一个上锁的子孙节点，并将所有上锁节点解锁
            if (dfs2(num)) {
                lockStatus[num] = user;
                return true;
            }
            return false;

        }

        private boolean dfs(int x, int num) {
            if (lockStatus[x] != -1) {
                return false;
            }
            if (x == num) {
                return true;
            }
            for (int y : g[x]) {
                if (dfs(y, num)) {
                    return true;
                }
            }
            return false;
        }

        private boolean dfs2(int x) {
            boolean flag = false;
            for (int y : g[x]) {
                if (dfs2(y)) {
                    flag = true;
                }
            }
            if (lockStatus[x] != -1) {
                flag = true;
                lockStatus[x] = -1;
            }
            return flag;
        }

    }

    // 638. 大礼包 (Shopping Offers)
    private List<List<Integer>> filterSpecial638;
    private Map<List<Integer>, Integer> memo638;
    private int n638;
    private List<Integer> price638;

    public int shoppingOffers(List<Integer> price, List<List<Integer>> special,
            List<Integer> needs) {
        this.n638 = price.size();
        this.price638 = price;
        this.filterSpecial638 = new ArrayList<>();
        search: for (List<Integer> sp : special) {
            int specialPrice = sp.get(n638);
            int count = 0;
            int total = 0;
            for (int i = 0; i < n638; ++i) {
                if (sp.get(i) > needs.get(i)) {
                    continue search;
                }
                total += sp.get(i) * price.get(i);
                count += sp.get(i);
            }
            if (count == 0) {
                continue;
            }
            if (total <= specialPrice) {
                continue;
            }
            filterSpecial638.add(sp);
        }
        this.memo638 = new HashMap<>();
        return dfs638(needs);

    }

    private int dfs638(List<Integer> needs) {
        if (memo638.containsKey(needs)) {
            return memo638.get(needs);
        }
        int minPrice = 0;
        for (int i = 0; i < n638; ++i) {
            minPrice += needs.get(i) * price638.get(i);
        }
        for (List<Integer> sp : filterSpecial638) {
            List<Integer> next = new ArrayList<>();
            for (int i = 0; i < n638; ++i) {
                if (sp.get(i) > needs.get(i)) {
                    break;
                }
                next.add(needs.get(i) - sp.get(i));
            }
            if (next.size() == n638) {
                minPrice = Math.min(minPrice, dfs638(next) + sp.get(n638));
            }
        }
        memo638.put(needs, minPrice);
        return minPrice;
    }

    // 188. 买卖股票的最佳时机 IV (Best Time to Buy and Sell Stock IV)
    private int[][][] memo188;
    private int n188;
    private int k188;
    private int[] prices188;

    public int maxProfit(int k, int[] prices) {
        this.n188 = prices.length;
        this.k188 = k;
        this.prices188 = prices;
        this.memo188 = new int[n188][k][2];
        for (int i = 0; i < n188; ++i) {
            for (int j = 0; j < k; ++j) {
                Arrays.fill(memo188[i][j], -1);
            }
        }
        return dfs188(0, 0, 0);

    }

    private int dfs188(int i, int count, int state) {
        if (i == n188 || count == k188) {
            return 0;
        }
        if (memo188[i][count][state] != -1) {
            return memo188[i][count][state];
        }
        int res = 0;
        // 已卖出状态 可买入
        if (state == 0) {
            // 买
            res = Math.max(res, -prices188[i] + dfs188(i + 1, count, state ^ 1));
            // 不买
            res = Math.max(res, dfs188(i + 1, count, state));
        }
        // 已买入状态 可卖出
        else {
            // 卖
            res = Math.max(res, prices188[i] + dfs188(i + 1, count + 1, state ^ 1));
            // 不卖
            res = Math.max(res, dfs188(i + 1, count, state));
        }
        return memo188[i][count][state] = res;
    }

    // 6323. 将钱分给最多的儿童 (Distribute Money to Maximum Children)
    public int distMoney(int money, int children) {
        money -= children;
        if (money < 0) {
            return -1;
        }
        int res = Math.min(money / 7, children);
        money -= res * 7;
        children -= res;

        if (money > 0 && children == 0 || children == 1 && money == 3) {
            --res;
        }
        return res;

    }

    // 6324. 最大化数组的伟大值 (Maximize Greatness of an Array)
    public int maximizeGreatness(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        int res = 0;
        int i = 0;
        int j = 0;
        while (i < n && j < n) {
            if (nums[j] > nums[i]) {
                ++res;
                ++i;
            }
            ++j;
        }
        return res;

    }

    // 6351. 标记所有元素后数组的分数 (Find Score of an Array After Marking All Elements)
    public long findScore(int[] nums) {
        int n = nums.length;
        boolean[] vis = new boolean[n];
        Integer[] ids = IntStream.range(0, n).boxed().toArray(Integer[]::new);
        Arrays.sort(ids, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                if (nums[o1] == nums[o2]) {
                    return Integer.compare(o1, o2);
                }
                return Integer.compare(nums[o1], nums[o2]);
            }

        });
        long res = 0l;
        for (int id : ids) {
            if (vis[id]) {
                continue;
            }
            vis[id] = true;
            res += nums[id];
            if (id - 1 >= 0) {
                vis[id - 1] = true;
            }
            if (id + 1 < n) {
                vis[id + 1] = true;
            }
        }
        return res;

    }

    // 6325. 修车的最少时间 (Minimum Time to Repair Cars)
    public long repairCars(int[] ranks, int cars) {
        long left = 1l;
        long right = (long) 1e15;
        long res = -1;
        while (left <= right) {
            long mid = left + ((right - left) >> 1);
            if (check6325(mid, ranks) >= cars) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;

    }

    private long check6325(long target, int[] ranks) {
        long cars = 0l;
        for (int r : ranks) {
            cars += Math.sqrt(target / r);
        }
        return cars;
    }

    // 6319. 奇偶位数 (Number of Even and Odd Bits)
    public int[] evenOddBit(int n) {
        int[] res = new int[2];
        int i = 0;
        while (n != 0) {
            res[i] += n & 1;
            i ^= 1;
            n >>= 1;
        }
        return res;
    }

    // 6319. 奇偶位数 (Number of Even and Odd Bits)
    public int[] evenOddBit2(int n) {
        final int MASK = 0x55555;
        return new int[] { Integer.bitCount(n & MASK), Integer.bitCount(n & (MASK >> 1)) };
    }

    // 6322. 检查骑士巡视方案 (Check Knight Tour Configuration)
    public boolean checkValidGrid(int[][] grid) {
        int n = grid.length;
        int num = 0;
        int[][] dirs = { { -2, 1 }, { -1, 2 }, { 1, 2 }, { 2, 1 }, { 2, -1 }, { 1, -2 }, { -1, -2 },
                { -2, -1 } };
        Queue<int[]> q = new LinkedList<>();
        q.offer(new int[] { 0, 0 });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int y = cur[1];
            for (int[] dir : dirs) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                if (nx >= 0 && nx < n && ny >= 0 && ny < n && grid[nx][ny] == num + 1) {
                    ++num;
                    q.offer(new int[] { nx, ny });
                }
            }
        }
        return num == n * n - 1;
    }

    // 6352. 美丽子集的数目 (The Number of Beautiful Subsets)
    private int n6352;
    private int[] nums6352;
    private int k6352;
    private int[] count6352;
    private int res6352;

    public int beautifulSubsets(int[] nums, int k) {
        this.n6352 = nums.length;
        this.nums6352 = nums;
        this.k6352 = k;
        this.count6352 = new int[2 * k + 1001];
        dfs6352(0);
        return res6352 - 1;

    }

    private void dfs6352(int i) {
        if (i == n6352) {
            ++res6352;
            return;
        }
        dfs6352(i + 1);
        int x = nums6352[i] + k6352;
        if (count6352[x - k6352] == 0 && count6352[x + k6352] == 0) {
            ++count6352[x];
            dfs6352(i + 1);
            --count6352[x];
        }
    }

    // 6321. 执行操作后的最大 MEX (Smallest Missing Non-negative Integer After Operations)
    public int findSmallestInteger(int[] nums, int value) {
        Map<Integer, Integer> count = new HashMap<>();
        for (int num : nums) {
            count.merge((num % value + value) % value, 1, Integer::sum);
        }
        int res = 0;
        while (count.merge(res % value, -1, Integer::sum) >= 0) {
            ++res;
        }
        return res;
    }

    // 1012. 至少有 1 位重复的数字 (Numbers With Repeated Digits) --数位dfs
    // (本题可以通过判断mask是否为0，即前面是否选过数字，从而去掉isNum)
    private int[][] memo1012;
    private char[] arr1012;
    private int k1012;

    public int numDupDigitsAtMostN(int n) {
        this.arr1012 = String.valueOf(n).toCharArray();
        this.k1012 = arr1012.length;
        this.memo1012 = new int[k1012][1 << 10];
        for (int i = 0; i < k1012; ++i) {
            Arrays.fill(memo1012[i], -1);
        }
        return n - dfs1012(0, 0, true, false);

    }

    private int dfs1012(int i, int mask, boolean isLimit, boolean isNum) {
        if (i == k1012) {
            return isNum ? 1 : 0;
        }
        if (!isLimit && isNum && memo1012[i][mask] != -1) {
            return memo1012[i][mask];
        }
        int res = 0;
        if (!isNum) {
            res = dfs1012(i + 1, mask, false, false);
        }
        int up = isLimit ? arr1012[i] - '0' : 9;
        for (int d = isNum ? 0 : 1; d <= up; ++d) {
            if ((mask & (1 << d)) == 0) {
                res += dfs1012(i + 1, mask | (1 << d), isLimit && d == up, true);
            }
        }
        if (!isLimit && isNum) {
            memo1012[i][mask] = res;
        }
        return res;
    }

    // 面试题 17.06. 2出现的次数 (Number Of 2s In Range LCCI) --数位dfs (本题可以不要isNum)
    private int[][] memo17_06;
    private char[] arr17_06;
    private int k17_06;

    public int numberOf2sInRange(int n) {
        this.arr17_06 = String.valueOf(n).toCharArray();
        this.k17_06 = arr17_06.length;
        this.memo17_06 = new int[k17_06][k17_06];
        for (int i = 0; i < k17_06; ++i) {
            Arrays.fill(memo17_06[i], -1);
        }
        return dfs17_06(0, 0, true, false);

    }

    private int dfs17_06(int i, int count, boolean isLimit, boolean isNum) {
        if (i == k17_06) {
            return isNum ? count : 0;
        }
        if (!isLimit && isNum && memo17_06[i][count] != -1) {
            return memo17_06[i][count];
        }
        int res = 0;
        if (!isNum) {
            res = dfs17_06(i + 1, count, false, false);
        }
        int up = isLimit ? arr17_06[i] - '0' : 9;
        for (int d = isNum ? 0 : 1; d <= up; ++d) {
            res += dfs17_06(i + 1, count + (d == 2 ? 1 : 0), isLimit && d == up, true);
        }
        if (!isLimit && isNum) {
            return memo17_06[i][count] = res;
        }
        return res;
    }

    // 2376. 统计特殊整数 (Count Special Integers) --数位dfs
    // (本题可以通过判断mask是否为0，即前面是否选过数字，从而去掉isNum)
    private int[][] memo2376;
    private char[] arr2376;
    private int k2376;

    public int countSpecialNumbers(int n) {
        this.arr2376 = String.valueOf(n).toCharArray();
        this.k2376 = arr2376.length;
        this.memo2376 = new int[k2376][1 << 10];
        for (int i = 0; i < k2376; ++i) {
            Arrays.fill(memo2376[i], -1);
        }
        return dfs2376(0, 0, true, false);

    }

    private int dfs2376(int i, int mask, boolean isLimit, boolean isNum) {
        if (i == k2376) {
            return isNum ? 1 : 0;
        }
        if (!isLimit && isNum && memo2376[i][mask] != -1) {
            return memo2376[i][mask];
        }
        int res = 0;
        if (!isNum) {
            res = dfs2376(i + 1, mask, false, false);
        }
        int up = isLimit ? arr2376[i] - '0' : 9;
        for (int d = isNum ? 0 : 1; d <= up; ++d) {
            if ((mask & (1 << d)) == 0) {
                res += dfs2376(i + 1, mask | (1 << d), isLimit && d == up, true);
            }
        }
        if (!isLimit && isNum) {
            memo2376[i][mask] = res;
        }
        return res;
    }

    // 309. 最佳买卖股票时机含冷冻期 (Best Time to Buy and Sell Stock with Cooldown)
    private int[][] memo309;
    private int n309;
    private int[] prices309;

    public int maxProfit309(int[] prices) {
        this.n309 = prices.length;
        this.memo309 = new int[n309][3];
        this.prices309 = prices;
        for (int i = 0; i < n309; ++i) {
            Arrays.fill(memo309[i], -1);
        }
        return dfs309(0, 0);

    }

    private int dfs309(int i, int state) {
        if (i == n309) {
            return 0;
        }
        if (memo309[i][state] != -1) {
            return memo309[i][state];
        }
        int max = 0;
        // 可买入
        if (state == 0) {
            // 买
            max = Math.max(max, -prices309[i] + dfs309(i + 1, (state + 1) % 3));
            // 不买
            max = Math.max(max, dfs309(i + 1, state));
        }
        // 可卖出
        else if (state == 1) {
            // 卖
            max = Math.max(max, prices309[i] + dfs309(i + 1, (state + 1) % 3));
            // 不卖
            max = Math.max(max, dfs309(i + 1, state));
        }
        // 冷冻期
        else {
            max = Math.max(max, dfs309(i + 1, (state + 1) % 3));
        }
        return memo309[i][state] = max;
    }

    // 902. 最大为 N 的数字组合 (Numbers At Most N Given Digit Set) --数位dfs
    private int[] memo902;
    private int k902;
    private char[] arr902;
    private String[] digits902;

    public int atMostNGivenDigitSet(String[] digits, int n) {
        this.arr902 = String.valueOf(n).toCharArray();
        this.k902 = arr902.length;
        this.memo902 = new int[k902];
        Arrays.fill(memo902, -1);
        this.digits902 = digits;
        return dfs902(0, true, false);
    }

    private int dfs902(int i, boolean isLimit, boolean isNum) {
        if (i == k902) {
            return isNum ? 1 : 0;
        }
        if (!isLimit && isNum && memo902[i] != -1) {
            return memo902[i];
        }
        int res = 0;
        if (!isNum) {
            res = dfs902(i + 1, false, false);
        }
        char up = isLimit ? arr902[i] : '9';
        for (String dight : digits902) {
            char d = dight.charAt(0);
            if (d > up) {
                break;
            }
            res += dfs902(i + 1, isLimit && d == up, true);
        }
        if (!isLimit && isNum) {
            memo902[i] = res;
        }
        return res;
    }

    // 357. 计算各个位数不同的数字个数 (Count Numbers with Unique Digits)
    public int countNumbersWithUniqueDigits(int n) {
        if (n == 0) {
            return 1;
        }
        if (n == 1) {
            return 10;
        }
        int res = 10;
        int cur = 9;
        for (int i = 0; i < n - 1; ++i) {
            cur *= 9 - i;
            res += cur;
        }
        return res;
    }

    // 357. 统计各位数字都不同的数字个数 (Count Numbers with Unique Digits) --数位dfs
    private int[][] memo357;
    private char[] arr357;
    private int k357;

    public int countNumbersWithUniqueDigits2(int n) {
        this.arr357 = String.valueOf((int) Math.pow(10, n) - 1).toCharArray();
        this.k357 = arr357.length;
        this.memo357 = new int[k357][1 << 10];
        for (int i = 0; i < k357; ++i) {
            Arrays.fill(memo357[i], -1);
        }
        return dfs357(0, 0, true, false) + 1;
    }

    private int dfs357(int i, int mask, boolean isLimit, boolean isNum) {
        if (i == k357) {
            return isNum ? 1 : 0;
        }
        if (!isLimit && isNum && memo357[i][mask] != -1) {
            return memo357[i][mask];
        }
        int res = 0;
        if (!isNum) {
            res = dfs357(i + 1, mask, false, false);
        }
        int up = isLimit ? arr357[i] - '0' : 9;
        for (int d = isNum ? 0 : 1; d <= up; ++d) {
            if ((mask & (1 << d)) == 0) {
                res += dfs357(i + 1, mask | (1 << d), isLimit && d == up, true);
            }
        }
        if (!isLimit && isNum) {
            memo357[i][mask] = res;
        }
        return res;
    }

    // 1626. 无矛盾的最佳球队 (Best Team With No Conflicts)
    private int[] memo1626;
    private int[][] arr1626;
    private int n1626;

    public int bestTeamScore(int[] scores, int[] ages) {
        this.n1626 = scores.length;
        this.memo1626 = new int[n1626];
        this.arr1626 = new int[n1626][2];
        for (int i = 0; i < n1626; ++i) {
            arr1626[i][0] = scores[i];
            arr1626[i][1] = ages[i];
        }
        Arrays.sort(arr1626, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] == o2[0]) {
                    return Integer.compare(o1[1], o2[1]);
                }
                return Integer.compare(o1[0], o2[0]);
            }

        });
        int res = 0;
        for (int i = 0; i < n1626; ++i) {
            res = Math.max(res, dfs1626(i));
        }
        return res;

    }

    private int dfs1626(int i) {
        if (memo1626[i] != 0) {
            return memo1626[i];
        }
        int max = 0;
        for (int j = 0; j < i; ++j) {
            if (arr1626[j][1] <= arr1626[i][1]) {
                max = Math.max(max, dfs1626(j));
            }
        }
        return memo1626[i] = max + arr1626[i][0];
    }

    // 1335. 工作计划的最低难度 (Minimum Difficulty of a Job Schedule)
    private int[] jobDifficulty1335;
    private int n1335;
    private int d1335;
    private int[][] memo1335;

    public int minDifficulty(int[] jobDifficulty, int d) {
        this.jobDifficulty1335 = jobDifficulty;
        this.d1335 = d;
        this.n1335 = jobDifficulty.length;
        if (n1335 < d) {
            return -1;
        }
        this.memo1335 = new int[d][n1335];
        for (int i = 0; i < d; ++i) {
            Arrays.fill(memo1335[i], -1);
        }
        return dfs1335(0, 0);

    }

    private int dfs1335(int days, int jobIndex) {
        if (days == d1335 || jobIndex == n1335) {
            if (days == d1335 && jobIndex == n1335) {
                return 0;
            }
            return (int) 1e7;
        }
        if (n1335 - jobIndex < d1335 - days) {
            return (int) 1e7;
        }
        if (memo1335[days][jobIndex] != -1) {
            return memo1335[days][jobIndex];
        }
        int max = 0;
        int res = (int) 1e7;
        int j = jobIndex;
        while (j < n1335) {
            max = Math.max(max, jobDifficulty1335[j]);
            res = Math.min(res, max + dfs1335(days + 1, j + 1));
            ++j;
        }
        return memo1335[days][jobIndex] = res;
    }

    // 2251. 花期内花的数目 (Number of Flowers in Full Bloom)
    public int[] fullBloomFlowers(int[][] flowers, int[] people) {
        int n = people.length;
        TreeMap<Integer, Integer> diff = new TreeMap<>();
        diff.put(0, 0);
        for (int[] flower : flowers) {
            int start = flower[0];
            int end = flower[1];
            diff.merge(start, 1, Integer::sum);
            diff.merge(end + 1, -1, Integer::sum);
        }
        int pre = 0;
        for (Map.Entry<Integer, Integer> entry : diff.entrySet()) {
            int key = entry.getKey();
            int val = entry.getValue();
            pre += val;
            diff.put(key, pre);
        }
        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            Integer val = diff.floorKey(people[i]);
            res[i] = diff.getOrDefault(val, 0);
        }
        return res;

    }

    // 1690. 石子游戏 VII (Stone Game VII)
    private int[] stones1690;
    private int n1690;
    private int[][] memo1690;

    public int stoneGameVII(int[] stones) {
        this.n1690 = stones.length;
        this.stones1690 = stones;
        this.memo1690 = new int[n1690][n1690];
        for (int i = 0; i < n1690; ++i) {
            Arrays.fill(memo1690[i], -1);
        }
        return dfs1690(0, n1690 - 1);

    }

    private int dfs1690(int left, int right) {
        if (left == right) {
            return 0;
        }
        if (left == right - 1) {
            return Math.max(stones1690[left], stones1690[right]);
        }
        if (memo1690[left][right] != -1) {
            return memo1690[left][right];
        }
        return memo1690[left][right] = Math.max(
                Math.min(stones1690[left + 1] + dfs1690(left + 2, right),
                        stones1690[right] + dfs1690(left + 1, right - 1)),
                Math.min(stones1690[left] + dfs1690(left + 1, right - 1),
                        stones1690[right - 1] + dfs1690(left, right - 2)));
    }

    // 924. 尽量减少恶意软件的传播 (Minimize Malware Spread)
    public int minMalwareSpread(int[][] graph, int[] initial) {
        int n = graph.length;
        Union924 union = new Union924(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (graph[i][j] == 1) {
                    union.union(i, j);
                }
            }
        }
        Map<Integer, Integer> countMap = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            int root = union.getRoot(i);
            countMap.merge(root, 1, Integer::sum);
        }
        int infectionsNodes = 0;
        Map<Integer, Integer> infection = new HashMap<>();
        for (int i : initial) {
            int root = union.getRoot(i);
            if (!infection.containsKey(root)) {
                infectionsNodes += countMap.getOrDefault(root, 0);
            }
            infection.merge(root, 1, Integer::sum);
        }
        Arrays.sort(initial);
        int res = initial[0];
        int min = infectionsNodes;
        for (int i : initial) {
            int root = union.getRoot(i);
            if (infection.getOrDefault(root, 0) == 1) {
                int actual = infectionsNodes - countMap.get(root);
                if (actual < min) {
                    min = actual;
                    res = i;
                }
            }
        }
        return res;

    }

    public class Union924 {
        private int[] parent;
        private int[] rank;

        public Union924(int n) {
            parent = new int[n];
            rank = new int[n];
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
                if (rank[root1] == rank[root2]) {
                    ++rank[root1];
                }
            }
        }

    }

    // 2218. 从栈中取出 K 个硬币的最大面值和 (Maximum Value of K Coins From Piles)
    private List<List<Integer>> piles2218;
    private int n2218;
    private int[][] memo2218;

    public int maxValueOfCoins(List<List<Integer>> piles, int k) {
        this.piles2218 = piles;
        this.n2218 = piles.size();
        this.memo2218 = new int[n2218][k + 1];
        return dfs2218(0, k);
    }

    private int dfs2218(int i, int count) {
        if (i == n2218 || count == 0) {
            return 0;
        }
        if (memo2218[i][count] != 0) {
            return memo2218[i][count];
        }
        int pre = 0;
        int res = 0;
        for (int j = 0; j <= Math.min(count, piles2218.get(i).size()); ++j) {
            res = Math.max(res, dfs2218(i + 1, count - j) + pre);
            if (j < piles2218.get(i).size()) {
                pre += piles2218.get(i).get(j);
            }
        }
        return memo2218[i][count] = res;
    }

    // 1449. 数位成本和为目标值的最大数字 (Form Largest Integer With Digits That Add up to Target)
    private String[] memo1449;
    private int[] cost1449;

    public String largestNumber(int[] cost, int target) {
        this.memo1449 = new String[target + 1];
        this.cost1449 = cost;
        return dfs1449(target);
    }

    private String dfs1449(int left) {
        if (left == 0) {
            return "";
        }
        if (memo1449[left] != null) {
            return memo1449[left];
        }
        String res = "";
        for (int j = 0; j < cost1449.length; ++j) {
            if (left - cost1449[j] >= 0) {
                String cur = String.valueOf(j + 1) + dfs1449(left - cost1449[j]);
                if (!cur.contains("0")) {
                    if (cur.length() > res.length() || cur.length() == res.length() && cur.compareTo(res) > 0) {
                        res = cur;
                    }
                }
            }
        }
        return memo1449[left] = res.isEmpty() ? "0" : res;
    }

    // 6354. K 件物品的最大和 (K Items With the Maximum Sum)
    public int kItemsWithMaximumSum(int numOnes, int numZeros, int numNegOnes, int k) {
        if (k <= numOnes + numZeros) {
            return Math.min(numOnes, k);
        }
        return numOnes - (k - numOnes - numZeros);

    }

    // 6355. 质数减法运算 (Prime Subtraction Operation)
    public boolean primeSubOperation(int[] nums) {
        int n = nums.length;
        List<Integer> list = getPrimes6355();
        int i = n - 2;
        while (i >= 0) {
            if (nums[i] < nums[i + 1]) {
                --i;
                continue;
            }
            int prime = binarySearch6355(list, nums[i] - nums[i + 1]);
            if (prime >= nums[i]) {
                return false;
            }
            nums[i] -= prime;
            --i;
        }
        return true;

    }

    // 找排序数组list中，严格大于target的最小值
    private int binarySearch6355(List<Integer> list, int target) {
        int left = 0;
        int right = list.size() - 1;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (list.get(mid) > target) {
                res = list.get(mid);
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    // 埃式筛质数
    private List<Integer> getPrimes6355() {
        List<Integer> list = new ArrayList<>();
        int[] isPrime = new int[1010];
        Arrays.fill(isPrime, 1);
        for (int i = 2; i < 1010; ++i) {
            if (isPrime[i] == 1) {
                list.add(i);
                if ((long) i * i < 1010) {
                    for (int j = i * i; j < 1010; j += i) {
                        isPrime[j] = 0;
                    }
                }
            }
        }
        return list;
    }

    // 6357. 使数组元素全部相等的最少操作次数 (Minimum Operations to Make All Array Elements Equal)
    public List<Long> minOperations(int[] nums, int[] queries) {
        Arrays.sort(nums);
        int n = nums.length;
        long[] preSum = new long[nums.length + 1];
        for (int i = 1; i <= nums.length; ++i) {
            preSum[i] = preSum[i - 1] + nums[i - 1];
        }
        List<Long> res = new ArrayList<>();
        for (int q : queries) {
            int less = binarySearch6357(nums, q);
            int more = n - less;
            long cur = (long) less * q - preSum[less] + (preSum[n] - preSum[less]) - (long) more * q;
            res.add(cur);
        }
        return res;

    }

    // 找排序数组nums中，严格小于target的元素个数
    private int binarySearch6357(int[] nums, int target) {
        int n = nums.length;
        if (target <= nums[0]) {
            return 0;
        }
        if (target > nums[n - 1]) {
            return n;
        }
        int left = 0;
        int right = n - 1;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (nums[mid] < target) {
                res = mid + 1;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    // 6356. 收集树中金币 (Collect Coins in a Tree)
    public int collectTheCoins(int[] coins, int[][] edges) {
        int n = coins.length;
        int leftEdges = n - 1;
        Map<Integer, List<Integer>> g = new HashMap<>();
        int[] deg = new int[n];
        for (int[] e : edges) {
            int a = e[0];
            int b = e[1];
            g.computeIfAbsent(a, k -> new ArrayList<>()).add(b);
            g.computeIfAbsent(b, k -> new ArrayList<>()).add(a);
            ++deg[a];
            ++deg[b];
        }
        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < n; ++i) {
            // 叶子结点 无金币
            if (deg[i] == 1 && coins[i] == 0) {
                q.offer(i);
            }
        }
        // 去除没有金币的叶子结点
        while (!q.isEmpty()) {
            int x = q.poll();
            --leftEdges;
            --deg[x];
            for (int y : g.getOrDefault(x, new ArrayList<>())) {
                // 没有金币的叶子结点
                if (--deg[y] == 1 && coins[y] == 0) {
                    q.offer(y);
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            // 叶子结点 有金币
            if (deg[i] == 1 && coins[i] == 1) {
                q.offer(i);
            }
        }
        int count = 0;
        while (!q.isEmpty() && count++ < 2) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                int x = q.poll();
                --leftEdges;
                --deg[x];
                for (int y : g.getOrDefault(x, new ArrayList<>())) {
                    if (--deg[y] == 1) {
                        q.offer(y);
                    }
                }
            }
        }
        return Math.max(0, leftEdges * 2);

    }

    // 600. 不含连续1的非负整数 (Non-negative Integers without Consecutive Ones)
    // 可以不要isNum参数
    private int[][] memo600;
    private char[] arr600;
    private int k600;

    public int findIntegers(int n) {
        this.arr600 = Integer.toBinaryString(n).toCharArray();
        this.k600 = arr600.length;
        this.memo600 = new int[k600][2];
        for (int i = 0; i < k600; ++i) {
            Arrays.fill(memo600[i], -1);
        }
        return dfs600(0, 0, true, false) + 1;
    }

    private int dfs600(int i, int pre, boolean isLimit, boolean isNum) {
        if (i == k600) {
            if (isNum) {
                return 1;
            }
            return 0;
        }
        if (!isLimit && isNum && memo600[i][pre] != -1) {
            return memo600[i][pre];
        }
        int res = 0;
        if (!isNum) {
            res = dfs600(i + 1, pre, false, false);
        }
        int up = isLimit ? arr600[i] - '0' : 1;
        for (int j = isNum ? 0 : 1; j <= up; ++j) {
            if (j + pre <= 1) {
                res += dfs600(i + 1, j, isLimit && j == up, true);
            }
        }
        if (!isLimit && isNum) {
            memo600[i][pre] = res;
        }
        return res;
    }

    // 1363. 形成三的最大倍数 (Largest Multiple of Three)
    public String largestMultipleOfThree(int[] digits) {
        int[] counts = new int[10];
        int sum = 0;
        for (int d : digits) {
            ++counts[d];
            sum += d;
        }
        if (sum % 3 == 1) {
            // 减去一个mod 3 == 1的数 或者 减去两个mod 3 == 2的数 (根据反证法 以上两种情况一定存在一种)
            if (!check1363(counts, 1, 1)) {
                check1363(counts, 2, 2);
            }
        } else if (sum % 3 == 2) {
            // 减去一个mod 3 == 2的数 或者 减去两个mod 3 == 1的数 (根据反证法 以上两种情况一定存在一种)
            if (!check1363(counts, 2, 1)) {
                check1363(counts, 1, 2);
            }
        }
        StringBuilder res = new StringBuilder();
        for (int i = 9; i >= 0; --i) {
            while (counts[i]-- > 0) {
                res.append(i);
            }
        }
        if (res.isEmpty()) {
            return "";
        }
        if (res.charAt(0) == '0') {
            return "0";
        }
        return res.toString();

    }

    private boolean check1363(int[] counts, int mod, int times) {
        for (int i = mod; i <= 9; i += 3) {
            while (counts[i] > 0) {
                --counts[i];
                if (--times == 0) {
                    return true;
                }
            }
        }
        return false;
    }

    // 1092. 最短公共超序列 (Shortest Common Supersequence)
    private int[][] memo1092;
    private String str11092;
    private String str21092;

    public String shortestCommonSupersequence(String str1, String str2) {
        int m = str1.length();
        int n = str2.length();
        this.str11092 = str1;
        this.str21092 = str2;
        this.memo1092 = new int[m][n];
        return makeAns(m - 1, n - 1);
    }

    private String makeAns(int i, int j) {
        if (i < 0) {
            return str21092.substring(0, j + 1);
        }
        if (j < 0) {
            return str11092.substring(0, i + 1);
        }
        if (str11092.charAt(i) == str21092.charAt(j)) {
            return makeAns(i - 1, j - 1) + str11092.charAt(i);
        }
        if (dfs1092(i, j) == dfs1092(i - 1, j) + 1) {
            return makeAns(i - 1, j) + str11092.charAt(i);
        }
        return makeAns(i, j - 1) + str21092.charAt(j);
    }

    private int dfs1092(int i, int j) {
        if (i < 0) {
            return j + 1;
        }
        if (j < 0) {
            return i + 1;
        }
        if (memo1092[i][j] != 0) {
            return memo1092[i][j];
        }
        if (str11092.charAt(i) == str21092.charAt(j)) {
            return memo1092[i][j] = dfs1092(i - 1, j - 1) + 1;
        }
        return memo1092[i][j] = Math.min(dfs1092(i - 1, j), dfs1092(i, j - 1)) + 1;
    }

    // 1478.安排邮筒 (Allocate Mailboxes)
    private int[] houses1478;
    private int k1478;
    private int n1478;
    private int[][] memo1478;

    public int minDistance(int[] houses, int k) {
        Arrays.sort(houses);
        this.n1478 = houses.length;
        this.k1478 = k;
        this.houses1478 = houses;
        this.memo1478 = new int[n1478][k];
        for (int i = 0; i < n1478; ++i) {
            Arrays.fill(memo1478[i], Integer.MAX_VALUE);
        }
        return dfs1478(0, 0);

    }

    private int dfs1478(int i, int count) {
        if (i == n1478 || count == k1478) {
            if (i == n1478 && count == k1478) {
                return 0;
            }
            return (int) 1e9;
        }
        if (memo1478[i][count] != Integer.MAX_VALUE) {
            return memo1478[i][count];
        }
        int res = (int) 1e9;
        for (int j = i; j < n1478; ++j) {
            res = Math.min(res, getDistance1478(i, j) + dfs1478(j + 1, count + 1));
        }
        return memo1478[i][count] = res;
    }

    private int getDistance1478(int i, int j) {
        int d = 0;
        while (i < j) {
            d += houses1478[j] - houses1478[i];
            ++i;
            --j;
        }
        return d;
    }

    // 2209. 用地毯覆盖后的最少白色砖块 (Minimum White Tiles After Covering With Carpets)
    private int[] pre2209;
    private int numCarpets2209;
    private int carpetLen2209;
    private int[][] memo2209;
    private int n2209;

    public int minimumWhiteTiles(String floor, int numCarpets, int carpetLen) {
        this.n2209 = floor.length();
        this.pre2209 = new int[n2209 + 1];
        for (int i = 1; i < n2209 + 1; ++i) {
            pre2209[i] = pre2209[i - 1] + floor.charAt(i - 1) - '0';
        }
        if (pre2209[n2209] == 0) {
            return 0;
        }
        this.numCarpets2209 = numCarpets;
        this.carpetLen2209 = carpetLen;
        this.memo2209 = new int[n2209][numCarpets];
        for (int i = 0; i < n2209; ++i) {
            Arrays.fill(memo2209[i], -1);
        }
        return pre2209[n2209] - dfs2209(0, 0);

    }

    // 从第i地砖开始、已使用count个地毯时，最多能覆盖多少白色的地砖
    private int dfs2209(int i, int count) {
        if (i >= n2209) {
            return 0;
        }
        // 地毯用完了
        if (count == numCarpets2209) {
            return 0;
        }
        // 从i开始的floor没有白色地砖了
        if (pre2209[n2209] - pre2209[i] == 0) {
            return 0;
        }
        // 剩余的地毯个数 * 每个地毯的长度 >= 剩余的地砖个数
        if ((numCarpets2209 - count) * carpetLen2209 >= n2209 - i) {
            return pre2209[n2209] - pre2209[i];
        }
        if (memo2209[i][count] != -1) {
            return memo2209[i][count];
        }
        // 不覆盖i地砖 or 从覆盖i地砖
        return memo2209[i][count] = Math.max(dfs2209(i + 1, count),
                pre2209[Math.min(n2209, i + carpetLen2209)] - pre2209[i] + dfs2209(i + carpetLen2209, count + 1));
    }

    // 975. 奇偶跳 (Odd Even Jump)
    private int n975;
    private int[][] memo975;
    private int[][] pos975;

    public int oddEvenJumps(int[] arr) {
        this.n975 = arr.length;
        this.memo975 = new int[n975][2];
        this.pos975 = new int[n975][2];
        for (int i = 0; i < n975; ++i) {
            Arrays.fill(memo975[i], -1);
            Arrays.fill(pos975[i], -1);
        }
        TreeMap<Integer, Integer> map = new TreeMap<>();
        map.put(arr[n975 - 1], n975 - 1);
        for (int i = n975 - 1; i >= 0; --i) {
            Integer ceiling = map.ceilingKey(arr[i]);
            if (ceiling != null) {
                pos975[i][1] = map.get(ceiling);
            }
            Integer floor = map.floorKey(arr[i]);
            if (floor != null) {
                pos975[i][0] = map.get(floor);
            }
            map.put(arr[i], i);
        }
        int res = 0;
        for (int i = 0; i < n975; ++i) {
            res += dfs975(i, 1);
        }
        return res;
    }

    private int dfs975(int i, int oddOrEven) {
        if (i == n975 - 1) {
            return 1;
        }
        if (pos975[i][oddOrEven & 1] == -1) {
            return 0;
        }
        if (memo975[i][oddOrEven] != -1) {
            return memo975[i][oddOrEven];
        }
        return memo975[i][oddOrEven] = dfs975(pos975[i][oddOrEven & 1], oddOrEven ^ 1);
    }

    // 871. 最低加油次数 (Minimum Number of Refueling Stops)
    public int minRefuelStops(int target, int startFuel, int[][] stations) {
        int n = stations.length;
        int curPos = 0;
        long curFuel = startFuel;
        Queue<Integer> q = new PriorityQueue<>(new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(o2, o1);
            }

        });
        int res = 0;
        for (int i = 0; i <= n; ++i) {
            int pos = i < n ? stations[i][0] : target;
            while (!q.isEmpty() && pos - curPos > curFuel) {
                ++res;
                curFuel += q.poll();
            }
            if (pos - curPos > curFuel) {
                return -1;
            }
            curFuel -= pos - curPos;
            curPos = pos;
            if (i < n) {
                q.offer(stations[i][1]);
            }
        }
        return res;

    }

    // 1547. 切棍子的最小成本 (Minimum Cost to Cut a Stick)
    private int[][] memo1547;
    private int k1547;
    private int[] cuts1547;

    public int minCost(int n, int[] cuts) {
        Arrays.sort(cuts);
        this.cuts1547 = cuts;
        this.k1547 = cuts.length;
        this.memo1547 = new int[k1547][k1547];
        for (int i = 0; i < k1547; ++i) {
            Arrays.fill(memo1547[i], -1);
        }
        return dfs1547(0, n, 0, k1547 - 1);

    }

    private int dfs1547(int left, int right, int i, int j) {
        if (i > j) {
            return 0;
        }
        if (i == j) {
            return right - left;
        }
        if (memo1547[i][j] != -1) {
            return memo1547[i][j];
        }
        int min = (int) 1e9;
        for (int cutPos = i; cutPos <= j; ++cutPos) {
            min = Math.min(min,
                    dfs1547(left, cuts1547[cutPos], i, cutPos - 1) + dfs1547(cuts1547[cutPos], right, cutPos + 1, j));
        }
        return memo1547[i][j] = min + right - left;
    }

    // 2576. 求出最多标记下标 (Find the Maximum Number of Marked Indices) --二分
    public int maxNumOfMarkedIndices(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        int res = 0;
        int left = 0;
        int right = n / 2;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (check2576(nums, mid)) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res * 2;
    }

    private boolean check2576(int[] nums, int k) {
        for (int i = 0; i < k; ++i) {
            if (nums[i] * 2 > nums[nums.length - k + i]) {
                return false;
            }
        }
        return true;
    }

    // 2576. 求出最多标记下标 (Find the Maximum Number of Marked Indices) --双指针
    public int maxNumOfMarkedIndices2(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        int i = 0;
        for (int j = (n + 1) / 2; j < n; ++j) {
            if (nums[i] * 2 <= nums[j]) {
                ++i;
            }
        }
        return i * 2;

    }

    // 2197. 替换数组中的非互质数 (Replace Non-Coprime Numbers in Array)
    public List<Integer> replaceNonCoprimes(int[] nums) {
        List<Integer> res = new ArrayList<>();
        for (int num : nums) {
            res.add(num);
            while (res.size() > 1) {
                int pop1 = res.get(res.size() - 1);
                int pop2 = res.get(res.size() - 2);
                int g = getGCD2197(pop1, pop2);
                if (g == 1) {
                    break;
                }
                res.remove(res.size() - 1);
                res.set(res.size() - 1, pop1 / g * pop2);
            }
        }

        return res;

    }

    private int getGCD2197(int a, int b) {
        return b == 0 ? a : getGCD2197(b, a % b);
    }

    // 212. 单词搜索 II (Word Search II)
    private Set<String> set212;
    private char[][] board212;
    private int m212;
    private int n212;
    private int[][] directions212 = { { 0, -1 }, { 0, 1 }, { 1, 0 }, { -1, 0 } };

    public List<String> findWords(char[][] board, String[] words) {
        Trie212 trie = new Trie212();
        for (String s : words) {
            trie.insert(s);
        }
        this.board212 = board;
        this.set212 = new HashSet<>();
        this.m212 = board.length;
        this.n212 = board[0].length;
        for (int i = 0; i < m212; ++i) {
            for (int j = 0; j < n212; ++j) {
                dfs212(i, j, trie);
            }
        }
        return new ArrayList<>(set212);

    }

    private void dfs212(int x, int y, Trie212 node) {
        if (node.children[board212[x][y] - 'a'] == null) {
            return;
        }
        node = node.children[board212[x][y] - 'a'];
        if (!node.word.isEmpty()) {
            set212.add(node.word);
        }
        char temp = board212[x][y];
        board212[x][y] = '*';
        for (int[] d : directions212) {
            int nx = x + d[0];
            int ny = y + d[1];
            if (nx >= 0 && nx < m212 && ny >= 0 && ny < n212 && board212[nx][ny] != '*') {
                dfs212(nx, ny, node);
            }
        }
        board212[x][y] = temp;

    }

    public class Trie212 {
        private Trie212[] children;
        private String word;

        public Trie212() {
            children = new Trie212[26];
            word = "";
        }

        public void insert(String s) {
            Trie212 node = this;
            for (char c : s.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new Trie212();
                }
                node = node.children[index];
            }
            node.word = s;
        }

    }

    // 1039. 多边形三角剖分的最低得分 (Minimum Score Triangulation of Polygon)
    private int[][] memo1039;
    private int[] values1039;

    public int minScoreTriangulation(int[] values) {
        int n = values.length;
        this.values1039 = values;
        this.memo1039 = new int[n][n];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(memo1039[i], -1);
        }
        return dfs1039(0, n - 1);
    }

    private int dfs1039(int i, int j) {
        if (memo1039[i][j] != -1) {
            return memo1039[i][j];
        }
        int res = Integer.MAX_VALUE;
        for (int k = i + 1; k < j; ++k) {
            res = Math.min(res, values1039[i] * values1039[j] * values1039[k] + dfs1039(i, k) + dfs1039(k, j));
        }
        return memo1039[i][j] = res == Integer.MAX_VALUE ? 0 : res;
    }

    // 2605. 从两个数字数组里生成最小数字 (Form Smallest Number From Two Digit Arrays)
    public int minNumber(int[] nums1, int[] nums2) {
        int res = Integer.MAX_VALUE;
        for (int i = 0; i < nums1.length; ++i) {
            for (int j = 0; j < nums2.length; ++j) {
                if (nums1[i] == nums2[j]) {
                    res = Math.min(res, nums1[i]);
                } else {
                    res = Math.min(res, nums1[i] * 10 + nums2[j]);
                    res = Math.min(res, nums2[j] * 10 + nums1[i]);
                }
            }
        }
        return res;

    }

    // 2606. 找到最大开销的子字符串 (Find the Substring With Maximum Cost)
    public int maximumCostSubstring(String s, String chars, int[] vals) {
        int[] fees = new int[26];
        for (int i = 0; i < 26; ++i) {
            fees[i] = i + 1;
        }
        for (int i = 0; i < chars.length(); ++i) {
            int index = chars.charAt(i) - 'a';
            fees[index] = vals[i];
        }
        int min = 0;
        int pre = 0;
        int res = 0;
        for (char c : s.toCharArray()) {
            pre += fees[c - 'a'];
            res = Math.max(res, pre - min);
            min = Math.min(pre, min);
        }
        return res;

    }

    // 2609. 最长平衡子字符串 (Find the Longest Balanced Substring of a Binary String)
    public int findTheLongestBalancedSubstring(String s) {
        int res = 0;
        int cnt0 = 0;
        int cnt1 = 0;
        int pre = 0;
        for (char c : s.toCharArray()) {
            if (c == '0') {
                if (pre == 0) {
                    ++cnt0;
                } else {
                    cnt0 = 1;
                    cnt1 = 0;
                }
            } else {
                ++cnt1;
            }
            res = Math.max(res, Math.min(cnt0, cnt1) * 2);
            pre = c - '0';
        }

        return res;

    }

    // 2610. 转换二维数组 (Convert an Array Into a 2D Array With Conditions)
    public List<List<Integer>> findMatrix(int[] nums) {
        int n = nums.length;
        int[] counts = new int[n + 1];
        int max = 0;
        for (int num : nums) {
            max = Math.max(max, ++counts[num]);
        }
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < max; ++i) {
            res.add(new ArrayList<>());
        }
        for (int i = 0; i < n + 1; ++i) {
            while (counts[i]-- > 0) {
                res.get(counts[i]).add(i);
            }
        }
        return res;

    }

    // 1912. 设计电影租借系统 (Design Movie Rental System)
    class MovieRentingSystem {
        // 借出的电影
        // key : price
        private TreeMap<Integer, TreeMap<Integer, List<Integer>>> loan;
        // 未借出的电影
        // key : movie
        private TreeMap<Integer, List<Bean>> mapMovie;
        // key : shop
        private Map<Integer, Map<Integer, Integer>> mapShop;

        // 统计 key : 来自shop的movie的price
        private Map<Bean, Integer> mapData;

        class Bean implements Comparable<Bean> {
            int shop;
            int movie;
            int price;

            Bean(int shop, int movie, int price) {
                this.shop = shop;
                this.movie = movie;
                this.price = price;
            }

            @Override
            public int compareTo(Bean o) {
                if (this.price == o.price) {
                    return Integer.compare(this.shop, o.shop);
                }
                return Integer.compare(this.price, o.price);
            }

            @Override
            public boolean equals(Object obj) {
                Bean o = (Bean) obj;
                return this.shop == o.shop && this.movie == o.movie;
            }

            @Override
            public int hashCode() {
                return (int) (((long) shop * 10000 + movie) % (1e9 + 7));
            }

        }

        public MovieRentingSystem(int n, int[][] entries) {
            loan = new TreeMap<>();
            mapMovie = new TreeMap<>();
            mapShop = new HashMap<>();
            mapData = new HashMap<>();
            for (int[] e : entries) {
                int shop = e[0];
                int movie = e[1];
                int price = e[2];
                mapMovie.computeIfAbsent(movie, k -> new List<>()).add(new Bean(shop, movie, price));
                mapShop.computeIfAbsent(shop, k -> new HashMap<>()).put(movie, price);
                mapData.put(new Bean(shop, movie, price), price);
            }

        }

        public List<Integer> search(int movie) {
            List<Integer> res = new ArrayList<>();
            List<Bean> set = mapMovie.getOrDefault(movie, new List<>());
            for (Bean b : set) {
                res.add(b.shop);
                if (res.size() == 5) {
                    break;
                }
            }
            return res;
        }

        public void rent(int shop, int movie) {
            // 从shop借走movie
            Map<Integer, Integer> map = mapShop.getOrDefault(shop, new HashMap<>());
            int price = map.get(movie);
            map.remove(movie);

            // 借走 movie
            List<Bean> set = mapMovie.getOrDefault(movie, new List<>());
            Bean removed = new Bean(shop, movie, price);
            set.remove(removed);

            // 借出的电影
            loan.computeIfAbsent(price, k -> new TreeMap<>()).computeIfAbsent(shop, k -> new List<>()).add(movie);
        }

        public void drop(int shop, int movie) {
            // 从已借出的还走
            Bean b = new Bean(shop, movie, 0);
            int price = mapData.get(b);
            TreeMap<Integer, List<Integer>> map = loan.getOrDefault(price, new TreeMap<>());
            List<Integer> movies = map.getOrDefault(shop, new List<>());
            movies.remove(movie);

            // 把已借出的还到商店
            Map<Integer, Integer> s = mapShop.getOrDefault(shop, new HashMap<>());
            s.put(movie, price);

            List<Bean> m = mapMovie.getOrDefault(movie, new List<>());
            m.add(new Bean(shop, movie, price));

        }

        public List<List<Integer>> report() {
            List<List<Integer>> res = new ArrayList<>();
            for (TreeMap<Integer, List<Integer>> shops : loan.values()) {
                for (Map.Entry<Integer, List<Integer>> movies : shops.entrySet()) {
                    int shop = movies.getKey();
                    for (int movie : movies.getValue()) {
                        res.add(List.of(shop, movie));
                        if (res.size() == 5) {
                            return res;
                        }
                    }
                }
            }
            return res;
        }
    }

    // 1383. 最大的团队表现值 (Maximum Performance of a Team)
    public int maxPerformance(int n, int[] speed, int[] efficiency, int k) {
        int[][] pairs = new int[n][2];
        for (int i = 0; i < n; ++i) {
            pairs[i][0] = speed[i];
            pairs[i][1] = efficiency[i];
        }
        Arrays.sort(pairs, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o2[1], o1[1]);
            }

        });
        Queue<Integer> q = new PriorityQueue<>();
        long res = 0l;
        long sum = 0l;
        for (int[] pair : pairs) {
            int s = pair[0];
            int e = pair[1];
            sum += s;
            res = Math.max(res, sum * e);
            q.offer(s);
            while (q.size() >= k) {
                sum -= q.poll();
            }
        }
        final int MOD = (int) (1e9 + 7);
        return (int) (res % MOD);

    }

    // 1000. 合并石头的最低成本 (Minimum Cost to Merge Stones)
    private int[][][] memo1000;
    private int k1000;
    private int n1000;
    private int[] pre1000;

    public int mergeStones(int[] stones, int k) {
        this.n1000 = stones.length;
        this.k1000 = k;
        if ((n1000 - 1) % (k - 1) != 0) {
            return -1;

        }
        this.memo1000 = new int[n1000][n1000][k + 1];
        this.pre1000 = new int[n1000 + 1];
        for (int i = 0; i < n1000; ++i) {
            pre1000[i + 1] = pre1000[i] + stones[i];
        }
        for (int i = 0; i < n1000; ++i) {
            for (int j = i; j < n1000; ++j) {
                Arrays.fill(memo1000[i][j], -1);
            }
        }
        return dfs1000(0, n1000 - 1, 1);

    }

    private int dfs1000(int i, int j, int p) {
        if (memo1000[i][j][p] != -1) {
            return memo1000[i][j][p];
        }
        if (p == 1) {
            return memo1000[i][j][p] = i == j ? 0 : dfs1000(i, j, k1000) + pre1000[j + 1] - pre1000[i];
        }
        int min = Integer.MAX_VALUE;
        for (int m = i; m < j; m += k1000 - 1) {
            min = Math.min(min, dfs1000(i, m, 1) + dfs1000(m + 1, j, p - 1));
        }
        return memo1000[i][j][p] = min;
    }

    // 375. II 猜数字大小 II (Guess Number Higher or Lower)
    private int[][] memo375;

    public int getMoneyAmount(int n) {
        this.memo375 = new int[n][n];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(memo375[i], -1);
        }
        return dfs375(1, n);
    }

    private int dfs375(int i, int j) {
        if (i == j) {
            return 0;
        }
        if (j - i == 1) {
            return i;
        }
        if (memo375[i][j] != -1) {
            return memo375[i][j];
        }
        int min = Integer.MAX_VALUE;
        for (int k = i + 1; k < j; ++k) {
            min = Math.min(min, Math.max(dfs375(i, k - 1), dfs375(k + 1, j)) + k);
        }
        return memo375[i][j] = min;
    }

    // 1201. 丑数 III (Ugly Number III) ---容斥原理 二分查找
    private long ab1201;
    private long bc1201;
    private long ca1201;
    private long abc1201;

    public int nthUglyNumber(int n, int a, int b, int c) {
        long res = -1;
        this.ab1201 = lcm1201(a, b);
        this.bc1201 = lcm1201(b, c);
        this.ca1201 = lcm1201(c, a);
        this.abc1201 = lcm1201(ab1201, c);
        int left = Math.min(Math.min(a, b), c);
        int right = Integer.MAX_VALUE;
        while (left <= right) {
            long mid = left + ((right - left) >> 1);
            long count = mid / a + mid / b + mid / c - mid / ab1201 - mid / bc1201 - mid / ca1201 + mid / abc1201;
            if (count >= n) {
                res = mid;
                right = (int) mid - 1;
            } else {
                left = (int) mid + 1;
            }
        }
        return (int) res;

    }

    private long lcm1201(long a, long b) {
        return a / gcd1201(a, b) * b;
    }

    private long gcd1201(long a, long b) {
        return b == 0 ? a : gcd1201(b, a % b);
    }

    // 391. 完美矩形 (Perfect Rectangle)
    public boolean isRectangleCover(int[][] rectangles) {
        int n = rectangles.length;
        int[][] arr = new int[n * 2][4];
        int index = 0;
        for (int i = 0; i < n; ++i) {
            arr[index++] = new int[] { rectangles[i][0], rectangles[i][1], rectangles[i][3], 1 };
            arr[index++] = new int[] { rectangles[i][2], rectangles[i][1], rectangles[i][3], -1 };
        }
        Arrays.sort(arr, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] == o2[0]) {
                    return Integer.compare(o1[1], o2[1]);
                }
                return Integer.compare(o1[0], o2[0]);
            }

        });
        int left = 0;
        while (left < index) {
            int right = left;
            while (right < index && arr[right][0] == arr[left][0]) {
                ++right;
            }
            List<int[]> list1 = new ArrayList<>();
            List<int[]> list2 = new ArrayList<>();
            for (int j = left; j < right; ++j) {
                int[] cur = new int[] { arr[j][1], arr[j][2] };
                List<int[]> list = arr[j][3] == 1 ? list1 : list2;
                if (list.isEmpty()) {
                    list.add(cur);
                } else {
                    int[] pre = list.get(list.size() - 1);
                    if (pre[1] > cur[0]) {
                        return false;
                    }
                    if (pre[1] == cur[0]) {
                        pre[1] = cur[1];
                    } else {
                        list.add(cur);
                    }
                }
            }
            if (left > 0 && right < index) {
                if (list1.size() != list2.size()) {
                    return false;
                }
                for (int j = 0; j < list1.size(); ++j) {
                    if (list1.get(j)[0] != list2.get(j)[0] || list1.get(j)[1] != list2.get(j)[1]) {
                        return false;
                    }
                }
            } else {
                if (list1.size() + list2.size() != 1) {
                    return false;
                }
            }
            left = right;
        }
        return true;

    }

    // 2611. 老鼠和奶酪 (Mice and Cheese) 参考1029
    public int miceAndCheese(int[] reward1, int[] reward2, int k) {
        int n = reward1.length;
        int[][] arr = new int[n][2];
        for (int i = 0; i < n; ++i) {
            arr[i][0] = reward1[i];
            arr[i][1] = reward2[i];
        }
        Arrays.sort(arr, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o2[0] - o2[1], o1[0] - o1[1]);
            }

        });

        int res = 0;
        for (int i = 0; i < n; ++i) {
            if (i < k) {
                res += arr[i][0];
            } else {
                res += arr[i][1];
            }
        }
        return res;

    }

    // 2608. 图中的最短环 (Shortest Cycle in a Graph)
    private List<Integer>[] g2608;
    private int n2608;

    public int findShortestCycle(int n, int[][] edges) {
        this.g2608 = new ArrayList[n];
        this.n2608 = n;
        for (int i = 0; i < n; ++i) {
            g2608[i] = new ArrayList<>();
        }
        for (int[] e : edges) {
            int a = e[0];
            int b = e[1];
            g2608[a].add(b);
            g2608[b].add(a);
        }
        int res = Integer.MAX_VALUE;
        for (int i = 0; i < n; ++i) {
            res = Math.min(res, getMinimalCycle(i));
        }
        return res == Integer.MAX_VALUE ? -1 : res;

    }

    private int getMinimalCycle(int start) {
        int res = Integer.MAX_VALUE;
        Queue<int[]> q = new LinkedList<>();
        int[] dis = new int[n2608];
        Arrays.fill(dis, -1);
        dis[start] = 0;
        q.offer(new int[] { start, -1 });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int fa = cur[1];
            for (int y : g2608[x]) {
                if (dis[y] == -1) {
                    dis[y] = dis[x] + 1;
                    q.offer(new int[] { y, x });
                } else if (y != fa) {
                    res = Math.min(res, dis[x] + dis[y] + 1);
                }
            }
        }
        return res;
    }

    // 1125. 最小的必要团队 (Smallest Sufficient Team) --状态压缩dp
    private int[] peopleSkills1125;
    private int n1125;
    private long[][] memo1125;
    private int peopleSize1125;

    public int[] smallestSufficientTeam(String[] req_skills, List<List<String>> people) {
        this.n1125 = req_skills.length;
        Map<String, Integer> map = new HashMap<>();
        int index = 0;
        for (String req : req_skills) {
            map.put(req, index++);
        }
        this.peopleSize1125 = people.size();
        peopleSkills1125 = new int[peopleSize1125];
        for (int i = 0; i < peopleSize1125; ++i) {
            for (int j = 0; j < people.get(i).size(); ++j) {
                peopleSkills1125[i] |= 1 << map.get(people.get(i).get(j));
            }
        }
        memo1125 = new long[people.size()][1 << n1125];
        for (int i = 0; i < people.size(); ++i) {
            Arrays.fill(memo1125[i], -1l);
        }
        long mask = dfs1125(0, 0);
        int[] res = new int[Long.bitCount(mask)];
        int i = 0;
        int j = 0;
        while (mask != 0) {
            if ((mask & 1) != 0) {
                res[j++] = i;
            }
            ++i;
            mask >>= 1;
        }
        return res;
    }

    private long dfs1125(int i, int mask) {
        if (mask == (1 << n1125) - 1) {
            return 0l;
        }
        if (i == peopleSize1125) {
            return (1l << peopleSize1125) - 1;
        }
        if (memo1125[i][mask] != -1l) {
            return memo1125[i][mask];
        }
        long mask1 = dfs1125(i + 1, mask);
        long mask2 = dfs1125(i + 1, mask | peopleSkills1125[i]) | (1l << i);
        return memo1125[i][mask] = Long.bitCount(mask1) < Long.bitCount(mask2) ? mask1 : mask2;
    }

    // 1172. 餐盘栈 (Dinner Plate Stacks)
    class DinnerPlates {
        private int capacity;
        private TreeMap<Integer, Stack<Integer>> fullStacks;
        private TreeMap<Integer, Stack<Integer>> notFullStacks;
        private List<Integer> emptyStacks;

        public DinnerPlates(int capacity) {
            this.capacity = capacity;
            this.fullStacks = new TreeMap<>();
            this.notFullStacks = new TreeMap<>();
            this.emptyStacks = new List<>();
            int stacks = 200000 / capacity + 1;
            for (int i = 0; i <= stacks; ++i) {
                emptyStacks.add(i);
            }
        }

        public void push(int val) {
            if (!notFullStacks.isEmpty() && notFullStacks.firstKey() < emptyStacks.first()) {
                Integer notFullIndex = notFullStacks.firstKey();
                Stack<Integer> stack = notFullStacks.get(notFullIndex);
                stack.push(val);
                if (stack.size() == capacity) {
                    fullStacks.put(notFullIndex, stack);
                    notFullStacks.remove(notFullIndex);
                }
            } else {
                int index = emptyStacks.first();
                Stack<Integer> stack = new Stack<>();
                stack.push(val);
                if (stack.size() == capacity) {
                    fullStacks.put(index, stack);
                } else {
                    notFullStacks.put(index, stack);
                }
                emptyStacks.remove(index);
            }

        }

        public int pop() {
            int res = -1;
            if (!notFullStacks.isEmpty() || !fullStacks.isEmpty()) {
                if (!notFullStacks.isEmpty() && !fullStacks.isEmpty() && notFullStacks.lastKey() > fullStacks.lastKey()
                        || fullStacks.isEmpty()) {
                    Integer notFullIndex = notFullStacks.lastKey();
                    Stack<Integer> stack = notFullStacks.get(notFullIndex);
                    res = stack.pop();
                    if (stack.isEmpty()) {
                        notFullStacks.remove(notFullIndex);
                        emptyStacks.add(notFullIndex);
                    }
                } else {
                    Integer index = fullStacks.lastKey();
                    Stack<Integer> stack = fullStacks.get(index);
                    res = stack.pop();
                    fullStacks.remove(index);
                    if (stack.isEmpty()) {
                        emptyStacks.add(index);
                    } else {
                        notFullStacks.put(index, stack);
                    }
                }

            }
            return res;
        }

        public int popAtStack(int index) {
            int res = -1;
            if (notFullStacks.containsKey(index)) {
                Stack<Integer> stack = notFullStacks.get(index);
                res = stack.pop();
                if (stack.isEmpty()) {
                    notFullStacks.remove(index);
                    emptyStacks.add(index);
                }
            } else if (fullStacks.containsKey(index)) {
                Stack<Integer> stack = fullStacks.get(index);
                res = stack.pop();
                fullStacks.remove(index);
                if (stack.isEmpty()) {
                    emptyStacks.add(index);
                } else {
                    notFullStacks.put(index, stack);
                }
            }
            return res;
        }
    }

    // 214. 最短回文串 (Shortest Palindrome) --rabin karp
    public String shortestPalindrome(String s) {
        int n = s.length();
        int left = 0;
        int right = 0;
        int base = 31;
        int mod = (int) (1e9 + 7);
        int mul = 1;
        int best = -1;
        for (int i = 0; i < n; ++i) {
            left = (int) (((long) left * base + s.charAt(i)) % mod);
            right = (int) ((right + (long) mul * s.charAt(i)) % mod);
            if (left == right) {
                best = i;
            }
            mul = (int) ((long) mul * base % mod);
        }
        if (best == n - 1) {
            return s;
        }
        StringBuilder res = new StringBuilder(s.substring(best + 1)).reverse();
        return res + s;

    }

    // 363. 矩形区域不超过 K 的最大数值和 (Max Sum of Rectangle No Larger Than K)
    public int maxSumSubmatrix(int[][] matrix, int k) {
        int m = matrix.length;
        int n = matrix[0].length;
        int res = Integer.MIN_VALUE;
        for (int up = 0; up < m; ++up) {
            int[] pre = new int[n];
            for (int down = up; down < m; ++down) {
                List<Integer> set = new List<>();
                set.add(0);
                int curPre = 0;
                for (int j = 0; j < n; ++j) {
                    pre[j] += matrix[down][j];
                    curPre += pre[j];
                    Integer ceiling = set.ceiling(curPre - k);
                    if (ceiling != null) {
                        res = Math.max(res, curPre - ceiling);
                    }
                    set.add(curPre);
                }
            }
        }
        return res;

    }

    // 363. 矩形区域不超过 K 的最大数值和 (Max Sum of Rectangle No Larger Than K)
    // follow up： 行数远大于列数时
    // follow up: What if the number of rows is much larger than the
    // number of columns?
    public int maxSumSubmatrix2(int[][] matrix, int k) {
        int m = matrix.length;
        int n = matrix[0].length;
        int res = Integer.MIN_VALUE;
        for (int left = 0; left < n; ++left) {
            int[] pre = new int[m];
            for (int right = left; right < n; ++right) {
                int curPre = 0;
                List<Integer> set = new List<>();
                set.add(0);
                for (int i = 0; i < m; ++i) {
                    pre[i] += matrix[i][right];
                    curPre += pre[i];
                    Integer ceiling = set.ceiling(curPre - k);
                    if (ceiling != null) {
                        res = Math.max(res, curPre - ceiling);
                    }
                    set.add(curPre);
                }
            }
        }
        return res;

    }

    // 6361. 对角线上的质数 (Prime In Diagonal)
    public int diagonalPrime(int[][] nums) {
        int res = 0;
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            if (checkPrime(nums[i][i])) {
                res = Math.max(res, nums[i][i]);
            }
            if (checkPrime(nums[i][n - i - 1])) {
                res = Math.max(res, nums[i][n - i - 1]);
            }
        }
        return res;

    }

    private boolean checkPrime(int num) {
        for (int i = 2; i * i <= num; ++i) {
            if (num % i == 0) {
                return false;
            }
        }
        return num != 1;
    }

    // 6360. 等值距离和 (Sum of Distances)
    public long[] distance(int[] nums) {
        int n = nums.length;
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            map.computeIfAbsent(nums[i], k -> new ArrayList<>()).add(i);
        }
        long[] res = new long[n];
        for (int i = 0; i < n; ++i) {
            List<Integer> list = map.getOrDefault(nums[i], new ArrayList<>());
            if (list.size() <= 1) {
                continue;
            }
            long sum = 0l;
            for (int num : list) {
                sum += num;
            }
            long pre = 0l;
            for (int j = 0; j < list.size(); ++j) {
                res[list.get(j)] = (long) list.get(j) * j - pre + sum - pre - (list.size() - j) * (long) list.get(j);
                pre += list.get(j);
            }
            map.remove(nums[i]);
        }
        return res;

    }

    // 6353. 网格图中最少访问的格子数 (Minimum Number of Visited Cells in a Grid) --超时
    public int minimumVisitedCells(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] dis = new int[m][n];
        for (int i = 0; i < m; ++i) {
            Arrays.fill(dis[i], Integer.MAX_VALUE);
        }
        dis[0][0] = 1;
        Queue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(dis[o1[0]][o1[1]], dis[o2[0]][o2[1]]);
            }

        });
        q.offer(new int[] { 0, 0 });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int y = cur[1];
            int d = dis[x][y];
            if (x == m - 1 && y == n - 1) {
                return d;
            }
            for (int k = y + 1; k <= Math.min(n - 1, grid[x][y] + y); ++k) {
                if (d + 1 < dis[x][k]) {
                    dis[x][k] = d + 1;
                    q.offer(new int[] { x, k, d + 1 });
                }
            }
            for (int k = x + 1; k <= Math.min(m - 1, grid[x][y] + x); ++k) {
                if (d + 1 < dis[k][y]) {
                    dis[k][y] = d + 1;
                    q.offer(new int[] { k, y, d + 1 });
                }
            }
        }
        return -1;
    }

    // 1955. 统计特殊子序列的数目 (Count Number of Special Subsequences)
    public int countSpecialSubsequences(int[] nums) {
        final int MOD = (int) (1e9 + 7);
        int[] f = new int[3];
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] == 0) {
                f[0] = (f[0] * 2 + 1) % MOD;
            } else if (nums[i] == 1) {
                f[1] = (f[1] * 2 % MOD + f[0]) % MOD;
            } else {
                f[2] = (f[2] * 2 % MOD + f[1]) % MOD;
            }
        }
        return f[2];

    }

    // 面试题 17.24. 最大子矩阵 (Max Submatrix LCCI)
    public int[] getMaxMatrix(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        int[] res = null;
        int max = Integer.MIN_VALUE;
        for (int up = 0; up < m; ++up) {
            int[] pre = new int[n];
            for (int down = up; down < m; ++down) {
                int[] minPre = new int[] { -1, 0 };
                int curPre = 0;
                for (int j = 0; j < n; ++j) {
                    pre[j] += matrix[down][j];
                    curPre += pre[j];
                    if (curPre - minPre[1] > max) {
                        max = curPre - minPre[1];
                        res = new int[] { up, minPre[0] + 1, down, j };
                    }
                    if (curPre < minPre[1]) {
                        minPre = new int[] { j, curPre };
                    }
                }
            }
        }
        return res;

    }

    // 2577. 在网格图中访问一个格子的最少时间 (Minimum Time to Visit a Cell In a Grid)
    public int minimumTime(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        if (grid[1][0] > 1 && grid[0][1] > 1) {
            return -1;
        }
        int[][] dirs = { { 0, -1 }, { 0, 1 }, { 1, 0 }, { -1, 0 } };
        Queue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[2], o2[2]);
            }

        });
        int[][] dis = new int[m][n];
        for (int i = 0; i < m; ++i) {
            Arrays.fill(dis[i], Integer.MAX_VALUE);
        }
        dis[0][0] = 0;
        q.offer(new int[] { 0, 0, 0 });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int y = cur[1];
            int d = cur[2];
            if (d > dis[x][y]) {
                continue;
            }
            if (x == m - 1 && y == n - 1) {
                return d;
            }
            for (int[] dir : dirs) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    int nd = Math.max(grid[nx][ny], d + 1);
                    nd += (nd - nx - ny) % 2;
                    if (nd < dis[nx][ny]) {
                        dis[nx][ny] = nd;
                        q.offer(new int[] { nx, ny, nd });
                    }
                }
            }
        }
        return -1;

    }

    // 372. 超级次方 (Super Pow)
    public int superPow(int a, int[] b) {
        final int MOD = 1337;
        a %= MOD;
        return dfs372(a, b, b.length - 1);
    }

    private int dfs372(int a, int[] b, int i) {
        if (i < 0) {
            return 1;
        }
        final int MOD = 1337;
        // pow(5, 123)
        // == pow(5, 3) * pow(5, 120)
        // == pow(5, 3) * pow(pow(5, 12), 10)
        int part1 = pow372(a, b[i]);
        int part2 = pow372(dfs372(a, b, i - 1), 10);
        return part1 * part2 % MOD;
    }

    // 快速幂
    private int pow372(int a, int b) {
        if (b == 0) {
            return 1;
        }
        final int MOD = 1337;
        if (b % 2 == 1) {
            return a * pow372(a, b - 1) % MOD;
        }
        int y = pow372(a, b / 2);
        return y * y % MOD;
    }

    // 968. 监控二叉树 (Binary Tree Cameras)
    private int res968;

    public int minCameraCover(TreeNode root) {
        int status = dfs968(root);
        return status == 0 ? ++res968 : res968;
    }

    private int dfs968(TreeNode root) {
        // 0：未被覆盖(当前节点未被照到)
        // 1：已被覆盖(摄像头已经照到这个节点)
        // 2：需放置摄像头
        if (root == null) {
            return 1;
        }
        // 00 01 10 11 02 20 22 12 21
        int left = dfs968(root.left);
        int right = dfs968(root.right);
        if (left == 0 || right == 0) {
            ++res968;
            return 2;
        }
        if (left == 1 && right == 1) {
            return 0;
        }
        return 1;
    }

    // 1639. 通过给定词典构造目标字符串的方案数 (Number of Ways to Form a Target String Given a
    // Dictionary)
    private int m1639;
    private int n1639;
    private int[][] memo1639;
    private int[][] counts1639;
    private String target1639;

    public int numWays(String[] words, String target) {
        this.m1639 = words[0].length();
        this.n1639 = target.length();
        this.memo1639 = new int[m1639][n1639];
        this.target1639 = target;
        for (int i = 0; i < m1639; ++i) {
            Arrays.fill(memo1639[i], -1);
        }
        this.counts1639 = new int[m1639][26];
        for (int i = 0; i < words.length; ++i) {
            for (int j = 0; j < words[i].length(); ++j) {
                ++counts1639[j][words[i].charAt(j) - 'a'];
            }
        }
        return dfs1639(0, 0);

    }

    private int dfs1639(int i, int j) {
        if (j == n1639) {
            return 1;
        }
        if (i == m1639) {
            return 0;
        }
        if (m1639 - i < n1639 - j) {
            return 0;
        }
        if (memo1639[i][j] != -1) {
            return memo1639[i][j];
        }
        final int MOD = (int) (1e9 + 7);
        int count = counts1639[i][target1639.charAt(j) - 'a'];
        return memo1639[i][j] = (int) (((long) count * dfs1639(i + 1, j + 1) % MOD + dfs1639(i + 1, j) % MOD) % MOD);
    }

    // 1643. 第 K 条最小指令 (Kth Smallest Instructions)
    public String kthSmallestPath(int[] destination, int k) {
        int h = destination[1];
        int v = destination[0];
        // 组合数
        int[][] comp = new int[h + v][h];
        for (int i = 0; i < h + v; ++i) {
            for (int j = 0; j <= i && j < h; ++j) {
                comp[i][j] = j == 0 ? 1 : (comp[i - 1][j - 1] + comp[i - 1][j]);
            }
        }
        StringBuilder res = new StringBuilder();
        while (h + v > 0) {
            if (h > 0) {
                int o = comp[h + v - 1][h - 1];
                if (k > o) {
                    res.append("V");
                    --v;
                    k -= o;
                } else {
                    res.append("H");
                    --h;
                }
            } else {
                res.append("V");
                --v;
            }
        }
        return res.toString();

    }

    // 639. 解码方法 II (Decode Ways II)
    private int n639;
    private char[] arr639;
    private int[] memo639;

    public int numDecodings(String s) {
        this.arr639 = s.toCharArray();
        this.n639 = s.length();
        if (s.charAt(0) == '0') {
            return 0;
        }
        this.memo639 = new int[n639];
        Arrays.fill(memo639, -1);
        return dfs639(n639 - 1);
    }

    private int dfs639(int i) {
        if (i < 0) {
            return 1;
        }
        if (memo639[i] != -1) {
            return memo639[i];
        }
        final int MOD = (int) (1e9 + 7);
        int res = 0;
        char c = arr639[i];
        if (c >= '0' && c <= '6') {
            if (c != '0') {
                res = (res + dfs639(i - 1)) % MOD;
            }
            if (i > 0) {
                char pre = arr639[i - 1];
                if (pre == '1' || pre == '2' || pre == '*') {
                    if (pre == '*') {
                        res = (res + dfs639(i - 2) * 2 % MOD) % MOD;
                    } else {
                        res = (res + dfs639(i - 2)) % MOD;
                    }
                }
            }
        } else if (c >= '7' && c <= '9') {
            res = (res + dfs639(i - 1)) % MOD;
            if (i > 0) {
                char pre = arr639[i - 1];
                if (pre == '1' || pre == '*') {
                    res = (res + dfs639(i - 2)) % MOD;
                }
            }
        } else {
            res = (int) (res + (long) dfs639(i - 1) * 9 % MOD) % MOD;
            if (i > 0) {
                char pre = arr639[i - 1];
                if (pre == '1') {
                    res = (int) (res + (long) dfs639(i - 2) * 9 % MOD) % MOD;
                } else if (pre == '2') {
                    res = (int) (res + (long) dfs639(i - 2) * 6 % MOD) % MOD;
                } else if (pre == '*') {
                    // 「**」 可以组成 11-19、21-26 共15种状态
                    res = (int) (res + (long) dfs639(i - 2) * 15 % MOD) % MOD;
                }
            }
        }
        return memo639[i] = res;
    }

    // 732. 我的日程安排表 III (My Calendar III) --差分 还需掌握 线段树
    class MyCalendarThree {
        private TreeMap<Integer, Integer> map;

        public MyCalendarThree() {
            map = new TreeMap<>();

        }

        public int book(int startTime, int endTime) {
            map.merge(startTime, 1, Integer::sum);
            map.merge(endTime, -1, Integer::sum);
            int max = 0;
            int cur = 0;
            for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
                cur += entry.getValue();
                max = Math.max(max, cur);
            }
            return max;
        }
    }

    // 699. 掉落的方块 (Falling Squares) --还需掌握 有续集合、线段树
    public List<Integer> fallingSquares(int[][] positions) {
        int n = positions.length;
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            int left1 = positions[i][0];
            int right1 = positions[i][0] + positions[i][1];
            int height = positions[i][1];
            for (int j = 0; j < i; ++j) {
                int left2 = positions[j][0];
                int right2 = positions[j][0] + positions[j][1];
                if (right1 <= left2 || left1 >= right2) {
                    continue;
                }
                height = Math.max(height, res.get(j) + positions[i][1]);
            }
            res.add(height);
        }
        for (int i = 1; i < n; ++i) {
            res.set(i, Math.max(res.get(i - 1), res.get(i)));
        }
        return res;

    }

    // 1851. 包含每个查询的最小区间 (Minimum Interval to Include Each Query)
    public int[] minInterval(int[][] intervals, int[] queries) {
        int n = queries.length;
        int[] res = new int[n];
        Arrays.fill(res, -1);
        Arrays.sort(intervals, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }
        });
        Integer[] ids = IntStream.range(0, n).boxed().toArray(Integer[]::new);
        Arrays.sort(ids, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(queries[o1], queries[o2]);
            }

        });
        Queue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[1] - o1[0], o2[1] - o2[0]);
            }

        });
        int i = 0;
        for (int id : ids) {
            int x = queries[id];
            while (i < intervals.length && intervals[i][0] <= x) {
                q.offer(intervals[i]);
                ++i;
            }
            while (!q.isEmpty() && q.peek()[1] < x) {
                q.poll();
            }
            if (!q.isEmpty()) {
                res[id] = q.peek()[1] - q.peek()[0] + 1;
            }
        }
        return res;

    }

    // 1157. 子数组中占绝大多数的元素 (Online Majority Element In Subarray) -- 还需掌握 线段树
    class MajorityChecker {
        private Map<Integer, List<Integer>> valToIndex;
        private List<int[]> sizeToVal;

        public MajorityChecker(int[] arr) {
            valToIndex = new HashMap<>();
            sizeToVal = new ArrayList<>();

            for (int i = 0; i < arr.length; ++i) {
                valToIndex.computeIfAbsent(arr[i], k -> new ArrayList<>()).add(i);
            }
            for (Map.Entry<Integer, List<Integer>> entry : valToIndex.entrySet()) {
                int size = entry.getValue().size();
                int val = entry.getKey();
                sizeToVal.add(new int[] { size, val });
            }
            Collections.sort(sizeToVal, new Comparator<int[]>() {

                @Override
                public int compare(int[] o1, int[] o2) {
                    return Integer.compare(o2[0], o1[0]);
                }

            });
        }

        public int query(int left, int right, int threshold) {
            for (int[] item : sizeToVal) {
                int size = item[0];
                int val = item[1];
                if (size < threshold) {
                    return -1;
                }
                int ceiling = binarySearchCeiling(valToIndex.get(val), left);
                if (ceiling == -1) {
                    continue;
                }
                int floor = binarySearchFloor(valToIndex.get(val), right);
                if (floor == -1) {
                    continue;
                }
                if (floor - ceiling + 1 >= threshold) {
                    return val;
                }
            }
            return -1;
        }

        // 找排序list中 <= target 的最大值对应的索引
        private int binarySearchFloor(List<Integer> list, int target) {
            if (list == null || list.isEmpty()) {
                return -1;
            }
            int n = list.size();
            if (list.get(0) > target) {
                return -1;
            }
            if (list.get(n - 1) <= target) {
                return n - 1;
            }
            int res = -1;
            int left = 0;
            int right = n - 1;
            while (left <= right) {
                int mid = left + ((right - left) >> 1);
                if (list.get(mid) <= target) {
                    res = mid;
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            return res;
        }

        // 找排序list中 >= target 的最小值对应的索引
        private int binarySearchCeiling(List<Integer> list, int target) {
            if (list == null || list.isEmpty()) {
                return -1;
            }
            int n = list.size();
            if (list.get(0) >= target) {
                return 0;
            }
            if (list.get(n - 1) < target) {
                return -1;
            }
            int res = -1;
            int left = 0;
            int right = n - 1;
            while (left <= right) {
                int mid = left + ((right - left) >> 1);
                if (list.get(mid) >= target) {
                    res = mid;
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            return res;
        }
    }


    // 2402. 会议室 III (Meeting Rooms III)
    // public int mostBooked(int n, int[][] meetings) {

    // }

    // 1316. 不同的循环子字符串 (Distinct Echo Substrings)
    // public int distinctEchoSubstrings(String text) {

    // }

    // 1473. 粉刷房子 III (Paint House III)
    // private int[][][] memo;
    // private int m1473;
    // private int n1473;
    // private int target1473;
    // private int[] houses1473;
    // private int[][] cost1473;

    // public int minCost(int[] houses, int[][] cost, int m, int n, int target) {
    // // memo[i][j][k] 将[0,i]房子涂色 第i个房子被涂成第j种颜色 且它属于第k个街区的 最小花销
    // memo = new int[m][n][target];
    // m1473 = m;
    // n1473 = n;
    // target1473 = target;
    // cost1473 = cost;
    // houses1473 = houses;
    // int res = dfs(0, n + 1, target);
    // return res == (int) 1e8 ? -1 : res;

    // }

    // private int dfs(int i, int lastColor, int kinds) {
    // if (i == m1473 || kinds < 0 || kinds > m1473 - i) {
    // if (i == m1473 && kinds == 0) {
    // return 0;
    // }
    // return (int) 1e8;
    // }

    // if (memo[i][lastColor][target1473] != 0) {
    // return memo[i][lastColor][target1473];
    // }
    // int min = (int) 1e8;
    // if (houses1473[i] != 0) {
    // min = Math.min(min, dfs(i + 1, houses1473[i], kinds + (lastColor !=
    // houses1473[i] ? -1 : 0)));
    // return memo[i][lastColor][target1473] = min;
    // }
    // for (int color = 1; color <= n1473; ++color) {
    // min = Math.min(min, cost1473[i][color - 1] + dfs(i + 1, color, kinds +
    // (lastColor != color ? -1 : 0)));
    // }
    // return memo[i][lastColor][target1473] = min;
    // }

    // 1671. 得到山形数组的最少删除次数 (Minimum Number of Removals to Make Mountain Array)
    // public int minimumMountainRemovals(int[] nums) {
    // }

    // 1937. 扣分后的最大得分 (Maximum Number of Points with Cost)
    // private int m;
    // private int n;
    // private int[][] points;
    // private long[][] memo;

    // public long maxPoints(int[][] points) {
    // this.m = points.length;
    // this.n = points[0].length;
    // this.points = points;
    // this.memo = new long[m][n];
    // for (int i = 0; i < m; ++i) {
    // Arrays.fill(memo[i], Long.MIN_VALUE);
    // }
    // long res = 0l;
    // for (int j = 0; j < n; ++j) {
    // res = Math.max(res, dfs(1, j) + points[0][j]);
    // }
    // return res;
    // }

    // private long dfs(int row, int lastCol) {
    // if (row == m) {
    // return 0;
    // }
    // if (memo[row][lastCol] != Long.MIN_VALUE) {
    // return memo[row][lastCol];
    // }
    // long res = Long.MIN_VALUE;
    // for (int j = 0; j < n; ++j) {
    // res = Math.max(res, dfs(row + 1, j) + points[row][j] - Math.abs(lastCol -
    // j));
    // }
    // return memo[row][lastCol] = res;
    // }

    // 1771. 由子序列构造的最长回文串的长度 (Maximize Palindrome Length From Subsequences)
    // private int m;
    // private int n;
    // private String s;
    // private int[][] memo;

    // public int longestPalindrome(String word1, String word2) {
    // this.m = word1.length();
    // this.n = word2.length();
    // this.s = word1 + word2;
    // this.memo = new int[m + n][m + n];
    // for (int i = 0; i < m + n; ++i) {
    // Arrays.fill(memo[i], -1);
    // }
    // int res = dfs(false, 0, m + n - 1);
    // return res > 1 ? res : 0;

    // }

    // private int dfs(boolean b, int i, int j) {
    // if (i > j) {
    // return 0;
    // }
    // if (i == j) {
    // return 1;
    // }
    // if (memo[i][j] != -1) {
    // return memo[i][j];
    // }
    // if (i >= m || j < m) {
    // if (!b) {
    // return 0;
    // }
    // }
    // if (s.charAt(i) == s.charAt(j)) {
    // return memo[i][j] = dfs(true, i + 1, j - 1) + 2;
    // }
    // return memo[i][j] = Math.max(dfs(b, i + 1, j), dfs(b, i, j - 1));
    // }

    // 面试题 17.08. 马戏团人塔 (Circus Tower LCCI)
    // public int bestSeqAtIndex(int[] height, int[] weight) {
    // int n = height.length;

    // }

    // 1406. 石子游戏 III (Stone Game III)
    // public String stoneGameIII(int[] stoneValue) {

    // }

}