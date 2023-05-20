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
import java.util.TreeSet;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.awt.Point;

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
        return dfs1105(0);
    }

    private int dfs1105(int i) {
        if (i == n1105) {
            return 0;
        }
        if (memo1105[i] != 0) {
            return memo1105[i];
        }
        int min = Integer.MAX_VALUE;
        int thick = 0;
        int height = 0;
        for (int j = i; j < n1105 && thick + books1105[j][0] <= shelfWidth1105; ++j) {
            thick += books1105[j][0];
            height = Math.max(height, books1105[j][1]);
            min = Math.min(min, dfs1105(j + 1) + height);
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
    private int[][] memo1335;
    private int n1335;
    private int[] jobDifficulty1335;
    private int d1335;

    public int minDifficulty(int[] jobDifficulty, int d) {
        this.n1335 = jobDifficulty.length;
        this.d1335 = d;
        this.jobDifficulty1335 = jobDifficulty;
        this.memo1335 = new int[n1335][d];
        for (int i = 0; i < n1335; ++i) {
            Arrays.fill(memo1335[i], (int) 1e6);
        }
        int res = dfs1335(0, 0);
        return res < (int) 1e6 ? res : -1;

    }

    private int dfs1335(int i, int count) {
        if (d1335 - count > n1335 - i) {
            return (int) 1e6;
        }
        if (i == n1335 || count == d1335) {
            return i == n1335 && count == d1335 ? 0 : (int) 1e6;
        }
        if (memo1335[i][count] != (int) 1e6) {
            return memo1335[i][count];
        }
        int max = 0;
        int res = (int) 1e6;
        for (int j = i; j < n1335; ++j) {
            max = Math.max(max, jobDifficulty1335[j]);
            res = Math.min(res, dfs1335(j + 1, count + 1) + max);
        }
        return memo1335[i][count] = res;
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
                mapMovie.computeIfAbsent(movie, k -> new ArrayList<>()).add(new Bean(shop, movie, price));
                mapShop.computeIfAbsent(shop, k -> new HashMap<>()).put(movie, price);
                mapData.put(new Bean(shop, movie, price), price);
            }

        }

        public List<Integer> search(int movie) {
            List<Integer> res = new ArrayList<>();
            List<Bean> set = mapMovie.getOrDefault(movie, new ArrayList<>());
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
            List<Bean> set = mapMovie.getOrDefault(movie, new ArrayList<>());
            Bean removed = new Bean(shop, movie, price);
            set.remove(removed);

            // 借出的电影
            loan.computeIfAbsent(price, k -> new TreeMap<>()).computeIfAbsent(shop, k -> new ArrayList<>()).add(movie);
        }

        public void drop(int shop, int movie) {
            // 从已借出的还走
            Bean b = new Bean(shop, movie, 0);
            int price = mapData.get(b);
            TreeMap<Integer, List<Integer>> map = loan.getOrDefault(price, new TreeMap<>());
            List<Integer> movies = map.getOrDefault(shop, new ArrayList<>());
            movies.remove(movie);

            // 把已借出的还到商店
            Map<Integer, Integer> s = mapShop.getOrDefault(shop, new HashMap<>());
            s.put(movie, price);

            List<Bean> m = mapMovie.getOrDefault(movie, new ArrayList<>());
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
        private TreeSet<Integer> emptyStacks;

        public DinnerPlates(int capacity) {
            this.capacity = capacity;
            this.fullStacks = new TreeMap<>();
            this.notFullStacks = new TreeMap<>();
            this.emptyStacks = new TreeSet<>();
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
                TreeSet<Integer> set = new TreeSet<>();
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
                TreeSet<Integer> set = new TreeSet<>();
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

    // 2639. 查询网格图中每一列的宽度 (Find the Width of Columns of a Grid)
    public int[] findColumnWidth(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[] res = new int[n];
        for (int j = 0; j < n; ++j) {
            int max = 0;
            for (int i = 0; i < m; ++i) {
                int num = grid[i][j];
                max = Math.max(max, String.valueOf(num).length());
            }
            res[j] = max;
        }
        return res;

    }

    // 2640. 一个数组所有前缀的分数 (Find the Score of All Prefixes of an Array)
    public long[] findPrefixScore(int[] nums) {
        int n = nums.length;
        long[] res = new long[n];
        long max = nums[0];
        res[0] = nums[0] * 2;
        for (int i = 1; i < n; ++i) {
            max = Math.max(nums[i], max);
            long co = max + nums[i];
            res[i] = res[i - 1] + co;
        }
        return res;

    }

    // 2641. 二叉树的堂兄弟节点 II (Cousins in Binary Tree II)
    public TreeNode replaceValueInTree(TreeNode root) {
        Map<Integer, Integer> map = new HashMap<>();
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int level = 0;
        while (!q.isEmpty()) {
            int size = q.size();
            int sum = 0;
            for (int i = 0; i < size; ++i) {
                TreeNode node = q.poll();
                sum += node.val;
                if (node.left != null) {
                    q.offer(node.left);
                }
                if (node.right != null) {
                    q.offer(node.right);
                }
            }
            map.put(level++, sum);
        }
        root.val = 0;
        level = 0;
        q.offer(root);
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                TreeNode node = q.poll();
                int cur = 0;
                if (node.left != null) {
                    cur += node.left.val;
                }
                if (node.right != null) {
                    cur += node.right.val;
                }
                if (node.left != null) {
                    node.left.val = map.get(level + 1) - cur;
                    q.offer(node.left);
                }
                if (node.right != null) {
                    node.right.val = map.get(level + 1) - cur;
                    q.offer(node.right);
                }
            }
            ++level;
        }
        return root;

    }

    // 2642. 设计可以求最短路径的图类 (Design Graph With Shortest Path Calculator)
    class Graph {
        private int n;

        private Map<Integer, List<int[]>> g;

        public Graph(int n, int[][] edges) {
            this.n = n;
            this.g = new HashMap<>();
            for (int[] e : edges) {
                addEdge(e);
            }
        }

        public void addEdge(int[] edge) {
            int a = edge[0];
            int b = edge[1];
            int cost = edge[2];
            g.computeIfAbsent(a, k -> new ArrayList<>()).add(new int[] { b, cost });
        }

        public int shortestPath(int node1, int node2) {
            if (node1 == node2) {
                return 0;
            }
            int[] dis = new int[n];
            Arrays.fill(dis, Integer.MAX_VALUE);
            dis[node1] = 0;
            Queue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

                @Override
                public int compare(int[] o1, int[] o2) {
                    return Integer.compare(o1[1], o2[1]);
                }

            });

            q.offer(new int[] { node1, 0 });
            dis[node1] = 0;
            while (!q.isEmpty()) {
                int[] cur = q.poll();
                int x = cur[0];
                int d = cur[1];
                if (d > dis[x]) {
                    continue;
                }
                if (x == node2) {
                    return d;
                }
                for (int[] nei : g.getOrDefault(x, new ArrayList<>())) {
                    int y = nei[0];
                    int c = nei[1];
                    if (d + c < dis[y]) {
                        dis[y] = d + c;
                        q.offer(new int[] { y, d + c });
                    }
                }
            }
            return -1;
        }
    }

    // 2643. 一最多的行 (Row With Maximum Ones)
    public int[] rowAndMaximumOnes(int[][] mat) {
        int m = mat.length;
        int[] res = new int[2];
        for (int i = 0; i < m; ++i) {
            int count = Arrays.stream(mat[i]).sum();
            if (count > res[1]) {
                res[0] = i;
                res[1] = count;
            }
        }
        return res;

    }

    // 2644. 找出可整除性得分最大的整数 (Find the Maximum Divisibility Score)
    public int maxDivScore(int[] nums, int[] divisors) {
        int count = 0;
        int res = divisors[0];
        for (int d : divisors) {
            int cur = 0;
            for (int num : nums) {
                if (num % d == 0) {
                    ++cur;
                }
            }
            if (cur > count) {
                count = cur;
                res = d;
            } else if (cur == count && res > d) {
                res = d;
            }
        }
        return res;

    }

    // 2645. 构造有效字符串的最少插入数 (Minimum Additions to Make Valid String)
    public int addMinimum(String word) {
        int n = word.length();
        char[] arr = word.toCharArray();
        int res = 0;
        if (n == 1) {
            return 2;
        }
        int i = 0;
        while (i < n) {
            if (arr[i] == 'a') {
                if (i + 1 < n && arr[i + 1] == 'b') {
                    if (i + 2 < n && arr[i + 2] == 'c') {
                        i += 3;
                    } else {
                        i += 2;
                        ++res;
                    }
                } else if (i + 1 < n && arr[i + 1] == 'c') {
                    ++res;
                    i += 2;
                } else {
                    res += 2;
                    ++i;
                }
            } else if (arr[i] == 'b') {
                if (i + 1 < n && arr[i + 1] == 'c') {
                    ++res;
                    i += 2;
                } else {
                    res += 2;
                    ++i;
                }
            } else {
                res += 2;
                ++i;
            }
        }
        return res;

    }

    // 2646. 最小化旅行的价格总和 (Minimize the Total Price of the Trips)
    private Map<Integer, List<Integer>> g2646;
    private int[] price2646;
    private int[] counts2646;
    private int end2646;

    public int minimumTotalPrice(int n, int[][] edges, int[] price, int[][] trips) {
        this.g2646 = new HashMap<>();
        for (int[] e : edges) {
            int a = e[0];
            int b = e[1];
            g2646.computeIfAbsent(a, k -> new ArrayList<>()).add(b);
            g2646.computeIfAbsent(b, k -> new ArrayList<>()).add(a);
        }
        this.price2646 = price;
        counts2646 = new int[n];
        for (int[] t : trips) {
            end2646 = t[1];
            paths2646(t[0], -1);
        }
        int[] res = dfs2646(0, -1);

        return Math.min(res[0], res[1]);

    }

    // [a,b] a价格减半 ，b价格不减半
    private int[] dfs2646(int x, int fa) {
        int half = price2646[x] * counts2646[x] / 2;
        int notHalf = price2646[x] * counts2646[x];
        for (int y : g2646.getOrDefault(x, new ArrayList<>())) {
            if (y != fa) {
                int[] cur = dfs2646(y, x);
                notHalf += Math.min(cur[0], cur[1]);
                half += cur[1];
            }
        }
        return new int[] { half, notHalf };
    }

    private boolean paths2646(int x, int fa) {
        if (x == end2646) {
            ++counts2646[x];
            return true;
        }
        for (int y : g2646.getOrDefault(x, new ArrayList<>())) {
            if (y != fa && paths2646(y, x)) {
                ++counts2646[x];
                return true;
            }
        }
        return false;
    }

    // 1621. 大小为 K 的不重叠线段的数目 (Number of Sets of K Non-Overlapping Line Segments)
    private int[][] memo1621;
    private int n1621;
    private int k1621;

    public int numberOfSets(int n, int k) {
        this.n1621 = n;
        this.k1621 = k;
        this.memo1621 = new int[n][k];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(memo1621[i], -1);
        }
        return dfs1621(0, 0);

    }

    private int dfs1621(int i, int count) {
        if (i == n1621 || count == k1621) {
            return count == k1621 ? 1 : 0;
        }
        if (n1621 - i - 1 < k1621 - count) {
            return 0;
        }
        if (memo1621[i][count] != -1) {
            return memo1621[i][count];
        }
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        res = (res + dfs1621(i + 1, count)) % MOD;
        for (int j = i + 1; j < n1621; ++j) {
            if (n1621 - j - 1 < k1621 - count - 1) {
                break;
            }
            res = (res + dfs1621(j, count + 1)) % MOD;
        }
        return memo1621[i][count] = res;
    }

    // 126. 单词接龙 II (Word Ladder II)
    private List<List<String>> res126;
    private Map<String, Set<String>> map126;
    private int count126;
    private String beginWord126;
    private Map<Integer, List<String>> adj126;

    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        this.res126 = new ArrayList<>();
        this.beginWord126 = beginWord;
        Set<String> wordSet = new HashSet<>(wordList);
        if (!wordSet.contains(endWord)) {
            return res126;
        }
        wordSet.remove(beginWord);
        this.adj126 = new HashMap<>();
        this.map126 = new HashMap<>();
        for (String s : wordSet) {
            char[] arr = s.toCharArray();
            for (int i = 0; i < arr.length; ++i) {
                char temp = arr[i];
                arr[i] = '_';
                map126.computeIfAbsent(String.valueOf(arr), k -> new HashSet<>()).add(s);
                arr[i] = temp;
            }
        }
        this.count126 = 1;
        boolean flag = false;
        Queue<String> q = new LinkedList<>();
        Set<String> set = new HashSet<>();
        set.add(beginWord);
        q.offer(beginWord);
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                String s = q.poll();
                if (s.equals(endWord)) {
                    flag = true;
                    break;
                }
                char[] arr = s.toCharArray();
                for (int j = 0; j < arr.length; ++j) {
                    char temp = arr[j];
                    arr[j] = '_';
                    for (String search : map126.getOrDefault(String.valueOf(arr), new HashSet<>())) {
                        if (!set.contains(search) && !search.equals(s)) {
                            adj126.computeIfAbsent(count126, k -> new ArrayList<>()).add(search);
                            set.add(search);
                            q.offer(search);
                        }
                    }
                    arr[j] = temp;
                }
            }
            if (flag) {
                break;
            }
            ++count126;
        }
        if (!flag) {
            return res126;
        }
        set.clear();
        set.add(endWord);
        List<String> list = new ArrayList<>();
        list.add(endWord);
        dfs126(list, set);
        return res126;
    }

    private void dfs126(List<String> list, Set<String> set) {
        boolean flag = check126(list.get(list.size() - 1), beginWord126);
        if (list.size() == count126 - 1 || flag) {
            if (list.size() == count126 - 1 && flag) {
                list.add(beginWord126);
                Collections.reverse(list);
                res126.add(new ArrayList<>(list));
                Collections.reverse(list);
                list.remove(list.size() - 1);
            }
            return;
        }
        String s = list.get(list.size() - 1);
        for (String neighbor : adj126.getOrDefault(count126 - list.size() - 1, new ArrayList<>())) {
            if (check126(s, neighbor) && !set.contains(neighbor)) {
                set.add(neighbor);
                list.add(neighbor);
                dfs126(list, set);
                set.remove(neighbor);
                list.remove(list.size() - 1);
            }

        }
    }

    private boolean check126(String s1, String s2) {
        int n = s1.length();
        int count = 0;
        for (int i = 0; i < n; ++i) {
            if (s1.charAt(i) != s2.charAt(i)) {
                if (++count > 1) {
                    return false;
                }
            }
        }
        return count == 1;
    }

    // 1617. 统计子树中城市之间最大距离 (Count Subtrees With Max Distance Between Cities)
    private Map<Integer, List<Integer>> g1617_1;
    private int n1617_1;
    private boolean[] inSet1617_1;
    private int[] res1617_1;
    private boolean[] vis1617_1;
    private int diameter1617_1;

    public int[] countSubgraphsForEachDiameter(int n, int[][] edges) {
        this.g1617_1 = new HashMap<>();
        this.n1617_1 = n;
        this.inSet1617_1 = new boolean[n];
        this.vis1617_1 = new boolean[n];
        this.res1617_1 = new int[n - 1];
        for (int[] e : edges) {
            int a = e[0] - 1;
            int b = e[1] - 1;
            g1617_1.computeIfAbsent(a, k -> new ArrayList<>()).add(b);
            g1617_1.computeIfAbsent(b, k -> new ArrayList<>()).add(a);
        }
        f1617_1(0);
        return res1617_1;

    }

    private void f1617_1(int i) {
        if (i == n1617_1) {
            Arrays.fill(vis1617_1, false);
            diameter1617_1 = 0;
            for (int j = 0; j < n1617_1; ++j) {
                if (inSet1617_1[j]) {
                    dfs1617_1(j);
                    break;
                }
            }
            if (diameter1617_1 > 0 && Arrays.equals(inSet1617_1, vis1617_1)) {
                ++res1617_1[diameter1617_1 - 1];
            }
            return;
        }
        // 不选
        f1617_1(i + 1);

        // 选
        inSet1617_1[i] = true;
        f1617_1(i + 1);
        inSet1617_1[i] = false;
    }

    private int dfs1617_1(int x) {
        vis1617_1[x] = true;
        int max = 0;
        for (int y : g1617_1.getOrDefault(x, new ArrayList<>())) {
            if (!vis1617_1[y] && inSet1617_1[y]) {
                int cur = dfs1617_1(y) + 1;
                diameter1617_1 = Math.max(diameter1617_1, cur + max);
                max = Math.max(max, cur);
            }
        }
        return max;
    }

    // 1617. 统计子树中城市之间最大距离 (Count Subtrees With Max Distance Between Cities)
    private Map<Integer, List<Integer>> g1617_2;
    private int inSetMask1617_2;
    private int visMask1617_2;
    private int diameter1617_2;
    private int[] res1617_2;

    public int[] countSubgraphsForEachDiameter2(int n, int[][] edges) {
        this.g1617_2 = new HashMap<>();
        for (int[] e : edges) {
            int a = e[0] - 1;
            int b = e[1] - 1;
            g1617_2.computeIfAbsent(a, k -> new ArrayList<>()).add(b);
            g1617_2.computeIfAbsent(b, k -> new ArrayList<>()).add(a);
        }
        this.res1617_2 = new int[n - 1];
        for (int i = 1; i < (1 << n); ++i) {
            inSetMask1617_2 = i;
            if (Integer.bitCount(inSetMask1617_2) == 1) {
                continue;
            }
            visMask1617_2 = 0;
            diameter1617_2 = 0;
            dfs1617_2(Integer.numberOfTrailingZeros(inSetMask1617_2));
            if (diameter1617_2 > 0 && visMask1617_2 == inSetMask1617_2) {
                ++res1617_2[diameter1617_2 - 1];
            }
        }
        return res1617_2;
    }

    private int dfs1617_2(int x) {
        visMask1617_2 |= 1 << x;
        int max = 0;
        for (int y : g1617_2.getOrDefault(x, new ArrayList<>())) {
            if (((inSetMask1617_2 >> y) & 1) == 1 && ((visMask1617_2 >> y) & 1) == 0) {
                int cur = dfs1617_2(y) + 1;
                diameter1617_2 = Math.max(diameter1617_2, max + cur);
                max = Math.max(max, cur);
            }
        }
        return max;
    }

    /**
     * 小明跑D公里的马拉松，总共有H的体力。小明跑步有五种模式，模式的配速越快，消耗体力越多，写一个程序求出跑马拉松最快的时间。
     * 例D=30,H=130
     * 1 2 3 4 5模式
     * 3 4 5 6 7跑每公里花的时间
     * 7 5 4 3 2消耗的体力
     */
    private int[][] memo_example;
    private int D_example;
    private int H_example;
    private int[][] mode_example;

    public int f_example(int D, int H, int[][] mode) {
        // memo[i][j] : 跑i公里且消耗了j体力时，所用的的最少时间
        this.memo_example = new int[D][H];
        this.D_example = D;
        this.H_example = H;
        this.mode_example = mode;
        for (int i = 0; i < D; ++i) {
            Arrays.fill(memo_example[i], -1);
        }
        this.mode_example = mode;
        int res = dfs_example(0, 0);
        return res == (int) 1e9 ? -1 : res;
    }

    private int dfs_example(int d, int h) {
        if (d == D_example || h == H_example) {
            if (d == D_example) {
                return 0;
            }
            return (int) 1e9;
        }
        if (memo_example[d][h] != -1) {
            return memo_example[d][h];
        }
        int res = (int) 1e9;
        for (int i = 4; i >= 0; --i) {
            // 标号越小的模式，消耗的体力越大，当前消耗的体力h + 第i号模式需要的体力时超过的总体力，则消耗更大体力的模式也不能选
            if (h + mode_example[i][1] > H_example) {
                break;
            }
            res = Math.min(res, dfs_example(d + 1, h + mode_example[i][1]) + mode_example[i][0]);
        }
        return memo_example[d][h] = res;
    }

    // 1473. 粉刷房子 III (Paint House III)
    private int[][][] memo1473;
    private int m1473;
    private int n1473;
    private int[] houses1473;
    private int[][] cost1473;

    public int minCost(int[] houses, int[][] cost, int m, int n, int target) {
        // memo[i][j][k] 将[0,i]房子涂色 第i个房子被涂成第j种颜色 且它属于第k个街区的 最小花销
        memo1473 = new int[m][n + 2][target + 1];
        m1473 = m;
        n1473 = n;
        cost1473 = cost;
        houses1473 = houses;
        int res = dfs1473(0, n + 1, target);
        return res == (int) 1e8 ? -1 : res;

    }

    private int dfs1473(int i, int lastColor, int kinds) {
        // kinds == 0 时，不可作为终止条件，
        // 因为如果此时 i < m1473，即还没有遍历完所有的房子，但是剩下的房子有可能都被染成了和上一个房子同样的颜色，无需再新建街区
        if (i == m1473 || kinds < 0 || kinds > m1473 - i) {
            if (i == m1473 && kinds == 0) {
                return 0;
            }
            return (int) 1e8;
        }
        if (memo1473[i][lastColor][kinds] != 0) {
            return memo1473[i][lastColor][kinds];
        }
        int min = (int) 1e8;
        if (houses1473[i] != 0) {
            min = Math.min(min, dfs1473(i + 1, houses1473[i], kinds + (lastColor != houses1473[i] ? -1 : 0)));
            return memo1473[i][lastColor][kinds] = min;
        }
        for (int color = 1; color <= n1473; ++color) {
            min = Math.min(min, cost1473[i][color - 1] + dfs1473(i + 1, color, kinds +
                    (lastColor != color ? -1 : 0)));
        }
        return memo1473[i][lastColor][kinds] = min;
    }

    // 715. Range 模块 (Range Module)
    class RangeModule {
        private TreeMap<Integer, int[]> treeMap;

        public RangeModule() {
            treeMap = new TreeMap<>();
        }

        public void addRange(int left, int right) {
            --right;
            int nLeft = left;
            int nRight = right;
            Map.Entry<Integer, int[]> entry = treeMap.ceilingEntry(left - 1);
            while (entry != null && entry.getValue()[0] - 1 <= right) {
                int[] cur = treeMap.remove(entry.getKey());
                nLeft = Math.min(nLeft, cur[0]);
                nRight = Math.max(nRight, cur[1]);
                entry = treeMap.ceilingEntry(left - 1);
            }
            treeMap.put(nRight, new int[] { nLeft, nRight });
        }

        public boolean queryRange(int left, int right) {
            --right;
            Map.Entry<Integer, int[]> entry = treeMap.ceilingEntry(right);
            return entry != null && entry.getValue()[0] <= left;
        }

        public void removeRange(int left, int right) {
            --right;
            Map.Entry<Integer, int[]> entry = treeMap.ceilingEntry(left);
            if (entry == null || entry.getValue()[0] > right) {
                return;
            }
            if (entry.getValue()[0] < left && right < entry.getValue()[1]) {
                int[] removed = treeMap.remove(entry.getKey());
                treeMap.put(left - 1, new int[] { removed[0], left - 1 });
                treeMap.put(removed[1], new int[] { right + 1, removed[1] });
                return;
            }
            if (entry.getValue()[1] <= right && entry.getValue()[0] < left) {
                int[] removed = treeMap.remove(entry.getKey());
                treeMap.put(left - 1, new int[] { removed[0], left - 1 });
                left = removed[1] + 1;
            }
            entry = treeMap.ceilingEntry(right);
            if (entry != null && entry.getValue()[0] <= right && entry.getValue()[0] >= left) {
                int[] removed = treeMap.remove(entry.getKey());
                treeMap.put(removed[1], new int[] { right + 1, removed[1] });
                right = removed[0] - 1;
            }
            entry = treeMap.ceilingEntry(left);
            while (left <= right && entry != null && entry.getValue()[1] <= right && entry.getValue()[0] >= left) {
                treeMap.remove(entry.getKey());
                entry = treeMap.ceilingEntry(left);
            }
        }
    }

    // 327. 区间和的个数 (Count of Range Sum) -- 还需实现 归并排序/线段树/动态增加节点的线段树/树状数组/平衡二叉搜索树
    public int countRangeSum(int[] nums, int lower, int upper) {
        List<Long> list = new ArrayList<>();
        list.add(0L);
        long pre = 0L;
        int res = 0;
        for (int num : nums) {
            pre += num;
            // 找 list中 <= pre - lower 的最大值的索引 不存在则为 -1
            int r = upperBound327(list, pre - lower);
            // 找 list中 >= pre - upper 的最小值的索引 不存在则为 -1
            int l = lowerBound327(list, pre - upper);
            if (r != -1 && l != -1) {
                res += r - l + 1;
            }
            insert327(list, pre);
        }
        return res;
    }

    private void insert327(List<Long> list, long target) {
        int n = list.size();
        if (target <= list.get(0)) {
            list.add(0, target);
            return;
        }
        if (target >= list.get(n - 1)) {
            list.add(target);
            return;
        }
        int left = 0;
        int right = n - 1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (list.get(mid) < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        list.add(left, target);
    }

    private int lowerBound327(List<Long> list, long target) {
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

    private int upperBound327(List<Long> list, long target) {
        int n = list.size();
        if (list.get(n - 1) <= target) {
            return n - 1;
        }
        if (target < list.get(0)) {
            return -1;
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

    // LCP 72. 补给马车
    public int[] supplyWagon(int[] supplies) {
        int n = supplies.length;
        List<Integer> list = Arrays.stream(supplies).boxed().collect(Collectors.toList());
        while (list.size() != n / 2) {
            int min = Integer.MAX_VALUE;
            int index = -1;
            for (int i = 0; i < list.size() - 1; ++i) {
                int curSum = list.get(i) + list.get(i + 1);
                if (curSum < min) {
                    min = curSum;
                    index = i;
                }
            }
            list.set(index, min);
            list.remove(index + 1);
        }
        return list.stream().mapToInt(Integer::intValue).toArray();

    }

    // LCP 73. 探险营地
    public int adventureCamp(String[] expeditions) {
        Set<String> set = new HashSet<>();
        int res = -1;
        int max = 0;
        for (int i = 0; i < expeditions.length; ++i) {
            if (expeditions[i].isEmpty()) {
                continue;
            }
            String[] split = expeditions[i].split("->");
            int cur = 0;
            for (int j = 0; j < split.length; ++j) {
                if (set.add(split[j])) {
                    ++cur;
                }
            }
            if (i != 0 && cur > max) {
                max = cur;
                res = i;
            }
        }
        return res;

    }

    // 6387. 计算列车到站时间 (Calculate Delayed Arrival Time)
    public int findDelayedArrivalTime(int arrivalTime, int delayedTime) {
        return (arrivalTime + delayedTime) % 24;
    }

    // 6391. 倍数求和
    public int sumOfMultiples(int n) {
        int res = 0;
        for (int i = 1; i <= n; ++i) {
            if (i % 3 == 0 || i % 5 == 0 || i % 7 == 0) {
                res += i;
            }
        }
        return res;

    }

    // 6391. 倍数求和
    public int sumOfMultiples2(int n) {
        return sum6391(n, 3) + sum6391(n, 5) + sum6391(n, 7) - sum6391(n, 15) - sum6391(n, 35) - sum6391(n, 21)
                + sum6391(n, 105);
    }

    private int sum6391(int n, int m) {
        return (1 + n / m) * (n / m) / 2 * m;
    }

    // 6390. 滑动子数组的美丽值
    public int[] getSubarrayBeauty(int[] nums, int k, int x) {
        int n = nums.length;
        int[] res = new int[n - k + 1];
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < k; ++i) {
            list.add(nums[i]);
        }
        Collections.sort(list);
        res[0] = Math.min(0, list.get(x - 1));
        for (int i = k; i < n; ++i) {
            insert6390(list, nums[i]);
            remove6390(list, nums[i - k]);
            res[i - k + 1] = Math.min(0, list.get(x - 1));
        }
        return res;

    }

    private void remove6390(List<Integer> list, int target) {
        int left = 0;
        int right = list.size() - 1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (list.get(mid) == target) {
                list.remove(mid);
                return;
            } else if (list.get(mid) < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }

    private void insert6390(List<Integer> list, int target) {
        int n = list.size();
        if (target <= list.get(0)) {
            list.add(0, target);
            return;
        }
        if (target >= list.get(n - 1)) {
            list.add(target);
            return;
        }
        int left = 0;
        int right = n - 1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (list.get(mid) < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        list.add(left, target);
    }

    // 6390. 滑动子数组的美丽值
    public int[] getSubarrayBeauty2(int[] nums, int k, int x) {
        int n = nums.length;
        int[] res = new int[n - k + 1];
        int[] counts = new int[101];
        for (int i = 0; i < k; ++i) {
            ++counts[nums[i] + 50];
        }
        int y = 0;
        for (int i = 0; i <= 100; ++i) {
            y += counts[i];
            if (y >= x) {
                res[0] = Math.min(0, i - 50);
                break;
            }
        }
        y = 0;
        for (int i = k; i < n; ++i) {
            --counts[nums[i - k] + 50];
            ++counts[nums[i] + 50];
            y = 0;
            for (int t = 0; t <= 100; ++t) {
                y += counts[t];
                if (y >= x) {
                    res[i - k + 1] = Math.min(0, t - 50);
                    break;
                }
            }
        }
        return res;

    }

    // 6392. 使数组所有元素变成 1 的最少操作次数
    public int minOperations(int[] nums) {
        int n = nums.length;
        int gcd = 0;
        int cnt1 = 0;
        for (int num : nums) {
            gcd = getGCD6392(gcd, num);
            if (num == 1) {
                ++cnt1;
            }
        }
        if (gcd > 1) {
            return -1;
        }
        if (cnt1 > 0) {
            return n - cnt1;
        }
        int min = n;
        for (int i = 0; i < n; ++i) {
            gcd = 0;
            for (int j = i; j < n; ++j) {
                gcd = getGCD6392(gcd, nums[j]);
                if (gcd == 1) {
                    min = Math.min(min, j - i + 1);
                }
            }
        }
        return min - 1 + n - 1;

    }

    private int getGCD6392(int a, int b) {
        return b == 0 ? a : getGCD6392(b, a % b);
    }

    // 174. 地下城游戏 (Dungeon Game)
    private int m174;
    private int n174;
    private int[][] dungeon174;
    private int[][] memo174;

    public int calculateMinimumHP(int[][] dungeon) {
        this.m174 = dungeon.length;
        this.n174 = dungeon[0].length;
        this.dungeon174 = dungeon;
        this.memo174 = new int[m174][n174];
        return dfs174(0, 0);
    }

    private int dfs174(int i, int j) {
        if (i == m174 - 1 && j == n174 - 1) {
            return Math.max(1 - dungeon174[i][j], 1);
        }
        if (memo174[i][j] != 0) {
            return memo174[i][j];
        }
        if (i == m174 - 1) {
            return memo174[i][j] = Math.max(dfs174(i, j + 1) - dungeon174[i][j], 1);
        }
        if (j == n174 - 1) {
            return memo174[i][j] = Math.max(dfs174(i + 1, j) - dungeon174[i][j], 1);
        }
        return memo174[i][j] = Math.max(Math.min(dfs174(i + 1, j), dfs174(i, j + 1)) - dungeon174[i][j], 1);
    }

    // 1889. 装包裹的最小浪费空间 (Minimum Space Wasted From Packaging)
    public int minWastedSpace(int[] packages, int[][] boxes) {
        Arrays.sort(packages);
        int n = packages.length;
        long[] pre = new long[n + 1];
        for (int i = 0; i < n; ++i) {
            pre[i + 1] = pre[i] + packages[i];
        }
        long res = Long.MAX_VALUE;
        search: for (int[] box : boxes) {
            Arrays.sort(box);
            int m = box.length;
            if (packages[n - 1] > box[m - 1]) {
                continue;
            }
            long cur = 0L;
            int i = 0;
            int j = 0;
            while (i < n) {
                int index = binarySearch1839(packages, i, box[j]);
                if (index == -1) {
                    ++j;
                    continue;
                }
                cur += (long) box[j] * (index - i + 1) - (pre[index + 1] - pre[i]);
                if (cur >= res) {
                    continue search;
                }
                ++j;
                i = index + 1;
            }
            res = Math.min(res, cur);
        }
        final int MOD = (int) (1e9 + 7);
        return (int) (res == Long.MAX_VALUE ? -1 : res % MOD);

    }

    // 在排序数组packages的i ～ n - 1的索引中 找 <= target的最大值对应的索引 若不存在 返回 -1
    private int binarySearch1839(int[] packages, int i, int target) {
        int n = packages.length;
        if (target < packages[i]) {
            return -1;
        }
        if (packages[n - 1] <= target) {
            return n - 1;
        }
        int left = i;
        int right = n - 1;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (packages[mid] <= target) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    // 1353. 最多可以参加的会议数目 (Maximum Number of Events That Can Be Attended)
    public int maxEvents(int[][] events) {
        int n = events.length;
        Queue<Integer> q = new PriorityQueue<>();

        Arrays.sort(events, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });
        int res = 0;
        int curDay = 1;
        int i = 0;
        while (i < n || !q.isEmpty()) {
            while (i < n && events[i][0] == curDay) {
                q.offer(events[i][1]);
                ++i;
            }

            while (!q.isEmpty() && q.peek() < curDay) {
                q.poll();
            }

            if (!q.isEmpty()) {
                q.poll();
                ++res;
            }
            ++curDay;
        }
        return res;

    }

    // 2402. 会议室 III (Meeting Rooms III)
    public int mostBooked(int n, int[][] meetings) {
        int[] times = new int[n];
        // 空闲会议室
        Queue<Integer> idle = new PriorityQueue<>();
        for (int i = 0; i < n; ++i) {
            idle.offer(i);
        }
        // 非空闲会议室 [i, j] : 编号为i的会议室被占用 结束时间为j
        Queue<long[]> wait = new PriorityQueue<>(new Comparator<long[]>() {

            @Override
            public int compare(long[] o1, long[] o2) {
                if (o1[1] == o2[1]) {
                    return Long.compare(o1[0], o2[0]);
                }
                return Long.compare(o1[1], o2[1]);
            }

        });

        Arrays.sort(meetings, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });
        for (int[] meeting : meetings) {
            while (!wait.isEmpty() && wait.peek()[1] <= (long) meeting[0]) {
                int id = (int) wait.poll()[0];
                idle.offer(id);
            }
            int id;
            if (!idle.isEmpty()) {
                id = idle.poll();
                wait.offer(new long[] { id, meeting[1] });
            } else {
                long[] cur = wait.poll();
                id = (int) cur[0];
                long end = cur[1];
                wait.offer(new long[] { id, end + meeting[1] - meeting[0] });
            }
            ++times[id];
        }

        int max = 0;
        int res = 0;
        for (int t = 0; t < times.length; ++t) {
            if (max < times[t]) {
                max = times[t];
                res = t;
            }
        }
        return res;
    }

    // 1751. 最多可以参加的会议数目 II (Maximum Number of Events That Can Be Attended II)
    private int n1751;
    private int k1751;
    private int[][] memo1751;
    private int[][] events1751;

    public int maxValue(int[][] events, int k) {
        Arrays.sort(events, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });
        this.events1751 = events;
        this.n1751 = events.length;
        this.k1751 = k;
        this.memo1751 = new int[n1751][k];
        return dfs1751(0, 0);

    }

    private int dfs1751(int i, int count) {
        if (i == n1751 || count == k1751) {
            return 0;
        }
        if (memo1751[i][count] != 0) {
            return memo1751[i][count];
        }
        int right = binarySearch1751(events1751, events1751[i][1]);
        return memo1751[i][count] = Math.max(dfs1751(i + 1, count), dfs1751(right, count + 1) + events1751[i][2]);
    }

    // 在「按照开始时间从小到大排序」的e数组中，找 >target 的开始时间中，最小值对应的索引，不存在返回n
    private int binarySearch1751(int[][] e, int target) {
        int n = e.length;
        if (target < e[0][0]) {
            return 0;
        }
        if (e[n - 1][0] <= target) {
            return n;
        }
        int left = 0;
        int right = n - 1;
        int res = n;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (e[mid][0] > target) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    // 6406. K 个元素的最大和 (Maximum Sum With Exactly K Elements)
    public int maximizeSum(int[] nums, int k) {
        int max = Arrays.stream(nums).max().getAsInt();
        return (2 * max + k - 1) * k / 2;

    }

    // 6405. 找到两个数组的前缀公共数组 (Find the Prefix Common Array of Two Arrays)
    public int[] findThePrefixCommonArray(int[] A, int[] B) {
        int n = A.length;
        int[] res = new int[n];
        long maskA = 0L;
        long maskB = 0L;
        for (int i = 0; i < n; ++i) {
            maskA |= 1L << A[i];
            maskB |= 1L << B[i];
            res[i] = Long.bitCount(maskA & maskB);
        }
        return res;

    }

    // 6403. 网格图中鱼的最大数目 (Maximum Number of Fish in a Grid)
    private int[][] grid6403;
    private int m6403;
    private int n6403;

    public int findMaxFish(int[][] grid) {
        this.m6403 = grid.length;
        this.n6403 = grid[0].length;
        this.grid6403 = grid;
        int res = 0;
        for (int i = 0; i < m6403; ++i) {
            for (int j = 0; j < n6403; ++j) {
                if (grid[i][j] != 0) {
                    res = Math.max(res, dfs6403(i, j));
                }
            }
        }
        return res;

    }

    private int dfs6403(int i, int j) {
        if (!(i >= 0 && i < m6403 && j >= 0 && j < n6403)) {
            return 0;
        }
        if (grid6403[i][j] == 0) {
            return 0;
        }
        int res = grid6403[i][j];
        grid6403[i][j] = 0;
        res += dfs6403(i + 1, j);
        res += dfs6403(i, j + 1);
        res += dfs6403(i - 1, j);
        res += dfs6403(i, j - 1);
        return res;
    }

    // 6341. 保龄球游戏的获胜者 (Determine the Winner of a Bowling Game)
    public int isWinner(int[] player1, int[] player2) {
        int n = player1.length;
        int sum1 = 0;
        int sum2 = 0;
        for (int i = 0; i < n; ++i) {
            if (i - 1 >= 0 && player1[i - 1] == 10 || i - 2 >= 0 && player1[i - 2] == 10) {
                sum1 += player1[i] * 2;
            } else {
                sum1 += player1[i];
            }
            if (i - 1 >= 0 && player2[i - 1] == 10 || i - 2 >= 0 && player2[i - 2] == 10) {
                sum2 += player2[i] * 2;
            } else {
                sum2 += player2[i];
            }
        }
        if (sum1 > sum2) {
            return 1;
        }
        if (sum1 < sum2) {
            return 2;
        }
        return 0;

    }

    // 6342. 找出叠涂元素 (First Completely Painted Row or Column)
    public int firstCompleteIndex(int[] arr, int[][] mat) {
        int m = mat.length;
        int n = mat[0].length;
        Map<Integer, int[]> map = new HashMap<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                map.put(mat[i][j], new int[] { i, j });
            }
        }
        int[] row = new int[m];
        int[] col = new int[n];
        for (int i = 0; i < m * n; ++i) {
            int[] cur = map.get(arr[i]);
            int x = cur[0];
            int y = cur[1];
            ++row[x];
            ++col[y];
            if (row[x] == n || col[y] == m) {
                return i;
            }
        }
        return -1;

    }

    // 2088. 统计农场中肥沃金字塔的数目 (Count Fertile Pyramids in a Land)
    public int countPyramids(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        if (m == 1 || n == 1) {
            return 0;
        }
        int res = 0;
        res += getPyramids(grid);
        reverse2088(grid);
        res += getPyramids(grid);
        return res;

    }

    private void reverse2088(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        for (int i = 0; i < m / 2; ++i) {
            for (int j = 0; j < n; ++j) {
                int temp = grid[i][j];
                grid[i][j] = grid[m - i - 1][j];
                grid[m - i - 1][j] = temp;
            }
        }
    }

    private int getPyramids(int[][] grid) {
        int res = 0;
        int m = grid.length;
        int n = grid[0].length;
        // dp[i][j] : 以[i，j] 为顶点的金字塔的最大高度
        int[][] dp = new int[m][n];
        dp[m - 1] = grid[m - 1];
        for (int i = m - 2; i >= 0; --i) {
            // 左右两列特判
            dp[i][0] = grid[i][0];
            dp[i][n - 1] = grid[i][n - 1];
            for (int j = 1; j < n - 1; ++j) {
                if (grid[i][j] == 0) {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = Math.min(dp[i + 1][j], Math.min(dp[i + 1][j + 1], dp[i + 1][j - 1])) + 1;
                    res += dp[i][j] - 1;
                }
            }
        }
        return res;
    }

    // 1671. 得到山形数组的最少删除次数 (Minimum Number of Removals to Make Mountain Array)
    // --O(n^2)
    public int minimumMountainRemovals(int[] nums) {
        int n = nums.length;
        int[] pre = new int[n];
        for (int i = 0; i < n; ++i) {
            pre[i] = 1;
            for (int j = 0; j < i; ++j) {
                if (nums[i] > nums[j]) {
                    pre[i] = Math.max(pre[i], pre[j] + 1);
                }
            }
        }
        int[] suf = new int[n];
        for (int i = n - 1; i >= 0; --i) {
            suf[i] = 1;
            for (int j = n - 1; j > i; --j) {
                if (nums[i] > nums[j]) {
                    suf[i] = Math.max(suf[i], suf[j] + 1);
                }
            }
        }
        int res = n;
        for (int i = 0; i < n; ++i) {
            // 山峰点的前提是左右两边都要有单调的非空子序列存在
            if (suf[i] != 1 && pre[i] != 1) {
                res = Math.min(res, n - pre[i] - suf[i] + 1);
            }

        }
        return res;
    }

    // 1671. 得到山形数组的最少删除次数 (Minimum Number of Removals to Make Mountain Array)
    // --O(n^2)
    private int[] pre1671;
    private int[] suf1671;
    private int[] nums1671;
    private int n1671;

    public int minimumMountainRemovals2(int[] nums) {
        this.n1671 = nums.length;
        this.pre1671 = new int[n1671];
        this.nums1671 = nums;
        for (int i = 0; i < n1671; ++i) {
            dfs1671(i);
        }
        this.suf1671 = new int[n1671];
        for (int i = n1671 - 1; i >= 0; --i) {
            dfs2_1671(i);
        }
        int res = n1671;
        for (int i = 0; i < n1671; ++i) {
            if (pre1671[i] != 1 && suf1671[i] != 1) {
                res = Math.min(res, n1671 - pre1671[i] - suf1671[i] + 1);
            }
        }
        return res;

    }

    private int dfs2_1671(int i) {
        if (suf1671[i] != 0) {
            return suf1671[i];
        }
        int max = 0;
        for (int j = i + 1; j < n1671; ++j) {
            if (nums1671[j] < nums1671[i]) {
                max = Math.max(max, dfs2_1671(j));
            }
        }
        return suf1671[i] = max + 1;
    }

    private int dfs1671(int i) {
        if (pre1671[i] != 0) {
            return pre1671[i];
        }
        int max = 0;
        for (int j = 0; j < i; ++j) {
            if (nums1671[j] < nums1671[i]) {
                max = Math.max(max, dfs1671(j));
            }
        }
        return pre1671[i] = max + 1;
    }

    // 1671. 得到山形数组的最少删除次数 (Minimum Number of Removals to Make Mountain Array)
    // --O(nlogn)
    public int minimumMountainRemovals3(int[] nums) {
        int n = nums.length;
        int[] pre = new int[n];
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            if (list.isEmpty() || nums[i] > list.get(list.size() - 1)) {
                list.add(nums[i]);
                pre[i] = list.size();
            } else {
                int j = binarySearchCeiling1671(list, nums[i]);
                list.set(j, nums[i]);
                pre[i] = j + 1;
            }
        }

        list.clear();
        int[] suf = new int[n];
        for (int i = n - 1; i >= 0; --i) {
            if (list.isEmpty() || nums[i] > list.get(list.size() - 1)) {
                list.add(nums[i]);
                suf[i] = list.size();
            } else {
                int j = binarySearchCeiling1671(list, nums[i]);
                list.set(j, nums[i]);
                suf[i] = j + 1;
            }
        }
        int res = n;
        for (int i = 0; i < n; ++i) {
            if (pre[i] != 1 && suf[i] != 1) {
                res = Math.min(res, n - pre[i] - suf[i] + 1);
            }
        }
        return res;

    }

    // 找排序list中，第一个 >= target 的值的索引
    private int binarySearchCeiling1671(List<Integer> list, int target) {
        int left = 0;
        int right = list.size() - 1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (list.get(mid) >= target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return right + 1;
    }

    // 354. 俄罗斯套娃信封问题 (Russian Doll Envelopes)
    public int maxEnvelopes(int[][] envelopes) {
        int n = envelopes.length;
        Arrays.sort(envelopes, new Comparator<int[]>() {

            // 优先级 ：按第 0 列升序，若相等，按第 1 列降序
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] == o2[0]) {
                    // 降序
                    return Integer.compare(o2[1], o1[1]);
                }
                return Integer.compare(o1[0], o2[0]);

            }

        });

        List<Integer> list = new ArrayList<>();
        list.add(envelopes[0][1]);
        for (int i = 1; i < n; ++i) {
            int num = envelopes[i][1];
            if (num > list.get(list.size() - 1)) {
                list.add(num);
            } else {
                int j = binarySearch354(list, num);
                list.set(j, num);
            }
        }
        return list.size();
    }

    // 找排序list中，第一个 >= target 的值的索引
    private int binarySearch354(List<Integer> list, int target) {
        int left = 0;
        int right = list.size() - 1;
        int res = 0;
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

    // 面试题 08.13. 堆箱子 (Pile Box LCCI)
    private int n08_13;
    private int[][] box08_13;
    private int[] memo08_13;

    public int pileBox(int[][] box) {
        Arrays.sort(box, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] == o2[0]) {
                    if (o1[1] == o2[1]) {
                        return Integer.compare(o1[2], o2[2]);
                    }
                    return Integer.compare(o1[1], o2[1]);
                }
                return Integer.compare(o1[0], o2[0]);
            }

        });
        this.n08_13 = box.length;
        this.box08_13 = box;
        this.memo08_13 = new int[n08_13];
        int res = 0;
        for (int i = 0; i < n08_13; ++i) {
            res = Math.max(res, dfs08_13(i));
        }
        return res;

    }

    private int dfs08_13(int i) {
        if (memo08_13[i] != 0) {
            return memo08_13[i];
        }
        int max = 0;
        for (int j = i - 1; j >= 0; --j) {
            if (box08_13[j][0] < box08_13[i][0] && box08_13[j][1] < box08_13[i][1] && box08_13[j][2] < box08_13[i][2]) {
                max = Math.max(max, dfs08_13(j));
            }
        }
        return memo08_13[i] = max + box08_13[i][2];
    }

    // 面试题 17.08. 马戏团人塔 (Circus Tower LCCI)
    public int bestSeqAtIndex(int[] height, int[] weight) {
        int n = height.length;
        Integer[] ids = IntStream.range(0, n).boxed().toArray(Integer[]::new);
        Arrays.sort(ids, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                if (height[o1] == height[o2]) {
                    return Integer.compare(weight[o2], weight[o1]);
                }
                return Integer.compare(height[o1], height[o2]);
            }

        });

        List<Integer> list = new ArrayList<>();
        for (int id : ids) {
            int num = weight[id];
            if (list.isEmpty() || num > list.get(list.size() - 1)) {
                list.add(num);
            } else {
                int j = binarySearch17_08(list, num);
                list.set(j, num);
            }
        }
        return list.size();

    }

    // 找排序list中，第一个 >= target 的值的索引
    private int binarySearch17_08(List<Integer> list, int target) {
        int left = 0;
        int right = list.size() - 1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (list.get(mid) >= target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return right + 1;
    }

    // 87. 扰乱字符串 (Scramble String)
    private Map<String, Boolean> memo_87_1;
    private int n_87_1;
    private String s2_87_1;

    public boolean isScramble(String s1, String s2) {
        this.memo_87_1 = new HashMap<>();
        this.n_87_1 = s1.length();
        this.s2_87_1 = s2;
        return dfs_87_1(s1, 0, n_87_1);

    }

    private boolean dfs_87_1(String s, int left, int right) {
        String p = s + "_" + left + "_" + right;
        if (memo_87_1.containsKey(p)) {
            return memo_87_1.get(p);
        }
        if (s.equals(s2_87_1.substring(left, right))) {
            return true;
        }
        if (!check_87_1(s, s2_87_1.substring(left, right))) {
            memo_87_1.put(p, false);
            return false;
        }
        for (int i = left + 1; i < right; ++i) {
            String sub1 = s.substring(left - left, i - left);
            String sub2 = s.substring(i - left, right - left);
            if (dfs_87_1(sub1, left, i) && dfs_87_1(sub2, i, right)) {
                memo_87_1.put(p, true);
                return true;
            }
            if (dfs_87_1(sub2, left, left + sub2.length())
                    && dfs_87_1(sub1, right - sub1.length(), right)) {
                memo_87_1.put(p, true);
                return true;
            }
        }
        memo_87_1.put(p, false);
        return false;
    }

    private boolean check_87_1(String a, String b) {
        int[] cnt = new int[26];
        for (char c : a.toCharArray()) {
            ++cnt[c - 'a'];
        }
        for (char c : b.toCharArray()) {
            --cnt[c - 'a'];
        }
        for (int i = 0; i < 26; ++i) {
            if (cnt[i] != 0) {
                return false;
            }
        }
        return true;
    }

    // 87. 扰乱字符串 (Scramble String)
    private int n_87_2;
    private String s1_87_2;
    private String s2_87_2;
    private int[][][] memo_87_2;

    public boolean isScramble2(String s1, String s2) {
        this.n_87_2 = s1.length();
        this.s1_87_2 = s1;
        this.s2_87_2 = s2;
        memo_87_2 = new int[n_87_2][n_87_2][n_87_2 + 1];
        return dfs_87_2(0, 0, n_87_2);

    }

    private boolean dfs_87_2(int i1, int i2, int len) {
        if (memo_87_2[i1][i2][len] != 0) {
            return memo_87_2[i1][i2][len] > 0;
        }
        if (s1_87_2.substring(i1, i1 + len).equals(s2_87_2.substring(i2, i2 + len))) {
            memo_87_2[i1][i2][len] = 1;
            return true;
        }
        if (!check_87_2(s1_87_2.substring(i1, i1 + len), s2_87_2.substring(i2, i2 + len))) {
            memo_87_2[i1][i2][len] = -1;
            return false;
        }
        for (int l = 1; l < len; ++l) {
            if (dfs_87_2(i1, i2, l) && dfs_87_2(i1 + l, i2 + l, len - l)) {
                memo_87_2[i1][i2][len] = 1;
                return true;
            }
            if (dfs_87_2(i1 + l, i2, len - l) && dfs_87_2(i1, i2 + len - l, l)) {
                memo_87_2[i1][i2][len] = 1;
                return true;
            }
        }
        memo_87_2[i1][i2][len] = -1;
        return false;
    }

    private boolean check_87_2(String a, String b) {
        int[] cnts = new int[26];
        for (char c : a.toCharArray()) {
            ++cnts[c - 'a'];
        }
        for (char c : b.toCharArray()) {
            --cnts[c - 'a'];
        }
        for (int c : cnts) {
            if (c != 0) {
                return false;
            }
        }
        return true;
    }

    // 1866. 恰有 K 根木棍可以看到的排列数目 (Number of Ways to Rearrange Sticks With K Sticks
    // Visible)
    private int[][] memo1866;

    public int rearrangeSticks(int n, int k) {
        this.memo1866 = new int[n + 1][k + 1];
        for (int i = 0; i < n + 1; ++i) {
            Arrays.fill(memo1866[i], -1);
        }
        return dfs1866(n, k);

    }

    private int dfs1866(int n, int k) {
        if (n == 0 || k == 0 || k >= n) {
            return k == n ? 1 : 0;
        }
        if (memo1866[n][k] != -1) {
            return memo1866[n][k];
        }
        final int MOD = (int) (1e9 + 7);
        return memo1866[n][k] = (int) (dfs1866(n - 1, k - 1) + ((long) (n - 1) * dfs1866(n - 1, k)) % MOD) % MOD;
    }

    // 44. 通配符匹配 (Wildcard Matching) -- 还需掌握贪心算法
    private int[][] memo44;
    private int sLen44;
    private int pLen44;
    private char[] sChars44;
    private char[] pChars44;
    // suf[i] ： 以p[i]为开头的字符串中「*」的个数
    private int[] suf44;

    public boolean isMatch(String s, String p) {
        this.sLen44 = s.length();
        this.pLen44 = p.length();
        this.memo44 = new int[sLen44][pLen44];
        this.sChars44 = s.toCharArray();
        this.pChars44 = p.toCharArray();
        this.suf44 = new int[pLen44];
        for (int i = pLen44 - 1; i >= 0; --i) {
            if (i < pLen44 - 1) {
                suf44[i] = suf44[i + 1] + (pChars44[i] == '*' ? 1 : 0);
            } else {
                suf44[i] = pChars44[i] == '*' ? 1 : 0;
            }
        }
        return dfs44(0, 0);

    }

    private boolean dfs44(int i, int j) {
        // s到头了 而且（p也到头了 或者 剩下的p都是「*」）
        if (i == sLen44 || j == pLen44) {
            return i == sLen44 && (j == pLen44 || suf44[j] == pLen44 - j);
        }
        // s和p都没到头 但是 剩下的p都是「*」 返回true
        if (suf44[j] == pLen44 - j) {
            memo44[i][j] = 1;
            return true;
        }
        if (memo44[i][j] != 0) {
            return memo44[i][j] > 0;
        }
        // 剩下的p没有「*」了 ，剩下的s和p长度不等 返回false
        if (suf44[j] == 0 && sLen44 - i != pLen44 - j) {
            memo44[i][j] = -1;
            return false;
        }
        if (Character.isLetter(pChars44[j])) {
            // p当前位是字母 s当前位和p当前位不等
            if (sChars44[i] != pChars44[j]) {
                memo44[i][j] = -1;
                return false;
            }
            // p当前位是字母 s当前位和p当前位相等
            boolean res = dfs44(i + 1, j + 1);
            memo44[i][j] = res ? 1 : -1;
            return res;
        }
        // p当前位是「?」
        if (pChars44[j] == '?') {
            boolean res = dfs44(i + 1, j + 1);
            memo44[i][j] = res ? 1 : -1;
            return res;
        }
        // p当前位是「*」，可以匹配p中的0个字符、1个字符、2个字符。。。。
        for (int index = i; index <= sLen44; ++index) {
            if (dfs44(index, j + 1)) {
                memo44[i][j] = 1;
                return true;
            }
        }
        memo44[i][j] = -1;
        return false;
    }

    // 963. 最小面积矩形 II (Minimum Area Rectangle II)
    public double minAreaFreeRect(int[][] points) {
        int n = points.length;
        Point[] P = new Point[n];
        double res = Double.MAX_VALUE;
        Set<Point> set = new HashSet<>();
        for (int i = 0; i < n; ++i) {
            P[i] = new Point(points[i][0], points[i][1]);
            set.add(P[i]);
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    if (i != j && j != k && i != k) {
                        Point p1 = P[i];
                        Point p2 = P[j];
                        Point p3 = P[k];
                        Point p4 = new Point(p2.x + p3.x - p1.x, p2.y + p3.y - p1.y);
                        if (set.contains(p4)) {
                            if ((p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) == 0) {
                                double cur = p1.distance(p2) * p1.distance(p3);
                                if (cur < res) {
                                    res = cur;
                                }
                            }
                        }

                    }
                }
            }
        }
        return res == Double.MAX_VALUE ? 0 : res;

    }

    // 629. K个逆序对数组 (K Inverse Pairs Array)
    private int[][] memo629;

    public int kInversePairs(int n, int k) {
        this.memo629 = new int[n + 1][k + 1];
        for (int i = 0; i < n + 1; ++i) {
            Arrays.fill(memo629[i], -1);
        }
        return dfs629(n, k);

    }

    private int dfs629(int n, int k) {
        if (n == 1 || k == 0) {
            return k == 0 ? 1 : 0;
        }
        if (k > n * (n - 1) / 2) {
            return 0;
        }
        if (memo629[n][k] != -1) {
            return memo629[n][k];
        }
        final int MOD = (int) (1e9 + 7);
        int res = 0;
        // 没理解
        for (int j = Math.max(0, k - n + 1); j <= k; ++j) {
            res = (res + dfs629(n - 1, j)) % MOD;
        }
        return memo629[n][k] = res;
    }

    // 1483. 树节点的第 K 个祖先 (Kth Ancestor of a Tree Node)
    class TreeAncestor {
        private Map<Integer, List<Integer>> g;
        // 时间戳
        private int times;
        // key : level
        // val : int[] { timeStamp, x }
        private Map<Integer, List<int[]>> levelNodes;
        // key : x
        // val : int[] { level, timeStamp}
        private Map<Integer, int[]> nodeToLevel;

        public TreeAncestor(int n, int[] parent) {
            this.nodeToLevel = new HashMap<>();
            g = new HashMap<>();
            levelNodes = new HashMap<>();
            for (int i = 0; i < n; ++i) {
                g.computeIfAbsent(parent[i], k -> new ArrayList<>()).add(i);
            }
            dfs(0, 0);

        }

        private void dfs(int x, int level) {
            nodeToLevel.put(x, new int[] { level, times });
            levelNodes.computeIfAbsent(level, k -> new ArrayList<>()).add(new int[] { times++, x });
            for (int y : g.getOrDefault(x, new ArrayList<>())) {
                dfs(y, level + 1);
            }
        }

        public int getKthAncestor(int node, int k) {
            int[] cur = nodeToLevel.get(node);
            int level = cur[0];
            int timeStamp = cur[1];
            List<int[]> kLevelNodes = levelNodes.getOrDefault(level - k, new ArrayList<>());
            return binarySearchFloor(kLevelNodes, timeStamp);
        }

        private int binarySearchFloor(List<int[]> list, int target) {
            int left = 0;
            int right = list.size() - 1;
            int res = -1;
            while (left <= right) {
                int mid = left + ((right - left) >> 1);
                if (list.get(mid)[0] < target) {
                    res = list.get(mid)[1];
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            return res;
        }
    }

    // 6416. 找出不同元素数目差数组 (Find the Distinct Difference Array)
    public int[] distinctDifferenceArray(int[] nums) {
        int[] cnts = new int[51];
        int suf = 0;
        for (int num : nums) {
            if (++cnts[num] == 1) {
                ++suf;
            }
        }
        int[] curCnts = new int[51];
        int pre = 0;
        int n = nums.length;
        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            if (++curCnts[nums[i]] == 1) {
                ++pre;
            }
            if (--cnts[nums[i]] == 0) {
                --suf;
            }
            res[i] = pre - suf;
        }
        return res;
    }

    // 6417. 频率跟踪器 (Frequency Tracker)
    class FrequencyTracker {
        private Map<Integer, Integer> map;
        private Map<Integer, Set<Integer>> freq;

        public FrequencyTracker() {
            map = new HashMap<>();
            freq = new HashMap<>();

        }

        public void add(int number) {
            map.merge(number, 1, Integer::sum);
            int count = map.get(number) - 1;
            Set<Integer> set = freq.getOrDefault(count, new HashSet<>());
            set.remove(number);
            freq.computeIfAbsent(count + 1, k -> new HashSet<>()).add(number);
        }

        public void deleteOne(int number) {
            int count = map.getOrDefault(number, 0);
            if (count == 0) {
                return;
            }
            map.merge(number, -1, Integer::sum);
            Set<Integer> ori = freq.getOrDefault(count, new HashSet<>());
            ori.remove(number);
            freq.computeIfAbsent(count - 1, k -> new HashSet<>()).add(number);
        }

        public boolean hasFrequency(int frequency) {
            return !freq.getOrDefault(frequency, new HashSet<>()).isEmpty();
        }
    }

    // 6418. 有相同颜色的相邻元素数目 (Number of Adjacent Elements With the Same Color)
    public int[] colorTheArray(int n, int[][] queries) {
        int[] colors = new int[n];
        int[] res = new int[queries.length];
        int cur = 0;
        for (int i = 0; i < queries.length; ++i) {
            int index = queries[i][0];
            int color = queries[i][1];
            if (index > 0 && colors[index] != 0 && colors[index] == colors[index - 1]) {
                --cur;
            }
            if (index < n - 1 && colors[index] != 0 && colors[index] == colors[index + 1]) {
                --cur;
            }
            colors[index] = color;
            if (index > 0 && colors[index] == colors[index - 1]) {
                ++cur;
            }
            if (index < n - 1 && colors[index] == colors[index + 1]) {
                ++cur;
            }
            res[i] = cur;
        }
        return res;

    }

    // 6419. 使二叉树所有路径值相等的最小代价 (Make Costs of Paths Equal in a Binary Tree)
    public int minIncrements(int n, int[] cost) {
        int res = 0;
        for (int i = n / 2; i > 0; --i) {
            res += Math.abs(cost[i * 2] - cost[i * 2 - 1]);
            cost[i - 1] += Math.max(cost[i * 2], cost[i * 2 - 1]);
        }
        return res;
    }

    // 1263. 推箱子 (Minimum Moves to Move a Box to Their Target)
    public int minPushBox(char[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] dis = new int[m * n][m * n];
        for (int i = 0; i < m * n; ++i) {
            Arrays.fill(dis[i], Integer.MAX_VALUE);
        }
        Deque<int[]> q = new LinkedList<>();
        int sx = -1;
        int sy = -1;
        int bx = -1;
        int by = -1;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 'S') {
                    sx = i;
                    sy = j;
                } else if (grid[i][j] == 'B') {
                    bx = i;
                    by = j;
                }
            }
        }
        int[][] dirs = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        q.offer(new int[] { transfer(sx, sy, n), transfer(bx, by, n), 0 });
        dis[transfer(sx, sy, n)][transfer(bx, by, n)] = 0;
        while (!q.isEmpty()) {
            int[] cur = q.pollFirst();
            int x = cur[0] / n;
            int y = cur[0] % n;
            int bi = cur[1] / n;
            int bj = cur[1] % n;
            int d = cur[2];
            if (grid[bi][bj] == 'T') {
                return d;
            }
            for (int[] dir : dirs) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] != '#') {
                    if (nx != bi || ny != bj) {
                        if (d < dis[transfer(nx, ny, n)][transfer(bi, bj, n)]) {
                            dis[transfer(nx, ny, n)][transfer(bi, bj, n)] = d;
                            q.offerFirst(new int[] { transfer(nx, ny, n), transfer(bi, bj, n), d });
                        }
                    } else {
                        int nbi = nx + dir[0];
                        int nbj = ny + dir[1];
                        if (nbi >= 0 && nbi < m && nbj >= 0 && nbj < n && grid[nbi][nbj] != '#') {
                            if (d + 1 < dis[transfer(nx, ny, n)][transfer(nbi, nbj, n)]) {
                                dis[transfer(nx, ny, n)][transfer(nbi, nbj, n)] = d + 1;
                                q.offerLast(new int[] { transfer(nx, ny, n), transfer(nbi, nbj, n), d + 1 });
                            }
                        }
                    }
                }
            }
        }
        return -1;

    }

    private int transfer(int i, int j, int n) {
        return i * n + j;
    }

    // 756. 金字塔转换矩阵 (Pyramid Transition Matrix)
    private Map<String, List<Character>> map756;
    private Map<Integer, Set<String>> cans756;
    private Map<String, Boolean> memo756;

    public boolean pyramidTransition(String bottom, List<String> allowed) {
        map756 = new HashMap<>();
        cans756 = new HashMap<>();
        memo756 = new HashMap<>();
        for (String a : allowed) {
            String b = a.substring(0, 2);
            char t = a.charAt(2);
            map756.computeIfAbsent(b, k -> new ArrayList<>()).add(t);
        }
        return dfs756(bottom);

    }

    private boolean dfs756(String s) {
        if (s.length() == 1) {
            return true;
        }
        if (memo756.containsKey(s)) {
            return memo756.get(s);
        }
        if (check756(s)) {
            construct756(s, 0, new StringBuilder());
            for (String c : cans756.getOrDefault(s.length() - 1, new HashSet<>())) {
                if (dfs756(c)) {
                    memo756.put(s, true);
                    return true;
                }
            }
        }
        memo756.put(s, false);
        return false;
    }

    private boolean check756(String s) {
        int n = s.length();
        for (int i = 0; i < n - 1; ++i) {
            if (!map756.containsKey(s.substring(i, i + 2))) {
                return false;
            }
        }
        return true;
    }

    private void construct756(String s, int i, StringBuilder builder) {
        if (builder.length() == s.length() - 1) {
            cans756.computeIfAbsent(builder.length(), k -> new HashSet<>()).add(builder.toString());
            return;
        }
        int n = s.length();
        for (int j = i; j < n - 1; ++j) {
            for (char c : map756.getOrDefault(s.substring(j, j + 2), new ArrayList<>())) {
                builder.append(c);
                construct756(s, j + 1, builder);
                builder.deleteCharAt(builder.length() - 1);
            }
        }
    }

    // 282. 给表达式添加运算符 (Expression Add Operators)
    private List<String> res282;
    private char[] arr282;
    private int target282;
    private int n282;

    public List<String> addOperators(String num, int target) {
        this.res282 = new ArrayList<>();
        this.arr282 = num.toCharArray();
        this.target282 = target;
        this.n282 = arr282.length;
        dfs282(0, 0L, 0L, new StringBuilder());
        return res282;

    }

    private void dfs282(int i, long sum, long mul, StringBuilder builder) {
        if (i == n282) {
            if (sum == target282) {
                res282.add(builder.toString());
            }
            return;
        }
        int expIndex = builder.length();
        if (i > 0) {
            builder.append(0);
        }
        long val = 0L;
        for (int j = i; j < n282 && (i == j || arr282[i] != '0'); ++j) {
            val = val * 10 + arr282[j] - '0';
            builder.append(arr282[j]);
            if (i == 0) {
                dfs282(j + 1, val, val, builder);
            } else {
                builder.setCharAt(expIndex, '+');
                dfs282(j + 1, sum + val, val, builder);
                builder.setCharAt(expIndex, '-');
                dfs282(j + 1, sum - val, -val, builder);
                builder.setCharAt(expIndex, '*');
                dfs282(j + 1, sum - mul + mul * val, mul * val, builder);
            }
        }
        builder.setLength(expIndex);
    }

    // LCP 77. 符文储备
    public int runeReserve(int[] runes) {
        Arrays.sort(runes);
        int res = 1;
        int pre = runes[0];
        int max = 1;
        for (int i = 1; i < runes.length; ++i) {
            if (runes[i] - pre <= 1) {
                ++max;
            } else {
                max = 1;
            }
            res = Math.max(res, max);
            pre = runes[i];
        }
        return res;

    }

    // LCP 78. 城墙防线
    public int rampartDefensiveLine(int[][] rampart) {
        int res = 0;
        int left = 0;
        int right = (int) 1e8;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            // 可以覆盖
            if (check_LCP_78(rampart, mid)) {
                res = mid;
                left = mid + 1;
            }
            // 有重叠
            else {
                right = mid - 1;
            }
        }
        return res;

    }

    private boolean check_LCP_78(int[][] rampart, int target) {
        int n = rampart.length;
        int give = rampart[1][0] - rampart[0][1];
        int i = 1;
        while (i < n - 1) {
            int r = rampart[i + 1][0] - rampart[i][1];
            if (target > give + r) {
                return false;
            }
            give = r - Math.max(0, target - give);
            ++i;
        }
        return true;
    }

    // LCP 79. 提取咒文
    public int extractMantra(String[] matrix, String mantra) {
        int m = matrix.length;
        int n = matrix[0].length();
        int len = mantra.length();
        int[][] dirs = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        boolean[][][] vis = new boolean[m][n][len + 1];
        Queue<int[]> q = new LinkedList<>();
        // i, j, l, d
        q.offer(new int[] { 0, 0, 0, 0 });
        vis[0][0][0] = true;
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int y = cur[1];
            int l = cur[2];
            int d = cur[3];
            if (l == len) {
                return d;
            }
            if (matrix[x].charAt(y) == mantra.charAt(l)) {
                if (!vis[x][y][l + 1]) {
                    vis[x][y][l + 1] = true;
                    q.offer(new int[] { x, y, l + 1, d + 1 });
                }
            } else {
                for (int[] dir : dirs) {
                    int nx = x + dir[0];
                    int ny = y + dir[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                        if (!vis[nx][ny][l]) {
                            vis[nx][ny][l] = true;
                            q.offer(new int[] { nx, ny, l, d + 1 });
                        }
                    }
                }
            }
        }
        return -1;

    }

    // 1718. 构建字典序最大的可行序列 (Construct the Lexicographically Largest Valid Sequence)
    private int n1718;
    private int[] res1718;
    private int vis1718;
    private boolean flag1718;

    public int[] constructDistancedSequence(int n) {
        this.n1718 = n;
        this.res1718 = new int[n * 2 - 1];
        dfs1718(0);
        return res1718;

    }

    private void dfs1718(int i) {
        if (i == n1718 * 2 - 1) {
            flag1718 = true;
            return;
        }
        if (res1718[i] != 0) {
            dfs1718(i + 1);
            return;
        }
        for (int num = n1718; num > 0; --num) {
            int nextIndex = num == 1 ? i : num + i;
            if ((vis1718 & (1 << num)) != 0 || nextIndex >= n1718 * 2 - 1 || res1718[nextIndex] != 0) {
                continue;
            }
            vis1718 ^= 1 << num;
            res1718[i] = num;
            res1718[nextIndex] = num;
            dfs1718(i + 1);
            if (flag1718) {
                return;
            }
            vis1718 ^= 1 << num;
            res1718[i] = 0;
            res1718[nextIndex] = 0;
        }
    }

    // 842. 将数组拆分成斐波那契序列 (Split Array into Fibonacci Sequence)
    private List<Integer> res842;
    private char[] arr842;
    private int n842;
    private boolean flag842;

    public List<Integer> splitIntoFibonacci(String num) {
        this.res842 = new ArrayList<>();
        this.arr842 = num.toCharArray();
        this.n842 = num.length();
        dfs842(0);
        return res842;
    }

    private void dfs842(int i) {
        if (i == n842) {
            if (res842.size() >= 3) {
                flag842 = true;
            }
            return;
        }
        long val = 0L;
        for (int j = i; j < n842 && (i == j || arr842[i] != '0'); ++j) {
            val = val * 10 + arr842[j] - '0';
            if (val > Integer.MAX_VALUE) {
                break;
            }
            if (res842.size() < 2) {
                res842.add((int) val);
                dfs842(j + 1);
                if (flag842) {
                    return;
                }
                res842.remove(res842.size() - 1);
            } else {
                int last = res842.get(res842.size() - 1);
                int last2 = res842.get(res842.size() - 2);
                if (((long) last + last2) < val) {
                    break;
                }
                if (((long) last + last2) == val) {
                    res842.add((int) val);
                    dfs842(j + 1);
                    if (flag842) {
                        return;
                    }
                    res842.remove(res842.size() - 1);
                }
            }
        }
    }

    // 306. 累加数 (Additive Number)
    private char[] arr306;
    private int n306;
    private boolean flag306;
    private List<Long> list306;

    public boolean isAdditiveNumber(String num) {
        this.arr306 = num.toCharArray();
        this.n306 = num.length();
        this.list306 = new ArrayList<>();
        if (n306 <= 2) {
            return false;
        }
        dfs306(0);
        return flag306;

    }

    private void dfs306(int i) {
        if (i == n306) {
            if (list306.size() >= 3) {
                flag306 = true;
            }
            return;
        }
        long val = 0L;
        for (int j = i; j < n306 && (i == j || arr306[i] != '0') && j - i + 1 <= n306 / 2; ++j) {
            val = val * 10 + arr306[j] - '0';
            if (list306.size() < 2) {
                list306.add(val);
                dfs306(j + 1);
                if (flag306) {
                    return;
                }
                list306.remove(list306.size() - 1);
            } else {
                long last = list306.get(list306.size() - 1);
                long last2 = list306.get(list306.size() - 2);
                if (last + last2 < val) {
                    break;
                }
                if (last + last2 == val) {
                    list306.add(val);
                    dfs306(j + 1);
                    if (flag306) {
                        return;
                    }
                    list306.remove(list306.size() - 1);
                }
            }
        }
    }

    // 691. 贴纸拼词 (Stickers to Spell Word)
    private int m691;
    private String[] stickers691;
    private int[] memo691;
    private char[] target691;

    public int minStickers(String[] stickers, String target) {
        this.m691 = target.length();
        this.stickers691 = stickers;
        this.target691 = target.toCharArray();
        this.memo691 = new int[1 << m691];
        Arrays.fill(memo691, -1);
        memo691[0] = 0;
        int res = dfs691((1 << m691) - 1);
        return res <= m691 ? res : -1;

    }

    private int dfs691(int mask) {
        if (memo691[mask] >= 0) {
            return memo691[mask];
        }
        int res = m691 + 1;
        for (String s : stickers691) {
            int left = mask;
            int[] cnts = new int[26];
            for (char c : s.toCharArray()) {
                ++cnts[c - 'a'];
            }
            for (int i = 0; i < m691; ++i) {
                if (((mask >> i) & 1) == 1 && cnts[target691[i] - 'a'] > 0) {
                    --cnts[target691[i] - 'a'];
                    left ^= 1 << i;
                }
            }
            if (left < mask) {
                res = Math.min(res, dfs691(left) + 1);
            }
        }
        return memo691[mask] = res;
    }

    // 996. 正方形数组的数目 (Number of Squareful Arrays)
    private int n996;
    private int[] nums996;
    private boolean[] used996;
    private int res996;
    private List<Integer> list996;

    public int numSquarefulPerms(int[] nums) {
        Arrays.sort(nums);
        this.n996 = nums.length;
        this.nums996 = nums;
        this.used996 = new boolean[n996];
        this.list996 = new ArrayList<>();
        dfs996();
        return res996;

    }

    private void dfs996() {
        if (list996.size() == n996) {
            ++res996;
            return;
        }
        for (int i = 0; i < n996; ++i) {
            if (used996[i] || i > 0 && nums996[i] == nums996[i - 1] && !used996[i - 1]) {
                continue;
            }
            if (list996.isEmpty()) {
                used996[i] = true;
                list996.add(nums996[i]);
                dfs996();
                used996[i] = false;
                list996.remove(list996.size() - 1);
            } else {
                int sqrt = (int) Math.sqrt(list996.get(list996.size() - 1) + nums996[i]);
                if (sqrt * sqrt == list996.get(list996.size() - 1) + nums996[i]) {
                    used996[i] = true;
                    list996.add(nums996[i]);
                    dfs996();
                    used996[i] = false;
                    list996.remove(list996.size() - 1);
                }
            }
        }
    }

    // 1799. N 次操作后的最大分数和 (Maximize Score After N Operations)
    private int n1799;
    private int[] memo1799;
    private int[][] gcdArr1799;

    public int maxScore(int[] nums) {
        this.n1799 = nums.length;
        this.memo1799 = new int[1 << n1799];
        this.gcdArr1799 = new int[n1799][n1799];
        for (int i = 0; i < n1799; ++i) {
            for (int j = i + 1; j < n1799; ++j) {
                gcdArr1799[i][j] = gcd1799(nums[i], nums[j]);
            }

        }
        return dfs1799(0);

    }

    private int dfs1799(int mask) {
        if (memo1799[mask] != 0) {
            return memo1799[mask];
        }
        int res = 0;
        for (int i = 0; i < n1799; ++i) {
            if (((mask >> i) & 1) == 0) {
                for (int j = i + 1; j < n1799; ++j) {
                    if (((mask >> j) & 1) == 0) {
                        res = Math.max(res,
                                (Integer.bitCount(mask) / 2 + 1)
                                        * (i < j ? gcdArr1799[i][j] : gcdArr1799[j][i])
                                        + dfs1799(mask | (1 << i) | (1 << j)));

                    }
                }
            }

        }
        return memo1799[mask] = res;
    }

    private int gcd1799(int a, int b) {
        return b == 0 ? a : gcd1799(b, a % b);
    }

    // 464. 我能赢吗 (Can I Win)
    private int[] memo464;
    private int maxChoosableInteger464;
    private int desiredTotal464;

    public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
        if ((1 + maxChoosableInteger) * maxChoosableInteger < desiredTotal) {
            return false;
        }
        this.memo464 = new int[1 << maxChoosableInteger];
        this.maxChoosableInteger464 = maxChoosableInteger;
        this.desiredTotal464 = desiredTotal;
        return dfs464(0, 0);

    }

    private boolean dfs464(int mask, int score) {
        if (memo464[mask] != 0) {
            return memo464[mask] > 0;
        }
        for (int i = 0; i < maxChoosableInteger464; ++i) {
            if (((mask >> i) & 1) == 0) {
                if (1 + i + score >= desiredTotal464) {
                    memo464[mask] = 1;
                    return true;
                }
                if (!dfs464(mask | (1 << i), score + 1 + i)) {
                    memo464[mask] = 1;
                    return true;
                }
            }
        }
        memo464[mask] = -1;
        return false;
    }

    // 面试题 08.14. 布尔运算 (Boolean Evaluation LCCI)
    private int[][][] memo08_14;
    private int n08_14;
    private char[] arr08_14;

    public int countEval(String s, int result) {
        this.n08_14 = s.length();
        this.arr08_14 = s.toCharArray();
        this.memo08_14 = new int[n08_14][n08_14][2];
        for (int i = 0; i < n08_14; ++i) {
            for (int j = 0; j < n08_14; ++j) {
                Arrays.fill(memo08_14[i][j], -1);
            }
        }
        return dfs08_14(0, n08_14 - 1, result);

    }

    private int dfs08_14(int i, int j, int result) {
        if (i == j) {
            return 1 - ((arr08_14[i] - '0') ^ result);
        }
        if (memo08_14[i][j][result] != -1) {
            return memo08_14[i][j][result];
        }
        int res = 0;
        for (int k = i; k < j; k += 2) {
            char op = arr08_14[k + 1];
            for (int x = 0; x <= 1; ++x) {
                for (int y = 0; y <= 1; ++y) {
                    if (calculate08_14(x, y, op) == result) {
                        res += dfs08_14(i, k, x) * dfs08_14(k + 2, j, y);
                    }
                }
            }
        }
        return memo08_14[i][j][result] = res;
    }

    private int calculate08_14(int x, int y, char op) {
        if (op == '&') {
            return x & y;
        }
        if (op == '|') {
            return x | y;
        }
        return x ^ y;
    }

    // 689. 三个无重叠子数组的最大和 (Maximum Sum of 3 Non-Overlapping Subarrays)
    public int[] maxSumOfThreeSubarrays(int[] nums, int k) {
        int n = nums.length;
        int maxIndex1 = 0;
        int maxIndex12_1 = 0;
        int maxIndex12_2 = 0;
        int max1 = 0;
        int max12 = 0;
        int max = 0;
        int sum1 = 0;
        int sum2 = 0;
        int sum3 = 0;
        int[] res = new int[3];
        for (int i = k * 2; i < n; ++i) {
            sum1 += nums[i - k * 2];
            sum2 += nums[i - k];
            sum3 += nums[i];
            if (i >= k * 3 - 1) {
                if (sum1 > max1) {
                    max1 = sum1;
                    maxIndex1 = i - (k * 3 - 1);
                }
                if (max1 + sum2 > max12) {
                    max12 = max1 + sum2;
                    maxIndex12_1 = maxIndex1;
                    maxIndex12_2 = i - (k * 2 - 1);
                }
                if (max12 + sum3 > max) {
                    max = max12 + sum3;
                    res[0] = maxIndex12_1;
                    res[1] = maxIndex12_2;
                    res[2] = i - k + 1;
                }
                sum1 -= nums[i - k * 3 + 1];
                sum2 -= nums[i - k * 2 + 1];
                sum3 -= nums[i - k + 1];
            }
        }
        return res;

    }

    // 689. 三个无重叠子数组的最大和 (Maximum Sum of 3 Non-Overlapping Subarrays)
    private int[][] memo689;
    private int n689;
    private int k689;
    private int[] pre689;
    private int[] res689;

    public int[] maxSumOfThreeSubarrays2(int[] nums, int k) {
        this.n689 = nums.length;
        this.k689 = k;
        this.memo689 = new int[4][n689];
        this.pre689 = new int[n689 + 1];
        this.res689 = new int[3];
        for (int i = 0; i < n689; ++i) {
            pre689[i + 1] = pre689[i] + nums[i];
        }
        dfs689(3, n689 - 1);
        int limit = n689 - 1;
        for (int i = 3; i >= 1; --i) {
            res689[i - 1] = limit - k + 1;
            for (int j = limit; j >= i * k; --j) {
                if (memo689[i][j] == memo689[i][j - 1]) {
                    res689[i - 1] = j - k;
                    limit = res689[i - 1] - 1;
                } else {
                    limit = res689[i - 1] - 1;
                    break;
                }
            }
        }
        return res689;
    }

    private int dfs689(int count, int i) {
        if (count <= 0 || i < 0) {
            return 0;
        }
        if (count * k689 > i + 1) {
            return 0;
        }
        if (memo689[count][i] != 0) {
            return memo689[count][i];
        }
        return memo689[count][i] = Math.max(dfs689(count, i - 1),
                dfs689(count - 1, i - k689) + pre689[i + 1] - pre689[i - k689 + 1]);
    }

    // 6366. 老人的数目 (Number of Senior Citizens)
    public int countSeniors(String[] details) {
        int res = 0;
        for (String detail : details) {
            int age = Integer.parseInt(detail.substring(11, 13));
            if (age > 60) {
                ++res;
            }
        }
        return res;
    }

    // 6367. 矩阵中的和 (Sum in a Matrix)
    public int matrixSum(int[][] nums) {
        int m = nums.length;
        int n = nums[0].length;
        for (int i = 0; i < m; ++i) {
            Arrays.sort(nums[i]);
        }
        int res = 0;
        for (int j = n - 1; j >= 0; --j) {
            int max = 0;
            for (int i = 0; i < m; ++i) {
                max = Math.max(max, nums[i][j]);
            }
            res += max;
        }
        return res;

    }

    // 6369. 最大或值 (Maximum OR)
    public long maximumOr(int[] nums, int k) {
        int n = nums.length;
        long[] suf = new long[n + 1];
        for (int i = n - 1; i >= 0; --i) {
            suf[i] = suf[i + 1] | nums[i];
        }
        long res = 0L;
        long pre = 0L;
        for (int i = 0; i < n; ++i) {
            res = Math.max(res, pre | ((long) nums[i] << k) | suf[i + 1]);
            pre |= nums[i];
        }
        return res;

    }

    // 2682. 找出转圈游戏输家 (Find the Losers of the Circular Game)
    public int[] circularGameLosers(int n, int k) {
        boolean[] vis = new boolean[n];
        int mul = 0;
        int cur = 0;
        // do...while
        do {
            vis[cur] = true;
            ++mul;
            cur = (cur + k * mul) % n;
        } while (!vis[cur]);

        // while...do
        // int mul = 1;
        // vis[cur] = true;
        // while (!vis[(cur + k * mul) % n]) {
        // vis[(cur + k * mul) % n] = true;
        // cur = (cur + k * mul) % n;
        // ++mul;
        // }
        int[] res = new int[n - mul];
        int index = 0;
        for (int i = 0; i < n; ++i) {
            if (!vis[i]) {
                res[index++] = i + 1;
            }
        }
        return res;

    }

    // 2683. 相邻值的按位异或 (Neighboring Bitwise XOR)
    public boolean doesValidArrayExist(int[] derived) {
        int xor = 0;
        for (int derive : derived) {
            xor ^= derive;
        }
        return xor == 0;

    }

    // 2684. 矩阵中移动的最大次数 (Maximum Number of Moves in a Grid)
    private int[][] memo2684;
    private int m2684;
    private int n2684;
    private int[][] grid2684;

    public int maxMoves(int[][] grid) {
        this.m2684 = grid.length;
        this.n2684 = grid[0].length;
        this.grid2684 = grid;
        this.memo2684 = new int[m2684][n2684];
        for (int i = 0; i < m2684; ++i) {
            Arrays.fill(memo2684[i], -1);
        }
        int res = 0;
        for (int i = 0; i < m2684; ++i) {
            res = Math.max(res, dfs2684(i, 0));
        }
        return res;

    }

    private int dfs2684(int i, int j) {
        if (j == n2684 - 1) {
            return 0;
        }
        if (memo2684[i][j] != -1) {
            return memo2684[i][j];
        }
        int max = 0;
        for (int r = Math.max(0, i - 1); r <= Math.min(m2684 - 1, i + 1); ++r) {
            if (grid2684[r][j + 1] > grid2684[i][j]) {
                max = Math.max(max, dfs2684(r, j + 1) + 1);
            }
        }
        return memo2684[i][j] = max;
    }

    // 2684. 矩阵中移动的最大次数 (Maximum Number of Moves in a Grid)
    public int maxMoves2(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        boolean[][] vis = new boolean[m][n];
        Queue<int[]> q = new LinkedList<>();
        for (int i = 0; i < m; ++i) {
            q.offer(new int[] { i, 0 });
            vis[i][0] = true;
        }
        int res = 0;
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int y = cur[1];
            res = Math.max(res, y);
            if (res == n - 1) {
                return res;
            }
            for (int nx = Math.max(0, x - 1); nx <= Math.min(m - 1, x + 1); ++nx) {
                if (y + 1 < n && grid[nx][y + 1] > grid[x][y] && !vis[nx][y + 1]) {
                    vis[nx][y + 1] = true;
                    q.offer(new int[] { nx, y + 1 });
                }
            }
        }
        return res;

    }

    // 2685. 统计完全连通分量的数量 (Count the Number of Complete Components)
    private Map<Integer, List<Integer>> map2685;
    private Union2685 union2685;
    private Set<Integer> set2685;
    private int n2685;

    public int countCompleteComponents(int n, int[][] edges) {
        this.map2685 = new HashMap<>();
        this.union2685 = new Union2685(n);
        this.set2685 = new HashSet<>();
        this.n2685 = n;
        for (int[] e : edges) {
            union2685.union(e[0], e[1]);
            set2685.add(e[0] * n + e[1]);
            set2685.add(e[1] * n + e[0]);
        }
        for (int i = 0; i < n; ++i) {
            int root = union2685.getRoot(i);
            map2685.computeIfAbsent(root, k -> new ArrayList<>()).add(i);
        }

        int res = 0;
        for (List<Integer> vals : map2685.values()) {
            if (check2685(vals)) {
                ++res;
            }
        }
        return res;
    }

    private boolean check2685(List<Integer> vals) {
        for (int i = 0; i < vals.size(); ++i) {
            for (int j = i + 1; j < vals.size(); ++j) {
                if (!set2685.contains(vals.get(i) * n2685 + vals.get(j))) {
                    return false;
                }
            }
        }
        return true;
    }

    public class Union2685 {
        private int[] parent;
        private int[] rank;

        public Union2685(int n) {
            parent = new int[n];
            rank = new int[n];
            Arrays.fill(rank, 1);
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

    // 2685. 统计完全连通分量的数量 (Count the Number of Complete Components)
    private List<Integer>[] g2685;
    private int e2685;
    private int v2685;
    private boolean[] vis2685;

    public int countCompleteComponents2(int n, int[][] edges) {
        this.g2685 = new ArrayList[n];
        Arrays.setAll(g2685, k -> new ArrayList<>());
        for (int[] e : edges) {
            g2685[e[0]].add(e[1]);
            g2685[e[1]].add(e[0]);
        }
        this.vis2685 = new boolean[n];
        int res = 0;
        for (int i = 0; i < n; ++i) {
            if (!vis2685[i]) {
                e2685 = 0;
                v2685 = 0;
                dfs2685(i);
                if (v2685 * (v2685 - 1) == e2685) {
                    ++res;
                }
            }
        }
        return res;
    }

    private void dfs2685(int x) {
        vis2685[x] = true;
        ++v2685;
        e2685 += g2685[x].size();
        for (int y : g2685[x]) {
            if (!vis2685[y]) {
                dfs2685(y);
            }
        }
    }

    // 351. 安卓系统手势解锁 (Android Unlock Patterns) --plus
    private int used351;
    private int res351;
    private int m351;
    private int n351;
    private List<Integer> list351;

    public int numberOfPatterns(int m, int n) {
        this.m351 = m;
        this.n351 = n;
        this.list351 = new ArrayList<>();
        dfs351();
        return res351;

    }

    private void dfs351() {
        if (list351.size() >= m351) {
            ++res351;
        }
        if (list351.size() == n351) {
            return;
        }
        for (int i = 1; i <= 9; ++i) {
            if (((used351 >> i) & 1) == 1) {
                continue;
            }
            if (!list351.isEmpty()) {
                if (i + list351.get(list351.size() - 1) == 10 && ((used351 >> 5) & 1) == 0) {
                    continue;
                }
                if (i == 1
                        && (list351.get(list351.size() - 1) == 3 && ((used351 >> 2) & 1) == 0
                                || list351.get(list351.size() - 1) == 7 && ((used351 >> 4) & 1) == 0)) {
                    continue;
                }
                if (i == 3
                        && (list351.get(list351.size() - 1) == 1 && ((used351 >> 2) & 1) == 0
                                || list351.get(list351.size() - 1) == 9 && ((used351 >> 6) & 1) == 0)) {
                    continue;
                }
                if (i == 9
                        && (list351.get(list351.size() - 1) == 3 && ((used351 >> 6) & 1) == 0
                                || list351.get(list351.size() - 1) == 7 && ((used351 >> 8)
                                        & 1) == 0)) {
                    continue;
                }
                if (i == 7
                        && (list351.get(list351.size() - 1) == 1 && ((used351 >> 4) & 1) == 0
                                || list351.get(list351.size() - 1) == 9 && ((used351 >> 8)
                                        & 1) == 0)) {
                    continue;
                }
            }
            list351.add(i);
            used351 ^= 1 << i;
            dfs351();
            list351.remove(list351.size() - 1);
            used351 ^= 1 << i;
        }
    }

    // 267. 回文排列 II --plus
    private List<String> res267;
    private char[] arr267;
    private int used267;
    private int n267;
    private StringBuilder builder267;
    private String mid267;

    public List<String> generatePalindromes(String s) {
        int[] cnts = new int[26];
        for (char c : s.toCharArray()) {
            ++cnts[c - 'a'];
        }
        res267 = new ArrayList<>();
        mid267 = "";
        int odds = 0;
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < 26; ++i) {
            if ((cnts[i] & 1) == 1) {
                if (++odds > 1) {
                    return res267;
                }
                mid267 = String.valueOf((char) (i + 'a'));
            }
            int c = cnts[i] / 2;
            while (c-- > 0) {
                builder.append((char) (i + 'a'));
            }
        }

        // 方法一：回溯 全排列
        // this.arr267 = builder.toString().toCharArray();
        // Arrays.sort(arr267);
        // this.n267 = arr267.length;
        // this.builder267 = new StringBuilder();
        // dfs267();

        // 方法二：下一个排列
        this.arr267 = builder.toString().toCharArray();
        Arrays.sort(arr267);
        char[] original = arr267.clone();
        do {
            res267.add(String.valueOf(arr267) + mid267 + String.valueOf(reverse267(arr267)));
            nextPermutation267(arr267);
        } while (!String.valueOf(original).equals(String.valueOf(arr267)));

        return res267;

    }

    private void nextPermutation267(char[] arr) {
        int n = arr.length;
        int i = n - 2;
        while (i >= 0) {
            if (arr[i] < arr[i + 1]) {
                break;
            }
            --i;
        }
        if (i < 0) {
            Arrays.sort(arr);
            return;
        }
        int j = n - 1;
        while (i < j) {
            if (arr[i] < arr[j]) {
                char temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
                Arrays.sort(arr, i + 1, n);
                break;
            }
            --j;
        }
    }

    private char[] reverse267(char[] arr) {
        int n = arr.length;
        char[] res = new char[n];
        for (int i = 0; i < n; ++i) {
            res[i] = arr[n - i - 1];
        }

        return res;
    }

    private void dfs267() {
        if (builder267.length() == n267) {
            res267.add(builder267 + mid267 + String.valueOf(reverse267(builder267.toString().toCharArray())));
            return;
        }
        for (int i = 0; i < n267; ++i) {
            if (((used267 >> i) & 1) == 1 || i > 0 && arr267[i] == arr267[i - 1] && ((used267 >> (i - 1)) & 1) == 0) {
                continue;
            }
            builder267.append(arr267[i]);
            used267 ^= 1 << i;
            dfs267();
            used267 ^= 1 << i;
            builder267.deleteCharAt(builder267.length() - 1);
        }
    }

    // 1066. 校园自行车分配 II (Campus Bikes II) --plus
    private int n1066;
    private int m1066;
    private int[][] workers1066;
    private int[][] bikes1066;
    private int[][] memo1066;

    public int assignBikes(int[][] workers, int[][] bikes) {
        this.n1066 = workers.length;
        this.m1066 = bikes.length;
        this.workers1066 = workers;
        this.bikes1066 = bikes;
        this.memo1066 = new int[n1066][1 << m1066];
        return dfs1066(0, 0);

    }

    private int dfs1066(int i, int mask) {
        if (i == n1066) {
            return 0;
        }
        if (memo1066[i][mask] != 0) {
            return memo1066[i][mask];
        }
        int min = Integer.MAX_VALUE;
        for (int j = 0; j < m1066; ++j) {
            if (((mask >> j) & 1) == 0) {
                min = Math.min(min,
                        Math.abs(workers1066[i][0] - bikes1066[j][0]) + Math.abs(workers1066[i][1] - bikes1066[j][1])
                                + dfs1066(i + 1, mask | (1 << j)));
            }
        }
        return memo1066[i][mask] = min;
    }

    // 2599. 使前缀和数组非负 (Make the Prefix Sum Non-negative) --plus
    public int makePrefSumNonNegative(int[] nums) {
        Queue<Long> q = new PriorityQueue<>();
        long prefix = 0L;
        int res = 0;
        for (long num : nums) {
            prefix += num;
            q.offer(num);
            if (prefix < 0) {
                prefix -= q.poll();
                ++res;
            }
        }
        return res;

    }

    // 2031. 1 比 0 多的子数组个数 (Count Subarrays With More Ones Than Zeros)
    public int subarraysWithMoreZerosThanOnes(int[] nums) {
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        List<Integer> list = new ArrayList<>();
        list.add(0);
        int prefix = 0;
        for (int num : nums) {
            prefix += num * 2 - 1;
            int index = binarySearch2031(list, prefix);
            res = (res + index + 1) % MOD;
            list.add(index + 1, prefix);
        }
        return res;

    }

    // 找排序list中，< target的最大值对应的最大索引，若不存在，返回-1
    private int binarySearch2031(List<Integer> list, int target) {
        int n = list.size();
        if (list.get(n - 1) < target) {
            return n - 1;
        }
        if (target <= list.get(0)) {
            return -1;
        }
        int left = 0;
        int right = n - 1;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (list.get(mid) < target) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    // 2510. 检查是否有路径经过相同数量的 0 和 1 (Check if There is a Path With Equal Number of 0's And 1's) --plus
    private int[][][] memo2510;
    private int m2510;
    private int n2510;
    private int[][] grid2510;

    public boolean isThereAPath(int[][] grid) {
        this.m2510 = grid.length;
        this.n2510 = grid[0].length;
        this.grid2510 = grid;
        if ((m2510 + n2510 - 1) % 2 == 1) {
            return false;
        }
        // memo[i][j][k] : 从 [i, j] 出发，且1比0多k个的情况下，能否在到达右下角时，存在1和0的个数相等的路径
        // == 0 未访问
        // == 1 存在合法路径
        // == -1 不存在合法路径
        this.memo2510 = new int[m2510][n2510][m2510 + n2510];
        return dfs2510(0, 0, 0);

    }

    private boolean dfs2510(int i, int j, int k) {
        if (i == m2510 || j == n2510) {
            return false;
        }
        if (i == m2510 - 1 && j == n2510 - 1) {
            return k + (grid2510[i][j] == 1 ? 1 : -1) == 0;
        }
        if (m2510 - i + n2510 - j - 1 < Math.abs(k)) {
            return false;
        }
        if (memo2510[i][j][k + (m2510 + n2510) / 2] != 0) {
            return memo2510[i][j][k + (m2510 + n2510) / 2] > 0;
        }
        boolean res = dfs2510(i + 1, j, grid2510[i][j] == 1 ? k + 1 : k - 1)
                || dfs2510(i, j + 1, grid2510[i][j] == 1 ? k + 1 : k - 1);
        memo2510[i][j][k + (m2510 + n2510) / 2] = res ? 1 : -1;
        return res;
    }

    // 2510. 检查是否有路径经过相同数量的 0 和 1 (Check if There is a Path With Equal Number of 0's
    // And 1's) --plus
    public boolean isThereAPath2(int[][] grid) {
        int[][] dirs = { { 0, 1 }, { 1, 0 } };
        int m = grid.length;
        int n = grid[0].length;
        if ((m + n - 1) % 2 == 1) {
            return false;
        }
        boolean[][][] vis = new boolean[m][n][m + n + m + n];
        Queue<int[]> q = new LinkedList<>();
        q.offer(new int[] { 0, 0, grid[0][0] > 0 ? 1 : -1 });
        vis[0][0][(grid[0][0] > 0 ? 1 : -1) + m + n] = true;
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int y = cur[1];
            int k = cur[2];
            if (x == m - 1 && y == n - 1) {
                if (k == 0) {
                    return true;
                }
                continue;
            }
            if (m - x + n - y - 1 < Math.abs(k)) {
                continue;
            }
            for (int[] dir : dirs) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                if (nx == m || ny == n) {
                    continue;
                }
                int nk = (grid[nx][ny] > 0 ? 1 : -1) + k + m + n;
                if (!vis[nx][ny][nk]) {
                    vis[nx][ny][nk] = true;
                    q.offer(new int[] { nx, ny, (grid[nx][ny] > 0 ? 1 : -1) + k });
                }
            }
        }
        return false;
    }

    // 1230. 抛掷硬币 (Toss Strange Coins) --plus
    private int n1230;
    private int target1230;
    private double[] prob1230;
    private double[][] memo1230;
    private double[] sufNeg1230;
    private double[] sufPos1230;

    public double probabilityOfHeads(double[] prob, int target) {
        this.n1230 = prob.length;
        this.target1230 = target;
        this.prob1230 = prob;
        this.memo1230 = new double[n1230][target];
        for (int i = 0; i < n1230; ++i) {
            Arrays.fill(memo1230[i], -1D);
        }
        this.sufNeg1230 = new double[n1230];
        this.sufPos1230 = new double[n1230];
        sufNeg1230[n1230 - 1] = 1D - prob[n1230 - 1];
        sufPos1230[n1230 - 1] = prob[n1230 - 1];
        for (int i = n1230 - 2; i >= 0; --i) {
            sufNeg1230[i] = sufNeg1230[i + 1] * (1D - prob[i]);
            sufPos1230[i] = sufPos1230[i + 1] * prob[i];
        }
        return dfs1230(0, 0);

    }

    private double dfs1230(int i, int j) {
        if (j == target1230) {
            return i < n1230 ? sufNeg1230[i] : 1D;
        }
        if (i == n1230) {
            return 0D;
        }
        // 需要再选 target - j 个硬币 > 还有 n - i 个硬币可选 
        if (n1230 - i < target1230 - j) {
            return 0D;
        }
        // 需要再选 target - j 个硬币 = 还有 n - i 个硬币可选 全部选正面朝上
        if (target1230 - j == n1230 - i) {
            return sufPos1230[i];
        }
        if (memo1230[i][j] != -1D) {
            return memo1230[i][j];
        }
        // 第 i 枚硬币朝反 + 第 i 枚硬币朝正
        return memo1230[i][j] = dfs1230(i + 1, j) * (1D - prob1230[i]) + dfs1230(i + 1, j + 1) * prob1230[i];

    }

    // 269. 火星词典 (Alien Dictionary) --plus
    public String alienOrder(String[] words) {
        int mask = 0;
        Map<Integer, List<Integer>> g = new HashMap<>();
        int[] degree = new int[26];
        int n = words.length;
        for (int i = 0; i < n; ++i) {
            for (char c : words[i].toCharArray()) {
                mask |= 1 << (c - 'a');
            }
            if (i < n - 1) {
                int[] cur = compare269(words[i], words[i + 1]);
                if (cur == null) {
                    return "";
                }
                if (cur[0] != -1) {
                    g.computeIfAbsent(cur[0], k -> new ArrayList<>()).add(cur[1]);
                    ++degree[cur[1]];
                }
            }
        }
        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < 26; ++i) {
            if (((mask >> i) & 1) == 1 && degree[i] == 0) {
                q.offer(i);
            }
        }
        StringBuilder res = new StringBuilder();
        while (!q.isEmpty()) {
            int x = q.poll();
            res.append((char) (x + 'a'));
            for (int y : g.getOrDefault(x, new ArrayList<>())) {
                if (--degree[y] == 0) {
                    q.offer(y);
                }
            }
        }
        return res.length() == Integer.bitCount(mask) ? res.toString() : "";

    }

    private int[] compare269(String s1, String s2) {
        for (int i = 0; i < Math.min(s1.length(), s2.length()); ++i) {
            if (s1.charAt(i) != s2.charAt(i)) {
                return new int[] { s1.charAt(i) - 'a', s2.charAt(i) - 'a' };
            }
        }
        if (s1.length() > s2.length()) {
            return null;
        }
        return new int[] { -1, -1 };
    }

    // 271. 字符串的编码与解码 (Encode and Decode Strings) --plus
    public class Codec {
        private String sEmpty = Character.toString((char) 258);
        private String mDivider = Character.toString((char) 257);

        // Encodes a list of strings to a single string.
        public String encode(List<String> strs) {
            if (strs.isEmpty()) {
                return sEmpty;
            }
            StringBuilder res = new StringBuilder();
            for (String s : strs) {
                res.append(s);
                res.append(mDivider);
            }
            res.deleteCharAt(res.length() - 1);
            return res.toString();
        }

        // Decodes a single string to a list of strings.
        public List<String> decode(String s) {
            if (sEmpty.equals(s)) {
                return new ArrayList<>();
            }
            return Arrays.asList(s.split(mDivider, -1));
        }
    }

    // 1121. 将数组分成几个递增序列 (Divide Array Into Increasing Sequences) --plus
    public boolean canDivideIntoSubsequences(int[] nums, int k) {
        int n = nums.length;
        int cnt = 0;
        int pre = nums[0];
        for (int num : nums) {
            if (pre == num) {
                ++cnt;
            } else {
                cnt = 1;
                pre = num;
            }
            if (cnt * k > n) {
                return false;
            }
        }
        return true;

    }

    // 1067. 范围内的数字计数 (Digit Count in Range) --plus
    private int d1067;
    private int[][] memo1067;
    private char[] arr1067;
    private int n1067;

    public int digitsCount(int d, int low, int high) {
        this.d1067 = d;
        return count1067(high) - count1067(low - 1);

    }

    private int count1067(int num) {
        this.arr1067 = String.valueOf(num).toCharArray();
        this.n1067 = arr1067.length;
        this.memo1067 = new int[n1067][n1067];
        for (int i = 0; i < n1067; ++i) {
            Arrays.fill(memo1067[i], -1);
        }
        return dfs1067(0, 0, true, false);
    }

    private int dfs1067(int i, int count, boolean isLimit, boolean isNum) {
        if (i == n1067) {
            return isNum ? count : 0;
        }
        if (isNum && !isLimit && memo1067[i][count] != -1) {
            return memo1067[i][count];
        }
        int res = 0;
        if (!isNum) {
            res = dfs1067(i + 1, count, false, false);
        }
        int up = isLimit ? arr1067[i] - '0' : 9;
        for (int num = isNum ? 0 : 1; num <= up; ++num) {
            res += dfs1067(i + 1, count + (num == d1067 ? 1 : 0), isLimit && num == up, true);
        }
        if (isNum && !isLimit) {
            memo1067[i][count] = res;
        }
        return res;
    }

    // 247. 中心对称数 II (Strobogrammatic Number II) --plus
    private int n247;
    private List<String> res247;
    private int[] dic247 = { 0, 1, 6, 8, 9 };
    private StringBuilder builder247;
    private int[] map247;

    public List<String> findStrobogrammatic(int n) {
        this.n247 = n;
        this.res247 = new ArrayList<>();
        this.builder247 = new StringBuilder();
        this.map247 = new int[10];
        map247[8] = 8;
        map247[1] = 1;
        map247[6] = 9;
        map247[9] = 6;
        dfs247();
        return res247;

    }

    private void dfs247() {
        if (builder247.length() == n247 / 2) {
            if ((n247 & 1) == 1) {
                res247.add(builder247 + "0" + upSideDown247(builder247.toString().toCharArray()));
                res247.add(builder247 + "1" + upSideDown247(builder247.toString().toCharArray()));
                res247.add(builder247 + "8" + upSideDown247(builder247.toString().toCharArray()));
            } else {
                res247.add(builder247 + upSideDown247(builder247.toString().toCharArray()).toString());
            }
            return;
        }
        for (int i = 0; i < dic247.length; ++i) {
            if (builder247.isEmpty() && dic247[i] == 0) {
                continue;
            }
            builder247.append(dic247[i]);
            dfs247();
            builder247.deleteCharAt(builder247.length() - 1);
        }
    }

    private StringBuilder upSideDown247(char[] array) {
        StringBuilder b = new StringBuilder();
        int n = array.length;
        for (int i = n - 1; i >= 0; --i) {
            b.append(map247[array[i] - '0']);
        }
        return b;
    }

    // 2674. 拆分循环链表 (Split a Circular Linked List) --plus
    public ListNode[] splitCircularLinkedList(ListNode list) {
        ListNode head = list;
        ListNode slow = head;
        ListNode fast = head;
        while (!(fast.next == head || fast.next.next == head)) {
            slow = slow.next;
            fast = fast.next.next;
        }
        if (fast.next.next == head) {
            fast = fast.next;
        }
        fast.next = slow.next;
        slow.next = head;
        return new ListNode[] { slow.next, fast.next };
    }

    // 2505. 所有子序列和的按位或 (Bitwise OR of All Subsequence Sums) --plus
    public long subsequenceSumOr(int[] nums) {
        long pre = 0L;
        long res = 0L;
        for (int num : nums) {
            pre = pre + num;
            res = res | num | pre;
        }
        return res;

    }

    // 2655. 寻找最大长度的未覆盖区间 (Find Maximal Uncovered Ranges) --plus
    public int[][] findMaximalUncoveredRanges(int n, int[][] ranges) {
        Arrays.sort(ranges, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });

        List<int[]> list = new ArrayList<>();
        int i = 0;
        int j = 1;
        while (i < ranges.length) {
            int left = ranges[i][0];
            int right = ranges[i][1];
            while (j < ranges.length && ranges[j][0] - 1 <= right) {
                right = Math.max(right, ranges[j][1]);
                ++j;
            }
            list.add(new int[] { left, right });
            i = j;
            ++j;
        }
        List<int[]> res = new ArrayList<>();
        int left = 0;
        for (int[] cur : list) {
            if (cur[0] - 1 >= left) {
                res.add(new int[] { left, cur[0] - 1 });
            }
            left = cur[1] + 1;
        }
        if (left <= n - 1) {
            res.add(new int[] { left, n - 1 });
        }

        return res.toArray(new int[0][]);
    }

    // 2431. 最大限度地提高购买水果的口味 (Maximize Total Tastiness of Purchased Fruits) --plus
    private int[][][] memo2431;
    private int n2431;
    private int[] price2431;
    private int[] tastiness2431;
    private int maxAmount2431;
    private int maxCoupons2431;

    public int maxTastiness(int[] price, int[] tastiness, int maxAmount, int maxCoupons) {
        this.n2431 = price.length;
        this.price2431 = price;
        this.tastiness2431 = tastiness;
        this.maxAmount2431 = maxAmount;
        this.maxCoupons2431 = maxCoupons;
        this.memo2431 = new int[n2431][maxAmount + 1][maxCoupons + 1];
        for (int i = 0; i < n2431; ++i) {
            for (int j = 0; j < maxAmount + 1; ++j) {
                Arrays.fill(memo2431[i][j], -1);
            }
        }
        return dfs2431(0, 0, 0);
    }

    private int dfs2431(int i, int amount, int coupons) {
        if (i == n2431) {
            return 0;
        }
        if (memo2431[i][amount][coupons] != -1) {
            return memo2431[i][amount][coupons];
        }
        // 不选
        int max = dfs2431(i + 1, amount, coupons);
        // 选 不用优惠券
        if (amount + price2431[i] <= maxAmount2431) {
            max = Math.max(max, dfs2431(i + 1, amount + price2431[i], coupons) + tastiness2431[i]);
        }
        // 选 用优惠券 (coupons < maxCoupons)
        if (coupons < maxCoupons2431) {
            int newAmount = amount + (price2431[i] >> 1);
            if (newAmount <= maxAmount2431) {
                max = Math.max(max, dfs2431(i + 1, newAmount, coupons + 1) + tastiness2431[i]);
            }
        }
        return memo2431[i][amount][coupons] = max;
    }

    // 2533. 好二进制字符串的数量 (Number of Good Binary Strings) --plus
    public int goodBinaryStrings(int minLength, int maxLength, int oneGroup, int zeroGroup) {
        int[] dp = new int[maxLength + 1];
        dp[0] = 1;
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        for (int i = 0; i <= maxLength; ++i) {
            if (i - oneGroup >= 0) {
                dp[i] = (dp[i] + dp[i - oneGroup]) % MOD;
            }
            if (i - zeroGroup >= 0) {
                dp[i] = (dp[i] + dp[i - zeroGroup]) % MOD;
            }
            if (i >= minLength) {
                res = (res + dp[i]) % MOD;
            }
        }
        return res;

    }

    // 2489. 固定比率的子字符串数 (Number of Substrings With Fixed Ratio) --plus
    public long fixedRatio(String s, int num1, int num2) {
        long[] cnts = new long[2];
        Map<Long, Long> map = new HashMap<>();
        map.put(0L, 1L);
        long res = 0L;
        for (char c : s.toCharArray()) {
            ++cnts[c - '0'];
            long key = cnts[0] * num2 - cnts[1] * num1;
            res += map.getOrDefault(key, 0L);
            map.merge(key, 1L, Long::sum);
        }
        return res;

    }

    // 2247. K 条高速公路的最大旅行费用 (Maximum Cost of Trip With K Highways) --plus
    private List<int[]>[] g2247;
    private int k2247;

    public int maximumCost(int n, int[][] highways, int k) {
        if (k > n - 1) {
            return -1;
        }
        this.k2247 = k;
        g2247 = new ArrayList[n];
        Arrays.setAll(g2247, o -> new ArrayList<>());
        for (int[] h : highways) {
            int a = h[0];
            int b = h[1];
            int fee = h[2];
            g2247[a].add(new int[] { b, fee });
            g2247[b].add(new int[] { a, fee });
        }
        int res = -1;
        for (int i = 0; i < n; ++i) {
            res = Math.max(res, calculateCost(i));
        }
        return res;

    }

    private int calculateCost(int start) {
        Queue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o2[1], o1[1]);
            }

        });
        // node, fee, road
        q.offer(new int[] { start, 0, 1 << start });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int curFee = cur[1];
            int mask = cur[2];
            if (Integer.bitCount(mask) - 1 == k2247) {
                return curFee;
            }
            for (int[] nei : g2247[x]) {
                int y = nei[0];
                int fee = nei[1];
                if (((mask >> y) & 1) == 0) {
                    q.offer(new int[] { y, curFee + fee, mask | (1 << y) });
                }
            }
        }
        return -1;
    }

    // 2098. 长度为 K 的最大偶数和子序列 (Subsequence of Size K With the Largest Even Sum) --plus
    public long largestEvenSum(int[] nums, int k) {
        int n = nums.length;
        Arrays.sort(nums);
        List<Long> odds = new ArrayList<>();
        List<Long> evens = new ArrayList<>();
        odds.add(0L);
        evens.add(0L);
        for (int i = n - 1; i >= 0; --i) {
            if ((nums[i] & 1L) == 0) {
                evens.add(evens.get(evens.size() - 1) + nums[i]);
            } else {
                odds.add(odds.get(odds.size() - 1) + nums[i]);
            }
        }
        long res = -1L;
        for (int i = 0; i < odds.size() && k - i >= 0; i += 2) {
            if (k - i < evens.size()) {
                res = Math.max(res, odds.get(i) + evens.get(k - i));
            }
        }
        return res;

    }



    // 1186. 删除一次得到子数组最大和 (Maximum Subarray Sum with One Deletion)
    // private int[] arr1186;
    // private int[][] memo1186;

    // public int maximumSum(int[] arr) {
    // int n = arr.length;
    // this.arr1186 = arr;
    // this.memo1186 = new int[n][2];
    // for (int i = 0; i < n; ++i) {
    // Arrays.fill(memo1186[i], Integer.MIN_VALUE);
    // }
    // // int res = Integer.MIN_VALUE;
    // // for (int i = 0; i < n; ++i) {
    // // res = Math.max(res, dfs(i, 1));
    // // }
    // return dfs(n - 1, 1);

    // }

    // private int dfs(int i, int count) {
    // if (i == 0) {
    // return arr1186[i];
    // }
    // if (count == 0) {
    // return Math.max(0, dfs(i - 1, 0)) + arr1186[i];
    // }
    // if (memo1186[i][count] != Integer.MIN_VALUE) {
    // return memo1186[i][count];
    // }
    // return memo1186[i][count] = Math.max(
    // arr1186[i], Math.max(arr1186[i] + dfs(i - 1, count), dfs(i - 1, count - 1)));
    // }

    // 1316. 不同的循环子字符串 (Distinct Echo Substrings)
    // public int distinctEchoSubstrings(String text) {

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

    // 1406. 石子游戏 III (Stone Game III)
    // public String stoneGameIII(int[] stoneValue) {

    // }
}