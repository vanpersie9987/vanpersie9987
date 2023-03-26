import java.sql.RowId;
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
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import javax.print.DocFlavor.INPUT_STREAM;

public class Leetcode_7 {
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

    // 1363. 形成三的最大倍数 (Largest Multiple of Three)
    // public String largestMultipleOfThree(int[] digits) {

    // }

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
        if (n == 1) {
            return true;
        }
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
            int less = binarySearch_6357_1(nums, q);
            int more = binarySearch_6357_2(nums, q);
            long cur = (long) less * q - preSum[less] + (preSum[n] - preSum[n - more]) - (long) more * q;
            res.add(cur);
        }
        return res;



    }

    private int binarySearch_6357_2(int[] nums, int target) {
        int n = nums.length;
        if (nums[n - 1] <= target) {
            return 0;
        }
        if (target < nums[0]) {
            return n;
        }
        int left = 0;
        int right = n - 1;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (nums[mid] > target) {
                res = n - mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    private int binarySearch_6357_1(int[] nums, int target) {
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
}
