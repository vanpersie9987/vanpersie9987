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
        return dfs(0, 0, 0);

    }

    private int dfs(int i, int count, int state) {
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
            res = Math.max(res, -prices188[i] + dfs(i + 1, count, state ^ 1));
            // 不买
            res = Math.max(res, dfs(i + 1, count, state));
        }
        // 已买入状态 可卖出
        else {
            // 卖
            res = Math.max(res, prices188[i] + dfs(i + 1, count + 1, state ^ 1));
            // 不卖
            res = Math.max(res, dfs(i + 1, count, state));
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
}
