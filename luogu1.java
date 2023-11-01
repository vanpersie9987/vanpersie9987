import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@SuppressWarnings("unchecked")
public class luogu1 {
    public static void main(String[] args) {
        // int[] res = calculateEachDigitsCounts(1L, 99L);
        // for (int i = 0; i < 10; ++i) {
        // System.out.println(res[i]);
        // }
        // int res = annoyingMathWork(1, 100);
        // System.out.println(res);
        // int res = binaryProblem(7, 2);
        // int res = roundNumbers(2, 12);
        // System.out.println(res);
        // int[][] arr = { { 1, 2 }, { 2, 3 } };
        // int res = GEPPETTO(3, arr);
        // System.out.println(res);
    }

    // P1122 最大子树和
    /**
     * input:
     * n朵花
     * len(exponent) = n
     * edges: a, b 之间有一条无向边
     */
    private List<Integer>[] g1122;
    private int[] exponent1122;
    private int res1122;

    public int maxSubTreeSum(int n, int[] exponent, int[][] edges) {
        this.g1122 = new ArrayList[n];
        Arrays.setAll(g1122, k -> new ArrayList<>());
        for (int[] e : edges) {
            g1122[e[0]].add(e[1]);
            g1122[e[1]].add(e[0]);
        }
        this.exponent1122 = exponent;
        this.res1122 = Integer.MIN_VALUE;
        dfs1122(0, -1);
        return res1122;

    }

    private int dfs1122(int x, int fa) {
        int sum = exponent1122[x];
        for (int y : g1122[x]) {
            if (y != fa) {
                sum += dfs1122(y, x);
            }
        }
        res1122 = Math.max(res1122, sum);
        return Math.max(0, sum);
    }

    // P1042 [NOIP2003 普及组] 乒乓球
    public List<List<String>> tableTennisResults(String s) {
        List<List<String>> res = new ArrayList<>();
        res.add(checkScores(s, 11));
        res.add(checkScores(s, 21));
        return res;
    }

    private List<String> checkScores(String s, int limit) {
        List<String> res = new ArrayList<>();
        int a = 0;
        int b = 0;
        for (char c : s.toCharArray()) {
            if (c == 'W') {
                ++a;
            } else {
                ++b;
            }
            if ((a >= limit || b >= limit) && Math.abs(a - b) >= 2) {
                res.add(a + ":" + b);
                a = 0;
                b = 0;
            }
        }
        if (a != 0 || b != 0) {
            res.add(a + ":" + b);
        }
        return res;
    }

    // P1831 杠杆数
    public long leverNumber(long left, long right) {
        return solve1831(right) - solve1831(left - 1);
    }

    private char[] arr1831;
    private int n1831;
    private long[][][] memo1831;

    private long solve1831(long num) {
        this.arr1831 = String.valueOf(num).toCharArray();
        this.n1831 = arr1831.length;
        this.memo1831 = new long[n1831][n1831][2000];
        for (int i = 0; i < n1831; ++i) {
            for (int j = 0; j < n1831; ++j) {
                Arrays.fill(memo1831[i][j], -1L);
            }
        }
        long res = 0L;
        for (int i = 0; i < n1831; ++i) {
            res += dfs1831(0, i, 0, true, false);
        }
        return res;
    }

    /**
     * @param i       选择了第几位
     * @param j       选择的杠杆位
     * @param k       杠杆和
     * @param isLimit 当前是否受限
     * @param isNum   是否为数
     * @return
     */
    private long dfs1831(int i, int j, int k, boolean isLimit, boolean isNum) {
        if (i == n1831) {
            return isNum && k == 0 ? 1L : 0L;
        }
        if (k < 0) {
            return 0L;
        }
        if (!isLimit && isNum && memo1831[i][j][k] != -1L) {
            return memo1831[i][j][k];
        }
        long res = 0L;
        if (!isNum) {
            res = dfs1831(i + 1, j, k, false, false);
        }
        int up = isLimit ? arr1831[i] - '0' : 9;
        for (int d = isNum ? 0 : 1; d <= up; ++d) {
            res += dfs1831(i + 1, j, k + (j - i) * d, isLimit && d == up, true);
        }
        if (!isLimit && isNum) {
            memo1831[i][j][k] = res;
        }
        return res;
    }

    // P6754 [BalticOI 2013 Day1] Palindrome-Free Numbers
    public long palindromeFreeNumbers(long left, long right) {
        return solve6754(right) - solve6754(left) + check6754(left);
    }

    private long check6754(long num) {
        char[] arr = String.valueOf(num).toCharArray();
        int n = arr.length;
        for (int i = 1; i < n; ++i) {
            if (arr[i] == arr[i - 1] || i >= 2 && arr[i] == arr[i - 2]) {
                return 0L;
            }
        }
        return 1L;
    }

    private char[] arr6754;
    private int n6754;
    private long[][][] memo6754;

    private long solve6754(long num) {
        this.arr6754 = String.valueOf(num).toCharArray();
        this.n6754 = arr6754.length;
        this.memo6754 = new long[n6754][11][11];
        for (int i = 0; i < n6754; ++i) {
            for (int j = 0; j < 11; ++j) {
                Arrays.fill(memo6754[i][j], -1L);
            }
        }
        return dfs6754(0, 10, 10, true, false);
    }

    /**
     * 
     * @param i       当前位
     * @param j       上一位选的数，若未选，则为10
     * @param k       上上一位选的数，若未选，则为10
     * @param isLimit 是否受限
     * @param isNum   是否为数
     * @return
     */
    private long dfs6754(int i, int j, int k, boolean isLimit, boolean isNum) {
        if (i == n6754) {
            return isNum ? 1L : 0L;
        }
        if (!isLimit && isNum && memo6754[i][j][k] != -1L) {
            return memo6754[i][j][k];
        }
        long res = 0L;
        if (!isNum) {
            res = dfs6754(i + 1, 10, 10, false, false);
        }
        int up = isLimit ? arr6754[i] - '0' : 9;
        for (int d = isNum ? 0 : 1; d <= up; ++d) {
            if (d != j && d != k) {
                res += dfs6754(i + 1, d, j, isLimit && d == up, true);
            }
        }
        if (!isLimit && isNum) {
            memo6754[i][j][k] = res;
        }
        return res;
    }

    // P2602[ZJOI2010] 数字计数
    public int[] calculateEachDigitsCounts(long left, long right) {
        int[] res = new int[10];
        for (int i = 0; i < 10; ++i) {
            res[i] = solve2602(String.valueOf(right), i) - solve2602(String.valueOf(left - 1), i);
        }
        return res;
    }

    private int n2602;
    private char[] arr2602;
    private int target2602;
    private int[][] memo2602;

    private int solve2602(String s, int target) {
        this.n2602 = s.length();
        this.arr2602 = s.toCharArray();
        this.target2602 = target;
        this.memo2602 = new int[n2602][n2602];
        for (int i = 0; i < n2602; ++i) {
            Arrays.fill(memo2602[i], -1);
        }
        return dfs2602(0, 0, true, false);
    }

    private int dfs2602(int i, int j, boolean isLimit, boolean isNum) {
        if (i == n2602) {
            return isNum ? j : 0;
        }
        if (!isLimit && isNum && memo2602[i][j] != -1) {
            return memo2602[i][j];
        }
        int res = 0;
        if (!isNum) {
            res = dfs2602(i + 1, j, false, false);
        }
        int up = isLimit ? arr2602[i] - '0' : 9;
        for (int d = isNum ? 0 : 1; d <= up; ++d) {
            res += dfs2602(i + 1, j + (d == target2602 ? 1 : 0), isLimit && d == up, true);
        }
        if (!isLimit && isNum) {
            memo2602[i][j] = res;
        }
        return res;
    }

    // P4999 烦人的数学作业
    public int annoyingMathWork(long left, long right) {
        final int MOD = (int) (1e9 + 7);
        return (solve4999(String.valueOf(right)) - solve4999(String.valueOf(left - 1)) + MOD) % MOD;
    }

    private int n4999;
    private char[] arr4999;
    private int[][] memo4999;

    private int solve4999(String s) {
        this.n4999 = s.length();
        this.arr4999 = s.toCharArray();
        this.memo4999 = new int[n4999][n4999 * 9];
        for (int i = 0; i < n4999; ++i) {
            Arrays.fill(memo4999[i], -1);
        }
        return dfs4999(0, 0, true, false);
    }

    private int dfs4999(int i, int j, boolean isLimit, boolean isNum) {
        if (i == n4999) {
            return isNum ? j : 0;
        }
        if (isNum && !isLimit && memo4999[i][j] != -1) {
            return memo4999[i][j];
        }
        int res = 0;
        if (!isNum) {
            res = dfs4999(i + 1, j, false, false);
        }
        final int MOD = (int) (1e9 + 7);
        int up = isLimit ? arr4999[i] - '0' : 9;
        for (int d = isNum ? 0 : 1; d <= up; ++d) {
            res += dfs4999(i + 1, j + d, isLimit && d == up, true);
            res %= MOD;
        }
        if (isNum && !isLimit) {
            memo4999[i][j] = res;
        }
        return memo4999[i][j] = res;
    }

    // P8764 [蓝桥杯 2021 国 BC] 二进制问题
    private int n8764;
    private char[] arr8764;
    private int k8764;
    private int[][] memo8764;

    public int binaryProblem(long n, int k) {
        this.arr8764 = Long.toBinaryString(n).toCharArray();
        this.n8764 = arr8764.length;
        this.k8764 = k;
        memo8764 = new int[this.n8764][k + 1];
        for (int i = 0; i < this.n8764; ++i) {
            Arrays.fill(memo8764[i], -1);
        }
        return dfs8764(0, 0, true, false);

    }

    private int dfs8764(int i, int j, boolean isLimit, boolean isNum) {
        if (i == n8764) {
            return isNum && j == k8764 ? 1 : 0;
        }
        if (isNum && !isLimit && memo8764[i][j] != -1) {
            return memo8764[i][j];
        }
        int res = 0;
        if (!isNum) {
            res = dfs8764(i + 1, j, false, false);
        }
        int up = isLimit ? arr8764[i] - '0' : 1;
        for (int d = isNum ? 0 : 1; d <= up; ++d) {
            if (d + j <= k8764) {
                res += dfs8764(i + 1, d + j, isLimit && up == d, true);
            }
        }
        if (isNum && !isLimit) {
            memo8764[i][j] = res;
        }
        return res;
    }

    // P4124 [CQOI2016] 手机号码
    public int phonwNumberChecking(long left, long right) {
        return solve4124(String.valueOf(right)) - solve4124(String.valueOf(left - 1));
    }

    private int n4124;
    private char[] arr4124;
    private int[][][][][] memo4124;

    private int solve4124(String s) {
        this.n4124 = s.length();
        this.arr4124 = s.toCharArray();
        this.memo4124 = new int[n4124][11][11][2][1 << 2];
        for (int i = 0; i < n4124; ++i) {
            for (int j = 0; j < 11; ++j) {
                for (int k = 0; k < 11; ++k) {
                    for (int x = 0; x < 2; ++x) {
                        Arrays.fill(memo4124[i][j][k][x], -1);
                    }
                }
            }
        }
        return dfs4124(0, 10, 10, 0, 0, true, false);
    }

    /**
     * 
     * @param i       当前位
     * @param j       上一位选的值 未选过为10
     * @param k       上上一位选的值 未选过为10
     * @param c1      号码中出现连续3个相同的数字 1：是 0：不是
     * @param c2      号码中出现8或4 用00 bit表示 高bit位为1表示出现过8 低bit位为1表示出现过4
     * @param isLimit 是否受到约束
     * @param isNum   是否选择了数
     * @return
     */
    private int dfs4124(int i, int j, int k, int c1, int c2, boolean isLimit, boolean isNum) {
        if (i == n4124) {
            return isNum && c1 == 1 ? 1 : 0;
        }
        if (!isLimit && isNum && memo4124[i][j][k][c1][c2] != -1) {
            return memo4124[i][j][k][c1][c2];
        }
        int res = 0;
        if (!isNum) {
            res += dfs4124(i + 1, j, k, c1, c2, false, false);
        }
        int up = isLimit ? arr4124[i] - '0' : 9;
        for (int d = isNum ? 0 : 1; d <= up; ++d) {
            if (d == 8 && (c2 & (1 << 0)) != 0 || d == 4 && (c2 & (1 << 1)) != 0) {
                continue;
            }
            res += dfs4124(i + 1, d, j, c1 | (d == j && d == k ? 1 : 0),
                    d == 4 ? (c2 | (1 << 0)) : (d == 8 ? (c2 | (1 << 1)) : c2), isLimit && d == up, true);
        }
        if (!isLimit && isNum) {
            memo4124[i][j][k][c1][c2] = res;
        }
        return res;
    }

    // P6218 [USACO06NOV] Round Numbers S 圆数
    public int roundNumbers(int left, int right) {
        return solve6218(Integer.toBinaryString(right)) - solve6218(Integer.toBinaryString(left - 1));
    }

    private int n6218;
    private char[] arr6218;
    private int[][] memo6218;

    private int solve6218(String s) {
        this.n6218 = s.length();
        this.arr6218 = s.toCharArray();
        this.memo6218 = new int[n6218][64];
        for (int i = 0; i < n6218; ++i) {
            Arrays.fill(memo6218[i], -1);
        }
        return dfs6218(0, 30, true, false);
    }

    private int dfs6218(int i, int diff, boolean isLimit, boolean isNum) {
        if (i == n6218) {
            return isNum && diff >= 30 ? 1 : 0;
        }
        if (!isLimit && isNum && memo6218[i][diff] != -1) {
            return memo6218[i][diff];
        }
        int res = 0;
        if (!isNum) {
            res = dfs6218(i + 1, diff, false, false);
        }
        int up = isLimit ? arr6218[i] - '0' : 1;
        for (int d = isNum ? 0 : 1; d <= up; ++d) {
            res += dfs6218(i + 1, diff + d * (-2) + 1, isLimit && d == up, true);
        }
        if (!isLimit && isNum) {
            memo6218[i][diff] = res;
        }
        return res;
    }

    // P1048[NOIP2005 普及组]采药
    private int t1048;
    private int[][] memo1048;
    private int n1048;
    private int[][] herbs1048;

    public int gatherHerbs(int[][] herbs, int t) {
        this.t1048 = t;
        this.n1048 = herbs.length;
        this.memo1048 = new int[n1048][t + 1];
        for (int i = 0; i < n1048; ++i) {
            Arrays.fill(memo1048[i], -1);
        }
        this.herbs1048 = herbs;
        return dfs1048(0, 0);

    }

    private int dfs1048(int i, int j) {
        if (i == n1048) {
            return 0;
        }
        if (memo1048[i][j] != -1) {
            return memo1048[i][j];
        }
        int res = dfs1048(i + 1, j);
        if (j + herbs1048[i][0] <= t1048) {
            res = Math.max(res, dfs1048(i + 1, j + herbs1048[i][0]) + herbs1048[i][1]);
        }
        return memo1048[i][j] = res;
    }

    // P2657 [SCOI2009] windy 数
    public int windyNumbers(int a, int b) {
        return solve2657(String.valueOf(b)) - solve2657(String.valueOf(a - 1));
    }

    private int n2657;
    private char[] arr2657;
    private int[][] memo2657;

    private int solve2657(String s) {
        this.n2657 = s.length();
        this.arr2657 = s.toCharArray();
        this.memo2657 = new int[n2657][10];
        for (int i = 0; i < n2657; ++i) {
            Arrays.fill(memo2657[i], -1);
        }
        return dfs2657(0, 0, true, false);
    }

    private int dfs2657(int i, int j, boolean isLimit, boolean isNum) {
        if (i == n2657) {
            return isNum ? 1 : 0;
        }
        if (!isLimit && isNum && memo2657[i][j] != -1) {
            return memo2657[i][j];
        }
        int res = 0;
        if (!isNum) {
            res = dfs2657(i + 1, j, false, false);
        }
        int up = isLimit ? arr2657[i] - '0' : 9;
        for (int d = isNum ? 0 : 1; d <= up; ++d) {
            if (!isNum || Math.abs(d - j) >= 2) {
                res += dfs2657(i + 1, d, isLimit && d == up, true);
            }
        }
        if (!isLimit && isNum) {
            memo2657[i][j] = res;
        }
        return memo2657[i][j] = res;
    }

    // P4317 花神的数论题
    private String s4317;
    private int n4317;
    private int[][] memo4317;

    public int flowerGodNumTheory(long N) {
        this.s4317 = Long.toBinaryString(N);
        this.n4317 = s4317.length();
        this.memo4317 = new int[n4317][n4317];
        for (int i = 0; i < n4317; ++i) {
            Arrays.fill(memo4317[i], -1);
        }
        return dfs4317(0, 0, true, false);
    }

    private int dfs4317(int i, int j, boolean isLimit, boolean isNum) {
        if (i == n4317) {
            return isNum ? j : 1;
        }
        if (!isLimit && isNum && memo4317[i][j] != -1) {
            return memo4317[i][j];
        }
        final int MOD = (int) (1e7 + 7);
        int res = 1;
        if (!isNum) {
            res = (int) (((long) res * dfs4317(i + 1, j, false, false)) % MOD);
        }
        int up = isLimit ? s4317.charAt(i) - '0' : 1;
        for (int d = isNum ? 0 : 1; d <= up; ++d) {
            res = (int) ((long) res * dfs4317(i + 1, j + d, isLimit && d == up, true) % MOD);
        }
        if (!isLimit && isNum) {
            memo4317[i][j] = res;
        }
        return res;
    }

    // P1060[NOIP2006 普及组]开心的金明
    private int n1060;
    private int m1060;
    private int[][] commodity1060;
    private int[][] memo1060;

    public int happyJinMing(int n, int[][] commodity) {
        this.n1060 = n;
        this.m1060 = commodity.length;
        this.commodity1060 = commodity;
        this.memo1060 = new int[m1060][n + 1];
        for (int i = 0; i < m1060; ++i) {
            Arrays.fill(memo1060[i], -1);
        }
        return dfs1060(0, 0);

    }

    private int dfs1060(int i, int j) {
        if (i == m1060) {
            return 0;
        }
        if (memo1060[i][j] != -1) {
            return memo1060[i][j];
        }
        int res = dfs1060(i + 1, j);
        if (j + commodity1060[i][0] <= n1060) {
            res = Math.max(res, dfs1060(i + 1, j + commodity1060[i][0]) + commodity1060[i][0] * commodity1060[i][1]);
        }
        return memo1060[i][j] = res;
    }

    // P7859[COCI2015-2016#2]GEPPETTO
    public int GEPPETTO(int n, int[][] conflict) {
        int res = 0;
        search: for (int i = 0; i < 1 << n; ++i) {
            for (int[] c : conflict) {
                if (((i >> (c[0] - 1)) & 1) != 0 && ((i >> (c[1] - 1)) & 1) != 0) {
                    continue search;
                }
            }
            ++res;
        }
        return res;
    }

    // P8687 [蓝桥杯 2019 省 A] 糖果
    private int[] bitMask8687;
    private int n8687;
    private int[][] memo8687;
    private int u8687;

    public int candy(int m, int[][] candy) {
        this.n8687 = candy.length;
        this.bitMask8687 = new int[n8687];
        for (int i = 0; i < n8687; ++i) {
            for (int c : candy[i]) {
                bitMask8687[i] |= 1 << (c - 1);
            }
        }
        this.u8687 = (1 << m) - 1;
        this.memo8687 = new int[n8687][1 << m];
        for (int i = 0; i < n8687; ++i) {
            Arrays.fill(memo8687[i], -1);
        }
        int res = dfs8687(0, 0);
        return res <= n8687 ? res : -1;

    }

    private int dfs8687(int i, int j) {
        if (j == u8687) {
            return 0;
        }
        if (i == n8687) {
            return n8687 + 1;
        }
        if (memo8687[i][j] != -1) {
            return memo8687[i][j];
        }
        return memo8687[i][j] = Math.min(dfs8687(i + 1, j), dfs8687(i + 1, j | bitMask8687[i]) + 1);
    }

    // P1164 小A点菜
    private int[][] memo1164;
    private int m1164;
    private int n1164;
    private int[] dishes1164;

    public int orderDishes(int m, int[] dishes) {
        this.n1164 = dishes.length;
        this.m1164 = m;
        this.dishes1164 = dishes;
        this.memo1164 = new int[n1164][m + 1];
        for (int i = 0; i < n1164; ++i) {
            Arrays.fill(memo1164[i], -1);
        }
        return dfs1164(0, 0);

    }

    private int dfs1164(int i, int j) {
        if (j > m1164) {
            return 0;
        }
        if (i == n1164) {
            return j == m1164 ? 1 : 0;
        }
        if (memo1164[i][j] != -1) {
            return memo1164[i][j];
        }
        return memo1164[i][j] = dfs1164(i + 1, j) + dfs1164(i + 1, j + dishes1164[i]);
    }

}
