import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@SuppressWarnings("unchecked")
public class luogu1 {
    public static void main(String[] args) {

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
            return isNum ? 1 : 0;
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
}
