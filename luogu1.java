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
}
