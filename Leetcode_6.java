import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;

public class Leetcode_6 {
    public static void main(String[] args) {
        // String[] strs = { "1.500", "2.500", "3.500" };
        // String s = minimizeError(strs, 9);

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

    // 1151. 最少交换次数来组合所有的 1 (Minimum Swaps to Group All 1's Together) --plus
    public int minSwaps(int[] data) {
        int n = data.length;
        int k = 0;
        for (int d : data) {
            k += d;
        }
        int cur = 0;
        for (int i = 0; i < k; ++i) {
            if (data[i] == 0) {
                ++cur;
            }
        }
        int res = cur;
        for (int i = k; i < n; ++i) {
            if (data[i] == 0) {
                ++cur;
            }
            if (data[i - k] == 0) {
                --cur;
            }
            res = Math.min(res, cur);
        }
        return res;

    }

    // 261. 以图判树 (Graph Valid Tree) --plus 是否有多个连通分量 (若无重边或自环，则只需判断「节点个数 - 1 ==
    // 边的条数」)、是否有环
    public boolean validTree(int n, int[][] edges) {
        return n - 1 == edges.length && checkCycle261(n, edges);

    }

    private boolean checkCycle261(int n, int[][] edges) {
        if (n == 1) {
            return true;
        }
        Map<Integer, List<Integer>> graph = new HashMap<>();
        int[] indegrees = new int[n];
        for (int[] edge : edges) {
            graph.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(edge[1]);
            graph.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(edge[0]);
            ++indegrees[edge[0]];
            ++indegrees[edge[1]];
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < n; ++i) {
            if (--indegrees[i] == 0) {
                queue.offer(i);
            }
        }
        while (!queue.isEmpty()) {
            int node = queue.poll();
            --n;
            for (int neighbor : graph.getOrDefault(node, new ArrayList<>())) {
                --indegrees[neighbor];
                if (indegrees[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }
        return n == 0;
    }

    // 285. 二叉搜索树中的中序后继 (Inorder Successor in BST) --plus
    private boolean flag285;
    private TreeNode res285;

    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        dfs285(root, p);
        return res285;

    }

    private void dfs285(TreeNode node, TreeNode p) {
        if (node == null) {
            return;
        }
        dfs285(node.left, p);
        if (flag285 && res285 == null) {
            res285 = node;
        }
        if (node == p) {
            flag285 = true;
        }
        dfs285(node.right, p);
    }

    // 288. 单词的唯一缩写 (Unique Word Abbreviation) --plus
    class ValidWordAbbr {
        private Map<String, String> map;

        public ValidWordAbbr(String[] dictionary) {
            map = new HashMap<>();
            for (String dic : dictionary) {
                int n = dic.length();
                String abbr = n <= 2 ? dic : dic.charAt(0) + String.valueOf(n - 2) + dic.charAt(n - 1);
                if (!map.containsKey(abbr)) {
                    map.put(abbr, dic);
                } else if (!map.get(abbr).equals(dic)) {
                    map.put(abbr, "_");
                }
            }
        }

        public boolean isUnique(String word) {
            int n = word.length();
            String abbr = n <= 2 ? word : word.charAt(0) + String.valueOf(n - 2) + word.charAt(n - 1);
            return !map.containsKey(abbr) || map.get(abbr).equals(word);

        }
    }

    // 1058. 最小化舍入误差以满足目标 (Minimize Rounding Error to Meet Target) --plus
    public String minimizeError(String[] prices, int target) {
        int min = 0;
        int max = 0;
        List<Integer> trans = new ArrayList<>();
        for (String price : prices) {
            int dot = price.indexOf(".");
            int tran = Integer.parseInt(price.substring(0, dot)) * 1000
                    + Integer.parseInt(price.substring(dot + 1));
            min += tran / 1000;
            max += tran / 1000 + (tran % 1000 == 0 ? 0 : 1);
            trans.add(tran);
        }
        if (min > target || max < target) {
            return "-1";
        }
        Collections.sort(trans, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return o2 % 1000 - o1 % 1000;

            }

        });

        int n = prices.length;
        int res = 0;
        for (int i = 0; i < n; ++i) {
            int num = trans.get(i);
            if (i < target - min) {
                res += (num / 1000 + 1) * 1000 - num;
            } else {
                res += num - num / 1000 * 1000;
            }
        }
        return String.format("%.3f", (double) res / 1000);

    }

    // 624. 数组列表中的最大距离 (Maximum Distance in Arrays) --plus
    public int maxDistance(List<List<Integer>> arrays) {
        int max1 = Integer.MIN_VALUE;
        int max2 = Integer.MIN_VALUE;
        int min1 = Integer.MAX_VALUE;
        int min2 = Integer.MAX_VALUE;
        int maxRowIndex = -1;
        int minRowIndex = -1;
        for (int i = 0; i < arrays.size(); ++i) {
            if (arrays.get(i) == null || arrays.get(i).isEmpty()) {
                continue;
            }
            int curMax = arrays.get(i).get(arrays.get(i).size() - 1);
            if (curMax >= max1) {
                max2 = max1;
                max1 = curMax;
                maxRowIndex = i;
            } else if (curMax >= max2) {
                max2 = curMax;
            }
            int curMin = arrays.get(i).get(0);
            if (curMin <= min1) {
                min2 = min1;
                min1 = curMin;
                minRowIndex = i;
            } else if (curMin <= min2) {
                min2 = curMin;
            }
        }
        if (maxRowIndex != minRowIndex) {
            return max1 - min1;
        }
        return Math.max(Math.abs(max1 - min2), Math.abs(max2 - min1));

    }

    // 2371. Minimize Maximum Value in a Grid
    // public int[][] minScore(int[][] grid) {

    // }

    // 1648. 销售价值减少的颜色球 (Sell Diminishing-Valued Colored Balls)
    // public int maxProfit(int[] inventory, int orders) {

    // }

    // 1562. 查找大小为 M 的最新分组 (Find Latest Group of Size M)
    // public int findLatestStep(int[] arr, int m) {

    // }

    // 813. 最大平均值和的分组 (Largest Sum of Averages)
    // public double largestSumOfAverages(int[] nums, int k) {

    // }

    // 2484. 统计回文子序列数目 (Count Palindromic Subsequences)
    // public int countPalindromes(String s) {

    // }

    // 2488. 统计中位数为 K 的子数组 (Count Subarrays With Median K)
    // public int countSubarrays(int[] nums, int k) {

    // }

    // 2467. 树上最大得分和路径 (Most Profitable Path in a Tree)
    // public int mostProfitablePath(int[][] edges, int bob, int[] amount) {

    // }

    // 2466. 统计构造好字符串的方案数 (Count Ways To Build Good Strings)
    // public int countGoodStrings(int low, int high, int zero, int one) {

    // }

}
