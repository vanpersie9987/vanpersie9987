import java.text.DecimalFormat;
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
import java.util.Random;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@SuppressWarnings("unchecked")
public class LeetCode_4 {
    public static void main(String[] args) {
        // String[] strings = { "mobile", "mouse", "moneypot", "monitor", "mousepad" };
        // suggestedProducts(strings, "mouse");
        // int res = nextGreaterElement(13);

    }

    // 6074. 字母在字符串中的百分比
    public int percentageLetter(String s, char letter) {
        int count = 0;
        for (char c : s.toCharArray()) {
            if (c == letter) {
                ++count;
            }
        }
        double percentage = (double) count / s.length();
        return (int) (percentage * 100);

    }

    // 6075. 装满石头的背包的最大数量
    public int maximumBags(int[] capacity, int[] rocks, int additionalRocks) {
        int res = 0;
        Queue<Integer> queue = new PriorityQueue<>();
        int n = rocks.length;
        for (int i = 0; i < n; ++i) {
            queue.offer(capacity[i] - rocks[i]);
        }
        while (!queue.isEmpty() && queue.peek() <= additionalRocks) {
            ++res;
            int cur = queue.poll();
            additionalRocks -= cur;
        }
        return res;

    }

    // 6076. 表示一个折线图的最少线段数
    public int minimumLines(int[][] stockPrices) {
        int n = stockPrices.length;
        if (n == 1) {
            return 0;
        }
        Arrays.sort(stockPrices, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }

        });

        int res = 1;
        // 分子
        int den = stockPrices[1][1] - stockPrices[0][1];
        // 分母
        int num = stockPrices[1][0] - stockPrices[0][0];
        // 符号 0 == 0 1 == 正号 2 == 负号
        int sign = 0;
        if (den == 0) {
            sign = 0;
        } else if ((den > 0 && num > 0) || (den < 0 && num < 0)) {
            sign = 1;
        } else {
            sign = 2;
        }
        den = Math.abs(den);
        num = Math.abs(num);
        int gcd = getGCD(den, num);
        den /= gcd;
        num /= gcd;
        for (int i = 2; i < stockPrices.length; ++i) {
            // 分子
            int curDen = stockPrices[i][1] - stockPrices[i - 1][1];
            // 分母
            int curNum = stockPrices[i][0] - stockPrices[i - 1][0];
            int curSign = 0;

            if (curDen == 0) {
                curSign = 0;
            } else if ((curDen > 0 && curNum > 0) || (curDen < 0 && curNum < 0)) {
                curSign = 1;
            } else {
                curSign = 2;
            }

            curDen = Math.abs(curDen);
            curNum = Math.abs(curNum);
            int curGcd = getGCD(curDen, curNum);
            curDen /= curGcd;
            curNum /= curGcd;
            if (!((curDen == 0 && den == 0) || (curSign == sign && curDen == den && curNum == num))) {
                ++res;
            }
            sign = curSign;
            den = curDen;
            num = curNum;

        }
        return res;

    }

    // 计算最大公约数
    private int getGCD(int a, int b) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
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

    // 110. 平衡二叉树 (Balanced Binary Tree) --dfs 自底向上
    // 剑指 Offer 55 - II. 平衡二叉树
    // 面试题 04.04. 检查平衡性 (Check Balance LCCI)
    private boolean flag110 = true;

    public boolean isBalanced(TreeNode root) {
        dfs110(root);
        return flag110;

    }

    private int dfs110(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int l = dfs110(root.left);
        int r = dfs110(root.right);
        if (Math.abs(l - r) > 1) {
            flag110 = false;
        }
        return Math.max(l, r) + 1;
    }

    // 110. 平衡二叉树 (Balanced Binary Tree) --dfs 自顶向下
    // 剑指 Offer 55 - II. 平衡二叉树
    // 面试题 04.04. 检查平衡性 (Check Balance LCCI)
    public boolean isBalanced2(TreeNode root) {
        if (root == null) {
            return true;
        }
        return isBalanced2(root.left) && isBalanced2(root.right)
                && Math.abs(height110(root.left) - height110(root.right)) <= 1;
    }

    private int height110(TreeNode node) {
        if (node == null) {
            return 0;
        }
        return Math.max(height110(node.left), height110(node.right)) + 1;
    }

    // 2235. 两整数相加 (Add Two Integers)
    public int sum(int num1, int num2) {
        return num1 + num2;
    }

    // 2235. 两整数相加 (Add Two Integers)
    public int sum2(int num1, int num2) {
        while (num2 != 0) {
            int carry = (num1 & num2) << 1;
            num1 ^= num2;
            num2 = carry;
        }
        return num1;

    }

    // 2236. 判断根结点是否等于子结点之和 (Root Equals Sum of Children)
    public boolean checkTree(TreeNode root) {
        return root.val == root.left.val + root.right.val;
    }

    // 1557. 可以到达所有点的最少点数目 (Minimum Number of Vertices to Reach All Nodes) --拓扑排序
    public List<Integer> findSmallestSetOfVertices(int n, List<List<Integer>> edges) {
        int[] inDegrees = new int[n];
        for (List<Integer> edge : edges) {
            ++inDegrees[edge.get(1)];
        }
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            if (inDegrees[i] == 0) {
                res.add(i);
            }
        }
        return res;

    }

    // 2119. 反转两次的数字 (A Number After a Double Reversal)
    public boolean isSameAfterReversals(int num) {
        return num == 0 || num % 10 != 0;
    }

    // 2169. 得到 0 的操作数 (Count Operations to Obtain Zero)
    public int countOperations(int num1, int num2) {
        int res = 0;
        while (num1 != 0 && num2 != 0) {
            if (num1 > num2) {
                num1 -= num2;
            } else {
                num2 -= num1;
            }
            ++res;
        }
        return res;

    }

    // 2169. 得到 0 的操作数 (Count Operations to Obtain Zero) --辗转相除
    public int countOperations2(int num1, int num2) {
        int res = 0;
        while (num1 != 0 && num2 != 0) {
            res += num1 / num2;
            num1 %= num2;
            int temp = num1;
            num1 = num2;
            num2 = temp;
        }
        return res;
    }

    // 129. 求根节点到叶节点数字之和 (Sum Root to Leaf Numbers) --bfs
    // 剑指 Offer II 049. 从根节点到叶节点的路径数字之和
    public int sumNumbers(TreeNode root) {
        int res = 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node.left == null && node.right == null) {
                res += node.val;
                continue;
            }
            if (node.left != null) {
                node.left.val += node.val * 10;
                queue.offer(node.left);
            }
            if (node.right != null) {
                node.right.val += node.val * 10;
                queue.offer(node.right);
            }

        }
        return res;

    }

    // 129. 求根节点到叶节点数字之和 (Sum Root to Leaf Numbers) --dfs
    // 剑指 Offer II 049. 从根节点到叶节点的路径数字之和
    private int res129;

    public int sumNumbers2(TreeNode root) {
        dfs129(root);
        return res129;

    }

    private void dfs129(TreeNode root) {
        if (root == null) {
            return;
        }
        if (root.left == null && root.right == null) {
            res129 += root.val;
        }
        if (root.left != null) {
            root.left.val += root.val * 10;
            dfs129(root.left);
        }
        if (root.right != null) {
            root.right.val += root.val * 10;
            dfs129(root.right);
        }
    }

    // 114. 二叉树展开为链表 (Flatten Binary Tree to Linked List) --dfs + 显式栈
    public void flatten(TreeNode root) {
        List<TreeNode> list = new ArrayList<>();
        dfs114(root, list);
        for (int i = 1; i < list.size(); ++i) {
            TreeNode pre = list.get(i - 1);
            TreeNode cur = list.get(i);
            pre.right = cur;
            pre.left = null;
        }

    }

    private void dfs114(TreeNode root, List<TreeNode> list) {
        if (root == null) {
            return;
        }
        list.add(root);
        dfs114(root.left, list);
        dfs114(root.right, list);
    }

    // 114. 二叉树展开为链表 (Flatten Binary Tree to Linked List) --隐式栈
    public void flatten2(TreeNode root) {
        List<TreeNode> list = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode node = root;
        while (!stack.isEmpty() || node != null) {
            while (node != null) {
                list.add(node);
                stack.push(node);
                node = node.left;
            }
            node = stack.pop();
            node = node.right;
        }
        for (int i = 1; i < list.size(); ++i) {
            TreeNode pre = list.get(i - 1);
            pre.left = null;
            pre.right = list.get(i);
        }
    }

    // 114. 二叉树展开为链表 (Flatten Binary Tree to Linked List) --展开和迭代同步进行
    public void flatten3(TreeNode root) {
        if (root == null) {
            return;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        TreeNode pre = null;
        while (!stack.isEmpty()) {
            TreeNode cur = stack.pop();
            if (pre != null) {
                pre.right = cur;
                pre.left = null;
            }
            pre = cur;
            if (cur.right != null) {
                stack.push(cur.right);
            }
            if (cur.left != null) {
                stack.push(cur.left);
            }
        }

    }

    // 124. 二叉树中的最大路径和 (Binary Tree Maximum Path Sum) --dfs
    // 剑指 Offer II 051. 节点之和最大的路径
    private int res124;

    public int maxPathSum(TreeNode root) {
        res124 = Integer.MIN_VALUE;
        dfs124(root);
        return res124;

    }

    private int dfs124(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = Math.max(0, dfs124(root.left));
        int right = Math.max(0, dfs124(root.right));
        res124 = Math.max(res124, left + right + root.val);
        return Math.max(left, right) + root.val;
    }

    // 200. 岛屿数量 (Number of Islands) --dfs
    public int numIslands(char[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == '1') {
                    ++res;
                    dfs200(grid, i, j);
                }
            }
        }
        return res;

    }

    private void dfs200(char[][] grid, int i, int j) {
        int m = grid.length;
        int n = grid[0].length;
        if (i >= 0 && i < m && j >= 0 && j < n && grid[i][j] == '1') {
            grid[i][j] = '0';
            dfs200(grid, i - 1, j);
            dfs200(grid, i + 1, j);
            dfs200(grid, i, j - 1);
            dfs200(grid, i, j + 1);
        }
    }

    // 200. 岛屿数量 (Number of Islands) --并查集
    public int numIslands2(char[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        Union200 union = new Union200(grid);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == '1') {
                    if (i - 1 >= 0 && grid[i - 1][j] == '1') {
                        union.union(getIndex200(n, i, j), getIndex200(n, i - 1, j));
                    }
                    if (j - 1 >= 0 && grid[i][j - 1] == '1') {
                        union.union(getIndex200(n, i, j), getIndex200(n, i, j - 1));
                    }
                }
            }
        }
        return union.getCount();

    }

    private int getIndex200(int n, int i, int j) {
        return n * i + j;
    }

    public class Union200 {
        private int[] parent;
        private int[] rank;
        private int count;

        public Union200(char[][] grid) {
            int m = grid.length;
            int n = grid[0].length;
            parent = new int[m * n];
            rank = new int[m * n];
            Arrays.fill(rank, 1);
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (grid[i][j] == '1') {
                        parent[i * n + j] = i * n + j;
                        ++count;
                    }
                }
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
            --count;
        }

        public int getCount() {
            return count;
        }
    }

    // 200. 岛屿数量 (Number of Islands) --bfs
    public int numIslands3(char[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == '1') {
                    grid[i][j] = '0';
                    Queue<int[]> queue = new LinkedList<>();
                    queue.offer(new int[] { i, j });
                    while (!queue.isEmpty()) {
                        int[] cur = queue.poll();
                        for (int[] direction : directions) {
                            int nx = cur[0] + direction[0];
                            int ny = cur[1] + direction[1];
                            if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                                if (grid[nx][ny] == '1') {
                                    grid[nx][ny] = '0';
                                    queue.offer(new int[] { nx, ny });
                                }
                            }
                        }
                    }
                    ++res;
                }
            }
        }
        return res;

    }

    // 1104. 二叉树寻路 (Path In Zigzag Labelled Binary Tree) --位运算
    public List<Integer> pathInZigZagTree(int label) {
        int row = 1;
        int rowStart = 1;
        while ((rowStart << 1) <= label) {
            ++row;
            rowStart <<= 1;
        }
        if ((row & 1) == 0) {
            label = getReverse1104(row, label);
        }
        List<Integer> res = new ArrayList<>();
        while (row > 0) {
            if ((row & 1) == 0) {
                res.add(getReverse1104(row, label));
            } else {
                res.add(label);
            }
            --row;
            label >>= 1;
        }
        Collections.reverse(res);
        return res;

    }

    private int getReverse1104(int row, int label) {
        return (1 << (row - 1)) + ((1 << row) - 1) - label;
    }

    // 1419. 数青蛙 (Minimum Number of Frogs Croaking)
    public int minNumberOfFrogs(String croakOfFrogs) {
        int[] cnts = new int[5];
        for (char ch : croakOfFrogs.toCharArray()) {
            switch (ch) {
                case 'c':
                    if (cnts[4] > 0) {
                        --cnts[4];
                    }
                    ++cnts[0];
                    break;
                case 'r':
                    if (cnts[0] > 0) {
                        --cnts[0];
                    } else {
                        return -1;
                    }
                    ++cnts[1];
                    break;
                case 'o':
                    if (cnts[1] > 0) {
                        --cnts[1];
                    } else {
                        return -1;
                    }
                    ++cnts[2];
                    break;
                case 'a':
                    if (cnts[2] > 0) {
                        --cnts[2];
                    } else {
                        return -1;
                    }
                    ++cnts[3];
                    break;
                case 'k':
                    if (cnts[3] > 0) {
                        --cnts[3];
                    } else {
                        return -1;
                    }
                    ++cnts[4];
                    break;

                default:
                    break;
            }
        }
        if (cnts[0] > 0 || cnts[1] > 0 || cnts[2] > 0 || cnts[3] > 0) {
            return -1;
        }
        return cnts[4];

    }

    // 6083. 判断一个数的数字计数是否等于数位的值
    public boolean digitCount(String num) {
        int[] counts = new int[10];
        for (char c : num.toCharArray()) {
            ++counts[c - '0'];
        }
        for (int i = 0; i < num.length(); ++i) {
            int c = num.charAt(i) - '0';

            if (c != counts[i]) {
                return false;
            }
        }
        return true;

    }

    // 6084. 最多单词数的发件人
    public String largestWordCount(String[] messages, String[] senders) {
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < senders.length; ++i) {
            int count = getWords(messages[i]);
            map.put(senders[i], map.getOrDefault(senders[i], 0) + count);
        }
        String res = "";
        int max = 0;
        for (Map.Entry<String, Integer> entry : map.entrySet()) {
            if (entry.getValue() > max) {
                max = entry.getValue();
                res = entry.getKey();
            } else if (entry.getValue() == max && entry.getKey().compareTo(res) > 0) {
                res = entry.getKey();
            }
        }
        return res;

    }

    private int getWords(String string) {
        int count = 1;
        for (char c : string.toCharArray()) {
            if (c == ' ') {
                ++count;
            }
        }
        return count;
    }

    // 6085. 道路的最大总重要性
    public long maximumImportance(int n, int[][] roads) {
        int[] degrees = new int[n];
        for (int[] road : roads) {
            ++degrees[road[0]];
            ++degrees[road[1]];
        }
        long res = 0l;
        Arrays.sort(degrees);
        for (int i = n - 1; i >= 0; --i) {
            res += (long) degrees[i] * (i + 1);
        }
        return res;

    }

    // 6078. 重排字符形成目标字符串
    public int rearrangeCharacters(String s, String target) {
        int[] counts = new int[26];
        for (char c : target.toCharArray()) {
            ++counts[c - 'a'];
        }
        int[] counts2 = new int[26];
        for (char c : s.toCharArray()) {
            ++counts2[c - 'a'];
        }
        int res = Integer.MAX_VALUE;
        for (int i = 0; i < 26; ++i) {
            if (counts[i] > 0) {
                res = Math.min(res, counts2[i] / counts[i]);
            }
        }
        return res;

    }

    // 2288. 价格减免 (Apply Discount to Prices)
    public String discountPrices(String sentence, int discount) {
        String[] arr = sentence.split(" ");
        for (int i = 0; i < arr.length; ++i) {
            if (arr[i].charAt(0) == '$' && checkAllDigit(arr[i].substring(1))) {
                long d = Long.valueOf(arr[i].substring(1));
                double c = d - d * discount * 0.01D;
                DecimalFormat df = new DecimalFormat("#0.00");
                arr[i] = "$" + df.format(c);
            }
        }
        return String.join(" ", arr);

    }

    private boolean checkAllDigit(String s) {
        for (char c : s.toCharArray()) {
            if (!Character.isDigit(c)) {
                return false;
            }
        }
        return !s.isEmpty();
    }

    // 2290. 到达角落需要移除障碍物的最小数目 (Minimum Obstacle Removal to Reach Corner)
    public int minimumObstacles(int[][] grid) {
        int[][] directions = { { 0, -1 }, { 1, 0 }, { -1, 0 }, { 0, 1 } };
        int m = grid.length;
        int n = grid[0].length;
        int[][] dis = new int[m][n];
        for (int i = 0; i < m; ++i) {
            Arrays.fill(dis[i], Integer.MAX_VALUE);
        }
        dis[0][0] = 0;
        Queue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[2] - o2[2];
            }

        });
        q.offer(new int[] { 0, 0, 0 });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int y = cur[1];
            int curDis = cur[2];
            if (x == m - 1 && y == n - 1) {
                return curDis;
            }
            if (curDis > dis[x][y]) {
                continue;
            }
            for (int[] d : directions) {
                int nx = x + d[0];
                int ny = y + d[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    int newDis = grid[nx][ny] == 0 ? curDis : curDis + 1;
                    if (newDis < dis[nx][ny]) {
                        dis[nx][ny] = newDis;
                        q.offer(new int[] { nx, ny, newDis });
                    }
                }
            }
        }
        return dis[m - 1][n - 1];

    }

    // 2290. 到达角落需要移除障碍物的最小数目 (Minimum Obstacle Removal to Reach Corner) -- 0-1 bfs
    public int minimumObstacles2(int[][] grid) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int m = grid.length;
        int n = grid[0].length;
        int[][] dis = new int[m][n];
        for (int i = 0; i < m; ++i) {
            Arrays.fill(dis[i], m + n - 1);
        }
        dis[0][0] = 0;
        Deque<int[]> deque = new ArrayDeque<>();
        deque.offer(new int[] { 0, 0 });
        while (!deque.isEmpty()) {
            int[] cur = deque.pollFirst();
            int x = cur[0];
            int y = cur[1];
            int d = dis[x][y];
            if (x == m - 1 && y == n - 1) {
                return d;
            }
            for (int[] direction : directions) {
                int nx = x + direction[0];
                int ny = y + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    int newD = d + grid[nx][ny];
                    if (newD < dis[nx][ny]) {
                        dis[nx][ny] = newD;
                        if (grid[nx][ny] == 0) {
                            deque.offerFirst(new int[] { nx, ny, newD });
                        } else {
                            deque.offerLast(new int[] { nx, ny, newD });
                        }
                    }
                }
            }
        }
        return -1;

    }

    // 208. 实现 Trie (前缀树) (Implement Trie (Prefix Tree))
    // 剑指 Offer II 062. 实现前缀树
    class Trie {
        private Trie[] children;
        private boolean isEnd;

        public Trie() {
            children = new Trie[26];
            isEnd = false;
        }

        public void insert(String word) {
            Trie node = this;
            for (char c : word.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new Trie();
                }
                node = node.children[index];
            }
            node.isEnd = true;
        }

        public boolean search(String word) {
            Trie node = searchPrefix(word);
            return node != null && node.isEnd;

        }

        private Trie searchPrefix(String word) {
            Trie node = this;
            for (char c : word.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    return null;
                }
                node = node.children[index];
            }
            return node;
        }

        public boolean startsWith(String prefix) {
            Trie node = searchPrefix(prefix);
            return node != null;
        }
    }

    // 211. 添加与搜索单词 - 数据结构设计 (Design Add and Search Words Data Structure) --dfs Trie
    class WordDictionary {
        private Trie211 trie;

        public WordDictionary() {
            trie = new Trie211();
        }

        public void addWord(String word) {
            trie.addWord(word);
        }

        public boolean search(String word) {
            return dfs211(word, 0, trie);
        }

        private boolean dfs211(String word, int index, Trie211 node) {
            if (index == word.length()) {
                return node.isEnd();
            }
            char c = word.charAt(index);
            if (Character.isLetter(c)) {
                Trie211 childNode = node.getChildren()[c - 'a'];
                if (childNode != null && dfs211(word, index + 1, childNode)) {
                    return true;
                }
            } else {
                for (Trie211 child : node.getChildren()) {
                    if (child != null && dfs211(word, index + 1, child)) {
                        return true;
                    }
                }
            }
            return false;
        }
    }

    class Trie211 {
        private Trie211[] children;
        private boolean isEnd;

        public Trie211() {
            children = new Trie211[26];
            isEnd = false;
        }

        public void addWord(String word) {
            Trie211 node = this;
            for (char c : word.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new Trie211();
                }
                node = node.children[index];
            }
            node.isEnd = true;
        }

        public Trie211[] getChildren() {
            return children;
        }

        public boolean isEnd() {
            return isEnd;
        }

    }

    // 1792. 最大平均通过率 (Maximum Average Pass Ratio) --贪心
    public double maxAverageRatio(int[][] classes, int extraStudents) {
        Queue<double[]> queue = new PriorityQueue<>(new Comparator<double[]>() {

            @Override
            public int compare(double[] o1, double[] o2) {
                Double x = (o1[0] + 1) / (o1[1] + 1) - o1[0] / o1[1];
                Double y = (o2[0] + 1) / (o2[1] + 1) - o2[0] / o2[1];
                return y.compareTo(x);
            }

        });
        for (int[] c : classes) {
            queue.offer(new double[] { c[0], c[1] });
        }

        while (extraStudents-- > 0) {
            double[] c = queue.poll();
            ++c[0];
            ++c[1];
            queue.offer(c);
        }
        double res = 0d;
        while (!queue.isEmpty()) {
            double[] c = queue.poll();
            res += c[0] / c[1];
        }
        return res / classes.length;

    }

    // 435. 无重叠区间 (Non-overlapping Intervals) --贪心
    public int eraseOverlapIntervals(int[][] intervals) {
        Arrays.sort(intervals, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[1] - o2[1];
            }

        });
        int right = intervals[0][1];
        int res = 1;
        for (int i = 1; i < intervals.length; ++i) {
            if (intervals[i][0] >= right) {
                ++res;
                right = intervals[i][1];
            }
        }
        return intervals.length - res;

    }

    // 435. 无重叠区间 (Non-overlapping Intervals)
    private Map<Integer, List<Integer>> g435;
    private int min435;
    private int max435;
    private Map<Integer, Integer> memo435;

    public int eraseOverlapIntervals2(int[][] intervals) {
        this.g435 = new HashMap<>();
        this.min435 = Integer.MAX_VALUE;
        this.max435 = Integer.MIN_VALUE;
        for (int[] i : intervals) {
            min435 = Math.min(min435, i[0]);
            max435 = Math.max(max435, i[1]);
            g435.computeIfAbsent(i[0], k -> new ArrayList<>()).add(i[1]);
        }
        this.memo435 = new HashMap<>();
        return intervals.length - dfs435(min435);
    }

    private int dfs435(int i) {
        if (i == max435) {
            return 0;
        }
        if (memo435.get(i) != null) {
            return memo435.get(i);
        }
        int res = dfs435(i + 1);
        for (int y : g435.getOrDefault(i, new ArrayList<>())) {
            res = Math.max(res, dfs435(y) + 1);
        }
        memo435.put(i, res);
        return res;
    }

    // 1775. 通过最少操作次数使数组的和相等 (Equal Sum Arrays With Minimum Number of Operations)
    public int minOperations(int[] nums1, int[] nums2) {
        int n1 = nums1.length;
        int n2 = nums2.length;
        if (n1 * 6 < n2 || n2 * 6 < n1) {
            return -1;
        }
        int d = 0;
        for (int num : nums1) {
            d -= num;
        }
        for (int num : nums2) {
            d += num;
        }
        if (d < 0) {
            d = -d;
            int[] temp = nums1;
            nums1 = nums2;
            nums2 = temp;
        }
        int[] count = new int[6];
        for (int num : nums1) {
            ++count[6 - num];
        }
        for (int num : nums2) {
            ++count[num - 1];
        }
        int res = 0;
        for (int i = 5; i >= 1; --i) {
            if (count[i] * i >= d) {
                return res + (d + i - 1) / i;
            }
            res += count[i];
            d -= count[i] * i;
        }
        return -1;
    }

    // 648. 单词替换 (Replace Words) --字典树
    // 剑指 Offer II 063. 替换单词
    public String replaceWords(List<String> dictionary, String sentence) {
        Trie648 trie = new Trie648();
        for (String dic : dictionary) {
            trie.insert(dic);
        }
        StringBuilder res = new StringBuilder();
        String[] strings = sentence.split(" ");
        for (String s : strings) {
            if (!res.isEmpty()) {
                res.append(" ");
            }
            String item = trie.findSmallestPrefix(s);
            if (item.isEmpty()) {
                res.append(s);
            } else {
                res.append(item);
            }
        }
        return res.toString();

    }

    class Trie648 {
        private Trie648[] children;
        private boolean isEnd;

        Trie648() {
            this.children = new Trie648[26];
            this.isEnd = false;
        }

        public void insert(String s) {
            Trie648 node = this;
            for (char c : s.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new Trie648();
                }
                node = node.children[index];
            }
            node.isEnd = true;
        }

        public String findSmallestPrefix(String s) {
            StringBuilder builder = new StringBuilder();
            Trie648 node = this;
            for (char c : s.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    return "";
                }
                builder.append(c);
                node = node.children[index];
                if (node.isEnd) {
                    return builder.toString();
                }
            }
            return "";
        }
    }

    // 648. 单词替换 (Replace Words) --哈希表
    // 剑指 Offer II 063. 替换单词
    public String replaceWords2(List<String> dictionary, String sentence) {
        Set<String> set = new HashSet<>(dictionary);
        StringBuilder res = new StringBuilder();
        for (String s : sentence.split("\\s+")) {
            String prefix = "";
            for (int i = 1; i <= s.length(); ++i) {
                prefix = s.substring(0, i);
                if (set.contains(prefix)) {
                    break;
                }
            }
            if (!res.isEmpty()) {
                res.append(" ");
            }
            res.append(prefix);
        }
        return res.toString();

    }

    // 1268. 搜索推荐系统 (Search Suggestions System) --字典树
    public List<List<String>> suggestedProducts(String[] products, String searchWord) {
        Trie1268 trie = new Trie1268();
        for (String product : products) {
            trie.insert(product);
        }
        return trie.startWith(searchWord);

    }

    class Trie1268 {
        private Trie1268[] children;
        private PriorityQueue<String> priorityQueue;

        public Trie1268() {
            children = new Trie1268[26];
            priorityQueue = new PriorityQueue<>(new Comparator<String>() {

                @Override
                public int compare(String o1, String o2) {
                    return o2.compareTo(o1);
                }

            });

        }

        public void insert(String s) {
            Trie1268 node = this;
            for (char c : s.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new Trie1268();
                }
                node = node.children[index];
                node.priorityQueue.offer(s);
                if (node.priorityQueue.size() > 3) {
                    node.priorityQueue.poll();
                }
            }
        }

        public List<List<String>> startWith(String searchWord) {
            Trie1268 node = this;
            List<List<String>> res = new ArrayList<>();
            boolean exists = true;
            for (char c : searchWord.toCharArray()) {
                List<String> sub = new ArrayList<>();
                int index = c - 'a';
                if (!exists || node.children[index] == null) {
                    exists = false;
                    res.add(sub);
                    continue;
                }
                node = node.children[index];
                while (!node.priorityQueue.isEmpty()) {
                    sub.add(node.priorityQueue.poll());
                }
                Collections.reverse(sub);
                res.add(sub);
            }
            return res;
        }
    }

    // 面试题 17.13.恢复空格 (Re-Space LCCI) --dp
    public int respace(String[] dictionary, String sentence) {
        Set<String> set = new HashSet<>(Arrays.asList(dictionary));
        int n = sentence.length();
        int[] dp = new int[n + 1];
        for (int i = 1; i <= sentence.length(); ++i) {
            dp[i] = dp[i - 1] + 1;
            for (int j = 0; j <= i; ++j) {
                if (set.contains(sentence.substring(j, i))) {
                    dp[i] = Math.min(dp[i], dp[j]);
                }
            }
        }
        return dp[n];

    }

    // 面试题 17.13.恢复空格 (Re-Space LCCI) --dp + Trie
    public int respace2(String[] dictionary, String sentence) {
        Trie17_13 trie = new Trie17_13();
        for (String dic : dictionary) {
            trie.insert(dic);
        }
        int n = sentence.length();
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; ++i) {
            dp[i] = dp[i - 1] + 1;
            for (int startPos : trie.getStartIndex(sentence, i - 1)) {
                dp[i] = Math.min(dp[i], dp[startPos]);
            }
        }
        return dp[n];

    }

    class Trie17_13 {
        private Trie17_13[] children;
        private boolean isEnd;

        public Trie17_13() {
            this.children = new Trie17_13[26];
            this.isEnd = false;
        }

        /**
         * 返回以sentence中endPos结尾的单词的起始坐标列表
         */
        public List<Integer> getStartIndex(String sentence, int endPos) {
            List<Integer> res = new ArrayList<>();
            Trie17_13 node = this;
            for (int i = endPos; i >= 0; --i) {
                int index = sentence.charAt(i) - 'a';
                if (node.children[index] == null) {
                    return res;
                }
                node = node.children[index];
                if (node.isEnd) {
                    res.add(i);
                }
            }
            return res;
        }

        public void insert(String s) {
            Trie17_13 node = this;
            for (int i = s.length() - 1; i >= 0; --i) {
                int index = s.charAt(i) - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new Trie17_13();
                }
                node = node.children[index];
            }
            node.isEnd = true;
        }
    }

    // 面试题 17.13. 恢复空格 (Re-Space LCCI) --dfs
    private String sentence17_13_2;
    private int[] memo17_13_2;
    private int n17_13_2;

    private Trie17_13_2 trie;

    public int respace3(String[] dictionary, String sentence) {
        this.sentence17_13_2 = sentence;
        this.n17_13_2 = sentence.length();
        this.memo17_13_2 = new int[n17_13_2];
        Arrays.fill(memo17_13_2, -1);
        this.trie = new Trie17_13_2();
        for (String d : dictionary) {
            trie.insert(d);
        }
        return dfs17_13_2(0);

    }

    private int dfs17_13_2(int i) {
        if (i == n17_13_2) {
            return 0;
        }
        if (memo17_13_2[i] != -1) {
            return memo17_13_2[i];
        }
        int res = dfs17_13_2(i + 1) + 1;
        for (int j : trie.getIndexes(sentence17_13_2.substring(i))) {
            res = Math.min(res, dfs17_13_2(i + j + 1));
        }
        return memo17_13_2[i] = res;
    }

    public class Trie17_13_2 {
        private Trie17_13_2[] children;
        private boolean isEnd;

        public Trie17_13_2() {
            this.children = new Trie17_13_2[26];
        }

        public void insert(String s) {
            Trie17_13_2 node = this;
            for (char c : s.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new Trie17_13_2();
                }
                node = node.children[index];
            }
            node.isEnd = true;
        }

        public List<Integer> getIndexes(String s) {
            List<Integer> res = new ArrayList<>();
            Trie17_13_2 node = this;
            for (int i = 0; i < s.length(); ++i) {
                int index = s.charAt(i) - 'a';
                if (node.children[index] == null) {
                    break;
                }
                node = node.children[index];
                if (node.isEnd) {
                    res.add(i);
                }
            }
            return res;
        }
    }

    // 面试题 17.17. 多次搜索 --字典树
    public int[][] multiSearch(String big, String[] smalls) {
        int n = smalls.length;
        Trie17_17 trie = new Trie17_17();
        for (int i = 0; i < n; ++i) {
            trie.insert(smalls[i], i);
        }
        List<List<Integer>> list = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            list.add(new ArrayList<>());
        }
        for (int i = 0; i < big.length(); ++i) {
            for (int id : trie.search(big.substring(i))) {
                list.get(id).add(i);
            }
        }
        int[][] res = new int[n][];
        for (int i = 0; i < n; ++i) {
            res[i] = list.get(i).stream().mapToInt(Integer::intValue).toArray();
        }
        return res;

    }

    class Trie17_17 {
        private Trie17_17[] children;
        private boolean isEnd;
        private int i;

        public Trie17_17() {
            this.children = new Trie17_17[26];
            this.isEnd = false;
            this.i = -1;
        }

        public void insert(String s, int pos) {
            Trie17_17 node = this;
            for (char c : s.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new Trie17_17();
                }
                node = node.children[index];
            }
            node.isEnd = true;
            node.i = pos;
        }

        public List<Integer> search(String s) {
            List<Integer> res = new ArrayList<>();
            Trie17_17 node = this;
            for (char c : s.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    break;
                }
                node = node.children[index];
                if (node.isEnd) {
                    res.add(node.i);
                }
            }
            return res;

        }

    }

    // 6090. 极大极小游戏
    public int minMaxGame(int[] nums) {
        int n = nums.length;
        while (n != 1) {
            for (int i = 0; i < (n >> 1); ++i) {
                if ((i & 1) == 0) {
                    nums[i] = Math.min(nums[i << 1], nums[(i << 1) + 1]);
                } else {
                    nums[i] = Math.max(nums[i << 1], nums[(i << 1) + 1]);
                }
            }
            n >>= 1;
        }
        return nums[0];

    }

    // 6091. 划分数组使最大差为 K
    public int partitionArray(int[] nums, int k) {
        Arrays.sort(nums);
        int res = 1;
        int min = nums[0];
        for (int i = 1; i < nums.length; ++i) {
            if (nums[i] - min > k) {
                ++res;
                min = nums[i];
            }
        }
        return res;

    }

    // 6092. 替换数组中的元素
    public int[] arrayChange(int[] nums, int[][] operations) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; ++i) {
            map.put(nums[i], i);
        }
        for (int[] operation : operations) {
            int index = map.remove(operation[0]);
            map.put(operation[1], index);
        }
        int[] res = new int[nums.length];
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            int index = entry.getValue();
            res[index] = entry.getKey();
        }
        return res;

    }

    // 2296. 设计一个文本编辑器 (Design a Text Editor)
    class TextEditor {
        private StringBuilder s;

        private int cursorIndex;

        public TextEditor() {
            s = new StringBuilder();
            cursorIndex = 0;

        }

        public void addText(String text) {
            s.insert(cursorIndex, text);
            cursorIndex += text.length();
        }

        public int deleteText(int k) {
            int leftCount = Math.min(k, cursorIndex);
            s.delete(cursorIndex - leftCount, cursorIndex);
            cursorIndex -= leftCount;
            return leftCount;

        }

        public String cursorLeft(int k) {
            cursorIndex = Math.max(cursorIndex - k, 0);
            return s.substring(Math.max(0, cursorIndex - 10), cursorIndex);

        }

        public String cursorRight(int k) {
            cursorIndex = Math.min(cursorIndex + k, s.length());
            return s.substring(Math.max(0, cursorIndex - 10), cursorIndex);
        }
    }

    // 2265. 统计值等于子树平均值的节点数 (Count Nodes Equal to Average of Subtree)
    private int res2265;

    public int averageOfSubtree(TreeNode root) {
        dfs2265(root);
        return res2265;

    }

    private int[] dfs2265(TreeNode root) {
        if (root == null) {
            return new int[] { 0, 0 };
        }
        int[] left = dfs2265(root.left);
        int[] right = dfs2265(root.right);
        int sum = left[0] + right[0] + root.val;
        int cnt = left[1] + right[1] + 1;
        if (root.val == sum / cnt) {
            ++res2265;
        }
        return new int[] { sum, cnt };

    }

    // 面试题 08.07. 无重复字符串的排列组合 (Permutation I LCCI) --回溯
    private List<String> res0807;
    private char[] chars0807;
    private int n0807;
    private StringBuilder path0807;

    public String[] permutation(String S) {
        this.res0807 = new ArrayList<>();
        this.n0807 = S.length();
        this.chars0807 = S.toCharArray();
        this.path0807 = new StringBuilder();
        dfs0807(0);
        return res0807.toArray(new String[0]);

    }

    private void dfs0807(int mask) {
        if (mask == ((1 << n0807) - 1)) {
            res0807.add(path0807.toString());
            return;
        }
        int candidates = (~mask) & ((1 << n0807) - 1);
        while (candidates != 0) {
            int last = candidates & (-candidates);
            int index = Integer.numberOfTrailingZeros(last);
            path0807.append(chars0807[index]);
            dfs0807(mask | last);
            path0807.deleteCharAt(path0807.length() - 1);
            candidates &= candidates - 1;
        }
    }

    // 46. 全排列 (Permutations) --回溯
    // 剑指 Offer II 083. 没有重复元素集合的全排列
    // 排列：需要用used数组
    // 无重复元素：不需要排序
    private int n46;
    private int[] nums46;
    private List<List<Integer>> res46;
    private List<Integer> list46;

    public List<List<Integer>> permute(int[] nums) {
        this.n46 = nums.length;
        this.nums46 = nums;
        this.res46 = new ArrayList<>();
        this.list46 = new ArrayList<>();
        dfs46(0);
        return res46;
    }

    private void dfs46(int mask) {
        if (mask == (1 << n46) - 1) {
            res46.add(new ArrayList<>(list46));
            return;
        }
        int pos = (~mask) & ((1 << n46) - 1);
        while (pos != 0) {
            int last = pos & (-pos);
            int index = Integer.numberOfTrailingZeros(last);
            list46.add(nums46[index]);
            dfs46(mask | last);
            list46.remove(list46.size() - 1);
            pos &= pos - 1;
        }
    }

    // 6095. 强密码检验器 II (Strong Password Checker II)
    public boolean strongPasswordCheckerII(String password) {
        int n = password.length();
        if (n < 8) {
            return false;
        }
        boolean a = false;
        boolean b = false;
        boolean c = false;
        boolean d = false;
        for (int i = 0; i < n; ++i) {
            char ch = password.charAt(i);
            if (i > 0 && ch == password.charAt(i - 1)) {
                return false;
            }
            if (Character.isLowerCase(ch)) {
                a = true;
            } else if (Character.isUpperCase(ch)) {
                b = true;
            } else if (Character.isDigit(ch)) {
                c = true;
            } else if ("!@#$%^&*()-+".indexOf(String.valueOf(ch)) != -1) {
                d = true;
            }
        }
        return a && b && c && d;

    }

    // 2300. 咒语和药水的成功对数 (Successful Pairs of Spells and Potions)
    public int[] successfulPairs(int[] spells, int[] potions, long success) {
        Arrays.sort(potions);
        int n = spells.length;
        int m = potions.length;
        int[] res = new int[n];
        Integer[] ids = IntStream.range(0, n).boxed().toArray(Integer[]::new);
        Arrays.sort(ids, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(spells[o2], spells[o1]);
            }

        });
        int i = 0;
        for (int id : ids) {
            while (i < m && (long) potions[i] * spells[id] < success) {
                ++i;
            }
            if (m == i) {
                break;
            }
            res[id] = m - i;
        }
        return res;

    }

    // 6097. 替换字符后匹配 (Match Substring After Replacement)
    public boolean matchReplacement(String s, String sub, char[][] mappings) {
        int m = s.length();
        int n = sub.length();
        if (m < n) {
            return false;
        }
        Map<Character, Set<Character>> map = new HashMap<>();
        for (char[] mapping : mappings) {
            map.computeIfAbsent(mapping[0], k -> new HashSet<>()).add(mapping[1]);
        }
        for (int i = 0; i <= m - n; ++i) {
            if (s.charAt(i) == sub.charAt(0)
                    || map.getOrDefault(sub.charAt(0), new HashSet<>()).contains(s.charAt(i))) {
                if (match6097(map, s.substring(i, i + n), sub)) {
                    return true;
                }
            }
        }
        return false;

    }

    private boolean match6097(Map<Character, Set<Character>> map, String s, String sub) {
        for (int i = 0; i < sub.length(); ++i) {
            if (sub.charAt(i) != s.charAt(i)
                    && !map.getOrDefault(sub.charAt(i), new HashSet<>()).contains(s.charAt(i))) {
                return false;
            }
        }
        return true;
    }

    // 6098. 统计得分小于 K 的子数组数目 (Count Subarrays With Score Less Than K)
    public long countSubarrays(int[] nums, long k) {
        int i = 0;
        int j = 0;
        long preSum = 0l;
        long res = 0l;
        while (j < nums.length) {
            preSum += nums[j];
            while (preSum * (j - i + 1) >= k) {
                preSum -= nums[i];
                ++i;
            }
            res += j - i + 1;
            ++j;
        }
        return res;

    }

    // 5259. 计算应缴税款总额 (Calculate Amount Paid in Taxes)
    public double calculateTax(int[][] brackets, int income) {
        double res = 0d;
        int min = 0;
        int pre = 0;
        for (int[] bracket : brackets) {
            min = Math.min(income, bracket[0] - pre);
            res += min * bracket[1] * 0.01d;
            pre = bracket[0];
            income -= min;
            if (income == 0) {
                break;
            }
        }
        return res;

    }

    // 2304. 网格中的最小路径代价 (Minimum Path Cost in a Grid)
    public int minPathCost(int[][] grid, int[][] moveCost) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < n; ++i) {
            dp[0][i] = grid[0][i];
        }
        for (int i = 1; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int min = Integer.MAX_VALUE;
                for (int k = 0; k < n; ++k) {
                    min = Math.min(min, dp[i - 1][k] + grid[i][j] + moveCost[grid[i - 1][k]][j]);
                }
                dp[i][j] = min;
            }
        }
        return Arrays.stream(dp[m - 1]).min().getAsInt();

    }

    // 2304. 网格中的最小路径代价 (Minimum Path Cost in a Grid)
    private int[][] grid2304;
    private int[][] moveCost2304;
    private int[][] memo2304;
    private int m2304;
    private int n2304;

    public int minPathCost2(int[][] grid, int[][] moveCost) {
        this.grid2304 = grid;
        this.moveCost2304 = moveCost;
        this.m2304 = grid.length;
        this.n2304 = grid[0].length;
        this.memo2304 = new int[m2304][n2304];
        for (int i = 0; i < m2304; ++i) {
            Arrays.fill(memo2304[i], -1);
        }
        int res = Integer.MAX_VALUE;
        for (int j = 0; j < n2304; ++j) {
            res = Math.min(res, dfs2304(0, j));
        }
        return res;
    }

    private int dfs2304(int i, int j) {
        if (i == m2304 - 1) {
            return grid2304[i][j];
        }
        if (memo2304[i][j] != -1) {
            return memo2304[i][j];
        }
        int res = Integer.MAX_VALUE;
        for (int k = 0; k < n2304; ++k) {
            res = Math.min(res, dfs2304(i + 1, k) + moveCost2304[grid2304[i][j]][k]);
        }
        return memo2304[i][j] = res + grid2304[i][j];
    }

    // 2305. 公平分发饼干 (Fair Distribution of Cookies)
    private int k2305;
    private int u2305;
    private int[] sum2305;
    private int[][] memo2305;

    public int distributeCookies(int[] cookies, int k) {
        int n = cookies.length;
        this.u2305 = (1 << n) - 1;
        this.k2305 = k;
        this.sum2305 = new int[1 << n];
        for (int i = 1; i < (1 << n); ++i) {
            int bit = Integer.numberOfTrailingZeros(i);
            sum2305[i] = sum2305[i ^ (1 << bit)] + cookies[bit];
        }
        this.memo2305 = new int[k][1 << n];
        return dfs2305(0, 0);
    }

    private int dfs2305(int i, int m) {
        if (i == k2305 || m == u2305) {
            return i == k2305 && m == u2305 ? 0 : (int) 1e8;
        }
        if (memo2305[i][m] != 0) {
            return memo2305[i][m];
        }
        int c = u2305 ^ m;
        int res = (int) 1e9;
        for (int j = c; j > 0; j = (j - 1) & c) {
            res = Math.min(res, Math.max(dfs2305(i + 1, j | m), sum2305[j]));
        }
        return memo2305[i][m] = res;
    }

    // 17. 电话号码的字母组合 (Letter Combinations of a Phone Number) --回溯
    private StringBuilder builder17;
    private int n17;
    private List<String> res17;
    private String[] map17 = { "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
    private String digits17;

    public List<String> letterCombinations(String digits) {
        res17 = new ArrayList<>();
        if (digits.isEmpty()) {
            return res17;
        }
        builder17 = new StringBuilder();
        this.n17 = digits.length();

        this.digits17 = digits;
        dfs17(0);
        return res17;

    }

    private void dfs17(int i) {
        if (i == n17) {
            res17.add(builder17.toString());
            return;
        }
        int index = digits17.charAt(i) - '0';
        for (char c : map17[index].toCharArray()) {
            builder17.append(c);
            dfs17(i + 1);
            builder17.deleteCharAt(builder17.length() - 1);
        }
    }

    // 17. 电话号码的字母组合 (Letter Combinations of a Phone Number) --队列
    public List<String> letterCombinations2(String digits) {
        List<String> res = new ArrayList<>();
        if (digits.length() == 0) {
            return res;
        }
        Queue<String> queue = new LinkedList<>();
        Map<Character, String> map = new HashMap<>() {
            {
                put('2', "abc");
                put('3', "def");
                put('4', "ghi");
                put('5', "jkl");
                put('6', "mno");
                put('7', "pqrs");
                put('8', "tuv");
                put('9', "wxyz");
            }
        };
        for (int i = 0; i < digits.length(); ++i) {
            handle17(queue, map.get(digits.charAt(i)));
        }
        return new ArrayList<>(queue);

    }

    private void handle17(Queue<String> queue, String string) {
        if (queue.isEmpty()) {
            for (char c : string.toCharArray()) {
                queue.offer(String.valueOf(c));
            }
        } else {
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                String cur = queue.poll();
                for (char c : string.toCharArray()) {
                    queue.offer(cur + String.valueOf(c));
                }
            }
        }
    }

    // 22. 括号生成 (Generate Parentheses)
    // 剑指 Offer II 085. 生成匹配的括号 --backtrack
    // 面试题 08.09.括号 (Bracket LCCI)

    private int n22;
    private List<String> res22;
    private StringBuilder builder22;

    public List<String> generateParenthesis(int n) {
        res22 = new ArrayList<>();
        this.n22 = n;
        this.builder22 = new StringBuilder();
        dfs22(0, 0);
        return res22;

    }

    private void dfs22(int left, int right) {
        if (left + right == n22 * 2) {
            res22.add(builder22.toString());
            return;
        }
        if (left < n22) {
            builder22.append("(");
            dfs22(left + 1, right);
            builder22.deleteCharAt(builder22.length() - 1);
        }
        if (left > right) {
            builder22.append(")");
            dfs22(left, right + 1);
            builder22.deleteCharAt(builder22.length() - 1);
        }
    }

    class Node22 {
        int left;
        int right;
        String cur;

        public Node22(String cur, int left, int right) {
            this.cur = cur;
            this.left = left;
            this.right = right;
        }
    }

    // 22. 括号生成 (Generate Parentheses)
    // 剑指 Offer II 085. 生成匹配的括号 --bfs
    // 面试题 08.09.括号 (Bracket LCCI)
    public List<String> generateParenthesis2(int n) {
        List<String> res = new ArrayList<>();
        Queue<Node22> queue = new LinkedList<>();
        queue.offer(new Node22("", 0, 0));
        while (!queue.isEmpty()) {
            Node22 node = queue.poll();
            String cur = node.cur;
            if (cur.length() == n * 2) {
                res.add(node.cur);
                continue;
            }
            int left = node.left;
            int right = node.right;
            if (left < n) {
                queue.offer(new Node22(cur + "(", left + 1, right));
            }
            if (right < left) {
                queue.offer(new Node22(cur + ")", left, right + 1));
            }
        }
        return res;

    }

    // 37. 解数独 (Sudoku Solver) --回溯
    private boolean[][] line_1;
    private boolean[][] column_1;
    private boolean[][][] block_1;
    private List<int[]> spaces_1;
    private boolean valid_1;

    public void solveSudoku(char[][] board) {
        int n = board.length;
        line_1 = new boolean[n][n];
        column_1 = new boolean[n][n];
        block_1 = new boolean[3][3][n];
        spaces_1 = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j] == '.') {
                    spaces_1.add(new int[] { i, j });
                } else {
                    int num = board[i][j] - '0' - 1;
                    line_1[i][num] = true;
                    column_1[j][num] = true;
                    block_1[i / 3][j / 3][num] = true;
                }
            }
        }
        dfs37(board, 0);

    }

    private void dfs37(char[][] board, int index) {
        if (index == spaces_1.size()) {
            valid_1 = true;
            return;
        }
        int[] cur = spaces_1.get(index);
        int x = cur[0];
        int y = cur[1];
        for (int val = 0; val < 9 && !valid_1; ++val) {
            if (!line_1[x][val] && !column_1[y][val] && !block_1[x / 3][y / 3][val]) {
                line_1[x][val] = true;
                column_1[y][val] = true;
                block_1[x / 3][y / 3][val] = true;
                board[x][y] = (char) (val + '0' + 1);
                dfs37(board, index + 1);
                line_1[x][val] = false;
                column_1[y][val] = false;
                block_1[x / 3][y / 3][val] = false;
            }
        }
    }

    // 37. 解数独 (Sudoku Solver) -- 位运算 + 回溯
    private int[] line37_2;
    private int[] column37_2;
    private int[][] block37_2;
    private List<int[]> spaces37_2;
    private boolean valid37_2;

    public void solveSudoku2(char[][] board) {
        int n = board.length;
        line37_2 = new int[n];
        column37_2 = new int[n];
        block37_2 = new int[n][n];
        spaces37_2 = new ArrayList<>();

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j] == '.') {
                    spaces37_2.add(new int[] { i, j });
                } else {
                    int digit = board[i][j] - '0' - 1;
                    flip37(i, j, digit);
                }
            }
        }
        dfs37_2(board, 0);

    }

    private void dfs37_2(char[][] board, int index) {
        if (index == spaces37_2.size()) {
            valid37_2 = true;
            return;
        }
        int[] cur = spaces37_2.get(index);
        int x = cur[0];
        int y = cur[1];
        int mask = ~(line37_2[x] | column37_2[y] | block37_2[x / 3][y / 3]) & 0x1FF;
        for (; mask != 0 && !valid37_2; mask &= (mask - 1)) {
            int bit = mask & (-mask);
            int digit = Integer.bitCount(bit - 1);
            flip37(x, y, digit);
            board[x][y] = (char) (digit + '0' + 1);
            dfs37_2(board, index + 1);
            flip37(x, y, digit);
        }
    }

    private void flip37(int i, int j, int digit) {
        line37_2[i] ^= 1 << digit;
        column37_2[j] ^= 1 << digit;
        block37_2[i / 3][j / 3] ^= 1 << digit;
    }

    // 37. 解数独 (Sudoku Solver) -- 位运算 + 回溯 + 预处理 (把只有唯一一个数可选的位置先填上值)
    private int[] line37_3;
    private int[] column37_3;
    private int[][] block37_3;
    private List<int[]> spaces37_3;
    private boolean valid37_3;

    public void solveSudoku3(char[][] board) {
        int n = board.length;
        line37_3 = new int[n];
        column37_3 = new int[n];
        block37_3 = new int[n / 3][n / 3];
        spaces37_3 = new ArrayList<>();

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j] != '.') {
                    int digit = board[i][j] - '0' - 1;
                    flip37_3(i, j, digit);
                }
            }
        }
        while (true) {
            boolean modified = false;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (board[i][j] == '.') {
                        int mask = ~(line37_3[i] | column37_3[j] | block37_3[i / 3][j / 3]) & ((1 << 9) - 1);
                        // 只有一个1，即该位置的值已唯一确定
                        if ((mask & (mask - 1)) == 0) {
                            int digit = Integer.numberOfTrailingZeros(mask);
                            board[i][j] = (char) (digit + '0' + 1);
                            flip37_3(i, j, digit);
                            modified = true;
                        }
                    }
                }
            }
            if (!modified) {
                break;
            }
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j] == '.') {
                    spaces37_3.add(new int[] { i, j });
                }
            }
        }
        dfs37_3(board, 0);

    }

    private void dfs37_3(char[][] board, int index) {
        if (index == spaces37_3.size()) {
            valid37_3 = true;
            return;
        }
        int[] cur = spaces37_3.get(index);
        int x = cur[0];
        int y = cur[1];
        int mask = ~(line37_3[x] | column37_3[y] | block37_3[x / 3][y / 3]) & ((1 << 9) - 1);
        for (; mask != 0 && !valid37_3; mask &= (mask - 1)) {
            int bit = mask & (-mask);
            int digit = Integer.numberOfTrailingZeros(bit);
            board[x][y] = (char) (digit + '0' + 1);
            flip37_3(x, y, digit);
            dfs37_3(board, index + 1);
            flip37_3(x, y, digit);
        }
    }

    private void flip37_3(int i, int j, int digit) {
        line37_3[i] ^= 1 << digit;
        column37_3[j] ^= 1 << digit;
        block37_3[i / 3][j / 3] ^= 1 << digit;
    }

    // 40. 组合总和 II (Combination Sum II) --回溯
    // LCR 082. 组合总和 II
    // 组合：不需要用used数组
    // 有重复元素：需要排序
    // 每个元素只能用一次 ：回溯的时候 index = i + 1
    private List<int[]> list40;
    private List<Integer> path40;
    private int n40;
    private int target40;
    private List<List<Integer>> res40;

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int c : candidates) {
            map.merge(c, 1, Integer::sum);
        }
        this.list40 = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            list40.add(new int[] { entry.getKey(), entry.getValue() });
        }
        this.path40 = new ArrayList<>();
        this.n40 = list40.size();
        this.target40 = target;
        this.res40 = new ArrayList<>();
        dfs40(0, 0);
        return res40;
    }

    private void dfs40(int i, int j) {
        if (i == n40) {
            if (j == target40) {
                res40.add(new ArrayList<>(path40));
            }
            return;
        }
        if (j > target40) {
            return;
        }
        dfs40(i + 1, j);
        List<Integer> cur = new ArrayList<>();
        for (int k = 1; k <= list40.get(i)[1]; ++k) {
            if (k * list40.get(i)[0] + j > target40) {
                break;
            }
            cur.add(list40.get(i)[0]);
            path40.addAll(cur);
            dfs40(i + 1, j + k * list40.get(i)[0]);
            int c = k;
            while (c-- > 0) {
                path40.remove(path40.size() - 1);
            }
        }
    }

    // 47. 全排列 II (Permutations II) --回溯
    // LCR 084. 全排列 II
    // 排列：需要用used数组
    // 有重复元素：需要排序
    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        backtrack47(res, path, nums, used);
        return res;
    }

    private void backtrack47(List<List<Integer>> res, List<Integer> path, int[] nums, boolean[] used) {
        if (path.size() == nums.length) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < nums.length; ++i) {
            if (used[i]) {
                continue;
            }
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
                continue;
            }
            used[i] = true;
            path.add(nums[i]);
            backtrack47(res, path, nums, used);
            path.remove(path.size() - 1);
            used[i] = false;
        }
    }

    // 面试题 08.08. 有重复字符串的排列组合
    public String[] permutation3(String S) {
        boolean[] used = new boolean[S.length()];
        List<String> res = new ArrayList<>();
        StringBuilder builder = new StringBuilder();
        char[] chars = S.toCharArray();
        Arrays.sort(chars);
        backtrack0808(res, used, builder, chars);
        return res.toArray(new String[0]);
    }

    private void backtrack0808(List<String> res, boolean[] used, StringBuilder builder, char[] chars) {
        if (builder.length() == chars.length) {
            res.add(builder.toString());
            return;
        }
        for (int i = 0; i < chars.length; ++i) {
            if (used[i]) {
                continue;
            }
            if (i > 0 && chars[i] == chars[i - 1] && !used[i - 1]) {
                continue;
            }
            used[i] = true;
            builder.append(chars[i]);
            backtrack0808(res, used, builder, chars);
            builder.deleteCharAt(builder.length() - 1);
            used[i] = false;
        }
    }

    // 39. 组合总和 (Combination Sum) --回溯
    // 剑指 Offer II 081. 允许重复选择元素的组合
    // 组合：不需要用used数组
    // 无重复元素：不需要排序
    // 每个元素可以多次使用 ：回溯的时候 index = i
    private int[] candidates39;
    private int target39;
    private List<List<Integer>> res39;
    private List<Integer> list39;

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        this.candidates39 = candidates;
        this.target39 = target;
        this.res39 = new ArrayList<>();
        this.list39 = new ArrayList<>();
        dfs39(0, 0);
        return res39;
    }

    private void dfs39(int index, int sum) {
        if (sum > target39) {
            return;
        }
        if (sum == target39) {
            res39.add(new ArrayList<>(list39));
            return;
        }
        for (int i = index; i < candidates39.length; ++i) {
            list39.add(candidates39[i]);
            dfs39(i, sum + candidates39[i]);
            list39.remove(list39.size() - 1);
        }

    }

    // 77. 组合 (Combinations)
    // LCR 080. 组合
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < (1 << n); ++i) {
            if (Integer.bitCount(i) == k) {
                List<Integer> list = new ArrayList<>();
                for (int j = 0; j < n; ++j) {
                    if ((i & (1 << j)) != 0) {
                        list.add(j + 1);
                    }
                }
                res.add(list);
            }
        }
        return res;

    }

    // 77. 组合 (Combinations) --回溯
    // LCR 080. 组合
    // 组合：不需要用used数组
    // 无重复元素：不需要排序
    // 每个元素只能使用一次 ：回溯的时候 index = i + 1
    private int n77;
    private int k77;
    private List<List<Integer>> res77;
    private List<Integer> path77;

    public List<List<Integer>> combine2(int n, int k) {
        this.res77 = new ArrayList<>();
        this.n77 = n;
        this.k77 = k;
        this.path77 = new ArrayList<>();
        dfs77(0);
        return res77;

    }

    private void dfs77(int i) {
        if (path77.size() > k77) {
            return;
        }
        if (path77.size() + n77 - i < k77) {
            return;
        }
        if (i == n77) {
            res77.add(new ArrayList<>(path77));
            return;
        }
        dfs77(i + 1);
        path77.add(i + 1);
        dfs77(i + 1);
        path77.remove(path77.size() - 1);
    }

    // 51. N 皇后 (N-Queens) --回溯
    // 面试题 08.12. 八皇后 (Eight Queens LCCI)
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> res = new ArrayList<>();
        int[] queens = new int[n];
        Arrays.fill(queens, -1);
        Set<Integer> colunms = new HashSet<>();
        Set<Integer> diagonals1 = new HashSet<>();
        Set<Integer> diagonals2 = new HashSet<>();
        backtrack51(res, queens, n, 0, colunms, diagonals1, diagonals2);
        return res;

    }

    private void backtrack51(List<List<String>> res, int[] queens, int n, int row, Set<Integer> colunms,
            Set<Integer> diagonals1, Set<Integer> diagonals2) {
        if (row == n) {
            List<String> board = generate51(queens);
            res.add(board);
            return;
        }
        for (int i = 0; i < n; ++i) {
            if (colunms.contains(i)) {
                continue;
            }
            int diagonal1 = row - i;
            if (diagonals1.contains(diagonal1)) {
                continue;
            }
            int diagonal2 = row + i;
            if (diagonals2.contains(diagonal2)) {
                continue;
            }
            colunms.add(i);
            diagonals1.add(diagonal1);
            diagonals2.add(diagonal2);
            queens[row] = i;
            backtrack51(res, queens, n, row + 1, colunms, diagonals1, diagonals2);
            queens[row] = -1;
            diagonals2.remove(diagonal2);
            diagonals1.remove(diagonal1);
            colunms.remove(i);
        }

    }

    private List<String> generate51(int[] queens) {
        int n = queens.length;
        List<String> res = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            char[] row = new char[n];
            Arrays.fill(row, '.');
            row[queens[i]] = 'Q';
            res.add(String.valueOf(row));
        }
        return res;
    }

    // 51. N 皇后 (N-Queens) --回溯 + 位运算
    // 面试题 08.12. 八皇后 (Eight Queens LCCI)
    public List<List<String>> solveNQueens2(int n) {
        List<List<String>> res = new ArrayList<>();
        dfs2_51(res, new ArrayList<>(), 0, 0, 0, 0, n);
        return res;

    }

    private void dfs2_51(List<List<String>> res, List<Integer> list, int i, int c0, int c1, int c2, int n) {
        if (i == n) {
            List<String> cur = new ArrayList<>();
            for (int x : list) {
                StringBuilder row = new StringBuilder();
                for (int j = 0; j < n; ++j) {
                    row.append(j == x ? 'Q' : '.');
                }
                cur.add(row.toString());
            }
            res.add(cur);
            return;
        }
        int u = (1 << n) - 1;
        int c = u ^ (c0 | c1 | c2);
        while (c != 0) {
            int lb = Integer.numberOfTrailingZeros(c);
            list.add(lb);
            dfs2_51(res, list, i + 1, c0 | (1 << lb), u & ((c1 | (1 << lb)) << 1), (c2 | 1 << lb) >> 1, n);
            list.remove(list.size() - 1);
            c &= c - 1;
        }
    }

    // 52. N皇后 II (N-Queens II) --回溯
    public int totalNQueens(int n) {
        return dfs52(n, 0, 0, 0, 0);
    }

    private int dfs52(int n, int i, int d0, int d1, int d2) {
        if (i == n) {
            return 1;
        }
        int res = 0;
        int u = (1 << n) - 1;
        int c = u ^ (colunms | diagonal1 | diagonal2);
        while (c != 0) {
            int lb = Integer.numberOfTrailingZeros(c);
            res += dfs52(n, i + 1, d0 | (1 << lb), u & ((d1 | (1 << lb)) << 1), (d2 | (1 << lb)) >> 1);
            c &= c - 1;
        }
        return res;
    }

    // 93. 复原 IP 地址 (Restore IP Addresses) --回溯
    // 剑指 Offer II 087. 复原 IP
    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
        if (s.length() > 12) {
            return res;
        }
        StringBuilder builder = new StringBuilder();
        backtrack93(res, builder, 0, s);
        return res;

    }

    private void backtrack93(List<String> res, StringBuilder builder, int start, String s) {
        if (start == s.length() && builder.length() == s.length() + 3) {
            res.add(builder.toString());
            return;
        }

        for (int end = start + 1; end <= s.length(); ++end) {
            // 超过1位时，首位不可为0
            if (end - start > 1 && s.charAt(start) == '0') {
                return;
            }
            int cur = Integer.parseInt(s.substring(start, end));
            if (cur > 255) {
                return;
            }
            if (!builder.isEmpty()) {
                builder.append(".");
            }
            builder.append(cur);
            backtrack93(res, builder, end, s);
            int lastIndexOfDot = builder.lastIndexOf(".");
            builder.setLength(Math.max(0, lastIndexOfDot));
        }
    }

    // LCP 55. 采集果实
    public int getMinimumTime(int[] time, int[][] fruits, int limit) {
        int res = 0;
        for (int fruit[] : fruits) {
            int t = time[fruit[0]];
            res += (fruit[1] / limit + (fruit[1] % limit == 0 ? 0 : 1)) * t;
        }
        return res;

    }

    // 2309. 兼具大小写的最好英文字母 (Greatest English Letter in Upper and Lower Case)
    public String greatestLetter(String s) {
        int mask1 = 0;
        int mask2 = 0;
        for (char c : s.toCharArray()) {
            if (Character.isLowerCase(c)) {
                mask1 |= 1 << (c - 'a');
            } else {
                mask2 |= 1 << (c - 'A');
            }
        }
        int mask = mask1 & mask2;
        if (mask == 0) {
            return "";
        }
        return String.valueOf((char) (Integer.bitCount(Integer.highestOneBit(mask) - 1) + 'A'));

    }

    // 473. 火柴拼正方形 (Matchsticks to Square) --回溯
    public boolean makesquare(int[] matchsticks) {
        int sum = Arrays.stream(matchsticks).sum();
        if (sum % 4 != 0) {
            return false;
        }
        int side = sum / 4;
        Arrays.sort(matchsticks);
        // 从大到小排序
        int left = 0;
        int right = matchsticks.length - 1;
        while (left < right) {
            int temp = matchsticks[left];
            matchsticks[left] = matchsticks[right];
            matchsticks[right] = temp;
            ++left;
            --right;
        }
        int[] square = new int[4];
        return backtrack473(square, matchsticks, 0, side);

    }

    private boolean backtrack473(int[] square, int[] matchsticks, int indexMatchsticks, int side) {
        if (indexMatchsticks == matchsticks.length) {
            return true;
        }
        for (int i = 0; i < square.length; ++i) {
            if (square[i] + matchsticks[indexMatchsticks] > side || i > 0 && square[i - 1] == square[i]) {
                continue;
            }
            square[i] += matchsticks[indexMatchsticks];
            if (backtrack473(square, matchsticks, indexMatchsticks + 1, side)) {
                return true;
            }
            square[i] -= matchsticks[indexMatchsticks];
        }
        return false;
    }

    // 473. 火柴拼正方形 (Matchsticks to Square)
    private int side473;
    private int n473;
    private int[] arr473;
    private int[] memo473;
    private int u473;

    public boolean makesquare2(int[] matchsticks) {
        int sum = Arrays.stream(matchsticks).sum();
        if (sum % 4 != 0) {
            return false;
        }
        this.side473 = sum / 4;
        this.n473 = matchsticks.length;
        this.arr473 = new int[1 << n473];
        for (int i = 1; i < (1 << n473); ++i) {
            int bit = Integer.numberOfTrailingZeros(i);
            arr473[i] = arr473[i ^ (1 << bit)] + matchsticks[bit];
        }
        this.u473 = (1 << n473) - 1;
        this.memo473 = new int[1 << n473];
        return dfs473(0);

    }

    private boolean dfs473(int i) {
        if (i == u473) {
            return true;
        }
        if (memo473[i] != 0) {
            return memo473[i] > 0;
        }
        int candidate = i ^ u473;
        for (int c = candidate; c > 0; c = (c - 1) & candidate) {
            if (arr473[c] == side473 && dfs473(i | c)) {
                memo473[i] = 1;
                return true;
            }
        }
        memo473[i] = -1;
        return false;
    }

    // 5218. 个位数字为 K 的整数之和 (Sum of Numbers With Units Digit K)
    public int minimumNumbers(int num, int k) {
        if (num == 0) {
            return 0;
        }
        for (int i = 1; i <= num; ++i) {
            int t = num - i * k;
            if (t >= 0 && t % 10 == 0) {
                return i;
            }
        }
        return -1;

    }

    // 6099. 小于等于 K 的最长二进制子序列 (Longest Binary Subsequence Less Than or Equal to K)
    public int longestSubsequence(String s, int k) {
        int res = 0;
        int sum = 0;
        for (int i = s.length() - 1; i >= 0; --i) {
            if (s.charAt(i) == '0') {
                ++res;
            } else {
                if (res <= 30 && sum + (1 << res) <= k) {
                    sum += 1 << res;
                    ++res;
                }
            }
        }
        return res;
    }

    // 131. 分割回文串 (Palindrome Partitioning) --回溯 + dp
    // 剑指 Offer II 086. 分割回文子字符串
    private int n131;
    private String s131;
    private List<List<String>> res131;
    private boolean[][] judge131;

    public List<List<String>> partition(String s) {
        this.n131 = s.length();
        this.s131 = s;
        this.judge131 = new boolean[n131][n131];
        for (int i = n131 - 1; i >= 0; --i) {
            for (int j = i; j < n131; ++j) {
                if (i == j || j - i == 1 && s.charAt(i) == s.charAt(j)
                        || j - i > 1 && s.charAt(i) == s.charAt(j) && judge131[i + 1][j - 1]) {
                    judge131[i][j] = true;
                }
            }
        }
        this.res131 = new ArrayList<>();
        dfs131(0, new ArrayList<>());
        return res131;

    }

    private void dfs131(int i, List<String> list) {
        if (i == n131) {
            res131.add(new ArrayList<>(list));
            return;
        }
        for (int j = i; j < n131; ++j) {
            if (judge131[i][j]) {
                list.add(s131.substring(i, j + 1));
                dfs131(j + 1, list);
                list.remove(list.size() - 1);
            }
        }
    }

    // 216. 组合总和 III (Combination Sum III) --回溯
    private int k216;
    private int n216;
    private List<Integer> list216;
    private List<List<Integer>> res216;

    public List<List<Integer>> combinationSum3(int k, int n) {
        this.k216 = k;
        this.n216 = n;
        this.list216 = new ArrayList<>();
        this.res216 = new ArrayList<>();
        dfs216(1, 0);
        return res216;

    }

    private void dfs216(int i, int j) {
        if (i == 10 || j >= n216 || list216.size() >= k216) {
            if (j == n216 && list216.size() == k216) {
                res216.add(new ArrayList<>(list216));
            }
            return;
        }
        dfs216(i + 1, j);
        list216.add(i);
        dfs216(i + 1, i + j);
        list216.remove(list216.size() - 1);
    }
    

    // 491. 递增子序列 (Increasing Subsequences) --枚举 + 位运算
    public List<List<Integer>> findSubsequences(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Set<Integer> set = new HashSet<>();
        int n = nums.length;
        for (int i = 0; i < (1 << n); ++i) {
            if (Integer.bitCount(i) >= 2) {
                List<Integer> candidateList = getSubSequence(i, nums);
                int hashVal = getHash(candidateList);
                if (!set.contains(hashVal) && checkNoneDecrease(candidateList)) {
                    set.add(hashVal);
                    res.add(candidateList);
                }
            }
        }
        return res;

    }

    private boolean checkNoneDecrease(List<Integer> candidateList) {
        for (int i = 1; i < candidateList.size(); ++i) {
            if (candidateList.get(i - 1) > candidateList.get(i)) {
                return false;
            }
        }
        return true;
    }

    private int getHash(List<Integer> candidateList) {
        int hashVal = 0;
        for (int num : candidateList) {
            hashVal = (int) (hashVal * 263 % 1E9 + 101 + num);
            hashVal %= 1E9;
        }
        return hashVal;
    }

    private List<Integer> getSubSequence(int mask, int[] nums) {
        List<Integer> res = new ArrayList<>();
        int index = nums.length - 1;
        while (mask != 0) {
            if ((mask & 1) != 0) {
                res.add(0, nums[index]);
            }
            --index;
            mask >>= 1;
        }
        return res;
    }

    // 491. 递增子序列 (Increasing Subsequences) --递归枚举 + 位运算
    private List<List<Integer>> res491;

    public List<List<Integer>> findSubsequences2(int[] nums) {
        res491 = new ArrayList<>();
        backtrack491(0, Integer.MIN_VALUE, nums, new ArrayList<>());
        return res491;

    }

    private void backtrack491(int index, int last, int[] nums, List<Integer> cur) {
        if (index == nums.length) {
            if (cur.size() >= 2) {
                res491.add(new ArrayList<>(cur));
            }
            return;
        }
        if (nums[index] >= last) {
            cur.add(nums[index]);
            backtrack491(index + 1, nums[index], nums, cur);
            cur.remove(cur.size() - 1);
        }
        if (nums[index] != last) {
            backtrack491(index + 1, last, nums, cur);
        }
    }

    // 1392. 最长快乐前缀 (Longest Happy Prefix) --Rabin Karp
    public String longestPrefix(String s) {
        final int base = 31;
        final int MOD = 1000000009;
        long prefix = 0l;
        long suffix = 0l;
        long mul = 1l;
        int happy = 0;
        for (int i = 1; i < s.length(); ++i) {
            prefix = (prefix * base + s.charAt(i - 1) - 'a') % MOD;
            suffix = (suffix + (s.charAt(s.length() - i) - 'a') * mul) % MOD;
            if (prefix == suffix) {
                happy = i;
            }
            mul = mul * base % MOD;
        }
        return s.substring(0, happy);
    }

    // 1079. 活字印刷 (Letter Tile Possibilities) --回溯 全排列
    private int n1079;
    private char[] arr1079;
    private int res1079;
    private int used1079;

    public int numTilePossibilities(String tiles) {
        this.n1079 = tiles.length();
        this.arr1079 = tiles.toCharArray();
        Arrays.sort(arr1079);
        dfs1079();
        return res1079 - 1;

    }

    private void dfs1079() {
        ++res1079;
        if (Integer.bitCount(used1079) == n1079) {
            return;
        }
        for (int i = 0; i < n1079; ++i) {
            if (((used1079 >> i) & 1) == 1
                    || i > 0 && arr1079[i] == arr1079[i - 1] && ((used1079 >> (i - 1)) & 1) == 0) {
                continue;
            }
            used1079 ^= 1 << i;
            dfs1079();
            used1079 ^= 1 << i;
        }
    }

    // 980. 不同路径 III (Unique Paths III) --回溯
    private int res980;
    private int[][] grid980;
    private int tr980;
    private int tc980;
    private int[][] directions980 = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };

    public int uniquePathsIII(int[][] grid) {
        this.grid980 = grid;
        int todo = 0;
        int sr = 0;
        int sc = 0;
        for (int i = 0; i < grid.length; ++i) {
            for (int j = 0; j < grid[0].length; ++j) {
                if (grid[i][j] != -1) {
                    ++todo;
                }
                if (grid[i][j] == 1) {
                    sr = i;
                    sc = j;
                } else if (grid[i][j] == 2) {
                    this.tr980 = i;
                    this.tc980 = j;
                }
            }
        }
        backtrack980(sr, sc, todo);
        return res980;

    }

    private void backtrack980(int i, int j, int todo) {
        --todo;
        if (todo < 0) {
            return;
        }
        if (tr980 == i && tc980 == j) {
            if (todo == 0) {
                ++res980;
            }
            return;
        }
        grid980[i][j] = 3;
        for (int[] direction : directions980) {
            int nx = i + direction[0];
            int ny = j + direction[1];
            if (nx >= 0 && nx < grid980.length && ny >= 0 && ny < grid980[0].length) {
                if (grid980[nx][ny] % 2 == 0) {
                    backtrack980(nx, ny, todo);
                }
            }
        }
        grid980[i][j] = 0;
    }

    // 980. 不同路径 III (Unique Paths III)
    public int uniquePathsIII2(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int obstacle = 0;
        int s = -1;
        int e = -1;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int x = grid[i][j];
                int p = i * n + j;
                if (x == 1) {
                    s = p;
                } else if (x == 2) {
                    e = p;
                } else if (x == -1) {
                    obstacle |= 1 << p;
                }
            }
        }
        int[][] dirs = { { 0, 1 }, { 1, 0 }, { -1, 0 }, { 0, -1 } };
        int res = 0;
        Queue<int[]> q = new LinkedList<>();
        q.offer(new int[] { s, (1 << s) | obstacle });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0] / n;
            int y = cur[0] % n;
            int mask = cur[1];
            if (cur[0] == e) {
                res += (mask == (1 << (m * n)) - 1) ? 1 : 0;
                continue;
            }
            for (int[] d : dirs) {
                int nx = x + d[0];
                int ny = y + d[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && ((mask >> (nx * n + ny)) & 1) == 0) {
                    q.offer(new int[] { nx * n + ny, mask | (1 << (nx * n + ny)) });
                }
            }
        }
        return res;

    }

    // 1415. 长度为 n 的开心字符串中字典序第 k 小的字符串 (The k-th Lexicographical String of All Happy
    // Strings of Length n) --回溯
    public String getHappyString(int n, int k) {
        char[] candidateChars = { 'a', 'b', 'c' };
        List<String> list = new ArrayList<>();
        backtrack1415(n, candidateChars, new StringBuilder(), list, new HashSet<>());
        if (list.size() < k) {
            return "";
        }
        return list.get(k - 1);
    }

    private void backtrack1415(int n, char[] candidateChars, StringBuilder cur, List<String> list, Set<String> set) {
        if (cur.length() == n) {
            if (set.add(cur.toString())) {
                list.add(cur.toString());
            }
            return;
        }
        for (int i = 0; i < candidateChars.length; ++i) {
            if (cur.isEmpty() || candidateChars[i] != cur.charAt(cur.length() - 1)) {
                cur.append(candidateChars[i]);
                backtrack1415(n, candidateChars, cur, list, set);
                cur.deleteCharAt(cur.length() - 1);
            }
        }
    }

    // 1887. 使数组元素相等的减少操作次数 (Reduction Operations to Make the Array Elements Equal)
    // -- 排序
    public int reductionOperations(int[] nums) {
        Arrays.sort(nums);
        int res = 0;
        int count = 0;
        for (int i = 1; i < nums.length; ++i) {
            if (nums[i] != nums[i - 1]) {
                ++count;
            }
            res += count;
        }
        return res;
    }

    // 1947. 最大兼容性评分和 (Maximum Compatibility Score Sum) -- dp + 下一个排列枚举
    public int maxCompatibilitySum(int[][] students, int[][] mentors) {
        int m = students.length;
        int n = students[0].length;
        int[][] dp = new int[m][m];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                for (int k = 0; k < n; ++k) {
                    if (students[i][k] == mentors[j][k]) {
                        ++dp[i][j];
                    }
                }
            }
        }
        int[] index = new int[m];
        for (int i = 0; i < m; ++i) {
            index[i] = i;
        }
        int res = 0;
        while (true) {
            int max = 0;
            for (int i = 0; i < m; ++i) {
                max += dp[index[i]][i];
            }
            res = Math.max(res, max);

            // next permutation
            int i = m - 2;
            while (i >= 0) {
                if (index[i] < index[i + 1]) {
                    break;
                }
                --i;
            }
            if (i < 0) {
                break;
            }
            int j = m - 1;
            while (i < j) {
                if (index[i] < index[j]) {
                    break;
                }
                --j;
            }
            swap1947(index, i, j);
            Arrays.sort(index, i + 1, index.length);
        }
        return res;

    }

    private void swap1947(int[] index, int i, int j) {
        int temp = index[i];
        index[i] = index[j];
        index[j] = temp;
    }

    // 1947. 最大兼容性评分和 (Maximum Compatibility Score Sum) -- dp + 回溯
    private int res1947;

    public int maxCompatibilitySum2(int[][] students, int[][] mentors) {
        int m = students.length;
        int n = students[0].length;
        int[][] dp = new int[m][m];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                for (int k = 0; k < n; ++k) {
                    if (students[i][k] == mentors[j][k]) {
                        ++dp[i][j];
                    }
                }
            }
        }
        List<Integer> path = new ArrayList<>();
        boolean[] used = new boolean[m];
        backtrack1947(path, used, dp);
        return res1947;

    }

    private void backtrack1947(List<Integer> path, boolean[] used, int[][] dp) {
        if (path.size() == used.length) {
            res1947 = Math.max(res1947, checkScore1947(path, dp));
            return;
        }
        for (int i = 0; i < used.length; ++i) {
            if (!used[i]) {
                used[i] = true;
                path.add(i);
                backtrack1947(path, used, dp);
                used[i] = false;
                path.remove(path.size() - 1);
            }
        }
    }

    private int checkScore1947(List<Integer> path, int[][] dp) {
        int res = 0;
        int n = path.size();
        for (int i = 0; i < n; ++i) {
            res += dp[path.get(i)][i];
        }
        return res;
    }

    // 1947. 最大兼容性评分和 (Maximum Compatibility Score Sum)
    private int m1947;
    private int[][] memo1947;
    private int u1947;
    private int[][] xor1947;

    public int maxCompatibilitySum3(int[][] students, int[][] mentors) {
        this.m1947 = students.length;
        this.xor1947 = new int[m1947][m1947];
        for (int i = 0; i < m1947; ++i) {
            for (int j = 0; j < m1947; ++j) {
                xor1947[i][j] = sum1947(students[i], mentors[j]);
            }
        }
        this.u1947 = (1 << m1947) - 1;
        this.memo1947 = new int[m1947][1 << m1947];
        for (int i = 0; i < m1947; ++i) {
            Arrays.fill(memo1947[i], -1);
        }
        return dfs1947(0, 0);

    }

    private int dfs1947(int i, int j) {
        if (j == u1947) {
            return 0;
        }
        if (memo1947[i][j] != -1) {
            return memo1947[i][j];
        }
        int res = 0;
        for (int c = j ^ u1947; c != 0; c &= c - 1) {
            int lb = Integer.numberOfTrailingZeros(c);
            res = Math.max(res, dfs1947(i + 1, j | (1 << lb)) + xor1947[i][lb]);
        }
        return memo1947[i][j] = res;
    }

    private int sum1947(int[] a, int[] b) {
        int res = 0;
        for (int i = 0; i < a.length; ++i) {
            res += a[i] ^ b[i] ^ 1;
        }
        return res;
    }

    // 1593.拆分字符串使唯一子字符串的数目最大 (Split a String Into the Max Number of Unique
    // Substrings) --回溯
    private int res1593;

    public int maxUniqueSplit(String s) {
        res1593 = 1;
        Set<String> set = new HashSet<>();
        backtrack1593(s, 0, 0, set);
        return res1593;

    }

    private void backtrack1593(String s, int index, int split, Set<String> set) {
        if (index == s.length()) {
            res1593 = Math.max(res1593, split);
            return;
        }
        for (int i = index; i < s.length(); ++i) {
            String sub = s.substring(index, i + 1);
            if (!set.contains(sub)) {
                set.add(sub);
                backtrack1593(s, i + 1, split + 1, set);
                set.remove(sub);
            }
        }
    }

    // 1784. 检查二进制字符串字段 (Check if Binary String Has at Most One Segment of Ones)
    public boolean checkOnesSegment(String s) {
        int index = 0;
        while (index < s.length()) {
            if (s.charAt(index) == '0') {
                break;
            }
            ++index;
        }
        while (index < s.length()) {
            if (s.charAt(index) == '1') {
                return false;
            }
            ++index;
        }
        return true;
    }

    // 1784. 检查二进制字符串字段 (Check if Binary String Has at Most One Segment of Ones)
    public boolean checkOnesSegment2(String s) {
        return !s.contains("01");
    }

    // 1286. 字母组合迭代器 (Iterator for Combination)
    class CombinationIterator {
        private int mask;
        private int combinationLength;
        private String characters;

        public CombinationIterator(String characters, int combinationLength) {
            this.mask = (1 << characters.length()) - 1;
            this.combinationLength = combinationLength;
            this.characters = characters;
        }

        public String next() {
            while (Integer.bitCount(mask) != combinationLength) {
                --mask;
            }
            StringBuilder builder = new StringBuilder();
            int copyMask = mask;
            while (copyMask != 0) {
                int index = characters.length() - Integer.numberOfTrailingZeros(copyMask) - 1;
                builder.insert(0, characters.charAt(index));
                copyMask &= copyMask - 1;
            }
            --mask;
            return builder.toString();

        }

        public boolean hasNext() {
            while (mask != 0) {
                if (Integer.bitCount(mask) == combinationLength) {
                    return true;
                }
                --mask;
            }
            return false;
        }
    }

    // 2248. 多个数组求交集 (Intersection of Multiple Arrays)
    public List<Integer> intersection(int[][] nums) {
        int[] counts = new int[1001];
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < nums[i].length; ++j) {
                ++counts[nums[i][j]];
            }
        }
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < counts.length; ++i) {
            if (counts[i] == n) {
                res.add(i);
            }

        }
        return res;

    }

    // 6104. 统计星号
    public int countAsterisks(String s) {
        int n = s.length();
        char[] arr = s.toCharArray();
        int res = 0;
        int count = 0;
        for (int i = 0; i < n; ++i) {
            if (arr[i] == '|') {
                ++count;
            }
            if ((count & 1) == 0 && arr[i] == '*') {
                ++res;
            }
        }
        return res;

    }

    // 2316. 统计无向图中无法互相到达点对数 (Count Unreachable Pairs of Nodes in an Undirected
    // Graph)
    public long countPairs(int n, int[][] edges) {
        UnionFind2316 union = new UnionFind2316(n);
        for (int[] edge : edges) {
            union.union(edge[0], edge[1]);
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            int root = union.getRoot(i);
            map.put(root, map.getOrDefault(root, 0) + 1);
        }
        long res = 0L;
        long pre = 0L;
        for (int cnt : map.values()) {
            res += cnt * pre;
            pre += cnt;
        }
        return res;
    }

    public class UnionFind2316 {
        private int[] parent;
        private int[] rank;

        public UnionFind2316(int n) {
            parent = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
            }
            rank = new int[n];
            Arrays.fill(rank, 1);
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
            if (rank[root1] > rank[root2]) {
                parent[root2] = root1;
            } else {
                parent[root1] = root2;
                if (rank[root1] == rank[root2]) {
                    ++rank[root2];
                }
            }
        }
    }

    // 2316. 统计无向图中无法互相到达点对数 (Count Unreachable Pairs of Nodes in an Undirected
    // Graph)
    private List<Integer>[] g2316;
    private boolean[] vis2316;
    private long cnt2316;

    public long countPairs2(int n, int[][] edges) {
        this.g2316 = new ArrayList[n];
        Arrays.setAll(g2316, k -> new ArrayList<>());
        for (int[] e : edges) {
            g2316[e[0]].add(e[1]);
            g2316[e[1]].add(e[0]);
        }
        this.vis2316 = new boolean[n];
        long res = 0L;
        long pre = 0L;
        for (int i = 0; i < n; ++i) {
            if (!vis2316[i]) {
                cnt2316 = 0L;
                dfs2316(i, -1);
                res += cnt2316 * pre;
                pre += cnt2316;
            }
        }
        return res;

    }

    private void dfs2316(int x, int fa) {
        ++cnt2316;
        vis2316[x] = true;
        for (int y : g2316[x]) {
            if (y != fa && !vis2316[y]) {
                dfs2316(y, x);
            }
        }
    }

    // 6105. 操作后的最大异或和
    public int maximumXOR(int[] nums) {
        int res = 0;
        for (int i = 0; i <= 27; ++i) {
            for (int num : nums) {
                if (((num >> i) & 1) != 0) {
                    res |= 1 << i;
                    break;
                }
            }
        }
        return res;
    }

    // 6105. 操作后的最大异或和
    public int maximumXOR2(int[] nums) {
        int res = 0;
        for (int num : nums) {
            res |= num;
        }
        return res;
    }

    // 2318. 不同骰子序列的数目 (Number of Distinct Roll Sequences)
    private int[][][] memo2318;
    private int n2318;

    public int distinctSequences(int n) {
        this.n2318 = n;
        this.memo2318 = new int[n][7][7];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < 7; ++j) {
                Arrays.fill(memo2318[i][j], -1);
            }
        }
        return dfs2318(0, 0, 0);
    }

    private int dfs2318(int i, int j, int k) {
        if (i == n2318) {
            return 1;
        }
        if (memo2318[i][j][k] != -1) {
            return memo2318[i][j][k];
        }
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        for (int s = 1; s <= 6; ++s) {
            if (s != j && s != k && (i == 0 || gcd2318(s, j) == 1)) {
                res = (res + dfs2318(i + 1, s, j)) % MOD;
            }
        }
        return memo2318[i][j][k] = res;
    }

    private int gcd2318(int a, int b) {
        return b == 0 ? a : gcd2318(b, a % b);
    }


    // 6101. 判断矩阵是否是一个 X 矩阵 (Check if Matrix Is X-Matrix)
    public boolean checkXMatrix(int[][] grid) {
        int n = grid.length;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j || i + j == n - 1) {
                    if (grid[i][j] == 0) {
                        return false;
                    }
                } else {
                    if (grid[i][j] != 0) {
                        return false;
                    }
                }
            }
        }
        return true;

    }

    // 2320. 统计放置房子的方式数 (Count Number of Ways to Place Houses)
    private int n2320;
    private int[] memo2320;

    public int countHousePlacements(int n) {
        this.n2320 = n;
        this.memo2320 = new int[n];
        int res = dfs2320(0);
        final int MOD = (int) (1e9 + 7);
        return (int) ((long) res * res % MOD);

    }

    private int dfs2320(int i) {
        if (i >= n2320) {
            return 1;
        }
        if (memo2320[i] != 0) {
            return memo2320[i];
        }
        final int MOD = (int) (1e9 + 7);
        return memo2320[i] = (dfs2320(i + 1) + dfs2320(i + 2)) % MOD;
    }

    // 2321. 拼接数组的最大分数 (Maximum Score Of Spliced Array)
    private int n2321;
    private int[] nums1_2321;
    private int[] nums2_2321;

    public int maximumsSplicedArray(int[] nums1, int[] nums2) {
        return Math.max(cal2321(nums1, nums2), cal2321(nums2, nums1));

    }

    private int[][] memo2321;

    public int cal2321(int[] nums1, int[] nums2) {
        this.n2321 = nums1.length;
        this.nums1_2321 = nums1;
        this.nums2_2321 = nums2;
        this.memo2321 = new int[n2321][3];
        return dfs2321(0, 0);
    }

    private int dfs2321(int i, int j) {
        if (i == n2321) {
            return 0;
        }
        if (memo2321[i][j] != 0) {
            return memo2321[i][j];
        }
        if (j == 0) {
            return memo2321[i][j] = Math.max(dfs2321(i + 1, j) + nums1_2321[i], dfs2321(i + 1, j + 1) + nums2_2321[i]);
        }
        if (j == 1) {
            return memo2321[i][j] = Math.max(dfs2321(i + 1, j) + nums2_2321[i], dfs2321(i + 1, j + 1) + nums1_2321[i]);
        }
        return memo2321[i][j] = dfs2321(i + 1, j) + nums1_2321[i];
    }

    // 2321. 拼接数组的最大分数 (Maximum Score Of Spliced Array)
    public int maximumsSplicedArray2(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int[] preSum1 = getPreSum5229(nums1);
        int[] preSum2 = getPreSum5229(nums2);
        int res = Math.max(preSum1[n], preSum2[n]);
        res = Math.max(res, getMax5229(preSum1, preSum2));
        res = Math.max(res, getMax5229(preSum2, preSum1));
        return res;

    }

    private int getMax5229(int[] preSum1, int[] preSum2) {
        int res = 0;
        int left = 0;
        int right = 1;
        int diffPre = 0;
        int n = preSum1.length;
        int sum = preSum1[n - 1];
        while (right < n) {
            diffPre = preSum2[right] - preSum2[left] - (preSum1[right] - preSum1[left]);
            if (diffPre > 0) {
                res = Math.max(res, sum + diffPre);
                ++right;
            } else if (left >= right) {
                ++right;
            } else {
                ++left;
            }
        }
        return res;
    }

    private int[] getPreSum5229(int[] nums) {
        int[] preSum = new int[nums.length + 1];
        for (int i = 1; i < preSum.length; ++i) {
            preSum[i] = preSum[i - 1] + nums[i - 1];
        }
        return preSum;
    }

    // 572. 另一棵树的子树 (Subtree of Another Tree) --dfs
    // 面试题 04.10. 检查子树
    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
        if (root == null && subRoot == null) {
            return true;
        }
        if (root == null || subRoot == null) {
            return false;
        }
        return isSameTree572(root, subRoot) || isSubtree(root.left, subRoot) || isSubtree(root.right, subRoot);

    }

    private boolean isSameTree572(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null) {
            return true;
        }
        if (root1 == null || root2 == null) {
            return false;
        }
        return root1.val == root2.val && isSameTree572(root1.left, root2.left)
                && isSameTree572(root1.right, root2.right);
    }

    // 572. 另一棵树的子树 (Subtree of Another Tree) --bfs
    // 面试题 04.10. 检查子树
    public boolean isSubtree2(TreeNode root, TreeNode subRoot) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node.val == subRoot.val && checkIsSameTree572(node, subRoot)) {
                return true;
            }
            if (node.left != null) {
                queue.offer(node.left);
            }
            if (node.right != null) {
                queue.offer(node.right);
            }
        }
        return false;

    }

    private boolean checkIsSameTree572(TreeNode A, TreeNode B) {
        Queue<TreeNode> queueA = new LinkedList<>();
        Queue<TreeNode> queueB = new LinkedList<>();
        queueA.offer(A);
        queueB.offer(B);
        while (!queueA.isEmpty() && !queueB.isEmpty()) {
            TreeNode nodeA = queueA.poll();
            TreeNode nodeB = queueB.poll();
            if (nodeA.val != nodeB.val || (nodeA.left == null) != (nodeB.left == null) || (nodeA.right == null) != (nodeB.right == null)) {
                return false;
            }
            if (nodeA.left != null && nodeB.left != null) {
                queueA.offer(nodeA.left);
                queueB.offer(nodeB.left);
            }
            if (nodeA.right != null && nodeB.right != null) {
                queueA.offer(nodeA.right);
                queueB.offer(nodeB.right);
            }
        }
        return true;
    }

    // 剑指 Offer 26. 树的子结构 --bfs
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if (B == null || A == null) {
            return false;
        }
        Queue<TreeNode> queueA = new LinkedList<>();
        queueA.offer(A);
        while (!queueA.isEmpty()) {
            TreeNode node = queueA.poll();
            if (node.val == B.val && checkOffer26(node, B)) {
                return true;
            }
            if (node.left != null) {
                queueA.offer(node.left);
            }
            if (node.right != null) {
                queueA.offer(node.right);
            }
        }
        return false;

    }

    private boolean checkOffer26(TreeNode A, TreeNode B) {
        Queue<TreeNode> queueA = new LinkedList<>();
        Queue<TreeNode> queueB = new LinkedList<>();
        queueA.offer(A);
        queueB.offer(B);
        while (!queueA.isEmpty() && !queueB.isEmpty()) {
            TreeNode nodeA = queueA.poll();
            TreeNode nodeB = queueB.poll();
            if (nodeA.val != nodeB.val) {
                return false;
            }
            if (nodeB.left != null) {
                if (nodeA.left == null) {
                    return false;
                }
                queueA.offer(nodeA.left);
                queueB.offer(nodeB.left);
            }
            if (nodeB.right != null) {
                if (nodeA.right == null) {
                    return false;
                }
                queueA.offer(nodeA.right);
                queueB.offer(nodeB.right);
            }
        }
        return true;
    }

    // 剑指 Offer 26. 树的子结构 --dfs
    public boolean isSubStructure2(TreeNode A, TreeNode B) {
        if (A == null || B == null) {
            return false;
        }
        return dfsOffer26(A, B) || isSubStructure2(A.left, B) || isSubStructure2(A.right, B);
    }

    private boolean dfsOffer26(TreeNode A, TreeNode B) {
        if (B == null) {
            return true;
        }
        if (A == null || A.val != B.val) {
            return false;
        }
        return dfsOffer26(A.left, B.left) && dfsOffer26(A.right, B.right);
    }

    // 413. 等差数列划分 (Arithmetic Slices)
    public int numberOfArithmeticSlices(int[] nums) {
        if (nums.length <= 1) {
            return 0;
        }
        int d = nums[1] - nums[0];
        int count = 0;
        int res = 0;
        for (int i = 2; i < nums.length; ++i) {
            if (nums[i] - nums[i - 1] == d) {
                ++count;
            } else {
                d = nums[i] - nums[i - 1];
                count = 0;
            }
            res += count;
        }
        return res;

    }

    // LCP 50. 宝石补给
    public int giveGem(int[] gem, int[][] operations) {
        for (int[] operation : operations) {
            int give = gem[operation[0]] / 2;
            gem[operation[0]] -= give;
            gem[operation[1]] += give;
        }
        int max = Arrays.stream(gem).max().getAsInt();
        int min = Arrays.stream(gem).min().getAsInt();
        return max - min;

    }

    // 931. 下降路径最小和 (Minimum Falling Path Sum) --dp
    public int minFallingPathSum(int[][] matrix) {
        int n = matrix.length;
        for (int i = 1; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                int min = Integer.MAX_VALUE;
                for (int k = Math.max(0, j - 1); k <= Math.min(j + 1, n - 1); ++k) {
                    min = Math.min(min, matrix[i - 1][k]);
                }
                matrix[i][j] += min;
            }
        }
        return Arrays.stream(matrix[n - 1]).min().getAsInt();
    }

    // 931. 下降路径最小和 (Minimum Falling Path Sum)
    private int n931;
    private int[][] grid931;
    private int[][] memo931;

    public int minFallingPathSum2(int[][] grid) {
        this.n931 = grid.length;
        this.grid931 = grid;
        this.memo931 = new int[n931][n931];
        for (int i = 0; i < n931; ++i) {
            Arrays.fill(memo931[i], Integer.MAX_VALUE);
        }
        int res = Integer.MAX_VALUE;
        for (int j = 0; j < n931; ++j) {
            res = Math.min(res, dfs931(0, j));
        }
        return res;

    }

    private int dfs931(int i, int j) {
        if (i == n931) {
            return 0;
        }
        if (memo931[i][j] != Integer.MAX_VALUE) {
            return memo931[i][j];
        }
        int min = Integer.MAX_VALUE;
        for (int k = Math.max(0, j - 1); k < Math.min(j + 2, n931); ++k) {
            min = Math.min(min, dfs931(i + 1, k) + grid931[i][j]);
        }
        return memo931[i][j] = min;
    }

    // 786. 第 K 个最小的素数分数 (K-th Smallest Prime Fraction) --暴力
    public int[] kthSmallestPrimeFraction(int[] arr, int k) {
        List<int[]> list = new ArrayList<>();
        int n = arr.length;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                list.add(new int[] { arr[i], arr[j] });
            }
        }
        Collections.sort(list, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] * o2[1] - o1[1] * o2[0];
            }

        });

        return list.get(k - 1);

    }

    // 786. 第 K 个最小的素数分数 (K-th Smallest Prime Fraction) --优先队列
    public int[] kthSmallestPrimeFraction2(int[] arr, int k) {
        int n = arr.length;
        PriorityQueue<int[]> priorityQueue = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return arr[o1[0]] * arr[o2[1]] - arr[o1[1]] * arr[o2[0]];
            }

        });

        priorityQueue.offer(new int[] { 0, n - 1 });
        Set<String> set = new HashSet<>();
        set.add(0 + "_" + (n - 1));
        while (--k > 0 && !priorityQueue.isEmpty()) {
            int[] cur = priorityQueue.poll();
            int index1 = cur[0];
            int index2 = cur[1];
            if (index1 + 1 < index2 && set.add((index1 + 1) + "_" + index2)) {
                priorityQueue.offer(new int[] { index1 + 1, index2 });
            }
            if (index1 < index2 - 1 && set.add(index1 + "_" + (index2 - 1))) {
                priorityQueue.offer(new int[] { index1, index2 - 1 });
            }
        }
        return new int[] { arr[priorityQueue.peek()[0]], arr[priorityQueue.peek()[1]] };
    }

    // 786. 第 K 个最小的素数分数 (K-th Smallest Prime Fraction) --二分查找
    public int[] kthSmallestPrimeFraction3(int[] arr, int k) {
        double left = 0d;
        double right = 1d;
        while (true) {
            double mid = (left + right) / 2;
            int count = 0;
            int x = 0;
            int y = 1;
            int i = -1;
            for (int j = 1; j < arr.length; ++j) {
                while ((double) arr[i + 1] / arr[j] < mid) {
                    ++i;
                    int a = arr[i];
                    int b = arr[j];
                    if (a * y > b * x) {
                        x = a;
                        y = b;
                    }
                }
                count += i + 1;
            }
            if (count == k) {
                return new int[] { x, y };
            }
            if (count < k) {
                left = mid;
            } else {
                right = mid;
            }
        }
    }

    // 508. 出现次数最多的子树元素和 (Most Frequent Subtree Sum)
    public int[] findFrequentTreeSum(TreeNode root) {
        Map<Integer, Integer> map = new HashMap<>();
        dfs508(root, map);
        int max = Collections.max(map.values());
        List<Integer> list = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (entry.getValue() == max) {
                list.add(entry.getKey());
            }
        }
        int n = list.size();
        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            res[i] = list.get(i);
        }
        return res;

    }

    private int dfs508(TreeNode root, Map<Integer, Integer> map) {
        if (root == null) {
            return 0;
        }
        root.val += dfs508(root.left, map) + dfs508(root.right, map);
        map.put(root.val, map.getOrDefault(root.val, 0) + 1);
        return root.val;
    }

    // 1175. 质数排列 (Prime Arrangements)
    public int numPrimeArrangements(int n) {
        final int MOD = 1000000007;
        int primeCount = 0;
        for (int i = 1; i <= n; ++i) {
            if (isPrime1175(i)) {
                ++primeCount;
            }
        }
        return (int) ((factorial1175(primeCount) * factorial1175(n - primeCount)) % MOD);

    }

    private long factorial1175(int n) {
        final int MOD = 1000000007;
        long res = 1l;
        while (n != 0) {
            res *= n;
            res %= MOD;
            --n;
        }
        return res;
    }

    private boolean isPrime1175(int num) {
        if (num == 1) {
            return false;
        }
        for (int i = 2; i * i <= num; ++i) {
            if (num % i == 0) {
                return false;
            }
        }
        return true;
    }

    // 2227. 加密解密字符串 (Encrypt and Decrypt Strings)
    class Encrypter {
        private Map<Character, String> map;
        private Map<String, Integer> count;

        public Encrypter(char[] keys, String[] values, String[] dictionary) {
            map = new HashMap<>();
            count = new HashMap<>();
            int n = keys.length;
            for (int i = 0; i < n; ++i) {
                map.put(keys[i], values[i]);
            }
            search: for (String dic : dictionary) {
                StringBuilder builder = new StringBuilder();
                for (char c : dic.toCharArray()) {
                    if (!map.containsKey(c)) {
                        continue search;
                    }
                    builder.append(map.get(c));
                }
                count.put(builder.toString(), count.getOrDefault(builder.toString(), 0) + 1);
            }

        }

        public String encrypt(String word1) {
            StringBuilder builder = new StringBuilder();
            for (char c : word1.toCharArray()) {
                builder.append(map.get(c));
            }
            return builder.toString();
        }

        public int decrypt(String word2) {
            return count.getOrDefault(word2, 0);
        }
    }

    // 2136. 全部开花的最早一天 (Earliest Possible Day of Full Bloom)
    public int earliestFullBloom(int[] plantTime, int[] growTime) {
        int n = plantTime.length;
        List<Integer> id = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            id.add(i);
        }
        Collections.sort(id, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return growTime[o2] - growTime[o1];
            }

        });

        int res = 0;
        int day = 0;
        for (int i : id) {
            day += plantTime[i];
            res = Math.max(res, day + growTime[i]);
        }
        return res;

    }

    // 2266. 统计打字方案数 (Count Number of Texts)
    private int n2266;
    private char[] p2266;
    private int[] memo2266;

    public int countTexts(String pressedKeys) {
        this.n2266 = pressedKeys.length();
        this.p2266 = pressedKeys.toCharArray();
        this.memo2266 = new int[n2266];
        return dfs2266(0);
    }

    private int dfs2266(int i) {
        if (i == n2266) {
            return 1;
        }
        if (memo2266[i] != 0) {
            return memo2266[i];
        }
        int j = 3;
        if (p2266[i] == '7' || p2266[i] == '9') {
            j = 4;
        }
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        int k = 0;
        while (i + k < n2266 && k < j && p2266[i] == p2266[i + k]) {
            res += dfs2266(i + k + 1);
            res %= MOD;
            ++k;
        }
        return memo2266[i] = res;
    }

    // 2325. 解密消息 (Decode the Message)
    public String decodeMessage(String key, String message) {
        int[] alpha = new int[26];
        Arrays.fill(alpha, -1);
        int index = 0;
        for (char c : key.toCharArray()) {
            if (Character.isLetter(c) && alpha[c - 'a'] == -1) {
                alpha[c - 'a'] = index++;
            }
        }
        int n = message.length();
        char[] res = message.toCharArray();
        for (int i = 0; i < n; ++i) {
            if (Character.isLetter(res[i])) {
                res[i] = (char) (alpha[res[i] - 'a'] + 'a');
            }
        }
        return String.valueOf(res);

    }

    public class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    // 6111. 螺旋矩阵 IV
    public int[][] spiralMatrix(int m, int n, ListNode head) {
        int[][] res = new int[m][n];
        int r1 = 0;
        int r2 = m - 1;
        int c1 = 0;
        int c2 = n - 1;
        for (int i = 0; i < m; ++i) {
            Arrays.fill(res[i], -1);
        }
        while (r1 <= r2 && c1 <= c2) {
            for (int c = c1; c <= c2; ++c) {
                if (head != null) {
                    res[r1][c] = head.val;
                    head = head.next;
                } else {
                    return res;
                }
            }
            for (int r = r1 + 1; r <= r2; ++r) {
                if (head != null) {
                    res[r][c2] = head.val;
                    head = head.next;
                } else {
                    return res;
                }
            }
            for (int c = c2 - 1; c >= c1; --c) {
                if (head != null) {
                    res[r2][c] = head.val;
                    head = head.next;
                } else {
                    return res;
                }
            }
            for (int r = r2 - 1; r >= r1 + 1; --r) {
                if (head != null) {
                    res[r][c1] = head.val;
                    head = head.next;
                } else {
                    return res;
                }
            }
            ++r1;
            --r2;
            ++c1;
            --c2;
        }
        return res;

    }

    // 6109. 知道秘密的人数 --dp
    public int peopleAwareOfSecret(int n, int delay, int forget) {
        final int MOD = 1000000007;
        long[] sum = new long[n + 1];
        sum[1] = 1l;
        for (int i = 2; i <= n; ++i) {
            long diff = (sum[Math.max(i - delay, 0)] - sum[Math.max(i - forget, 0)]) % MOD;
            sum[i] = (sum[i - 1] + diff) % MOD;
        }
        return (int) (((sum[n] - sum[Math.max(0, n - forget)]) % MOD + MOD) % MOD);

    }

    // 2328. 网格图中递增路径的数目 -- 记忆化搜索
    private int[][] memo2328;
    private final int MOD2328 = (int) (1e9 + 7);
    private int[][] grid2328;
    private int m2328;
    private int n2328;
    private int[][] directions2328 = { { 0, -1 }, { 0, 1 }, { 1, 0 }, { -1, 0 } };

    public int countPaths(int[][] grid) {
        this.m2328 = grid.length;
        this.n2328 = grid[0].length;
        this.grid2328 = grid;
        memo2328 = new int[m2328][n2328];
        for (int i = 0; i < m2328; ++i) {
            Arrays.fill(memo2328[i], -1);
        }
        int res = 0;
        for (int i = 0; i < m2328; ++i) {
            for (int j = 0; j < n2328; ++j) {
                res = (res + dfs2328(i, j)) % MOD2328;
            }
        }
        return res;

    }

    private int dfs2328(int i, int j) {
        if (memo2328[i][j] != -1) {
            return memo2328[i][j];
        }
        memo2328[i][j] = 1;
        for (int[] d : directions2328) {
            int nx = i + d[0];
            int ny = j + d[1];
            if (nx >= 0 && nx < m2328 && ny >= 0 && ny < n2328) {
                if (grid2328[nx][ny] < grid2328[i][j]) {
                    memo2328[i][j] = (memo2328[i][j] + dfs2328(nx, ny)) % MOD2328;
                }
            }
        }
        return memo2328[i][j];
    }

    // 1143. 最长公共子序列 (Longest Common Subsequence) --记忆化搜索
    // 剑指 Offer II 095. 最长公共子序列
    private int[][] memo1143;
    private String text1_1143;
    private String text2_1143;

    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length();
        int n = text2.length();
        memo1143 = new int[m][n];
        this.text1_1143 = text1;
        this.text2_1143 = text2;
        for (int i = 0; i < m; ++i) {
            Arrays.fill(memo1143[i], -1);
        }
        return dfs1143(m - 1, n - 1);

    }

    private int dfs1143(int i, int j) {
        if (i < 0 || j < 0) {
            return 0;
        }
        if (memo1143[i][j] != -1) {
            return memo1143[i][j];
        }
        if (text1_1143.charAt(i) == text2_1143.charAt(j)) {
            return memo1143[i][j] = dfs1143(i - 1, j - 1) + 1;
        }
        return memo1143[i][j] = Math.max(dfs1143(i, j - 1), dfs1143(i - 1, j));
    }

    // 1143. 最长公共子序列 (Longest Common Subsequence) --二维dp
    // 剑指 Offer II 095. 最长公共子序列
    public int longestCommonSubsequence2(String text1, String text2) {
        int m = text1.length();
        int n = text2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i < m + 1; ++i) {
            char c1 = text1.charAt(i - 1);
            for (int j = 1; j < n + 1; ++j) {
                char c2 = text2.charAt(j - 1);
                if (c1 == c2) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }

    // 1143. 最长公共子序列 (Longest Common Subsequence) --一维dp
    // 剑指 Offer II 095. 最长公共子序列
    public int longestCommonSubsequence3(String text1, String text2) {
        int m = text1.length();
        int n = text2.length();
        int[][] dp = new int[2][n + 1];
        for (int i = 1; i < m + 1; ++i) {
            char c1 = text1.charAt(i - 1);
            for (int j = 1; j < n + 1; ++j) {
                char c2 = text2.charAt(j - 1);
                if (c1 == c2) {
                    dp[1][j] = dp[0][j - 1] + 1;
                } else {
                    dp[1][j] = Math.max(dp[1][j - 1], dp[0][j]);
                }
            }
            System.arraycopy(dp[1], 0, dp[0], 0, n + 1);
            Arrays.fill(dp[1], 0);
        }
        return dp[0][n];
    }

    // 556. 下一个更大元素 III (Next Greater Element III)
    public int nextGreaterElement(int n) {
        char[] chars = String.valueOf(n).toCharArray();
        int len = chars.length;
        int i = len - 2;
        while (i >= 0) {
            if (chars[i] < chars[i + 1]) {
                break;
            }
            --i;
        }
        if (i == -1) {
            return -1;
        }
        int j = len - 1;
        while (i < j) {
            if (chars[i] < chars[j]) {
                char temp = chars[i];
                chars[i] = chars[j];
                chars[j] = temp;
                Arrays.sort(chars, i + 1, len);
                break;
            }
            --j;
        }

        String candicate = String.valueOf(chars);
        String max = String.valueOf(Integer.MAX_VALUE);
        if (candicate.length() == max.length() && candicate.compareTo(max) > 0) {
            return -1;
        }
        return Integer.parseInt(candicate);

    }

    // 面试题 05.04. 下一个数 --还需掌握位运算
    public int[] findClosedNumbers(int num) {
        char[] chars = Integer.toBinaryString(num).toCharArray();
        return new int[] { getNextSmallest(chars.clone()), getPreLargest(chars.clone()) };
    }

    private int getPreLargest(char[] chars) {
        int n = chars.length;
        int i = n - 2;
        while (i >= 0) {
            if (chars[i] > chars[i + 1]) {
                break;
            }
            --i;
        }
        if (i < 0) {
            return -1;
        }
        int j = i + 1;
        while (j < n) {
            if (chars[i] > chars[j]) {
                chars[i] = '0';
                chars[j] = '1';
                Arrays.sort(chars, i + 1, n);
                break;
            }
            ++j;
        }
        int left = i + 1;
        int right = n - 1;
        while (left < right) {
            char temp = chars[left];
            chars[left] = chars[right];
            chars[right] = temp;
            ++left;
            --right;
        }

        return binaryToDecimal(chars);
    }

    private int binaryToDecimal(char[] chars) {
        int n = chars.length;
        int res = 0;
        for (int index = n - 1; index >= 0; --index) {
            if (chars[index] == '1') {
                res += (chars[index] - '0') << (n - 1 - index);
            }
        }
        return res;
    }

    private int getNextSmallest(char[] chars) {
        int n = chars.length;
        int i = n - 2;
        while (i >= 0) {
            if (chars[i] < chars[i + 1]) {
                break;
            }
            --i;
        }
        if (i < 0) {
            if (n == 31) {
                return -1;
            }
            char[] newchars = new char[n + 1];
            System.arraycopy(chars, 0, newchars, 0, n);
            Arrays.sort(newchars, 1, n + 1);
            return binaryToDecimal(newchars);
        }
        int j = n - 1;
        while (i < j) {
            if (chars[i] < chars[j]) {
                chars[i] = '1';
                chars[j] = '0';
                Arrays.sort(chars, i + 1, n);
                break;
            }
            --j;
        }

        return binaryToDecimal(chars);
    }

    // 274. H 指数 (H-Index)
    public int hIndex(int[] citations) {
        Arrays.sort(citations);
        int i = citations.length - 1;
        int h = 0;
        while (i >= 0 && citations[i] > h) {
            ++h;
            --i;
        }
        return h;
    }

    // 275. H 指数 II (H-Index II)
    public int hIndex275(int[] citations) {
        int i = citations.length - 1;
        int h = 0;
        while (i >= 0 && citations[i] > h) {
            ++h;
            --i;
        }
        return h;
    }

    // 1641. 统计字典序元音字符串的数目
    private int[][] memo1641;
    private int n1641;

    public int countVowelStrings(int n) {
        this.n1641 = n;
        this.memo1641 = new int[n][5];
        return dfs1641(0, 0);
    }

    private int dfs1641(int i, int letter) {
        if (i == n1641) {
            return 1;
        }
        if (memo1641[i][letter] != 0) {
            return memo1641[i][letter];
        }
        int res = 0;
        for (int j = letter; j < 5; ++j) {
            res += dfs1641(i + 1, j);
        }
        return memo1641[i][letter] = res;
    }

    // 1641. 统计字典序元音字符串的数目 (Count Sorted Vowel Strings) --dp
    public int countVowelStrings2(int n) {
        int[] dp = { 1, 1, 1, 1, 1 };
        for (int i = 1; i < n; ++i) {
            for (int j = 3; j >= 0; --j) {
                dp[j] += dp[j + 1];
            }
        }
        return Arrays.stream(dp).sum();

    }

    // 1638. 统计只差一个字符的子串数目 (Count Substrings That Differ by One Character) --枚举
    public int countSubstrings(String s, String t) {
        int res = 0;
        int m = s.length();
        int n = t.length();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int diff = 0;
                for (int k = 0; i + k < m && j + k < n; ++k) {
                    if (s.charAt(i + k) != t.charAt(j + k)) {
                        ++diff;
                    }
                    if (diff == 1) {
                        ++res;
                    }
                    if (diff > 1) {
                        break;
                    }
                }
            }
        }
        return res;

    }

    // 1638. 统计只差一个字符的子串数目 (Count Substrings That Differ by One Character) --dp
    public int countSubstrings2(String s, String t) {
        int m = s.length();
        int n = t.length();
        // commonSuffixLength[i][j] 表示以s[i]、t[j] 为结尾的两个字符串的公共后缀长度
        int[][] commonSuffixLength = new int[m][n];
        // dp 表示以s[i]、t[j] 为结尾的两个字符串只有一个字符不同的数目
        int[][] dp = new int[m][n];

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == 0 || j == 0) {
                    if (s.charAt(i) == t.charAt(j)) {
                        commonSuffixLength[i][j] = 1;
                    }
                } else {
                    if (s.charAt(i) == t.charAt(j)) {
                        commonSuffixLength[i][j] = commonSuffixLength[i - 1][j - 1] + 1;
                    }
                }
            }
        }
        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == 0 || j == 0) {
                    if (s.charAt(i) != t.charAt(j)) {
                        dp[i][j] = 1;
                    }
                } else {
                    if (s.charAt(i) == t.charAt(j)) {
                        dp[i][j] = dp[i - 1][j - 1];
                    } else {
                        dp[i][j] = commonSuffixLength[i - 1][j - 1] + 1;
                    }
                }
                res += dp[i][j];
            }
        }
        return res;

    }

    // 1452. 收藏清单 (People Whose List of Favorite Companies Is Not a Subset of
    // Another List)
    public List<Integer> peopleIndexes(List<List<String>> favoriteCompanies) {
        Set<Integer> notAns = new HashSet<>();
        int n = favoriteCompanies.size();
        for (int i = 0; i < n; ++i) {
            Set<String> set = new HashSet<>(favoriteCompanies.get(i));
            search: for (int j = 0; j < n; ++j) {
                if (i == j) {
                    continue;
                }
                for (String item : favoriteCompanies.get(j)) {
                    if (!set.contains(item)) {
                        continue search;
                    }
                }
                notAns.add(j);
            }
        }
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            if (!notAns.contains(i)) {
                res.add(i);
            }
        }
        return res;

    }

    // 1796. 字符串中第二大的数字 (Second Largest Digit in a String)
    public int secondHighest(String s) {
        int mask = 0;
        for (char c : s.toCharArray()) {
            if (Character.isDigit(c)) {
                mask |= 1 << (c - '0');
            }
        }
        int max = -1;
        int secondMax = -1;
        while (mask != 0) {
            int last = mask & (-mask);
            int num = Integer.bitCount(last - 1);
            secondMax = max;
            max = num;
            mask &= mask - 1;
        }
        return secondMax;

    }

    // 1796. 字符串中第二大的数字 (Second Largest Digit in a String)
    public int secondHighest2(String s) {
        int max1 = -1;
        int max2 = -1;
        for (char c : s.toCharArray()) {
            if (Character.isDigit(c)) {
                int num = c - '0';
                if (num > max1) {
                    max2 = max1;
                    max1 = num;
                } else if (max1 > num && num > max2) {
                    max2 = num;
                }
            }
        }
        return max2;

    }

    // 522. 最长特殊序列 II (Longest Uncommon Subsequence II)
    public int findLUSlength(String[] strs) {
        Arrays.sort(strs, new Comparator<String>() {

            @Override
            public int compare(String o1, String o2) {
                return Integer.compare(o2.length(), o1.length());
            }

        });
        search: for (int i = 0; i < strs.length; ++i) {
            for (int j = 0; j < strs.length; ++j) {
                if (i == j) {
                    continue;
                }
                if (check522(strs[i], strs[j])) {
                    continue search;
                }
            }
            return strs[i].length();
        }
        return -1;

    }

    private boolean check522(String s, String t) {
        int i = 0;
        int j = 0;
        while (i < s.length() && j < t.length()) {
            if (s.charAt(i) == t.charAt(j)) {
                ++i;
            }
            ++j;
        }
        return i == s.length();
    }

    // 609. 在系统中查找重复文件 (Find Duplicate File in System)
    public List<List<String>> findDuplicate(String[] paths) {
        Map<String, List<String>> map = new HashMap<>();
        for (String path : paths) {
            String[] split = path.split("\\s+");
            String directory = split[0];
            for (int i = 1; i < split.length; ++i) {
                // 左括号索引
                int leftParentheseIndex = split[i].indexOf("(");
                // 文件名
                String fileName = split[i].substring(0, leftParentheseIndex);
                // 文件内容
                String fileContent = split[i].substring(leftParentheseIndex + 1, split[i].length() - 1);
                map.computeIfAbsent(fileContent, k -> new ArrayList<>()).add(directory + "/" + fileName);
            }
        }
        List<List<String>> res = new ArrayList<>();
        for (List<String> list : map.values()) {
            if (list.size() > 1) {
                res.add(list);
            }
        }
        return res;
    }

    // 1072. 按列翻转得到最大值等行数 (Flip Columns For Maximum Number of Equal Rows)
    public int maxEqualRowsAfterFlips(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < m; ++i) {
            StringBuilder b = new StringBuilder();
            for (int j = 0; j < n; ++j) {
                b.append(matrix[i][0] ^ matrix[i][j]);
            }
            map.merge(b.toString(), 1, Integer::sum);
        }
        return Collections.max(map.values());

    }

    // 2249. 统计圆内格点数目 (Count Lattice Points Inside a Circle)
    public int countLatticePoints(int[][] circles) {
        Set<Integer> set = new HashSet<>();
        for (int[] circle : circles) {
            int x = circle[0];
            int y = circle[1];
            int r = circle[2];
            for (int nx = x; nx <= x + r; ++nx) {
                for (int ny = y; ny <= y + r; ++ny) {
                    if ((nx - x) * (nx - x) + (ny - y) * (ny - y) <= r * r) {
                        set.add(nx * 200 + ny);
                        set.add((2 * x - nx) * 200 + ny);
                        set.add(nx * 200 + 2 * y - ny);
                        set.add((2 * x - nx) * 200 + 2 * y - ny);
                    } else {
                        break;
                    }
                }
            }
        }
        return set.size();

    }

    // 2013. 检测正方形 (Detect Squares)
    class DetectSquares {
        private Map<Integer, Integer> map;

        public DetectSquares() {
            map = new HashMap<>();
        }

        public void add(int[] point) {
            int x = point[0];
            int y = point[1];
            map.put((x << 10) + y, map.getOrDefault((x << 10) + y, 0) + 1);
        }

        public int count(int[] point) {
            int res = 0;
            int x = point[0];
            int y = point[1];
            for (int p : map.keySet()) {
                int x0 = p >> 10;
                int y0 = p & 1023;
                if (y != y0) {
                    continue;
                }
                if (x == x0) {
                    continue;
                }
                int side = Math.abs(x - x0);
                res += map.getOrDefault((x0 << 10) + y, 0) * map.getOrDefault((x0 << 10) + y + side, 0)
                        * map.getOrDefault((x << 10) + y + side, 0);

                res += map.getOrDefault((x0 << 10) + y, 0) * map.getOrDefault((x0 << 10) + y - side, 0)
                        * map.getOrDefault((x << 10) + y - side, 0);
            }
            return res;
        }
    }

    // 1455. 检查单词是否为句中其他单词的前缀 (Check If a Word Occurs As a Prefix of Any Word in a
    // Sentence)
    public int isPrefixOfWord(String sentence, String searchWord) {
        String[] split = sentence.split("\\s+");
        for (int i = 0; i < split.length; ++i) {
            if (split[i].indexOf(searchWord) == 0) {
                return i + 1;
            }
        }
        return -1;
    }

    // 1455. 检查单词是否为句中其他单词的前缀 (Check If a Word Occurs As a Prefix of Any Word in a
    // Sentence)
    public int isPrefixOfWord2(String sentence, String searchWord) {
        int left = 0;
        int right = 0;
        int count = 0;
        while (left < sentence.length()) {
            while (right < sentence.length() && sentence.charAt(right) != ' ') {
                ++right;
            }
            ++count;
            String s = sentence.substring(left, right);
            if (s.startsWith(searchWord)) {
                return count;
            }
            left = right + 1;
            right = left;
        }
        return -1;

    }

    // 1455. 检查单词是否为句中其他单词的前缀 (Check If a Word Occurs As a Prefix of Any Word in a
    // Sentence)
    public int isPrefixOfWord3(String sentence, String searchWord) {
        Trie1455 trie = new Trie1455();
        String[] split = sentence.split("\\s+");
        for (int i = 0; i < split.length; ++i) {
            trie.insert(split[i], i + 1);
        }
        return trie.getPrefixIndex(searchWord);
    }

    class Trie1455 {
        private Trie1455[] children;
        private int index;

        public Trie1455() {
            children = new Trie1455[26];
            index = -1;
        }

        public void insert(String s, int index) {
            Trie1455 node = this;
            for (char c : s.toCharArray()) {
                int i = c - 'a';
                if (node.children[i] == null) {
                    node.children[i] = new Trie1455();
                    node.children[i].index = index;
                }
                node = node.children[i];
            }
        }

        public int getPrefixIndex(String s) {
            Trie1455 node = this;
            for (char c : s.toCharArray()) {
                int i = c - 'a';
                if (node.children[i] == null) {
                    return -1;
                }
                node = node.children[i];
            }
            return node.index;
        }
    }

    // 1668. 最大重复子字符串 (Maximum Repeating Substring)
    public int maxRepeating(String sequence, String word) {
        StringBuilder builder = new StringBuilder(word);
        int res = 0;
        while (sequence.indexOf(builder.toString()) != -1) {
            ++res;
            builder.append(word);
        }
        return res;

    }

    // 1807. 替换字符串中的括号内容 (Evaluate the Bracket Pairs of a String)
    public String evaluate(String s, List<List<String>> knowledge) {
        Map<String, String> map = new HashMap<>();
        for (List<String> k : knowledge) {
            map.put(k.get(0), k.get(1));
        }
        int n = s.length();
        StringBuilder res = new StringBuilder();
        int i = 0;
        while (i < n) {
            if (s.charAt(i) == '(') {
                int j = i + 1;
                while (s.charAt(j) != ')') {
                    ++j;
                }
                String key = s.substring(i + 1, j);
                res.append(map.getOrDefault(key, "?"));
                i = j + 1;
            } else {
                res.append(s.charAt(i++));
            }
        }
        return res.toString();

    }

    // 652. 寻找重复的子树 (Find Duplicate Subtrees) --序列化
    public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
        Map<String, Integer> map = new HashMap<>();
        List<TreeNode> res = new ArrayList<>();
        collect652(root, map, res);
        return res;
    }

    private String collect652(TreeNode root, Map<String, Integer> map, List<TreeNode> res) {
        if (root == null) {
            return "#";
        }
        String key = root.val + "," + collect652(root.left, map, res) + "," + collect652(root.right, map, res);
        map.put(key, map.getOrDefault(key, 0) + 1);
        if (map.get(key) == 2) {
            res.add(root);
        }
        return key;
    }

    // 652. 寻找重复的子树 (Find Duplicate Subtrees)
    private int id652;

    public List<TreeNode> findDuplicateSubtrees2(TreeNode root) {
        id652 = 1;
        Map<String, Integer> map = new HashMap<>();
        Map<Integer, Integer> count = new HashMap<>();
        List<TreeNode> res = new ArrayList<>();
        collect652_2(root, map, count, res);
        return res;

    }

    private int collect652_2(TreeNode root, Map<String, Integer> map, Map<Integer, Integer> count,
            List<TreeNode> res) {
        if (root == null) {
            return 0;
        }
        String key = root.val + "," + collect652_2(root.left, map, count, res) + ","
                + collect652_2(root.right, map, count, res);
        int uid = map.computeIfAbsent(key, k -> id652++);
        count.put(uid, count.getOrDefault(uid, 0) + 1);
        if (count.get(uid) == 2) {
            res.add(root);
        }
        return uid;
    }

    // 2131. 连接两字母单词得到的最长回文串 (Longest Palindrome by Concatenating Two Letter Words)
    public int longestPalindrome(String[] words) {
        Map<String, Integer> map = new HashMap<>();
        for (String word : words) {
            map.put(word, map.getOrDefault(word, 0) + 1);
        }
        boolean hasOddSame = false;
        int countSame = 0;
        int countDiff = 0;
        for (Map.Entry<String, Integer> entry : map.entrySet()) {
            String key = entry.getKey();
            if (key.charAt(0) == key.charAt(1)) {
                if (entry.getValue() % 2 == 1) {
                    hasOddSame = true;
                    countSame += entry.getValue() - 1;
                } else {
                    countSame += entry.getValue();
                }
            } else {
                int count1 = entry.getValue();
                String reverse = String.valueOf(key.charAt(1)) + String.valueOf(key.charAt(0));
                int count2 = map.getOrDefault(reverse, 0);
                countDiff += Math.min(count1, count2);
            }
        }
        countSame += hasOddSame ? 1 : 0;
        return (countDiff + countSame) * 2;

    }

    // 2135. 统计追加字母可以获得的单词数 (Count Words Obtained After Adding a Letter)
    public int wordCount(String[] startWords, String[] targetWords) {
        Set<Integer> set = new HashSet<>();
        for (String s : startWords) {
            int m = 0;
            for (char c : s.toCharArray()) {
                m |= 1 << (c - 'a');
            }
            set.add(m);
        }
        int res = 0;
        search: for (String t : targetWords) {
            int m = 0;
            for (char c : t.toCharArray()) {
                m |= 1 << (c - 'a');
            }
            int c = m;
            while (c != 0) {
                int i = Integer.numberOfTrailingZeros(c);
                if (set.contains(m ^ (1 << i))) {
                    ++res;
                    continue search;
                }
                c &= c - 1;
            }
        }
        return res;
    }

    // 2121. 相同元素的间隔之和 (Intervals Between Identical Elements) --超时
    public long[] getDistances(int[] arr) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        int n = arr.length;
        long[] res = new long[n];
        for (int i = 0; i < n; ++i) {
            List<Integer> list = map.getOrDefault(arr[i], new ArrayList<>());
            for (int j = 0; j < list.size(); ++j) {
                res[list.get(j)] += i - list.get(j);
                res[i] += i - list.get(j);
            }
            list.add(i);
            map.put(arr[i], list);
        }
        return res;

    }

    // 2121. 相同元素的间隔之和 (Intervals Between Identical Elements) --前缀和
    public long[] getDistances2(int[] arr) {
        int n = arr.length;
        Map<Integer, int[]> map = new HashMap<>();
        // 前缀和
        // pre[i] 表示 索引i之前 所有值为arr[i]的元素 到i位置的间隔之和
        long[] pre = new long[n];
        for (int i = 0; i < n; ++i) {
            // cur[0] 表示 上一个值为arr[i]的索引
            // cur[1] 表示 索引i的前面，值为arr[i]的个数
            int[] cur = map.getOrDefault(arr[i], new int[2]);
            if (cur[1] != 0) {
                pre[i] += pre[cur[0]] + (i - cur[0]) * cur[1];
            }
            cur[0] = i;
            ++cur[1];
            map.put(arr[i], cur);
        }

        map.clear();
        long[] suf = new long[n];
        for (int i = n - 1; i >= 0; --i) {
            int[] cur = map.getOrDefault(arr[i], new int[2]);
            if (cur[1] != 0) {
                suf[i] += suf[cur[0]] + (cur[0] - i) * cur[1];
            }
            cur[0] = i;
            ++cur[1];
            map.put(arr[i], cur);
        }
        long[] res = new long[n];
        for (int i = 0; i < n; ++i) {
            res[i] = pre[i] + suf[i];
        }
        return res;

    }

    // 面试题 16.22. 兰顿蚂蚁 (Langtons Ant LCCI)
    public List<String> printKMoves(int K) {
        // false 白
        // true 黑
        boolean[][] board = new boolean[3000][3000];
        int x = 2000;
        int y = 2000;
        // 当前位置
        int minX = x;
        int maxX = x;
        int minY = y;
        int maxY = y;
        // 右、下、左、上
        int[][] directions = { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 } };
        char[] dir = { 'R', 'D', 'L', 'U' };
        char[] color = { '_', 'X' };
        // 当前方向
        int p = 0;
        for (int i = 0; i < K; ++i) {
            // true 当前为黑色
            if (board[x][y]) {
                // 变白
                board[x][y] = false;
                // 逆时针
                p = (p - 1 + 4) % 4;
            }
            // false 当前为白色
            else {
                // 变黑
                board[x][y] = true;
                // 顺时针
                p = (p + 1) % 4;
            }
            x += directions[p][0];
            y += directions[p][1];
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
        }
        List<String> res = new ArrayList<>();
        for (int i = minX; i <= maxX; ++i) {
            StringBuilder builder = new StringBuilder();
            for (int j = minY; j <= maxY; ++j) {
                if (x == i && y == j) {
                    builder.append(dir[p]);
                } else {
                    builder.append(color[board[i][j] ? 1 : 0]);
                }
            }
            res.add(builder.toString());
        }
        return res;

    }

    // 957. N 天后的牢房 (Prison Cells After N Days)
    public int[] prisonAfterNDays(int[] cells, int n) {
        Map<Integer, Integer> map = new HashMap<>();
        int state = 0;
        for (int i = 0; i < 8; ++i) {
            state |= cells[i] << i;
        }
        while (n > 0) {
            if (map.containsKey(state)) {
                n %= map.get(state) - n;
            }
            map.put(state, n);
            if (n >= 1) {
                state = getNextState957(state);
                --n;
            }
        }
        int[] res = new int[8];
        for (int i = 1; i < 7; ++i) {
            res[i] = (state >> i) & 1;
        }
        return res;

    }

    private int getNextState957(int state) {
        int res = 0;
        for (int i = 1; i < 7; ++i) {
            if (((state >> (i - 1)) & 1) == ((state >> (i + 1)) & 1)) {
                res |= 1 << i;
            }
        }
        return res;
    }

    // 1325. 删除给定值的叶子节点 (Delete Leaves With a Given Value) --bfs + 拓扑排序
    public TreeNode removeLeafNodes(TreeNode root, int target) {
        Queue<TreeNode> queue = new LinkedList<>();
        Map<TreeNode, TreeNode> map = new HashMap<>();
        Map<TreeNode, Integer> degree = new HashMap<>();
        Queue<TreeNode> topologicalQueue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node.left == null && node.right == null && node.val == target) {
                topologicalQueue.offer(node);
            }
            if (node.left != null) {
                map.put(node.left, node);
                degree.put(node, degree.getOrDefault(node, 0) + 1);
                queue.offer(node.left);
            }
            if (node.right != null) {
                map.put(node.right, node);
                degree.put(node, degree.getOrDefault(node, 0) + 1);
                queue.offer(node.right);
            }
        }
        while (!topologicalQueue.isEmpty()) {
            TreeNode node = topologicalQueue.poll();
            TreeNode parent = map.get(node);
            if (parent == null) {
                return null;
            }
            if (parent.left == node) {
                parent.left = null;
            } else {
                parent.right = null;
            }
            degree.put(parent, degree.get(parent) - 1);
            if (degree.get(parent) == 0 && parent.val == target) {
                topologicalQueue.offer(parent);
            }
        }
        return root;

    }

    // 1325. 删除给定值的叶子节点 (Delete Leaves With a Given Value) --递归
    public TreeNode removeLeafNodes2(TreeNode root, int target) {
        if (root == null) {
            return null;
        }
        root.left = removeLeafNodes(root.left, target);
        root.right = removeLeafNodes(root.right, target);
        if (root.left == null && root.right == null && root.val == target) {
            return null;
        }
        return root;

    }

    // 519. 随机翻转矩阵 (Random Flip Matrix)
    class Solution {
        private Random random;
        private int count;
        private Map<Integer, Integer> map;
        private int m;
        private int n;

        public Solution(int m, int n) {
            count = m * n;
            random = new Random();
            map = new HashMap<>();
            this.m = m;
            this.n = n;

        }

        public int[] flip() {
            int x = random.nextInt(count);
            --count;
            int index = map.getOrDefault(x, x);
            map.put(x, map.getOrDefault(count, count));
            return new int[] { index / n, index % n };
        }

        public void reset() {
            map.clear();
            count = m * n;
        }
    }

    // 1592. 重新排列单词间的空格 (Rearrange Spaces Between Words)
    public String reorderSpaces(String text) {
        int n = text.length();
        int space = 0;
        List<String> words = new ArrayList<>();
        int left = 0;
        int right = 0;
        while (right < n) {
            while (right < n && !Character.isLetter(text.charAt(right))) {
                ++right;
                ++space;
            }
            left = right;
            while (right < n && Character.isLetter(text.charAt(right))) {
                ++right;
            }
            if (left != right) {
                words.add(text.substring(left, right));
            }
            left = right;
        }
        StringBuilder builder = new StringBuilder();
        int mod = space;
        int tab = 0;
        if (words.size() != 1) {
            tab = space / (words.size() - 1);
            mod = space % (words.size() - 1);
        }
        for (String word : words) {
            if (!builder.isEmpty()) {
                int count = tab;
                while (count-- > 0) {
                    builder.append(" ");
                }
            }
            builder.append(word);
        }
        while (mod-- > 0) {
            builder.append(" ");
        }
        return builder.toString();
    }

    // 1736. 替换隐藏数字得到的最晚时间 (Latest Time by Replacing Hidden Digits)
    public String maximumTime(String time) {
        char[] chars = time.toCharArray();
        // 小时
        if (chars[0] == '?' && chars[1] == '?') {
            chars[0] = '2';
            chars[1] = '3';
        } else if (chars[0] == '?') {
            if (chars[1] >= '0' && chars[1] <= '3') {
                chars[0] = '2';
            } else {
                chars[0] = '1';
            }
        } else if (chars[1] == '?') {
            if (chars[0] == '0' || chars[0] == '1') {
                chars[1] = '9';
            } else {
                chars[1] = '3';
            }
        }
        // 分钟
        if (chars[3] == '?') {
            chars[3] = '5';
        }
        if (chars[4] == '?') {
            chars[4] = '9';
        }
        return String.valueOf(chars);

    }

    // 6116. 计算布尔二叉树的值
    public boolean evaluateTree(TreeNode root) {
        if (root.left == null && root.right == null) {
            return root.val != 0;
        }
        boolean left = evaluateTree(root.left);
        boolean right = evaluateTree(root.right);
        if (root.val == 2) {
            return left | right;
        }
        return left & right;

    }

    // 6117. 坐上公交的最晚时间
    public int latestTimeCatchTheBus(int[] buses, int[] passengers, int capacity) {
        Arrays.sort(buses);
        Arrays.sort(passengers);
        int j = 0;
        int c = 0;
        for (int i = 0; i < buses.length; ++i) {
            for (c = capacity; c > 0 && j < passengers.length && passengers[j] <= buses[i]; ++j) {
                --c;
            }
        }
        --j;
        int res = c > 0 ? buses[buses.length - 1] : passengers[j];
        while (j >= 0 && res == passengers[j--]) {
            --res;
        }
        return res;

    }

    // 6118. 最小差值平方和 (Minimum Sum of Squared Difference)
    public long minSumSquareDiff(int[] nums1, int[] nums2, int k1, int k2) {
        Map<Integer, Integer> map = new HashMap<>();
        int n = nums1.length;
        int max = 0;
        long sum = 0l;
        for (int i = 0; i < n; ++i) {
            int diff = Math.abs(nums1[i] - nums2[i]);
            map.put(diff, map.getOrDefault(diff, 0) + 1);
            max = Math.max(max, diff);
            sum += diff;
        }
        int k = k1 + k2;
        if (k >= sum) {
            return 0;
        }
        while (k > 0) {
            int count = map.get(max);
            if (k >= count) {
                map.remove(max);
                --max;
                map.put(max, map.getOrDefault(max, 0) + count);
                k -= count;
            } else {
                int diff = count - k;
                map.put(max, diff);
                --max;
                map.put(max, map.getOrDefault(max, 0) + k);
                k = 0;
            }
        }
        long res = 0l;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            res += (long) entry.getKey() * entry.getKey() * entry.getValue();
        }
        return res;

    }

    // 2335. 装满杯子需要的最短总时长 (Minimum Amount of Time to Fill Cups)
    public int fillCups(int[] amount) {
        int res = 0;
        for (Arrays.sort(amount); amount[1] > 0; Arrays.sort(amount)) {
            --amount[1];
            --amount[2];
            ++res;
        }
        return res + amount[2];

    }

    // 2336. 无限集中的最小数字 (Smallest Number in Infinite Set)
    class SmallestInfiniteSet {
        private int min;
        private TreeSet<Integer> set;

        public SmallestInfiniteSet() {
            this.min = 1;
            this.set = new TreeSet<>();
        }

        public int popSmallest() {
            if (set.isEmpty()) {
                int res = min;
                ++min;
                return res;
            }
            return set.pollFirst();
        }

        public void addBack(int num) {
            if (num < min) {
                set.add(num);
            }
        }

    }

    // 2337. 移动片段得到字符串 (Move Pieces to Obtain a String)
    public boolean canChange(String start, String target) {
        if (!start.replaceAll("_", "").equals(target.replaceAll("_", ""))) {
            return false;
        }
        for (int i = 0, j = 0; i < start.length(); ++i) {
            if (start.charAt(i) == '_') {
                continue;
            }
            while (target.charAt(j) == '_') {
                ++j;
            }
            if (i != j && (start.charAt(i) == 'L') == (i < j)) {
                return false;
            }
            ++j;
        }
        return true;

    }

    // 1487. 保证文件名唯一 (Making File Names Unique)
    public String[] getFolderNames(String[] names) {
        int n = names.length;
        Map<String, Integer> map = new HashMap<>();
        String[] res = new String[n];
        for (int i = 0; i < n; ++i) {
            if (!map.containsKey(names[i])) {
                res[i] = names[i];
                map.put(names[i], 0);
            } else {
                int index = map.get(names[i]) + 1;
                String cur = names[i] + "(" + index + ")";
                while (map.containsKey(cur)) {
                    ++index;
                    cur = names[i] + "(" + index + ")";
                }
                res[i] = cur;
                map.put(cur, 0);
                map.put(names[i], index);
            }
        }
        return res;

    }

    class Node {
        public int val;
        public List<Node> children;
        public Node prev;
        public Node next;
        public Node child;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, List<Node> _children) {
            val = _val;
            children = _children;
        }

        public Node(int _val, Node _next) {
            val = _val;
            next = _next;
        }
    }

    // 589. N 叉树的前序遍历 (N-ary Tree Preorder Traversal)
    public List<Integer> preorder(Node root) {
        List<Integer> res = new ArrayList<>();
        dfs589(root, res);
        return res;

    }

    private void dfs589(Node root, List<Integer> res) {
        if (root == null) {
            return;
        }
        res.add(root.val);
        for (Node child : root.children) {
            dfs589(child, res);
        }
    }

    // 589. N 叉树的前序遍历 (N-ary Tree Preorder Traversal) --迭代
    public List<Integer> preorder2(Node root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Stack<Node> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            res.add(node.val);
            for (int i = node.children.size() - 1; i >= 0; --i) {
                stack.push(node.children.get(i));
            }
        }
        return res;

    }

    // 590. N 叉树的后序遍历 (N-ary Tree Postorder Traversal)
    public List<Integer> postorder(Node root) {
        List<Integer> res = new ArrayList<>();
        dfs590(root, res);
        return res;

    }

    private void dfs590(Node node, List<Integer> res) {
        if (node == null) {
            return;
        }
        for (int i = 0; i < node.children.size(); ++i) {
            dfs590(node.children.get(i), res);
        }
        res.add(node.val);
    }

    // 590. N 叉树的后序遍历 (N-ary Tree Postorder Traversal) --迭代
    public List<Integer> postorder2(Node root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Stack<Node> stack = new Stack<>();
        stack.push(root);
        Set<Node> visited = new HashSet<>();
        while (!stack.isEmpty()) {
            Node node = stack.peek();
            if (node.children.isEmpty() || visited.contains(node)) {
                stack.pop();
                res.add(node.val);
                continue;
            }
            for (int i = node.children.size() - 1; i >= 0; --i) {
                stack.push(node.children.get(i));
            }
            visited.add(node);
        }
        return res;

    }

    public interface NestedInteger {

        // @return true if this NestedInteger holds a single integer, rather than a
        // nested list.
        public boolean isInteger();

        // @return the single integer that this NestedInteger holds, if it holds a
        // single integer
        // Return null if this NestedInteger holds a nested list
        public Integer getInteger();

        // @return the nested list that this NestedInteger holds, if it holds a nested
        // list
        // Return empty list if this NestedInteger holds a single integer
        public List<NestedInteger> getList();
    }

    // 341. 扁平化嵌套列表迭代器 (Flatten Nested List Iterator)
    public class NestedIterator implements Iterator<Integer> {
        private List<Integer> list;
        private int index;

        public NestedIterator(List<NestedInteger> nestedList) {
            list = new ArrayList<>();
            dfs341(nestedList);

        }

        private void dfs341(List<NestedInteger> nestedList) {
            for (int i = 0; i < nestedList.size(); ++i) {
                NestedInteger nestedInteger = nestedList.get(i);
                if (nestedInteger.isInteger()) {
                    list.add(nestedInteger.getInteger());
                } else {
                    dfs341(nestedInteger.getList());
                }
            }
        }

        @Override
        public Integer next() {
            return list.get(index++);
        }

        @Override
        public boolean hasNext() {
            return index < list.size();
        }
    }

    // 341. 扁平化嵌套列表迭代器 (Flatten Nested List Iterator) --迭代
    public class NestedIterator2 implements Iterator<Integer> {
        private Stack<Iterator<NestedInteger>> stack;

        public NestedIterator2(List<NestedInteger> nestedList) {
            stack = new Stack<>();
            stack.push(nestedList.iterator());
        }

        @Override
        public Integer next() {
            return stack.peek().next().getInteger();
        }

        @Override
        public boolean hasNext() {
            while (!stack.isEmpty()) {
                Iterator<NestedInteger> nIterator = stack.peek();
                if (!nIterator.hasNext()) {
                    stack.pop();
                    continue;
                }
                NestedInteger nest = nIterator.next();
                if (nest.isInteger()) {
                    List<NestedInteger> list = new ArrayList<>();
                    list.add(nest);
                    stack.push(list.iterator());
                    return true;
                }
                stack.push(nest.getList().iterator());
            }
            return false;
        }
    }

    // 814. 二叉树剪枝 (Binary Tree Pruning) --dfs
    // 剑指 Offer II 047. 二叉树剪枝
    public TreeNode pruneTree(TreeNode root) {
        dfs814(root);
        if (root.val == 2) {
            return null;
        }
        return root;

    }

    private void dfs814(TreeNode root) {
        if (root == null) {
            return;
        }
        dfs814(root.left);
        dfs814(root.right);
        if (root.left != null && root.left.val == 2) {
            root.left = null;
        }
        if (root.right != null && root.right.val == 2) {
            root.right = null;
        }
        if (root.left == null && root.right == null) {
            if (root.val == 0) {
                root.val = 2;
            }
        }
    }

    // 814. 二叉树剪枝 (Binary Tree Pruning) --dfs
    // 剑指 Offer II 047. 二叉树剪枝
    public TreeNode pruneTree2(TreeNode root) {
        if (root == null) {
            return null;
        }
        root.left = pruneTree2(root.left);
        root.right = pruneTree2(root.right);
        if (root.left == null && root.right == null && root.val == 0) {
            return null;
        }
        return root;
    }

    // 814. 二叉树剪枝 (Binary Tree Pruning) --bfs
    // 剑指 Offer II 047. 二叉树剪枝
    public TreeNode pruneTree3(TreeNode root) {
        Map<TreeNode, Integer> degrees = new HashMap<>();
        Map<TreeNode, TreeNode> map = new HashMap<>();
        Queue<TreeNode> topologicalQueue = new LinkedList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node.left == null && node.right == null && node.val == 0) {
                topologicalQueue.offer(node);
                continue;
            }
            if (node.left != null) {
                map.put(node.left, node);
                degrees.put(node, degrees.getOrDefault(node, 0) + 1);
                queue.offer(node.left);
            }
            if (node.right != null) {
                map.put(node.right, node);
                degrees.put(node, degrees.getOrDefault(node, 0) + 1);
                queue.offer(node.right);
            }
        }
        while (!topologicalQueue.isEmpty()) {
            TreeNode node = topologicalQueue.poll();
            TreeNode parent = map.get(node);
            if (parent == null) {
                return null;
            }
            if (parent.left == node) {
                parent.left = null;
            } else {
                parent.right = null;
            }
            degrees.put(parent, degrees.get(parent) - 1);
            if (degrees.get(parent) == 0 && parent.val == 0) {
                topologicalQueue.offer(parent);
            }
        }
        return root;
    }

    // 1043. 分隔数组以得到最大和 (Partition Array for Maximum Sum)
    private int[] arr1043;
    private int k1043;
    private int n1043;
    private int[] memo1043;

    public int maxSumAfterPartitioning(int[] arr, int k) {
        this.arr1043 = arr;
        this.k1043 = k;
        this.n1043 = arr.length;
        this.memo1043 = new int[n1043];
        Arrays.fill(memo1043, -1);
        return dfs1043(0);

    }

    private int dfs1043(int i) {
        if (i == n1043) {
            return 0;
        }
        if (memo1043[i] != -1) {
            return memo1043[i];
        }
        int res = 0;
        int max = 0;
        for (int j = i; j < Math.min(i + k1043, n1043); ++j) {
            max = Math.max(max, arr1043[j]);
            res = Math.max(res, max * (j - i + 1) + dfs1043(j + 1));
        }
        return memo1043[i] = res;
    }

    // 712. 两个字符串的最小ASCII删除和 (Minimum ASCII Delete Sum for Two Strings) --dp
    public int minimumDeleteSum(String s1, String s2) {
        int m = s1.length();
        int n = s2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = m - 1; i >= 0; --i) {
            dp[i][n] = dp[i + 1][n] + s1.codePointAt(i);
        }
        for (int j = n - 1; j >= 0; --j) {
            dp[m][j] = dp[m][j + 1] + s2.codePointAt(j);
        }
        for (int i = m - 1; i >= 0; --i) {
            for (int j = n - 1; j >= 0; --j) {
                if (s1.charAt(i) == s2.charAt(j)) {
                    dp[i][j] = dp[i + 1][j + 1];
                } else {
                    dp[i][j] = Math.min(dp[i + 1][j] + s1.codePointAt(i), dp[i][j + 1] + s2.codePointAt(j));
                }
            }
        }
        return dp[0][0];

    }

    // 712. 两个字符串的最小ASCII删除和 (Minimum ASCII Delete Sum for Two Strings)
    private int n1_712;
    private int n2_712;
    private int[][] memo712;
    private char[] arr1_712;
    private char[] arr2_712;
    private int[] suf1_712;
    private int[] suf2_712;

    public int minimumDeleteSum2(String s1, String s2) {
        this.n1_712 = s1.length();
        this.arr1_712 = s1.toCharArray();
        this.n2_712 = s2.length();
        this.arr2_712 = s2.toCharArray();
        this.memo712 = new int[n1_712][n2_712];
        for (int i = 0; i < n1_712; ++i) {
            Arrays.fill(memo712[i], -1);
        }
        this.suf1_712 = new int[n1_712];
        this.suf2_712 = new int[n2_712];
        for (int i = n1_712 - 1; i >= 0; --i) {
            if (i == n1_712 - 1) {
                suf1_712[i] = arr1_712[i];
            } else {
                suf1_712[i] = arr1_712[i] + suf1_712[i + 1];
            }
        }
        for (int i = n2_712 - 1; i >= 0; --i) {
            if (i == n2_712 - 1) {
                suf2_712[i] = arr2_712[i];
            } else {
                suf2_712[i] = arr2_712[i] + suf2_712[i + 1];
            }
        }

        return dfs712(0, 0);

    }

    private int dfs712(int i, int j) {
        if (i == n1_712 && j == n2_712) {
            return 0;
        }
        if (i == n1_712) {
            return suf2_712[j];
        }
        if (j == n2_712) {
            return suf1_712[i];
        }
        if (memo712[i][j] != -1) {
            return memo712[i][j];
        }
        if (arr1_712[i] == arr2_712[j]) {
            return memo712[i][j] = dfs712(i + 1, j + 1);
        }
        return memo712[i][j] = Math.min(dfs712(i + 1, j) + arr1_712[i], dfs712(i, j + 1) + arr2_712[j]);
    }

    // 390. 消除游戏 (Elimination Game)
    public int lastRemaining(int n) {
        boolean fromLeft = true;
        // 记录最左侧的元素
        int head = 1;
        // 公差
        int diff = 1;
        while (n > 1) {
            if (fromLeft || (n & 1) == 1) {
                head += diff;
            }
            diff <<= 1;
            fromLeft = !fromLeft;
            n >>= 1;
        }
        return head;

    }

    // 393. UTF-8 编码验证 (UTF-8 Validation)
    public boolean validUtf8(int[] data) {
        int n = data.length;
        int index = 0;
        while (index < n) {
            int count = getBytes393(data[index]);
            if (count < 0 || index + count - 1 >= n) {
                return false;
            }
            for (int j = 1; j < count; ++j) {
                if (!isValid393(data[index + j])) {
                    return false;
                }
            }
            index += count;
        }

        return true;

    }

    private boolean isValid393(int i) {
        int mask1 = 1 << 7;
        int mask2 = (1 << 7) | (1 << 6);
        return (mask2 & i) == mask1;
    }

    private int getBytes393(int n) {
        int mask = 1 << 7;
        int count = 0;
        if ((n & mask) == 0) {
            return 1;
        }
        while ((n & mask) != 0) {
            ++count;
            if (count > 4) {
                return -1;
            }
            mask >>= 1;
        }
        return count >= 2 ? count : -1;
    }

    // 91. 解码方法 (Decode Ways) --dp
    public int numDecodings(String s) {
        int n = s.length();
        int[] dp = new int[n];
        char[] chars = s.toCharArray();
        dp[0] = chars[0] == '0' ? 0 : 1;
        for (int i = 1; i < chars.length; ++i) {
            char c = chars[i];
            char pre = chars[i - 1];
            if (c != '0') {
                dp[i] += dp[i - 1];
                if (i - 2 >= 0) {
                    if (c >= '7' && c <= '9' && pre == '1') {
                        dp[i] += dp[i - 2];
                    } else if (c >= '1' && c <= '6' && pre >= '1' && pre <= '2') {
                        dp[i] += dp[i - 2];
                    }
                } else {
                    if (pre == '1' || (pre == '2' && c >= '0' && c <= '6')) {
                        ++dp[i];
                    }
                }

            } else {
                if (i - 2 >= 0) {
                    if (pre >= '1' && pre <= '2') {
                        dp[i] += dp[i - 2];
                    }
                } else {
                    if (pre == '1' || (pre == '2' && c >= '0' && c <= '6')) {
                        ++dp[i];
                    }
                }
            }
        }
        return dp[n - 1];

    }

    // 91. 解码方法 (Decode Ways) --dp
    public int numDecodings2(String s) {
        int n = s.length();
        int[] dp = new int[n + 1];
        dp[0] = 1;
        for (int i = 1; i <= n; ++i) {
            if (s.charAt(i - 1) != '0') {
                dp[i] += dp[i - 1];
            }
            if (i - 2 >= 0 && s.charAt(i - 2) != '0' && (s.charAt(i - 2) - '0') * 10 + s.charAt(i - 1) - '0' <= 26) {
                dp[i] += dp[i - 2];
            }
        }
        return dp[n];
    }

    // 91. 解码方法 (Decode Ways) --dp
    public int numDecodings3(String s) {
        int n = s.length();
        int a = 0;
        int b = 1;
        int c = 0;
        for (int i = 1; i <= n; ++i) {
            c = 0;
            if (s.charAt(i - 1) != '0') {
                c += b;
            }
            if (i - 2 >= 0 && s.charAt(i - 2) != '0' && (s.charAt(i - 2) - '0') * 10 + s.charAt(i - 1) - '0' <= 26) {
                c += a;
            }
            a = b;
            b = c;
        }
        return c;

    }

    // 91. 解码方法 (Decode Ways)
    private int[] memo91;
    private char[] arr91;
    private int n91;

    public int numDecodings4(String s) {
        if (s.charAt(0) == '0') {
            return 0;
        }
        this.arr91 = s.toCharArray();
        this.n91 = s.length();
        memo91 = new int[n91];
        Arrays.fill(memo91, -1);
        return dfs91(0);

    }

    private int dfs91(int i) {
        if (i == n91) {
            return 1;
        }
        if (arr91[i] == '0') {
            return 0;
        }
        if (memo91[i] != -1) {
            return memo91[i];
        }
        int res = dfs91(i + 1);
        if (i + 1 < n91 && (arr91[i] == '1' || arr91[i] == '2' && arr91[i + 1] <= '6')) {
            res += dfs91(i + 2);
        }
        return memo91[i] = res;
    }

    // 337. 打家劫舍 III (House Robber III)
    public int rob(TreeNode root) {
        return Arrays.stream(dfs337(root)).max().getAsInt();
    }

    private int[] dfs337(TreeNode root) {
        if (root == null) {
            return new int[] { 0, 0 };
        }
        // 不偷 // 偷
        int[] left = dfs337(root.left);
        int[] right = dfs337(root.right);
        return new int[] { Math.max(left[0], left[1]) + Math.max(right[0], right[1]), left[0] + right[0] + root.val };
    }

    // 1054. 距离相等的条形码 (Distant Barcodes)
    public int[] rearrangeBarcodes(int[] barcodes) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int barcode : barcodes) {
            map.put(barcode, map.getOrDefault(barcode, 0) + 1);
        }
        Queue<int[]> queue = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o2[1] - o1[1];
            }

        });

        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            queue.offer(new int[] { entry.getKey(), entry.getValue() });
        }
        int index = 0;
        int n = barcodes.length;
        int[] res = new int[n];
        while (index < n) {
            int[] max = queue.poll();
            if (queue.isEmpty()) {
                res[index++] = max[0];
            } else {
                int[] secondMax = queue.poll();
                if (index > 0 && max[0] == res[index - 1]) {
                    res[index++] = secondMax[0];
                    res[index++] = max[0];
                } else {
                    res[index++] = max[0];
                    res[index++] = secondMax[0];
                }
                if (--secondMax[1] != 0) {
                    queue.offer(secondMax);
                }
            }
            if (--max[1] != 0) {
                queue.offer(max);
            }
        }
        return res;

    }

    // 2001. 可互换矩形的组数 (Number of Pairs of Interchangeable Rectangles)
    public long interchangeableRectangles(int[][] rectangles) {
        Map<Long, Integer> map = new HashMap<>();
        for (int[] rectangle : rectangles) {
            int radio = getGCD2001(rectangle[0], rectangle[1]);
            rectangle[0] /= radio;
            rectangle[1] /= radio;
            long key = rectangle[0] * 10001 + rectangle[1];
            map.put(key, map.getOrDefault(key, 0) + 1);
        }
        long res = 0l;
        for (Map.Entry<Long, Integer> entry : map.entrySet()) {
            res += (long) entry.getValue() * (entry.getValue() - 1) / 2;
        }
        return res;

    }

    // 计算最大公约数
    private int getGCD2001(int a, int b) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    // 1519. 子树中标签相同的节点数 (Number of Nodes in the Sub-Tree With the Same Label)
    private int[] res1519;
    private boolean[] visited1519;

    public int[] countSubTrees(int n, int[][] edges, String labels) {
        res1519 = new int[n];
        visited1519 = new boolean[n];
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int[] edge : edges) {
            map.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(edge[1]);
            map.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(edge[0]);
        }
        dfs1519(map, labels, visited1519, 0);
        return res1519;
    }

    private int[] dfs1519(Map<Integer, List<Integer>> map, String labels, boolean[] visited, int node) {
        visited[node] = true;
        int[] count = new int[26];
        for (int child : map.getOrDefault(node, new ArrayList<>())) {
            if (!visited[child]) {
                int[] res = dfs1519(map, labels, visited, child);
                for (int i = 0; i < 26; ++i) {
                    count[i] += res[i];
                }
            }
        }
        ++count[labels.charAt(node) - 'a'];
        res1519[node] = count[labels.charAt(node) - 'a'];
        return count;
    }

    // 6120. 数组能形成多少数对
    public int[] numberOfPairs(int[] nums) {
        int[] counts = new int[101];
        for (int num : nums) {
            ++counts[num];
        }
        int a = 0;
        int b = 0;
        for (int i = 0; i < 101; ++i) {
            a += counts[i] / 2;
            b += counts[i] % 2;
        }
        return new int[] { a, b };
    }

    // 6164. 数位和相等数对的最大和
    public int maximumSum(int[] nums) {
        int res = -1;
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int num : nums) {
            int sum = getBitSum(num);

            map.computeIfAbsent(sum, k -> new ArrayList<>()).add(num);
        }
        for (List<Integer> list : map.values()) {
            if (list.size() < 2) {
                continue;
            }
            Collections.sort(list);
            res = Math.max(res, list.get(list.size() - 1) + list.get(list.size() - 2));
        }
        return res;
    }

    private int getBitSum(int num) {
        int res = 0;
        while (num != 0) {
            res += num % 10;
            num /= 10;
        }
        return res;
    }

    // 6121. 裁剪数字后查询第 K 小的数字
    public int[] smallestTrimmedNumbers(String[] nums, int[][] queries) {
        Map<Integer, List<Bean6121>> map = new HashMap<>();
        int[] res = new int[queries.length];
        for (int i = 0; i < queries.length; ++i) {
            int[] cur = queries[i];
            int suf = cur[1];
            if (map.containsKey(suf)) {
                Bean6121 bean = map.get(suf).get(cur[0] - 1);
                res[i] = bean.index;
            } else {
                List<Bean6121> list = new ArrayList<>();
                for (int j = 0; j < nums.length; ++j) {
                    String s = nums[j];

                    String sub = s.substring(s.length() - suf);
                    list.add(new Bean6121(j, sub));
                }
                Collections.sort(list);
                Bean6121 bean = list.get(cur[0] - 1);
                res[i] = bean.index;
                map.put(suf, list);
            }
        }
        return res;

    }

    class Bean6121 implements Comparable<Bean6121> {
        int index;
        String num;

        public Bean6121(int index, String num) {
            this.index = index;
            int i = 0;
            while (i < num.length()) {
                if (num.charAt(i) != '0') {
                    break;
                }
                ++i;
            }
            if (i == num.length()) {
                this.num = "0";
            } else {
                this.num = num.substring(i);
            }

        }

        @Override
        public int compareTo(Bean6121 o) {
            if (o.num.length() == this.num.length()) {
                return this.num.compareTo(o.num);
            } else {
                return this.num.length() - o.num.length();
            }
        }
    }

    // 6122. 使数组可以被整除的最少删除次数
    public int minOperations6122(int[] nums, int[] numsDivide) {
        Set<Integer> set = new HashSet<>();
        for (int num : numsDivide) {
            set.add(num);
        }
        Map<Integer, Integer> map = new TreeMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        int res = 0;
        search: for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            int y = entry.getKey();
            for (int x : set) {
                if (x % y != 0) {
                    res += entry.getValue();
                    continue search;
                }
            }
            return res;
        }
        return -1;
    }

    // 6122. 使数组可以被整除的最少删除次数
    public int minOperations6122_2(int[] nums, int[] numsDivide) {
        int g = 0;
        for (int num : numsDivide) {
            g = gcd6122(num, g);
        }
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; ++i) {
            if (g % nums[i] == 0) {
                return i;
            }
        }
        return -1;

    }

    private int gcd6122(int a, int b) {
        return b == 0 ? a : gcd6122(b, a % b);
    }

    // 6122. 使数组可以被整除的最少删除次数
    public int minOperations6122_3(int[] nums, int[] numsDivide) {
        int g = 0;
        for (int num : numsDivide) {
            g = gcd6122(num, g);
        }
        int min = Integer.MAX_VALUE;
        for (int num : nums) {
            if (g % num == 0) {
                min = Math.min(min, num);
            }
        }
        if (min == Integer.MAX_VALUE) {
            return -1;
        }
        int res = 0;
        for (int num : nums) {
            if (num < min) {
                ++res;
            }
        }
        return res;

    }

    // 面试题 17.01. 不用加号的加法
    // 剑指 Offer 65. 不用加减乘除做加法
    public int add(int a, int b) {
        while (b != 0) {
            int c = (a & b) << 1;
            a ^= b;
            b = c;
        }
        return a;
    }

    // 991. 坏了的计算器 (Broken Calculator)
    public int brokenCalc(int startValue, int target) {
        int res = 0;
        while (startValue < target) {
            ++res;
            if (target % 2 == 0) {
                target >>= 1;
            } else {
                ++target;
            }
        }
        return res + startValue - target;

    }

    // 543. 二叉树的直径 (Diameter of Binary Tree) --统计点的数量
    private int res543;

    public int diameterOfBinaryTree(TreeNode root) {
        dfs543(root);
        return res543 - 1;

    }

    private int dfs543(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = dfs543(root.left);
        int right = dfs543(root.right);
        res543 = Math.max(res543, left + right + 1);
        return Math.max(left, right) + 1;
    }

    // 543. 二叉树的直径 (Diameter of Binary Tree) --统计边的数量
    private int res543_2;

    public int diameterOfBinaryTree2(TreeNode root) {
        dfs543_2(root);
        return res543_2;

    }

    private int dfs543_2(TreeNode root) {
        if (root == null) {
            return -1;
        }
        int left = dfs543_2(root.left) + 1;
        int right = dfs543_2(root.right) + 1;
        res543_2 = Math.max(res543_2, left + right);
        return Math.max(left, right);
    }

    // 731. 我的日程安排表 II (My Calendar II)
    class MyCalendarTwo {
        private List<int[]> booked;
        private List<int[]> overlaped;

        public MyCalendarTwo() {
            booked = new ArrayList<>();
            overlaped = new ArrayList<>();
        }

        public boolean book(int start, int end) {
            for (int[] overlape : overlaped) {
                if (!(end <= overlape[0] || start >= overlape[1])) {
                    return false;
                }
            }
            for (int[] book : booked) {
                if (book[0] < end && start < book[1]) {
                    int min = Math.max(start, book[0]);
                    int max = Math.min(end, book[1]);
                    overlaped.add(new int[] { min, max });
                }
            }
            booked.add(new int[] { start, end });
            return true;
        }
    }

    // 731. 我的日程安排表 II (My Calendar II) --哈希表 还需掌握线段树
    class MyCalendarTwo_2 {
        private Map<Integer, Integer> map;

        public MyCalendarTwo_2() {
            map = new TreeMap<>();

        }

        public boolean book(int start, int end) {
            int sum = 0;
            map.put(start, map.getOrDefault(start, 0) + 1);
            map.put(end, map.getOrDefault(end, 0) - 1);
            for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
                sum += entry.getValue();
                if (sum > 2) {
                    map.put(start, map.getOrDefault(start, 0) - 1);
                    map.put(end, map.getOrDefault(end, 0) + 1);
                    return false;
                }
            }
            return true;
        }
    }

    // 164. 最大间距 (Maximum Gap) --基数排序 时间：O(kn) k:最大数的位数
    public int maximumGap(int[] nums) {
        int n = nums.length;
        if (n < 2) {
            return 0;
        }
        int[] buf = new int[n];
        int max = Arrays.stream(nums).max().getAsInt();
        // 最大数的位数
        int times = getBits164(max);
        int div = 1;
        for (int i = 0; i < times; ++i) {
            int[] count = new int[10];
            for (int num : nums) {
                int digit = (num / div) % 10;
                ++count[digit];
            }
            for (int j = 1; j < 10; ++j) {
                count[j] += count[j - 1];
            }
            for (int j = n - 1; j >= 0; --j) {
                int digit = (nums[j] / div) % 10;
                buf[count[digit] - 1] = nums[j];
                --count[digit];
            }
            nums = Arrays.copyOf(buf, n);
            div *= 10;
        }
        int res = 0;
        for (int i = 1; i < n; ++i) {
            res = Math.max(res, nums[i] - nums[i - 1]);
        }
        return res;

    }

    private int getBits164(int num) {
        int res = 0;
        while (num != 0) {
            ++res;
            num /= 10;
        }
        return res;
    }

    // 912. 排序数组 (Sort an Array) --基数排序
    // 基数排序：低位优先
    private static final int OFFSET = 50000;

    public int[] sortArray(int[] nums) {
        int n = nums.length;
        // 预处理，让所有的数都大于等于 0，这样才可以使用基数排序
        for (int i = 0; i < n; i++) {
            nums[i] += OFFSET;
        }
        // 第 1 步：找出最大的数字
        int max = nums[0];
        for (int num : nums) {
            if (num > max) {
                max = num;
            }
        }

        // 第 2 步：计算出最大的数字有几位，这个数值决定了我们要将整个数组看几遍
        int maxLen = getMaxLen(max);

        // 计数排序需要使用的计数数组和临时数组
        int[] count = new int[10];
        int[] temp = new int[n];

        // 表征关键字的量：除数
        // 1 表示按照个位关键字排序
        // 10 表示按照十位关键字排序
        // 100 表示按照百位关键字排序
        // 1000 表示按照千位关键字排序
        int divisor = 1;
        // 有几位数，外层循环就得执行几次
        for (int i = 0; i < maxLen; i++) {

            // 每一步都使用计数排序，保证排序结果是稳定的
            // 这一步需要额外空间保存结果集，因此把结果保存在 temp 中
            countingSort(nums, temp, divisor, n, count);

            // 交换 nums 和 temp 的引用，下一轮还是按照 nums 做计数排序
            int[] t = nums;
            nums = temp;
            temp = t;

            // divisor 自增，表示采用低位优先的基数排序
            divisor *= 10;
        }

        int[] res = new int[n];
        for (int i = 0; i < n; i++) {
            res[i] = nums[i] - OFFSET;
        }
        return res;
    }

    private void countingSort(int[] nums, int[] res, int divisor, int len, int[] count) {
        // 1、计算计数数组
        for (int i = 0; i < len; i++) {
            // 计算数位上的数是几，先取个位，再十位、百位
            int remainder = (nums[i] / divisor) % 10;
            count[remainder]++;
        }

        // 2、变成前缀和数组
        for (int i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }

        // 3、从后向前赋值
        for (int i = len - 1; i >= 0; i--) {
            int remainder = (nums[i] / divisor) % 10;
            int index = count[remainder] - 1;
            res[index] = nums[i];
            count[remainder]--;
        }

        // 4、count 数组需要设置为 0 ，以免干扰下一次排序使用
        for (int i = 0; i < 10; i++) {
            count[i] = 0;
        }
    }

    /**
     * 获取一个整数的最大位数
     *
     * @param num
     * @return
     */
    private int getMaxLen(int num) {
        int maxLen = 0;
        while (num > 0) {
            num /= 10;
            maxLen++;
        }
        return maxLen;
    }

    // 912. 排序数组 (Sort an Array) --归并排序
    public int[] sortArray2(int[] nums) {
        int n = nums.length;
        if (n <= 1) {
            return nums;
        }
        int mid = n >> 1;
        int[] a = new int[mid];
        int[] b = new int[n - mid];
        for (int i = 0; i < n; ++i) {
            if (i < mid) {
                a[i] = nums[i];
            } else {
                b[i - mid] = nums[i];
            }
        }
        int[] res1 = sortArray(a);
        int[] res2 = sortArray(b);

        int i = 0;
        int j = 0;
        int index = 0;
        int[] res = new int[n];
        while (i < res1.length || j < res2.length) {
            if (i < res1.length && j < res2.length) {
                if (res1[i] < res2[j]) {
                    res[index++] = res1[i++];
                } else {
                    res[index++] = res2[j++];
                }
            } else if (i < res1.length) {
                res[index++] = res1[i++];
            } else {
                res[index++] = res2[j++];
            }
        }
        return res;

    }

    // 2351. 第一个出现两次的字母 (First Letter to Appear Twice)
    public char repeatedCharacter(String s) {
        int mask = 0;
        for (char c : s.toCharArray()) {
            if ((mask | (1 << (c - 'a'))) == mask) {
                return c;
            }
            mask |= 1 << (c - 'a');
        }
        throw new IllegalStateException();

    }

    // 6125. 相等行列对
    public int equalPairs(int[][] grid) {
        int n = grid.length;
        Map<List<Integer>, Integer> map = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            List<Integer> list = new ArrayList<>();
            for (int j = 0; j < n; ++j) {
                list.add(grid[i][j]);
            }
            map.merge(list, 1, Integer::sum);
        }
        int res = 0;
        for (int j = 0; j < n; ++j) {
            List<Integer> list = new ArrayList<>();
            for (int i = 0; i < n; ++i) {
                list.add(grid[i][j]);
            }
            res += map.getOrDefault(list, 0);
        }
        return res;
    }

    // 2353. 设计食物评分系统
    class FoodRatings {
        class Bean implements Comparable<Bean> {
            String food;
            int rating;

            Bean(String food, int rating) {
                this.food = food;
                this.rating = rating;
            }

            @Override
            public int compareTo(Bean o) {
                return o.rating == this.rating ? this.food.compareTo(o.food) : o.rating - this.rating;

            }

            @Override
            public int hashCode() {
                return food.hashCode() * 31 + rating;
            }

            @Override
            public boolean equals(Object obj) {
                Bean b = (Bean) obj;
                return b.food.equals(this.food) && b.rating == this.rating;
            }

        }

        private TreeMap<String, TreeSet<Bean>> treeMap;
        private Map<String, Integer> map;
        private Map<String, String> cuiMap;

        public FoodRatings(String[] foods, String[] cuisines, int[] ratings) {
            treeMap = new TreeMap<>();
            map = new HashMap<>();
            cuiMap = new HashMap<>();
            int n = foods.length;
            for (int i = 0; i < n; ++i) {
                treeMap.computeIfAbsent(cuisines[i], k -> new TreeSet<>()).add(new Bean(foods[i], ratings[i]));
                map.put(foods[i], ratings[i]);
                cuiMap.put(foods[i], cuisines[i]);
            }

        }

        public void changeRating(String food, int newRating) {
            int old = map.get(food);
            map.put(food, newRating);
            String cui = cuiMap.get(food);
            TreeSet<Bean> set = treeMap.get(cui);
            set.remove(new Bean(food, old));
            set.add(new Bean(food, newRating));
        }

        public String highestRated(String cuisine) {
            return treeMap.get(cuisine).first().food;
        }
    }

    // 6128. 最好的扑克手牌
    public String bestHand(int[] ranks, char[] suits) {
        int[] counts = new int[14];
        int max = 0;
        boolean flush = true;
        for (int i = 0; i < 5; ++i) {
            ++counts[ranks[i]];
            max = Math.max(counts[ranks[i]], max);
            if (i > 0) {
                if (suits[i] != suits[i - 1]) {
                    flush = false;
                }
            }
        }
        if (flush) {
            return "Flush";
        }
        if (max >= 3) {
            return "Three of a Kind";
        }
        if (max == 2) {
            return "Pair";
        }
        return "High Card";

    }

    // 6129. 全 0 子数组的数目
    public long zeroFilledSubarray(int[] nums) {
        long res = 0l;
        int count = 0;
        for (int num : nums) {
            if (num != 0) {
                count = 0;
            } else {
                ++count;
                res += count;
            }
        }
        return res;

    }

    // 2349. 设计数字容器系统
    class NumberContainers {
        private TreeMap<Integer, Integer> indexMap;
        private Map<Integer, TreeSet<Integer>> map;

        public NumberContainers() {
            indexMap = new TreeMap<>();
            map = new HashMap<>();
        }

        public void change(int index, int number) {
            int original = indexMap.getOrDefault(index, -1);
            if (original != -1) {
                map.get(original).remove(index);
            }
            indexMap.put(index, number);
            map.computeIfAbsent(number, k -> new TreeSet<>()).add(index);
        }

        public int find(int number) {
            TreeSet<Integer> set = map.getOrDefault(number, new TreeSet<>());
            return set.isEmpty() ? -1 : set.first();
        }
    }

    // 6131. 不可能得到的最短骰子序列
    public int shortestSequence(int[] rolls, int k) {
        Set<Integer> set = new HashSet<>();
        int res = 0;
        for (int roll : rolls) {
            set.add(roll);
            if (set.size() == k) {
                ++res;
                set.clear();
            }
        }
        return res + 1;

    }

    // 343. 整数拆分 (Integer Break) --记忆化搜索
    // 剑指 Offer 14- I. 剪绳子
    private int[] memo343;

    public int integerBreak(int n) {
        if (n <= 3) {
            return n - 1;
        }
        memo343 = new int[n + 1];
        Arrays.fill(memo343, -1);
        return dfs343(n);

    }

    private int dfs343(int n) {
        if (n <= 3) {
            return n;
        }
        if (memo343[n] != -1) {
            return memo343[n];
        }
        int max = 0;
        for (int i = 2; i <= n - 2; ++i) {
            max = Math.max(max, dfs343(n - i) * dfs343(i));
        }
        return memo343[n] = max;

    }

    // 343. 整数拆分 (Integer Break) --dp
    // 剑指 Offer 14- I. 剪绳子
    public int integerBreak2(int n) {
        int[] dp = new int[n + 1];
        for (int i = 2; i <= n; ++i) {
            for (int j = 1; j < i; ++j) {
                dp[i] = Math.max(dp[i], Math.max(j * (i - j), j * dp[i - j]));
            }
        }
        return dp[n];

    }

    // 384. 打乱数组 (Shuffle an Array)
    class Solution384 {
        private int[] original;
        private int[] nums;
        private int n;
        private Random random;

        public Solution384(int[] nums) {
            this.n = nums.length;
            this.nums = nums;
            this.random = new Random();
            this.original = new int[n];
            System.arraycopy(nums, 0, original, 0, n);
        }

        public int[] reset() {
            System.arraycopy(original, 0, nums, 0, n);
            return nums;
        }

        public int[] shuffle() {
            List<Integer> list = new ArrayList<>();
            for (int num : nums) {
                list.add(num);
            }
            int index = 0;
            int i = 0;
            while (!list.isEmpty()) {
                index = random.nextInt(list.size());
                nums[i++] = list.remove(index);
            }
            return nums;

            /*
             * Fisher-Yates 洗牌算法
             * int[] shuffle = new int[n];
             * for (int i = 0; i < n; ++i) {
             * int j = i + random.nextInt(n - i);
             * shuffle[i] = nums[j];
             * int temp = nums[j];
             * nums[j] = nums[i];
             * nums[i] = temp;
             * }
             * return shuffle;
             */
        }
    }

    // 658. 找到 K 个最接近的元素 (Find K Closest Elements) --自定义排序
    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        List<Integer> list = Arrays.stream(arr).boxed().collect(Collectors.toList());
        Collections.sort(list, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                if (Math.abs(o1 - x) == Math.abs(o2 - x)) {
                    return o1 - o2;
                }
                return Math.abs(o1 - x) - Math.abs(o2 - x);
            }

        });
        list = list.subList(0, k);
        Collections.sort(list);
        return list;

    }

    // 658. 找到 K 个最接近的元素 (Find K Closest Elements) --双指针
    public List<Integer> findClosestElements2(int[] arr, int k, int x) {
        int remove = arr.length - k;
        int left = 0;
        int right = arr.length - 1;
        while (remove-- > 0) {
            if (Math.abs(arr[left] - x) <= Math.abs(arr[right] - x)) {
                --right;
            } else {
                ++left;
            }
        }
        return Arrays.stream(arr).boxed().collect(Collectors.toList()).subList(left, right + 1);

    }

    // 1110. 删点成林 (Delete Nodes And Return Forest) --dfs 后序遍历
    public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {
        List<TreeNode> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Set<Integer> set = Arrays.stream(to_delete).boxed().collect(Collectors.toSet());
        res.add(root);
        dfs1110(root, res, set);
        return res;

    }

    private TreeNode dfs1110(TreeNode node, List<TreeNode> res, Set<Integer> set) {
        if (node == null) {
            return null;
        }
        node.left = dfs1110(node.left, res, set);
        node.right = dfs1110(node.right, res, set);
        if (set.contains(node.val)) {
            if (node.left != null) {
                res.add(node.left);
            }
            if (node.right != null) {
                res.add(node.right);
            }
            if (res.contains(node)) {
                res.remove(node);
            }
            return null;
        }
        return node;
    }

    // 1276. 不浪费原料的汉堡制作方案 (Number of Burgers with No Waste of Ingredients) --二元一次方程组
    public List<Integer> numOfBurgers(int tomatoSlices, int cheeseSlices) {
        List<Integer> res = new ArrayList<>();
        if (tomatoSlices % 2 == 1) {
            return res;
        }
        int JumboBurge = tomatoSlices / 2 - cheeseSlices;
        int SmallBurger = cheeseSlices - JumboBurge;
        if (JumboBurge >= 0 && SmallBurger >= 0) {
            res.add(JumboBurge);
            res.add(SmallBurger);
            return res;
        }
        return res;

    }

    // 926. 将字符串翻转到单调递增 (Flip String to Monotone Increasing) -- 空间 O(n)
    // LCR 092. 将字符串翻转到单调递增
    public int minFlipsMonoIncr(String s) {
        int n = s.length();
        int[][] dp = new int[n][2];
        dp[0][0] = s.charAt(0) - '0';
        dp[0][1] = '1' - s.charAt(0);
        for (int i = 1; i < n; ++i) {
            dp[i][0] = dp[i - 1][0] + s.charAt(i) - '0';
            dp[i][1] = Math.min(dp[i - 1][0], dp[i - 1][1]) + '1' - s.charAt(i);
        }
        return Math.min(dp[n - 1][0], dp[n - 1][1]);

    }

    // 926. 将字符串翻转到单调递增 (Flip String to Monotone Increasing) -- 空间 O(1)
    // LCR 092. 将字符串翻转到单调递增
    public int minFlipsMonoIncr2(String s) {
        int n = s.length();
        int put0 = s.charAt(0) - '0';
        int put1 = '1' - s.charAt(0);
        for (int i = 1; i < n; ++i) {
            put1 = Math.min(put0, put1) + '1' - s.charAt(i);
            put0 = put0 + s.charAt(i) - '0';
        }
        return Math.min(put0, put1);

    }

    // 926. 将字符串翻转到单调递增 (Flip String to Monotone Increasing)
    // LCR 092. 将字符串翻转到单调递增
    private int[][] memo926;
    private int[] arr926;
    private int n926;

    public int minFlipsMonoIncr3(String s) {
        this.n926 = s.length();
        this.arr926 = new int[n926];
        for (int i = 0; i < n926; ++i) {
            arr926[i] = s.charAt(i) - '0';
        }
        this.memo926 = new int[n926][2];
        for (int i = 0; i < n926; ++i) {
            Arrays.fill(memo926[i], -1);
        }
        return dfs926(0, 0);
    }

    private int dfs926(int i, int j) {
        if (i == n926) {
            return 0;
        }
        if (memo926[i][j] != -1) {
            return memo926[i][j];
        }
        if (j == 0) {
            return memo926[i][j] = Math.min(dfs926(i + 1, arr926[i]), dfs926(i + 1, arr926[i] ^ 1) + 1);
        }
        return memo926[i][j] = dfs926(i + 1, j) + (arr926[i] ^ 1);

    }

    // 516. 最长回文子序列 (Longest Palindromic Subsequence) --记忆化搜索
    private int[][] memo516;
    private String s516;

    public int longestPalindromeSubseq(String s) {
        int n = s.length();
        memo516 = new int[n][n];
        this.s516 = s;
        return dfs516(0, n - 1);

    }

    private int dfs516(int i, int j) {
        if (i > j) {
            return 0;
        }
        if (i == j) {
            return 1;
        }
        if (memo516[i][j] != 0) {
            return memo516[i][j];
        }
        if (s516.charAt(i) == s516.charAt(j)) {
            return memo516[i][j] = dfs516(i + 1, j - 1) + 2;
        }
        return memo516[i][j] = Math.max(dfs516(i + 1, j), dfs516(i, j - 1));
    }

    // 516. 最长回文子序列 (Longest Palindromic Subsequence) --dp
    public int longestPalindromeSubseq2(String s) {
        int n = s.length();
        int[][] dp = new int[n][n];
        for (int i = n - 1; i >= 0; --i) {
            dp[i][i] = 1;
            char c1 = s.charAt(i);
            for (int j = i + 1; j < n; ++j) {
                char c2 = s.charAt(j);
                if (c1 == c2) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][n - 1];

    }

    // 872. 叶子相似的树 (Leaf-Similar Trees) --dfs
    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        List<Integer> list1 = new ArrayList<>();
        dfs872(root1, list1);

        List<Integer> list2 = new ArrayList<>();
        dfs872(root2, list2);

        return list1.equals(list2);

    }

    private void dfs872(TreeNode node, List<Integer> list) {
        if (node.left == null && node.right == null) {
            list.add(node.val);
        } else {
            if (node.left != null) {
                dfs872(node.left, list);
            }
            if (node.right != null) {
                dfs872(node.right, list);
            }
        }
    }

    // 749. 隔离病毒 (Contain Virus) --bfs
    public int containVirus(int[][] isInfected) {
        int[][] directions = { { 0, -1 }, { 0, 1 }, { 1, 0 }, { -1, 0 } };
        int m = isInfected.length;
        int n = isInfected[0].length;
        int res = 0;
        while (true) {
            List<Set<Integer>> neighbors = new ArrayList<>();
            List<Integer> firewalls = new ArrayList<>();
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (isInfected[i][j] == 1) {
                        Queue<int[]> queue = new LinkedList<>();
                        queue.offer(new int[] { i, j });
                        // 某一个感染区块周围的非感染区域
                        Set<Integer> neighbor = new HashSet<>();
                        // 隔离该感染区块所需要建立的墙的数量
                        int firewall = 0;
                        int idx = neighbors.size() + 1;
                        isInfected[i][j] = -idx;
                        while (!queue.isEmpty()) {
                            int[] cur = queue.poll();
                            int x = cur[0];
                            int y = cur[1];
                            for (int[] direction : directions) {
                                int nx = direction[0] + x;
                                int ny = direction[1] + y;
                                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                                    if (isInfected[nx][ny] == 1) {
                                        queue.offer(new int[] { nx, ny });
                                        isInfected[nx][ny] = -idx;
                                    } else if (isInfected[nx][ny] == 0) {
                                        ++firewall;
                                        neighbor.add(getHash749(nx, ny));
                                    }
                                }
                            }
                        }
                        neighbors.add(neighbor);
                        firewalls.add(firewall);
                    }
                }
            }
            // 没有病毒
            if (neighbors.isEmpty()) {
                break;
            }
            // 找到最大的扩散区域
            int idx = 0;
            for (int i = 1; i < neighbors.size(); ++i) {
                if (neighbors.get(i).size() > neighbors.get(idx).size()) {
                    idx = i;
                }
            }
            res += firewalls.get(idx);
            // 恢复非隔离区域的病毒🦠为1
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (isInfected[i][j] < 0) {
                        if (isInfected[i][j] != -idx - 1) {
                            isInfected[i][j] = 1;
                        } else {
                            isInfected[i][j] = 2;
                        }
                    }
                }
            }
            // 将非隔离区域的病毒🦠扩散
            for (int i = 0; i < neighbors.size(); ++i) {
                if (i != idx) {
                    for (int val : neighbors.get(i)) {
                        int x = val >> 16;
                        int y = val & ((1 << 16) - 1);
                        isInfected[x][y] = 1;
                    }
                }
            }
            // 病毒块只有一个，则不会再扩散
            if (neighbors.size() == 1) {
                break;
            }
        }
        return res;

    }

    private int getHash749(int x, int y) {
        return (x << 16) | y;
    }

    // 664. 奇怪的打印机 (Strange Printer) --dp
    public int strangePrinter(String s) {
        int n = s.length();
        int[][] dp = new int[n][n];
        for (int i = n - 1; i >= 0; --i) {
            dp[i][i] = 1;
            for (int j = i + 1; j < n; ++j) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i][j - 1];
                } else {
                    int min = Integer.MAX_VALUE;
                    for (int k = i; k < j; ++k) {
                        min = Math.min(min, dp[i][k] + dp[k + 1][j]);
                    }
                    dp[i][j] = min;
                }
            }
        }
        return dp[0][n - 1];

    }

    // 583. 两个字符串的删除操作 (Delete Operation for Two Strings) --最长公共子序列
    public int minDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        // dp[i][j] ： word1[0,i] 和 word2[0,j] 的最长公共子序列
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; ++i) {
            char c1 = word1.charAt(i - 1);
            for (int j = 1; j <= n; ++j) {
                char c2 = word2.charAt(j - 1);
                if (c1 == c2) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return m - dp[m][n] + n - dp[m][n];
    }

    // 583. 两个字符串的删除操作 (Delete Operation for Two Strings) --dp
    public int minDistance2(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        // dp[i][j] ： 使 word1[0,i] 和 word2[0,j] 相同，所需删除的最少字符
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; ++i) {
            dp[i][0] = i;
        }
        for (int j = 1; j <= n; ++j) {
            dp[0][j] = j;
        }
        for (int i = 1; i <= m; ++i) {
            char c1 = word1.charAt(i - 1);
            for (int j = 1; j <= n; ++j) {
                char c2 = word2.charAt(j - 1);
                if (c1 == c2) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + 1;
                }
            }
        }
        return dp[m][n];

    }

    // 583. 两个字符串的删除操作 (Delete Operation for Two Strings)
    private int n1_583;
    private int n2_583;
    private String w1_583;
    private String w2_583;
    private int[][] memo583;

    public int minDistance583(String word1, String word2) {
        this.n1_583 = word1.length();
        this.n2_583 = word2.length();
        this.w1_583 = word1;
        this.w2_583 = word2;
        this.memo583 = new int[n1_583][n2_583];
        for (int i = 0; i < n1_583; ++i) {
            Arrays.fill(memo583[i], -1);
        }
        return dfs583(n1_583 - 1, n2_583 - 1);

    }

    private int dfs583(int i, int j) {
        if (i < 0) {
            return j + 1;
        }
        if (j < 0) {
            return i + 1;
        }
        if (memo583[i][j] != -1) {
            return memo583[i][j];
        }
        if (w1_583.charAt(i) == w2_583.charAt(j)) {
            return memo583[i][j] = dfs583(i - 1, j - 1);
        }
        return memo583[i][j] = Math.min(dfs583(i - 1, j), dfs583(i, j - 1)) + 1;
    }

    // 2323. Find Minimum Time to Finish All Jobs II --plus
    public int minimumTime(int[] jobs, int[] workers) {
        int res = 0;
        Arrays.sort(jobs);
        Arrays.sort(workers);
        int n = jobs.length;
        for (int i = 0; i < n; ++i) {
            res = Math.max(res, (jobs[i] - 1) / workers[i] + 1);
        }
        return res;

    }

    // 2330. Valid Palindrome IV --plus
    public boolean makePalindrome(String s) {
        int left = 0;
        int right = s.length() - 1;
        int change = 0;
        while (left < right) {
            if (s.charAt(left) != s.charAt(right)) {
                if (++change > 2) {
                    return false;
                }
            }
            ++left;
            --right;
        }
        return true;

    }

    // 1490. 克隆 N 叉树 (Clone N-ary Tree) --bfs --plus
    public Node cloneTree(Node root) {
        if (root == null) {
            return null;
        }
        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);
        Node copyRoot = new Node(root.val);
        Queue<Node> copyQueue = new LinkedList<>();
        copyQueue.offer(copyRoot);
        while (!queue.isEmpty()) {
            Node node = queue.poll();
            Node copy = copyQueue.poll();
            List<Node> children = node.children;
            if (children != null) {
                for (Node child : children) {
                    Node copyChild = new Node(child.val);
                    copy.children.add(copyChild);
                    queue.offer(child);
                    copyQueue.offer(copyChild);
                }
            }
        }
        return copyRoot;

    }

    // 1490. 克隆 N 叉树 (Clone N-ary Tree) --dfs --plus
    public Node cloneTree2(Node root) {
        if (root == null) {
            return null;
        }
        Node clone = new Node(root.val);
        for (Node node : root.children) {
            clone.children.add(cloneTree(node));
        }
        return clone;

    }

    // 339. 嵌套列表权重和 (Nested List Weight Sum) --plus
    private int res339;

    public int depthSum(List<NestedInteger> nestedList) {
        dfs339(nestedList, 1);
        return res339;
    }

    private void dfs339(List<NestedInteger> nestedList, int level) {
        for (NestedInteger nested : nestedList) {
            if (nested.isInteger()) {
                res339 += level * nested.getInteger();
            } else {
                dfs339(nested.getList(), level + 1);
            }
        }
    }

    // 1756. 设计最近使用（MRU）队列 (Design Most Recently Used Queue) --plus
    class MRUQueue {
        private List<Integer> list;

        public MRUQueue(int n) {
            list = new ArrayList<>();
            for (int i = 1; i <= n; ++i) {
                list.add(i);
            }
        }

        public int fetch(int k) {
            int res = list.get(k - 1);
            list.remove(k - 1);
            list.add(res);
            return res;
        }
    }

    // 1660. 纠正二叉树 (Correct a Binary Tree) --bfs --plus
    public TreeNode correctBinaryTree(TreeNode root) {
        Map<Integer, TreeNode> parent = new HashMap<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            Map<Integer, Integer> map = new HashMap<>();
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                if (map.containsKey(node.val)) {
                    int deleteNodeVal = map.get(node.val);
                    TreeNode p = parent.get(deleteNodeVal);
                    if (p.left != null && p.left.val == deleteNodeVal) {
                        p.left = null;
                    } else {
                        p.right = null;
                    }
                    return root;
                }

                if (node.left != null) {
                    queue.offer(node.left);
                    parent.put(node.left.val, node);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                    parent.put(node.right.val, node);
                    map.put(node.right.val, node.val);
                }
            }
        }
        return root;
    }

    // 2340. Minimum Adjacent Swaps to Make a Valid Array --plus
    public int minimumSwaps(int[] nums) {
        int n = nums.length;
        int maxIndex = -1;
        int max = -1;
        int minIndex = n;
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < n; ++i) {
            if (nums[i] < min) {
                min = nums[i];
                minIndex = i;
            }
        }
        for (int i = n - 1; i >= 0; --i) {
            if (nums[i] > max) {
                max = nums[i];
                maxIndex = i;
            }
        }
        if (maxIndex >= minIndex) {
            return n - 1 - maxIndex + minIndex;
        }
        return n - 1 - maxIndex + minIndex - 1;

    }

    // 1676. 二叉树的最近公共祖先 IV (Lowest Common Ancestor of a Binary Tree IV) --dfs
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode[] nodes) {
        if (root == null) {
            return null;
        }
        for (TreeNode node : nodes) {
            if (root == node) {
                return root;
            }
        }
        TreeNode left = lowestCommonAncestor(root.left, nodes);
        TreeNode right = lowestCommonAncestor(root.right, nodes);
        if (left != null && right != null) {
            return root;
        }
        return left != null ? left : right;

    }

    // 1213. 三个有序数组的交集 (Intersection of Three Sorted Arrays)
    public List<Integer> arraysIntersection(int[] arr1, int[] arr2, int[] arr3) {
        int n1 = arr1.length;
        int n2 = arr2.length;
        int n3 = arr3.length;
        int index1 = 0;
        int index2 = 0;
        int index3 = 0;
        List<Integer> res = new ArrayList<>();
        while (index1 < n1 && index2 < n2 && index3 < n3) {
            if (arr1[index1] == arr2[index2] && arr2[index2] == arr3[index3]) {
                res.add(arr1[index1]);
                ++index1;
                ++index2;
                ++index3;
            } else {
                if (arr1[index1] <= arr2[index2] && arr1[index1] <= arr3[index3]) {
                    ++index1;
                } else if (arr2[index2] <= arr1[index1] && arr2[index2] <= arr3[index3]) {
                    ++index2;
                } else {
                    ++index3;
                }
            }
        }
        return res;

    }

    // 952. 按公因数计算最大组件大小 (Largest Component Size by Common Factor) --并查集
    public int largestComponentSize(int[] nums) {
        int max = Arrays.stream(nums).max().getAsInt();
        Union952 union = new Union952(max + 1);
        for (int num : nums) {
            for (int i = 2; i * i <= num; ++i) {
                if (num % i == 0) {
                    union.union(num, i);
                    union.union(num, num / i);
                }
            }
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            int root = union.getRoot(num);
            map.put(root, map.getOrDefault(root, 0) + 1);
        }
        return Collections.max(map.values());
    }

    public class Union952 {
        private int[] rank;
        private int[] parent;

        public Union952(int n) {
            rank = new int[n];
            Arrays.fill(rank, 1);
            parent = new int[n];
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
            if (rank[root1] > rank[root2]) {
                parent[root2] = root1;
            } else {
                parent[root1] = root2;
                if (rank[root1] == rank[root2]) {
                    ++rank[root2];
                }
            }
        }
    }

    // 2355. Maximum Number of Books You Can Take --单调栈 + dp
    public long maximumBooks(int[] books) {
        Stack<long[]> stack = new Stack<>();
        // index , maxVal
        stack.push(new long[] { -1l, 0l });
        long res = 0l;
        for (int i = 0; i < books.length; ++i) {
            while (stack.size() > 1 && books[(int) stack.peek()[0]] - stack.peek()[0] >= books[i] - i) {
                stack.pop();
            }
            long size = Math.min(books[i], i - (int) stack.peek()[0]);
            long max = (books[i] + books[i] - size + 1) * size / 2 + stack.peek()[1];
            res = Math.max(res, max);
            stack.push(new long[] { i, max });
        }
        return res;
    }

    // 359. 日志速率限制器 (Logger Rate Limiter) --哈希表
    class Logger {
        private Map<String, Integer> map;

        public Logger() {
            map = new HashMap<>();
        }

        public boolean shouldPrintMessage(int timestamp, String message) {
            if (!map.containsKey(message) || timestamp - map.get(message) >= 10) {
                map.put(message, timestamp);
                return true;
            }
            return false;
        }
    }

    // 359. 日志速率限制器 (Logger Rate Limiter) --队列
    class Logger2 {
        private Queue<Pair359> queue;
        private Set<String> set;

        public Logger2() {
            queue = new LinkedList<>();
            set = new HashSet<>();
        }

        public boolean shouldPrintMessage(int timestamp, String message) {
            while (!queue.isEmpty() && timestamp - queue.peek().timestamp >= 10) {
                set.remove(queue.peek().message);
                queue.poll();
            }
            if (!set.contains(message)) {
                queue.offer(new Pair359(message, timestamp));
                set.add(message);
                return true;
            }
            return false;
        }
    }

    public class Pair359 {
        public String message;
        public int timestamp;

        public Pair359(String message, int timestamp) {
            this.message = message;
            this.timestamp = timestamp;
        }
    }

    // 305. 岛屿数量 II (Number of Islands II) -- 并查集
    public List<Integer> numIslands2(int m, int n, int[][] positions) {
        Union305 union = new Union305(m * n);
        boolean[] visited = new boolean[m * n];
        int[][] directions = { { -1, 0 }, { 0, -1 }, { 1, 0 }, { 0, 1 } };
        List<Integer> res = new ArrayList<>();
        for (int[] position : positions) {
            int x = position[0];
            int y = position[1];
            int index = x * n + y;
            if (!visited[index]) {
                union.addCount();
                visited[index] = true;
                for (int[] direction : directions) {
                    int nx = x + direction[0];
                    int ny = y + direction[1];
                    int nIndex = nx * n + ny;
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && visited[nIndex]
                            && !union.isConnected(nIndex, index)) {
                        union.union(nIndex, index);
                    }
                }
            }
            res.add(union.getCount());
        }
        return res;
    }

    public class Union305 {
        private int[] rank;
        private int[] parent;
        private int count;

        public Union305(int n) {
            rank = new int[n];
            Arrays.fill(rank, 1);
            parent = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
            }
            count = 0;
        }

        public void addCount() {
            ++count;
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
            if (rank[root1] > rank[root2]) {
                parent[root2] = root1;
            } else {
                parent[root1] = root2;
                if (rank[root1] == rank[root2]) {
                    ++rank[root2];
                }
            }
            --count;
        }

        public int getCount() {
            return count;
        }
    }

    // 2357. 使数组中所有元素都等于零 (Make Array Zero by Subtracting Equal Amounts)
    public int minimumOperations(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        return set.size() - (set.contains(0) ? 1 : 0);

    }

    // 6133. 分组的最大数量
    public int maximumGroups(int[] grades) {
        int res = 1;
        int i = 1;
        int count = 2;
        while (i + count - 1 < grades.length) {
            ++res;
            i += count;
            ++count;
        }
        return res;

    }

    // 6133. 分组的最大数量
    public int maximumGroups2(int[] grades) {
        // (1+x)* x/2 = n;
        // x^2+x-2*n=0;
        return (-1 + (int) Math.sqrt(1 + 8 * grades.length)) / 2;
    }

    // 6134. 找到离给定两个节点最近的节点
    public int closestMeetingNode(int[] edges, int node1, int node2) {
        int[] dis1 = getDis6134(node1, edges);
        int[] dis2 = getDis6134(node2, edges);
        int n = edges.length;
        int res = -1;
        int dis = n;
        for (int i = 0; i < n; ++i) {
            int max = Math.max(dis1[i], dis2[i]);
            if (max < dis) {
                dis = max;
                res = i;
            }
        }
        return res;

    }

    private int[] getDis6134(int node, int[] edges) {
        int n = edges.length;
        int[] res = new int[n];
        Arrays.fill(res, n);
        int d = 0;
        while (node != -1 && res[node] == n) {
            res[node] = d++;
            node = edges[node];
        }
        return res;
    }

    // 2268. Minimum Number of Keypresses --plus
    public int minimumKeypresses(String s) {
        int[] counts = new int[26];
        for (char c : s.toCharArray()) {
            ++counts[c - 'a'];
        }
        Arrays.sort(counts);
        int res = 0;
        for (int i = counts.length - 1; i >= 0; --i) {
            if (counts[i] == 0) {
                break;
            }
            if (i >= 17) {
                res += counts[i];
            } else if (i >= 8) {
                res += counts[i] * 2;
            } else {
                res += counts[i] * 3;
            }
        }
        return res;

    }

    // 364. 加权嵌套序列和 II (Nested List Weight Sum II) --plus
    private int maxDepth364;
    private int res364;

    public int depthSumInverse(List<NestedInteger> nestedList) {
        if (nestedList.isEmpty()) {
            return 0;
        }
        dfs364(nestedList, 1);
        dfs364_getRes(nestedList, 1);
        return res364;
    }

    private void dfs364_getRes(List<NestedInteger> nestedList, int level) {
        if (nestedList.isEmpty()) {
            return;
        }
        for (NestedInteger nestedInteger : nestedList) {
            if (nestedInteger.isInteger()) {
                res364 += nestedInteger.getInteger() * (maxDepth364 - level + 1);
            } else {
                dfs364_getRes(nestedInteger.getList(), level + 1);
            }
        }
    }

    private void dfs364(List<NestedInteger> nestedList, int level) {
        if (nestedList.isEmpty()) {
            return;
        }
        maxDepth364 = Math.max(maxDepth364, level);
        for (NestedInteger nestedInteger : nestedList) {
            if (!nestedInteger.isInteger()) {
                dfs364(nestedInteger.getList(), level + 1);
            }
        }
    }

    // 2219. Maximum Sum Score of Array --plus
    public long maximumSumScore(int[] nums) {
        int n = nums.length;
        long[] prefix = new long[n + 1];
        for (int i = 1; i <= n; ++i) {
            prefix[i] = prefix[i - 1] + nums[i - 1];
        }
        long res = Integer.MIN_VALUE;
        for (int i = 0; i < n; ++i) {
            res = Math.max(res, prefix[i + 1]);
            res = Math.max(res, prefix[n] - prefix[i]);
        }
        return res;

    }

    // 2219. Maximum Sum Score of Array --plus
    public long maximumSumScore2(int[] nums) {
        long sum = 0l;
        for (int num : nums) {
            sum += num;
        }
        long preSum = 0l;
        long res = Integer.MIN_VALUE;
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            preSum += nums[i];
            res = Math.max(res, preSum);
            res = Math.max(res, sum - preSum + nums[i]);
        }
        return res;

    }

    // 750. 角矩形的数量 (Number Of Corner Rectangles) --plus
    public int countCornerRectangles(int[][] grid) {
        Map<Integer, Integer> map = new HashMap<>();
        int res = 0;
        int n = grid[0].length;
        for (int[] row : grid) {
            for (int c1 = 0; c1 < n; ++c1) {
                if (row[c1] == 1) {
                    for (int c2 = c1 + 1; c2 < n; ++c2) {
                        if (row[c2] == 1) {
                            int index = c1 * 200 + c2;
                            int c = map.getOrDefault(index, 0);
                            res += c;
                            map.put(index, c + 1);
                        }
                    }
                }
            }
        }
        return res;

    }

    // 750. 角矩形的数量 (Number Of Corner Rectangles) --plus
    public int countCornerRectangles2(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int res = 0;
        for (int r1 = 0; r1 < m; ++r1) {
            for (int r2 = r1 + 1; r2 < m; ++r2) {
                int count = 0;
                for (int c = 0; c < n; ++c) {
                    if (grid[r1][c] == 1 && grid[r2][c] == 1) {
                        ++count;
                    }
                }
                res += count * (count - 1) / 2;
            }
        }
        return res;

    }

    // 2237. Count Positions on Street With Required Brightness --差分数组
    public int meetRequirement(int n, int[][] lights, int[] requirement) {
        int[] diff = new int[n + 1];
        for (int[] light : lights) {
            int min = Math.max(0, light[0] - light[1]);
            int max = Math.min(n - 1, light[0] + light[1]);
            ++diff[min];
            --diff[max + 1];
        }
        for (int i = 1; i < n; ++i) {
            diff[i] += diff[i - 1];
        }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            if (diff[i] >= requirement[i]) {
                ++res;
            }
        }
        return res;

    }

    // 2204. Distance to a Cycle in Undirected Graph --拓扑排序
    public int[] distanceToCycle(int n, int[][] edges) {
        int[] degrees = new int[n];
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int[] edge : edges) {
            map.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(edge[1]);
            map.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(edge[0]);
            ++degrees[edge[0]];
            ++degrees[edge[1]];
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < n; ++i) {
            if (degrees[i] == 1) {
                queue.offer(i);
            }
        }
        int[] res = new int[n];
        if (queue.isEmpty()) {
            return res;
        }
        boolean[] visited = new boolean[n];
        Arrays.fill(visited, true);
        while (!queue.isEmpty()) {
            int node = queue.poll();
            visited[node] = false;
            --degrees[node];
            for (int neighbor : map.getOrDefault(node, new ArrayList<>())) {
                --degrees[neighbor];
                if (degrees[neighbor] == 1) {
                    queue.offer(neighbor);
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            if (visited[i]) {
                queue.offer(i);
            }
        }
        int level = 0;
        while (!queue.isEmpty()) {
            ++level;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int node = queue.poll();
                for (int neighbor : map.getOrDefault(node, new ArrayList<>())) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        res[neighbor] = level;
                        queue.offer(neighbor);
                    }
                }
            }
        }
        return res;

    }

    // 2229. Check if an Array Is Consecutive
    public boolean isConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();

        int min = Arrays.stream(nums).min().getAsInt();
        int max = min + nums.length - 1;
        for (int num : nums) {
            if (num < min || num > max) {
                return false;
            }
            if (!set.add(num)) {
                return false;
            }
        }
        return set.size() == nums.length;

    }

    // 2229. Check if an Array Is Consecutive
    public boolean isConsecutive2(int[] nums) {
        Set<Integer> set = Arrays.stream(nums).boxed().collect(Collectors.toSet());
        int min = Arrays.stream(nums).min().getAsInt();
        int max = Arrays.stream(nums).max().getAsInt();
        return max - min == nums.length - 1 && set.size() == nums.length;

    }

    // 1858. 包含所有前缀的最长单词 (Longest Word With All Prefixes) --字典树 --plus
    public String longestWord(String[] words) {
        // Arrays.sort(words, new Comparator<String>() {

        // @Override
        // public int compare(String o1, String o2) {
        // if (o1.length() == o2.length()) {
        // return o2.compareTo(o1);
        // }
        // return o1.length() - o2.length();
        // }

        // });
        Trie1858 trie = new Trie1858();
        for (String word : words) {
            trie.insert(word);
        }
        String res = "";
        int maxLen = 0;
        for (String word : words) {
            if (trie.isLegal(word)) {
                if (word.length() > maxLen) {
                    maxLen = word.length();
                    res = word;
                } else if (word.length() == maxLen && word.compareTo(res) < 0) {
                    res = word;
                }
            }
        }
        return res;
    }

    public class Trie1858 {
        public Trie1858[] children;
        public boolean isEnd;

        public Trie1858() {
            children = new Trie1858[26];
            isEnd = false;
        }

        public void insert(String word) {
            Trie1858 node = this;
            for (char c : word.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new Trie1858();
                }
                node = node.children[index];
            }
            node.isEnd = true;
        }

        public boolean isLegal(String word) {
            Trie1858 node = this;
            for (char c : word.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null || !node.children[index].isEnd) {
                    return false;
                }
                node = node.children[index];
            }
            return node.isEnd;
        }
    }

    // 1229. 安排会议日程 (Meeting Scheduler) --plus
    public List<Integer> minAvailableDuration(int[][] slots1, int[][] slots2, int duration) {
        int n1 = slots1.length;
        int n2 = slots2.length;
        Arrays.sort(slots1, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }

        });

        Arrays.sort(slots2, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }

        });

        int i = 0;
        int j = 0;
        while (i < n1 && j < n2) {
            int[] cur1 = slots1[i];
            int[] cur2 = slots2[j];
            int min = Math.max(cur1[0], cur2[0]);
            int max = Math.min(cur1[1], cur2[1]);
            if (max - min >= duration) {
                return List.of(min, min + duration);
            }
            if (cur1[1] < cur2[1]) {
                ++i;
            } else {
                ++j;
            }
        }
        return List.of();

    }

    // 2127. 参加会议的最多员工数 (Maximum Employees to Be Invited to a Meeting) --基环内向树 (拓扑排序
    // + dp)
    public int maximumInvitations(int[] favorite) {
        int n = favorite.length;
        int[] degrees = new int[n];
        for (int f : favorite) {
            ++degrees[f];
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < n; ++i) {
            if (degrees[i] == 0) {
                queue.offer(i);
            }
        }
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        while (!queue.isEmpty()) {
            int node = queue.poll();
            int neighbor = favorite[node];
            dp[neighbor] = Math.max(dp[node] + 1, dp[neighbor]);
            --degrees[neighbor];
            if (degrees[neighbor] == 0) {
                queue.offer(neighbor);
            }
        }
        int twoNodesRing = 0;
        int threeOrMoreNodesRing = 0;
        for (int i = 0; i < n; ++i) {
            if (degrees[i] != 0) {
                int neighbor = favorite[i];
                if (favorite[neighbor] == i) {
                    degrees[i] = 0;
                    degrees[neighbor] = 0;
                    twoNodesRing += dp[neighbor] + dp[i];
                } else {
                    int count = 0;
                    int node = i;
                    while (degrees[node] != 0) {
                        degrees[node] = 0;
                        ++count;
                        node = favorite[node];
                    }
                    threeOrMoreNodesRing = Math.max(threeOrMoreNodesRing, count);
                }
            }
        }
        return Math.max(threeOrMoreNodesRing, twoNodesRing);

    }

    // 1506. 找到 N 叉树的根节点 (Find Root of N-Ary Tree) --plus
    public Node findRoot(List<Node> tree) {
        int xor = 0;
        for (Node node : tree) {
            xor ^= node.val;
            for (Node sub : node.children) {
                xor ^= sub.val;
            }
        }
        for (Node node : tree) {
            if (node.val == xor) {
                return node;
            }
        }
        return null;
    }

    // 311. 稀疏矩阵的乘法 (Sparse Matrix Multiplication) --plus
    public int[][] multiply(int[][] mat1, int[][] mat2) {
        int m = mat1.length;
        int n = mat2[0].length;
        int k = mat1[0].length;
        int[][] res = new int[m][n];
        for (int i = 0; i < m; ++i) {
            for (int c = 0; c < k; ++c) {
                if (mat1[i][c] == 0) {
                    continue;
                }
                for (int j = 0; j < n; ++j) {
                    res[i][j] += mat1[i][c] * mat2[c][j];
                }
            }
        }
        return res;
    }

    // 899. 有序队列 (Orderly Queue)
    public String orderlyQueue(String s, int k) {
        if (k == 1) {
            String res = s;
            StringBuilder builder = new StringBuilder(s);
            for (int i = 0; i < s.length(); ++i) {
                char first = builder.charAt(0);
                builder.deleteCharAt(0);
                builder.append(first);
                if (builder.toString().compareTo(res) < 0) {
                    res = builder.toString();
                }
            }
            return res;
        } else {
            char[] chars = s.toCharArray();
            Arrays.sort(chars);
            return String.valueOf(chars);
        }

    }

    // 159. 至多包含两个不同字符的最长子串 (Longest Substring with At Most Two Distinct Characters)
    public int lengthOfLongestSubstringTwoDistinct(String s) {
        int res = 0;
        int[] countsLower = new int[26];
        int[] countsUpper = new int[26];
        int left = 0;
        int right = 0;
        int kinds = 0;
        while (right < s.length()) {
            if (Character.isUpperCase(s.charAt(right))) {
                ++countsUpper[s.charAt(right) - 'A'];
                if (countsUpper[s.charAt(right) - 'A'] == 1) {
                    ++kinds;
                }
            } else {
                ++countsLower[s.charAt(right) - 'a'];
                if (countsLower[s.charAt(right) - 'a'] == 1) {
                    ++kinds;
                }
            }

            while (kinds > 2) {
                if (Character.isUpperCase(s.charAt(left))) {
                    --countsUpper[s.charAt(left) - 'A'];
                    if (countsUpper[s.charAt(left) - 'A'] == 0) {
                        --kinds;
                    }
                } else {
                    --countsLower[s.charAt(left) - 'a'];
                    if (countsLower[s.charAt(left) - 'a'] == 0) {
                        --kinds;
                    }
                }
                ++left;
            }
            res = Math.max(res, right - left + 1);
            ++right;
        }
        return res;

    }

    // 163. 缺失的区间 (Missing Ranges)
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        List<String> res = new ArrayList<>();
        int i = 0;
        while (i < nums.length) {
            if (nums[i] - lower == 1) {
                res.add(String.valueOf(lower));
            } else if (nums[i] - lower > 1) {
                res.add(lower + "->" + (nums[i] - 1));
            }
            lower = nums[i] + 1;
            ++i;
        }
        if (upper == lower) {
            res.add(String.valueOf(upper));
        } else if (upper > lower) {
            res.add(lower + "->" + upper);
        }
        return res;

    }

    // 340. 至多包含 K 个不同字符的最长子串 (Longest Substring with At Most K Distinct Characters)
    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        Map<Character, Integer> map = new HashMap<>();
        int left = 0;
        int right = 0;
        int kinds = 0;
        int res = 0;
        while (right < s.length()) {
            map.put(s.charAt(right), map.getOrDefault(s.charAt(right), 0) + 1);
            if (map.get(s.charAt(right)) == 1) {
                ++kinds;
            }
            while (kinds > k) {
                map.put(s.charAt(left), map.get(s.charAt(left)) - 1);
                if (map.get(s.charAt(left)) == 0) {
                    --kinds;
                }
                ++left;
            }
            res = Math.max(res, right - left + 1);
            ++right;
        }
        return res;

    }

    // 1056. 易混淆数 (Confusing Number)
    public boolean confusingNumber(int n) {
        int rotating = 0;
        int original = n;
        while (n != 0) {
            int mod = n % 10;
            if (mod == 2 || mod == 3 || mod == 4 || mod == 5 || mod == 7) {
                return false;
            }
            if (mod == 6) {
                mod = 9;
            } else if (mod == 9) {
                mod = 6;
            }
            rotating = rotating * 10 + mod;
            n /= 10;
        }
        return rotating != original;

    }

    // 1933. 判断字符串是否可分解为值均等的子串 (Check if String Is Decomposable Into Value-Equal
    // Substrings)
    public boolean isDecomposable(String s) {
        int index = 1;
        int count = 1;
        boolean flag = false;
        while (index < s.length()) {
            if (s.charAt(index) == s.charAt(index - 1)) {
                ++count;
            } else {
                if (count % 3 == 1) {
                    return false;
                }
                if (count % 3 == 2) {
                    if (flag) {
                        return false;
                    }
                    flag = true;
                }
                count = 1;
            }
            ++index;
        }
        if (count % 3 == 1) {
            return false;
        }
        if (count % 3 == 0) {
            return flag;
        }
        return !flag;
    }

    // 1933. 判断字符串是否可分解为值均等的子串 (Check if String Is Decomposable Into Value-Equal
    // Substrings)
    public boolean isDecomposable2(String s) {
        int two = 0;
        int left = 0;
        int right = 0;
        while (left < s.length()) {
            right = left + 1;
            while (right < s.length() && s.charAt(left) == s.charAt(right)) {
                ++right;
            }
            int count = right - left;
            if (count % 3 == 1) {
                return false;
            }
            if (count % 3 == 2) {
                if (++two > 1) {
                    return false;
                }
            }
            left = right;
        }
        return two == 1;

    }

    // 1403. 非递增顺序的最小子序列 (Minimum Subsequence in Non-Increasing Order)
    public List<Integer> minSubsequence(int[] nums) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        Arrays.sort(nums);
        int curSum = 0;
        List<Integer> res = new ArrayList<>();
        for (int i = nums.length - 1; i >= 0; --i) {
            curSum += nums[i];
            res.add(nums[i]);
            if (curSum > sum - curSum) {
                return res;
            }
        }
        return res;
    }

    // 2360. 图中的最长环 (Longest Cycle in a Graph)
    public int longestCycle(int[] edges) {
        int n = edges.length;
        int[] degrees = new int[n];
        for (int edge : edges) {
            if (edge != -1) {
                ++degrees[edge];
            }
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < n; ++i) {
            if (degrees[i] == 0) {
                queue.offer(i);
            }
        }
        while (!queue.isEmpty()) {
            int node = queue.poll();
            int neighbor = edges[node];
            if (neighbor != -1) {
                --degrees[neighbor];
                if (degrees[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }
        int res = -1;
        for (int i = 0; i < n; ++i) {
            if (degrees[i] != 0) {
                int cur = 0;
                int node = i;
                while (degrees[node] != 0) {
                    degrees[node] = 0;
                    ++cur;
                    node = edges[node];
                }
                res = Math.max(res, cur);
            }
        }
        return res;

    }

    // 2360. 图中的最长环 (Longest Cycle in a Graph) --时间戳
    public int longestCycle2(int[] edges) {
        int res = -1;
        int n = edges.length;
        int[] time = new int[n];
        int clock = 1;
        for (int i = 0; i < n; ++i) {
            if (time[i] > 0) {
                continue;
            }
            for (int x = i, start_time = clock; x >= 0; x = edges[x]) {
                if (time[x] > 0) {
                    if (time[x] >= start_time) {
                        res = Math.max(res, clock - time[x]);
                    }
                    break;
                }
                time[x] = clock++;
            }
        }
        return res;
    }

    // 1944. 队列中可以看到的人数 (Number of Visible People in a Queue) --单调栈
    public int[] canSeePersonsCount(int[] heights) {
        int n = heights.length;
        Stack<Integer> stack = new Stack<>();
        int[] res = new int[n];
        for (int i = n - 1; i >= 0; --i) {
            while (!stack.isEmpty() && heights[stack.peek()] < heights[i]) {
                stack.pop();
                ++res[i];
            }
            if (!stack.isEmpty()) {
                ++res[i];
            }
            stack.push(i);
        }
        return res;

    }

    // 2282. Number of People That Can Be Seen in a Grid
    public int[][] seePeople(int[][] heights) {
        int m = heights.length;
        int n = heights[0].length;
        int[][] res = new int[m][n];
        for (int i = 0; i < m; ++i) {
            int[] counts = getPeople2282(heights[i]);
            for (int j = 0; j < n; ++j) {
                res[i][j] += counts[j];
            }
        }
        for (int j = 0; j < n; ++j) {
            int[] height = new int[m];
            for (int i = 0; i < m; ++i) {
                height[i] = heights[i][j];
            }
            int[] counts = getPeople2282(height);
            for (int i = 0; i < m; ++i) {
                res[i][j] += counts[i];
            }
        }
        return res;

    }

    private int[] getPeople2282(int[] height) {
        int n = height.length;
        int[] res = new int[n];
        Stack<Integer> stack = new Stack<>();
        for (int i = n - 1; i >= 0; --i) {
            while (!stack.isEmpty() && height[i] > height[stack.peek()]) {
                ++res[i];
                stack.pop();
            }
            if (!stack.isEmpty()) {
                ++res[i];
            }
            while (!stack.isEmpty() && height[i] == height[stack.peek()]) {
                stack.pop();
            }
            stack.push(i);
        }
        return res;
    }

    // 259. 较小的三数之和 (3Sum Smaller)
    public int threeSumSmaller(int[] nums, int target) {
        Arrays.sort(nums);
        int n = nums.length;
        int res = 0;
        for (int i = 0; i < n - 2; ++i) {
            res += getCounts259(nums, i + 1, target - nums[i]);
        }
        return res;

    }

    private int getCounts259(int[] nums, int i, int target) {
        int left = i;
        int right = nums.length - 1;
        int count = 0;
        while (left < right) {
            if (nums[left] + nums[right] < target) {
                count += right - left;
                ++left;
            } else {
                --right;
            }
        }
        return count;
    }

    // 671. 二叉树中第二小的节点 (Second Minimum Node In a Binary Tree)
    private int res671;
    private int min671;

    public int findSecondMinimumValue(TreeNode root) {
        res671 = -1;
        min671 = root.val;
        dfs671(root);
        return res671;

    }

    private void dfs671(TreeNode node) {
        if (node == null) {
            return;
        }
        if (res671 != -1 && node.val >= res671) {
            return;
        }
        if (node.val > min671) {
            res671 = node.val;
        }
        dfs671(node.left);
        dfs671(node.right);
    }

    // 8. 字符串转换整数 (atoi) (String to Integer (atoi))
    // 剑指 Offer 67. 把字符串转换成整数
    public int myAtoi(String str) {
        int res = 0;
        int i = 0;
        while (i < str.length() && str.charAt(i) == ' ') {
            ++i;
        }
        if (i == str.length()) {
            return res;
        }
        int sign = 1;
        if (str.charAt(i) == '-') {
            sign = -1;
            ++i;
        } else if (str.charAt(i) == '+') {
            sign = 1;
            ++i;
        }
        while (i < str.length() && Character.isDigit(str.charAt(i))) {
            if (res > Integer.MAX_VALUE / 10
                    || (res == Integer.MAX_VALUE / 10 && (str.charAt(i) - '0') > (Integer.MAX_VALUE % 10))) {
                if (sign == 1) {
                    return Integer.MAX_VALUE;
                } else {
                    return Integer.MIN_VALUE;
                }
            }
            res = res * 10 + str.charAt(i) - '0';
            ++i;
        }
        return res * sign;
    }

    // 2363. 合并相似的物品 (Merge Similar Items)
    public List<List<Integer>> mergeSimilarItems(int[][] items1, int[][] items2) {
        int[] counts = new int[1001];
        for (int[] item : items1) {
            counts[item[0]] += item[1];
        }
        for (int[] item : items2) {
            counts[item[0]] += item[1];
        }
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < 1001; ++i) {
            if (counts[i] != 0) {
                res.add(List.of(i, counts[i]));
            }
        }
        return res;

    }

    // 2364. 统计坏数对的数目 (Count Number of Bad Pairs)
    public long countBadPairs(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            nums[i] -= i;
        }
        long res = (long) (n) * (n - 1) / 2;
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        for (int val : map.values()) {
            res -= (long) val * (val - 1) / 2;
        }
        return res;

    }

    // 6174. 任务调度器 II (Task Scheduler II)
    public long taskSchedulerII(int[] tasks, int space) {
        long res = 0l;
        Map<Integer, Long> map = new HashMap<>();
        for (int task : tasks) {
            if (map.containsKey(task) && res - map.get(task) < space) {
                res += space - (res - map.get(task));
            }
            map.put(task, ++res);
        }
        return res;

    }

    // 6144. 将数组排序的最少替换次数 (Minimum Replacements to Sort the Array)
    public long minimumReplacement(int[] nums) {
        int n = nums.length;
        int last = nums[n - 1];
        long res = 0;
        for (int i = n - 2; i >= 0; --i) {
            int k = (nums[i] - 1) / last;
            res += k;
            last = nums[i] / (k + 1);
        }
        return res;

    }

    // 6136. 算术三元组的数目 (Number of Arithmetic Triplets) --暴力 O(n^3)
    public int arithmeticTriplets(int[] nums, int diff) {
        int n = nums.length;
        int res = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                for (int k = j + 1; k < n; ++k) {
                    if (nums[j] - nums[i] == diff && nums[k] - nums[j] == diff) {
                        ++res;
                    }
                }
            }
        }
        return res;
    }

    // 6136. 算术三元组的数目 (Number of Arithmetic Triplets) --哈希表/计数 O(n) 空间：O(n)
    public int arithmeticTriplets2(int[] nums, int diff) {
        int max = 0;
        for (int num : nums) {
            max = Math.max(max, num);
        }
        int[] counts = new int[max + 1];
        int res = 0;
        for (int num : nums) {
            ++counts[num];
            if (num - diff >= 0 && num - 2 * diff >= 0) {
                res += counts[num] * counts[num - diff] * counts[num - 2 * diff];
            }
        }
        return res;
    }

    // 6136. 算术三元组的数目 (Number of Arithmetic Triplets) --三指针 O(n) 空间：O(1)
    public int arithmeticTriplets3(int[] nums, int diff) {
        int n = nums.length;
        int res = 0;
        int i = 0;
        int j = 1;
        for (int k = 2; k < n; ++k) {
            while (nums[j] + diff < nums[k]) {
                ++j;
            }
            if (nums[j] + diff > nums[k]) {
                continue;
            }
            while (nums[i] + 2 * diff < nums[k]) {
                ++i;
            }
            if (nums[i] + 2 * diff == nums[k]) {
                ++res;
            }
        }
        return res;

    }

    // 2368. 受限条件下可到达节点的数目 (Reachable Nodes With Restrictions) --dfs
    private Map<Integer, List<Integer>> graph2368;
    private Set<Integer> restrictedSet;

    public int reachableNodes(int n, int[][] edges, int[] restricted) {
        restrictedSet = new HashSet<>();
        for (int r : restricted) {
            restrictedSet.add(r);
        }
        graph2368 = new HashMap<>();
        for (int[] edge : edges) {
            graph2368.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(edge[1]);
            graph2368.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(edge[0]);
        }
        return dfs2368(0, -1);

    }

    private int dfs2368(int x, int fa) {
        if (restrictedSet.contains(x)) {
            return 0;
        }
        int res = 1;
        for (int y : graph2368.getOrDefault(x, new ArrayList<>())) {
            if (y != fa) {
                res += dfs2368(y, x);
            }
        }
        return res;
    }

    // 2368. 受限条件下可到达节点的数目 (Reachable Nodes With Restrictions) --bfs
    public int reachableNodes2(int n, int[][] edges, int[] restricted) {
        Set<Integer> restrictedSet = Arrays.stream(restricted).boxed().collect(Collectors.toSet());
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int[] edge : edges) {
            graph.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(edge[1]);
            graph.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(edge[0]);
        }
        int res = 0;
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(0);
        restrictedSet.add(0);
        while (!queue.isEmpty()) {
            ++res;
            int x = queue.poll();
            for (int y : graph.getOrDefault(x, new ArrayList<>())) {
                if (restrictedSet.add(y)) {
                    queue.offer(y);
                }
            }
        }
        return res;
    }

    // 2368. 受限条件下可到达节点的数目 (Reachable Nodes With Restrictions) --并查集
    public int reachableNodes3(int n, int[][] edges, int[] restricted) {
        Set<Integer> restrictedSet = Arrays.stream(restricted).boxed().collect(Collectors.toSet());
        Union2368 union = new Union2368(n);
        for (int[] edge : edges) {
            int p1 = edge[0];
            int p2 = edge[1];
            if (!restrictedSet.contains(p1) && !restrictedSet.contains(p2)) {
                union.union(p1, p2);
            }
        }
        return union.getCount(0);

    }

    public class Union2368 {
        private int[] rank;
        private int[] parent;
        private int[] size;

        public Union2368(int n) {
            rank = new int[n];
            parent = new int[n];
            size = new int[n];
            for (int i = 0; i < n; ++i) {
                rank[i] = 1;
                size[i] = 1;
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
                size[root2] += size[root1];
            } else {
                parent[root2] = root1;
                size[root1] += size[root2];
                if (rank[root1] == rank[root2]) {
                    ++rank[root1];
                }
            }
        }

        public int getCount(int p) {
            int root = getRoot(p);
            return size[root];

        }

    }

    // 2369. 检查数组是否存在有效划分 (Check if There is a Valid Partition For The Array)
    private int[] nums2369;
    private int n2369;
    private int[] memo2369;

    public boolean validPartition(int[] nums) {
        this.nums2369 = nums;
        this.n2369 = nums.length;
        this.memo2369 = new int[n2369];
        return dfs2369(0);
    }

    private boolean dfs2369(int i) {
        if (i == n2369) {
            return true;
        }
        if (memo2369[i] != 0) {
            return memo2369[i] > 0;
        }
        boolean res1 = false;
        boolean res2 = false;
        if (i + 1 < n2369 && nums2369[i] == nums2369[i + 1]) {
            res1 = dfs2369(i + 2);
        }
        if (i + 2 < n2369 && (nums2369[i] == nums2369[i + 1] && nums2369[i + 1] == nums2369[i + 2]
                || nums2369[i] + 1 == nums2369[i + 1] && nums2369[i + 1] + 1 == nums2369[i + 2])) {
            res2 = dfs2369(i + 3);
        }
        memo2369[i] = (res1 || res2) ? 1 : -1;
        return memo2369[i] > 0;
    }

    // 2369. 检查数组是否存在有效划分 (Check if There is a Valid Partition For The Array)
    public boolean validPartition2(int[] nums) {
        int n = nums.length;
        boolean[] valid = new boolean[n + 1];
        valid[0] = true;
        for (int i = 2; i <= n; ++i) {
            if (valid[i - 2] && nums[i - 1] == nums[i - 2]) {
                valid[i] = true;
            }
            if (i > 2 && valid[i - 3]) {
                if (nums[i - 1] == nums[i - 2] && nums[i - 2] == nums[i - 3]) {
                    valid[i] = true;
                }
                if (nums[i - 1] - nums[i - 2] == 1 && nums[i - 2] - nums[i - 3] == 1) {
                    valid[i] = true;
                }
            }
        }
        return valid[n];

    }

    // 2370. 最长理想子序列 (Longest Ideal Subsequence) (参考第300题)
    public int longestIdealString(String s, int k) {
        int[] dp = new int[26];
        int res = 0;
        for (char c : s.toCharArray()) {
            int x = c - 'a';
            int cur = 0;
            for (int i = 0; i < 26; ++i) {
                if (Math.abs(x - i) <= k) {
                    cur = Math.max(cur, dp[i] + 1);
                }
            }
            dp[x] = Math.max(dp[x], cur);
            res = Math.max(res, dp[x]);
        }
        return res;

    }

    // 2370. 最长理想子序列 (Longest Ideal Subsequence)
    private int n2370;
    private char[] arr2370;
    private int k2370;
    private int[][] memo2370;

    public int longestIdealString2(String s, int k) {
        this.n2370 = s.length();
        this.arr2370 = s.toCharArray();
        this.k2370 = k;
        this.memo2370 = new int[n2370][27];
        for (int i = 0; i < n2370; ++i) {
            Arrays.fill(memo2370[i], -1);
        }
        return dfs2370(0, 26);

    }

    private int dfs2370(int i, int j) {
        if (i == n2370) {
            return 0;
        }
        if (memo2370[i][j] != -1) {
            return memo2370[i][j];
        }
        int res = dfs2370(i + 1, j);
        if (j == 26 || Math.abs(arr2370[i] - 'a' - j) <= k2370) {
            res = Math.max(res, dfs2370(i + 1, arr2370[i] - 'a') + 1);
        }
        return memo2370[i][j] = res;
    }

    // 1422. 分割字符串的最大得分 (Maximum Score After Splitting a String)
    public int maxScore(String s) {
        int n = s.length();
        int zero = 0;
        for (char c : s.toCharArray()) {
            if (c - '0' == 0) {
                ++zero;
            }
        }
        int res = 0;
        int curZeroLeft = 0;
        for (int i = 1; i < n; ++i) {
            if (s.charAt(i - 1) == '0') {
                ++curZeroLeft;
            }
            int curOneRight = (n - i) - (zero - curZeroLeft);
            res = Math.max(res, curZeroLeft + curOneRight);
        }
        return res;

    }

    // 1576. 替换所有的问号 (Replace All ?'s to Avoid Consecutive Repeating Characters)
    public String modifyString(String s) {
        char[] chars = s.toCharArray();
        int n = chars.length;
        for (int i = 0; i < n; ++i) {
            if (chars[i] == '?') {
                chars[i] = 'a';
                while ((i - 1 >= 0 && chars[i] == chars[i - 1]) || (i + 1 < n && chars[i] == chars[i + 1])) {
                    ++chars[i];
                }
            }
        }
        return String.valueOf(chars);

    }

    // 1507. 转变日期格式 (Reformat Date)
    public String reformatDate(String date) {
        Map<String, String> map = new HashMap<>();
        map.put("Jan", "01");
        map.put("Feb", "02");
        map.put("Mar", "03");
        map.put("Apr", "04");
        map.put("May", "05");
        map.put("Jun", "06");
        map.put("Jul", "07");
        map.put("Aug", "08");
        map.put("Sep", "09");
        map.put("Oct", "10");
        map.put("Nov", "11");
        map.put("Dec", "12");
        StringBuilder res = new StringBuilder();
        int n = date.length();
        // year
        res.append(date.substring(n - 4, n)).append("-");

        // month
        for (Map.Entry<String, String> entry : map.entrySet()) {
            if (date.contains(entry.getKey())) {
                res.append(entry.getValue()).append("-");
                break;
            }
        }

        // day
        for (int i = 31; i > 0; --i) {
            if (date.substring(0, 4).contains(String.valueOf(i))) {
                if (i < 10) {
                    res.append(0);
                }
                res.append(i);
                break;
            }
        }
        return res.toString();

    }

    // 1507. 转变日期格式 (Reformat Date)
    public String reformatDate2(String date) {
        String[] months = { "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };
        Map<String, Integer> s2month = new HashMap<String, Integer>();
        for (int i = 1; i <= 12; i++) {
            s2month.put(months[i - 1], i);
        }
        String[] array = date.split(" ");
        int year = Integer.parseInt(array[2]);
        int month = s2month.get(array[1]);
        int day = Integer.parseInt(array[0].substring(0, array[0].length() - 2));
        return String.format("%d-%02d-%02d", year, month, day);
    }

    // 436. 寻找右区间 (Find Right Interval)
    public int[] findRightInterval(int[][] intervals) {
        int n = intervals.length;
        List<int[]> list = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            list.add(new int[] { intervals[i][0], i });
        }
        Collections.sort(list, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }

        });
        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            int right = intervals[i][1];
            int index = binarySearch436(list, right);
            if (index < 0) {
                res[i] = -1;
            } else {
                res[i] = list.get(index)[1];
            }
        }
        return res;

    }

    private int binarySearch436(List<int[]> list, int num) {
        int res = -1;
        int left = 0;
        int right = list.size() - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (list.get(mid)[0] >= num) {
                res = mid;
                right = mid - 1;
            } else if (list.get(mid)[0] < num) {
                left = mid + 1;
            }
        }
        return res;

    }

    // 面试题 17.15. 最长单词 (Longest Word LCCI) -- 回溯 + 字典树
    public String longestWord_17_15(String[] words) {
        Arrays.sort(words, new Comparator<String>() {

            @Override
            public int compare(String o1, String o2) {
                if (o1.length() == o2.length()) {
                    return o1.compareTo(o2);
                }
                return o2.length() - o1.length();
            }

        });

        Trie17_15 trie = new Trie17_15();

        for (String word : words) {
            trie.add(word);
        }

        for (String word : words) {
            if (dfs17_15(word.toCharArray(), 0, 0, trie) > 1) {
                return word;
            }
        }
        return "";

    }

    private int dfs17_15(char[] chars, int start, int size, Trie17_15 trie) {
        if (start == chars.length) {
            return size;
        }
        Trie17_15 node = trie;
        for (int i = start; i < chars.length; ++i) {
            int index = chars[i] - 'a';
            if (node.children[index] == null) {
                return 0;
            }
            node = node.children[index];
            if (node.isEnd) {
                int res = dfs17_15(chars, i + 1, size + 1, trie);
                if (res > 0) {
                    return res;
                }
            }
        }
        return 0;
    }

    public class Trie17_15 {
        public Trie17_15[] children;
        public boolean isEnd;

        public Trie17_15() {
            children = new Trie17_15[26];
            isEnd = false;
        }

        public void add(String s) {
            Trie17_15 node = this;
            for (char c : s.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new Trie17_15();
                }
                node = node.children[index];
            }
            node.isEnd = true;
        }

        public boolean search(String s) {
            Trie17_15 node = this;
            for (char c : s.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    return false;
                }
                node = node.children[index];
            }
            return node.isEnd;
        }
    }

    // 833. 字符串中的查找与替换 (Find And Replace in String)
    public String findReplaceString(String s, int[] indices, String[] sources, String[] targets) {
        int n = s.length();
        int m = indices.length;
        Integer[] ids = IntStream.range(0, m).boxed().toArray(Integer[]::new);
        Arrays.sort(ids, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(indices[o1], indices[o2]);
            }

        });
        StringBuilder res = new StringBuilder();
        int i = 0;
        int j = 0;
        while (i < n && j < m) {
            int index = indices[ids[j]];
            if (index != i) {
                res.append(s.charAt(i));
                ++i;
            } else if (s.substring(index).startsWith(sources[ids[j]])) {
                res.append(targets[ids[j]]);
                i += sources[ids[j]].length();
                ++j;
            } else {
                res.append(s.charAt(i));
                ++i;
                ++j;
            }
        }
        res.append(s.substring(i));
        return res.toString();

    }

    // 708. 循环有序列表的插入 (Insert into a Sorted Circular Linked List) --plus
    // 剑指 Offer II 029. 排序的循环链表
    public Node insert(Node head, int insertVal) {
        if (head == null) {
            head = new Node(insertVal);
            head.next = head;
            return head;
        }
        Node node = head;
        while (node.next != head) {
            if (node.val <= insertVal && insertVal <= node.next.val) {
                break;
            }
            if (node.next.val < node.val && (insertVal < node.next.val || insertVal > node.val)) {
                break;
            }
            node = node.next;
        }
        Node insertNode = new Node(insertVal);
        insertNode.next = node.next;
        node.next = insertNode;
        return head;

    }

    // 233. 数字 1 的个数 (Number of Digit One)
    // 剑指 Offer 43. 1～n 整数中 1 出现的次数
    public int countDigitOne(int n) {
        // mulk 表示 10^k
        // 在下面的代码中，可以发现 k 并没有被直接使用到（都是使用 10^k）
        // 但为了让代码看起来更加直观，这里保留了 k
        long mulk = 1;
        int ans = 0;
        while (n >= mulk) {
            ans += (n / (mulk * 10)) * mulk + Math.min(Math.max(n % (mulk * 10) - mulk + 1, 0), mulk);
            mulk *= 10;
        }
        return ans;
    }

    // 233. 数字 1 的个数 (Number of Digit One) --数位dfs (本题可以不要isNum)
    private char[] arr233;
    private int k233;
    private int[][] memo233;

    public int countDigitOne2(int n) {
        this.arr233 = String.valueOf(n).toCharArray();
        this.k233 = arr233.length;
        this.memo233 = new int[k233][k233];
        for (int i = 0; i < k233; ++i) {
            Arrays.fill(memo233[i], -1);
        }
        return dfs233(0, 0, true, false);

    }

    private int dfs233(int i, int count, boolean isLimit, boolean isNum) {
        if (i == k233) {
            return count;
        }
        if (!isLimit && isNum && memo233[i][count] != -1) {
            return memo233[i][count];
        }
        int res = 0;
        if (!isNum) {
            res = dfs233(i + 1, count, false, false);
        }
        int up = isLimit ? arr233[i] - '0' : 9;
        for (int d = isNum ? 0 : 1; d <= up; ++d) {
            res += dfs233(i + 1, count + (d == 1 ? 1 : 0), isLimit && d == up, true);
        }
        if (!isLimit && isNum) {
            memo233[i][count] = res;
        }
        return res;
    }

    // 1492. n 的第 k 个因子 (The kth Factor of n)
    public int kthFactor(int n, int k) {
        int d = (int) Math.sqrt(n);
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>();
        for (int i = 1; i <= d; ++i) {
            if (n % i == 0) {
                priorityQueue.offer(i);
                if (n / i != i) {
                    priorityQueue.offer(n / i);
                }
            }
        }
        if (priorityQueue.size() < k) {
            return -1;
        }
        while (k-- > 0) {
            int poll = priorityQueue.poll();
            if (k == 0) {
                return poll;
            }
        }
        return -1;

    }

    // 1492. n 的第 k 个因子 (The kth Factor of n) --O(√n)
    public int kthFactor2(int n, int k) {
        int factor = 1;
        while (factor * factor <= n) {
            if (n % factor == 0) {
                --k;
                if (k == 0) {
                    return factor;
                }
            }
            ++factor;
        }
        --factor;
        if (factor * factor == n) {
            --factor;
        }
        while (factor > 0) {
            if (n % factor == 0) {
                --k;
                if (k == 0) {
                    return n / factor;
                }
            }
            --factor;
        }
        return -1;
    }

    // 479. 最大回文数乘积 (Largest Palindrome Product)
    public int largestPalindrome(int n) {
        if (n == 1) {
            return 9;
        }
        int upper = (int) (Math.pow(10, n) - 1);
        for (int left = upper; left > 0; --left) {
            long p = left;
            for (int x = left; x > 0; x /= 10) {
                p = p * 10 + x % 10;
            }
            for (long x = upper; x * x >= p; --x) {
                if (p % x == 0 && String.valueOf(x).length() == n && String.valueOf(p / x).length() == n) {
                    return (int) (p % 1337);
                }
            }
        }
        return -1;

    }

    // 1049. 最后一块石头的重量 II (Last Stone Weight II)
    // -- 0-1背包（至多选择一次；外层循环：nums ； 内层循环：target；倒序遍历)
    // --求最值dp[i]=max/min(dp[i], dp[i-nums] + 1)或dp[i]=max/min(dp[i], dp[i-num] +
    // nums);
    public int lastStoneWeightII(int[] stones) {
        int sum = Arrays.stream(stones).sum();
        int target = sum / 2;
        // dp[i] ：容量为i的背包可装下石头的最大重量
        int[] dp = new int[target + 1];
        for (int stone : stones) {
            for (int i = target; i >= stone; --i) {
                dp[i] = Math.max(dp[i], dp[i - stone] + stone);
            }
        }
        return sum - 2 * dp[target];

    }

    // 1049. 最后一块石头的重量 II (Last Stone Weight II)
    private int[] stones1049;
    private int n1049;
    private int[][] memo1049;

    public int lastStoneWeightII2(int[] stones) {
        this.n1049 = stones.length;
        this.stones1049 = stones;
        this.memo1049 = new int[n1049][3001];
        for (int i = 0; i < n1049; ++i) {
            Arrays.fill(memo1049[i], -1);
        }
        return dfs1049(0, 0);

    }

    private int dfs1049(int i, int sum) {
        if (i == n1049) {
            return Math.abs(sum);
        }
        if (memo1049[i][Math.abs(sum)] != -1) {
            return memo1049[i][Math.abs(sum)];
        }
        return memo1049[i][Math.abs(sum)] = Math.min(dfs1049(i + 1, sum + stones1049[i]),
                dfs1049(i + 1, sum - stones1049[i]));
    }

    // 416. 分割等和子集 (Partition Equal Subset Sum)
    // 剑指 Offer II 101. 分割等和子集
    // -- 0-1背包（至多选择一次；外层循环：nums ； 内层循环：target；倒序遍历)
    // 存在问题(boolean)：dp[i]=dp[i]||dp[i-num];
    public boolean canPartition(int[] nums) {
        int sum = Arrays.stream(nums).sum();
        if (sum % 2 == 1) {
            return false;
        }
        int target = sum / 2;
        boolean[] dp = new boolean[target + 1];
        dp[0] = true;
        for (int num : nums) {
            for (int i = target; i >= num; --i) {
                dp[i] = dp[i] || dp[i - num];
            }
        }
        return dp[target];

    }

    // 416. 分割等和子集 (Partition Equal Subset Sum)
    // 剑指 Offer II 101. 分割等和子集
    private int n416;
    private int[] nums416;
    private int[][] memo416;
    private int s416;

    public boolean canPartition2(int[] nums) {
        this.n416 = nums.length;
        this.nums416 = nums;
        this.s416 = 0;
        for (int x : nums) {
            s416 += x;
        }
        if ((s416 & 1) == 1) {
            return false;
        }
        s416 >>= 1;
        this.memo416 = new int[n416][s416 + 1];
        return dfs416(0, 0);
    }

    private boolean dfs416(int i, int j) {
        if (j == s416) {
            return true;
        }
        if (j > s416 || i == n416) {
            return false;
        }
        if (memo416[i][j] != 0) {
            return memo416[i][j] > 0;
        }
        memo416[i][j] = (dfs416(i + 1, j) || dfs416(i + 1, j + nums416[i])) ? 1 : -1;
        return memo416[i][j] > 0;
    }

    // 494. 目标和 (Target Sum)
    // 剑指 Offer II 102. 加减的目标值
    // -- 0-1背包（至多选择一次；外层循环：nums ； 内层循环：target；倒序遍历)
    // 组合问题 dp[i]+=dp[i-num];
    public int findTargetSumWays(int[] nums, int target) {
        int s = Arrays.stream(nums).sum() - Math.abs(target);
        if (s < 0 || s % 2 != 0) {
            return 0;
        }
        int m = s / 2;
        int[] dp = new int[m + 1];
        dp[0] = 1;
        for (int num : nums) {
            for (int i = m; i >= num; --i) {
                dp[i] += dp[i - num];
            }
        }
        return dp[m];

    }

    // 494. 目标和 (Target Sum)
    // 剑指 Offer II 102. 加减的目标值
    private int[] nums494;
    private int[][] memo494;

    public int findTargetSumWays2(int[] nums, int target) {
        this.nums494 = nums;
        int n = nums.length;
        target = Arrays.stream(nums).sum() - Math.abs(target);
        if (target < 0 || target % 2 != 0) {
            return 0;
        }
        target /= 2;
        this.memo494 = new int[n][target + 1];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(memo494[i], -1);
        }
        return dfs494(n - 1, target);

    }

    private int dfs494(int i, int c) {
        if (i < 0) {
            return c == 0 ? 1 : 0;
        }
        if (memo494[i][c] != -1) {
            return memo494[i][c];
        }
        if (c < nums494[i]) {
            return memo494[i][c] = dfs494(i - 1, c);
        }
        return memo494[i][c] = dfs494(i - 1, c - nums494[i]) + dfs494(i - 1, c);
    }

    // 279. 完全平方数 (Perfect Squares) --记忆化搜索
    private int[][] memo279;
    private int n279;

    public int numSquares(int n) {
        this.n279 = n;
        memo279 = new int[(int) Math.sqrt(n) + 1][n + 1];
        for (int i = 0; i < (int) Math.sqrt(n) + 1; ++i) {
            Arrays.fill(memo279[i], -1);
        }
        return dfs279((int) Math.sqrt(n), 0);

    }

    private int dfs279(int i, int j) {
        if (j >= n279 || i == 0) {
            return j == n279 ? 0 : (int) 1e8;
        }
        if (memo279[i][j] != -1) {
            return memo279[i][j];
        }
        return memo279[i][j] = Math.min(dfs279(i - 1, j), dfs279(i, j + i * i) + 1);
    }

    // 279. 完全平方数 (Perfect Squares)
    // -- 完全背包（可重复选择；外层循环：nums ； 内层循环：target；正序遍历)
    // -- 求最值dp[i]=max/min(dp[i], dp[i-nums] + 1)或dp[i]=max/min(dp[i], dp[i-num] +
    // nums);
    public int numSquares2(int n) {
        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int num = 1; num <= Math.sqrt(n); ++num) {
            for (int i = 1; i <= n; ++i) {
                if (i - num * num >= 0) {
                    dp[i] = Math.min(dp[i], dp[i - num * num] + 1);
                }
            }
        }
        return dp[n];

    }

    // 279. 完全平方数 (Perfect Squares) --数学：四平方和定理
    public int numSquares3(int n) {
        if (isPerfectSquare279(n)) {
            return 1;
        }
        if (isAnswer4(n)) {
            return 4;
        }
        for (int i = 1; i * i < n; ++i) {
            int j = n - i * i;
            if (isPerfectSquare279(j)) {
                return 2;
            }
        }
        return 3;

    }

    private boolean isAnswer4(int n) {
        while (n % 4 == 0) {
            n /= 4;
        }
        return n % 8 == 7;
    }

    private boolean isPerfectSquare279(int n) {
        int m = (int) Math.sqrt(n);
        return m * m == n;
    }

    // 279. 完全平方数 (Perfect Squares) --bfs
    public int numSquares4(int n) {
        Queue<Integer> queue = new LinkedList<>();
        Set<Integer> visited = new HashSet<>();
        queue.offer(0);
        visited.add(0);
        int level = 0;
        while (!queue.isEmpty()) {
            ++level;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int cur = queue.poll();
                for (int j = 1; j <= Math.sqrt(n); ++j) {
                    int neighbor = cur + j * j;
                    if (neighbor == n) {
                        return level;
                    }
                    if (neighbor > n) {
                        break;
                    }
                    if (!visited.contains(neighbor)) {
                        visited.add(neighbor);
                        queue.offer(neighbor);
                    }
                }
            }
        }
        return -1;
    }

    // 377. 组合总和 Ⅳ (Combination Sum IV)
    // 剑指 Offer II 104. 排列的数目
    // 考虑顺序的组合问题（可重复选择；外层循环：target； 内层循环：nums)
    // -- 组合 dp[i] += dp[i - num];
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int i = 1; i <= target; ++i) {
            for (int num : nums) {
                if (i - num >= 0) {
                    dp[i] += dp[i - num];
                }
            }
        }
        return dp[target];

    }

    // 322. 零钱兑换 (Coin Change)
    // 剑指 Offer II 103. 最少的硬币数目
    // -- 完全背包(可重复选择；外层循环：nums ； 内层循环：target；正序遍历)
    // --求最值dp[i]=max/min(dp[i], dp[i-nums] + 1)或dp[i]=max/min(dp[i], dp[i-num] +
    // nums);
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        for (int coin : coins) {
            for (int i = 0; i < amount + 1; ++i) {
                if (i - coin >= 0) {
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        return dp[amount] == amount + 1 ? -1 : dp[amount];

    }

    // 322. 零钱兑换 (Coin Change) --bfs
    public int coinChange2(int[] coins, int amount) {
        if (amount == 0) {
            return 0;
        }
        Queue<int[]> queue = new LinkedList<>();
        Set<Integer> visited = new HashSet<>();
        for (int coin : coins) {
            if (coin == amount) {
                return 1;
            }
            if (coin < amount) {
                queue.offer(new int[] { coin, 1 });
                visited.add(coin);
            }
        }
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int sum = cur[0];
            int count = cur[1];
            if (sum == amount) {
                return count;
            }
            for (int coin : coins) {
                if (sum + coin <= amount && !visited.contains(sum + coin)) {
                    visited.add(sum + coin);
                    queue.offer(new int[] { sum + coin, count + 1 });
                }
            }
        }
        return -1;

    }

    // 322. 零钱兑换 (Coin Change)
    private int[][] memo322;
    private int[] coins322;
    private int amount322;
    private int n322;

    public int coinChange3(int[] coins, int amount) {
        Arrays.sort(coins);
        this.coins322 = coins;
        this.amount322 = amount;
        this.n322 = coins.length;
        this.memo322 = new int[n322][amount + 1];
        for (int i = 0; i < n322; ++i) {
            Arrays.fill(memo322[i], -1);
        }
        int res = dfs322(n322 - 1, 0);
        return res < amount + 1 ? res : -1;

    }

    private int dfs322(int i, int sum) {
        if (sum == amount322 || i < 0) {
            return sum == amount322 ? 0 : amount322 + 1;
        }
        if (memo322[i][sum] != -1) {
            return memo322[i][sum];
        }
        int min = amount322 + 1;
        // 不选
        min = Math.min(min, dfs322(i - 1, sum));
        // 选
        if ((long) sum + coins322[i] <= (long) amount322) {
            min = Math.min(min, dfs322(i, sum + coins322[i]) + 1);
        }
        return memo322[i][sum] = min;
    }

    // 518. 零钱兑换 II (Coin Change 2)
    // -- 完全背包(可重复选择；外层循环：nums ； 内层循环：target；正序遍历)
    // -- 组合 dp[i] += dp[i - num];
    public int change(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int coin : coins) {
            for (int i = 1; i <= amount; ++i) {
                if (i >= coin) {
                    dp[i] += dp[i - coin];
                }
            }
        }
        return dp[amount];
    }

    // 518. 零钱兑换 II (Coin Change 2)
    private int amount518;
    private int[] coins518;
    private int n518;
    private int[][] memo518;

    public int change2(int amount, int[] coins) {
        Arrays.sort(coins);
        this.amount518 = amount;
        this.coins518 = coins;
        this.n518 = coins.length;
        this.memo518 = new int[n518][amount];
        for (int i = 0; i < n518; ++i) {
            Arrays.fill(memo518[i], -1);
        }
        return dfs518(n518 - 1, 0);

    }

    private int dfs518(int i, int sum) {
        if (i < 0 || sum == amount518) {
            return sum == amount518 ? 1 : 0;
        }
        if (memo518[i][sum] != -1) {
            return memo518[i][sum];
        }
        int res = 0;
        // 不选
        res += dfs518(i - 1, sum);
        // 选
        if (sum + coins518[i] <= amount518) {
            res += dfs518(i, sum + coins518[i]);
        }
        return memo518[i][sum] = res;
    }

    // 1155. 掷骰子等于目标和的方法数 (Number of Dice Rolls With Target Sum)
    // 分组背包的组合问题：dp[i][j]表示投掷i个骰子点数和为j的方法数;三层循环：最外层为背包d,然后先遍历target后遍历点数f
    public int numRollsToTarget(int n, int k, int target) {
        final int MOD = (int) (1e9 + 7);
        int[][] dp = new int[n + 1][target + 1];
        dp[0][0] = 1;
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= target; ++j) {
                for (int m = 1; m <= k; ++m) {
                    if (j - m >= 0) {
                        dp[i][j] = (dp[i][j] + dp[i - 1][j - m]) % MOD;
                    }
                }
            }
        }
        return dp[n][target];
    }

    // 1155. 掷骰子等于目标和的方法数 (Number of Dice Rolls With Target Sum)
    private int[][] memo1155;
    private int k1155;

    public int numRollsToTarget2(int n, int k, int target) {
        this.memo1155 = new int[n + 1][target + 1];
        this.k1155 = k;
        return dfs1155(n, target);
    }

    private int dfs1155(int i, int j) {
        if (i * k1155 < j || i > j || j < 0) {
            return 0;
        }
        if (i == 0) {
            return j == 0 ? 1 : 0;
        }
        if (memo1155[i][j] != 0) {
            return memo1155[i][j];
        }
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        for (int x = 1; x <= k1155 && j - x >= 0; ++x) {
            res += dfs1155(i - 1, j - x);
            res %= MOD;
        }
        return memo1155[i][j] = res;
    }

    // 474. 一和零 (Ones and Zeroes)
    // -- 0-1背包
    // -- 求最值dp[i]=max/min(dp[i], dp[i-nums] + 1)或dp[i]=max/min(dp[i], dp[i-num] +
    // nums);
    public int findMaxForm(String[] strs, int m, int n) {
        int len = strs.length;
        // dp[i][j][k] ：在前i个字符串中，使用j个0和k个1可获得的最多字符串个数
        int[][][] dp = new int[len + 1][m + 1][n + 1];
        for (int i = 1; i <= len; ++i) {
            int[] count = getCounts474(strs[i - 1]);
            int zeroes = count[0];
            int ones = count[1];
            for (int j = 0; j <= m; ++j) {
                for (int k = 0; k <= n; ++k) {
                    dp[i][j][k] = dp[i - 1][j][k];
                    if (j >= zeroes && k >= ones) {
                        dp[i][j][k] = Math.max(dp[i - 1][j - zeroes][k - ones] + 1, dp[i][j][k]);
                    }
                }
            }
        }
        return dp[len][m][n];

    }

    private int[] getCounts474(String str) {
        int[] counts = new int[2];
        for (char c : str.toCharArray()) {
            ++counts[c - '0'];
        }
        return counts;
    }

    // 474. 一和零 (Ones and Zeroes)
    private int[][] arr474;
    private int m474;
    private int n474;
    private int[][][] memo474;
    private int len474;

    public int findMaxForm2(String[] strs, int m, int n) {
        this.len474 = strs.length;
        this.arr474 = new int[len474][2];
        this.m474 = m;
        this.n474 = n;
        this.memo474 = new int[len474][m + 1][n + 1];
        for (int i = 0; i < len474; ++i) {
            for (int j = 0; j < m + 1; ++j) {
                Arrays.fill(memo474[i][j], -1);
            }
        }
        for (int i = 0; i < len474; ++i) {
            int cnt1 = 0;
            for (char c : strs[i].toCharArray()) {
                cnt1 += c - '0';
            }
            arr474[i][0] = strs[i].length() - cnt1;
            arr474[i][1] = cnt1;
        }
        return dfs474(0, 0, 0);

    }

    private int dfs474(int i, int j, int k) {
        if (j > m474 || k > n474) {
            return Integer.MIN_VALUE / 2;
        }
        if (i == len474) {
            return 0;
        }
        if (memo474[i][j][k] != -1) {
            return memo474[i][j][k];
        }
        return memo474[i][j][k] = Math.max(dfs474(i + 1, j, k), dfs474(i + 1, j + arr474[i][0], k + arr474[i][1]) + 1);
    }

    // 879. 盈利计划 (Profitable Schemes) --多维0-1背包
    public int profitableSchemes(int n, int minProfit, int[] group, int[] profit) {
        int MOD = (int) 1e9 + 7;
        int size = group.length;
        // dp[i][j][k] : 选择前i项工作，选择j个人，可获得利润至少为k的方案数
        int[][][] dp = new int[size + 1][n + 1][minProfit + 1];
        dp[0][0][0] = 1;
        for (int i = 1; i <= size; ++i) {
            int people = group[i - 1];
            int earn = profit[i - 1];
            for (int j = 0; j <= n; ++j) {
                for (int k = 0; k <= minProfit; ++k) {
                    dp[i][j][k] = dp[i - 1][j][k];
                    if (j >= people) {
                        dp[i][j][k] = (dp[i - 1][j][k] + dp[i - 1][j - people][Math.max(0, k - earn)]) % MOD;
                    }
                }
            }
        }
        int res = 0;
        for (int j = 0; j <= n; ++j) {
            res = (res + dp[size][j][minProfit]) % MOD;
        }
        return res;
    }

    // 879. 盈利计划 (Profitable Schemes)
    private int n879;
    private int minProfit879;
    private int[] group879;
    private int[] profit879;
    private int[][][] memo879;

    public int profitableSchemes2(int n, int minProfit, int[] group, int[] profit) {
        this.n879 = n;
        this.minProfit879 = minProfit;
        this.group879 = group;
        this.profit879 = profit;
        this.memo879 = new int[group.length][n + 1][minProfit + 1];
        for (int i = 0; i < group.length; ++i) {
            for (int j = 0; j < n + 1; ++j) {
                Arrays.fill(memo879[i][j], -1);
            }
        }
        return dfs879(0, 0, 0);

    }

    private int dfs879(int i, int j, int k) {
        if (j > n879) {
            return 0;
        }
        if (i == group879.length) {
            return k == minProfit879 ? 1 : 0;
        }
        if (memo879[i][j][k] != -1) {
            return memo879[i][j][k];
        }
        final int MOD = (int) (1e9 + 7);
        return memo879[i][j][k] = (dfs879(i + 1, j, k) + dfs879(i + 1, j + group879[i], Math.min(minProfit879, k + profit879[i]))) % MOD;
    }

    // 2116. 判断一个括号字符串是否有效 (Check if a Parentheses String Can Be Valid)
    public boolean canBeValid(String s, String locked) {
        int n = s.length();
        if (n % 2 == 1) {
            return false;
        }
        int count = 0;
        for (int i = 0; i < n; ++i) {
            if (s.charAt(i) == '(' || locked.charAt(i) == '0') {
                ++count;
            } else if (count > 0) {
                --count;
            } else {
                return false;
            }
        }
        count = 0;
        for (int i = n - 1; i >= 0; --i) {
            if (s.charAt(i) == ')' || locked.charAt(i) == '0') {
                ++count;
            } else if (count > 0) {
                --count;
            } else {
                return false;
            }
        }
        return true;

    }

    // 6148. 矩阵中的局部最大值
    public int[][] largestLocal(int[][] grid) {
        int n = grid.length;
        int[][] res = new int[n - 2][n - 2];
        for (int i = 1; i < n - 1; ++i) {
            for (int j = 1; j < n - 1; ++j) {
                int max = 0;
                for (int x = i - 1; x <= i + 1; ++x) {
                    for (int y = j - 1; y <= j + 1; ++y) {
                        max = Math.max(max, grid[x][y]);
                    }
                }
                res[i - 1][j - 1] = max;
            }
        }
        return res;

    }

    // 2374. 边积分最高的节点 (Node With Highest Edge Score)
    public int edgeScore(int[] edges) {
        int n = edges.length;
        long[] res = new long[n];

        for (int i = 0; i < n; ++i) {
            res[edges[i]] += i;
        }
        long max = 0;
        int ret = -1;
        for (int i = 0; i < n; ++i) {
            if (res[i] > max) {
                max = res[i];
                ret = i;
            }
        }
        return ret;

    }

    // 6150. 根据模式串构造最小数字 (Construct Smallest Number From DI String)
    public String smallestNumber(String pattern) {
        int n = pattern.length();
        char cur = '1';
        char[] res = new char[n + 1];
        for (int i = 0; i < n; ++i) {
            if (pattern.charAt(i) == 'I') {
                res[i] = cur++;
                int j = i - 1;
                while (j >= 0 && pattern.charAt(j) == 'D') {
                    res[j] = cur++;
                    --j;
                }
            }
        }
        for (int i = n; i >= 0; --i) {
            if (res[i] != 0) {
                break;
            }
            res[i] = cur++;
        }
        return String.valueOf(res);

    }

    // 剑指 Offer 60. n个骰子的点数
    public double[] dicesProbability(int n) {
        double denomiator = Math.pow(6, n);
        // dp[i][j] : 投i个骰子，得到j分的投法数
        int[][] dp = new int[n + 1][6 * n + 1];
        for (int j = 1; j <= 6; ++j) {
            dp[1][j] = 1;
        }
        for (int i = 2; i <= n; ++i) {
            for (int j = i; j <= 6 * i; ++j) {
                for (int k = 1; k <= 6; ++k) {
                    if (j - k >= 0) {
                        dp[i][j] += dp[i - 1][j - k];
                    }
                }
            }
        }
        double[] res = new double[6 * n - n + 1];
        int index = 0;
        int score = n;
        while (index < res.length) {
            res[index++] = dp[n][score++] / denomiator;
        }
        return res;
    }

    // 剑指 Offer 60. n个骰子的点数
    public double[] dicesProbability2(int n) {
        int[] dp = new int[70];
        for (int i = 1; i <= 6; ++i) {
            dp[i] = 1;
        }
        for (int i = 2; i <= n; ++i) {
            for (int j = 6 * i; j >= i; --j) {
                dp[j] = 0;
                for (int cur = 1; cur <= 6; ++cur) {
                    // 前 i - 1 个骰子的和应最少为 i - 1
                    if (j - cur < i - 1) {
                        break;
                    }
                    dp[j] += dp[j - cur];
                }
            }
        }
        double denomiator = Math.pow(6, n);
        double[] res = new double[6 * n - n + 1];
        int index = 0;
        int score = n;
        while (index < res.length) {
            res[index++] = dp[score++] / denomiator;
        }
        return res;

    }

    // 面试题 16.14. 最佳直线 (Best Line LCCI)
    public int[] bestLine(int[][] points) {
        int n = points.length;
        int[] res = new int[2];
        int max = 0;
        for (int i = 0; i < n; ++i) {
            if (max > n / 2 || max > n - i) {
                break;
            }
            for (int j = i + 1; j < n; ++j) {
                int count = 2;
                long x1 = points[i][0] - points[j][0];
                long y1 = points[i][1] - points[j][1];
                for (int k = j + 1; k < n; ++k) {
                    long x2 = points[i][0] - points[k][0];
                    long y2 = points[i][1] - points[k][1];
                    if (x1 * y2 == x2 * y1) {
                        ++count;
                    }
                }
                if (count > max) {
                    max = count;
                    res[0] = i;
                    res[1] = j;
                }
            }
        }
        return res;

    }

    // 149. 直线上最多的点数 (Max Points on a Line) --暴力枚举
    public int maxPoints(int[][] points) {
        int n = points.length;
        if (n <= 2) {
            return n;
        }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            if (res > n / 2 || res > n - i) {
                break;
            }
            int x1 = points[i][0];
            int y1 = points[i][1];
            for (int j = i + 1; j < n; ++j) {
                int x2 = points[j][0];
                int y2 = points[j][1];
                int count = 2;
                for (int k = j + 1; k < n; ++k) {
                    int x3 = points[k][0];
                    int y3 = points[k][1];
                    if ((y2 - y3) * (x1 - x2) == (x2 - x3) * (y1 - y2)) {
                        ++count;
                    }
                }
                res = Math.max(res, count);
            }
        }
        return res;

    }

    // 149. 直线上最多的点数 (Max Points on a Line) --哈希表
    public int maxPoints2(int[][] points) {
        int n = points.length;
        if (n <= 2) {
            return n;
        }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            if (res > n / 2 || res > n - i) {
                break;
            }
            Map<Integer, Integer> map = new HashMap<>();
            for (int j = i + 1; j < n; ++j) {
                int deltaX = points[i][0] - points[j][0];
                int deltaY = points[i][1] - points[j][1];
                if (deltaX == 0) {
                    deltaY = 1;
                } else if (deltaY == 0) {
                    deltaX = 1;
                } else {
                    if (deltaY < 0) {
                        deltaX = -deltaX;
                        deltaY = -deltaY;
                    }
                    int abs = gcd149(Math.abs(deltaX), Math.abs(deltaY));
                    deltaX /= abs;
                    deltaY /= abs;
                }
                int key = deltaX * 20001 + deltaY;
                map.put(key, map.getOrDefault(key, 0) + 1);
            }
            int max = Collections.max(map.values()) + 1;
            res = Math.max(res, max);
        }
        return res;

    }

    private int gcd149(int a, int b) {
        return b == 0 ? a : gcd149(b, a % b);
    }

    // 430. 扁平化多级双向链表 (Flatten a Multilevel Doubly Linked List)
    // 剑指 Offer II 028. 展平多级双向链表
    public Node flatten(Node head) {
        dfs430(head);
        return head;
    }

    private Node dfs430(Node node) {
        Node cur = node;
        Node last = null;

        while (cur != null) {
            Node next = cur.next;
            if (cur.child == null) {
                last = cur;
            } else {
                Node childLast = dfs430(cur.child);

                cur.next = cur.child;
                cur.child.prev = cur;

                if (next != null) {
                    childLast.next = next;
                    next.prev = childLast;
                }

                cur.child = null;
                last = childLast;
            }
            cur = next;
        }
        return last;
    }

    // 745. 前缀和后缀搜索 (Prefix and Suffix Search) --暴力 + 哈希表 (还需掌握字典树)
    class WordFilter {
        private Map<String, Integer> map;

        public WordFilter(String[] words) {
            map = new HashMap<>();
            for (int index = 0; index < words.length; ++index) {
                int n = words[index].length();
                for (int i = 1; i <= n; ++i) {
                    for (int j = n - 1; j >= 0; --j) {
                        map.put(words[index].substring(0, i) + "#" + words[index].substring(j), index);
                    }
                }
            }
        }

        public int f(String pref, String suff) {
            return map.getOrDefault(pref + "#" + suff, -1);
        }
    }

    // 1224. 最大相等频率 (Maximum Equal Frequency) --哈希表
    public int maxEqualFreq(int[] nums) {
        int n = nums.length;
        // key : nums[i] , val : 出现的频率
        Map<Integer, Integer> count = new HashMap<>();
        // key:数字出现的频率 , val:出现key频率的个数
        Map<Integer, Integer> freq = new HashMap<>();
        // 出现最大频率
        int maxFreq = 0;
        int res = 0;
        for (int i = 0; i < n; ++i) {
            if (count.getOrDefault(nums[i], 0) > 0) {
                freq.put(count.get(nums[i]), freq.get(count.get(nums[i])) - 1);
            }
            count.put(nums[i], count.getOrDefault(nums[i], 0) + 1);
            maxFreq = Math.max(maxFreq, count.get(nums[i]));
            freq.put(count.get(nums[i]), freq.getOrDefault(count.get(nums[i]), 0) + 1);
            // maxFreq == 1
            // 数字出现的频率只有 maxFreq 和 maxFreq - 1，而且出现值为 maxFreq 的元素只有1个
            // 数字出现的频率只有 maxFreq 和 非maxFreq，而且出现值为 非maxFreq 的元素只有1个
            if (maxFreq == 1
                    || ((freq.get(maxFreq) * maxFreq + freq.get(maxFreq - 1) * (maxFreq - 1) == i + 1)
                            && (freq.get(maxFreq) == 1))
                    || (freq.get(maxFreq) * maxFreq + 1 == i + 1)) {
                res = Math.max(res, i + 1);
            }
        }
        return res;

    }

    // 761. 特殊的二进制序列 (Special Binary String) --分治
    public String makeLargestSpecial(String s) {
        int n = s.length();
        if (n <= 2) {
            return s;
        }
        List<String> list = new ArrayList<>();
        int left = 0;
        int count = 0;
        for (int i = 0; i < n; ++i) {
            if (s.charAt(i) == '1') {
                ++count;
            } else if (s.charAt(i) == '0') {
                --count;
                if (count == 0) {
                    list.add('1' + makeLargestSpecial(s.substring(left + 1, i)) + '0');
                    left = i + 1;
                }
            }
        }
        Collections.sort(list, new Comparator<String>() {

            @Override
            public int compare(String o1, String o2) {
                return o2.compareTo(o1);
            }

        });

        StringBuilder res = new StringBuilder();
        for (String string : list) {
            res.append(string);
        }
        return res.toString();

    }

    // 2171. 拿出最少数目的魔法豆 (Removing Minimum Number of Magic Beans)
    public long minimumRemoval(int[] beans) {
        int n = beans.length;
        Arrays.sort(beans);
        long sum = 0l;
        long max = 0l;
        for (var i = 0; i < n; ++i) {
            sum += beans[i];
            max = Math.max(max, (long) (n - i) * beans[i]);
        }
        return sum - max;

    }

    // 478. 在圆内随机生成点 (Generate Random Point in a Circle) --拒绝采样
    class Solution478 {
        private Random random;
        private double x_center;
        private double y_center;
        private double radius;

        public Solution478(double radius, double x_center, double y_center) {
            random = new Random();
            this.radius = radius;
            this.x_center = x_center;
            this.y_center = y_center;

        }

        public double[] randPoint() {
            while (true) {
                double x = random.nextDouble() * 2 * radius - radius;
                double y = random.nextDouble() * 2 * radius - radius;
                if ((x * x + y * y <= radius * radius)) {
                    return new double[] { x_center + x, y_center + y };
                }
            }
        }
    }

    // 2110. 股票平滑下跌阶段的数目 (Number of Smooth Descent Periods of a Stock)
    public long getDescentPeriods(int[] prices) {
        int n = prices.length;
        long res = 0l;
        long count = 0l;
        for (int i = 0; i < n; ++i) {
            if (i - 1 >= 0 && prices[i - 1] - prices[i] == 1) {
                ++count;
            } else {
                count = 1l;
            }
            res += count;
        }
        return res;

    }

    // 2134. 最少交换次数来组合所有的 1 II (Minimum Swaps to Group All 1's Together II) --滑动窗口
    public int minSwaps(int[] nums) {
        int n = nums.length;
        int countOne = 0;
        for (int num : nums) {
            countOne += num;
        }
        if (countOne <= 1) {
            return 0;
        }
        int curZero = 0;
        for (int i = 0; i < countOne; ++i) {
            curZero += 1 - nums[i];
        }
        int res = curZero;
        for (int i = 0; i < n - 1; ++i) {
            if (nums[i] == 0) {
                --curZero;
            }
            if (nums[(i + countOne) % n] == 0) {
                ++curZero;
            }
            res = Math.min(res, curZero);
        }
        return res;

    }

    // 273. 整数转换英文表示 (Integer to English Words)
    // 面试题 16.08. English Int LCCI
    private String[] singles = new String[] { "", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
            "Nine" };
    private String[] teens = new String[] { "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
            "Seventeen", "Eighteen", "Nineteen" };
    private String[] tens = new String[] { "", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy",
            "Eighty",
            "Ninety" };
    private String[] thousand = new String[] { "", "Thousand", "Million", "Billion" };

    public String numberToWords(int num) {
        if (num == 0) {
            return "Zero";
        }
        StringBuilder res = new StringBuilder();
        int unit = 1000000000;
        for (int i = 3; i >= 0; --i) {
            int cur = num / unit;
            if (cur != 0) {
                num -= cur * unit;
                res.append(getHundred273(cur)).append(" ").append(thousand[i]).append(" ");
            }
            unit /= 1000;
        }
        return res.toString().trim();

    }

    private String getHundred273(int num) {
        StringBuilder builder = new StringBuilder();
        int hundred = num / 100;
        if (hundred != 0) {
            builder.append(singles[hundred]).append(" Hundred ");
            num %= 100;
        }
        int ten = num / 10;
        if (ten >= 2) {
            builder.append(tens[ten]).append(" ");
            num %= 10;
        }
        if (num >= 10) {
            builder.append(teens[num - 10]);
        } else {
            builder.append(singles[num]);
        }
        return builder.toString().trim();
    }

    // 115. 不同的子序列 (Distinct Subsequences)
    // 剑指 Offer II 097. 子序列的数目
    public int numDistinct(String s, String t) {
        int m = s.length();
        int n = t.length();
        int[][] dp = new int[n + 1][m + 1];
        Arrays.fill(dp[0], 1);
        for (int i = 1; i <= n; ++i) {
            char tChar = t.charAt(i - 1);
            for (int j = 1; j <= m; ++j) {
                char sChar = s.charAt(j - 1);
                if (sChar == tChar) {
                    dp[i][j] = dp[i - 1][j - 1] + dp[i][j - 1];
                } else {
                    dp[i][j] = dp[i][j - 1];
                }
            }
        }
        return dp[n][m];

    }

    // 115. 不同的子序列 (Distinct Subsequences)
    // 剑指 Offer II 097. 子序列的数目
    private int[][] memo115;
    private String s115;
    private String t115;
    private int n115;
    private int m115;

    public int numDistinct2(String s, String t) {
        this.n115 = s.length();
        this.m115 = t.length();
        this.s115 = s;
        this.t115 = t;
        this.memo115 = new int[n115][m115];
        for (int i = 0; i < n115; ++i) {
            Arrays.fill(memo115[i], -1);
        }
        return dfs115(0, 0);

    }

    private int dfs115(int i, int j) {
        if (j == m115) {
            return 1;
        }
        if (i == n115 || n115 - i < m115 - j) {
            return 0;
        }
        if (memo115[i][j] != -1) {
            return memo115[i][j];
        }
        int res = dfs115(i + 1, j);
        if (s115.charAt(i) == t115.charAt(j)) {
            res += dfs115(i + 1, j + 1);
        }
        final int MOD = (int) (1e9 + 7);
        return memo115[i][j] = res % MOD;
    }

    // 2379. 得到 K 个黑块的最少涂色次数 (Minimum Recolors to Get K Consecutive Black Blocks)
    public int minimumRecolors(String blocks, int k) {
        int n = blocks.length();
        int res = Integer.MAX_VALUE;
        int s = 0;
        for (int i = 0; i < n; ++i) {
            s += (blocks.charAt(i) == 'W' ? 1 : 0);
            if (i >= k) {
                s -= (blocks.charAt(i - k) == 'W' ? 1 : 0);
            }
            if (i >= k - 1) {
                res = Math.min(res, s);
            }
        }
        return res;

    }

    // 6157. 二进制字符串重新安排顺序需要的时间
    public int secondsToRemoveOccurrences(String s) {
        char[] chars = s.toCharArray();
        int res = 0;
        while (String.valueOf(chars).indexOf("01") != -1) {
            ++res;
            for (int i = 1; i < chars.length; ++i) {
                if (chars[i] == '1' && chars[i - 1] == '0') {
                    chars[i] = '0';
                    chars[i - 1] = '1';
                    ++i;
                }
            }
        }
        return res;

    }

    // 2381. 字母移位 II (Shifting Letters II)
    public String shiftingLetters(String s, int[][] shifts) {
        int n = s.length();
        int[] diff = new int[n + 1];
        for (int[] shift : shifts) {
            if (shift[2] == 1) {
                ++diff[shift[0]];
                --diff[shift[1] + 1];
            } else {
                --diff[shift[0]];
                ++diff[shift[1] + 1];
            }
        }
        char[] res = new char[n];
        for (int i = 0; i < n; ++i) {
            if (i > 0) {
                diff[i] += diff[i - 1];
            }
            char c = (char) (((s.charAt(i) - 'a' + diff[i]) % 26 + 26) % 26 + 'a');
            res[i] = c;
        }
        return String.valueOf(res);

    }

    // 2090. 半径为 k 的子数组平均值 (K Radius Subarray Averages)
    public int[] getAverages(int[] nums, int k) {
        int n = nums.length;
        int[] res = new int[n];
        Arrays.fill(res, -1);
        long sum = 0L;
        for (int i = 0; i < n; ++i) {
            sum += nums[i];
            if (i > 2 * k) {
                sum -= nums[i - 2 * k - 1];
            }
            if (i >= 2 * k) {
                res[i - k] = (int) (sum / (2 * k + 1));
            }
        }
        return res;

    }

    // 2070. 每一个查询的最大美丽值 (Most Beautiful Item for Each Query) -- 排序 + 二分查找
    public int[] maximumBeauty(int[][] items, int[] queries) {

        Arrays.sort(items, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] == o2[0]) {
                    return o2[1] - o1[1];
                }
                return o1[0] - o2[0];
            }

        });
        for (int i = 1; i < items.length; ++i) {
            items[i][1] = Math.max(items[i - 1][1], items[i][1]);
        }

        int n = queries.length;
        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            int bound = queries[i];
            int beauty = binarySearch2070(items, bound);
            if (beauty >= 0) {
                res[i] = beauty;
            }
        }
        return res;

    }

    private int binarySearch2070(int[][] items, int bound) {
        int left = 0;
        int right = items.length - 1;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (items[mid][0] <= bound) {
                res = items[mid][1];
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    // 2383. 赢得比赛需要的最少训练时长 (Minimum Hours of Training to Win a Competition)
    public int minNumberOfHours(int initialEnergy, int initialExperience, int[] energy, int[] experience) {
        int res = 0;
        int n = energy.length;
        int curEn = initialEnergy;
        int curEx = initialExperience;
        for (int i = 0; i < n; ++i) {
            int en = energy[i];
            int ex = experience[i];
            if (curEn <= en) {
                res += en - curEn + 1;
                curEn = 1;
            } else {
                curEn -= en;
            }
            if (curEx <= ex) {
                res += ex - curEx + 1;
                curEx = ex + 1 + ex;
            } else {
                curEx += ex;

            }
        }
        return res;

    }

    // 6166. 最大回文数字
    public String largestPalindromic(String num) {
        int[] counts = new int[10];
        for (char c : num.toCharArray()) {
            ++counts[c - '0'];
        }
        int oddIndex = -1;
        StringBuilder res = new StringBuilder();
        for (int i = 9; i >= 0; --i) {
            if (counts[i] % 2 == 1) {
                if (oddIndex == -1) {
                    oddIndex = i;
                }
            }
            counts[i] /= 2;
            while (counts[i]-- > 0) {
                res.append(i);
            }

        }
        StringBuilder rev = new StringBuilder(res).reverse();
        if (oddIndex != -1) {
            res.append(oddIndex);
        }
        res.append(rev);
        int i = 0;
        while (i < res.length()) {
            if (res.charAt(i) != '0') {
                break;
            }
            ++i;
        }
        if (i == res.length()) {
            return "0";
        }
        return res.substring(i, res.length() - i);

    }

    // 2385. 感染二叉树需要的总时间 --bfs + bfs
    public int amountOfTime(TreeNode root, int start) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node.left != null) {
                map.computeIfAbsent(node.val, k -> new ArrayList<>()).add(node.left.val);
                map.computeIfAbsent(node.left.val, k -> new ArrayList<>()).add(node.val);
                queue.offer(node.left);

            }
            if (node.right != null) {
                map.computeIfAbsent(node.val, k -> new ArrayList<>()).add(node.right.val);
                map.computeIfAbsent(node.right.val, k -> new ArrayList<>()).add(node.val);
                queue.offer(node.right);
            }

        }
        int res = -1;
        Queue<Integer> queue2 = new LinkedList<>();
        queue2.offer(start);
        Set<Integer> visited = new HashSet<>();
        visited.add(start);
        while (!queue2.isEmpty()) {
            int size = queue2.size();
            for (int i = 0; i < size; ++i) {
                int cur = queue2.poll();
                for (int neighbor : map.getOrDefault(cur, new ArrayList<>())) {
                    if (!visited.contains(neighbor)) {
                        visited.add(neighbor);
                        queue2.offer(neighbor);
                    }
                }
            }
            ++res;
        }
        return res;

    }

    // 2385. 感染二叉树需要的总时间 --dfs + bfs
    private Map<Integer, List<Integer>> graph2385;

    public int amountOfTime2(TreeNode root, int start) {
        graph2385 = new HashMap<>();
        dfs2385(root);
        int res = -1;
        Queue<Integer> queue = new LinkedList<>();
        Set<Integer> visited = new HashSet<>();
        queue.offer(start);
        visited.add(start);
        while (!queue.isEmpty()) {
            ++res;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int x = queue.poll();
                for (int y : graph2385.getOrDefault(x, new ArrayList<>())) {
                    if (visited.add(y)) {
                        queue.offer(y);
                    }
                }
            }
        }
        return res;

    }

    private void dfs2385(TreeNode x) {

        if (x.left != null) {
            graph2385.computeIfAbsent(x.val, k -> new ArrayList<>()).add(x.left.val);
            graph2385.computeIfAbsent(x.left.val, k -> new ArrayList<>()).add(x.val);
            dfs2385(x.left);
        }
        if (x.right != null) {
            graph2385.computeIfAbsent(x.val, k -> new ArrayList<>()).add(x.right.val);
            graph2385.computeIfAbsent(x.right.val, k -> new ArrayList<>()).add(x.val);
            dfs2385(x.right);
        }
    }

    // 6159. 删除操作后的最大子段和 --并查集运用
    private int[] parent6159;

    public long[] maximumSegmentSum(int[] nums, int[] removeQueries) {
        int n = nums.length;
        long[] res = new long[n];
        parent6159 = new int[n + 1];
        for (int i = 0; i <= n; ++i) {
            parent6159[i] = i;
        }
        long[] sum = new long[n + 1];
        for (int i = n - 1; i > 0; --i) {
            int p = removeQueries[i];
            int pa = getRoot6159(p + 1);
            parent6159[p] = pa;
            sum[pa] += sum[p] + nums[p];
            res[i - 1] = Math.max(res[i], sum[pa]);
        }
        return res;

    }

    private int getRoot6159(int p) {
        if (parent6159[p] == p) {
            return p;
        }
        return parent6159[p] = getRoot6159(parent6159[p]);
    }

    // 1594. 矩阵的最大非负积 (Maximum Non Negative Product in a Matrix)
    public int maxProductPath(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int MOD = (int) (1e9 + 7);
        long[][] maxDP = new long[m][n];
        long[][] minDP = new long[m][n];
        maxDP[0][0] = grid[0][0];
        minDP[0][0] = grid[0][0];
        for (int i = 1; i < m; ++i) {
            minDP[i][0] = maxDP[i][0] = maxDP[i - 1][0] * grid[i][0];
        }
        for (int j = 1; j < n; ++j) {
            minDP[0][j] = maxDP[0][j] = maxDP[0][j - 1] * grid[0][j];
        }
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                if (grid[i][j] >= 0) {
                    maxDP[i][j] = Math.max(maxDP[i - 1][j], maxDP[i][j - 1]) * grid[i][j];
                    minDP[i][j] = Math.min(minDP[i - 1][j], minDP[i][j - 1]) * grid[i][j];
                } else {
                    maxDP[i][j] = Math.min(minDP[i - 1][j], minDP[i][j - 1]) * grid[i][j];
                    minDP[i][j] = Math.max(maxDP[i - 1][j], maxDP[i][j - 1]) * grid[i][j];
                }
            }
        }
        return (int) Math.max(-1, maxDP[m - 1][n - 1] % MOD);

    }

    // 1594. 矩阵的最大非负积 (Maximum Non Negative Product in a Matrix)
    private int m1594;
    private int n1594;
    private int[][] grid1594;
    private long[][][] memo1594;

    public int maxProductPath2(int[][] grid) {
        this.m1594 = grid.length;
        this.n1594 = grid[0].length;
        this.grid1594 = grid;
        this.memo1594 = new long[m1594][n1594][2];
        for (int i = 0; i < m1594; ++i) {
            for (int j = 0; j < n1594; ++j) {
                Arrays.fill(memo1594[i][j], Integer.MAX_VALUE);
            }
        }
        final int MOD = (int) (1e9 + 7);
        return (int) (Math.max(-1L, dfs1594(0, 0, 1)) % MOD);

    }

    private long dfs1594(int i, int j, int k) {
        if (i == m1594 - 1 && j == n1594 - 1) {
            return grid1594[i][j];
        }
        if (memo1594[i][j][k] != Integer.MAX_VALUE) {
            return memo1594[i][j][k];
        }
        long res = (k == 1 ? Integer.MIN_VALUE : Integer.MAX_VALUE);
        if (k == 1) {
            if (grid1594[i][j] < 0) {
                if (i + 1 < m1594) {
                    res = Math.max(res, dfs1594(i + 1, j, 0) * grid1594[i][j]);
                }
                if (j + 1 < n1594) {
                    res = Math.max(res, dfs1594(i, j + 1, 0) * grid1594[i][j]);
                }
            } else {
                if (i + 1 < m1594) {
                    res = Math.max(res, dfs1594(i + 1, j, 1) * grid1594[i][j]);
                }
                if (j + 1 < n1594) {
                    res = Math.max(res, dfs1594(i, j + 1, 1) * grid1594[i][j]);
                }
            }
        } else {
            if (grid1594[i][j] < 0) {
                if (i + 1 < m1594) {
                    res = Math.min(res, dfs1594(i + 1, j, 1) * grid1594[i][j]);
                }
                if (j + 1 < n1594) {
                    res = Math.min(res, dfs1594(i, j + 1, 1) * grid1594[i][j]);
                }
            } else {
                if (i + 1 < m1594) {
                    res = Math.min(res, dfs1594(i + 1, j, 0) * grid1594[i][j]);
                }
                if (j + 1 < n1594) {
                    res = Math.min(res, dfs1594(i, j + 1, 0) * grid1594[i][j]);
                }
            }
        }
        return memo1594[i][j][k] = res;
    }

    // 2064. 分配给商店的最多商品的最小值 (Minimized Maximum of Products Distributed to Any Store)
    // --二分查找
    public int minimizedMaximum(int n, int[] quantities) {
        int res = -1;
        int left = 1;
        int right = 0;
        for (int quantity : quantities) {
            right = Math.max(right, quantity);
        }
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (canDistribute2064(mid, quantities, n)) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    private boolean canDistribute2064(int max, int[] quantities, int n) {
        int count = 0;
        for (int quantity : quantities) {
            count += (quantity - 1) / max + 1;
            if (count > n) {
                return false;
            }
        }
        return true;
    }

    // 2033. 获取单值网格的最小操作数 (Minimum Operations to Make a Uni-Value Grid)
    public int minOperations(int[][] grid, int x) {
        int m = grid.length;
        int n = grid[0].length;
        int[] nums = new int[m * n];
        int index = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if ((grid[i][j] - grid[0][0]) % x != 0) {
                    return -1;
                }
                nums[index++] = grid[i][j];
            }
        }
        Arrays.sort(nums);
        int midNum = nums[m * n / 2];
        int res = 0;
        for (int num : nums) {
            res += Math.abs(num - midNum) / x;
        }
        return res;

    }

    // 764. 最大加号标志 (Largest Plus Sign) --dp
    public int orderOfLargestPlusSign(int n, int[][] mines) {
        Set<Integer> set = new HashSet<>();
        for (int[] mine : mines) {
            set.add(mine[0] * n + mine[1]);
        }
        int[][] dp = new int[n][n];
        int count = 0;
        for (int i = 0; i < n; ++i) {
            count = 0;
            for (int j = 0; j < n; ++j) {
                count = set.contains(i * n + j) ? 0 : count + 1;
                dp[i][j] = count;
            }
            count = 0;
            for (int j = n - 1; j >= 0; --j) {
                count = set.contains(i * n + j) ? 0 : count + 1;
                dp[i][j] = Math.min(dp[i][j], count);
            }
        }
        int res = 0;

        for (int j = 0; j < n; ++j) {
            count = 0;
            for (int i = 0; i < n; ++i) {
                count = set.contains(i * n + j) ? 0 : count + 1;
                dp[i][j] = Math.min(dp[i][j], count);
            }
            count = 0;
            for (int i = n - 1; i >= 0; --i) {
                count = set.contains(i * n + j) ? 0 : count + 1;
                dp[i][j] = Math.min(dp[i][j], count);
                res = Math.max(res, dp[i][j]);
            }
        }
        return res;

    }

    /*
     * |arr1[i] - arr1[j]| + |arr2[i] - arr2[j]| + |i - j|
     * 
     * = (arr1[i] + arr2[i] + i) - (arr1[j] + arr2[j] + j)
     * = (arr1[i] + arr2[i] - i) - (arr1[j] + arr2[j] - j)
     * = (arr1[i] - arr2[i] + i) - (arr1[j] - arr2[j] + j)
     * = (arr1[i] - arr2[i] - i) - (arr1[j] - arr2[j] - j)
     * = -(arr1[i] + arr2[i] + i) + (arr1[j] + arr2[j] + j)
     * = -(arr1[i] + arr2[i] - i) + (arr1[j] + arr2[j] - j)
     * = -(arr1[i] - arr2[i] + i) + (arr1[j] - arr2[j] + j)
     * = -(arr1[i] - arr2[i] - i) + (arr1[j] - arr2[j] - j)
     * 
     * 因为存在四组两两等价的展开，所以可以优化为四个表达式：
     * A = arr1[i] + arr2[i] + i
     * B = arr1[i] + arr2[i] - i
     * C = arr1[i] - arr2[i] + i
     * D = arr1[i] - arr2[i] - i
     * 
     * max( |arr1[i] - arr1[j]| + |arr2[i] - arr2[j]| + |i - j|)
     * = max(max(A) - min(A),
     * max(B) - min(B),
     * max(C) - min(C),
     * max(D) - min(D))
     * 
     */
    // 1131. 绝对值表达式的最大值 (Maximum of Absolute Value Expression)
    public int maxAbsValExpr(int[] arr1, int[] arr2) {
        int n = arr1.length;

        int maxA = Integer.MIN_VALUE;
        int maxB = Integer.MIN_VALUE;
        int maxC = Integer.MIN_VALUE;
        int maxD = Integer.MIN_VALUE;

        int minA = Integer.MAX_VALUE;
        int minB = Integer.MAX_VALUE;
        int minC = Integer.MAX_VALUE;
        int minD = Integer.MAX_VALUE;

        for (int i = 0; i < n; ++i) {
            maxA = Math.max(arr1[i] + arr2[i] + i, maxA);
            minA = Math.min(arr1[i] + arr2[i] + i, minA);

            maxB = Math.max(arr1[i] + arr2[i] - i, maxB);
            minB = Math.min(arr1[i] + arr2[i] - i, minB);

            maxC = Math.max(arr1[i] - arr2[i] + i, maxC);
            minC = Math.min(arr1[i] - arr2[i] + i, minC);

            maxD = Math.max(arr1[i] - arr2[i] - i, maxD);
            minD = Math.min(arr1[i] - arr2[i] - i, minD);

        }
        return Math.max(Math.max(maxA - minA, maxB - minB), Math.max(maxC - minC, maxD - minD));
    }

    // 916. 单词子集 (Word Subsets)
    public List<String> wordSubsets(String[] words1, String[] words2) {
        int[] maxCounts = new int[26];
        for (String word : words2) {
            int[] counts = new int[26];
            for (char c : word.toCharArray()) {
                int index = c - 'a';
                ++counts[index];
                maxCounts[index] = Math.max(maxCounts[index], counts[index]);
            }
        }

        List<String> res = new ArrayList<>();
        search: for (String word : words1) {
            int[] counts = new int[26];
            for (char c : word.toCharArray()) {
                ++counts[c - 'a'];
            }
            for (int i = 0; i < 26; ++i) {
                if (counts[i] < maxCounts[i]) {
                    continue search;
                }
            }
            res.add(word);
        }
        return res;

    }

    // 1760. 袋子里最少数目的球 (Minimum Limit of Balls in a Bag) --二分查找
    public int minimumSize(int[] nums, int maxOperations) {
        int left = 1;
        int right = (int) 1e9;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (judge1760(nums, mid) <= maxOperations) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;

    }

    private int judge1760(int[] nums, int max) {
        int operations = 0;
        for (int num : nums) {
            operations += (num - 1) / max;
        }
        return operations;
    }

}
