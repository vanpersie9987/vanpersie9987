import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
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
    private int res124 = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        dfs124(root);
        return res124;

    }

    private int dfs124(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = Math.max(dfs124(root.left), 0);
        int right = Math.max(dfs124(root.right), 0);
        int max = root.val + left + right;
        res124 = Math.max(max, res124);
        return root.val + Math.max(left, right);
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
        char[] chars = croakOfFrogs.toCharArray();
        int c = 0;
        int r = 0;
        int o = 0;
        int a = 0;
        int k = 0;
        int res = 0;
        for (int i = 0; i < chars.length; ++i) {
            char ch = chars[i];
            if (ch == 'c') {
                if (k > 0) {
                    --k;
                } else {
                    ++res;
                }
                ++c;
            } else if (ch == 'r') {
                ++r;
                --c;
            } else if (ch == 'o') {
                ++o;
                --r;
            } else if (ch == 'a') {
                ++a;
                --o;
            } else if (ch == 'k') {
                ++k;
                --a;
            }
            if (c < 0 || r < 0 || o < 0 || a < 0) {
                break;
            }
        }
        if (c != 0 || r != 0 || o != 0 || a != 0) {
            return -1;
        }
        return res;

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

    // 6079. 价格减免
    public String discountPrices(String sentence, int discount) {
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < sentence.length(); ++i) {
            if (i == 0 && sentence.charAt(i) == '$' && (i + 1) < sentence.length()) {
                int spaceIndex = sentence.indexOf(" ", i + 1);
                if (spaceIndex != -1) {
                    String sub = sentence.substring(i + 1, spaceIndex);
                    if (isNum(sub)) {
                        double dis = (100 - discount) * 0.01d;
                        double d = Double.parseDouble(sub) * dis;
                        String r = String.format("%.2f", d);
                        res.append("$");
                        res.append(r);

                        i = spaceIndex - 1;
                    } else {
                        res.append(sentence.charAt(i));
                    }
                } else if (i + 1 < sentence.length()) {
                    String sub = sentence.substring(i + 1);
                    if (isNum(sub)) {
                        double dis = (100 - discount) * 0.01d;
                        double d = Double.parseDouble(sub) * dis;
                        String r = String.format("%.2f", d);
                        res.append("$");
                        res.append(r);
                        i = sentence.length() - 1;
                    } else {
                        res.append(sentence.charAt(i));
                    }
                } else {
                    res.append(sentence.charAt(i));
                }
            } else if (sentence.charAt(i) == ' ' && (i + 1) < sentence.length() && sentence.charAt(i + 1) == '$') {
                int spaceIndex = sentence.indexOf(" ", i + 1);
                if (spaceIndex != -1) {
                    String sub = sentence.substring(i + 2, spaceIndex);
                    if (isNum(sub)) {
                        double dis = (100 - discount) * 0.01d;
                        double d = Double.parseDouble(sub) * dis;
                        String r = String.format("%.2f", d);
                        res.append(" ");
                        res.append("$");
                        res.append(r);

                        i = spaceIndex - 1;
                    } else {
                        res.append(sentence.charAt(i));
                    }
                } else if (i + 2 < sentence.length()) {
                    String sub = sentence.substring(i + 2);
                    if (isNum(sub)) {
                        double dis = (100 - discount) * 0.01d;
                        double d = Double.parseDouble(sub) * dis;
                        String r = String.format("%.2f", d);
                        res.append(" ");
                        res.append("$");
                        res.append(r);
                        i = sentence.length() - 1;
                    } else {
                        res.append(sentence.charAt(i));
                    }
                } else {
                    res.append(sentence.charAt(i));
                }

            } else {
                res.append(sentence.charAt(i));
            }
        }
        return res.toString();

    }

    private boolean isNum(String str) {
        if (str.isEmpty()) {
            return false;
        }
        Matcher isNum = Pattern.compile("[0-9]*").matcher(str);
        return isNum.matches();
    }

    public int minimumObstacles(int[][] grid) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int m = grid.length;
        int n = grid[0].length;
        boolean[][] visited = new boolean[m][n];
        int[][] dp = new int[m][n];
        PriorityQueue<int[]> queue = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return dp[o1[0]][o1[1]] - dp[o2[0]][o2[1]];
            }

        });
        queue.offer(new int[] { 0, 0 });
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0];
            int y = cur[1];
            for (int[] direction : directions) {
                int nx = x + direction[0];
                int ny = y + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny]) {
                    visited[nx][ny] = true;
                    dp[nx][ny] = grid[nx][ny] + dp[x][y];
                    queue.offer(new int[] { nx, ny });
                }
            }
        }
        return dp[m - 1][n - 1];

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

    // 1775. 通过最少操作次数使数组的和相等 (Equal Sum Arrays With Minimum Number of Operations)
    // --贪心
    public int minOperations(int[] nums1, int[] nums2) {
        int sumA = Arrays.stream(nums1).sum();
        int sumB = Arrays.stream(nums2).sum();

        if (sumA == sumB) {
            return 0;
        }
        if (sumA > sumB) {
            int temp = sumA;
            sumA = sumB;
            sumB = temp;

            int[] tempArr = nums1;
            nums1 = nums2;
            nums2 = tempArr;
        }
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int res = 0;
        int i = 0;
        int j = nums2.length - 1;
        while (i < nums1.length && j >= 0 && sumA < sumB) {
            if (6 - nums1[i] > nums2[j] - 1) {
                sumA += 6 - nums1[i++];
            } else {
                sumB -= nums2[j--] - 1;
            }
            ++res;
        }
        while (i < nums1.length && sumA < sumB) {
            sumA += 6 - nums1[i++];
            ++res;
        }

        while (j >= 0 && sumA < sumB) {
            sumB -= nums2[j--] - 1;
            ++res;
        }
        return sumA >= sumB ? res : -1;

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
            for (int i = 0; i < n / 2; ++i) {
                if (i % 2 == 0) {
                    nums[i] = Math.min(nums[i * 2], nums[i * 2 + 1]);
                } else {
                    nums[i] = Math.max(nums[i * 2], nums[i * 2 + 1]);
                }
            }
            n /= 2;
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

    // 2265. 统计值等于子树平均值的节点数 (Count Nodes Equal to Average of Subtree) --dfs
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
        int count = left[1] + right[1] + 1;
        if (sum / count == root.val) {
            ++res2265;
        }
        return new int[] { sum, count };
    }

    // 面试题 08.07. 无重复字符串的排列组合 (Permutation I LCCI) --回溯
    private List<String> res0807;

    public String[] permutation(String S) {
        res0807 = new ArrayList<>();
        int n = S.length();
        boolean[] visited = new boolean[n];
        char[] chars = S.toCharArray();
        StringBuilder path = new StringBuilder();
        dfs0807(chars, 0, visited, path);
        return res0807.toArray(new String[0]);

    }

    private void dfs0807(char[] chars, int index, boolean[] visited, StringBuilder path) {
        if (chars.length == index) {
            res0807.add(path.toString());
            return;
        }
        for (int i = 0; i < chars.length; ++i) {
            if (!visited[i]) {
                path.append(chars[i]);
                visited[i] = true;
                dfs0807(chars, index + 1, visited, path);
                visited[i] = false;
                path.deleteCharAt(path.length() - 1);
            }
        }
    }

    // 46. 全排列 (Permutations) --回溯
    // 剑指 Offer II 083. 没有重复元素集合的全排列
    // 排列：需要用used数组
    // 无重复元素：不需要排序
    private List<List<Integer>> res46;

    public List<List<Integer>> permute(int[] nums) {
        res46 = new ArrayList<>();
        int n = nums.length;
        boolean[] visited = new boolean[n];
        List<Integer> path = new ArrayList<>();
        dfs46(nums, 0, visited, path);
        return res46;

    }

    private void dfs46(int[] nums, int index, boolean[] visited, List<Integer> path) {
        if (index == nums.length) {
            res46.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < nums.length; ++i) {
            if (!visited[i]) {
                path.add(nums[i]);
                visited[i] = true;
                dfs46(nums, index + 1, visited, path);
                visited[i] = false;
                path.remove(path.size() - 1);
            }
        }
    }

    // 6095. 强密码检验器 II (Strong Password Checker II)
    public boolean strongPasswordCheckerII(String password) {
        if (password.length() < 8) {
            return false;
        }
        boolean hasLower = false;
        boolean hasUpper = false;
        boolean hasDigit = false;
        boolean hasSpecial = false;
        char[] chars = password.toCharArray();
        for (int i = 0; i < chars.length; ++i) {
            if (i != 0) {
                if (chars[i] == chars[i - 1]) {
                    return false;
                }
            }
            if (Character.isUpperCase(chars[i])) {
                hasUpper = true;
            }
            if (Character.isLowerCase(chars[i])) {
                hasLower = true;
            }
            if (Character.isDigit(chars[i])) {
                hasDigit = true;
            }
            if (chars[i] == '!' || chars[i] == '@' || chars[i] == '$' || chars[i] == '#' || chars[i] == '%'
                    || chars[i] == '^' || chars[i] == '&' || chars[i] == '*' || chars[i] == '(' || chars[i] == ')'
                    || chars[i] == '-' || chars[i] == '+') {
                hasSpecial = true;
            }
        }
        return hasLower && hasUpper && hasDigit && hasSpecial;

    }

    // 6096. 咒语和药水的成功对数 (Successful Pairs of Spells and Potions) --二分查找
    public int[] successfulPairs(int[] spells, int[] potions, long success) {
        Arrays.sort(potions);
        long[] longPotions = new long[potions.length];
        for (int i = 0; i < potions.length; ++i) {
            longPotions[i] = potions[i];
        }
        int[] res = new int[spells.length];
        for (int i = 0; i < spells.length; ++i) {
            int index = search6096(spells[i], longPotions, success);
            res[i] = longPotions.length - index;
        }
        return res;

    }

    private int search6096(int spell, long[] longPotions, long success) {
        if (longPotions[longPotions.length - 1] * spell < success) {
            return longPotions.length;
        }
        int left = 0;
        int right = longPotions.length - 1;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (longPotions[mid] * spell >= success) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
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
        for (int i = 0; i < brackets.length; ++i) {
            if (income == 0) {
                break;
            }
            int min = 0;
            if (i == 0) {
                min = Math.min(income, brackets[i][0]);
            } else {
                min = Math.min(income, brackets[i][0] - brackets[i - 1][0]);
            }
            res += min * brackets[i][1] * 0.01d;
            income -= min;
        }
        return res;

    }

    // 5270. 网格中的最小路径代价 (Minimum Path Cost in a Grid)
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

    // 2305. 公平分发饼干 (Fair Distribution of Cookies)
    private int res2305;

    public int distributeCookies(int[] cookies, int k) {
        res2305 = Integer.MAX_VALUE;
        Arrays.sort(cookies);
        dfs2305(cookies, cookies.length - 1, new int[k]);
        return res2305;

    }

    private void dfs2305(int[] cookies, int start, int[] cur) {
        if (start < 0) {
            int max = Integer.MIN_VALUE;
            for (int c : cur) {
                max = Math.max(c, max);
            }
            res2305 = Math.min(res2305, max);
            return;
        }
        for (int i = 0; i < cur.length; ++i) {
            if (i > 0 && start == cookies.length) {
                return;
            }
            cur[i] += cookies[start];
            dfs2305(cookies, start - 1, cur);
            cur[i] -= cookies[start];
        }
    }

    // 17. 电话号码的字母组合 (Letter Combinations of a Phone Number) --回溯
    public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();
        if (digits.length() == 0) {
            return res;
        }
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

        StringBuilder builder = new StringBuilder();
        backtrack17(res, builder, digits, 0, map);
        return res;
    }

    private void backtrack17(List<String> res, StringBuilder builder, String digits, int index,
            Map<Character, String> map) {
        if (index == digits.length()) {
            res.add(builder.toString());
            return;
        }
        char c = digits.charAt(index);
        String s = map.get(c);
        for (int i = 0; i < s.length(); ++i) {
            builder.append(s.charAt(i));
            backtrack17(res, builder, digits, index + 1, map);
            builder.deleteCharAt(builder.length() - 1);
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

    // 剑指 Offer II 085. 生成匹配的括号 --backtrack
    // 22. 括号生成 (Generate Parentheses)
    // 面试题 08.09.括号 (Bracket LCCI)
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        StringBuilder builder = new StringBuilder();
        backtrack22(res, builder, 0, 0, n);
        return res;

    }

    private void backtrack22(List<String> res, StringBuilder builder, int open, int close, int n) {
        if (builder.length() == n * 2) {
            res.add(builder.toString());
            return;
        }
        if (open < n) {
            builder.append('(');
            backtrack22(res, builder, open + 1, close, n);
            builder.deleteCharAt(builder.length() - 1);
        }
        if (open > close) {
            builder.append(')');
            backtrack22(res, builder, open, close + 1, n);
            builder.deleteCharAt(builder.length() - 1);
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

    // 剑指 Offer II 085. 生成匹配的括号 --bfs
    // 22. 括号生成 (Generate Parentheses)
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
        block37_3 = new int[n][n];
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
                        int mask = ~(line37_3[i] | column37_3[j] | block37_3[i / 3][j / 3]) & 0x1FF;
                        // 只有一个1，即该位置的值已唯一确定
                        if ((mask & (mask - 1)) == 0) {
                            int digit = Integer.bitCount(mask - 1);
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
        int mask = ~(line37_3[x] | column37_3[y] | block37_3[x / 3][y / 3]) & 0x1FF;
        for (; mask != 0 && !valid37_3; mask &= (mask - 1)) {
            int bit = mask & (-mask);
            int digit = Integer.bitCount(bit - 1);
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
    // 剑指 Offer II 082. 含有重复元素集合的组合
    // 组合：不需要用used数组
    // 有重复元素：需要排序
    // 每个元素只能用一次 ：回溯的时候 index = i + 1
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        int sum = 0;
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        backtrack40(res, candidates, path, sum, target, 0);
        return res;

    }

    private void backtrack40(List<List<Integer>> res, int[] candidates, List<Integer> path, int sum, int target,
            int index) {
        if (sum == target) {
            res.add(new ArrayList<>(path));
            return;
        }
        if (sum > target) {
            return;
        }
        for (int i = index; i < candidates.length; ++i) {
            if (i > index && candidates[i] == candidates[i - 1]) {
                continue;
            }
            sum += candidates[i];
            path.add(candidates[i]);
            backtrack40(res, candidates, path, sum, target, i + 1);
            path.remove(path.size() - 1);
            sum -= candidates[i];
        }
    }

    // 47. 全排列 II (Permutations II) --回溯
    // 剑指 Offer II 084. 含有重复元素集合的全排列
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
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        backtrack39(candidates, path, res, target, 0, 0);
        return res;

    }

    private void backtrack39(int[] candidates, List<Integer> path, List<List<Integer>> res, int target, int sum,
            int index) {
        if (target == sum) {
            res.add(new ArrayList<>(path));
            return;
        }
        if (sum > target) {
            return;
        }
        for (int i = index; i < candidates.length; ++i) {
            sum += candidates[i];
            path.add(candidates[i]);
            backtrack39(candidates, path, res, target, sum, i);
            path.remove(path.size() - 1);
            sum -= candidates[i];
        }
    }

    // 77. 组合 (Combinations)
    // 剑指 Offer II 080. 含有 k 个元素的组合
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
    // 剑指 Offer II 080. 含有 k 个元素的组合
    // 组合：不需要用used数组
    // 无重复元素：不需要排序
    // 每个元素只能使用一次 ：回溯的时候 index = i + 1
    public List<List<Integer>> combine2(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        int[] nums = new int[n];
        for (int i = 1; i <= n; ++i) {
            nums[i - 1] = i;
        }
        List<Integer> path = new ArrayList<>();
        backtrack77(res, nums, path, 0, k);
        return res;

    }

    private void backtrack77(List<List<Integer>> res, int[] nums, List<Integer> path, int index, int k) {
        if (path.size() == k) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = index; i < nums.length; ++i) {
            path.add(nums[i]);
            backtrack77(res, nums, path, i + 1, k);
            path.remove(path.size() - 1);
        }
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
        int[] queens = new int[n];
        Arrays.fill(queens, -1);
        backtrack51_2(res, queens, n, 0, 0, 0, 0);
        return res;
    }

    private void backtrack51_2(List<List<String>> res, int[] queens, int n, int row, int colunms, int diagonal1,
            int diagonal2) {
        if (row == n) {
            List<String> board = generate51_2(queens);
            res.add(board);
            return;
        }
        int availablePositions = ((1 << n) - 1) & (~(colunms | diagonal1 | diagonal2));
        while (availablePositions != 0) {
            int position = availablePositions & (-availablePositions);
            int index = Integer.bitCount(position - 1);
            queens[row] = index;
            backtrack51_2(res, queens, n, row + 1, colunms | position, (diagonal1 | position) << 1,
                    (diagonal2 | position) >> 1);
            queens[row] = -1;
            availablePositions &= availablePositions - 1;
        }

    }

    private List<String> generate51_2(int[] queens) {
        int n = queens.length;
        List<String> res = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            char[] sub = new char[n];
            Arrays.fill(sub, '.');
            sub[queens[i]] = 'Q';
            res.add(String.valueOf(sub));
        }
        return res;
    }

    // 52. N皇后 II (N-Queens II) --回溯
    private int res52;

    public int totalNQueens(int n) {
        backtrack52(n, 0, 0, 0, 0);
        return res52;

    }

    private void backtrack52(int n, int row, int colunms, int diagonal1, int diagonal2) {
        if (row == n) {
            ++res52;
            return;
        }
        int availablePositions = ((1 << n) - 1) & (~(colunms | diagonal1 | diagonal2));
        while (availablePositions != 0) {
            int position = availablePositions & (-availablePositions);
            backtrack52(n, row + 1, colunms | position, (diagonal1 | position) << 1, (diagonal2 | position) >> 1);
            availablePositions &= availablePositions - 1;
        }
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
        if (start == s.length() || builder.length() == s.length() + 3) {
            if (start == s.length() && builder.length() == s.length() + 3) {
                res.add(builder.toString());
            }
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
            if (lastIndexOfDot == -1) {
                builder.setLength(0);
            } else {
                builder.delete(lastIndexOfDot, builder.length());
            }
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

    public String greatestLetter(String s) {
        // 第0维 : 大写
        // 第1维 : 小写
        boolean[][] map = new boolean[26][2];
        for (char c : s.toCharArray()) {
            if (Character.isUpperCase(c)) {
                map[c - 'A'][0] = true;
            } else {
                map[c - 'a'][1] = true;
            }
        }
        for (int i = map.length - 1; i >= 0; --i) {
            if (map[i][0] && map[i][1]) {
                char c = (char) (i + 'A');
                return String.valueOf(c);
            }
        }
        return "";

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
    public List<List<String>> partition(String s) {
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(dp[i], true);
        }
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                dp[i][j] = dp[i + 1][j - 1] && s.charAt(i) == s.charAt(j);
            }
        }
        List<List<String>> res = new ArrayList<>();
        List<String> cur = new ArrayList<>();
        backtrack131(res, cur, s, 0, dp);
        return res;
    }

    private void backtrack131(List<List<String>> res, List<String> cur, String s, int index, boolean[][] dp) {
        if (index == s.length()) {
            res.add(new ArrayList<>(cur));
            return;
        }
        for (int i = index; i < s.length(); ++i) {
            if (dp[index][i]) {
                cur.add(s.substring(index, i + 1));
                backtrack131(res, cur, s, i + 1, dp);
                cur.remove(cur.size() - 1);
            }
        }
    }

    // 216. 组合总和 III (Combination Sum III) --回溯
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> res = new ArrayList<>();
        // 1--9的和最大是45
        if (n > 45) {
            return res;
        }

        List<Integer> cur = new ArrayList<>();
        backtrack216(res, cur, k, n, 0, 0);
        return res;
    }

    private void backtrack216(List<List<Integer>> res, List<Integer> cur, int k, int n, int index, int sum) {
        if (cur.size() > k || sum > n) {
            return;
        }
        if (sum == n && cur.size() == k) {
            res.add(new ArrayList<>(cur));
            return;
        }
        for (int i = index; i < 9; ++i) {
            int num = i + 1;
            cur.add(num);
            sum += num;
            backtrack216(res, cur, k, n, i + 1, sum);
            sum -= num;
            cur.remove(cur.size() - 1);
        }
    }

    // 216. 组合总和 III (Combination Sum III) --位运算 + 二进制枚举
    public List<List<Integer>> combinationSum3_2(int k, int n) {
        List<List<Integer>> res = new ArrayList<>();
        if (n > 45) {
            return res;
        }
        List<Integer> cur = new ArrayList<>();
        // 第0位---第8位 表示整数 1---9
        for (int i = 0; i < (1 << 9); ++i) {
            if (Integer.bitCount(i) == k) {
                int sum = 0;
                int mask = i;
                while (mask != 0) {
                    int last = mask & (-mask);
                    int num = Integer.bitCount(last - 1) + 1;
                    sum += num;
                    cur.add(num);
                    if (sum > n) {
                        break;
                    }
                    mask &= mask - 1;
                }
                if (sum == n) {
                    res.add(new ArrayList<>(cur));
                }
                cur.clear();
            }
        }
        return res;

    }

    // 494. 目标和 (Target Sum) --回溯
    // 剑指 Offer II 102. 加减的目标值
    private int res494;

    public int findTargetSumWays(int[] nums, int target) {
        backtrack494(nums, target, 0, 0);
        return res494;
    }

    private void backtrack494(int[] nums, int target, int index, int sum) {
        if (index == nums.length) {
            if (sum == target) {
                ++res494;
            }
            return;
        }
        backtrack494(nums, target, index + 1, sum + nums[index]);
        backtrack494(nums, target, index + 1, sum - nums[index]);
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

    // 1079. 活字印刷 (Letter Tile Possibilities)
    private int res1079;

    public int numTilePossibilities(String tiles) {
        int n = tiles.length();
        char[] chars = tiles.toCharArray();
        Arrays.sort(chars);
        boolean[] used = new boolean[n];
        for (int i = 1; i <= n; ++i) {
            backtrack1079(chars, i, used);
        }
        return res1079;

    }

    private void backtrack1079(char[] chars, int n, boolean[] used) {
        if (n == 0) {
            ++res1079;
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
            backtrack1079(chars, n - 1, used);
            used[i] = false;
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
        Map<Integer, Integer> map = new TreeMap<>();
        for (int[] num : nums) {
            for (int n : num) {
                map.put(n, map.getOrDefault(n, 0) + 1);
            }
        }
        List<Integer> res = new ArrayList<>();
        int n = nums.length;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (entry.getValue() == n) {
                res.add(entry.getKey());
            }
        }
        return res;
    }

    // 2248. 多个数组求交集 (Intersection of Multiple Arrays)
    public List<Integer> intersection2(int[][] nums) {
        int n = nums.length;
        Set<Integer> cur = new HashSet<>();
        for (int num : nums[0]) {
            cur.add(num);
        }
        for (int i = 1; i < n; ++i) {
            Set<Integer> temp = new HashSet<>();
            for (int num : nums[i]) {
                if (cur.contains(num)) {
                    temp.add(num);
                }
            }
            cur = temp;
        }
        List<Integer> res = new ArrayList<>(cur);
        Collections.sort(res);
        return res;
    }

    // 6104. 统计星号
    public int countAsterisks(String s) {
        int count = 0;
        int res = 0;
        for (char c : s.toCharArray()) {
            if (c == '|') {
                ++count;
            } else if (c == '*' && ((count & 1) == 0)) {
                ++res;
            }
        }
        return res;
    }

    // 6106. 统计无向图中无法互相到达点对数
    public long countPairs(int n, int[][] edges) {
        UnionFind6106 union = new UnionFind6106(n);
        for (int[] edge : edges) {
            union.union(edge[0], edge[1]);
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            int root = union.getRoot(i);
            map.put(root, map.getOrDefault(root, 0) + 1);
        }
        long res = 0l;
        long sum = 0l;
        for (int num : map.values()) {
            sum += num;
        }
        for (int num : map.values()) {
            sum -= num;
            res += num * sum;
        }
        return res;
    }

    public class UnionFind6106 {
        private int[] parent;
        private int[] rank;

        public UnionFind6106(int n) {
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

    // 6107. 不同骰子序列的数目
    public int distinctSequences(int n) {
        return backtrack6107(0, 0, 0, n);
    }

    private int backtrack6107(int lastButOne, int last, int index, int n) {
        if (index == n) {
            return 1;
        }
        int res = 0;
        for (int i = 1; i <= 6; ++i) {
            if (lastButOne != i && last != i && getGCD6107(last, i) == 1 || last == 0) {
                res += backtrack6107(last, i, index + 1, n);
                res %= 1000000007;
            }
        }
        return res;
    }

    private int getGCD6107(int a, int b) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
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

    // 6100. 统计放置房子的方式数 (Count Number of Ways to Place Houses)
    public int countHousePlacements(int n) {
        final int MOD = 1000000007;
        // 放置
        int put = 1;
        // 不放置
        int notPut = 1;
        for (int i = 1; i < n; ++i) {
            int temp = put;
            // 当前放置的数量，为之前一个地方不放置的值
            put = notPut;
            // 当前不放置的数量，为之前一个地方不放置与放置之和
            notPut = (notPut + temp) % MOD;
        }
        // 最终结果为最后一个地方放置与不放置之和的平方
        long res = (put + notPut) % MOD;
        return (int) ((res * res) % MOD);

    }

    // 5229. 拼接数组的最大分数 (Maximum Score Of Spliced Array)
    public int maximumsSplicedArray(int[] nums1, int[] nums2) {
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
            if (nodeA.val != nodeB.val) {
                return false;
            }
            if ((nodeA.left == null && nodeB.left != null) || (nodeA.left != null && nodeB.left == null)) {
                return false;
            }
            if (nodeA.left != null && nodeB.left != null) {
                queueA.offer(nodeA.left);
                queueB.offer(nodeB.left);
            }

            if ((nodeA.right == null && nodeB.right != null) || (nodeA.right != null && nodeB.right == null)) {
                return false;
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
        Queue<int[]> priorityQueue = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return arr[o1[0]] * arr[o2[1]] - arr[o2[0]] * arr[o1[1]];
            }

        });
        int n = arr.length;
        for (int j = 1; j < n; ++j) {
            priorityQueue.offer(new int[] { 0, j });
        }
        for (int i = 1; i < k; ++i) {
            int[] cur = priorityQueue.poll();
            if (cur[0] + 1 < cur[1]) {
                priorityQueue.offer(new int[] { cur[0] + 1, cur[1] });
            }
        }
        int[] index = priorityQueue.peek();
        return new int[] { arr[index[0]], arr[index[1]] };

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

    // 2271. 毯子覆盖的最多白色砖块数 (Maximum White Tiles Covered by a Carpet) --贪心
    public int maximumWhiteTiles(int[][] tiles, int carpetLen) {
        int n = tiles.length;
        Arrays.sort(tiles, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }

        });
        int r = 0;
        int res = 0;
        int count = 0;
        for (int l = 0; l < n; ++l) {
            if (l != 0) {
                count -= tiles[l - 1][1] - tiles[l - 1][0] + 1;
            }
            while (r < n && tiles[l][0] + carpetLen > tiles[r][1]) {
                count += tiles[r][1] - tiles[r][0] + 1;
                ++r;
            }
            if (r == n) {
                return Math.max(res, count);
            }
            int extra = Math.max(0, tiles[l][0] + carpetLen - tiles[r][0]);
            res = Math.max(res, count + extra);
        }
        return res;

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

    // 2266. 统计打字方案数 (Count Number of Texts) --dp
    public int countTexts(String pressedKeys) {
        final int MOD = 1000000007;
        int n = pressedKeys.length();
        List<Long> dp3 = new ArrayList<>();
        dp3.add(1L);
        dp3.add(1L);
        dp3.add(2L);
        dp3.add(4L);
        List<Long> dp4 = new ArrayList<>();
        dp4.add(1L);
        dp4.add(1L);
        dp4.add(2L);
        dp4.add(4L);
        for (int i = 4; i < n + 1; ++i) {
            dp3.add((dp3.get(i - 1) + dp3.get(i - 2) + dp3.get(i - 3)) % MOD);
            dp4.add((dp4.get(i - 1) + dp4.get(i - 2) + dp4.get(i - 3) + dp4.get(i - 4)) % MOD);
        }
        long res = 1l;
        int count = 1;
        for (int i = 1; i < n; ++i) {
            if (pressedKeys.charAt(i) == pressedKeys.charAt(i - 1)) {
                ++count;
            } else {
                if (pressedKeys.charAt(i - 1) == '7' || pressedKeys.charAt(i - 1) == '9') {
                    res *= dp4.get(count);
                } else {
                    res *= dp3.get(count);
                }
                res %= MOD;
                count = 1;
            }
        }
        if (pressedKeys.charAt(n - 1) == '7' || pressedKeys.charAt(n - 1) == '9') {
            res *= dp4.get(count);
        } else {
            res *= dp3.get(count);
        }
        res %= MOD;
        return (int) res;

    }

    // 2325. 解密消息 (Decode the Message)
    public String decodeMessage(String key, String message) {
        Map<Character, Character> map = new HashMap<>();
        char c = 'a';
        for (char ch : key.toCharArray()) {
            if (!Character.isWhitespace(ch)) {
                if (!map.containsKey(ch)) {
                    map.put(ch, c++);
                }
                // if (map.putIfAbsent(ch, c) == null) {
                // ++c;
                // }
            }
        }
        StringBuilder res = new StringBuilder();
        for (char ch : message.toCharArray()) {
            if (Character.isWhitespace(ch)) {
                res.append(ch);
            } else {
                res.append(map.get(ch));
            }
        }
        return res.toString();

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

    // 6110. 网格图中递增路径的数目 -- 记忆化搜索
    public int countPaths(int[][] grid) {
        final int MOD = 1000000007;
        final int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        int res = 0;
        for (int i = 0; i < m; ++i) {
            Arrays.fill(dp[i], -1);
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                res = (res + dfs6110(i, j, grid, dp, directions, MOD)) % MOD;
            }
        }
        return res;
    }

    private int dfs6110(int i, int j, int[][] grid, int[][] dp, int[][] directions, int MOD) {
        if (dp[i][j] != -1) {
            return dp[i][j];
        }
        int m = grid.length;
        int n = grid[0].length;
        int res = 1;
        for (int[] direction : directions) {
            int nx = direction[0] + i;
            int ny = direction[1] + j;
            if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] > grid[i][j]) {
                res = (res + dfs6110(nx, ny, grid, dp, directions, MOD)) % MOD;
            }
        }
        return dp[i][j] = res;
    }

    // 1143. 最长公共子序列 (Longest Common Subsequence) --二维dp
    // 剑指 Offer II 095. 最长公共子序列
    public int longestCommonSubsequence(String text1, String text2) {
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
    public int longestCommonSubsequence2(String text1, String text2) {
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

    // 1641. 统计字典序元音字符串的数目 (Count Sorted Vowel Strings) --dp
    public int countVowelStrings(int n) {
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
        int max = -1;
        int secondMax = -1;
        for (char c : s.toCharArray()) {
            if (Character.isDigit(c)) {
                int num = c - '0';
                if (num > max) {
                    secondMax = max;
                    max = num;
                } else if (max > num && num > secondMax) {
                    secondMax = num;
                }
            }
        }
        return secondMax;
    }

    // 522. 最长特殊序列 II (Longest Uncommon Subsequence II)
    public int findLUSlength(String[] strs) {
        int n = strs.length;
        int res = -1;
        search: for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    continue;
                }
                if (isLUS522(strs[i], strs[j])) {
                    continue search;
                }
            }
            res = Math.max(res, strs[i].length());
        }
        return res;

    }

    // 判断s1是否为s2的子序列
    private boolean isLUS522(String s1, String s2) {
        int i = 0;
        int j = 0;
        while (i < s1.length() && j < s2.length()) {
            if (s1.charAt(i) == s2.charAt(j)) {
                ++i;
            }
            ++j;
        }
        return i == s1.length();
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
            StringBuilder builder = new StringBuilder();
            for (int j = n - 1; j >= 0; --j) {
                matrix[i][j] ^= matrix[i][0];
                builder.append(matrix[i][j]);
            }
            map.put(builder.toString(), map.getOrDefault(builder.toString(), 0) + 1);
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
        for (List<String> know : knowledge) {
            map.put(know.get(0), know.get(1));
        }
        StringBuilder res = new StringBuilder();
        int index = 0;
        while (index < s.length()) {
            if (s.charAt(index) != '(') {
                res.append(s.charAt(index++));
            } else {
                int right = index + 1;
                while (right < s.length() && s.charAt(right) != ')') {
                    ++right;
                }
                res.append(map.getOrDefault(s.substring(index + 1, right), "?"));
                index = right + 1;
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
    public int wordCount1(String[] startWords, String[] targetWords) {
        Map<Integer, Integer> map = new HashMap<>();
        for (String targetWord : targetWords) {
            int mask = getMask2153(targetWord);
            map.put(mask, map.getOrDefault(mask, 0) + 1);
        }
        int res = 0;
        for (String startWord : startWords) {
            int mask = getMask2153(startWord);
            for (int i = 0; i < 26; ++i) {
                int candidateMask = mask | (1 << i);
                if (candidateMask != mask && map.containsKey(candidateMask)) {
                    res += map.get(candidateMask);
                    map.remove(candidateMask);
                }
            }
        }
        return res;
    }

    private int getMask2153(String word) {
        int mask = 0;
        for (char c : word.toCharArray()) {
            mask |= 1 << (c - 'a');
        }
        return mask;
    }

    // 2135. 统计追加字母可以获得的单词数 (Count Words Obtained After Adding a Letter)
    public int wordCount2(String[] startWords, String[] targetWords) {
        Set<Integer> set = new HashSet<>();
        for (String startWord : startWords) {
            int mask = getMask2153(startWord);
            set.add(mask);
        }
        int res = 0;
        for (String targetWord : targetWords) {
            int mask = getMask2153(targetWord);
            for (int i = 0; i < targetWord.length(); ++i) {
                if (set.contains(mask ^ (1 << (targetWord.charAt(i) - 'a')))) {
                    ++res;
                    break;
                }
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
        int space = 0;
        List<String> list = new ArrayList<>();
        for (int i = 0; i < text.length(); ++i) {
            char c = text.charAt(i);
            if (c == ' ') {
                ++space;
            } else {
                int right = i;
                while (right < text.length() && text.charAt(right) != ' ') {
                    ++right;
                }
                list.add(text.substring(i, right));
                i = right - 1;
            }
        }

        int intervalSpaceCount = 0;
        if (list.size() > 1) {
            intervalSpaceCount = space / (list.size() - 1);
            space = space % (list.size() - 1);
        }
        int index = 0;
        StringBuilder res = new StringBuilder();
        while (index < list.size()) {
            res.append(list.get(index));
            for (int i = 0; i < intervalSpaceCount && index != list.size() - 1; ++i) {
                res.append(" ");
            }
            ++index;
        }
        while (space-- != 0) {
            res.append(" ");
        }
        return res.toString();

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
            return root.val == 1;
        }
        if (root.val == 2) {
            return evaluateTree(root.left) || evaluateTree(root.right);
        } else {
            return evaluateTree(root.left) && evaluateTree(root.right);
        }
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

    // 6112. 装满杯子需要的最短总时长
    public int fillCups(int[] amount) {
        PriorityQueue<Integer> queue = new PriorityQueue<>(new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return o2.compareTo(o1);
            }

        });
        for (int a : amount) {
            queue.offer(a);
        }
        int count = 0;
        while (!queue.isEmpty()) {
            int poll1 = queue.poll();
            int poll2 = queue.poll();
            int poll3 = queue.poll();
            if (poll1 == 0) {
                return count;
            }
            if (poll2 == 0) {
                return poll1 + count;
            }
            if (poll3 == 0) {
                return poll1 + count;
            }
            --poll1;
            --poll2;
            count += 1;
            queue.offer(poll1);
            queue.offer(poll2);
            queue.offer(poll3);
        }
        return count;

    }

    // 6113. 无限集中的最小数字
    class SmallestInfiniteSet {
        private TreeSet<Integer> set;

        public SmallestInfiniteSet() {
            set = new TreeSet<>();
            for (int i = 1; i <= 1000; ++i) {
                set.add(i);
            }
        }

        public int popSmallest() {
            int res = set.first();
            set.remove(res);
            return res;
        }

        public void addBack(int num) {
            set.add(num);
        }
    }

    // 6114. 移动片段得到字符串
    public boolean canChange(String start, String target) {
        List<int[]> starts = new ArrayList<>();
        List<int[]> targets = new ArrayList<>();
        int n = start.length();
        for (int i = 0; i < n; ++i) {
            if (start.charAt(i) != '_') {
                if (start.charAt(i) == 'L') {
                    starts.add(new int[] { 0, i });
                } else {
                    starts.add(new int[] { 1, i });
                }
            }

            if (target.charAt(i) != '_') {
                if (target.charAt(i) == 'L') {
                    targets.add(new int[] { 0, i });
                } else {
                    targets.add(new int[] { 1, i });
                }
            }
        }
        if (starts.size() != targets.size()) {
            return false;
        }
        int m = starts.size();
        for (int i = 0; i < m; ++i) {
            if (starts.get(i)[0] != targets.get(i)[0]) {
                return false;
            }
            if (starts.get(i)[0] == 0 && starts.get(i)[1] < targets.get(i)[1]) {
                return false;
            }
            if (starts.get(i)[0] == 1 && starts.get(i)[1] > targets.get(i)[1]) {
                return false;
            }
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

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, List<Node> _children) {
            val = _val;
            children = _children;
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

    // 1043. 分隔数组以得到最大和 (Partition Array for Maximum Sum) --dp
    public int maxSumAfterPartitioning(int[] arr, int k) {
        int n = arr.length;
        int[] dp = new int[n];
        for (int i = 0; i < k; ++i) {
            dp[i] = Arrays.stream(arr, 0, i + 1).max().getAsInt() * (i + 1);
        }
        for (int i = k; i < n; ++i) {
            for (int j = 1; j <= k; ++j) {
                dp[i] = Math.max(dp[i], dp[i - j] + Arrays.stream(arr, i - j + 1, i + 1).max().getAsInt() * j);
            }
        }
        return dp[n - 1];

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

    // 337. 打家劫舍 III (House Robber III)
    public int rob(TreeNode root) {
        Map<TreeNode, int[]> map = new HashMap<>();
        dfs337(root, map);
        return Math.max(map.get(root)[0], map.get(root)[1]);
    }

    private void dfs337(TreeNode node, Map<TreeNode, int[]> map) {
        if (node == null) {
            return;
        }
        dfs337(node.left, map);
        dfs337(node.right, map);
        // dp[0]:不偷当前节点时的最大值
        // dp[1]:偷当前节点时的最大值
        int[] dp = new int[2];
        if (node.left != null) {
            int[] childLeft = map.get(node.left);
            dp[0] += Math.max(childLeft[0], childLeft[1]);
            dp[1] += childLeft[0];
        }
        if (node.right != null) {
            int[] childRight = map.get(node.right);
            dp[0] += Math.max(childRight[0], childRight[1]);
            dp[1] += childRight[0];
        }
        dp[1] += node.val;
        map.put(node, dp);

    }

    // 337. 打家劫舍 III (House Robber III)
    public int rob2(TreeNode root) {
        int[] max = dfs337_2(root);
        return Math.max(max[0], max[1]);

    }

    private int[] dfs337_2(TreeNode node) {
        if (node == null) {
            return new int[] { 0, 0 };
        }
        // l[0]: 不偷node节点的左子树的最大值
        // l[1]: 偷node节点的左子树的最大值
        int[] l = dfs337_2(node.left);
        // r[0]: 不偷node节点的右子树的最大值
        // r[1]: 偷node节点的右子树的最大值
        int[] r = dfs337_2(node.right);
        int notStole = Math.max(l[0], l[1]) + Math.max(r[0], r[1]);
        int stole = node.val + l[0] + r[0];
        return new int[] { notStole, stole };
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
        int pairs = 0;
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            if (!set.add(num)) {
                set.remove(num);
                ++pairs;
            }
        }
        return new int[] { pairs, set.size() };
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

    // 543. 二叉树的直径 (Diameter of Binary Tree)
    private int res543;

    public int diameterOfBinaryTree(TreeNode root) {
        dfs543(root);
        return res543;

    }

    private int dfs543(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = dfs543(root.left);
        int right = dfs543(root.right);
        res543 = Math.max(res543, left + right);
        return Math.max(left, right) + 1;
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

    // 6124. 第一个出现两次的字母
    public char repeatedCharacter(String s) {
        Set<Character> set = new HashSet<>();
        for (char c : s.toCharArray()) {
            if (set.contains(c)) {
                return c;
            }
            set.add(c);
        }
        return '1';

    }

    // 6125. 相等行列对
    public int equalPairs(int[][] grid) {
        int n = grid.length;
        int res = 0;
        for (int i = 0; i < n; ++i) {
            int[] row = grid[i];
            int[] col = new int[n];
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    col[k] = grid[k][j];
                }
                if (Arrays.equals(row, col)) {
                    ++res;
                }
            }
        }
        return res;

    }

    // 6126. 设计食物评分系统
    class FoodRatings {
        public class Bean6126 implements Comparable<Bean6126> {
            int ratings;
            String food;

            public Bean6126(int ratings, String food) {
                this.ratings = ratings;
                this.food = food;
            }

            @Override
            public int compareTo(FoodRatings.Bean6126 o) {
                if (o.ratings == this.ratings) {
                    return this.food.compareTo(o.food);
                }
                return o.ratings - this.ratings;
            }
        }

        private Map<String, Bean6126> foodToBean;
        private Map<String, String> foodToCusine;

        private Map<String, TreeSet<Bean6126>> cuisineToFoodRatings;

        public FoodRatings(String[] foods, String[] cuisines, int[] ratings) {
            int n = foods.length;
            foodToBean = new HashMap<>();
            foodToCusine = new HashMap<>();
            cuisineToFoodRatings = new HashMap<>();
            for (int i = 0; i < n; ++i) {
                Bean6126 bean = new Bean6126(ratings[i], foods[i]);
                foodToBean.put(foods[i], bean);
                foodToCusine.put(foods[i], cuisines[i]);
                cuisineToFoodRatings.computeIfAbsent(cuisines[i], k -> new TreeSet<>()).add(bean);
            }
        }

        public void changeRating(String food, int newRating) {
            String c = foodToCusine.get(food);
            TreeSet<Bean6126> set = cuisineToFoodRatings.get(c);
            set.remove(foodToBean.get(food));
            Bean6126 bean = new Bean6126(newRating, food);
            set.add(bean);
            foodToBean.put(food, bean);
        }

        public String highestRated(String cuisine) {
            TreeSet<Bean6126> set = cuisineToFoodRatings.get(cuisine);
            return set.first().food;
        }
    }

    // 6128. 最好的扑克手牌
    public String bestHand(int[] ranks, char[] suits) {
        Set<Character> set = new HashSet<>();
        for (char c : suits) {
            set.add(c);
        }
        if (set.size() == 1) {
            return "Flush";
        }
        int[] counts = new int[14];
        for (int rank : ranks) {
            ++counts[rank];
        }
        boolean three = false;
        boolean two = false;
        for (int count : counts) {
            if (count >= 3) {
                three = true;
            } else if (count == 2) {
                two = true;
            }
        }
        if (three) {
            return "Three of a Kind";
        }
        if (two) {
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

    // 6130. 设计数字容器系统
    class NumberContainers {
        private Map<Integer, Integer> map;
        private Map<Integer, TreeSet<Integer>> map2;

        public NumberContainers() {
            map = new HashMap<>();
            map2 = new HashMap<>();
        }

        public void change(int index, int number) {
            int original = map.getOrDefault(index, 0);
            if (original != 0) {
                TreeSet<Integer> set = map2.getOrDefault(original, new TreeSet<>());
                set.remove(index);
            }
            map.put(index, number);
            map2.computeIfAbsent(number, k -> new TreeSet<>()).add(index);
        }

        public int find(int number) {
            TreeSet<Integer> set = map2.getOrDefault(number, new TreeSet<>());
            if (set.size() < 1) {
                return -1;
            }
            return set.first();
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

    // 343. 整数拆分 (Integer Break) --dp
    public int integerBreak(int n) {
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
        int left = 0;
        int right = arr.length - 1;
        int times = arr.length - k;
        while (times-- > 0) {
            if (x - arr[left] <= arr[right] - x) {
                --right;
            } else {
                ++left;
            }
        }

        List<Integer> res = new ArrayList<>();
        for (int i = left; i <= right; ++i) {
            res.add(arr[i]);
        }
        return res;

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
    // 剑指 Offer II 092. 翻转字符
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
    // 剑指 Offer II 092. 翻转字符
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

    // 516. 最长回文子序列 (Longest Palindromic Subsequence) --dp
    public int longestPalindromeSubseq(String s) {
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

    // 474. 一和零 (Ones and Zeroes) --0-1背包
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
                        dp[i][j][k] = Math.max(dp[i - 1][j - zeroes][k - ones] + 1, dp[i - 1][j][k]);
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

    // 526. 优美的排列 (Beautiful Arrangement)
    // public int countArrangement(int n) {

    // }

    // 1049. 最后一块石头的重量 II (Last Stone Weight II)
    // public int lastStoneWeightII(int[] stones) {

    // }

    // 979. 在二叉树中分配硬币 (Distribute Coins in Binary Tree)
    // public int distributeCoins(TreeNode root) {

    // }

    // 1026. 节点与其祖先之间的最大差值 (Maximum Difference Between Node and Ancestor)
    // public int maxAncestorDiff(TreeNode root) {

    // }

}
