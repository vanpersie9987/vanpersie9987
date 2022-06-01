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

import javax.swing.text.html.HTMLDocument.RunElement;

public class LeetCode_4 {
    public static void main(String[] args) {

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

    // 110. 平衡二叉树 (Balanced Binary Tree) --dfs
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

    // 473. 火柴拼正方形 (Matchsticks to Square)
    // public boolean makesquare(int[] matchsticks) {

    // }

}
