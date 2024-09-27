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

@SuppressWarnings("unchecked")
public class Leetcode_6 {
    public static void main(String[] args) {
        // String[] strs = { "1.500", "2.500", "3.500" };
        // String s = minimizeError(strs, 9);
        // int res = maxValue(4, 2, 6);
        // houses = [0,2,1,2,0], cost = [[1,10],[10,1],[10,1],[1,10],[5,1]], m = 5, n =
        // 2, target = 3 int[] houses = {}
        // int[] houses = { 0, 2, 1, 2, 0 };
        // int[][] cost = { { 1, 10 }, { 10, 1 }, { 10, 1 }, { 1, 10 }, { 5, 1 } };
        // int m = 5;
        // int n = 2;
        // int target = 3;
        // int res = minCost(houses, cost, m, n, target);
        // int min = minimumCost("i love leetcode", 12);
        // System.out.println(min);
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
        return Math.max(max1 - min2, max2 - min1);

    }

    // 663. 均匀树划分 (Equal Tree Partition) --plus
    private int sum663;
    private boolean res663;

    public boolean checkEqualTree(TreeNode root) {
        dfs663(root);
        dfs_div663(root.left);
        dfs_div663(root.right);
        return res663;
    }

    private void dfs663(TreeNode node) {
        if (node == null) {
            return;
        }
        sum663 += node.val;
        dfs663(node.left);
        dfs663(node.right);
    }

    private int dfs_div663(TreeNode node) {
        if (node == null) {
            return 0;
        }
        int cur = dfs_div663(node.left) + dfs_div663(node.right) + node.val;
        if (cur * 2 == sum663) {
            res663 = true;
        }
        return cur;
    }

    // 1120. 子树的最大平均值 (Maximum Average Subtree) --plus
    private double res1120;

    public double maximumAverageSubtree(TreeNode root) {
        dfs1120(root);
        return res1120;
    }

    private int[] dfs1120(TreeNode node) {
        if (node == null) {
            return new int[] { 0, 0 };
        }
        int[] left = dfs1120(node.left);
        int[] right = dfs1120(node.right);
        int sum = left[0] + right[0] + node.val;
        int count = left[1] + right[1] + 1;
        res1120 = Math.max(res1120, (double) sum / count);
        return new int[] { sum, count };
    }

    // 1181. 前后拼接 (Before and After Puzzle) --plus
    public List<String> beforeAndAfterPuzzles(String[] phrases) {
        Map<String, Set<String>> prefix = new HashMap<>();
        Map<String, Set<String>> suffix = new HashMap<>();
        TreeSet<String> set = new TreeSet<>();
        for (String phrase : phrases) {
            int firstSpace = phrase.indexOf(" ");
            String curPrefix = firstSpace != -1 ? phrase.substring(0, firstSpace) : phrase;
            String curPreRemain = firstSpace != -1 ? phrase.substring(firstSpace + 1) : "";
            for (String s : suffix.getOrDefault(curPrefix, new HashSet<>())) {
                set.add(String.join(" ", new String[] { s, phrase }).trim());
            }

            int lastSpace = phrase.lastIndexOf(" ");
            String curSuffix = lastSpace != -1 ? phrase.substring(lastSpace + 1) : phrase;
            String curSufRemain = lastSpace != -1 ? phrase.substring(0, lastSpace) : "";
            for (String s : prefix.getOrDefault(curSuffix, new HashSet<>())) {
                set.add(String.join(" ", new String[] { phrase, s }).trim());
            }
            prefix.computeIfAbsent(curPrefix, k -> new HashSet<>()).add(curPreRemain);
            suffix.computeIfAbsent(curSuffix, k -> new HashSet<>()).add(curSufRemain);
        }

        return new ArrayList<>(set);

    }

    // 1167. 连接棒材的最低费用 (Minimum Cost to Connect Sticks) --plus
    public int connectSticks(int[] sticks) {
        Queue<Integer> queue = new PriorityQueue<>();
        for (int stick : sticks) {
            queue.offer(stick);
        }
        int res = 0;
        while (queue.size() > 1) {
            int cost = queue.poll() + queue.poll();
            res += cost;
            queue.offer(cost);
        }
        return res;

    }

    // 1182. 与目标颜色间的最短距离 (Shortest Distance to Target Color) --plus
    public List<Integer> shortestDistanceColor(int[] colors, int[][] queries) {
        int n = queries.length;
        List<Integer> res = new ArrayList<>();
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < colors.length; ++i) {
            map.computeIfAbsent(colors[i], k -> new ArrayList<>()).add(i);
        }
        for (int i = 0; i < n; ++i) {
            List<Integer> index = map.get(queries[i][1]);
            if (index == null) {
                res.add(-1);
                continue;
            }
            int min = Integer.MAX_VALUE;
            int target1 = binarySearchFloor1182(index, queries[i][0]);
            if (target1 != -1) {
                min = Math.min(min, queries[i][0] - target1);
            }
            int target2 = binarySearchCeiling1182(index, queries[i][0]);
            if (target2 != -1) {
                min = Math.min(min, target2 - queries[i][0]);
            }
            res.add(min);
        }
        return res;

    }

    private int binarySearchCeiling1182(List<Integer> index, int target) {
        int n = index.size();
        if (target > index.get(n - 1)) {
            return -1;
        }
        if (target <= index.get(0)) {
            return index.get(0);
        }
        int left = 0;
        int right = index.size() - 1;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (index.get(mid) >= target) {
                res = index.get(mid);
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    private int binarySearchFloor1182(List<Integer> index, int target) {
        int n = index.size();
        if (target < index.get(0)) {
            return -1;
        }
        if (target >= index.get(n - 1)) {
            return index.get(n - 1);
        }
        int left = 0;
        int right = index.size() - 1;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (index.get(mid) <= target) {
                res = index.get(mid);
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    // 2445. Number of Nodes With Value One --plus
    public int numberOfNodes(int n, int[] queries) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 1; i <= n; ++i) {
            if (i * 2 <= n) {
                map.computeIfAbsent(i, k -> new ArrayList<>()).add(i * 2);
            }
            if (i * 2 + 1 <= n) {
                map.computeIfAbsent(i, k -> new ArrayList<>()).add(i * 2 + 1);
            }
        }
        int[] count = new int[n + 1];
        for (int query : queries) {
            ++count[query];
        }
        int res = 0;
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(1);
        while (!queue.isEmpty()) {
            int node = queue.poll();
            if (count[node] % 2 == 1) {
                ++res;
            }
            for (int neighbor : map.getOrDefault(node, new ArrayList<>())) {
                count[neighbor] += count[node];
                queue.offer(neighbor);
            }
        }
        return res;

    }

    // 2445. Number of Nodes With Value One --plus
    public int numberOfNodes2(int n, int[] queries) {
        int[] count = new int[n + 1];
        for (int query : queries) {
            ++count[query];
        }
        return dfs2445(1, 0, n, count);
    }

    private int dfs2445(int node, int sum, int n, int[] count) {
        if (node > n) {
            return 0;
        }
        int cur = (sum + count[node]) % 2;
        int res = cur;
        res += dfs2445(node * 2, cur, n, count);
        res += dfs2445(node * 2 + 1, cur, n, count);
        return res;
    }

    // 2488. 统计中位数为 K 的子数组 (Count Subarrays With Median K)
    public int countSubarrays(int[] nums, int k) {
        int n = nums.length;
        int res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int prefix = 0;
        boolean flag = false;
        for (int i = 0; i < n; ++i) {
            prefix += nums[i] > k ? 1 : (nums[i] < k ? -1 : 0);
            if (!flag && nums[i] == k) {
                flag = true;
            }
            if (flag) {
                res += map.getOrDefault(prefix, 0) + map.getOrDefault(prefix - 1, 0);
            } else {
                map.put(prefix, map.getOrDefault(prefix, 0) + 1);
            }
        }
        return res;

    }

    // 490. 迷宫 (The Maze) --plus
    public boolean hasPath(int[][] maze, int[] start, int[] destination) {
        int[][] directions = { { 0, -1 }, { 1, 0 }, { -1, 0 }, { 0, 1 } };
        int m = maze.length;
        int n = maze[0].length;
        boolean[][] visited = new boolean[m][n];
        visited[start[0]][start[1]] = true;
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(start);
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            if (cur[0] == destination[0] && cur[1] == destination[1]) {
                return true;
            }
            for (int[] direction : directions) {
                int nx = cur[0] + direction[0];
                int ny = cur[1] + direction[1];
                while (nx >= 0 && nx < m && ny >= 0 && ny < n && maze[nx][ny] == 0) {
                    nx += direction[0];
                    ny += direction[1];
                }
                nx -= direction[0];
                ny -= direction[1];
                if (!visited[nx][ny]) {
                    visited[nx][ny] = true;
                    queue.offer(new int[] { nx, ny });
                }
            }
        }
        return false;

    }

    // 505. 迷宫 II (The Maze II) --plus
    public int shortestDistance(int[][] maze, int[] start, int[] destination) {
        int[][] directions = { { 0, -1 }, { 1, 0 }, { -1, 0 }, { 0, 1 } };
        int m = maze.length;
        int n = maze[0].length;
        int[][] distance = new int[m][n];
        for (int i = 0; i < m; ++i) {
            Arrays.fill(distance[i], Integer.MAX_VALUE);
        }
        distance[start[0]][start[1]] = 0;
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(start);
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0];
            int y = cur[1];
            int dis = distance[x][y];
            for (int[] direction : directions) {
                int nx = x + direction[0];
                int ny = y + direction[1];
                while (nx >= 0 && nx < m && ny >= 0 && ny < n && maze[nx][ny] == 0) {
                    nx += direction[0];
                    ny += direction[1];
                }
                nx -= direction[0];
                ny -= direction[1];
                int curDis = Math.max(Math.abs(x - nx), Math.abs(y - ny));
                int nDis = curDis + dis;
                if (nDis < distance[nx][ny]) {
                    distance[nx][ny] = nDis;
                    queue.offer(new int[] { nx, ny });
                }
            }
        }
        return distance[destination[0]][destination[1]] == Integer.MAX_VALUE ? -1
                : distance[destination[0]][destination[1]];

    }

    // 499. 迷宫 III (The Maze III) --plus
    public String findShortestWay(int[][] maze, int[] ball, int[] hole) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        char[] d = { 'r', 'l', 'd', 'u' };
        int m = maze.length;
        int n = maze[0].length;
        int[][] distance = new int[m][n];
        List<List<String>> list = new ArrayList<>();
        for (int i = 0; i < m; ++i) {
            List<String> rowList = new ArrayList<>();
            for (int j = 0; j < n; ++j) {
                distance[i][j] = Integer.MAX_VALUE;
                rowList.add("");
            }
            list.add(rowList);
        }
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(ball);
        distance[ball[0]][ball[1]] = 0;
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0];
            int y = cur[1];
            search: for (int i = 0; i < 4; ++i) {
                int nx = x + directions[i][0];
                int ny = y + directions[i][1];
                while (nx >= 0 && nx < m && ny >= 0 && ny < n && maze[nx][ny] == 0) {
                    if (nx == hole[0] && ny == hole[1]) {
                        int steps = Math.max(Math.abs(nx - x), Math.abs(ny - y));
                        if (distance[x][y] + steps < distance[nx][ny]) {
                            distance[nx][ny] = distance[x][y] + steps;
                            String s = list.get(x).get(y) + d[i];
                            list.get(nx).set(ny, s);
                        } else if (distance[x][y] + steps == distance[nx][ny]) {
                            String s = list.get(x).get(y) + d[i];
                            if (s.compareTo(list.get(nx).get(ny)) < 0) {
                                list.get(nx).set(ny, s);
                            }
                        }
                        continue search;
                    }
                    nx += directions[i][0];
                    ny += directions[i][1];
                }
                nx -= directions[i][0];
                ny -= directions[i][1];
                int steps = Math.max(Math.abs(nx - x), Math.abs(ny - y));
                if (distance[x][y] + steps < distance[nx][ny]) {
                    distance[nx][ny] = distance[x][y] + steps;
                    String s = list.get(x).get(y) + d[i];
                    list.get(nx).set(ny, s);
                    queue.offer(new int[] { nx, ny });
                } else if (distance[x][y] + steps == distance[nx][ny]) {
                    String s = list.get(x).get(y) + d[i];
                    if (s.compareTo(list.get(nx).get(ny)) < 0) {
                        list.get(nx).set(ny, s);
                        queue.offer(new int[] { nx, ny });
                    }
                }
            }
        }
        return distance[hole[0]][hole[1]] == Integer.MAX_VALUE ? "impossible" : list.get(hole[0]).get(hole[1]);
    }

    // 528. 按权重随机选择 (Random Pick with Weight)
    // 剑指 Offer II 071. 按权重生成随机数
    class Solution528 {
        private Random random;
        private int[] prefix;
        private int n;

        public Solution528(int[] w) {
            random = new Random();
            n = w.length;
            prefix = new int[n];
            prefix[0] = w[0];
            for (int i = 1; i < n; ++i) {
                prefix[i] = prefix[i - 1] + w[i];
            }

        }

        public int pickIndex() {
            int target = random.nextInt(prefix[n - 1]) + 1;
            return binarySearch528(prefix, target);
        }

        private int binarySearch528(int[] nums, int target) {
            int left = 0;
            int right = nums.length - 1;
            int res = -1;
            while (left <= right) {
                int mid = left + ((right - left) >>> 1);
                if (nums[mid] >= target) {
                    res = mid;
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            return res;
        }
    }

    // 1804. 实现 Trie （前缀树） II (Implement Trie II (Prefix Tree)) --plus
    class Trie1804 {
        class Inner_Trie {
            int prefixCount;
            int wordCount;
            Inner_Trie[] children;

            Inner_Trie() {
                this.children = new Inner_Trie[26];
                this.prefixCount = 0;
                this.wordCount = 0;
            }
        }

        private Inner_Trie trie;

        public Trie1804() {
            trie = new Inner_Trie();
        }

        public void insert(String word) {
            Inner_Trie node = trie;
            for (char c : word.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new Inner_Trie();
                }
                node = node.children[index];
                ++node.prefixCount;
            }
            ++node.wordCount;
        }

        public int countWordsEqualTo(String word) {
            Inner_Trie node = this.trie;
            for (char c : word.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    return 0;
                }
                node = node.children[index];
            }
            return node.wordCount;

        }

        public int countWordsStartingWith(String prefix) {
            Inner_Trie node = this.trie;
            for (char c : prefix.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    return 0;
                }
                node = node.children[index];
            }
            return node.prefixCount;

        }

        public void erase(String word) {
            Inner_Trie node = this.trie;
            for (char c : word.toCharArray()) {
                int index = c - 'a';
                node = node.children[index];
                --node.prefixCount;
            }
            --node.wordCount;
        }
    }

    // 2168. 每个数字的频率都相同的独特子字符串的数量 (Unique Substrings With Equal Digit Frequency)
    // --plus
    public int equalDigitFrequency(String s) {
        int n = s.length();
        int[][] prefix = new int[n + 1][10];
        for (int i = 1; i <= n; ++i) {
            prefix[i] = prefix[i - 1].clone();
            ++prefix[i][s.charAt(i - 1) - '0'];
        }
        Set<String> set = new HashSet<>();
        for (int right = 0; right <= n; ++right) {
            int[] countRight = prefix[right];
            search: for (int left = 0; left < right; ++left) {
                int[] countLeft = prefix[left];
                int diff = -1;
                for (int i = 0; i < 10; ++i) {
                    int d = countRight[i] - countLeft[i];
                    if (d == 0) {
                        continue;
                    }
                    if (diff == -1) {
                        diff = d;
                    } else if (diff != d) {
                        continue search;
                    }
                }
                set.add(s.substring(left, right));
            }
        }
        return set.size();

    }

    // 2214. Minimum Health to Beat Game --plus
    public long minimumHealth(int[] damage, int armor) {
        long right = 1l;
        for (int d : damage) {
            right += d;
        }
        long left = 1l;
        long res = -1;
        while (left <= right) {
            long mid = left + ((right - left) >>> 1);
            if (check2214(mid, damage, armor)) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;

    }

    private boolean check2214(long target, int[] damage, int armor) {
        int n = damage.length;
        int i = 0;
        boolean usedArmor = false;
        int max = 0;
        while (i < n) {
            max = Math.max(max, damage[i]);
            if (target > damage[i]) {
                target -= damage[i];
                ++i;
            } else if (!usedArmor) {
                usedArmor = true;
                int recovery = Math.min(max, armor);
                target += recovery;
                if (target > damage[i]) {
                    target -= damage[i];
                    ++i;
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }
        return true;
    }

    // 2214. Minimum Health to Beat Game --plus
    public long minimumHealth2(int[] damage, int armor) {
        long sum = 0l;
        int max = 0;
        for (int d : damage) {
            sum += d;
            max = Math.max(max, d);
        }
        return sum + 1 - Math.min(max, armor);

    }

    // LCP 56. 信物传送 -- 0-1 bfs
    public int conveyorBelt(String[] matrix, int[] start, int[] end) {
        int m = matrix.length;
        int n = matrix[0].length();
        int[][] steps = new int[m][n];
        for (int i = 0; i < m; ++i) {
            Arrays.fill(steps[i], Integer.MAX_VALUE);
        }
        steps[start[0]][start[1]] = 0;
        int[][] directions = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
        Map<Character, Integer> map = new HashMap<>();
        map.put('^', 0);
        map.put('v', 1);
        map.put('<', 2);
        map.put('>', 3);

        Deque<int[]> deque = new LinkedList<>();
        deque.offer(new int[] { start[0], start[1], 0 });
        while (!deque.isEmpty()) {
            int[] cur = deque.pollFirst();
            int x = cur[0];
            int y = cur[1];
            int signDir = map.get(matrix[x].charAt(y));
            int curStep = cur[2];
            if (x == end[0] && y == end[1]) {
                break;
            }
            if (curStep > steps[x][y]) {
                continue;
            }
            for (int i = 0; i < 4; ++i) {
                int nStep = curStep + (i == signDir ? 0 : 1);
                int nx = x + directions[i][0];
                int ny = y + directions[i][1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && nStep < steps[nx][ny]) {
                    steps[nx][ny] = nStep;
                    if (i == signDir) {
                        deque.offerFirst(new int[] { nx, ny, nStep });
                    } else {
                        deque.offerLast(new int[] { nx, ny, nStep });
                    }

                }
            }
        }
        return steps[end[0]][end[1]];

    }

    // 295. 数据流的中位数 (Find Median from Data Stream)
    // 剑指 Offer 41. 数据流中的中位数
    // 面试题 17.20. 连续中值
    class MedianFinder {
        private Queue<Integer> minQueue;
        private Queue<Integer> maxQueue;

        /** initialize your data structure here. */
        public MedianFinder() {
            minQueue = new PriorityQueue<>(new Comparator<Integer>() {

                @Override
                public int compare(Integer o1, Integer o2) {
                    return o2 - o1;
                }

            });

            maxQueue = new PriorityQueue<>(new Comparator<Integer>() {

                @Override
                public int compare(Integer o1, Integer o2) {
                    return o1 - o2;
                }

            });

        }

        public void addNum(int num) {
            if (minQueue.isEmpty() || num <= minQueue.peek()) {
                minQueue.offer(num);
                if (minQueue.size() - 1 > maxQueue.size()) {
                    maxQueue.offer(minQueue.poll());
                }
            } else {
                maxQueue.offer(num);
                if (maxQueue.size() - 1 == minQueue.size()) {
                    minQueue.offer(maxQueue.poll());
                }
            }
        }

        public double findMedian() {
            int total = minQueue.size() + maxQueue.size();
            if (total % 2 == 0) {
                return ((double) minQueue.peek() + maxQueue.peek()) / 2;
            }
            return minQueue.peek();

        }
    }

    // 2065. 最大化一张图中的路径价值 (Maximum Path Quality of a Graph)
    private List<int[]>[] g2065;
    private boolean[] vis2065;
    private int[] values2065;
    private int res2065;
    private int maxTime2065;

    public int maximalPathQuality(int[] values, int[][] edges, int maxTime) {
        int n = values.length;
        this.g2065 = new ArrayList[n];
        Arrays.setAll(g2065, k -> new ArrayList<>());
        for (int[] e : edges) {
            int a = e[0];
            int b = e[1];
            int t = e[2];
            g2065[a].add(new int[] { b, t });
            g2065[b].add(new int[] { a, t });
        }
        this.maxTime2065 = maxTime;
        this.vis2065 = new boolean[n];
        this.values2065 = values;
        vis2065[0] = true;
        dfs2065(0, values[0], 0);
        return res2065;

    }

    private void dfs2065(int x, int val, int time) {
        if (x == 0) {
            res2065 = Math.max(res2065, val);
        }
        for (int[] nei : g2065[x]) {
            int y = nei[0];
            int t = nei[1];
            if (time + t > maxTime2065) {
                continue;
            }
            if (!vis2065[y]) {
                vis2065[y] = true;
                dfs2065(y, val + values2065[y], time + t);
                vis2065[y] = false;
            } else {
                dfs2065(y, val, time + t);
            }
        }
    }

    // 2440. 创建价值相同的连通块 (Create Components With Same Value)
    private Map<Integer, List<Integer>> graph2440;
    private int target2440;
    private int[] nums2440;

    public int componentValue(int[] nums, int[][] edges) {
        graph2440 = new HashMap<>();
        nums2440 = nums;
        for (int[] edge : edges) {
            graph2440.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(edge[1]);
            graph2440.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(edge[0]);
        }
        int sum = Arrays.stream(nums).sum();
        int max = Arrays.stream(nums).max().getAsInt();
        for (int i = sum / max; i > 0; --i) {
            if (sum % i == 0) {
                target2440 = sum / i;
                if (dfs2440(0, -1) == 0) {
                    return i - 1;
                }
            }
        }
        return -1;

    }

    private int dfs2440(int x, int fa) {
        int sum = nums2440[x];
        for (int neighbor : graph2440.getOrDefault(x, new ArrayList<>())) {
            if (neighbor != fa) {
                int res = dfs2440(neighbor, x);
                if (res < 0) {
                    return -1;
                }
                sum += res;
            }
        }
        if (sum > target2440) {
            return -1;
        }
        return sum < target2440 ? sum : 0;

    }

    // 6261. 数组中字符串的最大值 (Maximum Value of a String in an Array)
    public int maximumValue(String[] strs) {
        int res = 0;
        search: for (String s : strs) {
            int num = 0;
            for (char c : s.toCharArray()) {
                if (Character.isDigit(c)) {
                    num = num * 10 + c - '0';
                } else {
                    res = Math.max(res, s.length());
                    continue search;
                }
            }
            res = Math.max(res, num);
        }
        return res;

    }

    // 6262. 图中最大星和 (Maximum Star Sum of a Graph)
    public int maxStarSum(int[] vals, int[][] edges, int k) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int[] edge : edges) {
            map.computeIfAbsent(edge[0], o -> new ArrayList<>()).add(edge[1]);
            map.computeIfAbsent(edge[1], o -> new ArrayList<>()).add(edge[0]);
        }
        int n = vals.length;
        int res = Integer.MIN_VALUE;
        for (int i = 0; i < n; ++i) {
            int cur = vals[i];
            List<Integer> neighbors = map.getOrDefault(i, new ArrayList<>());
            List<Integer> valss = new ArrayList<>();
            for (int nei : neighbors) {
                valss.add(vals[nei]);
            }
            Collections.sort(valss);
            int j = valss.size() - 1;
            int copyK = k;
            while (j >= 0 && valss.get(j) > 0 && copyK > 0) {
                cur += valss.get(j);
                --j;
                --copyK;
            }
            res = Math.max(res, cur);

        }
        return res;

    }

    // 6263. 青蛙过河 II (Frog Jump II)
    public int maxJump(int[] stones) {
        int n = stones.length;
        int res = stones[1] - stones[0];
        for (int i = 2; i < n; ++i) {
            res = Math.max(res, stones[i] - stones[i - 2]);
        }
        return res;

    }

    // 6257. 删除每行中的最大值
    public int deleteGreatestValue(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int res = 0;
        int nCopy = n;
        while (nCopy-- > 0) {
            int curMax = 0;
            for (int i = 0; i < m; ++i) {
                int max = Arrays.stream(grid[i]).max().getAsInt();
                for (int j = 0; j < n; ++j) {
                    if (max == grid[i][j]) {
                        grid[i][j] = 0;
                        curMax = Math.max(curMax, max);
                        break;
                    }
                }
            }
            res += curMax;
        }
        return res;

    }

    // 6258. 数组中最长的方波
    public int longestSquareStreak(int[] nums) {
        int res = 0;
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        for (int i = 2; i <= 316; ++i) {
            if (set.contains(i)) {
                int cur = 0;
                int base = i;
                set.remove(i);
                while (set.contains(base * base)) {
                    set.remove(base * base);
                    base = base * base;
                    ++cur;
                }
                if (cur != 0) {
                    res = Math.max(res, cur + 1);
                }
            }
        }
        return res == 0 ? -1 : res;

    }

    // 6259. 设计内存分配器
    class Allocator {
        private int n;
        private int[] arr;

        public Allocator(int n) {
            arr = new int[n];
            this.n = n;

        }

        public int allocate(int size, int mID) {
            if (size > n) {
                return -1;
            }
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (arr[i] == 0) {
                    ++count;
                    if (count == size) {
                        Arrays.fill(arr, i - count + 1, i + 1, mID);
                        return i - count + 1;
                    }
                } else {
                    count = 0;
                }
            }
            return -1;

        }

        public int free(int mID) {
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (arr[i] == mID) {
                    arr[i] = 0;
                    ++count;
                }
            }
            return count;
        }
    }

    // 2322. 从树中删除边的最小分数 (Minimum Score After Removals on a Tree)
    private Map<Integer, List<Integer>> graph2322;
    private int[] nums2322;
    private int[] xor2322;
    private int[] in2322;
    private int[] out2322;
    private int clock2322;

    public int minimumScore(int[] nums, int[][] edges) {
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int[] edge : edges) {
            graph.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(edge[1]);
            graph.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(edge[0]);
        }
        int n = nums.length;
        graph2322 = graph;
        nums2322 = nums;
        in2322 = new int[n];
        out2322 = new int[n];
        xor2322 = new int[n];
        dfs2322(0, -1);
        int res = Integer.MAX_VALUE;
        for (int i = 2, x, y, z; i < n; ++i) {
            for (int j = 1; j < i; ++j) {
                if (in2322[i] < in2322[j] && in2322[j] <= out2322[i]) {
                    x = xor2322[j];
                    y = xor2322[i] ^ x;
                    z = xor2322[0] ^ xor2322[i];
                } else if (in2322[j] < in2322[i] && in2322[i] <= out2322[j]) {
                    x = xor2322[i];
                    y = xor2322[j] ^ x;
                    z = xor2322[0] ^ xor2322[j];
                } else {
                    x = xor2322[i];
                    y = xor2322[j];
                    z = xor2322[0] ^ x ^ y;
                }
                res = Math.min(res, Math.max(Math.max(x, y), z) - Math.min(Math.min(x, y), z));
                if (res == 0) {
                    return res;
                }
            }
        }
        return res;

    }

    private void dfs2322(int x, int fa) {
        in2322[x] = ++clock2322;
        xor2322[x] = nums2322[x];
        for (int y : graph2322.getOrDefault(x, new ArrayList<>())) {
            if (y != fa) {
                dfs2322(y, x);
                xor2322[x] ^= xor2322[y];
            }
        }
        out2322[x] = clock2322;
    }

    // 2467. 树上最大得分和路径 (Most Profitable Path in a Tree)
    private int[] time_bob;
    private Map<Integer, List<Integer>> graph2467;
    private int res2467;
    private int[] amount2467;

    public int mostProfitablePath(int[][] edges, int bob, int[] amount) {
        int n = amount.length;
        time_bob = new int[n];
        Arrays.fill(time_bob, n);
        graph2467 = new HashMap<>();
        for (int[] edge : edges) {
            graph2467.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(edge[1]);
            graph2467.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(edge[0]);
        }
        dfs_bob(bob, -1, 0);
        amount2467 = amount;
        res2467 = Integer.MIN_VALUE;
        graph2467.computeIfAbsent(0, k -> new ArrayList<>()).add(-1);
        dfs_alice(0, -1, 0, 0);
        return res2467;

    }

    private boolean dfs_bob(int x, int fa, int t) {
        if (x == 0) {
            time_bob[x] = t;
            return true;
        }
        for (int y : graph2467.getOrDefault(x, new ArrayList<>())) {
            if (y != fa && dfs_bob(y, x, t + 1)) {
                time_bob[x] = t;
                return true;
            }
        }
        return false;
    }

    private void dfs_alice(int x, int fa, int alice_time, int val) {
        if (alice_time < time_bob[x]) {
            val += amount2467[x];
        } else if (alice_time == time_bob[x]) {
            val += amount2467[x] / 2;
        }
        if (graph2467.getOrDefault(x, new ArrayList<>()).size() == 1) {
            res2467 = Math.max(res2467, val);
            return;
        }
        for (int y : graph2467.getOrDefault(x, new ArrayList<>())) {
            if (y != fa) {
                dfs_alice(y, x, alice_time + 1, val);
            }
        }
    }

    // 6260. 矩阵查询可获得的最大分数
    public int[] maxPoints(int[][] grid, int[] queries) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { -1, 0 }, { 1, 0 } };
        int m = grid.length;
        int n = grid[0].length;
        int[][] arr = new int[m * n][3];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                arr[i * n + j] = new int[] { grid[i][j], i, j };
            }
        }
        Arrays.sort(arr, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }

        });
        int k = queries.length;
        int[] res = new int[k];
        Integer[] ids = IntStream.range(0, k).boxed().toArray(Integer[]::new);
        Arrays.sort(ids, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return queries[o1] - queries[o2];
            }

        });
        Union6260 union = new Union6260(m * n);

        int j = 0;
        for (int i : ids) {
            int q = queries[i];
            while (j < m * n && arr[j][0] < q) {
                int x = arr[j][1];
                int y = arr[j][2];
                for (int[] direction : directions) {
                    int nx = x + direction[0];
                    int ny = y + direction[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] < q) {
                        union.union(x * n + y, nx * n + ny);
                    }
                }
                ++j;
            }
            if (grid[0][0] < q) {
                res[i] = union.getSize(0);
            }
        }

        return res;

    }

    class Union6260 {
        private int[] parent;
        private int[] rank;
        private int[] size;

        Union6260(int n) {
            parent = new int[n];
            rank = new int[n];
            size = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
                rank[i] = 1;
                size[i] = 1;
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

        public int getSize(int p) {
            int root = getRoot(p);
            return size[root];
        }

    }

    // 6260. 矩阵查询可获得的最大分数
    public int[] maxPoints2(int[][] grid, int[] queries) {
        int m = grid.length;
        int n = grid[0].length;
        Queue<int[]> queue = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }

        });

        int k = queries.length;
        Integer[] ids = IntStream.range(0, k).boxed().toArray(Integer[]::new);
        Arrays.sort(ids, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return queries[o1] - queries[o2];
            }

        });
        int[] res = new int[k];
        int[][] directions = { { 0, 1 }, { 0, -1 }, { -1, 0 }, { 1, 0 } };
        queue.offer(new int[] { grid[0][0], 0, 0 });
        grid[0][0] = 0;
        int count = 0;
        for (int i : ids) {
            int q = queries[i];
            while (!queue.isEmpty() && queue.peek()[0] < q) {
                ++count;
                int[] cur = queue.poll();
                int x = cur[1];
                int y = cur[2];
                for (int[] direction : directions) {
                    int nx = x + direction[0];
                    int ny = y + direction[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] != 0) {
                        queue.offer(new int[] { grid[nx][ny], nx, ny });
                        grid[nx][ny] = 0;
                    }
                }
            }
            res[i] = count;
        }
        return res;

    }

    // 1697. 检查边长度限制的路径是否存在 (Checking Existence of Edge Length Limited Paths)
    public boolean[] distanceLimitedPathsExist(int n, int[][] edgeList, int[][] queries) {
        Union1697 union = new Union1697(n);
        Arrays.sort(edgeList, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[2] - o2[2];
            }

        });
        int k = queries.length;
        boolean[] res = new boolean[k];
        Integer[] ids = IntStream.range(0, k).boxed().toArray(Integer[]::new);
        Arrays.sort(ids, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return queries[o1][2] - queries[o2][2];
            }

        });
        int j = 0;
        for (int i : ids) {
            int limit = queries[i][2];
            while (j < edgeList.length && edgeList[j][2] < limit) {
                union.union(edgeList[j][0], edgeList[j][1]);
                ++j;
            }
            res[i] = union.isConnected(queries[i][0], queries[i][1]);
        }
        return res;

    }

    class Union1697 {
        private int[] parent;
        private int[] rank;

        public Union1697(int n) {
            parent = new int[n];
            rank = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
                rank[i] = 1;
            }
        }

        public int getRoot(int p) {
            if (p == parent[p]) {
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

    // 1627. 带阈值的图连通性 (Graph Connectivity With Threshold)
    public List<Boolean> areConnected(int n, int threshold, int[][] queries) {
        Union1627 union = new Union1627(n + 1);
        for (int i = threshold + 1; i <= n; i++) {
            for (int j = i; j <= n; j += i) {
                union.union(i, j);
            }
        }
        List<Boolean> res = new ArrayList<>();
        for (int[] query : queries) {
            res.add(union.isConnected(query[0], query[1]));
        }
        return res;

    }

    class Union1627 {
        private int[] parent;
        private int[] rank;

        public Union1627(int n) {
            parent = new int[n];
            rank = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
                rank[i] = 1;
            }
        }

        public int getRoot(int p) {
            if (p == parent[p]) {
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

    // 2203. 得到要求路径的最小带权子图 (Minimum Weighted Subgraph With the Required Paths)
    public long minimumWeight(int n, int[][] edges, int src1, int src2, int dest) {
        Map<Integer, List<Pair2203>> graph = new HashMap<>();
        Map<Integer, List<Pair2203>> rGraph = new HashMap<>();
        for (int[] edge : edges) {
            graph.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(new Pair2203(edge[1], edge[2]));
            rGraph.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(new Pair2203(edge[0], edge[2]));
        }
        long[] d1 = dijkstra2203(graph, src1, n);
        long[] d2 = dijkstra2203(graph, src2, n);
        long[] d3 = dijkstra2203(rGraph, dest, n);
        long res = Long.MAX_VALUE / 3;
        for (int i = 0; i < n; ++i) {
            res = Math.min(res, d1[i] + d2[i] + d3[i]);
        }
        return res == Long.MAX_VALUE / 3 ? -1 : res;

    }

    private long[] dijkstra2203(Map<Integer, List<Pair2203>> graph, int start, int n) {
        long[] distance = new long[n];
        Arrays.fill(distance, Long.MAX_VALUE / 3);
        distance[start] = 0l;
        Queue<Pair2203> queue = new PriorityQueue<>(new Comparator<Pair2203>() {

            @Override
            public int compare(Pair2203 o1, Pair2203 o2) {
                return Long.valueOf(o1.dis).compareTo(Long.valueOf(o2.dis));
            }

        });

        queue.offer(new Pair2203(start, 0L));
        while (!queue.isEmpty()) {
            Pair2203 p = queue.poll();
            int x = p.n;
            long wt = p.dis;
            if (wt > distance[x]) {
                continue;
            }
            for (Pair2203 pair : graph.getOrDefault(x, new ArrayList<>())) {
                long nDis = wt + pair.dis;
                if (nDis < distance[pair.n]) {
                    distance[pair.n] = nDis;
                    queue.offer(new Pair2203(pair.n, nDis));
                }
            }
        }
        return distance;
    }

    class Pair2203 {
        int n;
        long dis;

        Pair2203(int n, long dis) {
            this.n = n;
            this.dis = dis;
        }
    }

    // 2246. 相邻字符不同的最长路径 (Longest Path With Different Adjacent Characters)
    private int res2246;
    private List<Integer>[] g2246;
    private String s2246;

    public int longestPath(int[] parent, String s) {
        int n = s.length();
        this.g2246 = new ArrayList[n];
        this.s2246 = s;
        Arrays.setAll(g2246, k -> new ArrayList<>());
        for (int i = 1; i < n; ++i) {
            g2246[parent[i]].add(i);
        }
        dfs2246(0);
        return res2246;

    }

    private int dfs2246(int x) {
        int mx = 0;
        int pre = 0;
        for (int y : g2246[x]) {
            int cur = dfs2246(y);
            if (s2246.charAt(x) != s2246.charAt(y)) {
                mx = Math.max(mx, cur + pre);
                pre = Math.max(pre, cur);
            }
        }
        res2246 = Math.max(res2246, mx + 1);
        return pre + 1;
    }

    // 687. 最长同值路径 ( Longest Univalue Path)
    private int res687;

    public int longestUnivaluePath(TreeNode root) {
        dfs687(root);
        return res687;

    }

    private int dfs687(TreeNode root) {
        if (root == null) {
            return 0;
        }

        int left = dfs687(root.left);
        int right = dfs687(root.right);
        left = root.left != null && root.left.val == root.val ? left + 1 : 0;
        right = root.right != null && root.right.val == root.val ? right + 1 : 0;
        res687 = Math.max(res687, left + right);
        return Math.max(left, right);
    }

    // 1080. 根到叶路径上的不足节点 (Insufficient Nodes in Root to Leaf Paths)
    public TreeNode sufficientSubset(TreeNode root, int limit) {
        TreeNode dummy = new TreeNode(0);
        dummy.left = root;
        dfs1080(root, dummy, 0, limit);
        return dummy.left;

    }

    private boolean dfs1080(TreeNode node, TreeNode fa, int sum, int limit) {
        if (node == null) {
            return true;
        }
        if (node.left == null && node.right == null) {
            if (sum + node.val < limit) {
                if (fa.left == node) {
                    fa.left = null;
                } else {
                    fa.right = null;
                }
                return true;
            }
            return false;
        }
        boolean left = dfs1080(node.left, node, sum + node.val, limit);
        boolean right = dfs1080(node.right, node, sum + node.val, limit);
        if (left && right) {
            if (fa.left == node) {
                fa.left = null;
            } else {
                fa.right = null;
            }
            return true;
        }
        return false;
    }

    // 1026. 节点与其祖先之间的最大差值 (Maximum Difference Between Node and Ancestor)
    private int res1026;

    public int maxAncestorDiff(TreeNode root) {
        dfs1026(root);
        return res1026;
    }

    private int[] dfs1026(TreeNode root) {
        int min = root.val;
        int max = root.val;
        if (root.left != null) {
            int[] left = dfs1026(root.left);
            min = Math.min(min, left[0]);
            max = Math.max(max, left[1]);
        }
        if (root.right != null) {
            int[] right = dfs1026(root.right);
            min = Math.min(min, right[0]);
            max = Math.max(max, right[1]);
        }
        res1026 = Math.max(res1026, Math.abs(root.val - min));
        res1026 = Math.max(res1026, Math.abs(root.val - max));
        return new int[] { min, max };
    }

    // 2049. 统计最高分的节点数目 (Count Nodes With the Highest Score)
    private int count2049;
    private long maxScore2049;
    private List<Integer>[] g2049;
    private int n2049;

    public int countHighestScoreNodes(int[] parents) {
        this.n2049 = parents.length;
        this.g2049 = new ArrayList[n2049];
        Arrays.setAll(g2049, k -> new ArrayList<>());
        for (int i = 1; i < n2049; ++i) {
            g2049[parents[i]].add(i);
        }
        dfs2049(0);
        return count2049;
    }

    private int dfs2049(int x) {
        long cur = 1L;
        int s = 0;
        for (int y : g2049[x]) {
            int cnt = dfs2049(y);
            cur = cur * cnt;
            s += cnt;
        }
        if (s + 1 != n2049) {
            cur = cur * (n2049 - s - 1);
        }
        if (cur > maxScore2049) {
            maxScore2049 = cur;
            count2049 = 1;
        } else if (cur == maxScore2049) {
            ++count2049;
        }
        return s + 1;
    }

    // 2096. 从二叉树一个节点到另一个节点每一步的方向 (Step-By-Step Directions From a Binary Tree Node
    // to Another)
    private StringBuilder builder2096;

    public String getDirections(TreeNode root, int startValue, int destValue) {
        TreeNode lca = lca2096(root, startValue, destValue);
        builder2096 = new StringBuilder();
        dfs2096(lca, startValue, true);
        StringBuilder res = new StringBuilder(builder2096);
        builder2096.setLength(0);
        dfs2096(lca, destValue, false);
        builder2096.reverse();
        res.append(builder2096);
        return res.toString();

    }

    private boolean dfs2096(TreeNode root, int val, boolean isStart) {
        if (root == null) {
            return false;
        }
        if (root.val == val) {
            return true;
        }
        if (dfs2096(root.left, val, isStart)) {
            builder2096.append(isStart ? 'U' : 'L');
            return true;
        }
        if (dfs2096(root.right, val, isStart)) {
            builder2096.append(isStart ? 'U' : 'R');
            return true;
        }
        return false;
    }

    private TreeNode lca2096(TreeNode root, int startValue, int destValue) {
        if (root == null || root.val == startValue || root.val == destValue) {
            return root;
        }
        TreeNode left = lca2096(root.left, startValue, destValue);
        TreeNode right = lca2096(root.right, startValue, destValue);
        if (left != null && right != null) {
            return root;
        }
        return left != null ? left : right;
    }

    // 1802. 有界数组中指定下标处的最大值 (Maximum Value at a Given Index in a Bounded Array)
    public int maxValue(int n, int index, int maxSum) {
        int left = 1;
        int right = maxSum;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (check1802(n, index, mid) <= (long) maxSum) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;

    }

    private long check1802(int n, int index, long val) {
        long res = 0l;
        if (index + 1 >= val) {
            res += (1 + val) * val / 2 + (index - val) + 1;
        } else {
            res += ((val - index) + val) * (index + 1) / 2;
        }

        if (index + val <= n) {
            res += (1 + val) * val / 2 + n - (index + val);
        } else {
            res += ((val - (n - index) + 1) + val) * (n - index) / 2;
        }
        return res - val;

    }

    // 1849. 将字符串拆分为递减的连续值 (Splitting a String Into Descending Consecutive Values)
    public boolean splitString(String s) {
        StringBuilder builder = new StringBuilder(s);
        while (builder.length() > 0 && builder.charAt(0) == '0') {
            builder.deleteCharAt(0);
        }
        s = builder.toString();
        int n = s.length();
        search: for (int i = 1; i < n; ++i) {
            String ss = s.substring(0, i);
            if (ss.length() > 10) {
                return false;
            }
            long num = Long.parseLong(ss.toString());
            int j = i;
            search1: while (true) {
                if (num == 1l) {
                    while (j < n) {
                        if (s.charAt(j) != '0') {
                            continue search;
                        }
                        ++j;
                    }
                    return true;
                } else {
                    long cur = 0l;
                    while (j < n && cur < num) {
                        cur = cur * 10 + s.charAt(j) - '0';
                        if (cur + 1 == num) {
                            break;
                        }
                        ++j;
                    }
                    if (cur + 1 == num) {
                        ++j;
                        if (j == n) {
                            return true;
                        }
                        --num;
                        continue search1;
                    }
                    continue search;
                }
            }
        }
        return false;

    }

    // 6265. 统计相似字符串对的数目
    public int similarPairs(String[] words) {
        int n = words.length;
        Map<Integer, Integer> counts = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            int bit = 0;
            for (char c : words[i].toCharArray()) {
                bit |= (1 << (c - 'a'));
            }
            counts.put(bit, counts.getOrDefault(bit, 0) + 1);
        }
        int res = 0;
        for (int count : counts.values()) {
            res += count * (count - 1) / 2;
        }
        return res;

    }

    // 6266. 使用质因数之和替换后可以取到的最小值
    public int smallestValue(int n) {
        while (true) {
            int sum = getFactorsSum(n);
            if (sum == n) {
                return n;
            }
            n = sum;
        }
    }

    private int getFactorsSum(int n) {
        int sum = 0;
        for (int i = 2; i * i <= n; ++i) {
            while (n % i == 0) {
                sum += i;
                n /= i;
            }
        }
        if (n != 1) {
            sum += n;
        }
        return sum;
    }

    // 6268. 查询树中环的长度
    public int[] cycleLengthQueries(int n, int[][] queries) {
        int[] res = new int[queries.length];
        for (int i = 0; i < queries.length; ++i) {
            int u = queries[i][0];
            int v = queries[i][1];
            int count = 1;
            while (u != v) {
                if (u < v) {
                    int temp = u;
                    u = v;
                    v = temp;
                }
                u >>>= 1;
                ++count;
            }
            res[i] = count;
        }
        return res;
    }

    // 6267. 添加边使所有节点度数都为偶数
    public boolean isPossible(int n, List<List<Integer>> edges) {
        Map<Integer, Set<Integer>> map = new HashMap<>();
        int[] degrees = new int[n + 1];
        for (List<Integer> edge : edges) {
            map.computeIfAbsent(edge.get(0), k -> new HashSet<>()).add(edge.get(1));
            map.computeIfAbsent(edge.get(1), k -> new HashSet<>()).add(edge.get(0));
            ++degrees[edge.get(0)];
            ++degrees[edge.get(1)];
        }
        List<Integer> odd = new ArrayList<>();
        for (int i = 1; i <= n; ++i) {
            if (degrees[i] > 0 && degrees[i] % 2 == 1) {
                odd.add(i);
            }
        }
        if (odd.size() == 0) {
            return true;
        }
        if (odd.size() == 2) {
            int u = odd.get(0);
            int v = odd.get(1);
            if (!map.getOrDefault(u, new HashSet<>()).contains(v)) {
                return true;
            }
            for (int i = 1; i <= n; ++i) {
                if (i != u && i != v) {
                    Set<Integer> set = map.getOrDefault(i, new HashSet<>());
                    if (!set.contains(u) && !set.contains(v)) {
                        return true;
                    }
                }
            }
        }
        if (odd.size() == 4) {
            int a = odd.get(0);
            int b = odd.get(1);
            int c = odd.get(2);
            int d = odd.get(3);
            if (!map.getOrDefault(a, new HashSet<>()).contains(b)
                    && !map.getOrDefault(c, new HashSet<>()).contains(d)) {
                return true;
            }
            if (!map.getOrDefault(a, new HashSet<>()).contains(c)
                    && !map.getOrDefault(b, new HashSet<>()).contains(d)) {
                return true;
            }
            if (!map.getOrDefault(a, new HashSet<>()).contains(d)
                    && !map.getOrDefault(b, new HashSet<>()).contains(c)) {
                return true;
            }
        }
        return false;
    }

    // 923. 三数之和的多种可能 (3Sum With Multiplicity)
    public int threeSumMulti(int[] arr, int target) {
        int[] counts = new int[101];
        for (int a : arr) {
            ++counts[a];
        }
        long res = 0l;
        final int mod = (int) (1e9 + 7);

        // x != y != z
        for (int x = 0; x <= 100; ++x) {
            for (int y = x + 1; y <= 100; ++y) {
                int z = target - x - y;
                if (y < z && z <= 100) {
                    res = (res + (long) counts[x] * counts[y] * counts[z]) % mod;
                }
            }
        }
        // x == y != z
        for (int x = 0; x <= 100; ++x) {
            int z = target - 2 * x;
            if (x < z && z <= 100) {
                res = (res + (long) counts[x] * (counts[x] - 1) / 2 * counts[z]) % mod;
            }
        }
        // x != y == z
        for (int x = 0; x <= 100; ++x) {
            if (target % 2 == x % 2) {
                int y = (target - x) / 2;
                if (x < y && y <= 100) {
                    res = (res + (long) counts[x] * counts[y] * (counts[y] - 1) / 2) % mod;
                }
            }
        }

        // x == y == z
        for (int x = 0; x <= 100; ++x) {
            if (x * 3 == target) {
                res = (res + (long) counts[x] * (counts[x] - 1) * (counts[x] - 2) / 6) % mod;
            }
        }
        return (int) (res % mod);

    }

    // 1600. 王位继承顺序 (Throne Inheritance)
    class ThroneInheritance {
        private String kingName;
        private Map<String, List<String>> inheritanceTree;
        private Set<String> setDeath;
        private List<String> res;

        public ThroneInheritance(String kingName) {
            this.kingName = kingName;
            this.inheritanceTree = new HashMap<>();
            this.setDeath = new HashSet<>();
            inheritanceTree.put(kingName, new ArrayList<>());
        }

        public void birth(String parentName, String childName) {
            inheritanceTree.computeIfAbsent(parentName, k -> new ArrayList<>()).add(childName);
        }

        public void death(String name) {
            setDeath.add(name);
        }

        public List<String> getInheritanceOrder() {
            res = new ArrayList<>();
            dfs1600(kingName);
            return res;
        }

        private void dfs1600(String x) {
            if (!setDeath.contains(x)) {
                res.add(x);
            }
            for (String y : inheritanceTree.getOrDefault(x, new ArrayList<>())) {
                dfs1600(y);
            }
        }

    }

    // 1797. 设计一个验证系统 (Design Authentication Manager)
    class AuthenticationManager {

        private TreeMap<Integer, String> treeMap;
        private Map<String, Integer> map;
        private int timeToLive;

        public AuthenticationManager(int timeToLive) {
            treeMap = new TreeMap<>();
            map = new HashMap<>();
            this.timeToLive = timeToLive;
        }

        public void generate(String tokenId, int currentTime) {
            map.put(tokenId, currentTime);
            treeMap.put(currentTime, tokenId);
        }

        public void renew(String tokenId, int currentTime) {
            if (!map.containsKey(tokenId) || currentTime - map.get(tokenId) >= timeToLive) {
                return;
            }
            treeMap.remove(map.get(tokenId));
            map.put(tokenId, currentTime);
            treeMap.put(currentTime, tokenId);
        }

        public int countUnexpiredTokens(int currentTime) {
            while (!treeMap.isEmpty() && currentTime - treeMap.firstKey() >= timeToLive) {
                map.remove(treeMap.pollFirstEntry().getValue());
            }
            return map.size();
        }
    }

    // 2102. 序列顺序查询 (Sequentially Ordinal Rank Tracker)
    class SORTracker {
        class Pair2102 {
            String name;
            int score;

            Pair2102(String name, int score) {
                this.name = name;
                this.score = score;
            }

        }

        private Queue<Pair2102> min;
        private Queue<Pair2102> max;

        public SORTracker() {
            min = new PriorityQueue<>(new Comparator<Pair2102>() {

                @Override
                public int compare(Pair2102 o1, Pair2102 o2) {
                    return o1.score == o2.score ? o2.name.compareTo(o1.name) : o1.score - o2.score;
                }

            });

            max = new PriorityQueue<>(new Comparator<Pair2102>() {

                @Override
                public int compare(Pair2102 o1, Pair2102 o2) {
                    return o1.score == o2.score ? o1.name.compareTo(o2.name) : o2.score - o1.score;
                }

            });

        }

        public void add(String name, int score) {
            min.offer(new Pair2102(name, score));
            max.offer(min.poll());
        }

        public String get() {
            String res = max.peek().name;
            min.offer(max.poll());
            return res;
        }
    }

    // 1835. 所有数对按位与结果的异或和 (Find XOR Sum of All Pairs Bitwise AND)
    // (a&b)^(a&c) == a&(b^c)
    public int getXORSum(int[] arr1, int[] arr2) {
        int xor1 = 0;
        int xor2 = 0;
        for (int a : arr1) {
            xor1 ^= a;
        }
        for (int a : arr2) {
            xor2 ^= a;
        }
        return xor1 & xor2;

    }

    // 333. 最大 BST 子树 (Largest BST Subtree) --plus
    private int res333;

    public int largestBSTSubtree(TreeNode root) {
        dfs333(root);
        return res333;

    }

    private int[] dfs333(TreeNode node) {
        if (node == null) {
            return new int[] { Integer.MAX_VALUE, Integer.MIN_VALUE, 0 };
        }
        int[] left = dfs333(node.left);
        int[] right = dfs333(node.right);
        if (left[1] >= node.val || right[0] <= node.val) {
            return new int[] { Integer.MIN_VALUE, Integer.MAX_VALUE, 0 };
        }
        int count = left[2] + right[2] + 1;
        res333 = Math.max(res333, count);
        return new int[] { Math.min(node.val, left[0]), Math.max(node.val, right[1]), count };
    }

    // Definition for a Node.
    class Node {
        public int val;
        public List<Node> children;

        public Node() {
            children = new ArrayList<Node>();
        }

        public Node(int _val) {
            val = _val;
            children = new ArrayList<Node>();
        }

        public Node(int _val, ArrayList<Node> _children) {
            val = _val;
            children = _children;
        }
    };

    // 1522. N 叉树的直径 (Diameter of N-Ary Tree) --plus
    private int res1522;

    public int diameter(Node root) {
        dfs1522(root);
        return res1522;

    }

    private int dfs1522(Node node) {
        if (node == null) {
            return 0;
        }
        int max1 = 0;
        int max2 = 0;
        for (Node y : node.children) {
            int cur = dfs1522(y);
            if (cur >= max1) {
                max2 = max1;
                max1 = cur;
            } else if (cur >= max2) {
                max2 = cur;
            }
        }
        res1522 = Math.max(res1522, max1 + max2);
        return max1 + 1;
    }

    // 250. 统计同值子树 (Count Univalue Subtrees)
    private int res250;

    public int countUnivalSubtrees(TreeNode root) {
        dfs250(root);
        return res250;

    }

    private Integer dfs250(TreeNode node) {
        if (node == null) {
            return null;
        }
        Integer left = dfs250(node.left);
        Integer right = dfs250(node.right);
        if (left != null && left != node.val) {
            return 1001;
        }
        if (right != null && right != node.val) {
            return 1001;
        }
        ++res250;
        return node.val;
    }

    // 1214. 查找两棵二叉搜索树之和 (Two Sum BSTs) --plus
    public boolean twoSumBSTs(TreeNode root1, TreeNode root2, int target) {
        List<Integer> list1 = new ArrayList<>();
        List<Integer> list2 = new ArrayList<>();
        dfs1214(root1, list1);
        dfs1214(root2, list2);
        int n1 = list1.size();
        int n2 = list2.size();
        int i = 0;
        int j = n2 - 1;
        while (i < n1 && j >= 0) {
            int cur = list1.get(i) + list2.get(j);
            if (cur == target) {
                return true;
            }
            if (cur < target) {
                ++i;
            } else {
                --j;
            }
        }
        return false;

    }

    private void dfs1214(TreeNode root, List<Integer> list) {
        if (root == null) {
            return;
        }
        dfs1214(root.left, list);
        list.add(root.val);
        dfs1214(root.right, list);
    }

    // 1245. 树的直径 (Tree Diameter) --plus
    private int res1245;

    public int treeDiameter(int[][] edges) {
        Map<Integer, List<Integer>> tree = new HashMap<>();
        for (int[] edge : edges) {
            tree.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(edge[1]);
            tree.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(edge[0]);
        }
        dfs1245(0, -1, tree);
        return res1245;
    }

    private int dfs1245(int x, int fa, Map<Integer, List<Integer>> tree) {
        int max1 = 0;
        int max2 = 0;
        for (int y : tree.getOrDefault(x, new ArrayList<>())) {
            if (y != fa) {
                int cur = dfs1245(y, x, tree);
                if (cur >= max1) {
                    max2 = max1;
                    max1 = cur;
                } else if (cur >= max2) {
                    max2 = cur;
                }
            }
        }
        res1245 = Math.max(res1245, max1 + max2);
        return max1 + 1;
    }

    // 1273. 删除树节点 (Delete Tree Nodes) --plus
    private int[] value1273;
    private Map<Integer, List<Integer>> tree1273;

    public int deleteTreeNodes(int nodes, int[] parent, int[] value) {
        value1273 = value;
        tree1273 = new HashMap<>();
        for (int i = 0; i < parent.length; ++i) {
            tree1273.computeIfAbsent(i, k -> new ArrayList<>()).add(parent[i]);
            tree1273.computeIfAbsent(parent[i], k -> new ArrayList<>()).add(i);
        }
        return dfs1273(0, -1)[1];

    }

    private int[] dfs1273(int x, int fa) {
        int sum = value1273[x];
        int count = 1;
        for (int y : tree1273.getOrDefault(x, new ArrayList<>())) {
            if (y != fa) {
                int[] cur = dfs1273(y, x);
                sum += cur[0];
                count += cur[1];
            }
        }
        if (sum == 0) {
            count = 0;
        }
        return new int[] { sum, count };
    }

    // 1273. 删除树节点 (Delete Tree Nodes) --plus 拓扑排序
    public int deleteTreeNodes2(int nodes, int[] parent, int[] value) {
        int[] count = new int[nodes];
        Arrays.fill(count, 1);
        int[] degrees = new int[nodes];
        for (int i = 1; i < parent.length; ++i) {
            ++degrees[parent[i]];
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < nodes; ++i) {
            if (degrees[i] == 0) {
                queue.offer(i);
            }
        }
        while (!queue.isEmpty()) {
            int x = queue.poll();
            if (value[x] == 0) {
                count[x] = 0;
            }
            int y = parent[x];
            if (y == -1) {
                continue;
            }
            --degrees[y];
            value[y] += value[x];
            count[y] += count[x];
            if (degrees[y] == 0) {
                queue.offer(y);
            }

        }
        return count[0];

    }

    // 1059. 从始点到终点的所有路径 (All Paths from Source Lead to Destination) --plus
    public boolean leadsToDestination(int n, int[][] edges, int source, int destination) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        int[] inDegrees = new int[n];
        for (int[] edge : edges) {
            map.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(edge[0]);
            ++inDegrees[edge[0]];
        }
        if (inDegrees[destination] != 0) {
            return false;
        }
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(destination);
        while (!queue.isEmpty()) {
            int x = queue.poll();
            if (x == source) {
                return true;
            }
            for (int y : map.getOrDefault(x, new ArrayList<>())) {
                if (--inDegrees[y] == 0) {
                    queue.offer(y);
                }
            }
        }
        return false;

    }

    // 1852. 每个子数组的数字种类数 (Distinct Numbers in Each Subarray) --plus
    public int[] distinctNumbers(int[] nums, int k) {
        int n = nums.length;
        int max = 0;
        for (int num : nums) {
            max = Math.max(max, num);
        }
        int[] counts = new int[max + 1];
        int[] res = new int[n - k + 1];
        int kinds = 0;
        for (int i = 0; i < k; ++i) {
            if (counts[nums[i]]++ == 0) {
                ++kinds;
            }
        }
        res[0] = kinds;
        for (int i = k; i < n; ++i) {
            if (counts[nums[i]]++ == 0) {
                ++kinds;
            }
            if (--counts[nums[i - k]] == 0) {
                --kinds;
            }
            res[i - k + 1] = kinds;
        }
        return res;

    }

    // 1215. 步进数 (Stepping Numbers) --plus
    public List<Integer> countSteppingNumbers(int low, int high) {
        Queue<Long> queue = new LinkedList<>();
        List<Integer> res = new ArrayList<>();
        if (low == 0) {
            res.add(0);
        }
        for (long i = 1; i <= 9; ++i) {
            queue.offer(i);
        }

        while (!queue.isEmpty()) {
            long cur = queue.poll();
            int lastDigit = (int) (cur % 10);
            if (cur > high) {
                return res;
            }
            if (cur >= low) {
                res.add((int) cur);
            }
            if (lastDigit != 0) {
                queue.offer(cur * 10 + lastDigit - 1);
            }
            if (lastDigit != 9) {
                queue.offer(cur * 10 + lastDigit + 1);
            }
        }
        return res;

    }

    // 1215. 步进数 (Stepping Numbers) --plus
    public List<Integer> countSteppingNumbers2(int low, int high) {
        List<Integer> list = new ArrayList<>();
        if (low == 0) {
            list.add(low);
        }
        for (long i = 1; i <= 9; ++i) {
            dfs1215(i, high, list);
        }
        List<Integer> res = new ArrayList<>();
        for (int num : list) {
            if (num >= low && num <= high) {
                res.add(num);
            }
        }
        Collections.sort(res);
        return res;

    }

    private void dfs1215(long num, int high, List<Integer> list) {
        if (num > high) {
            return;
        }
        list.add((int) num);
        int lastDigit = (int) (num % 10);

        if (lastDigit != 0) {
            dfs1215(num * 10 + lastDigit - 1, high, list);
        }
        if (lastDigit != 9) {
            dfs1215(num * 10 + lastDigit + 1, high, list);
        }

    }

    // 2511. 最多可以摧毁的敌人城堡数目 (Maximum Enemy Forts That Can Be Captured)
    public int captureForts(int[] forts) {
        int n = forts.length;
        int res = 0;
        int i = 0;
        while (i < n) {
            while (i < n && forts[i] == 0) {
                ++i;
            }
            int j = i;
            ++i;
            while (i < n && forts[i] == 0) {
                ++i;
            }
            if (i < n && forts[i] + forts[j] == 0) {
                res = Math.max(res, i - j - 1);
            }
        }
        return res;

    }

    // 6274. 奖励最顶尖的 K 名学生
    public List<Integer> topStudents(String[] positive_feedback, String[] negative_feedback, String[] report,
            int[] student_id, int k) {
        Set<String> postive = new HashSet<>();
        for (String p : positive_feedback) {
            postive.add(p);
        }
        Set<String> negative = new HashSet<>();
        for (String n : negative_feedback) {
            negative.add(n);
        }
        List<int[]> list = new ArrayList<>();
        int n = report.length;
        for (int i = 0; i < n; ++i) {
            String[] split = report[i].split(" ");
            int score = 0;
            for (String s : split) {
                if (postive.contains(s)) {
                    score += 3;
                } else if (negative.contains(s)) {
                    score -= 1;
                }
            }
            list.add(new int[] { student_id[i], score });
        }
        Collections.sort(list, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[1] == o2[1] ? o1[0] - o2[0] : o2[1] - o1[1];
            }

        });
        List<Integer> res = new ArrayList<>();

        for (int i = 0; i < k; ++i) {
            res.add(list.get(i)[0]);
        }
        return res;
    }

    // 6269. 到目标字符串的最短距离
    public int closetTarget(String[] words, String target, int startIndex) {
        int n = words.length;
        int res = n;
        for (int i = 0; i < n; ++i) {
            if (words[i].equals(target)) {
                res = Math.min(res, Math.abs(startIndex - i));
                res = Math.min(res, n - Math.abs(startIndex - i));
            }
        }
        return res == n ? -1 : res;

    }

    // 2516. 每种字符至少取 K 个 (Take K of Each Character From Left and Right)
    public int takeCharacters(String s, int k) {
        int[] cnt = new int[3];
        for (char c : s.toCharArray()) {
            ++cnt[c - 'a'];
        }
        for (int c : cnt) {
            if (c < k) {
                return -1;
            }
        }
        int left = 0;
        int right = s.length() - 3 * k;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (check2516(mid, cnt, s, k)) {
                res = s.length() - mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;

    }

    private boolean check2516(int w, int[] cnt, String s, int k) {
        int[] cnt_cur = new int[3];
        for (int i = 0; i < s.length(); ++i) {
            ++cnt_cur[s.charAt(i) - 'a'];
            if (i >= w) {
                --cnt_cur[s.charAt(i - w) - 'a'];
            }
            if (i >= w - 1 && cnt[0] - cnt_cur[0] >= k && cnt[1] - cnt_cur[1] >= k && cnt[2] - cnt_cur[2] >= k) {
                return true;
            }
        }
        return false;
    }

    // 1989. 捉迷藏中可捕获的最大人数 (Maximum Number of People That Can Be Caught in Tag)
    // --plus
    public int catchMaximumAmountofPeople(int[] team, int dist) {
        int n = team.length;
        // 人
        int i = 0;
        // 鬼
        int j = 0;
        int res = 0;
        while (i < n && j < n) {
            while (i < n && team[i] == 1) {
                ++i;
            }
            while (j < n && team[j] == 0) {
                ++j;
            }
            if (i == n || j == n) {
                break;
            }
            if (j - dist <= i && i <= j + dist) {
                ++i;
                ++j;
                ++res;
            } else if (j - dist > i) {
                ++i;
            } else {
                ++j;
            }

        }
        return res;

    }

    // 1272. 删除区间 (Remove Interval) --plus
    public List<List<Integer>> removeInterval(int[][] intervals, int[] toBeRemoved) {
        List<List<Integer>> res = new ArrayList<>();
        for (int[] interval : intervals) {
            if (interval[0] < toBeRemoved[0]) {
                res.add(List.of(interval[0], Math.min(interval[1], toBeRemoved[0])));
            }

            if (interval[1] > toBeRemoved[1]) {
                res.add(List.of(Math.max(interval[0], toBeRemoved[1]), interval[1]));
            }

        }
        return res;
    }

    // 2093. 前往目标城市的最小费用 (Minimum Cost to Reach City With Discounts) --plus
    public int minimumCost(int n, int[][] highways, int discounts) {
        Map<Integer, List<int[]>> graph = new HashMap<>();
        for (int[] highway : highways) {
            graph.computeIfAbsent(highway[0], k -> new ArrayList<>()).add(new int[] { highway[1], highway[2] });
            graph.computeIfAbsent(highway[1], k -> new ArrayList<>()).add(new int[] { highway[0], highway[2] });
        }
        int[][] fees = new int[n][discounts + 1];
        for (int i = 1; i < n; ++i) {
            Arrays.fill(fees[i], Integer.MAX_VALUE);
        }
        Queue<int[]> queue = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[2] - o2[2];
            }

        });
        // node, usedDiscounts, fee
        queue.offer(new int[] { 0, 0, 0 });
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0];
            int usedDiscounts = cur[1];
            int xFee = cur[2];
            if (x == n - 1) {
                return xFee;
            }
            for (int[] neighbor : graph.getOrDefault(x, new ArrayList<>())) {
                int y = neighbor[0];
                int fee = neighbor[1];
                if (usedDiscounts < discounts) {
                    int nFee2 = xFee + fee / 2;
                    if (nFee2 < fees[y][usedDiscounts + 1]) {
                        fees[y][usedDiscounts + 1] = nFee2;
                        queue.offer(new int[] { y, usedDiscounts + 1, nFee2 });
                    }
                }
                int nFee = fee + xFee;
                if (nFee < fees[y][usedDiscounts]) {
                    fees[y][usedDiscounts] = nFee;
                    queue.offer(new int[] { y, usedDiscounts, nFee });
                }

            }
        }
        return -1;

    }

    // 444. 序列重建 (Sequence Reconstruction) --plus
    public boolean sequenceReconstruction(int[] nums, List<List<Integer>> sequences) {
        int n = nums.length;
        int[] degees = new int[n + 1];
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (List<Integer> sequence : sequences) {
            for (int i = 0; i < sequence.size() - 1; ++i) {
                graph.computeIfAbsent(sequence.get(i), k -> new ArrayList<>()).add(sequence.get(i + 1));
                ++degees[sequence.get(i + 1)];
            }
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 1; i <= n; ++i) {
            if (degees[i] == 0) {
                queue.offer(i);
            }
        }
        int index = 0;
        while (!queue.isEmpty()) {
            int x = queue.poll();
            if (!queue.isEmpty()) {
                return false;
            }
            if (index == n) {
                return false;
            }
            if (x != nums[index]) {
                return false;
            }
            ++index;
            for (int y : graph.getOrDefault(x, new ArrayList<>())) {
                if (--degees[y] == 0) {
                    queue.offer(y);
                }
            }
        }
        return index == n;

    }

    // 1730. 获取食物的最短路径 (Shortest Path to Get Food) --plus
    public int getFood(char[][] grid) {
        int[][] directions = { { 0, -1 }, { 0, 1 }, { -1, 0 }, { 1, 0 } };
        int m = grid.length;
        int n = grid[0].length;
        int x = -1;
        int y = -1;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == '*') {
                    x = i;
                    y = j;
                    break;
                }
            }
            if (x != -1) {
                break;
            }
        }
        int res = 0;
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[] { x, y });
        grid[x][y] = 'X';
        while (!queue.isEmpty()) {
            ++res;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int[] cur = queue.poll();
                x = cur[0];
                y = cur[1];
                for (int[] d : directions) {
                    int nx = x + d[0];
                    int ny = y + d[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                        if (grid[nx][ny] == '#') {
                            return res;
                        }
                        if (grid[nx][ny] == 'X') {
                            continue;
                        }
                        grid[nx][ny] = 'X';
                        queue.offer(new int[] { nx, ny });
                    }

                }

            }

        }
        return -1;

    }

    // 1430. 判断给定的序列是否是二叉树从根到叶的路径 (Check If a String Is a Valid Sequence from Root
    // to Leaves Path in a Binary Tree)
    public boolean isValidSequence(TreeNode root, int[] arr) {
        return dfs1430(root, arr, 0);

    }

    private boolean dfs1430(TreeNode root, int[] arr, int index) {
        if (root == null || arr[index] != root.val) {
            return false;
        }
        if (index == arr.length - 1) {
            return root.left == null && root.right == null;

        }
        return dfs1430(root.left, arr, index + 1) || dfs1430(root.right, arr, index + 1);
    }

    // 1429. 第一个唯一数字 (First Unique Number) --plus
    class FirstUnique {
        private Map<Integer, Integer> counts;
        private List<Integer> list;
        private int index;

        public FirstUnique(int[] nums) {
            list = new ArrayList<>();
            counts = new HashMap<>();
            for (int num : nums) {
                list.add(num);
                counts.put(num, counts.getOrDefault(num, 0) + 1);
            }
        }

        public int showFirstUnique() {
            int n = list.size();
            while (index < n) {
                if (counts.get(list.get(index)) == 1) {
                    return list.get(index);
                }
                ++index;
            }
            return -1;

        }

        public void add(int value) {
            list.add(value);
            counts.put(value, counts.getOrDefault(value, 0) + 1);
        }
    }

    // 1166. 设计文件系统 (Design File System) --plus
    class FileSystem {
        private Map<String, Integer> map;

        public FileSystem() {
            map = new HashMap<>();
        }

        public boolean createPath(String path, int value) {
            if (map.containsKey(path)) {
                return false;
            }
            int index = path.lastIndexOf("/");
            String fa = path.substring(0, index);

            if (!"".equals(fa) && !map.containsKey(fa)) {
                return false;
            }
            map.put(path, value);
            return true;
        }

        public int get(String path) {
            return map.getOrDefault(path, -1);
        }
    }

    // 1891. 割绳子 (Cutting Ribbons) --plus
    public int maxLength(int[] ribbons, int k) {
        int left = 1;
        int right = (int) 1e5;
        int res = 0;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (check1891(ribbons, mid, k)) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;

    }

    private boolean check1891(int[] ribbons, int target, int k) {
        int count = 0;
        for (int ribbon : ribbons) {
            count += ribbon / target;
            if (count >= k) {
                return true;
            }
        }
        return false;
    }

    // 1868. 两个行程编码数组的积 (Product of Two Run-Length Encoded Arrays) --plus
    public List<List<Integer>> findRLEArray(int[][] encoded1, int[][] encoded2) {
        int i = 0;
        int j = 0;
        int n1 = encoded1.length;
        List<List<Integer>> res = new ArrayList<>();
        while (i < n1) {
            int count = Math.min(encoded1[i][1], encoded2[j][1]);
            int num = encoded1[i][0] * encoded2[j][0];
            int size = res.size();
            if (size != 0 && res.get(size - 1).get(0) == num) {
                res.set(size - 1, List.of(num, res.get(size - 1).get(1) + count));
            } else {
                res.add(List.of(num, count));
            }
            encoded1[i][1] -= count;
            encoded2[j][1] -= count;
            if (encoded1[i][1] == 0) {
                ++i;
            }
            if (encoded2[j][1] == 0) {
                ++j;
            }
        }
        return res;

    }

    // 1244. 力扣排行榜 (Design A Leaderboard) --plus
    class Leaderboard {
        private TreeMap<Integer, Set<Integer>> treeMap;
        private Map<Integer, Integer> map;

        public Leaderboard() {
            treeMap = new TreeMap<>(new Comparator<Integer>() {

                @Override
                public int compare(Integer o1, Integer o2) {
                    return o2 - o1;
                }

            });
            map = new HashMap<>();
        }

        public void addScore(int playerId, int score) {
            int originalScore = map.getOrDefault(playerId, 0);
            map.put(playerId, map.getOrDefault(playerId, 0) + score);
            treeMap.getOrDefault(originalScore, new HashSet<>()).remove(playerId);
            if (treeMap.getOrDefault(originalScore, new HashSet<>()).isEmpty()) {
                treeMap.remove(originalScore);
            }
            treeMap.computeIfAbsent(score + originalScore, k -> new HashSet<>()).add(playerId);

        }

        public int top(int K) {
            int sum = 0;
            for (Map.Entry<Integer, Set<Integer>> entry : treeMap.entrySet()) {
                int size = entry.getValue().size();
                if (K <= size) {
                    sum += entry.getKey() * K;
                    break;
                }
                sum += entry.getKey() * size;
                K -= size;
            }
            return sum;

        }

        public void reset(int playerId) {
            int score = map.remove(playerId);
            treeMap.get(score).remove(playerId);
            if (treeMap.get(score).isEmpty()) {
                treeMap.remove(score);
            }
        }
    }

    // 2495. 乘积为偶数的子数组数 (Number of Subarrays Having Even Product) --plus
    public long evenProduct(int[] nums) {
        int n = nums.length;
        long res = 0l;
        int lastEvenIndex = -1;
        for (int i = 0; i < n; ++i) {
            if (nums[i] % 2 == 0) {
                lastEvenIndex = i;
            }
            res += lastEvenIndex + 1;
        }
        return res;

    }

    // 1786. 从第一个节点出发到最后一个节点的受限路径数 (Number of Restricted Paths From First to Last
    // Node)
    public int countRestrictedPaths(int n, int[][] edges) {
        Map<Integer, List<int[]>> graph = new HashMap<>();
        for (int[] edge : edges) {
            graph.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(new int[] { edge[1], edge[2] });
            graph.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(new int[] { edge[0], edge[2] });
        }
        int[] dist = new int[n + 1];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[n] = 0;
        Queue<int[]> queue = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[1] - o2[1];
            }

        });
        queue.offer(new int[] { n, 0 });
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0];
            int d = cur[1];
            if (d > dist[x]) {
                continue;
            }
            for (int[] neighbor : graph.getOrDefault(x, new ArrayList<>())) {
                int y = neighbor[0];
                int nDist = d + neighbor[1];
                if (nDist < dist[y]) {
                    dist[y] = nDist;
                    queue.offer(new int[] { y, nDist });
                }
            }
        }
        int[][] ids = new int[n][2];
        for (int i = 0; i < n; ++i) {
            ids[i][0] = i + 1;
            ids[i][1] = dist[i + 1];
        }
        Arrays.sort(ids, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[1] - o2[1];
            }

        });
        final int mod = (int) (1e9 + 7);
        int[] dp = new int[n + 1];
        dp[n] = 1;
        for (int i = 0; i < n; ++i) {
            int x = ids[i][0];
            int d = ids[i][1];
            for (int[] neighbor : graph.getOrDefault(x, new ArrayList<>())) {
                int y = neighbor[0];
                if (dist[y] < d) {
                    dp[x] = (dp[x] + dp[y]) % mod;
                }
            }
            if (x == 1) {
                break;
            }
        }
        return dp[1];

    }

    // 2473. Minimum Cost to Buy Apples --plus
    public long[] minCost(int n, int[][] roads, int[] appleCost, int k) {
        long[] res = new long[n];
        Map<Integer, List<int[]>> graph = new HashMap<>();
        for (int[] road : roads) {
            graph.computeIfAbsent(road[0] - 1, o -> new ArrayList<>()).add(new int[] { road[1] - 1, road[2] });
            graph.computeIfAbsent(road[1] - 1, o -> new ArrayList<>()).add(new int[] { road[0] - 1, road[2] });
        }
        for (int i = 0; i < n; ++i) {
            res[i] = dijkstra2473(i, graph, appleCost, k);
        }
        return res;

    }

    private long dijkstra2473(int start, Map<Integer, List<int[]>> graph, int[] appleCost, int k) {
        int n = appleCost.length;
        long[] dis = new long[n];
        Arrays.fill(dis, Long.MAX_VALUE);
        dis[start] = 0;
        Queue<long[]> queue = new PriorityQueue<>(new Comparator<long[]>() {

            @Override
            public int compare(long[] o1, long[] o2) {
                return Long.valueOf(o1[1]).compareTo(Long.valueOf(o2[1]));
            }
        });

        long res = appleCost[start];
        queue.offer(new long[] { start, 0l });
        while (!queue.isEmpty()) {
            long[] cur = queue.poll();
            int x = (int) cur[0];
            long d = cur[1];
            if (d > dis[x]) {
                continue;
            }
            res = Math.min(res, d + appleCost[x]);
            for (int[] neighbor : graph.getOrDefault(x, new ArrayList<>())) {
                int y = neighbor[0];
                long fee = neighbor[1];
                long total = d + fee + fee * k;
                if (total < dis[y]) {
                    dis[y] = total;
                    queue.offer(new long[] { y, total });
                }
            }
        }
        return res;
    }

    // 2077. 殊途同归 (Paths in Maze That Lead to Same Room) --plus
    public int numberOfPaths(int n, int[][] corridors) {
        Map<Integer, Set<Integer>> graph = new HashMap<>();
        for (int[] corridor : corridors) {
            graph.computeIfAbsent(corridor[0], k -> new HashSet<>()).add(corridor[1]);
            graph.computeIfAbsent(corridor[1], k -> new HashSet<>()).add(corridor[0]);
        }
        int res = 0;
        for (int[] corridor : corridors) {
            Set<Integer> set1 = graph.getOrDefault(corridor[0], new HashSet<>());
            Set<Integer> set2 = graph.getOrDefault(corridor[1], new HashSet<>());
            for (int x : set1) {
                if (set2.contains(x)) {
                    ++res;
                }
            }
        }
        return res / 3;

    }

    // 951. 翻转等价二叉树 (Flip Equivalent Binary Trees)
    public boolean flipEquiv(TreeNode root1, TreeNode root2) {
        if (root1 == null || root2 == null) {
            return root1 == null && root2 == null;
        }
        if (root1.val != root2.val) {
            return false;
        }
        return flipEquiv(root1.left, root2.left) && flipEquiv(root1.right, root2.right)
                || flipEquiv(root1.left, root2.right) && flipEquiv(root1.right, root2.left);
    }

    // 2517. 礼盒的最大甜蜜度 (Maximum Tastiness of Candy Basket)
    public int maximumTastiness(int[] price, int k) {
        Arrays.sort(price);
        int n = price.length;
        int left = 0;
        int right = (price[n - 1] - price[0]) / (k - 1);
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (check2517(price, mid) >= k) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;

    }

    private int check2517(int[] price, int target) {
        int cnt = 1;
        int pre = price[0];
        int i = 1;
        while (i < price.length) {
            if (price[i] - pre >= target) {
                ++cnt;
                pre = price[i];
            }
            ++i;
        }
        return cnt;
    }

    // 616. 给字符串添加加粗标签 (Add Bold Tag in String) --plus
    // 758. 字符串中的加粗单词 (Bold Words in String) --plus
    public String addBoldTag(String s, String[] words) {
        int n = s.length();
        boolean[] bold = new boolean[n];
        for (int i = 0; i < n; ++i) {
            search: for (String word : words) {
                int k = i;
                int j = 0;
                int m = word.length();
                while (k < n && j < m) {
                    if (s.charAt(k) != word.charAt(j)) {
                        continue search;
                    }
                    ++k;
                    ++j;
                }
                if (j == m) {
                    for (int x = i; x < i + m; ++x) {
                        bold[x] = true;
                    }
                }
            }
        }
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < n; ++i) {
            if (bold[i] && (i == 0 || !bold[i - 1])) {
                res.append("<b>");
            }
            res.append(s.charAt(i));
            if (bold[i] && (i == n - 1 || !bold[i + 1])) {
                res.append("</b>");
            }
        }
        return res.toString();

    }

    // 1902. 给定二叉搜索树的插入顺序求深度 (Depth of BST Given Insertion Order) --plus
    public int maxDepthBST(int[] order) {
        TreeMap<Integer, Integer> map = new TreeMap<>();
        map.put(0, 0);
        map.put(Integer.MAX_VALUE, 0);
        map.put(order[0], 1);
        int res = 1;
        for (int i = 1; i < order.length; ++i) {
            int val = order[i];
            int max = Math.max(map.lowerEntry(val).getValue(), map.higherEntry(val).getValue()) + 1;
            res = Math.max(res, max);
            map.put(val, max);
        }
        return res;

    }

    // 6278. 统计能整除数字的位数
    public int countDigits(int num) {
        int copy = num;
        int res = 0;
        while (copy > 0) {
            int mod = copy % 10;
            if (num % mod == 0) {
                ++res;
            }
            copy /= 10;
        }
        return res;

    }

    // 6279. 数组乘积中的不同质因数数目
    public int distinctPrimeFactors(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            for (int i = 2; i * i <= num; ++i) {
                while (num % i == 0) {
                    set.add(i);
                    num /= i;
                }
            }
            if (num != 1) {
                set.add(num);
            }
        }
        return set.size();

    }

    // 2522. 将字符串分割成值不超过 K 的子字符串 (Partition String Into Substrings With Values at
    // Most K)
    public int minimumPartition(String s, int k) {
        int res = 0;
        int n = s.length();
        int i = 0;
        while (i < n) {
            long num = s.charAt(i) - '0';
            if (num > k) {
                return -1;
            }
            int j = i + 1;
            while (j < n) {
                num = num * 10 + s.charAt(j) - '0';
                if (num > k) {
                    break;
                }
                ++j;
            }
            ++res;
            i = j;

        }
        return res;

    }

    // 2522. 将字符串分割成值不超过 K 的子字符串 (Partition String Into Substrings With Values at
    // Most K)
    private int n2522;
    private char[] arr2522;
    private int k2522;
    private int[] memo2522;

    public int minimumPartition2(String s, int k) {
        this.n2522 = s.length();
        this.k2522 = k;
        this.arr2522 = s.toCharArray();
        this.memo2522 = new int[n2522];
        Arrays.fill(memo2522, -1);
        int res = dfs2522(0);
        return res < n2522 + 1 ? res : -1;
    }

    private int dfs2522(int i) {
        if (i == n2522) {
            return 0;
        }
        if (memo2522[i] != -1) {
            return memo2522[i];
        }
        int res = n2522 + 1;
        long val = 0L;
        for (int j = i; j < n2522; ++j) {
            val = val * 10 + arr2522[j] - '0';
            if (val > k2522) {
                break;
            }
            res = Math.min(res, dfs2522(j + 1) + 1);
        }
        return memo2522[i] = res;
    }

    // 6280. 范围内最接近的两个质数
    public int[] closestPrimes(int left, int right) {
        boolean[] isPrim = new boolean[right + 1];
        Arrays.fill(isPrim, true);
        isPrim[1] = false;
        // 从 2 开始枚举到 sqrt(n)。
        for (int i = 2; i * i <= right; i++) {
            // 如果当前是素数
            if (isPrim[i]) {
                // 就把从 i*i 开始，i 的所有倍数都设置为 false。
                for (int j = i * i; j <= right; j += i) {
                    isPrim[j] = false;
                }
            }
        }

        int[] res = { -1, -1 };
        int num1 = -1;
        int num2 = -1;
        int diff = Integer.MAX_VALUE;
        for (int i = left; i <= right; i++) {
            if (isPrim[i]) {
                num1 = num2;
                num2 = i;
                if (num1 != -1 && num2 != -1) {
                    if (num2 - num1 < diff) {
                        diff = num2 - num1;
                        res = new int[] { num1, num2 };
                    }
                }
            }
        }
        return res;

    }

    // 803. 打砖块 (Bricks Falling When Hit)
    public int[] hitBricks(int[][] grid, int[][] hits) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] copy = new int[m][n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                copy[i][j] = grid[i][j];
            }
        }
        for (int[] hit : hits) {
            copy[hit[0]][hit[1]] = 0;
        }
        Union803 union = new Union803(m * n + 1);
        int dummy = m * n;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (copy[i][j] == 0) {
                    continue;
                }
                if (i == 0) {
                    union.union(getIndex803(i, j, n), dummy);
                } else {
                    if (i - 1 >= 0 && copy[i - 1][j] == 1) {
                        union.union(getIndex803(i, j, n), getIndex803(i - 1, j, n));
                    }
                    if (j - 1 >= 0 && copy[i][j - 1] == 1) {
                        union.union(getIndex803(i, j, n), getIndex803(i, j - 1, n));
                    }
                }
            }
        }
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };

        int[] res = new int[hits.length];
        for (int i = hits.length - 1; i >= 0; --i) {
            int x = hits[i][0];
            int y = hits[i][1];
            if (grid[x][y] == 0) {
                continue;
            }
            int origin = union.getSize(dummy);
            if (x == 0) {
                union.union(getIndex803(x, y, n), dummy);
            }
            for (int[] direction : directions) {
                int nx = x + direction[0];
                int ny = y + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && copy[nx][ny] == 1) {
                    union.union(getIndex803(x, y, n), getIndex803(nx, ny, n));
                }
            }
            int current = union.getSize(dummy);
            res[i] = Math.max(0, current - origin - 1);
            copy[x][y] = 1;
        }
        return res;

    }

    private int getIndex803(int i, int j, int n) {
        return i * n + j;
    }

    public class Union803 {
        private int[] rank;
        private int[] parent;
        private int[] size;

        public Union803(int n) {
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
            if (rank[root1] > rank[root2]) {
                parent[root2] = root1;
                size[root1] += size[root2];
            } else {
                parent[root1] = root2;
                size[root2] += size[root1];
                if (rank[root1] == rank[root2]) {
                    ++rank[root2];
                }
            }
        }

        public int getSize(int p) {
            int root = getRoot(p);
            return size[root];
        }

    }

    // 1036. 逃离大迷宫 (Escape a Large Maze)
    private static final int BLOCKED1036 = -1;
    private static final int CONNECTED1036 = 0;
    private static final int VALID1036 = 1;
    private static final int SIDE = (int) 1e6;

    public boolean isEscapePossible(int[][] blocked, int[] source, int[] target) {
        int n = blocked.length;
        if (n <= 1) {
            return true;
        }
        Set<Bean1036> blockedSet = new HashSet<>();
        for (int[] b : blocked) {
            blockedSet.add(new Bean1036(b[0], b[1]));
        }
        int check = check1036(blockedSet, source, target);
        if (check == BLOCKED1036) {
            return false;
        }
        if (check == CONNECTED1036) {
            return true;
        }
        check = check1036(blockedSet, target, source);
        return check != BLOCKED1036;

    }

    private int check1036(Set<Bean1036> blockedSet, int[] source, int[] target) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        Set<Bean1036> visitedSet = new HashSet<>();
        Queue<Bean1036> queue = new LinkedList<>();
        visitedSet.add(new Bean1036(source[0], source[1]));
        queue.offer(new Bean1036(source[0], source[1]));
        int limit = blockedSet.size() * (blockedSet.size() - 1) / 2;
        while (!queue.isEmpty() && limit > 0) {
            Bean1036 bean = queue.poll();
            int x = bean.x;
            int y = bean.y;
            if (x == target[0] && y == target[1]) {
                return CONNECTED1036;
            }
            for (int[] d : directions) {
                int nx = x + d[0];
                int ny = y + d[1];
                Bean1036 nbean = new Bean1036(nx, ny);
                if (nx >= 0 && nx < SIDE && ny >= 0 && ny < SIDE && !blockedSet.contains(nbean)
                        && !visitedSet.contains(nbean)) {
                    --limit;
                    visitedSet.add(nbean);
                    queue.offer(nbean);
                }
            }
        }
        if (limit > 0) {
            return BLOCKED1036;
        }
        return VALID1036;
    }

    class Bean1036 {
        int x;
        int y;

        public Bean1036(int x, int y) {
            this.x = x;
            this.y = y;
        }

        @Override
        public int hashCode() {
            return (int) ((long) x << 20 | y);
        }

        @Override
        public boolean equals(Object obj) {
            if (obj instanceof Bean1036) {
                Bean1036 b = (Bean1036) obj;
                return b.x == this.x && b.y == this.y;
            }
            return false;
        }
    }

    // 剑指 Offer 51. 数组中的逆序对 --二分查找 还需掌握：树状数组
    public int reversePairs(int[] nums) {
        List<Integer> list = new ArrayList<>();
        int res = 0;
        for (int i = nums.length - 1; i >= 0; --i) {
            res += binarySearchOffer51(list, nums[i]);
        }
        return res;

    }

    // 查找有序list中，小于target的个数，并插入，保持list有序
    private int binarySearchOffer51(List<Integer> list, int target) {
        int count = 0;
        if (list.isEmpty() || list.get(0) >= target) {
            list.add(0, target);
            return count;
        }
        if (list.get(list.size() - 1) < target) {
            count = list.size();
            list.add(target);
            return count;
        }
        int left = 0;
        int right = list.size() - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (target > list.get(mid)) {
                count = mid + 1;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        list.add(left, target);
        return count;
    }

    // 剑指 Offer 51. 数组中的逆序对 --归并排序 还需掌握：树状数组
    private int res_offer_051;

    public int reversePairs2(int[] nums) {
        int n = nums.length;
        if (n <= 1) {
            return 0;
        }
        dfs_offer_051(nums);
        return res_offer_051;

    }

    private int[] dfs_offer_051(int[] nums) {
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
        int[] sorted1 = dfs_offer_051(a);
        int[] sorted2 = dfs_offer_051(b);
        int i = 0;
        for (int x : sorted1) {
            while (i < sorted2.length && x > sorted2[i]) {
                ++i;
            }
            res_offer_051 += i;
        }
        int[] res = new int[n];
        int j = 0;
        int k = 0;
        int index = 0;
        while (j < sorted1.length && k < sorted2.length) {
            if (sorted1[j] < sorted2[k]) {
                res[index++] = sorted1[j++];
            } else {
                res[index++] = sorted2[k++];
            }
        }
        while (j < sorted1.length) {
            res[index++] = sorted1[j++];
        }
        while (k < sorted2.length) {
            res[index++] = sorted2[k++];
        }
        return res;

    }

    // 剑指 Offer 51. 数组中的逆序对 -- 离散化、线段树 还需掌握：树状数组
    private int[] seg_offer_51;

    public int reversePairs3(int[] nums) {
        // 离散化
        TreeSet<Integer> treeSet = new TreeSet<>();
        for (int num : nums) {
            treeSet.add(num);
        }
        int n = treeSet.size() + 1;
        // key : original num
        // val : 离散化后的值 不影响计算逆序对
        Map<Integer, Integer> map = new HashMap<>();
        int cnt = 1;
        for (int i : treeSet) {
            map.put(i, cnt++);
        }
        int res = 0;
        this.seg_offer_51 = new int[n * 4];
        for (int num : nums) {
            res += query_offer_51(1, 1, n, map.get(num) + 1, n);
            insert_offer_51(1, 1, n, map.get(num));
        }
        return res;
    }

    private void insert_offer_51(int o, int l, int r, int id) {
        if (l == r) {
            ++seg_offer_51[o];
            return;
        }
        int mid = l + ((r - l) >> 1);
        if (id <= mid) {
            insert_offer_51(o * 2, l, mid, id);
        } else {
            insert_offer_51(o * 2 + 1, mid + 1, r, id);
        }
        seg_offer_51[o] = seg_offer_51[o * 2] + seg_offer_51[o * 2 + 1];
    }

    private int query_offer_51(int o, int l, int r, int L, int R) {
        if (L <= l && r <= R) {
            return seg_offer_51[o];
        }
        int res = 0;
        int mid = l + ((r - l) >> 1);
        if (L <= mid) {
            res += query_offer_51(o * 2, l, mid, L, R);
        }
        if (R >= mid + 1) {
            res += query_offer_51(o * 2 + 1, mid + 1, r, L, R);
        }
        return res;
    }

    // 剑指 Offer 51. 数组中的逆序对 -- 动态开点线段树 还需掌握：树状数组
    public int reversePairs4(int[] nums) {
        SegNode_Offer_51 root = new SegNode_Offer_51((long) Integer.MIN_VALUE, (long) Integer.MAX_VALUE);
        int res = 0;
        for (long x : nums) {
            res += query_offer_51(root, x + 1, (long) Integer.MAX_VALUE);
            insert_offer_51(root, x);
        }
        return res;
    }

    private int query_offer_51(SegNode_Offer_51 node, long L, long R) {
        if (node == null) {
            return 0;
        }
        if (L > node.hi || R < node.lo) {
            return 0;
        }
        if (L <= node.lo && node.hi <= R) {
            return node.val;
        }
        return query_offer_51(node.left, L, R) + query_offer_51(node.right, L, R);
    }

    private void insert_offer_51(SegNode_Offer_51 node, long x) {
        ++node.val;
        if (node.lo == node.hi) {
            return;
        }
        long mid = node.lo + ((node.hi - node.lo) >> 1);
        if (x <= mid) {
            if (node.left == null) {
                node.left = new SegNode_Offer_51(node.lo, mid);
            }
            insert_offer_51(node.left, x);
        } else {
            if (node.right == null) {
                node.right = new SegNode_Offer_51(mid + 1, node.hi);
            }
            insert_offer_51(node.right, x);
        }
    }

    public class SegNode_Offer_51 {
        public long lo;
        public long hi;
        public int val;
        public SegNode_Offer_51 left;
        public SegNode_Offer_51 right;

        public SegNode_Offer_51(long lo, long hi) {
            this.lo = lo;
            this.hi = hi;
        }
    }

    // 834. 树中距离之和 (Sum of Distances in Tree) --树型dp
    private int n834;
    private List<Integer>[] g834;
    private int[] cnts834;
    private int[] res834;

    public int[] sumOfDistancesInTree(int n, int[][] edges) {
        this.n834 = n;
        this.g834 = new ArrayList[n];
        Arrays.setAll(g834, k -> new ArrayList<>());
        for (int[] e : edges) {
            int a = e[0];
            int b = e[1];
            g834[a].add(b);
            g834[b].add(a);
        }
        this.cnts834 = new int[n];
        this.res834 = new int[n];
        dfs834(0, -1, 0);
        reRoot834(0, -1);
        return res834;

    }

    private void reRoot834(int x, int fa) {
        for (int y : g834[x]) {
            if (y != fa) {
                res834[y] = res834[x] + n834 - cnts834[y] * 2;
                reRoot834(y, x);
            }
        }
    }

    private void dfs834(int x, int fa, int d) {
        cnts834[x] = 1;
        res834[0] += d;
        for (int y : g834[x]) {
            if (y != fa) {
                dfs834(y, x, d + 1);
                cnts834[x] += cnts834[y];
            }
        }
    }

    // 1803. 统计异或值在范围内的数对有多少 (Count Pairs With XOR in a Range)
    public int countPairs(int[] nums, int low, int high) {
        Map<Integer, Integer> counts = new HashMap<>();
        for (int num : nums) {
            counts.put(num, counts.getOrDefault(num, 0) + 1);
        }
        int res = 0;
        for (++high; high > 0; high >>= 1, low >>= 1) {
            Map<Integer, Integer> next = new HashMap<>();
            for (Map.Entry<Integer, Integer> entry : counts.entrySet()) {
                int k = entry.getKey();
                int c = entry.getValue();
                if ((high & 1) == 1) {
                    res += c * counts.getOrDefault(k ^ (high - 1), 0);
                }
                if ((low & 1) == 1) {
                    res -= c * counts.getOrDefault(k ^ (low - 1), 0);
                }
                next.put(k >> 1, next.getOrDefault(k >> 1, 0) + c);
            }
            counts = next;
        }
        return res / 2;

    }

    // 2458. 移除子树后的二叉树高度 (Height of Binary Tree After Subtree Removal Queries)
    private int[] left2458;
    private int[] right2458;
    private static final int N2458 = (int) (1e5 + 1);
    private int maxHeight2458;

    public int[] treeQueries(TreeNode root, int[] queries) {
        left2458 = new int[N2458];
        right2458 = new int[N2458];
        maxHeight2458 = 0;
        dfs2458_1(root, 0);
        maxHeight2458 = 0;
        dfs2458_2(root, 0);
        for (int i = 0; i < queries.length; ++i) {
            queries[i] = Math.max(left2458[queries[i]], right2458[queries[i]]);
        }
        return queries;

    }

    private void dfs2458_1(TreeNode node, int h) {
        if (node == null) {
            return;
        }
        left2458[node.val] = maxHeight2458;
        maxHeight2458 = Math.max(maxHeight2458, h);
        dfs2458_1(node.left, h + 1);
        dfs2458_1(node.right, h + 1);
    }

    private void dfs2458_2(TreeNode node, int h) {
        if (node == null) {
            return;
        }
        right2458[node.val] = maxHeight2458;
        maxHeight2458 = Math.max(maxHeight2458, h);
        dfs2458_2(node.right, h + 1);
        dfs2458_2(node.left, h + 1);
    }

    // 315. 计算右侧小于当前元素的个数 (Count of Smaller Numbers After Self) --二分查找
    // 还需掌握：树状数组 归并排序
    private List<Integer> list315;

    public List<Integer> countSmaller(int[] nums) {
        int n = nums.length;
        list315 = new ArrayList<>();
        int[] res = new int[n];
        for (int i = n - 1; i >= 0; --i) {
            res[i] = binarySearch315(nums[i]);
        }
        return Arrays.stream(res).boxed().collect(Collectors.toList());

    }

    private int binarySearch315(int target) {
        if (list315.isEmpty() || target <= list315.get(0)) {
            list315.add(0, target);
            return 0;
        }
        int count = 0;
        if (list315.get(list315.size() - 1) < target) {
            count = list315.size();
            list315.add(target);
            return count;
        }
        int left = 0;
        int right = list315.size() - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (list315.get(mid) < target) {
                count = mid + 1;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        list315.add(left, target);
        return count;
    }

    // 315. 计算右侧小于当前元素的个数 (Count of Smaller Numbers After Self) --线段树
    private int[] cnts315;

    public List<Integer> countSmaller2(int[] nums) {
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        int n = nums.length;
        for (int x : nums) {
            min = Math.min(min, x);
            max = Math.max(max, x);
        }
        int add = Math.max(0, 2 - min);
        int m = max + add;
        this.cnts315 = new int[m * 4];
        List<Integer> res = new ArrayList<>();
        for (int i = n - 1; i >= 0; --i) {
            int x = nums[i] + add;
            int cnt = query315(1, 1, m, 1, x - 1);
            res.add(cnt);
            add315(1, 1, m, x);
        }
        Collections.reverse(res);
        return res;

    }

    private void add315(int o, int l, int r, int id) {
        if (l == r) {
            ++cnts315[o];
            return;
        }
        int m = l + ((r - l) >> 1);
        if (id <= m) {
            add315(o * 2, l, m, id);
        } else {
            add315(o * 2 + 1, m + 1, r, id);
        }
        cnts315[o] = cnts315[o * 2] + cnts315[o * 2 + 1];
    }

    private int query315(int o, int l, int r, int L, int R) {
        if (L <= l && r <= R) {
            return cnts315[o];
        }
        int m = l + ((r - l) >> 1);
        int cnt = 0;
        if (L <= m) {
            cnt += query315(o * 2, l, m, L, R);
        }
        if (R >= m + 1) {
            cnt += query315(o * 2 + 1, m + 1, r, L, R);
        }
        return cnt;
    }

    // 315. 计算右侧小于当前元素的个数 (Count of Smaller Numbers After Self) --动态开点线段树
    public List<Integer> countSmaller3(int[] nums) {
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for (int x : nums) {
            min = Math.min(min, x);
            max = Math.max(max, x);
        }
        SegNode315 root = new SegNode315(min, max);
        List<Integer> list = new ArrayList<>();
        for (int i = nums.length - 1; i >= 0; --i) {
            list.add(query315(root, min, nums[i] - 1));
            insert315(root, nums[i]);
        }
        Collections.reverse(list);
        return list;

    }

    private int query315(SegNode315 node, int L, int R) {
        if (node == null) {
            return 0;
        }
        if (L > node.hi || R < node.lo) {
            return 0;
        }
        if (L <= node.lo && node.hi <= R) {
            return node.val;
        }
        return query315(node.left, L, R) + query315(node.right, L, R);
    }

    private void insert315(SegNode315 node, int x) {
        ++node.val;
        if (node.hi == node.lo) {
            return;
        }
        int mid = node.lo + ((node.hi - node.lo) >> 1);
        if (x <= mid) {
            if (node.left == null) {
                node.left = new SegNode315(node.lo, mid);
            }
            insert315(node.left, x);
        } else {
            if (node.right == null) {
                node.right = new SegNode315(mid + 1, node.hi);
            }
            insert315(node.right, x);
        }
    }

    public class SegNode315 {
        public int lo;
        public int hi;
        public int val;
        public SegNode315 left;
        public SegNode315 right;

        public SegNode315(int lo, int hi) {
            this.lo = lo;
            this.hi = hi;
        }

    }

    // 2426. 满足不等式的数对数目 (Number of Pairs Satisfying Inequality) --二分查找 还需掌握 树状数组
    public long numberOfPairs(int[] nums1, int[] nums2, int diff) {
        int n = nums1.length;
        List<Integer> list = new ArrayList<>();
        long res = 0l;
        for (int i = 0; i < n; ++i) {
            int target = nums1[i] - nums2[i] + diff;
            res += binarySearch2426(list, target);
            binarySearch2426_2(list, nums1[i] - nums2[i]);
        }
        return res;

    }

    private int binarySearch2426(List<Integer> list, int target) {
        int n = list.size();
        if (n == 0 || target < list.get(0)) {
            return 0;
        }
        if (target >= list.get(n - 1)) {
            return n;
        }
        int count = 0;
        int left = 0;
        int right = n - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (list.get(mid) <= target) {
                count = mid + 1;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return count;
    }

    private void binarySearch2426_2(List<Integer> list, int target) {
        int n = list.size();
        if (n == 0 || target <= list.get(0)) {
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
            int mid = left + ((right - left) >>> 1);
            if (list.get(mid) < target) {
                left = mid + 1;
            } else if (list.get(mid) > target) {
                right = mid - 1;
            } else {
                list.add(mid, target);
                return;
            }
        }
        list.add(left, target);
        return;
    }

    // 2426. 满足不等式的数对数目 (Number of Pairs Satisfying Inequality) --归并排序 还需掌握 树状数组
    private long res2426;
    private int diff2426;

    public long numberOfPairs2(int[] nums1, int[] nums2, int diff) {
        int n = nums1.length;
        int[] arr = new int[n];
        for (int i = 0; i < n; ++i) {
            arr[i] = nums1[i] - nums2[i];
        }
        this.diff2426 = diff;
        mergingSort2426(arr);
        return res2426;
    }

    private int[] mergingSort2426(int[] arr) {
        int n = arr.length;
        if (n <= 1) {
            return arr;
        }
        int mid = n >> 1;
        int[] a = new int[mid];
        int[] b = new int[n - mid];
        for (int i = 0; i < n; ++i) {
            if (i < mid) {
                a[i] = arr[i];
            } else {
                b[i - mid] = arr[i];
            }
        }
        int[] sorted1 = mergingSort2426(a);
        int[] sorted2 = mergingSort2426(b);
        int i = 0;
        int j = 0;
        while (i < sorted1.length && j < sorted2.length) {
            if (sorted2[j] + diff2426 >= sorted1[i]) {
                res2426 += sorted2.length - j;
                ++i;
            } else {
                ++j;
            }
        }

        int[] res = new int[n];
        i = 0;
        j = 0;
        int index = 0;
        while (i < sorted1.length && j < sorted2.length) {
            if (sorted1[i] < sorted2[j]) {
                res[index++] = sorted1[i++];
            } else {
                res[index++] = sorted2[j++];
            }
        }
        while (i < sorted1.length) {
            res[index++] = sorted1[i++];
        }
        while (j < sorted2.length) {
            res[index++] = sorted2[j++];
        }
        return res;

    }

    // 2466. 统计构造好字符串的方案数 (Count Ways To Build Good Strings)
    private int[] memo2466;
    private int zero2466;
    private int one2466;
    private final int MOD2466 = (int) (1e9 + 7);

    public int countGoodStrings(int low, int high, int zero, int one) {
        memo2466 = new int[high + 1];
        Arrays.fill(memo2466, -1);
        this.zero2466 = zero;
        this.one2466 = one;
        int res = 0;
        for (int i = high; i >= low; --i) {
            res = (res + dfs2466(i)) % MOD2466;
        }
        return res;

    }

    private int dfs2466(int i) {
        if (i < 0) {
            return 0;
        }
        if (i == 0) {
            return 1;
        }
        if (memo2466[i] != -1) {
            return memo2466[i];
        }
        return memo2466[i] = (dfs2466(i - zero2466) + dfs2466(i - one2466)) % MOD2466;
    }

    // 2466. 统计构造好字符串的方案数 (Count Ways To Build Good Strings)
    public int countGoodStrings2(int low, int high, int zero, int one) {
        int[] dp = new int[high + 1];
        int res = 0;
        dp[0] = 1;
        final int MOD = (int) (1e9 + 7);
        for (int i = 1; i <= high; ++i) {
            if (i - zero >= 0) {
                dp[i] = (dp[i] + dp[i - zero]) % MOD;
            }
            if (i - one >= 0) {
                dp[i] = (dp[i] + dp[i - one]) % MOD;
            }
            if (i >= low) {
                res = (res + dp[i]) % MOD;
            }
        }
        return res;

    }

    // 2525. 根据规则将箱子分类 (Categorize Box According to Criteria)
    public String categorizeBox(int length, int width, int height, int mass) {
        long a = 1l;
        a = a * length;
        a = a * width;
        a = a * height;
        boolean tiji = a >= 1e9;
        int b = (int) 1e4;
        boolean weidu = length >= b || width >= b || height >= b || mass >= b;
        boolean isBluky = tiji || weidu;
        boolean isHeavy = mass >= 100;
        if (isBluky && isHeavy) {
            return "Both";
        }
        if (isBluky) {
            return "Bulky";
        }
        if (isHeavy) {
            return "Heavy";
        }
        return "Neither";

    }

    // 6288. 找到数据流中的连续整数
    class DataStream {
        private int count;
        private int value;
        private int k;

        public DataStream(int value, int k) {
            this.value = value;
            this.k = k;
        }

        public boolean consec(int num) {
            if (value == num) {
                ++count;
            } else {
                count = 0;
            }
            return count >= k;
        }
    }

    // 6289. 查询数组 Xor 美丽值
    public int xorBeauty(int[] nums) {
        int res = 0;
        for (int num : nums) {
            res ^= num;
        }
        return res;

    }

    // 2529. 正整数和负整数的最大计数 (Maximum Count of Positive Integer and Negative Integer)
    public int maximumCount(int[] nums) {
        int pos = 0;
        int neg = 0;
        for (int num : nums) {
            if (num < 0) {
                ++neg;
            } else if (num > 0) {
                ++pos;
            }
        }
        return Math.max(pos, neg);

    }

    // 2529. 正整数和负整数的最大计数 (Maximum Count of Positive Integer and Negative Integer)
    public int maximumCount2(int[] nums) {
        int neg = binarySearch6283_neg(nums);
        int pos = binarySearch6283_pos(nums);
        return Math.max(neg, pos);

    }

    private int binarySearch6283_pos(int[] nums) {
        int n = nums.length;
        if (nums[n - 1] <= 0) {
            return 0;
        }
        if (nums[0] > 0) {
            return n;
        }
        int left = 0;
        int right = n - 1;
        int count = 0;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (nums[mid] > 0) {
                count = n - mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return count;
    }

    private int binarySearch6283_neg(int[] nums) {
        int n = nums.length;
        if (nums[0] >= 0) {
            return 0;
        }
        if (nums[n - 1] < 0) {
            return n;
        }
        int left = 0;
        int right = n - 1;
        int count = 0;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (nums[mid] < 0) {
                count = mid + 1;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return count;
    }

    // 6285. 执行 K 次操作后的最大分数
    public long maxKelements(int[] nums, int k) {
        Queue<Integer> queue = new PriorityQueue<>(new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return o2 - o1;
            }

        });
        for (int num : nums) {
            queue.offer(num);
        }
        long res = 0l;
        while (queue.peek() != 0 && k-- > 0) {
            int cur = queue.poll();
            res += cur;
            queue.offer((cur + 2) / 3);
        }
        return res;

    }

    // 6284. 使字符串总不同字符的数目相等
    public boolean isItPossible(String word1, String word2) {
        int mask1 = 0;
        int mask2 = 0;
        int[] counts1 = new int[26];
        int[] counts2 = new int[26];
        for (char c : word1.toCharArray()) {
            mask1 |= 1 << (c - 'a');
            ++counts1[c - 'a'];
        }
        for (char c : word2.toCharArray()) {
            mask2 |= 1 << (c - 'a');
            ++counts2[c - 'a'];
        }

        for (int i = 0; i < 26; ++i) {
            if (counts1[i] > 0) {
                for (int j = 0; j < 26; ++j) {
                    if (counts2[j] > 0) {
                        int copy1 = mask1;
                        int copy2 = mask2;
                        if (i != j) {
                            if (counts1[i] == 1) {
                                copy1 ^= 1 << i;
                            }
                            if (counts2[i] == 0) {
                                copy2 ^= 1 << i;
                            }
                            if (counts2[j] == 1) {
                                copy2 ^= 1 << j;
                            }
                            if (counts1[j] == 0) {
                                copy1 ^= 1 << j;
                            }
                        }
                        if (Integer.bitCount(copy1) == Integer.bitCount(copy2)) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;

    }

    // 6290. 最大化城市的最小供电站数目
    public long maxPower(int[] stations, int r, int k) {
        int n = stations.length;
        long[] diff = new long[n];
        for (int i = 0; i < n; ++i) {
            diff[Math.max(0, i - r)] += stations[i];
            if (i + r + 1 < n) {
                diff[i + r + 1] -= stations[i];
            }
        }
        long mn = diff[0];
        for (int i = 1; i < n; ++i) {
            diff[i] += diff[i - 1];
            mn = Math.min(mn, diff[i]);
        }

        long left = mn;
        long right = (long) (mn + k);
        long res = 0l;
        while (left <= right) {
            long mid = left + ((right - left) >> 1);
            if (check6290(diff, r, k, mid)) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;

    }

    private boolean check6290(long[] stations, int r, int k, long target) {
        int n = stations.length;
        long[] diff = new long[n];
        long delta = 0l;
        for (int i = 0; i < n; ++i) {
            long d = 0l;
            if (i == 0) {
                d = target - stations[i];
            } else {
                diff[i] += diff[i - 1];
                d = target - stations[i] - diff[i];
            }
            if (d > 0) {
                delta += d;
                if (delta > k) {
                    return false;
                }
                diff[i] += d;
                if (i + 2 * r + 1 < n) {
                    diff[i + 2 * r + 1] -= d;
                }
            }
        }
        return true;
    }

    // 1032. 字符流 (Stream of Characters)
    class StreamChecker {
        private class Trie1032 {
            private Trie1032[] children;
            private boolean isEnd;

            public Trie1032() {
                children = new Trie1032[26];
            }

            public void insert(String s) {
                Trie1032 node = this;
                for (int i = s.length() - 1; i >= 0; --i) {
                    int index = s.charAt(i) - 'a';
                    if (node.children[index] == null) {
                        node.children[index] = new Trie1032();
                    }
                    node = node.children[index];
                }
                node.isEnd = true;
            }

            public boolean checkSuffix(String s) {
                Trie1032 node = this;
                for (int i = s.length() - 1; i >= 0; --i) {
                    int index = s.charAt(i) - 'a';
                    if (node.children[index] == null) {
                        return false;
                    }
                    node = node.children[index];
                    if (node.isEnd) {
                        return true;
                    }
                }
                return false;

            }

        }

        private Trie1032 trie;
        private StringBuilder builder;

        public StreamChecker(String[] words) {
            builder = new StringBuilder();
            trie = new Trie1032();
            for (String word : words) {
                trie.insert(word);
            }
        }

        public boolean query(char letter) {
            return trie.checkSuffix(builder.append(letter).toString());
        }
    }

    // 1819. 序列中不同最大公约数的数目 (Number of Different Subsequences GCDs)
    public int countDifferentSubsequenceGCDs(int[] nums) {
        int max = Arrays.stream(nums).max().getAsInt();
        boolean[] occour = new boolean[max + 1];
        for (int num : nums) {
            occour[num] = true;
        }
        int res = 0;
        for (int i = 1; i <= max; ++i) {
            int subGCD = 0;
            for (int j = i; j <= max; j += i) {
                if (occour[j]) {
                    if (subGCD == 0) {
                        subGCD = j;
                    } else {
                        subGCD = getGCD1839(subGCD, j);
                    }

                    if (subGCD == i) {
                        ++res;
                        break;
                    }
                }
            }
        }
        return res;

    }

    private int getGCD1839(int a, int b) {
        return b == 0 ? a : getGCD1839(b, a % b);
    }

    // 307. 区域和检索 - 数组可修改 (Range Sum Query - Mutable) --线段树
    class NumArray {
        private int[] segmentTree;
        private int[] nums;
        private int n;

        public NumArray(int[] nums) {
            this.n = nums.length;
            this.nums = nums;
            this.segmentTree = new int[Integer.highestOneBit(n << 2)];
            build(1, 1, n);
        }

        private void build(int o, int l, int r) {
            if (l == r) {
                segmentTree[o] = nums[l - 1];
                return;
            }
            int mid = l + ((r - l) >> 1);
            build(o * 2, l, mid);
            build(o * 2 + 1, mid + 1, r);
            segmentTree[o] = segmentTree[o * 2] + segmentTree[o * 2 + 1];
        }

        public void update(int index, int val) {
            // nums[index] = val;
            change(1, 1, n, index + 1, val);
        }

        private void change(int o, int l, int r, int index, int val) {
            if (l == r) {
                segmentTree[o] = val;
                return;
            }
            int mid = l + ((r - l) >> 1);
            if (index <= mid) {
                change(o * 2, l, mid, index, val);
            } else {
                change(o * 2 + 1, mid + 1, r, index, val);
            }
            segmentTree[o] = segmentTree[o * 2] + segmentTree[o * 2 + 1];
        }

        public int sumRange(int left, int right) {
            return query(1, 1, n, left + 1, right + 1);
        }

        private int query(int o, int l, int r, int L, int R) {
            if (L <= l && r <= R) {
                return segmentTree[o];
            }
            int mid = l + ((r - l) >> 1);
            if (R <= mid) {
                return query(o * 2, l, mid, L, R);
            }
            if (L >= mid + 1) {
                return query(o * 2 + 1, mid + 1, r, L, R);
            }
            return query(o * 2, l, mid, L, mid) + query(o * 2 + 1, mid + 1, r, mid + 1, R);
        }
    }

    // 307. 区域和检索 - 数组可修改 (Range Sum Query - Mutable) --动态开点线段树
    class NumArray2 {
        class SegNode {
            int lo;
            int hi;
            int sum;
            SegNode left;
            SegNode right;

            SegNode(int lo, int hi) {
                this.lo = lo;
                this.hi = hi;
            }
        }

        private SegNode root;
        private int[] nums;

        public NumArray2(int[] nums) {
            int n = nums.length;
            this.nums = nums;
            root = new SegNode(1, n);
            build(root, 1, n);
        }

        private void build(SegNode node, int l, int r) {
            if (l == r) {
                node.sum = nums[l - 1];
                return;
            }
            int mid = l + ((r - l) >> 1);
            if (node.left == null) {
                node.left = new SegNode(l, mid);
            }
            build(node.left, l, mid);
            if (node.right == null) {
                node.right = new SegNode(mid + 1, r);
            }
            build(node.right, mid + 1, r);
            node.sum = node.left.sum + node.right.sum;
        }

        public void update(int index, int val) {
            modify(root, index + 1, val);

        }

        private void modify(SegNode node, int index, int val) {
            if (node == null) {
                return;
            }
            if (node.lo == node.hi) {
                node.sum = val;
                return;
            }
            int mid = node.lo + ((node.hi - node.lo) >> 1);
            if (index <= mid) {
                modify(node.left, index, val);
            } else {
                modify(node.right, index, val);
            }
            node.sum = node.left.sum + node.right.sum;
        }

        public int sumRange(int left, int right) {
            return query(root, left + 1, right + 1);
        }

        private int query(SegNode node, int L, int R) {
            if (node == null) {
                return 0;
            }
            if (L > node.hi || R < node.lo) {
                return 0;
            }
            if (L <= node.lo && node.hi <= R) {
                return node.sum;
            }
            return query(node.left, L, R) + query(node.right, L, R);
        }
    }

    // 1649. 通过指令创建有序数组 (Create Sorted Array through Instructions) 二分 --还需掌握 树状数组
    public int createSortedArray(int[] instructions) {
        final int MOD = (int) (1e9 + 7);
        int res = 0;
        List<Integer> list = new ArrayList<>();
        for (int instruction : instructions) {
            int count1 = binarySearchLower1649(list, instruction);
            int count2 = binarySearchHigher1649(list, instruction);
            res = (res + Math.min(count1, count2)) % MOD;
        }
        return res;

    }

    private int binarySearchLower1649(List<Integer> list, int target) {
        int n = list.size();
        if (list.isEmpty() || target <= list.get(0)) {
            return 0;
        }
        if (target > list.get(n - 1)) {
            return n;
        }
        int left = 0;
        int right = n - 1;
        int count = 0;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (list.get(mid) < target) {
                count = mid + 1;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return count;
    }

    private int binarySearchHigher1649(List<Integer> list, int target) {
        int n = list.size();
        if (list.isEmpty() || target >= list.get(n - 1)) {
            list.add(target);
            return 0;
        }
        if (target < list.get(0)) {
            list.add(0, target);
            return n;
        }
        int left = 0;
        int right = n - 1;
        int count = 0;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (list.get(mid) > target) {
                count = n - mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        list.add(left, target);
        return count;
    }

    // 1649. 通过指令创建有序数组 (Create Sorted Array through Instructions) 线段树 --还需掌握 树状数组
    private int[] seg1649;

    public int createSortedArray2(int[] instructions) {
        int max = Integer.MIN_VALUE;
        for (int i : instructions) {
            max = Math.max(max, i);
        }
        final int MOD = (int) (1e9 + 7);
        int n = max + 1;
        int[] cnts = new int[n + 1];
        int res = 0;
        int nums = 0;
        this.seg1649 = new int[n * 4];
        for (int i : instructions) {
            ++i;
            int less = query1649(1, 1, n, 1, i - 1);
            // int more = query(1, 1, n, i + 1, n);
            int more = nums - cnts[i]++ - less;
            res = (res + Math.min(less, more)) % MOD;
            insert1649(1, 1, n, i);
            ++nums;
        }
        return res;
    }

    private void insert1649(int o, int l, int r, int id) {
        if (l == r) {
            ++seg1649[o];
            return;
        }
        int mid = l + ((r - l) >> 1);
        if (id <= mid) {
            insert1649(o * 2, l, mid, id);
        } else {
            insert1649(o * 2 + 1, mid + 1, r, id);
        }
        seg1649[o] = seg1649[o * 2] + seg1649[o * 2 + 1];
    }

    private int query1649(int o, int l, int r, int L, int R) {
        if (L <= l && r <= R) {
            return seg1649[o];
        }
        int mid = l + ((r - l) >> 1);
        int cnt = 0;
        if (L <= mid) {
            cnt += query1649(o * 2, l, mid, L, R);
        }
        if (R >= mid + 1) {
            cnt += query1649(o * 2 + 1, mid + 1, r, L, R);
        }
        return cnt;
    }

    // 493. 翻转对 (Reverse Pairs) --归并排序
    private int res493;

    public int reversePairs493(int[] nums) {
        int n = nums.length;
        if (n <= 1) {
            return 0;
        }
        dfs493(nums);

        return res493;
    }

    private int[] dfs493(int[] nums) {
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
        int[] res1 = dfs493(a);
        int[] res2 = dfs493(b);
        int i = 0;
        for (int x : res1) {
            while (i < res2.length && (long) x > res2[i] * 2l) {
                ++i;
            }
            res493 += i;
        }

        int j = 0;
        int k = 0;
        int index = 0;
        int[] res = new int[n];
        while (j < res1.length || k < res2.length) {
            if (j < res1.length && k < res2.length) {
                if (res1[j] < res2[k]) {
                    res[index++] = res1[j++];
                } else {
                    res[index++] = res2[k++];
                }
            } else if (j < res1.length) {
                res[index++] = res1[j++];
            } else {
                res[index++] = res2[k++];
            }
        }
        return res;
    }

    // 493. 翻转对 (Reverse Pairs) --离散化 线段树
    private int[] seg493;

    public int reversePairs493_2(int[] nums) {
        TreeSet<Long> set = new TreeSet<>();
        for (long num : nums) {
            set.add(num);
            set.add(num * 2);
        }
        int id = 1;
        Map<Long, Integer> map = new HashMap<>();
        for (long x : set) {
            map.put(x, id++);
        }
        this.seg493 = new int[id * 4];
        int res = 0;
        for (long x : nums) {
            res += count493(1, 1, id, map.get(x * 2) + 1, id);
            insert493(1, 1, id, map.get(x));
        }
        return res;

    }

    private int count493(int o, int l, int r, int L, int R) {
        if (L <= l && r <= R) {
            return seg493[o];
        }
        int res = 0;
        int mid = l + ((r - l) >> 1);
        if (L <= mid) {
            res += count493(o * 2, l, mid, L, R);
        }
        if (R >= mid + 1) {
            res += count493(o * 2 + 1, mid + 1, r, L, R);
        }
        return res;
    }

    private void insert493(int o, int l, int r, int x) {
        if (l == r) {
            ++seg493[o];
            return;
        }
        int mid = l + ((r - l) >> 1);
        if (x <= mid) {
            insert493(o * 2, l, mid, x);
        } else {
            insert493(o * 2 + 1, mid + 1, r, x);
        }
        seg493[o] = seg493[o * 2] + seg493[o * 2 + 1];
    }

    // 493. 翻转对 (Reverse Pairs) --动态开点的线段树
    public int reversePairs493_3(int[] nums) {
        long min = Long.MAX_VALUE;
        long max = Long.MIN_VALUE;
        for (long num : nums) {
            min = Math.min(min, num);
            min = Math.min(min, num * 2);
            max = Math.max(max, num);
            max = Math.max(max, num * 2);
        }
        SegNode493 root = new SegNode493(min, max);
        int res = 0;
        for (long x : nums) {
            res += count493(root, x * 2 + 1, max);
            insert493(root, x);
        }
        return res;

    }

    private int count493(SegNode493 root, long L, long R) {
        if (root == null) {
            return 0;
        }
        if (L > root.hi || R < root.lo) {
            return 0;
        }
        if (L <= root.lo && root.hi <= R) {
            return root.add;
        }
        return count493(root.left, L, R) + count493(root.right, L, R);
    }

    private void insert493(SegNode493 root, long x) {
        ++root.add;
        if (root.lo == root.hi) {
            return;
        }
        long mid = root.lo + ((root.hi - root.lo) >> 1);
        if (x <= mid) {
            if (root.left == null) {
                root.left = new SegNode493(root.lo, mid);
            }
            insert493(root.left, x);
        } else {
            if (root.right == null) {
                root.right = new SegNode493(mid + 1, root.hi);
            }
            insert493(root.right, x);
        }
    }

    public class SegNode493 {
        long lo;
        long hi;
        int add;
        SegNode493 left;
        SegNode493 right;

        public SegNode493(long lo, long hi) {
            this.lo = lo;
            this.hi = hi;
        }

    }

    // 6291. 数组元素和与数字和的绝对差
    public int differenceOfSum(int[] nums) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
            sum -= getSum(num);
        }
        return sum;

    }

    private int getSum(int num) {
        int sum = 0;
        while (num != 0) {
            sum += num % 10;
            num /= 10;
        }
        return sum;
    }

    // 6292. 子矩阵元素加 1 --二维差分
    public int[][] rangeAddQueries(int n, int[][] queries) {
        int[][] diff = new int[n + 2][n + 2];
        for (int[] q : queries) {
            int r1 = q[0];
            int c1 = q[1];
            int r2 = q[2];
            int c2 = q[3];
            ++diff[r1 + 1][c1 + 1];
            --diff[r1 + 1][c2 + 2];
            --diff[r2 + 2][c1 + 1];
            ++diff[r2 + 2][c2 + 2];
        }
        int[][] res = new int[n][n];
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                diff[i][j] += diff[i - 1][j] + diff[i][j - 1] - diff[i - 1][j - 1];
                res[i - 1][j - 1] = diff[i][j];
            }
        }
        return res;

    }

    // 2537. 统计好子数组的数目 (Count the Number of Good Subarrays)
    public long countGood(int[] nums, int k) {
        int n = nums.length;
        long res = 0L;
        int j = 0;
        long s = 0L;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            s += map.getOrDefault(nums[i], 0);
            map.merge(nums[i], 1, Integer::sum);
            while (s >= k) {
                res += n - i;
                map.merge(nums[j], -1, Integer::sum);
                s -= map.get(nums[j]);
                ++j;
            }
        }
        return res;

    }

    // 2538. 最大价值和与最小价值和的差值 (Difference Between Maximum and Minimum Price Sum)
    private Map<Integer, List<Integer>> map2538;
    private int[] price2538;
    private long res2538;

    public long maxOutput(int n, int[][] edges, int[] price) {
        map2538 = new HashMap<>();
        for (int[] edge : edges) {
            map2538.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(edge[1]);
            map2538.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(edge[0]);
        }
        this.price2538 = price;
        dfs2538(0, -1);
        return res2538;

    }

    private long[] dfs2538(int x, int fa) {
        long withLeafMax = price2538[x];
        long withoutLeafMax = 0l;
        for (int y : map2538.getOrDefault(x, new ArrayList<>())) {
            if (y != fa) {
                long[] cur = dfs2538(y, x);
                res2538 = Math.max(res2538, Math.max(withLeafMax + cur[1], withoutLeafMax + cur[0]));
                withLeafMax = Math.max(withLeafMax, cur[0] + price2538[x]);
                withoutLeafMax = Math.max(withoutLeafMax, cur[1] + price2538[x]);
            }
        }
        return new long[] { withLeafMax, withoutLeafMax };
    }

    // 1095. 山脉数组中查找目标值 (Find in Mountain Array)
    abstract class MountainArray {
        public int get(int index) {
            return 0;
        }

        public int length() {
            return 0;

        }
    }

    // 1095. 山脉数组中查找目标值 (Find in Mountain Array)
    private MountainArray mountainArray;

    public int findInMountainArray(int target, MountainArray mountainArr) {
        this.mountainArray = mountainArr;
        int maxIndex = binarySearchToGetMax(0, mountainArr.length() - 1);
        int res = binarySearchLeft1095(0, maxIndex, target);
        if (res != -1) {
            return res;
        }
        return binarySearchRight1095(maxIndex + 1, mountainArr.length() - 1, target);

    }

    private int binarySearchToGetMax(int left, int right) {
        while (left < right) {
            int mid = left + ((right - left) >> 1);
            if (mountainArray.get(mid) < mountainArray.get(mid + 1)) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return left;
    }

    private int binarySearchLeft1095(int left, int right, int target) {
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (mountainArray.get(mid) == target) {
                return mid;
            }
            if (mountainArray.get(mid) > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return -1;
    }

    private int binarySearchRight1095(int left, int right, int target) {
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (mountainArray.get(mid) == target) {
                return mid;
            }
            if (mountainArray.get(mid) > target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }

    // 853. 车队 (Car Fleet)
    public int carFleet(int target, int[] position, int[] speed) {
        int n = position.length;
        Car853[] cars = new Car853[n];
        for (int i = 0; i < n; ++i) {
            cars[i] = new Car853(position[i], (double) (target - position[i]) / speed[i]);
        }
        Arrays.sort(cars, new Comparator<Car853>() {

            @Override
            public int compare(Car853 o1, Car853 o2) {
                return Integer.compare(o1.pos, o2.pos);
            }

        });

        int res = 0;
        for (int i = n - 1; i > 0; --i) {
            if (cars[i].time < cars[i - 1].time) {
                ++res;
            } else {
                cars[i - 1] = cars[i];
            }
        }
        return res + 1;

    }

    public class Car853 {
        private int pos;
        private double time;

        public Car853(int pos, double time) {
            this.pos = pos;
            this.time = time;
        }

    }

    // 6300. 最小公共值
    public int getCommon(int[] nums1, int[] nums2) {
        int i = 0;
        int j = 0;
        int n1 = nums1.length;
        int n2 = nums2.length;
        while (i < n1 && j < n2) {
            if (nums1[i] == nums2[j]) {
                return nums1[i];
            }
            if (nums1[i] < nums2[j]) {
                ++i;
            } else {
                ++j;
            }
        }
        return -1;

    }

    // 6275. 使数组中所有元素相等的最小操作数 II
    public long minOperations(int[] nums1, int[] nums2, int k) {
        int n = nums1.length;
        if (k == 0) {
            return Arrays.equals(nums1, nums2) ? 0 : -1;
        }
        long pairs = 0;
        long bal = 0l;

        for (int i = 0; i < n; ++i) {
            if ((nums1[i] - nums2[i]) % k != 0) {
                return -1;
            }
            long cur = (nums1[i] - nums2[i]) / k;
            pairs += Math.abs(cur);
            bal += cur;

        }
        return bal == 0 ? pairs / 2 : -1;

    }

    // 6296. 交替数字和
    public int alternateDigitSum(int n) {
        int sign = 1;
        int sum = 0;
        while (n != 0) {
            int mod = n % 10;
            sum += sign * mod;
            sign = -sign;
            n /= 10;
        }
        return sum * -sign;

    }

    // 6297. 根据第 K 场考试的分数排序
    public int[][] sortTheStudents(int[][] score, int k) {
        Arrays.sort(score, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o2[k], o1[k]);
            }

        });
        return score;

    }

    // 6298. 执行逐位运算使字符串相等
    public boolean makeStringsEqual(String s, String target) {
        return s.contains("1") == target.contains("1");

    }

    // 2547. 拆分数组的最小代价 (Minimum Cost to Split an Array)
    public int minCost(int[] nums, int k) {
        int max = 0;
        for (int num : nums) {
            max = Math.max(max, num);
        }
        int n = nums.length;
        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int i = 1; i <= n; ++i) {
            int unique = 0;
            int[] counts = new int[max + 1];
            int min = Integer.MAX_VALUE;
            for (int j = i; j > 0; --j) {
                ++counts[nums[j - 1]];
                if (counts[nums[j - 1]] == 1) {
                    ++unique;
                } else if (counts[nums[j - 1]] == 2) {
                    --unique;
                }
                min = Math.min(min, dp[j - 1] + i - j + 1 - unique + k);
            }
            dp[i] = min;
        }
        return dp[n];

    }

    // 2547. 拆分数组的最小代价 (Minimum Cost to Split an Array)
    private int[] nums2547;
    private int k2547;
    private long[] memo2547;
    private int n2547;

    public int minCost2(int[] nums, int k) {
        this.nums2547 = nums;
        this.n2547 = nums.length;
        this.k2547 = k;
        this.memo2547 = new long[n2547];
        Arrays.fill(memo2547, -1L);
        return (int) dfs2547(0);

    }

    private long dfs2547(int i) {
        if (i == n2547) {
            return 0L;
        }
        if (memo2547[i] != -1L) {
            return memo2547[i];
        }
        long min = Long.MAX_VALUE;
        int unique = 0;
        int[] cnt = new int[n2547];
        for (int j = i; j < n2547; ++j) {
            ++cnt[nums2547[j]];
            if (cnt[nums2547[j]] == 1) {
                ++unique;
            } else if (cnt[nums2547[j]] == 2) {
                --unique;
            }
            min = Math.min(min, dfs2547(j + 1) + j - i + 1 - unique + k2547);
        }
        return memo2547[i] = min;
    }

    // 970. 强整数 (Powerful Integers)
    public List<Integer> powerfulIntegers(int x, int y, int bound) {
        Set<Integer> set = new HashSet<>();
        for (int i = 1; i <= bound; i *= x) {
            for (int j = 1; j <= bound; j *= y) {
                if (i + j <= bound) {
                    set.add(i + j);
                }
                if (y == 1) {
                    break;
                }
            }
            if (x == 1) {
                break;
            }
        }
        return new ArrayList<>(set);

    }

    // 2542. 最大子序列的分数 (Maximum Subsequence Score)
    public long maxScore(int[] nums1, int[] nums2, int k) {
        int n = nums1.length;
        int[][] pairs = new int[n][2];
        for (int i = 0; i < n; ++i) {
            pairs[i][0] = nums1[i];
            pairs[i][1] = nums2[i];
        }
        Arrays.sort(pairs, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o2[1], o1[1]);
            }

        });

        long res = 0l;
        long sum = 0l;
        Queue<Integer> q = new PriorityQueue<>();
        for (int[] pair : pairs) {
            sum += pair[0];
            q.offer(pair[0]);
            if (q.size() == k) {
                res = Math.max(res, sum * pair[1]);
            }
            while (q.size() >= k) {
                sum -= q.poll();
            }
        }
        return res;

    }

    // 995. K 连续位的最小翻转次数 (Minimum Number of K Consecutive Bit Flips) --差分 空间O(n)
    public int minKBitFlips(int[] nums, int k) {
        int n = nums.length;
        int[] diff = new int[n + 1];
        int res = 0;
        int revCnt = 0;
        for (int i = 0; i < n; ++i) {
            revCnt += diff[i];
            if (((revCnt + nums[i]) & 1) == 0) {
                if (i + k > n) {
                    return -1;
                }
                ++res;
                ++revCnt;
                --diff[i + k];
            }
        }
        return res;

    }

    // 995. K 连续位的最小翻转次数 (Minimum Number of K Consecutive Bit Flips) --差分 空间O(1)
    public int minKBitFlips2(int[] nums, int k) {
        int n = nums.length;
        int res = 0;
        int revCnt = 0;
        for (int i = 0; i < n; ++i) {
            if (i >= k && nums[i - k] > 1) {
                revCnt ^= 1;
                // 恢复原数组
                nums[i - k] -= 2;
            }
            if (nums[i] == revCnt) {
                if (i + k > n) {
                    return -1;
                }
                revCnt ^= 1;
                ++res;
                nums[i] += 2;
            }
        }
        return res;
    }

    // 780. 到达终点 (Reaching Points)
    public boolean reachingPoints(int sx, int sy, int tx, int ty) {
        while (tx > sx && ty > sy && tx != ty) {
            if (tx > ty) {
                tx %= ty;
            } else {
                ty %= tx;
            }
        }
        if (sx == tx && sy == ty) {
            return true;
        }

        if (sx == tx) {
            return ty > sy && (ty - sy) % sx == 0;
        }

        if (sy == ty) {
            return tx > sx && (tx - sx) % sy == 0;
        }
        return false;

    }

    // 6301. 判断一个点是否可以到达
    public boolean isReachable(int targetX, int targetY) {
        int g = getGCD6301(targetX, targetY);
        return Integer.bitCount(g) == 1;

    }

    private int getGCD6301(int a, int b) {
        return b == 0 ? a : getGCD6301(b, a % b);
    }

    // 1962. 移除石子使总数最小 (Remove Stones to Minimize the Total)
    public int minStoneSum(int[] piles, int k) {
        Queue<Integer> queue = new PriorityQueue<>(new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(o2, o1);
            }

        });
        int sum = 0;

        for (int pile : piles) {
            sum += pile;
            queue.offer(pile);
        }
        while (k-- > 0 && !queue.isEmpty()) {
            int num = queue.poll();
            sum -= num / 2;
            int added = (num + 1) / 2;
            if (added > 1) {
                queue.offer(added);
            }
        }
        return sum;

    }

    // 1975. 最大方阵和 (Maximum Matrix Sum)
    public long maxMatrixSum(int[][] matrix) {
        long res = 0l;
        int countNegative = 0;
        int min = Integer.MAX_VALUE;
        int n = matrix.length;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                min = Math.min(min, Math.abs(matrix[i][j]));
                if (matrix[i][j] < 0) {
                    ++countNegative;
                }
                res += Math.abs(matrix[i][j]);
            }
        }
        if (countNegative % 2 == 0) {
            return res;
        }
        return res - min * 2;
    }

    // 1834. 单线程 CPU (Single-Threaded CPU)
    public int[] getOrder(int[][] tasks) {
        int n = tasks.length;
        int[][] arr = new int[n][3];
        for (int i = 0; i < n; ++i) {
            arr[i][0] = tasks[i][0];
            arr[i][1] = tasks[i][1];
            arr[i][2] = i;
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

        Queue<int[]> queue = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[1] == o2[1]) {
                    return Integer.compare(o1[2], o2[2]);
                }
                return Integer.compare(o1[1], o2[1]);

            }

        });
        int[] res = new int[n];
        int i = 0;

        long time = 0l;
        int index = 0;
        queue.offer(arr[index++]);
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int enqueueTime = cur[0];
            int processingTime = cur[1];
            int id = cur[2];
            res[i++] = id;
            time = Math.max(time, enqueueTime) + processingTime;
            while (index < n && (arr[index][0] <= time || queue.isEmpty())) {
                queue.offer(arr[index++]);
            }
        }
        return res;

    }

    // 1443. 收集树上所有苹果的最少时间 (Minimum Time to Collect All Apples in a Tree)
    private int res1443;
    private Map<Integer, List<Integer>> graph1443;
    private List<Boolean> hasApple1443;

    public int minTime(int n, int[][] edges, List<Boolean> hasApple) {
        graph1443 = new HashMap<>();
        for (int[] edge : edges) {
            graph1443.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(edge[1]);
            graph1443.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(edge[0]);
        }
        this.hasApple1443 = hasApple;
        dfs1443(0, -1);
        return res1443;

    }

    private boolean dfs1443(int x, int fa) {
        boolean b = hasApple1443.get(x);
        for (int y : graph1443.getOrDefault(x, new ArrayList<>())) {
            if (y != fa) {
                if (dfs1443(y, x)) {
                    res1443 += 2;
                    b = true;
                }
            }
        }
        return b;
    }

    // 1530. 好叶子节点对的数量 (Number of Good Leaf Nodes Pairs)
    private int res1530;
    private int distance1530;

    public int countPairs(TreeNode root, int distance) {
        distance1530 = distance;
        dfs1530(root);
        return res1530;

    }

    private List<Integer> dfs1530(TreeNode node) {
        List<Integer> list = new ArrayList<>();
        if (node == null) {
            return list;
        }
        if (node.left == null && node.right == null) {
            list.add(1);
            return list;
        }
        List<Integer> left = dfs1530(node.left);
        List<Integer> right = dfs1530(node.right);
        for (int i : left) {
            if (i < distance1530) {
                for (int j : right) {
                    if (j < distance1530) {
                        if (i + j <= distance1530) {
                            ++res1530;
                        }
                    }
                }
            }
        }
        for (int i : left) {
            if (i + 1 < distance1530) {
                list.add(i + 1);
            }
        }
        for (int i : right) {
            if (i + 1 < distance1530) {
                list.add(i + 1);
            }

        }
        return list;

    }

    // 1145. 二叉树着色游戏 (Binary Tree Coloring Game)
    private int leftCnt1145;
    private int rightCnt1145;

    public boolean btreeGameWinningMove(TreeNode root, int n, int x) {
        dfs1145(root, x);
        int cntParent = n - leftCnt1145 - rightCnt1145 - 1;
        return leftCnt1145 * 2 > n || rightCnt1145 * 2 > n || cntParent * 2 > n;
    }

    private int dfs1145(TreeNode root, int x) {
        if (root == null) {
            return 0;
        }
        int left = dfs1145(root.left, x);
        int right = dfs1145(root.right, x);
        if (root.val == x) {
            leftCnt1145 = left;
            rightCnt1145 = right;
        }
        return left + right + 1;
    }

    // 1477. 找两个和为目标值且不重叠的子数组 (Find Two Non-overlapping Sub-arrays Each With Target
    // Sum)
    private Map<Integer, List<int[]>> map1477;
    private int[][] memo1477;
    private int n1477;

    public int minSumOfLengths(int[] arr, int target) {
        this.n1477 = arr.length;
        int i = 0;
        int j = 0;
        int sum = 0;
        this.map1477 = new HashMap<>();
        while (j < n1477) {
            sum += arr[j];
            while (sum > target) {
                sum -= arr[i++];
            }
            if (sum == target) {
                map1477.computeIfAbsent(i, k -> new ArrayList<>()).add(new int[] { i, j });
            }
            ++j;
        }
        this.memo1477 = new int[n1477][2];
        for (int k = 0; k < n1477; ++k) {
            Arrays.fill(memo1477[k], -1);
        }
        int res = dfs1477(0, 0);
        return res <= n1477 ? res : -1;
    }

    private int dfs1477(int i, int j) {
        if (j == 2) {
            return 0;
        }
        if (i == n1477) {
            return n1477 + 1;
        }
        if (memo1477[i][j] != -1) {
            return memo1477[i][j];
        }
        int res = dfs1477(i + 1, j);
        for (int[] item : map1477.getOrDefault(i, new ArrayList<>())) {
            res = Math.min(res, dfs1477(item[1] + 1, j + 1) + item[1] - item[0] + 1);
        }
        return memo1477[i][j] = res;
    }

    // 1372. 二叉树中的最长交错路径 (Longest ZigZag Path in a Binary Tree) --先序遍历
    private int res1372;

    public int longestZigZag(TreeNode root) {
        dfs1372(root, 0, 0);
        return res1372;

    }

    private void dfs1372(TreeNode root, int l, int r) {
        if (root == null) {
            return;
        }
        res1372 = Math.max(res1372, Math.max(l, r));
        dfs1372(root.left, r + 1, 0);
        dfs1372(root.right, 0, l + 1);

    }

    // 1774. 最接近目标价格的甜点成本 (Closest Dessert Cost) --二进制枚举
    public int closestCost(int[] baseCosts, int[] toppingCosts, int target) {
        int n = toppingCosts.length;
        int[] topCost = new int[n * 2];
        for (int i = 0; i < n; ++i) {
            topCost[i] = toppingCosts[i];
            topCost[i + n] = toppingCosts[i];
        }
        Set<Integer> set = getTopCosts1774(topCost);
        int res = Integer.MAX_VALUE;
        int diff = Integer.MAX_VALUE;
        for (int top : set) {
            for (int base : baseCosts) {
                int sum = base + top;
                if (Math.abs(sum - target) < diff) {
                    diff = Math.abs(sum - target);
                    res = sum;
                } else if (Math.abs(sum - target) == diff && sum < res) {
                    res = sum;
                }
                if (diff == 0) {
                    return target;
                }
            }
        }
        return res;

    }

    private Set<Integer> getTopCosts1774(int[] topCost) {
        Set<Integer> set = new HashSet<>();
        int n = topCost.length;
        for (int i = 0; i < (1 << n); ++i) {
            int mask = i;
            int index = 0;
            int sum = 0;
            while (mask != 0) {
                if ((mask & 1) == 1) {
                    sum += topCost[index];
                }
                mask >>= 1;
                ++index;
            }
            set.add(sum);

        }
        return set;
    }

    // 1774. 最接近目标价格的甜点成本 (Closest Dessert Cost) --三进制枚举
    public int closestCost2(int[] baseCosts, int[] toppingCosts, int target) {
        Set<Integer> topCost = getTopCosts1774_2(toppingCosts);
        int res = Integer.MAX_VALUE;
        int diff = Integer.MAX_VALUE;
        for (int base : baseCosts) {
            for (int top : topCost) {
                int sum = base + top;
                if (Math.abs(sum - target) < diff) {
                    diff = Math.abs(sum - target);
                    res = sum;
                } else if (Math.abs(sum - target) == diff && sum < res) {
                    res = sum;
                }
                if (diff == 0) {
                    return target;
                }
            }
        }
        return res;

    }

    private Set<Integer> getTopCosts1774_2(int[] toppingCosts) {
        int n = toppingCosts.length;
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < Math.pow(3, n); ++i) {
            int sum = 0;
            int mask = i;
            int index = 0;
            while (mask != 0) {
                sum += toppingCosts[index] * (mask % 3);
                mask /= 3;
                ++index;
            }
            set.add(sum);
        }
        return set;
    }

    // 1986. 完成任务的最少工作时间段 (Minimum Number of Work Sessions to Finish the Tasks)
    // --状态压缩dp
    private int sessionTime1986;
    private int n1986;
    private int[] memo1986;
    private int[] arr1986;
    private int u1986;

    public int minSessions(int[] tasks, int sessionTime) {
        this.sessionTime1986 = sessionTime;
        this.n1986 = tasks.length;
        this.memo1986 = new int[1 << n1986];
        this.arr1986 = new int[1 << n1986];
        for (int i = 1; i < 1 << n1986; ++i) {
            int index = Integer.numberOfTrailingZeros(i);
            arr1986[i] = arr1986[i ^ (1 << index)] + tasks[index];
        }
        Arrays.fill(memo1986, -1);
        this.u1986 = (1 << n1986) - 1;
        return dfs1986(0);

    }

    private int dfs1986(int i) {
        if (i == u1986) {
            return 0;
        }
        if (memo1986[i] != -1) {
            return memo1986[i];
        }
        int res = n1986 + 1;
        int c = u1986 ^ i;
        for (int j = c; j > 0; j = (j - 1) & c) {
            if (arr1986[j] <= sessionTime1986) {
                res = Math.min(res, dfs1986(i | j) + 1);
            }
        }
        return memo1986[i] = res;
    }

    /**
     * You are given a string
     * S of length
     * N consisting of A and B.
     * 
     * You can repeat the following operation zero or more times:
     * 
     * choose a pair of adjacent characters in
     * S and replace them with AB.
     * Determine whether
     * S can be turned into a palindrome.
     */
    public boolean isValid(String s) {
        if ("BA".equals(s) || s.charAt(0) == 'A' && s.charAt(s.length() - 1) == 'B') {
            return false;
        }
        return true;
    }

    // 2106. 摘水果 (Maximum Fruits Harvested After at Most K Steps) --二分查找
    public int maxTotalFruits(int[][] fruits, int startPos, int k) {
        int sum = 0;
        for (int[] fruit : fruits) {
            sum += fruit[1];
        }
        int res = 0;
        int left = 0;
        int right = sum;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (check2106(fruits, startPos, mid) <= k) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;

    }

    // 从startPos出发，摘至少target个草莓，至少需要走多少步？
    private int check2106(int[][] fruits, int startPos, int target) {
        int minStep = Integer.MAX_VALUE;
        int n = fruits.length;
        int sum = 0;
        int i = 0;
        int j = 0;
        while (j < n) {
            sum += fruits[j][1];
            while (i <= j && sum >= target) {
                minStep = Math.min(minStep, fruits[j][0] - fruits[i][0]
                        + Math.min(Math.abs(startPos - fruits[i][0]), Math.abs(fruits[j][0] - startPos)));
                sum -= fruits[i++][1];
            }
            ++j;
        }
        return minStep;
    }

    // 2106. 摘水果 (Maximum Fruits Harvested After at Most K Steps)
    public int maxTotalFruits2(int[][] fruits, int startPos, int k) {
        int n = fruits.length;
        int res = 0;
        int s = 0;
        int j = 0;
        for (int i = 0; i < n; ++i) {
            s += fruits[i][1];
            while (j <= i && fruits[i][0] - fruits[j][0]
                    + Math.min(Math.abs(startPos - fruits[i][0]), Math.abs(startPos - fruits[j][0])) > k) {
                s -= fruits[j++][1];
            }
            res = Math.max(res, s);
        }
        return res;
    }

    // 2267. 检查是否有合法括号字符串路径 (Check if There Is a Valid Parentheses String Path)
    public boolean hasValidPath(char[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int len = m + n - 1;
        if (len % 2 == 1) {
            return false;
        }
        if (grid[0][0] != '(') {
            return false;
        }
        if (grid[m - 1][n - 1] != ')') {
            return false;
        }
        Queue<int[]> queue = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[2], o2[2]);
            }
        });
        int[][] directions = { { 1, 0 }, { 0, 1 } };

        boolean[][][] vis = new boolean[m][n][(m + n - 1) / 2 + 1];

        // {i, j, 左括号数量 - 右括号数量}
        queue.offer(new int[] { 0, 0, 1 });
        vis[0][0][1] = true;
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0];
            int y = cur[1];
            int diff = cur[2];
            if (diff < 0) {
                continue;
            }
            if (diff > m - 1 - x + n - 1 - y) {
                continue;
            }
            if (x == m - 1 && y == n - 1 && diff == 0) {
                return true;
            }
            for (int[] d : directions) {
                int nx = x + d[0];
                int ny = y + d[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    if (grid[nx][ny] == '(' && diff + 1 <= (m + n - 1) / 2 && !vis[nx][ny][diff + 1]) {
                        vis[nx][ny][diff + 1] = true;
                        queue.offer(new int[] { nx, ny, diff + 1 });
                    } else if (grid[nx][ny] == ')' && diff - 1 >= 0 && !vis[nx][ny][diff - 1]) {
                        vis[nx][ny][diff - 1] = true;
                        queue.offer(new int[] { nx, ny, diff - 1 });
                    }
                }
            }
        }
        return false;

    }

    // 2267. 检查是否有合法括号字符串路径 (Check if There Is a Valid Parentheses String Path)
    private int m2267;
    private int n2267;
    private char[][] grid2267;
    private int[][][] memo2267;

    public boolean hasValidPath2(char[][] grid) {
        this.m2267 = grid.length;
        this.n2267 = grid[0].length;
        this.grid2267 = grid;
        if ((m2267 + n2267) % 2 == 0 || grid[0][0] == ')' || grid[m2267 - 1][n2267 - 1] == '(') {
            return false;
        }
        this.memo2267 = new int[m2267][n2267][(m2267 + n2267 - 1) / 2 + 1];
        return dfs2267(0, 0, 0);

    }

    private boolean dfs2267(int i, int j, int k) {
        if (i == m2267 - 1 && j == n2267 - 1) {
            return k == 1;
        }
        if (i == m2267 || j == n2267 || k < 0 || m2267 - i + n2267 - j - 1 < k) {
            return false;
        }
        if (memo2267[i][j][k] != 0) {
            return memo2267[i][j][k] > 0;
        }
        boolean res = dfs2267(i + 1, j, k + (grid2267[i][j] == '(' ? 1 : -1))
                || dfs2267(i, j + 1, k + (grid2267[i][j] == '(' ? 1 : -1));
        memo2267[i][j][k] = res ? 1 : -1;
        return res;
    }

    // 1801. 积压订单中的订单总数 (Number of Orders in the Backlog)
    public int getNumberOfBacklogOrders(int[][] orders) {
        Queue<int[]> sell = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });

        Queue<int[]> buy = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o2[0], o1[0]);
            }

        });

        for (int[] order : orders) {
            int price = order[0];
            int amount = order[1];
            int type = order[2];
            if (type == 0) {
                while (!sell.isEmpty() && sell.peek()[0] <= price && amount != 0) {
                    int minus = Math.min(sell.peek()[1], amount);
                    sell.peek()[1] -= minus;
                    amount -= minus;
                    if (sell.peek()[1] == 0) {
                        sell.poll();
                    }
                }
                if (amount != 0) {
                    buy.offer(new int[] { price, amount });
                }
            } else {
                while (!buy.isEmpty() && buy.peek()[0] >= price && amount != 0) {
                    int minus = Math.min(buy.peek()[1], amount);
                    buy.peek()[1] -= minus;
                    amount -= minus;
                    if (buy.peek()[1] == 0) {
                        buy.poll();
                    }
                }
                if (amount != 0) {
                    sell.offer(new int[] { price, amount });
                }
            }
        }

        long res = 0l;
        final int MOD = (int) (1e9 + 7);
        while (!sell.isEmpty()) {
            res = (res + sell.poll()[1]) % MOD;
        }
        while (!buy.isEmpty()) {
            res = (res + buy.poll()[1]) % MOD;
        }
        return (int) res;

    }

    // 2549. 统计桌面上的不同数字 (Count Distinct Numbers on Board)
    public int distinctIntegers(int n) {
        return Math.max(1, n - 1);
    };

    // 2550. 猴子碰撞的方法数 (Count Collisions of Monkeys on a Polygon)
    public int monkeyMove(int n) {
        final int MOD = (int) (1e9 + 7);
        return ((dfs2550(n) - 2) % MOD + MOD) % MOD;
    }

    private int dfs2550(int n) {
        if (n == 0) {
            return 1;
        }
        int res = dfs2550(n / 2);
        final int MOD = (int) (1e9 + 7);
        res = (int) (((long) res * res) % MOD);
        if ((n & 1) == 1) {
            res = res * 2 % MOD;
        }
        return res;
    }

    // 2551. 将珠子放入背包中 (Put Marbles in Bags)
    public long putMarbles(int[] weights, int k) {
        int n = weights.length;
        for (int i = 0; i < n - 1; ++i) {
            weights[i] += weights[i + 1];
        }
        Arrays.sort(weights, 0, n - 1);
        long res = 0l;
        for (int i = 0; i < k - 1; ++i) {
            res += weights[n - i - 2] - weights[i];
        }
        return res;

    }

    // 1163. 按字典序排在最后的子串 (Last Substring in Lexicographical Order)
    public String lastSubstring(String s) {
        int n = s.length();
        char max = 'a';
        int pos = n - 1;
        for (int i = n - 1; i >= 0; --i) {
            if (s.charAt(i) >= max) {
                max = s.charAt(i);
                pos = i;
            }
        }
        int left = pos;
        int count = 1;
        while (left + count < n) {
            if (s.charAt(left + count) < max) {
                ++count;
            } else {
                int i = left;
                int j = left + count;
                while (j < n) {
                    if (s.charAt(i) == s.charAt(j)) {
                        ++i;
                        ++j;
                    } else {
                        if (s.charAt(i) < s.charAt(j)) {
                            left += count;
                            count = 1;
                        } else {
                            ++count;
                        }
                        break;
                    }
                }
                if (j == n) {
                    break;
                }

            }
        }
        return s.substring(left);

    }

    // 1562. 查找大小为 M 的最新分组 (Find Latest Group of Size M)
    public int findLatestStep(int[] arr, int m) {
        int n = arr.length;
        int[][] ends = new int[n + 2][2];
        for (int i = 0; i < n + 2; ++i) {
            ends[i][0] = -1;
            ends[i][1] = -1;
        }
        int res = -1;
        Map<Integer, Integer> counts = new HashMap<>();
        int step = 0;
        for (int pos : arr) {
            ++step;
            if (ends[pos - 1][0] != -1 && ends[pos + 1][0] != -1) {
                int len1 = ends[pos - 1][1] - ends[pos - 1][0] + 1;
                int len2 = ends[pos + 1][1] - ends[pos + 1][0] + 1;
                ends[ends[pos - 1][0]][1] = ends[pos + 1][1];
                ends[ends[pos + 1][1]][0] = ends[pos - 1][0];
                counts.put(len1, counts.get(len1) - 1);
                counts.put(len2, counts.get(len2) - 1);
                counts.put(len1 + len2 + 1, counts.getOrDefault(len1 + len2 + 1, 0) + 1);
            } else if (ends[pos - 1][0] != -1) {
                int len = ends[pos - 1][1] - ends[pos - 1][0] + 1;
                ends[ends[pos - 1][0]][1] = pos;
                ends[pos][0] = ends[ends[pos - 1][0]][0];
                ends[pos][1] = pos;
                counts.put(len, counts.get(len) - 1);
                counts.put(len + 1, counts.getOrDefault(len + 1, 0) + 1);
            } else if (ends[pos + 1][0] != -1) {
                int len = ends[pos + 1][1] - ends[pos + 1][0] + 1;
                ends[ends[pos + 1][1]][0] = pos;
                ends[pos][0] = pos;
                ends[pos][1] = ends[ends[pos + 1][1]][1];
                counts.put(len, counts.get(len) - 1);
                counts.put(len + 1, counts.getOrDefault(len + 1, 0) + 1);
            } else {
                ends[pos][0] = pos;
                ends[pos][1] = pos;
                counts.put(1, counts.getOrDefault(1, 0) + 1);
            }
            if (counts.getOrDefault(m, 0) > 0) {
                res = step;
            }
        }
        return res;

    }

    // 1648. 销售价值减少的颜色球 (Sell Diminishing-Valued Colored Balls) --优先队列
    public int maxProfit(int[] inventory, int orders) {
        TreeMap<Integer, Integer> map = new TreeMap<>(new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(o2, o1);
            }

        });
        for (int i : inventory) {
            map.put(i, map.getOrDefault(i, 0) + 1);
        }
        final int MOD = (int) (1e9 + 7);
        long res = 0l;
        while (true) {
            Map.Entry<Integer, Integer> entry = map.pollFirstEntry();
            int val = entry.getKey();
            int count = entry.getValue();
            if (map.isEmpty() || ((long) (val - map.firstKey())) * count >= orders) {
                int d = orders / count;
                int m = orders % count;
                res = (res + ((long) (val - d + 1 + val)) * d / 2 * (long) count) % MOD;
                res = (res + ((long) (val - d)) * m) % MOD;
                return (int) res;
            } else {
                int nextVal = map.firstKey();
                res = (res + ((long) (nextVal + 1 + val)) * (val - nextVal) / 2 * (long) count) % MOD;
                orders -= (val - nextVal) * count;
                map.put(nextVal, map.getOrDefault(nextVal, 0) + count);
            }
        }
    }

    // 1648. 销售价值减少的颜色球 (Sell Diminishing-Valued Colored Balls) --二分查找
    public int maxProfit2(int[] inventory, int orders) {
        final int MOD = (int) (1e9 + 7);
        int T = -1;
        int left = 0;
        int right = 0;
        long rest = 0;
        for (int i : inventory) {
            right = Math.max(right, i);
        }
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            long curOrders = check1648(inventory, mid);
            if (curOrders <= orders) {
                T = mid;
                right = mid - 1;
                rest = orders - curOrders;
            } else {
                left = mid + 1;
            }
        }
        long res = 0l;
        for (int i : inventory) {
            if (i >= T) {
                if (rest-- > 0) {
                    res = (res + getSum1648(T, i)) % MOD;
                } else {
                    res = (res + getSum1648(T + 1, i)) % MOD;
                }
            }
        }
        return (int) (res % MOD);

    }

    private long getSum1648(int low, int high) {
        return ((long) (low + high)) * (high - low + 1) / 2;
    }

    private long check1648(int[] inventory, int target) {
        long sum = 0l;
        for (int i : inventory) {
            sum += Math.max(i - target, 0l);
        }
        return sum;
    }

    // 1192. 查找集群内的关键连接 (Critical Connections in a Network)
    private Map<Integer, List<Integer>> graph1192;
    private int time1192;
    private int[] visited1192;
    private List<List<Integer>> res1192;

    public List<List<Integer>> criticalConnections(int n, List<List<Integer>> connections) {
        graph1192 = new HashMap<>();
        for (List<Integer> connection : connections) {
            int a = connection.get(0);
            int b = connection.get(1);
            graph1192.computeIfAbsent(a, k -> new ArrayList<>()).add(b);
            graph1192.computeIfAbsent(b, k -> new ArrayList<>()).add(a);
        }
        res1192 = new ArrayList<>();
        visited1192 = new int[n];
        Arrays.fill(visited1192, -1);
        dfs1192(0, -1);
        return res1192;

    }

    private int dfs1192(int x, int fa) {
        if (visited1192[x] != -1) {
            return visited1192[x];
        }
        int curTime = time1192++;
        visited1192[x] = curTime;
        for (int y : graph1192.getOrDefault(x, new ArrayList<>())) {
            if (y != fa) {
                int nextTime = dfs1192(y, x);
                visited1192[x] = Math.min(visited1192[x], nextTime);
                if (curTime < nextTime) {
                    res1192.add(List.of(x, y));
                }
            }
        }
        return visited1192[x];
    }

    // 2553. 分割数组中数字的数位 (Separate the Digits in an Array)
    public int[] separateDigits(int[] nums) {
        List<Integer> list = new ArrayList<>();
        for (int num : nums) {
            int index = list.size();
            while (num != 0) {
                list.add(index, num % 10);
                num /= 10;
            }
        }
        return list.stream().mapToInt(Integer::intValue).toArray();

    }

    // 2554. 从一个范围内选择最多整数 I (Maximum Number of Integers to Choose From a Range I)
    public int maxCount(int[] banned, int n, int maxSum) {
        long curSum = 0l;
        Set<Integer> set = new HashSet<>();
        for (int b : banned) {
            set.add(b);
        }
        int res = 0;
        for (int i = 1; i <= n; ++i) {
            if (set.contains(i)) {
                continue;
            }
            if (curSum + i <= maxSum) {
                curSum += i;
                ++res;
            } else {
                break;
            }
        }
        return res;

    }

    // 2558. 从数量最多的堆取走礼物 (Take Gifts From the Richest Pile)
    public long pickGifts(int[] gifts, int k) {
        Queue<Integer> q = new PriorityQueue<>(new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(o2, o1);
            }

        });
        for (int g : gifts) {
            q.offer(g);
        }
        while (k-- > 0 && q.peek() > 1) {
            int cur = q.poll();
            cur = (int) Math.sqrt(cur);
            q.offer(cur);
        }
        long res = 0l;
        while (!q.isEmpty()) {
            res += q.poll();
        }
        return res;

    }

    // 2559. 统计范围内的元音字符串数 (Count Vowel Strings in Ranges)
    public int[] vowelStrings(String[] words, int[][] queries) {
        int n = words.length;
        int[] pre = new int[n + 1];
        for (int i = 0; i < n; ++i) {
            pre[i + 1] = pre[i] + (checkLegal2559(words[i]) ? 1 : 0);
        }
        int[] res = new int[queries.length];
        for (int i = 0; i < queries.length; ++i) {
            res[i] = pre[queries[i][1] + 1] - pre[queries[i][0]];
        }
        return res;

    }

    private boolean checkLegal2559(String s) {
        return checkVowel2559(s.charAt(0)) && checkVowel2559(s.charAt(s.length() - 1));
    }

    private boolean checkVowel2559(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }

    // 2560. 打家劫舍 IV (House Robber IV)
    public int minCapability(int[] nums, int k) {
        int left = Arrays.stream(nums).min().getAsInt();
        int right = Arrays.stream(nums).max().getAsInt();
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (check6346(nums, mid) >= k) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;

    }

    private int check6346(int[] nums, int target) {
        int res = 0;
        int count = 0;
        for (int num : nums) {
            if (num <= target) {
                ++count;
            } else {
                res += (count + 1) / 2;
                count = 0;
            }
        }
        res += (count + 1) / 2;
        return res;
    }

    // 2561. 重排水果 (Rearranging Fruits)
    public long minCost(int[] basket1, int[] basket2) {
        long res = 0l;
        int n = basket1.length;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            map.merge(basket1[i], 1, Integer::sum);
            map.merge(basket2[i], -1, Integer::sum);
        }
        int min = Integer.MAX_VALUE;
        List<Integer> list = new ArrayList<>();
        for (Map.Entry<Integer, Integer> e : map.entrySet()) {
            min = Math.min(e.getKey(), min);
            if ((e.getValue() & 1) != 0) {
                return -1l;
            }
            for (int c = Math.abs(e.getValue() / 2); c > 0; --c) {
                list.add(e.getKey());
            }
        }
        Collections.sort(list);
        for (int i = 0; i < list.size() / 2; ++i) {
            res += Math.min(min * 2, list.get(i));
        }
        return res;

    }

    // 2556. 二进制矩阵中翻转最多一次使路径不连通 (Disconnect Path in a Binary Matrix by at Most One
    // Flip)
    private int[][] grid2556;

    public boolean isPossibleToCutPath(int[][] grid) {
        this.grid2556 = grid;
        return !dfs2556(0, 0) || !dfs2556(0, 0);

    }

    private boolean dfs2556(int i, int j) {
        int m = grid2556.length;
        int n = grid2556[0].length;
        if (m - 1 == i && n - 1 == j) {
            return true;
        }
        grid2556[i][j] = 0;
        return i + 1 < m && grid2556[i + 1][j] == 1 && dfs2556(i + 1, j)
                || j + 1 < n && grid2556[i][j + 1] == 1 && dfs2556(i, j + 1);
    }

    // 2555. 两个线段获得的最多奖品 (Maximize Win From Two Segments)
    public int maximizeWin(int[] prizePositions, int k) {
        int n = prizePositions.length;
        int[] pre = new int[n + 1];
        int left = 0;
        int right = 0;
        int res = 0;
        while (right < n) {
            while (prizePositions[right] - prizePositions[left] > k) {
                ++left;
            }
            res = Math.max(res, right - left + 1 + pre[left]);
            pre[right + 1] = Math.max(pre[right], right - left + 1);
            ++right;
        }
        return res;

    }

    // 1545. 找出第 N 个二进制字符串中的第 K 位 (Find Kth Bit in Nth Binary String)
    public char findKthBit(int n, int k) {
        if (k == 1) {
            return '0';
        }
        int mid = 1 << (n - 1);
        if (mid == k) {
            return '1';
        } else if (k < mid) {
            return findKthBit(n - 1, k);
        } else {
            k = (mid << 1) - k;
            return invert1545(findKthBit(n - 1, k));
        }

    }

    private char invert1545(char c) {
        return (char) (((c - '0') ^ 1) + '0');
    }

    // 2111. 使数组 K 递增的最少操作次数 (Minimum Operations to Make the Array K-Increasing)
    public int kIncreasing(int[] arr, int k) {
        int res = 0;
        for (int i = 0; i < k; ++i) {
            List<Integer> list = new ArrayList<>();
            int j = i;
            while (j < arr.length) {
                list.add(arr[j]);
                j += k;
            }
            int n = list.size();
            int[] dp = new int[n + 1];
            int len = 1;
            dp[len] = list.get(0);
            for (int x = 1; x < n; ++x) {
                if (list.get(x) >= dp[len]) {
                    dp[++len] = list.get(x);
                } else {
                    int left = 1;
                    int right = len;
                    int pos = 0;
                    while (left <= right) {
                        int mid = left + ((right - left) >>> 1);
                        if (dp[mid] <= list.get(x)) {
                            pos = mid;
                            left = mid + 1;
                        } else {
                            right = mid - 1;
                        }
                    }
                    dp[pos + 1] = list.get(x);
                }
            }
            res += len;
        }
        return arr.length - res;

    }

    // 2008. 出租车的最大盈利 (Maximum Earnings From Taxi)
    public long maxTaxiEarnings(int n, int[][] rides) {
        long[] dp = new long[n + 1];
        Map<Integer, List<int[]>> map = new HashMap<>();
        for (int[] r : rides) {
            map.computeIfAbsent(r[1], k -> new ArrayList<>()).add(new int[] { r[0], r[2] });
        }
        long res = 0l;
        for (int i = 1; i < n + 1; ++i) {
            dp[i] = dp[i - 1];
            for (int[] r : map.getOrDefault(i, new ArrayList<>())) {
                int start = r[0];
                int tips = r[1];
                dp[i] = Math.max(dp[i], dp[start] + i - start + tips);
            }
            res = Math.max(res, dp[i]);
        }
        return res;

    }

    // 2008. 出租车的最大盈利 (Maximum Earnings From Taxi)
    private List<int[]>[] g2008;
    private long[] memo2008;
    private int n2008;

    public long maxTaxiEarnings2(int n, int[][] rides) {
        this.n2008 = n;
        this.g2008 = new ArrayList[n];
        Arrays.setAll(g2008, k -> new ArrayList<>());
        for (int[] r : rides) {
            g2008[r[0] - 1].add(new int[] { r[1] - 1, r[2] });
        }
        this.memo2008 = new long[n];
        Arrays.fill(memo2008, -1L);
        return dfs2008(0);
    }

    private long dfs2008(int i) {
        if (i == n2008) {
            return 0;
        }
        if (memo2008[i] != -1L) {
            return memo2008[i];
        }
        long res = dfs2008(i + 1);
        for (int[] nxt : g2008[i]) {
            res = Math.max(res, dfs2008(nxt[0]) + nxt[0] - i + nxt[1]);
        }
        return memo2008[i] = res;
    }

    // 2009. 使数组连续的最少操作数 (Minimum Number of Operations to Make Array Continuous)
    public int minOperations(int[] nums) {
        int n = nums.length;
        List<Integer> list = new ArrayList<>(Arrays.stream(nums).boxed().collect(Collectors.toSet()));
        Collections.sort(list);
        int res = 0;
        int j = 0;
        for (int i = 0; i < list.size(); ++i) {
            while (list.get(j) < list.get(i) - n + 1) {
                ++j;
            }
            res = Math.max(res, i - j + 1);
        }
        return n - res;
    }

    // 2156. 查找给定哈希值的子串 (Find Substring With Given Hash Value)
    public String subStrHash(String s, int power, int modulo, int k, int hashValue) {
        int n = s.length();
        int[] pre = new int[k];
        pre[0] = 1;
        for (int i = 1; i < k; ++i) {
            pre[i] = (int) ((long) pre[i - 1] * power % modulo);
        }
        int h = 0;
        int res = n;
        for (int i = n - k; i < n; ++i) {
            h = (int) ((h + (long) (s.charAt(i) - 'a' + 1) * pre[i - n + k] % modulo) % modulo);
        }
        if (h == hashValue) {
            res = n - k;
        }
        for (int i = n - k - 1; i >= 0; --i) {
            h = (int) (((h - (long) (s.charAt(i + k) - 'a' + 1) * pre[k - 1] % modulo) % modulo + modulo) % modulo);
            h = (int) ((long) h * power % modulo);
            h = (int) ((h + (long) s.charAt(i) - 'a' + 1) % modulo);
            if (h == hashValue) {
                res = i;
            }
        }
        return s.substring(res, res + k);
    }

    // 2156. 查找给定哈希值的子串 (Find Substring With Given Hash Value)
    // --秦九韶算法、反向Rabin-Karp(滚动哈希)
    public String subStrHash2(String s, int power, int modulo, int k, int hashValue) {
        long hash = 0l;
        long mult = 1l;
        int pos = -1;
        int n = s.length();
        for (int i = n - 1; i >= n - k; --i) {
            hash = (hash * power + (s.charAt(i) & 31)) % modulo;
            if (i != n - k) {
                mult = (mult * power) % modulo;
            }
        }
        if (hash == hashValue) {
            pos = n - k;
        }
        for (int i = n - k - 1; i >= 0; --i) {
            hash = ((hash - (s.charAt(i + k) & 31) * mult % modulo + modulo) % modulo * power + (s.charAt(i) & 31))
                    % modulo;
            if (hash == hashValue) {
                pos = i;
            }
        }
        return s.substring(pos, pos + k);

    }

    // 1223. 掷骰子模拟 (Dice Roll Simulation) --记忆化搜索
    private int[][][] memo1223;
    private int n1223;
    private int[] rollMax1223;

    public int dieSimulator(int n, int[] rollMax) {
        this.n1223 = n;
        this.rollMax1223 = rollMax;
        this.memo1223 = new int[n][7][16];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < 7; ++j) {
                Arrays.fill(memo1223[i][j], -1);
            }
        }
        return dfs1223(0, 0, 0);

    }

    private int dfs1223(int i, int j, int k) {
        if (i == n1223) {
            return 1;
        }
        if (memo1223[i][j][k] != -1) {
            return memo1223[i][j][k];
        }
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        for (int x = 1; x <= 6; ++x) {
            if (x != j) {
                res += dfs1223(i + 1, x, 1) % MOD;
            } else if (k < rollMax1223[j - 1]) {
                res += dfs1223(i + 1, x, k + 1) % MOD;
            }
            res %= MOD;
        }
        return memo1223[i][j][k] = res;
    }

    // 1223. 掷骰子模拟 (Dice Roll Simulation) --动态规划
    public int dieSimulator2(int n, int[] rollMax) {
        final int MOD = (int) (1e9 + 7);
        int m = rollMax.length;
        // f[i][j][k] 表示「投了i + 1次，上一次投的点数是j，该点数还剩k次可投掷」的不同方案数
        int[][][] f = new int[n][m][15];
        for (int j = 0; j < m; ++j) {
            Arrays.fill(f[0][j], 1);
        }
        for (int i = 1; i < n; ++i) {
            for (int last = 0; last < m; ++last) {
                for (int left = 0; left < rollMax[last]; ++left) {
                    long res = 0l;
                    for (int j = 0; j < m; ++j) {
                        if (j != last) {
                            res += f[i - 1][j][rollMax[j] - 1];
                        } else if (left > 0) {
                            res += f[i - 1][j][left - 1];
                        }
                    }
                    f[i][last][left] = (int) (res % MOD);
                }
            }
        }
        long res = 0l;
        for (int j = 0; j < m; ++j) {
            res += f[n - 1][j][rollMax[j] - 1];
        }
        return (int) (res % MOD);

    }

    // 1553. 吃掉 N 个橘子的最少天数 (Minimum Number of Days to Eat N Oranges) --记忆化搜索
    private Map<Integer, Integer> memo1553;

    public int minDays(int n) {
        memo1553 = new HashMap<>();
        return dfs1553(n);
    }

    private int dfs1553(int n) {
        if (n == 1) {
            return 1;
        }
        if (memo1553.containsKey(n)) {
            return memo1553.get(n);
        }
        int min = Integer.MAX_VALUE;
        if (n % 2 == 0) {
            min = Math.min(min, dfs1553(n / 2) + 1);
        }
        if (n % 3 == 0) {
            min = Math.min(min, dfs1553(n / 3) + 1);
        }
        if (n % 6 != 0) {
            min = Math.min(min, dfs1553(n - 1) + 1);
        }
        memo1553.put(n, min);
        return min;
    }

    // 1373. 二叉搜索子树的最大键值和 (Maximum Sum BST in Binary Tree)
    private int res1373;

    public int maxSumBST(TreeNode root) {
        dfs1373(root);
        return res1373;

    }

    private int[] dfs1373(TreeNode root) {
        if (root == null) {
            return new int[] { Integer.MAX_VALUE, Integer.MIN_VALUE, 0 };
        }
        int sum = root.val;
        int[] left = dfs1373(root.left);
        int[] right = dfs1373(root.right);

        if (root.val > left[1]) {
            sum += left[2];
        } else {
            return new int[] { Integer.MIN_VALUE, Integer.MAX_VALUE, 0 };
        }

        if (root.val < right[0]) {
            sum += right[2];
        } else {
            return new int[] { Integer.MIN_VALUE, Integer.MAX_VALUE, 0 };
        }

        res1373 = Math.max(res1373, sum);
        return new int[] { Math.min(root.val, left[0]), Math.max(root.val, right[1]), sum };
    }

    // 10. 正则表达式匹配 (Regular Expression Matching) --暴力dfs
    // 剑指 Offer 19. 正则表达式匹配
    public boolean isMatch(String s, String p) {
        if (p.isEmpty()) {
            return s.isEmpty();
        }
        boolean match = !s.isEmpty() && (s.charAt(0) == p.charAt(0) || p.charAt(0) == '.');
        if (p.length() >= 2 && p.charAt(1) == '*') {
            return isMatch(s, p.substring(2)) || (match && isMatch(s.substring(1), p));
        }
        return match && isMatch(s.substring(1), p.substring(1));

    }

    // 10. 正则表达式匹配 (Regular Expression Matching) --记忆化搜索
    // 剑指 Offer 19. 正则表达式匹配
    private int[][] memo10;
    private char[] s10;
    private char[] p10;

    public boolean isMatch2(String s, String p) {
        int m = s.length();
        int n = p.length();
        memo10 = new int[m + 1][n + 1];
        s10 = s.toCharArray();
        p10 = p.toCharArray();
        return dfs10(0, 0);

    }

    private boolean dfs10(int i, int j) {
        if (j == p10.length) {
            return i == s10.length;
        }
        if (memo10[i][j] != 0) {
            return memo10[i][j] > 0;
        }
        boolean match = i < s10.length && (s10[i] == p10[j] || p10[j] == '.');
        if (j + 1 < p10.length && p10[j + 1] == '*') {
            boolean b = dfs10(i, j + 2) || (match && dfs10(i + 1, j));
            memo10[i][j] = b ? memo10[i][j] = 1 : -1;
            return b;
        }
        boolean b2 = match && dfs10(i + 1, j + 1);
        memo10[i][j] = b2 ? memo10[i][j] = 1 : -1;
        return b2;
    }

    // 1510. 石子游戏 IV (Stone Game IV) --记忆化搜索
    private int[] memo1510;

    public boolean winnerSquareGame(int n) {
        memo1510 = new int[n + 1];
        return dfs1510(n);

    }

    private boolean dfs1510(int n) {
        if (n == 0) {
            return false;
        }
        if (n == 1) {
            return true;
        }

        if (memo1510[n] != 0) {
            return memo1510[n] > 0;
        }
        for (int i = 1; i * i <= n; ++i) {
            boolean b = dfs1510(n - i * i);
            if (!b) {
                memo1510[n] = 1;
                return true;
            }
        }
        memo1510[n] = -1;
        return false;
    }

    // 6354. 找出数组的串联值
    public long findTheArrayConcVal(int[] nums) {
        int n = nums.length;
        int i = 0;
        int j = n - 1;
        long res = 0l;
        while (i < j) {
            String s = String.valueOf(nums[i]) + String.valueOf(nums[j]);
            int num = Integer.parseInt(s);
            res += num;
            ++i;
            --j;
        }
        if (i == j) {
            res += nums[i];
        }
        return res;

    }

    // 6355. 统计公平数对的数目
    private int[] nums6355;

    public long countFairPairs(int[] nums, int lower, int upper) {
        Arrays.sort(nums);
        this.nums6355 = nums;
        int n = nums.length;
        long res = 0l;
        for (int i = 0; i < n; ++i) {
            int r = binarySearch6335(i + 1, n - 1, upper - nums[i]);
            int l = binarySearch6335_2(i + 1, n - 1, lower - nums[i]);
            if (r != -1 && l != -1 && r >= l) {
                res += r - l + 1;
            }
        }
        return res;

    }

    private int binarySearch6335_2(int left, int right, int target) {
        if (left > right) {
            return -1;
        }
        if (nums6355[right] < target) {
            return -1;
        }
        if (nums6355[left] >= target) {
            return left;
        }
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (nums6355[mid] >= target) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }

        }
        return res;
    }

    private int binarySearch6335(int left, int right, int target) {
        if (left > right) {
            return -1;
        }
        if (nums6355[left] > target) {
            return -1;
        }
        if (nums6355[right] <= target) {
            return right;
        }
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (nums6355[mid] <= target) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    // 6356. 子字符串异或查询
    public int[][] substringXorQueries(String s, int[][] queries) {
        Map<Integer, int[]> map = new HashMap<>();
        int n = queries.length;
        for (int i = 0; i < s.length(); ++i) {
            int num = 0;
            for (int j = i; j < Math.min(i + 30, s.length()); ++j) {
                num = (num << 1) | (s.charAt(j) - '0');
                if (!map.containsKey(num) || j - i < map.get(num)[1] - map.get(num)[0]) {
                    map.put(num, new int[] { i, j });
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            int[] q = queries[i];
            int xor = q[0] ^ q[1];
            queries[i] = map.getOrDefault(xor, new int[] { -1, -1 });
        }
        return queries;

    }

    // 823. 带因子的二叉树 (Binary Trees With Factors) --记忆化搜索
    private Map<Integer, Long> memo823;
    private Set<Integer> set823;
    private int[] arr823;

    public int numFactoredBinaryTrees(int[] arr) {
        memo823 = new HashMap<>();
        Arrays.sort(arr);
        this.arr823 = arr;
        set823 = Arrays.stream(arr).boxed().collect(Collectors.toSet());
        long res = 0l;
        final int MOD = (int) (1e9 + 7);
        for (int a : arr) {
            res = (res + dfs823(a)) % MOD;
        }
        return (int) res;

    }

    private long dfs823(int num) {
        final int MOD = (int) (1e9 + 7);
        if (memo823.containsKey(num)) {
            return memo823.get(num);
        }
        long res = 1l;
        for (int a : arr823) {
            if (a * arr823[0] > num) {
                break;
            }
            if (num % a == 0 && set823.contains(num / a)) {
                res = (res + dfs823(a) * dfs823(num / a)) % MOD;
            }
        }
        memo823.put(num, res);
        return res;
    }

    // 1220. 统计元音字母序列的数目 (Count Vowels Permutation)
    private int[][] memo1220;
    private int n1220;
    private List<Integer>[] g1220;

    public int countVowelPermutation(int n) {
        this.n1220 = n;
        this.memo1220 = new int[n][6];
        this.g1220 = new ArrayList[6];
        Arrays.setAll(g1220, k -> new ArrayList<>());
        g1220[0].addAll(List.of(1));
        g1220[1].addAll(List.of(0, 2));
        g1220[2].addAll(List.of(0, 1, 3, 4));
        g1220[3].addAll(List.of(2, 4));
        g1220[4].addAll(List.of(0));
        g1220[5].addAll(List.of(0, 1, 2, 3, 4));
        return dfs1220(0, 5);

    }

    private int dfs1220(int i, int j) {
        if (i == n1220) {
            return 1;
        }
        if (memo1220[i][j] != 0) {
            return memo1220[i][j];
        }
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        for (int k : g1220[j]) {
            res = (res + dfs1220(i + 1, k)) % MOD;
        }
        return memo1220[i][j] = res;
    }

    // 1458. 两个子序列的最大点积 (Max Dot Product of Two Subsequences)
    private int n1_1458;
    private int n2_1458;
    private int[] nums1_1458;
    private int[] nums2_1458;
    private int[][][] memo1458;

    public int maxDotProduct(int[] nums1, int[] nums2) {
        this.n1_1458 = nums1.length;
        this.n2_1458 = nums2.length;
        this.nums1_1458 = nums1;
        this.nums2_1458 = nums2;
        this.memo1458 = new int[n1_1458][n2_1458][2];
        for (int i = 0; i < n1_1458; ++i) {
            for (int j = 0; j < n2_1458; ++j) {
                Arrays.fill(memo1458[i][j], Integer.MIN_VALUE);
            }
        }
        return dfs1458(0, 0, 0);

    }

    private int dfs1458(int i, int j, int k) {
        if (i == n1_1458 || j == n2_1458) {
            return k == 1 ? 0 : (int) -1e9;
        }
        if (memo1458[i][j][k] != Integer.MIN_VALUE) {
            return memo1458[i][j][k];
        }
        return memo1458[i][j][k] = Math.max(dfs1458(i + 1, j + 1, 1) + nums1_1458[i] * nums2_1458[j],
                Math.max(dfs1458(i + 1, j, k), dfs1458(i, j + 1, k)));
    }

    // 1289. 下降路径最小和 II (Minimum Falling Path Sum II)
    private int[][] memo1289;
    private int n1289;
    private int[][] grid1289;

    public int minFallingPathSum(int[][] grid) {
        this.n1289 = grid.length;
        memo1289 = new int[n1289][n1289 + 1];
        grid1289 = grid;
        for (int i = 0; i < n1289; ++i) {
            Arrays.fill(memo1289[i], Integer.MAX_VALUE);
        }
        return dfs1289(0, n1289);
    }

    private int dfs1289(int i, int j) {
        if (i == n1289) {
            return 0;
        }
        if (memo1289[i][j] != Integer.MAX_VALUE) {
            return memo1289[i][j];
        }
        int min = Integer.MAX_VALUE;
        for (int k = 0; k < n1289; ++k) {
            if (j != k) {
                min = Math.min(min, grid1289[i][k] + dfs1289(i + 1, k));
            }
        }
        return memo1289[i][j] = min;
    }

    // 1411. 给 N x 3 网格图涂色的方案数 (Number of Ways to Paint N × 3 Grid)
    private int[][] memo1411;
    private int n1411;
    private List<Integer>[] g1411;

    public int numOfWays(int n) {
        final int MOD = (int) (1e9 + 7);
        memo1411 = new int[n][12];
        g1411 = new ArrayList[12];
        for (int i = 0; i < 12; ++i) {
            g1411[i] = new ArrayList<>();
        }
        // 状态定义：
        // 0--010
        // 1--020
        // 2--012
        // 3--021
        // 4--120
        // 5--102
        // 6--101
        // 7--121
        // 8--201
        // 9--210
        // 10--202
        // 11--212
        for (int i = 0; i < 12; ++i) {
            switch (i) {
                case 0:
                    g1411[i].addAll(List.of(5, 6, 7, 8, 10));
                    break;
                case 1:
                    g1411[i].addAll(List.of(5, 6, 8, 10, 11));
                    break;
                case 2:
                    g1411[i].addAll(List.of(4, 6, 7, 8));
                    break;
                case 3:
                    g1411[i].addAll(List.of(5, 9, 10, 11));
                    break;
                case 4:
                    g1411[i].addAll(List.of(2, 8, 10, 11));
                    break;
                case 5:
                    g1411[i].addAll(List.of(0, 1, 3, 9));
                    break;
                case 6:
                    g1411[i].addAll(List.of(0, 1, 2, 9, 11));
                    break;
                case 7:
                    g1411[i].addAll(List.of(0, 2, 9, 10, 11));
                    break;
                case 8:
                    g1411[i].addAll(List.of(0, 1, 2, 4));
                    break;
                case 9:
                    g1411[i].addAll(List.of(3, 5, 6, 7));
                    break;
                case 10:
                    g1411[i].addAll(List.of(0, 1, 3, 4, 7));
                    break;
                case 11:
                    g1411[i].addAll(List.of(1, 3, 4, 6, 7));
                    break;
                default:
                    break;
            }

        }
        for (int i = 0; i < n; ++i) {
            Arrays.fill(memo1411[i], -1);
        }
        this.n1411 = n;
        int res = 0;
        for (int type = 0; type < 12; ++type) {
            res = (res + dfs1411(0, type)) % MOD;
        }
        return res;

    }

    private int dfs1411(int i, int type) {
        if (i == n1411 - 1) {
            return 1;
        }
        if (memo1411[i][type] != -1) {
            return memo1411[i][type];
        }
        final int MOD = (int) (1e9 + 7);
        int res = 0;
        for (int next : g1411[type]) {
            res = (res + dfs1411(i + 1, next)) % MOD;
        }
        return memo1411[i][type] = res;
    }

    // 1964. 找出到每个位置为止最长的有效障碍赛跑路线 (Find the Longest Valid Obstacle Course at Each
    // Position) --300
    public int[] longestObstacleCourseAtEachPosition(int[] obstacles) {
        int n = obstacles.length;
        int[] dp = new int[n + 1];
        int len = 1;
        dp[len] = obstacles[0];
        int[] res = new int[n];
        res[0] = 1;
        for (int i = 1; i < n; ++i) {
            if (dp[len] <= obstacles[i]) {
                dp[++len] = obstacles[i];
                res[i] = len;
            } else {
                int left = 1;
                int right = len;
                int pos = 0;
                while (left <= right) {
                    int mid = left + ((right - left) >>> 1);
                    if (dp[mid] <= obstacles[i]) {
                        pos = mid;
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
                }
                dp[pos + 1] = obstacles[i];
                res[i] = pos + 1;
            }
        }
        return res;

    }

    // 1981. 最小化目标值与所选元素的差 (Minimize the Difference Between Target and Chosen
    // Elements)
    private int[] suf1981;
    private int m1981;
    private int[][] mat1981;
    private int[][] memo1981;

    public int minimizeTheDifference(int[][] mat, int target) {
        this.mat1981 = mat;
        this.m1981 = mat.length;
        this.suf1981 = new int[m1981];
        this.memo1981 = new int[m1981][target + 1];
        for (int i = 0; i < m1981; ++i) {
            Arrays.fill(memo1981[i], -1);
        }
        for (int i = m1981 - 1; i >= 0; --i) {
            Arrays.sort(mat[i]);
            if (i < m1981 - 1) {
                suf1981[i] = suf1981[i + 1] + mat[i][0];
            } else {
                suf1981[i] = mat[i][0];
            }
        }
        return dfs1981(0, target);
    }

    private int dfs1981(int i, int j) {
        if (i == m1981) {
            return Math.abs(j);
        }
        if (j <= 0) {
            return Math.abs(j) + suf1981[i];
        }
        if (memo1981[i][j] != -1) {
            return memo1981[i][j];
        }
        int res = Integer.MAX_VALUE;
        for (int x : mat1981[i]) {
            res = Math.min(res, dfs1981(i + 1, j - x));
            if (j - x <= 0) {
                break;
            }
        }
        return memo1981[i][j] = res;
    }

    // 1377. T 秒后青蛙的位置 (Frog Position After T Seconds)
    public double frogPosition(int n, int[][] edges, int t, int target) {
        List<Integer>[] g = new ArrayList[n + 1];
        Arrays.setAll(g, k -> new ArrayList<>());
        for (int[] e : edges) {
            g[e[0]].add(e[1]);
            g[e[1]].add(e[0]);
        }
        boolean[] vis = new boolean[n + 1];
        vis[1] = true;
        Queue<long[]> q = new LinkedList<>();
        // node, prob, time
        q.offer(new long[] { 1, 1, 0 });
        while (!q.isEmpty()) {
            long[] cur = q.poll();
            int x = (int) cur[0];
            long p = cur[1];
            int curT = (int) cur[2];
            if (curT > t) {
                continue;
            }
            if (x == target && curT == t) {
                return 1D / p;
            }
            int size = 0;
            for (int y : g[x]) {
                if (!vis[y]) {
                    ++size;
                }
            }
            if (x == target) {
                return size == 0 ? 1D / p : 0D;
            }
            if (size > 0) {
                for (int y : g[x]) {
                    if (!vis[y]) {
                        vis[y] = true;
                        q.offer(new long[] { y, p * size, curT + 1 });
                    }
                }
            }
        }
        return 0D;

    }

    // 1253. 重构 2 行二进制矩阵 (Reconstruct a 2-Row Binary Matrix)
    public List<List<Integer>> reconstructMatrix(int upper, int lower, int[] colsum) {
        int n = colsum.length;
        List<List<Integer>> res = new ArrayList<>();
        res.add(new ArrayList<>());
        res.add(new ArrayList<>());
        for (int i = 0; i < n; ++i) {
            if (colsum[i] == 1) {
                if (upper > lower) {
                    --upper;
                    res.get(0).add(1);
                    res.get(1).add(0);
                } else {
                    --lower;
                    res.get(0).add(0);
                    res.get(1).add(1);
                }
            } else {
                int add = colsum[i] / 2;
                upper -= add;
                lower -= add;
                res.get(0).add(add);
                res.get(1).add(add);

            }
            if (lower < 0 || upper < 0) {
                break;
            }
        }
        if (lower != 0 || upper != 0) {
            return new ArrayList<>();
        }
        return res;

    }

    // 1537. 最大得分 (Get the Maximum Score) --建图 记忆化搜索
    private Map<Integer, List<Integer>> g1537;
    private Map<Integer, Long> memo1537;

    public int maxSum(int[] nums1, int[] nums2) {
        g1537 = new HashMap<>();
        memo1537 = new HashMap<>();
        for (int i = 0; i < nums1.length; ++i) {
            if (i + 1 < nums1.length) {
                g1537.computeIfAbsent(nums1[i], k -> new ArrayList<>(2)).add(nums1[i + 1]);
            }
        }
        for (int i = 0; i < nums2.length; ++i) {
            if (i + 1 < nums2.length) {
                g1537.computeIfAbsent(nums2[i], k -> new ArrayList<>(2)).add(nums2[i + 1]);
            }
        }
        final int MOD = (int) (1e9 + 7);
        return (int) (Math.max(dfs1537(nums1[0]), dfs1537(nums2[0])) % MOD);

    }

    private long dfs1537(int x) {
        if (memo1537.containsKey(x)) {
            return memo1537.get(x);
        }
        long max = 0l;
        for (int y : g1537.getOrDefault(x, new ArrayList<>())) {
            max = Math.max(max, dfs1537(y));
        }
        memo1537.put(x, max + x);
        return max + x;
    }

    // 1537. 最大得分 (Get the Maximum Score) --双指针 贪心
    public int maxSum2(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        int i = 0;
        int j = 0;
        long sum1 = 0l;
        long sum2 = 0l;
        while (i < m && j < n) {
            if (nums1[i] < nums2[j]) {
                sum1 += nums1[i++];
            } else if (nums1[i] > nums2[j]) {
                sum2 += nums2[j++];
            } else {
                sum1 = sum2 = Math.max(sum1, sum2) + nums1[i];
                ++i;
                ++j;
            }
        }
        while (i < m) {
            sum1 += nums1[i++];
        }
        while (j < n) {
            sum2 += nums2[j++];
        }
        final int MOD = (int) (1e9 + 7);
        return (int) (Math.max(sum1, sum2) % MOD);
    }

    // 1312. 让字符串成为回文串的最少插入次数 (Minimum Insertion Steps to Make a String Palindrome)
    private int[][] memo1312;
    private String s1312;

    public int minInsertions(String s) {
        int n = s.length();
        memo1312 = new int[n][n];
        this.s1312 = s;
        for (int i = 0; i < n; ++i) {
            Arrays.fill(memo1312[i], Integer.MAX_VALUE);
        }
        return dfs1312(0, n - 1);
    }

    private int dfs1312(int i, int j) {
        if (i >= j) {
            return 0;
        }
        if (memo1312[i][j] != Integer.MAX_VALUE) {
            return memo1312[i][j];
        }
        if (s1312.charAt(i) == s1312.charAt(j)) {
            return memo1312[i][j] = dfs1312(i + 1, j - 1);
        }
        return memo1312[i][j] = Math.min(dfs1312(i + 1, j), dfs1312(i, j - 1)) + 1;
    }

    // 935. 骑士拨号器 (Knight Dialer)
    private int[][] memo935;
    private int[][] map935 = { { 4, 6 }, { 6, 8 }, { 7, 9 }, { 4, 8 }, { 0, 3, 9 }, {}, { 0, 1, 7 }, { 2, 6 }, { 1, 3 },
            { 2, 4 } };

    public int knightDialer(int n) {
        memo935 = new int[10][n];
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        for (int i = 0; i <= 9; ++i) {
            res = (res + dfs935(i, n - 1)) % MOD;
        }
        return res;

    }

    private int dfs935(int num, int left) {
        if (left == 0) {
            return 1;
        }
        if (memo935[num][left] != 0) {
            return memo935[num][left];
        }
        final int MOD = (int) (1e9 + 7);
        int res = 0;
        for (int next : map935[num]) {
            res = (res + dfs935(next, left - 1)) % MOD;
        }
        return memo935[num][left] = res;
    }

    // 1463. 摘樱桃 II (Cherry Pickup II)
    private int[][][] memo1463;
    private int m1463;
    private int n1463;
    private int[][] grid1463;

    public int cherryPickupⅡ(int[][] grid) {
        this.m1463 = grid.length;
        this.n1463 = grid[0].length;
        this.grid1463 = grid;
        memo1463 = new int[m1463][n1463][n1463];
        for (int i = 0; i < m1463; ++i) {
            for (int j = 0; j < n1463; ++j) {
                Arrays.fill(memo1463[i][j], -1);
            }
        }
        return dfs1463(0, 0, n1463 - 1);

    }

    private int dfs1463(int i, int j1, int j2) {
        if (i == m1463) {
            return 0;
        }
        if (memo1463[i][j1][j2] != -1) {
            return memo1463[i][j1][j2];
        }
        int max = 0;
        for (int x = Math.max(0, j1 - 1); x <= Math.min(n1463 - 1, j1 + 1); ++x) {
            for (int y = Math.max(0, j2 - 1); y <= Math.min(n1463 - 1, j2 + 1); ++y) {
                max = Math.max(max, dfs1463(i + 1, x, y));
            }
        }
        return memo1463[i][j1][j2] = max + (j1 == j2 ? grid1463[i][j1] : (grid1463[i][j1] + grid1463[i][j2]));
    }

    // 1269. 停在原地的方案数 (Number of Ways to Stay in the Same Place After Some Steps)
    private int[][] memo1269;
    private int arrLen1269;

    public int numWays(int steps, int arrLen) {
        this.arrLen1269 = arrLen;
        memo1269 = new int[steps + 1][steps + 1];
        for (int i = 0; i < steps + 1; ++i) {
            Arrays.fill(memo1269[i], -1);
        }
        return dfs1269(0, steps);

    }

    private int dfs1269(int i, int j) {
        if (i >= arrLen1269 || i < 0 || i > j) {
            return 0;
        }
        if (j == 0) {
            return i == 0 ? 1 : 0;
        }
        if (memo1269[i][j] != -1) {
            return memo1269[i][j];
        }
        final int MOD = (int) (1e9 + 7);
        return memo1269[i][j] = (int) (((long) dfs1269(i, j - 1) + dfs1269(i + 1, j - 1) + dfs1269(i - 1, j - 1))
                % MOD);
    }

    // 6359. 替换一个数字后的最大差值 (Maximum Difference by Remapping a Digit)
    public int minMaxDifference(int num) {
        char[] chars = String.valueOf(num).toCharArray();
        char[] max = chars.clone();
        char c = '_';
        for (int i = 0; i < max.length; ++i) {
            if (max[i] != '9' && c == '_') {
                c = max[i];
            }
            if (max[i] == c) {
                max[i] = '9';
            }
        }
        char[] min = chars.clone();
        c = min[0];
        for (int i = 0; i < min.length; ++i) {
            if (min[i] == c) {
                min[i] = '0';
            }
        }
        int max1 = Integer.parseInt(String.valueOf(max));
        int min1 = Integer.parseInt(String.valueOf(min));
        return max1 - min1;

    }

    // 2567. 修改两个元素的最小分数 (Minimum Score by Changing Two Elements)
    public int minimizeSum(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        return Math.min(nums[n - 2] - nums[1], Math.min(nums[n - 3] - nums[0], nums[n - 1] - nums[2]));

    }

    // 6360. 最小无法得到的或值 (Minimum Impossible OR)
    public int minImpossibleOR(int[] nums) {
        int mask = 0;
        for (int num : nums) {
            // 判断「num为2的整数次幂」的三种方法
            // Integer.bitCount(num) == 1;
            // num & -num == num;
            // num & (num - 1) == 0
            if ((num & (num - 1)) == 0) {
                mask |= num;
            }
        }
        mask = ~mask;
        return mask & -mask;

    }

    // 2570. 合并两个二维数组 - 求和法 (Merge Two 2D Arrays by Summing Values)
    public int[][] mergeArrays(int[][] nums1, int[][] nums2) {
        int i = 0;
        int j = 0;
        int n1 = nums1.length;
        int n2 = nums2.length;
        List<int[]> res = new ArrayList<>();
        while (i < n1 && j < n2) {
            if (nums1[i][0] == nums2[j][0]) {
                res.add(new int[] { nums1[i][0], nums1[i][1] + nums2[j][1] });
                ++i;
                ++j;
            } else if (nums1[i][0] < nums2[j][0]) {
                res.add(nums1[i++]);
            } else {
                res.add(nums2[j++]);
            }
        }
        while (i < n1) {
            res.add(nums1[i++]);
        }
        while (j < n2) {
            res.add(nums2[j++]);
        }
        return res.toArray(new int[res.size()][]);
    }

    // 2571. 将整数减少到零需要的最少操作数 (Minimum Operations to Reduce an Integer to 0)
    private Map<Integer, Integer> memo2571;

    public int minOperations(int n) {
        memo2571 = new HashMap<>();
        return dfs2571(n);

    }

    private int dfs2571(int n) {
        if (n == 0) {
            return 0;
        }
        if (memo2571.containsKey(n)) {
            return memo2571.get(n);
        }
        int x = n & (-n);
        int min = Math.min(dfs2571(n - x), dfs2571(n + x)) + 1;
        memo2571.put(n, min);
        return min;
    }

    // 1301. 最大得分的路径数目 (Number of Paths with Max Score)
    private int n1301;
    private List<String> board1301;
    // 最大分数
    private int[][] memo1301;
    // 最大分数的方案数
    private int[][] ways1301;

    public int[] pathsWithMaxScore(List<String> board) {
        this.n1301 = board.size();
        this.board1301 = board;
        this.memo1301 = new int[n1301][n1301];
        for (int i = 0; i < n1301; ++i) {
            Arrays.fill(memo1301[i], -1);
        }
        this.ways1301 = new int[n1301][n1301];
        ways1301[n1301 - 1][n1301 - 1] = 1;
        int max = dfs1301(0, 0);
        if (max < 0) {
            return new int[] { 0, 0 };
        }
        return new int[] { max, ways1301[0][0] };
    }

    // 从[i, j]出发，向右、右下、下走的最大分数
    private int dfs1301(int i, int j) {
        if (i == n1301 - 1 && j == n1301 - 1) {
            return 0;
        }
        if (memo1301[i][j] != -1) {
            return memo1301[i][j];
        }
        int max1 = (int) -1e5;
        int max2 = (int) -1e5;
        int max3 = (int) -1e5;
        if (i + 1 < n1301 && board1301.get(i + 1).charAt(j) != 'X') {
            max1 = dfs1301(i + 1, j);
        }
        if (i + 1 < n1301 && j + 1 < n1301 && board1301.get(i + 1).charAt(j + 1) != 'X') {
            max2 = dfs1301(i + 1, j + 1);
        }
        if (j + 1 < n1301 && board1301.get(i).charAt(j + 1) != 'X') {
            max3 = dfs1301(i, j + 1);
        }
        int max = (int) -1e5;
        max = Math.max(max, max1);
        max = Math.max(max, max2);
        max = Math.max(max, max3);
        if (max >= 0) {
            final int MOD = (int) (1e9 + 7);
            if (max == max1) {
                ways1301[i][j] = (ways1301[i][j] + ways1301[i + 1][j]) % MOD;
            }
            if (max == max2) {
                ways1301[i][j] = (ways1301[i][j] + ways1301[i + 1][j + 1]) % MOD;
            }
            if (max == max3) {
                ways1301[i][j] = (ways1301[i][j] + ways1301[i][j + 1]) % MOD;
            }
        }
        // [0, 0]是起始点没有分数
        if (!(i == 0 && j == 0)) {
            max += board1301.get(i).charAt(j) - '0';
        }
        return memo1301[i][j] = max;
    }

    // 1326. 灌溉花园的最少水龙头数目 (Minimum Number of Taps to Open to Water a Garden)
    // 相似题目：55. 跳跃游戏 // 45. 跳跃游戏 II // 1024. 视频拼接
    public int minTaps(int n, int[] ranges) {
        int[] rightMost = new int[n + 1];
        for (int i = 0; i <= n; ++i) {
            int r = ranges[i];
            if (i - r > 0) {
                rightMost[i - r] = i + r;
            } else {
                rightMost[0] = Math.max(rightMost[0], i + r);
            }
        }
        int res = 0;
        int cur = 0;
        int right = 0;
        for (int i = 0; i < n; ++i) {
            right = Math.max(right, rightMost[i]);
            if (i == cur) {
                cur = right;
                ++res;
            }
        }
        if (cur >= n) {
            return res;
        }
        return -1;

    }

    // 741. 摘樱桃 (Cherry Pickup)
    private int n741;
    private int[][] grid741;
    private int[][][] memo741;

    public int cherryPickup(int[][] grid) {
        this.n741 = grid.length;
        this.grid741 = grid;
        this.memo741 = new int[n741][n741][n741];
        for (int i = 0; i < n741; ++i) {
            for (int j = 0; j < n741; ++j) {
                Arrays.fill(memo741[i][j], -1);
            }
        }
        return Math.max(0, dfs741(0, 0, 0));
    }

    private int dfs741(int i, int j, int k) {
        int l = i + j - k;
        if (i == n741 || j == n741 || k == n741 || l == n741 || grid741[i][j] == -1 || grid741[k][l] == -1) {
            return (int) -1e5;
        }
        if (i == n741 - 1 && j == n741 - 1) {
            return grid741[i][j];
        }
        if (memo741[i][j][k] != -1) {
            return memo741[i][j][k];
        }
        return memo741[i][j][k] = Math.max(Math.max(dfs741(i + 1, j, k), dfs741(i + 1, j, k + 1)),
                Math.max(dfs741(i, j + 1, k), dfs741(i, j + 1, k + 1))) + grid741[i][j]
                + (i != k || j != l ? grid741[k][l] : 0);
    }

    // LCP 40. 心算挑战
    public int maxmiumScore(int[] cards, int cnt) {
        List<Integer>[] list = new ArrayList[2];
        Arrays.setAll(list, o -> new ArrayList<>());
        list[0].add(0);
        list[1].add(0);
        Arrays.sort(cards);
        for (int i = cards.length - 1; i >= 0; --i) {
            int j = cards[i] % 2;
            list[j].add(list[j].get(list[j].size() - 1) + cards[i]);
        }
        int res = 0;
        for (int k = 0; k < list[1].size(); k += 2) {
            if (cnt >= k && cnt - k < list[0].size()) {
                res = Math.max(res, list[0].get(cnt - k) + list[1].get(k));
            }
        }
        return res;

    }

    // 1140. 石子游戏 II (Stone Game II)
    private int[][] memo1140;
    private int[] piles1140;

    public int stoneGameII(int[] piles) {
        this.piles1140 = piles;
        int n = piles.length;
        for (int i = n - 2; i >= 0; --i) {
            piles[i] += piles[i + 1];
        }
        memo1140 = new int[n - 1][n];
        for (int i = 0; i < n - 1; ++i) {
            Arrays.fill(memo1140[i], -1);
        }
        return dfs1140(0, 1);
    }

    private int dfs1140(int i, int m) {
        if (i + 2 * m >= piles1140.length) {
            return piles1140[i];
        }
        if (memo1140[i][m] != -1) {
            return memo1140[i][m];
        }
        int min = Integer.MAX_VALUE;
        for (int j = 1; j <= 2 * m; ++j) {
            min = Math.min(min, dfs1140(i + j, Math.max(j, m)));
        }
        return memo1140[i][m] = piles1140[i] - min;
    }

    // 1444. 切披萨的方案数 (Number of Ways of Cutting a Pizza)
    private int m1444;
    private int n1444;
    private int[][] pre1444;
    private int[][][] memo1444;

    public int ways(String[] pizza, int k) {
        this.m1444 = pizza.length;
        this.n1444 = pizza[0].length();
        this.pre1444 = new int[m1444 + 1][n1444 + 1];
        for (int i = 0; i < m1444; ++i) {
            for (int j = 0; j < n1444; ++j) {
                pre1444[i + 1][j + 1] = pre1444[i + 1][j] + pre1444[i][j + 1] - pre1444[i][j]
                        + (pizza[i].charAt(j) == 'A' ? 1 : 0);
            }

        }
        this.memo1444 = new int[m1444][n1444][k + 1];
        for (int i = 0; i < m1444; ++i) {
            for (int j = 0; j < n1444; ++j) {
                Arrays.fill(memo1444[i][j], -1);
            }
        }
        return dfs1444(0, 0, k);

    }

    private int dfs1444(int i, int j, int l) {
        int c = getCounts1444(i, j, m1444 - 1, n1444 - 1);
        if (l == 1) {
            return c > 0 ? 1 : 0;
        }
        if (memo1444[i][j][l] != -1) {
            return memo1444[i][j][l];
        }
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        for (int k = i; k < m1444 - 1; ++k) {
            int cnt = getCounts1444(i, j, k, n1444 - 1);
            if (cnt > 0) {
                res = (res + dfs1444(k + 1, j, l - 1)) % MOD;
            }
        }
        for (int k = j; k < n1444 - 1; ++k) {
            int cnt = getCounts1444(i, j, m1444 - 1, k);
            if (cnt > 0) {
                res = (res + dfs1444(i, k + 1, l - 1)) % MOD;
            }
        }
        return memo1444[i][j][l] = res;
    }

    private int getCounts1444(int x1, int y1, int x2, int y2) {
        return pre1444[x2 + 1][y2 + 1] - pre1444[x2 + 1][y1] - pre1444[x1][y2 + 1] + pre1444[x1][y1];
    }

    // LCP 52. 二叉搜索树染色
    private List<Integer> listLCP_52;

    public int getNumber(TreeNode root, int[][] ops) {
        listLCP_52 = new ArrayList<>();
        dfs_LCP52(root);
        int res = 0;
        for (int i = ops.length - 1; i >= 0; --i) {
            int type = ops[i][0];
            int left = ops[i][1];
            int right = ops[i][2];
            List<Integer> temp = new ArrayList<>(listLCP_52);
            int index1 = binarySearchLCP52_1(left);
            int index2 = binarySearchLCP52_2(right);
            if (index1 != -1 && index2 != -1) {
                if (type == 1) {
                    res += index2 - index1 + 1;
                }
                listLCP_52 = temp.subList(0, index1);
                listLCP_52.addAll(temp.subList(index2 + 1, temp.size()));
            }
        }
        return res;
    }

    // <= target 的最大索引
    private int binarySearchLCP52_2(int target) {
        if (listLCP_52.isEmpty()) {
            return -1;
        }
        int n = listLCP_52.size();
        if (target < listLCP_52.get(0)) {
            return -1;
        }
        if (listLCP_52.get(n - 1) <= target) {
            return n - 1;
        }
        int left = 0;
        int right = n - 1;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (listLCP_52.get(mid) <= target) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    // >= target 的最小索引
    private int binarySearchLCP52_1(int target) {
        if (listLCP_52.isEmpty()) {
            return -1;
        }
        int n = listLCP_52.size();
        if (listLCP_52.get(0) >= target) {
            return 0;
        }
        if (target > listLCP_52.get(n - 1)) {
            return -1;
        }
        int left = 0;
        int right = n - 1;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (listLCP_52.get(mid) >= target) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;

    }

    private void dfs_LCP52(TreeNode root) {
        if (root == null) {
            return;
        }
        dfs_LCP52(root.left);
        listLCP_52.add(root.val);
        dfs_LCP52(root.right);
    }

    // 403. 青蛙过河 (Frog Jump)
    private int[][] memo403;
    private int[] stones403;
    private int n;

    public boolean canCross(int[] stones) {
        if (stones[1] != 1) {
            return false;
        }
        this.n = stones.length;
        memo403 = new int[n][n + 1];
        this.stones403 = stones;
        for (int i = 0; i < n; ++i) {
            Arrays.fill(memo403[i], -1);
        }
        return dfs403(1, 1);

    }

    private boolean dfs403(int i, int k) {
        if (i == stones403.length - 1) {
            return true;
        }
        if (memo403[i][k] != -1) {
            return memo403[i][k] > 0;
        }
        for (int j = i + 1; j < n && stones403[j] - stones403[i] - k <= 1; ++j) {
            if (stones403[j] - stones403[i] - k < -1) {
                continue;
            }
            if (dfs403(j, stones403[j] - stones403[i])) {
                memo403[i][k] = 1;
                return true;
            }
        }
        memo403[i][k] = 0;
        return false;
    }

    // 132. 分割回文串 II (Palindrome Partitioning II)
    // 剑指 Offer II 094. 最少回文分割
    private boolean[][] isPalindromes132;
    private int n132;
    private int[] memo132;

    public int minCut(String s) {
        this.n132 = s.length();
        isPalindromes132 = new boolean[n132][n132];
        for (int i = n132 - 1; i >= 0; --i) {
            for (int j = i; j < n132; ++j) {
                if (s.charAt(i) == s.charAt(j) && (j - i < 2 || isPalindromes132[i + 1][j - 1])) {
                    isPalindromes132[i][j] = true;
                }
            }
        }
        memo132 = new int[n132];
        Arrays.fill(memo132, -1);
        return dfs132(0) - 1;

    }

    private int dfs132(int i) {
        if (i == n132) {
            return 0;
        }
        if (isPalindromes132[i][n132 - 1]) {
            return 1;
        }
        if (memo132[i] != -1) {
            return memo132[i];
        }

        int cur = n132;
        for (int j = i; j < n132; ++j) {
            if (isPalindromes132[i][j]) {
                cur = Math.min(cur, dfs132(j + 1) + 1);
            }
        }
        return memo132[i] = cur;
    }

    // 2472. 不重叠回文子字符串的最大数目 (Maximum Number of Non-overlapping Palindrome
    // Substrings)
    private boolean[][] isPalindromes2472;
    private int[] memo2472;
    private int k2472;
    private int n2472;

    public int maxPalindromes(String s, int k) {
        this.n2472 = s.length();
        this.k2472 = k;
        isPalindromes2472 = new boolean[n2472][n2472];
        char[] arr = s.toCharArray();
        for (int i = n2472 - 1; i >= 0; --i) {
            for (int j = i; j < n2472; ++j) {
                if (arr[i] == arr[j] && (j - i < 2 || isPalindromes2472[i + 1][j - 1])) {
                    isPalindromes2472[i][j] = true;
                }
            }
        }
        memo2472 = new int[n2472];
        Arrays.fill(memo2472, -1);
        return dfs2472(0);

    }

    private int dfs2472(int i) {
        if (i == n2472) {
            return 0;
        }
        if (n2472 - i < k2472) {
            return 0;
        }
        if (memo2472[i] != -1) {
            return memo2472[i];
        }
        // 不选
        int max = dfs2472(i + 1);
        // 选
        for (int j = i + k2472 - 1; j < n2472; ++j) {
            if (isPalindromes2472[i][j]) {
                max = Math.max(max, dfs2472(j + 1) + 1);
                break;
            }
        }
        return memo2472[i] = max;
    }

    // 410. 分割数组的最大值 (Split Array Largest Sum)
    public int splitArray(int[] nums, int k) {
        int left = 0;
        int right = 0;
        for (int num : nums) {
            left = Math.max(left, num);
            right += num;
        }
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (check410(mid, nums) > k) {
                left = mid + 1;
            } else {
                res = mid;
                right = mid - 1;
            }
        }
        return res;

    }

    private int check410(int target, int[] nums) {
        int sum = 0;
        int count = 0;
        for (int num : nums) {
            if (num + sum > target) {
                sum = 0;
                ++count;
            }
            sum += num;
        }
        return count + 1;
    }

    // 410. 分割数组的最大值 (Split Array Largest Sum)
    private int n410;
    private int[] nums410;
    private int k410;
    private int[][] memo410;

    public int splitArray2(int[] nums, int k) {
        this.n410 = nums.length;
        this.nums410 = nums;
        this.k410 = k;
        this.memo410 = new int[n410][k];
        for (int i = 0; i < n410; ++i) {
            Arrays.fill(memo410[i], -1);
        }
        return dfs410(0, 0);

    }

    private int dfs410(int i, int j) {
        if (i == n410) {
            return j == k410 ? 0 : (int) 1e9;
        }
        if (j == k410) {
            return (int) 1e9;
        }
        if (memo410[i][j] != -1) {
            return memo410[i][j];
        }
        int res = (int) 1e9;
        int s = 0;
        for (int x = i; x <= n410 - k410 + j; ++x) {
            s += nums410[x];
            res = Math.min(res, Math.max(s, dfs410(x + 1, j + 1)));
        }
        return memo410[i][j] = res;
    }

    // 312. 戳气球 (Burst Balloons)
    private int[][] memo312;
    private int[] arr312;

    public int maxCoins(int[] nums) {
        int n = nums.length;
        int[] arr = new int[n + 2];
        for (int i = 0; i < n; ++i) {
            arr[i + 1] = nums[i];
        }
        arr[0] = arr[n + 1] = 1;
        this.arr312 = arr;
        memo312 = new int[n + 2][n + 2];
        for (int i = 0; i < n + 2; ++i) {
            Arrays.fill(memo312[i], -1);
        }
        return dfs312(0, n + 1);

    }

    private int dfs312(int left, int right) {
        if (right - left <= 1) {
            return 0;
        }
        if (memo312[left][right] != -1) {
            return memo312[left][right];
        }
        int max = 0;
        for (int i = left + 1; i < right; ++i) {
            int sum = arr312[left] * arr312[i] * arr312[right] + dfs312(left, i) + dfs312(i, right);
            max = Math.max(max, sum);
        }
        return memo312[left][right] = max;
    }

    // 面试题 17.23. 最大黑方阵
    public int[] findSquare(int[][] matrix) {
        int n = matrix.length;
        int[][] preRow = new int[n][n + 1];
        for (int i = 0; i < n; ++i) {
            for (int j = 1; j < n + 1; ++j) {
                preRow[i][j] = preRow[i][j - 1] + matrix[i][j - 1];
            }

        }
        int[][] preCol = new int[n][n + 1];
        for (int i = 0; i < n; ++i) {
            for (int j = 1; j < n + 1; ++j) {
                preCol[i][j] = preCol[i][j - 1] + matrix[j - 1][i];
            }
        }
        for (int r = n; r > 0; --r) {
            for (int i = 0; i < n - r + 1; ++i) {
                for (int j = 0; j < n - r + 1; ++j) {
                    if (preRow[i][j + r] - preRow[i][j] == 0
                            && preRow[i + r - 1][j + r] - preRow[i + r - 1][j] == 0
                            && preCol[j][i + r] - preCol[j][i] == 0
                            && preCol[j + r - 1][i + r] - preCol[j + r - 1][i] == 0) {
                        return new int[] { i, j, r };
                    }
                }
            }
        }
        return new int[0];

    }

    // 2574. 左右元素和的差值 (Left and Right Sum Differences)
    public int[] leftRigthDifference(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        int[] leftSum = new int[n];
        for (int i = 1; i < n; ++i) {
            leftSum[i] = leftSum[i - 1] + nums[i - 1];
        }
        int[] rightSum = new int[n];
        for (int i = n - 2; i >= 0; --i) {
            rightSum[i] = rightSum[i + 1] + nums[i + 1];
        }
        for (int i = 0; i < n; ++i) {
            res[i] = Math.abs(Math.abs(leftSum[i] - rightSum[i]));
        }
        return res;

    }

    // 2575. 找出字符串的可整除数组 (Find the Divisibility Array of a String)
    public int[] divisibilityArray(String word, int m) {
        int n = word.length();
        int[] res = new int[n];
        long mod = 0L;
        for (int i = 0; i < n; ++i) {
            mod = (mod * 10 + word.charAt(i) - '0') % m;
            if (mod == 0) {
                res[i] = 1;
            }
        }
        return res;

    }

    // 1340. 跳跃游戏 V (Jump Game V)
    private int n1340;
    private int[] arr1340;
    private int[] memo1340;
    private int d1340;

    public int maxJumps(int[] arr, int d) {
        this.n1340 = arr.length;
        this.arr1340 = arr;
        this.d1340 = d;
        memo1340 = new int[n1340];
        Arrays.fill(memo1340, -1);
        int res = 0;
        for (int i = 0; i < n1340; ++i) {
            res = Math.max(res, dfs1340(i));
        }
        return res;

    }

    private int dfs1340(int i) {
        if (i < 0 || i >= n1340) {
            return 0;
        }
        if (memo1340[i] != -1) {
            return memo1340[i];
        }
        int res = 1;
        int j = i - 1;
        while (j >= 0 && i - j <= d1340) {
            if (arr1340[j] >= arr1340[i]) {
                break;
            }
            res = Math.max(res, dfs1340(j--) + 1);
        }
        j = i + 1;
        while (j < n1340 && j - i <= d1340) {
            if (arr1340[j] >= arr1340[i]) {
                break;
            }
            res = Math.max(res, dfs1340(j++) + 1);
        }
        return memo1340[i] = res;
    }

    // 140. 单词拆分 II (Word Break II)
    private List<String> res140;
    private Set<String> set140;
    private String s140;

    public List<String> wordBreak(String s, List<String> wordDict) {
        res140 = new ArrayList<>();
        set140 = new HashSet<>(wordDict);
        this.s140 = s;
        dfs140(0, new ArrayList<>());
        return res140;

    }

    private void dfs140(int i, List<String> list) {
        if (i == s140.length()) {
            res140.add(String.join(" ", list));
        }
        StringBuilder b = new StringBuilder();
        int j = i;
        while (j < s140.length()) {
            b.append(s140.charAt(j));
            if (set140.contains(b.toString())) {
                list.add(b.toString());
                dfs140(j + 1, list);
                list.remove(list.size() - 1);
            }
            ++j;
        }
    }

    // 955. 删列造序 II (Delete Columns to Make Sorted II)
    public int minDeletionSize(String[] strs) {
        int n = strs.length;
        int m = strs[0].length();
        int res = 0;
        char[][] arr = new char[n][m];
        for (int i = 0; i < n; ++i) {
            arr[i] = strs[i].toCharArray();
        }
        boolean[] cuts = new boolean[n - 1];
        search: for (int j = 0; j < m; ++j) {
            for (int i = 0; i < n - 1; ++i) {
                if (!cuts[i] && arr[i][j] > arr[i + 1][j]) {
                    ++res;
                    continue search;
                }
            }
            for (int i = 0; i < n - 1; ++i) {
                if (arr[i][j] < arr[i + 1][j]) {
                    cuts[i] = true;
                }
            }
        }
        return res;

    }

    // 1745. 回文串分割 IV (Palindrome Partitioning IV)
    private boolean[][] isPalindromes1745;
    private int[][] memo1745;
    private int n1745;

    public boolean checkPartitioning(String s) {
        this.n1745 = s.length();
        isPalindromes1745 = new boolean[n1745][n1745];
        for (int i = n1745 - 1; i >= 0; --i) {
            for (int j = i; j < n1745; ++j) {
                if (s.charAt(i) == s.charAt(j) && (j - i < 2 || isPalindromes1745[i + 1][j - 1])) {
                    isPalindromes1745[i][j] = true;
                }
            }
        }
        memo1745 = new int[n1745][3];
        return dfs1745(0, 0);

    }

    private boolean dfs1745(int i, int j) {
        if (i == n1745) {
            return j == 3;
        }
        if (j == 3) {
            return false;
        }
        if (memo1745[i][j] != 0) {
            return memo1745[i][j] > 0;
        }
        for (int k = i; k < n1745; ++k) {
            if (isPalindromes1745[i][k] && dfs1745(k + 1, j + 1)) {
                memo1745[i][j] = 1;
                return true;
            }
        }
        memo1745[i][j] = -1;
        return false;
    }

    // 526. 优美的排列 (Beautiful Arrangement)
    private int n526;
    private int[][] memo526;
    private int u526;

    public int countArrangement(int n) {
        this.n526 = n;
        memo526 = new int[n][1 << n];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(memo526[i], -1);
        }
        this.u526 = (1 << n) - 1;
        return dfs526(0, 0);

    }

    private int dfs526(int i, int j) {
        if (i == n526) {
            return 1;
        }
        if (memo526[i][j] != -1) {
            return memo526[i][j];
        }
        int res = 0;
        for (int c = u526 ^ j; c != 0; c &= c - 1) {
            int lb = Integer.numberOfTrailingZeros(c);
            if ((i + 1) % (lb + 1) == 0 || (lb + 1) % (i + 1) == 0) {
                res += dfs526(i + 1, j | (1 << lb));
            }
        }
        return memo526[i][j] = res;
    }

    // 1001. 网格照明 (Grid Illumination)
    public int[] gridIllumination(int n, int[][] lamps, int[][] queries) {
        Map<Integer, Integer> row = new HashMap<>();
        Map<Integer, Integer> col = new HashMap<>();
        Map<Integer, Integer> mainDiag = new HashMap<>();
        Map<Integer, Integer> counterDiag = new HashMap<>();
        Set<Long> set = new HashSet<>();
        for (int[] lamp : lamps) {
            int r = lamp[0];
            int c = lamp[1];
            if (set.add((long) r * n + c)) {
                row.merge(r, 1, Integer::sum);
                col.merge(c, 1, Integer::sum);
                mainDiag.merge(r - c, 1, Integer::sum);
                counterDiag.merge(r + c, 1, Integer::sum);
            }
        }
        int m = queries.length;
        int[] res = new int[m];
        for (int i = 0; i < m; ++i) {
            int r = queries[i][0];
            int c = queries[i][1];
            if (row.getOrDefault(r, 0) != 0
                    || col.getOrDefault(c, 0) != 0
                    || mainDiag.getOrDefault(r - c, 0) != 0
                    || counterDiag.getOrDefault(r + c, 0) != 0) {
                res[i] = 1;
                for (int x = Math.max(0, r - 1); x <= Math.min(n - 1, r + 1); ++x) {
                    for (int y = Math.max(0, c - 1); y <= Math.min(n - 1, c + 1); ++y) {
                        if (set.remove((long) x * n + y)) {
                            row.merge(x, -1, Integer::sum);
                            col.merge(y, -1, Integer::sum);
                            mainDiag.merge(x - y, -1, Integer::sum);
                            counterDiag.merge(x + y, -1, Integer::sum);
                        }
                    }
                }
            }

        }
        return res;
    }

    // 1278. 分割回文串 III (Palindrome Partitioning III)
    private int[][] p1278;
    private String s1278;
    private int n1278;
    private int[][] memo1278;
    private int k1278;

    public int palindromePartition(String s, int k) {
        this.n1278 = s.length();
        this.k1278 = k;
        this.p1278 = new int[n1278][n1278];
        this.memo1278 = new int[n1278][k];
        this.s1278 = s;
        for (int i = 0; i < n1278; ++i) {
            Arrays.fill(memo1278[i], -1);
        }
        for (int i = 0; i < n1278; ++i) {
            for (int j = i; j < n1278; ++j) {
                p1278[i][j] = cal1278(i, j);
            }
        }
        return dfs1278(0, 0);

    }

    private int cal1278(int i, int j) {
        int res = 0;
        while (i < j) {
            res += s1278.charAt(i++) != s1278.charAt(j--) ? 1 : 0;
        }
        return res;
    }

    private int dfs1278(int i, int j) {
        if (i == n1278) {
            return j == k1278 ? 0 : n1278;
        }
        if (j == k1278) {
            return n1278;
        }
        if (memo1278[i][j] != -1) {
            return memo1278[i][j];
        }
        int res = n1278;
        for (int x = i; x <= n1278 - k1278 + j; ++x) {
            res = Math.min(res, dfs1278(x + 1, j + 1) + p1278[i][x]);
        }
        return memo1278[i][j] = res;
    }

    // 355. 设计推特 (Design Twitter)
    class Twitter {
        private int globalTime;
        private Map<Integer, Node> userToTwitter;
        private Map<Integer, Set<Integer>> userToFollower;
        private Queue<Node> q;

        class Node {
            int time;
            int twitterId;
            Node next;

            Node(int time, int twitterId, Node next) {
                this.time = time;
                this.twitterId = twitterId;
                this.next = next;
            }

            Node(int time, int twitterId) {
                this.time = time;
                this.twitterId = twitterId;
            }
        }

        public Twitter() {
            userToTwitter = new HashMap<>();
            userToFollower = new HashMap<>();
            q = new PriorityQueue<>(new Comparator<Node>() {

                @Override
                public int compare(Node o1, Node o2) {
                    return Integer.compare(o2.time, o1.time);
                }

            });
        }

        public void postTweet(int userId, int tweetId) {
            Node head = new Node(globalTime++, tweetId);
            Node next = userToTwitter.getOrDefault(userId, null);
            head.next = next;
            userToTwitter.put(userId, head);
        }

        public List<Integer> getNewsFeed(int userId) {
            q.clear();
            List<Integer> res = new ArrayList<>();
            if (userToTwitter.get(userId) != null) {
                q.offer(userToTwitter.get(userId));
            }
            for (int followeeId : userToFollower.getOrDefault(userId, new HashSet<>())) {
                if (userToTwitter.get(followeeId) != null) {
                    q.offer(userToTwitter.get(followeeId));
                }
            }
            while (!q.isEmpty() && res.size() != 10) {
                Node node = q.poll();
                if (node == null) {
                    continue;
                }
                res.add(node.twitterId);
                node = node.next;
                if (node != null) {
                    q.offer(node);
                }
            }
            return res;
        }

        public void follow(int followerId, int followeeId) {
            userToFollower.computeIfAbsent(followerId, k -> new HashSet<>()).add(followeeId);
        }

        public void unfollow(int followerId, int followeeId) {
            userToFollower.getOrDefault(followerId, new HashSet<>()).remove(followeeId);
        }
    }

    // 1488. 避免洪水泛滥 (Avoid Flood in The City)
    public int[] avoidFlood(int[] rains) {
        int n = rains.length;
        TreeSet<Integer> set = new TreeSet<>();
        Map<Integer, Integer> last = new HashMap<>();
        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            if (rains[i] == 0) {
                set.add(i);
            } else {
                if (last.containsKey(rains[i])) {
                    int lastIndex = last.get(rains[i]);
                    Integer val = set.higher(lastIndex);
                    if (val == null) {
                        return new int[0];
                    }
                    res[val] = rains[i];
                    set.remove(val);
                }
                last.put(rains[i], i);
                res[i] = -1;
            }
        }
        for (int i = 0; i < n; ++i) {
            if (res[i] == 0) {
                res[i] = 1;
            }
        }
        return res;

    }

    // LCP 51. 烹饪料理
    public int perfectMenu(int[] materials, int[][] cookbooks, int[][] attribute, int limit) {
        int res = -1;
        int n = cookbooks.length;
        List<int[]> copyBooks = new ArrayList<>();
        List<int[]> copyAttribute = new ArrayList<>();
        search: for (int i = 0; i < n; ++i) {
            for (int j = 0; j < 5; ++j) {
                if (materials[j] < cookbooks[i][j]) {
                    continue search;
                }
            }
            copyBooks.add(cookbooks[i]);
            copyAttribute.add(attribute[i]);
        }
        n = copyBooks.size();
        if (n == 0) {
            return -1;
        }
        search: for (int i = 1; i < (1 << n); ++i) {
            int mask = i;
            int[] curCounts = new int[5];
            int index = 0;
            int x = 0;
            int y = 0;
            while (index < n && mask != 0) {
                if ((mask & 1) == 1) {
                    for (int j = 0; j < 5; ++j) {
                        curCounts[j] += copyBooks.get(index)[j];
                        if (curCounts[j] > materials[j]) {
                            continue search;
                        }
                    }
                    x += copyAttribute.get(index)[0];
                    y += copyAttribute.get(index)[1];
                }
                ++index;
                mask >>= 1;
            }
            if (y >= limit) {
                res = Math.max(res, x);
            }
        }
        return res;

    }

    // LCP 64. 二叉树灯饰
    public int closeLampInTree(TreeNode root) {
        int[] res = dfs_LCP_64(root);
        return res[1];

    }

    // return int[4]: 1、全亮；2、全灭； 3、根亮，其余全灭； 4、根灭，其余全亮
    private int[] dfs_LCP_64(TreeNode root) {
        if (root == null) {
            return new int[] { 0, 0, 0, 0 };
        }

        int[] l = dfs_LCP_64(root.left);
        int[] r = dfs_LCP_64(root.right);
        int min1 = Integer.MAX_VALUE;
        int min2 = Integer.MAX_VALUE;
        int min3 = Integer.MAX_VALUE;
        int min4 = Integer.MAX_VALUE;
        if (root.val == 1) {
            min1 = Math.min(Math.min(l[0] + r[0], l[1] + r[1] + 2), Math.min(l[2] + r[2] + 2, l[3] + r[3] + 2));
            min2 = Math.min(Math.min(l[0] + r[0] + 1, l[1] + r[1] + 1), Math.min(l[2] + r[2] + 1, l[3] + r[3] + 3));
            min3 = Math.min(Math.min(l[0] + r[0] + 2, l[1] + r[1]), Math.min(l[2] + r[2] + 2, l[3] + r[3] + 2));
            min4 = Math.min(Math.min(l[0] + r[0] + 1, l[1] + r[1] + 1), Math.min(l[2] + r[2] + 3, l[3] + r[3] + 1));
        } else {
            min1 = Math.min(Math.min(l[0] + r[0] + 1, l[1] + r[1] + 1), Math.min(l[2] + r[2] + 3, l[3] + r[3] + 1));
            min2 = Math.min(Math.min(l[0] + r[0] + 2, l[1] + r[1]), Math.min(l[2] + r[2] + 2, l[3] + r[3] + 2));
            min3 = Math.min(Math.min(l[0] + r[0] + 1, l[1] + r[1] + 1), Math.min(l[2] + r[2] + 1, l[3] + r[3] + 3));
            min4 = Math.min(Math.min(l[0] + r[0], l[1] + r[1] + 2), Math.min(l[2] + r[2] + 2, l[3] + r[3] + 2));
        }
        return new int[] { min1, min2, min3, min4 };
    }

    // 1616. 分割两个字符串得到回文串 (Split Two Strings to Make Palindrome)
    public boolean checkPalindromeFormation(String a, String b) {
        int n = a.length();
        if (isPalindrome1616(a, 0, n - 1) || isPalindrome1616(b, 0, n - 1)) {
            return true;
        }
        return check1616(a, b) || check1616(b, a);

    }

    private boolean isPalindrome1616(String s, int i, int j) {
        while (i < j) {
            if (s.charAt(i) != s.charAt(j)) {
                return false;
            }
            ++i;
            --j;
        }
        return true;
    }

    private boolean check1616(String a, String b) {
        int n = a.length();
        int i = 0;
        int j = n - 1;
        while (i < j) {
            if (a.charAt(i) != b.charAt(j)) {
                return isPalindrome1616(a, i, j) || isPalindrome1616(b, i, j);
            }
            ++i;
            --j;
        }
        return true;
    }

    // 面试题 05.08. 绘制直线
    public int[] drawLine(int length, int w, int x1, int x2, int y) {
        int[] res = new int[length];
        int mask = 0;
        int index = 0;
        for (int i = 0; i < w; ++i) {
            mask = (mask << 1) | ((i >= x1 && i <= x2) ? 1 : 0);
            if (i % 32 == 31) {
                res[w / 32 * y + index++] = mask;
                mask = 0;
            }
        }
        return res;

    }

    // 2250. 统计包含每个点的矩形数目 (Count Number of Rectangles Containing Each Point)
    public int[] countRectangles(int[][] rectangles, int[][] points) {
        int n = points.length;
        int[] res = new int[n];
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int[] r : rectangles) {
            map.computeIfAbsent(r[1], k -> new ArrayList<>()).add(r[0]);
        }
        for (List<Integer> vals : map.values()) {
            Collections.sort(vals);
        }
        for (int i = 0; i < n; ++i) {
            int x = points[i][0];
            int y = points[i][1];
            int cur = 0;
            for (int j = y; j <= 100; ++j) {
                cur += binarySearch2250(map.getOrDefault(j, new ArrayList<>()), x);
            }
            res[i] = cur;
        }
        return res;
    }

    private int binarySearch2250(List<Integer> list, int target) {
        if (list.isEmpty()) {
            return 0;
        }
        int n = list.size();
        if (target <= list.get(0)) {
            return n;
        }
        if (target > list.get(n - 1)) {
            return 0;
        }
        int left = 0;
        int right = n - 1;
        int res = 0;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (list.get(mid) >= target) {
                res = n - mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    // 982. 按位与为零的三元组 (Triples with Bitwise AND Equal To Zero)
    public int countTriplets(int[] nums) {
        int[] counts = new int[1 << 16];
        for (int x : nums) {
            for (int y : nums) {
                ++counts[x & y];
            }
        }
        int res = 0;
        for (int z : nums) {
            for (int i = 0; i < counts.length; ++i) {
                if ((i & z) == 0) {
                    res += counts[i];
                }
            }
        }
        return res;

    }

    // 982. 按位与为零的三元组 (Triples with Bitwise AND Equal To Zero) --枚举子集技巧
    public int countTriplets2(int[] nums) {
        int[] counts = new int[1 << 16];
        for (int x : nums) {
            for (int y : nums) {
                ++counts[x & y];
            }
        }
        int res = 0;
        for (int x : nums) {
            x ^= 0xffff;
            int sub = x;
            do {
                res += counts[sub];
                sub = (sub - 1) & x;
            } while (sub != x);

        }
        return res;

    }

    // 6312. 最小和分割 (Split With Minimum Sum)
    public int splitNum(int num) {
        int[] cnt = new int[10];
        while (num != 0) {
            ++cnt[num % 10];
            num /= 10;
        }
        int[] res = new int[2];
        int i = 0;
        for (int j = 1; j < 10; ++j) {
            while (cnt[j]-- != 0) {
                res[i] = res[i] * 10 + j;
                i ^= 1;
            }
        }
        return res[0] + res[1];

    }

    // 6311. 统计染色格子数 (Count Total Number of Colored Cells)
    public long coloredCells(int n) {
        long res = 1l;
        for (int i = 2; i <= n; ++i) {
            res = res + (i - 1) * 4;
        }
        return res;

    }

    // 2580. 统计将重叠区间合并成组的方案数 (Count Ways to Group Overlapping Ranges)
    public int countWays(int[][] ranges) {
        Arrays.sort(ranges, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });
        int res = 1;
        final int MOD = (int) (1e9 + 7);
        int i = 0;
        int n = ranges.length;
        while (i < n) {
            int right = ranges[i][1];
            int j = i;
            while (j < n && ranges[j][0] <= right) {
                right = Math.max(right, ranges[j][1]);
                ++j;
            }
            res <<= 1;
            res %= MOD;
            i = j;
        }
        return res;

    }

    // 2581. 统计可能的树根数目 (Count Number of Possible Root Nodes)
    private List<Integer>[] g2581;
    private Set<Long> set2581;
    private int res2581;
    private int k2581;
    private int s2581;

    public int rootCount(int[][] edges, int[][] guesses, int k) {
        int n = edges.length + 1;
        this.g2581 = new ArrayList[n + 1];
        Arrays.setAll(g2581, o -> new ArrayList<>());
        this.set2581 = new HashSet<>();
        this.k2581 = k;
        for (int[] e : edges) {
            g2581[e[0]].add(e[1]);
            g2581[e[1]].add(e[0]);
        }
        for (int[] gu : guesses) {
            set2581.add(gu[0] * (long) 1e5 + gu[1]);
        }
        dfs2581(0, -1);
        res2581 = s2581 >= k ? 1 : 0;
        reRoot2581(0, -1, s2581);
        return res2581;

    }

    private void reRoot2581(int x, int fa, int cnt) {
        for (int y : g2581[x]) {
            if (y != fa) {
                int copy = cnt;
                if (set2581.contains(x * (long) 1e5 + y)) {
                    --copy;
                }
                if (set2581.contains(y * (long) 1e5 + x)) {
                    ++copy;
                }
                if (copy >= k2581) {
                    ++res2581;
                }
                reRoot2581(y, x, copy);
            }
        }
    }

    private void dfs2581(int x, int fa) {
        for (int y : g2581[x]) {
            if (y != fa) {
                if (set2581.contains(x * (long) 1e5 + y)) {
                    ++s2581;
                }
                dfs2581(y, x);
            }
        }
    }

    // 2582. 递枕头 (Pass the Pillow)
    public int passThePillow(int n, int time) {
        int t = time / (n - 1);
        return (t & 1) == 1 ? n - time % (n - 1) : 1 + time % (n - 1);

    }

    // 2583. 二叉树中的第 K 大层和 (Kth Largest Sum in a Binary Tree)
    public long kthLargestLevelSum(TreeNode root, int k) {
        List<Long> list = new ArrayList<>();
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int size = q.size();
            long curSum = 0l;
            for (int i = 0; i < size; ++i) {
                TreeNode node = q.poll();
                curSum += node.val;
                if (node.left != null) {
                    q.offer(node.left);
                }
                if (node.right != null) {
                    q.offer(node.right);
                }
            }
            list.add(curSum);
        }
        if (k > list.size()) {
            return -1;
        }
        Collections.sort(list, new Comparator<Long>() {

            @Override
            public int compare(Long o1, Long o2) {
                return Long.compare(o2, o1);
            }

        });
        return list.get(k - 1);

    }

    // 2584. 分割数组使乘积互质 (Split the Array to Make Coprime Products)
    public int findValidSplit(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        int n = nums.length;
        if (n == 1) {
            return -1;
        }
        if (nums[0] == 1) {
            return 0;
        }
        for (int i = 0; i < n; ++i) {
            int num = nums[i];
            for (int j = 2; j * j <= num; ++j) {
                while (num % j == 0) {
                    map.put(j, i);
                    num /= j;
                }
            }
            if (num != 1) {
                map.put(num, i);
            }

        }

        int right = -1;
        for (int i = 0; i < n; ++i) {
            int num = nums[i];
            if (num == 1) {
                continue;
            }
            int curRight = i;
            for (int j = 2; j * j <= num; ++j) {
                while (num % j == 0) {
                    int r = map.get(j);
                    curRight = Math.max(r, curRight);
                    num /= j;
                }
            }
            if (num != 1) {
                curRight = Math.max(curRight, map.get(num));
            }
            if (curRight == n - 1) {
                return -1;
            }
            right = Math.max(right, curRight);
            if (i == right) {
                return right;
            }
        }
        return -1;

    }

    // 2585. 获得分数的方法数 (Number of Ways to Earn Points)
    private int target2585;
    private int[][] types2585;
    private int n2585;
    private int[][] memo2585;

    public int waysToReachTarget(int target, int[][] types) {
        this.target2585 = target;
        this.types2585 = types;
        this.n2585 = types.length;
        this.memo2585 = new int[n2585][target + 1];
        for (int i = 0; i < n2585; ++i) {
            Arrays.fill(memo2585[i], -1);
        }
        return dfs2585(0, 0);

    }

    private int dfs2585(int i, int j) {
        if (j == target2585) {
            return 1;
        }
        if (i == n2585) {
            return 0;
        }
        if (memo2585[i][j] != -1) {
            return memo2585[i][j];
        }
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        for (int k = 0; k <= types2585[i][0] && j + k * types2585[i][1] <= target2585; ++k) {
            res += dfs2585(i + 1, j + k * types2585[i][1]);
            res %= MOD;
        }
        return memo2585[i][j] = res;
    }

    // 688. 骑士在棋盘上的概率 (Knight Probability in Chessboard)
    private double[][][] memo688;
    private int n688;
    private int[][] dirs688 = { { -2, 1 }, { -1, 2 }, { 1, 2 }, { 2, 1 }, { 2, -1 }, { 1, -2 }, { -1, -2 },
            { -2, -1 } };

    public double knightProbability(int n, int k, int row, int column) {
        memo688 = new double[n][n][k + 1];
        this.n688 = n;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                Arrays.fill(memo688[i][j], -1.0d);
            }
        }
        return dfs688(row, column, k);
    }

    private double dfs688(int r, int c, int k) {
        if (r < 0 || r >= n688 || c < 0 || c >= n688) {
            return 0d;
        }
        if (k == 0) {
            return 1.0d;
        }
        if (memo688[r][c][k] >= 0d) {
            return memo688[r][c][k];
        }
        double res = 0d;
        for (int[] d : dirs688) {
            int nr = r + d[0];
            int nc = c + d[1];
            res += dfs688(nr, nc, k - 1) / 8.0d;
        }

        return memo688[r][c][k] = res;
    }

    // 1563. 石子游戏 V (Stone Game V)
    private int[][] memo1563;
    private int[] stoneValue1563;

    public int stoneGameV(int[] stoneValue) {
        int n = stoneValue.length;
        this.stoneValue1563 = stoneValue;
        memo1563 = new int[n][n];
        return dfs1563(0, n - 1);

    }

    private int dfs1563(int i, int j) {
        if (i == j) {
            return 0;
        }
        if (memo1563[i][j] != 0) {
            return memo1563[i][j];
        }
        int sum = 0;
        for (int k = i; k <= j; ++k) {
            sum += stoneValue1563[k];
        }
        int leftSum = 0;
        for (int k = i; k < j; ++k) {
            leftSum += stoneValue1563[k];
            int rightSum = sum - leftSum;
            if (leftSum < rightSum) {
                memo1563[i][j] = Math.max(memo1563[i][j], dfs1563(i, k) + leftSum);
            } else if (leftSum > rightSum) {
                memo1563[i][j] = Math.max(memo1563[i][j], dfs1563(k + 1, j) + rightSum);
            } else {
                memo1563[i][j] = Math.max(memo1563[i][j], Math.max(dfs1563(i, k), dfs1563(k + 1, j)) + leftSum);
            }
        }
        return memo1563[i][j];
    }

    // 1712. 将数组分成三个子数组的方案数 (Ways to Split Array Into Three Subarrays)
    public int waysToSplit(int[] nums) {
        int res = 0;
        int n = nums.length;
        int[] pre = new int[n];
        pre[0] = nums[0];
        for (int i = 1; i < n; ++i) {
            pre[i] = pre[i - 1] + nums[i];
        }
        int cut = pre[n - 1] / 3;
        final int MOD = (int) (1e9 + 7);
        for (int i = 0; i < n && pre[i] <= cut; ++i) {
            // 第二个分割位置的左边界
            int left = binarySearch1712(pre, i + 1, n - 1, pre[i] * 2);
            // 第二个分割位置的右边界
            int right = binarySearch1712_2(pre, i + 1, n - 1, pre[i] + (pre[n - 1] - pre[i]) / 2);
            if (right >= left) {
                res = (res + right - left + 1) % MOD;
            }
        }
        return res;

    }

    // 找大于等于target的最小值对应的索引
    private int binarySearch1712(int[] nums, int left, int right, int target) {
        while (left < right) {
            int mid = left + ((right - left) >>> 1);
            if (nums[mid] >= target) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    // 找小于等于target的最大值对应的索引
    private int binarySearch1712_2(int[] nums, int left, int right, int target) {
        while (left < right) {
            int mid = left + ((right - left) >>> 1);
            if (nums[mid] <= target) {
                // res = mid;
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return right - 1;

    }

    // 487. 最大连续1的个数 II (Max Consecutive Ones II) --plus
    public int findMaxConsecutiveOnes(int[] nums) {
        int n = nums.length;
        int left = 0;
        int right = 0;
        int cnt0 = 0;
        int res = 0;
        while (right < n) {
            cnt0 += 1 - nums[right];
            while (cnt0 > 1) {
                cnt0 -= 1 - nums[left++];
            }
            res = Math.max(res, right - left + 1);
            ++right;
        }
        return res;
    }

    // 2297. 跳跃游戏 VIII (Jump Game VIII) --plus
    private long[] memo2297;
    private int[] nums2297;
    private int[] cost2297;
    private int n2297;
    private int[] rightCeiling2297;
    private int[] rightLower2297;

    public long minCost2297(int[] nums, int[] costs) {
        this.n2297 = nums.length;
        this.nums2297 = nums;
        this.cost2297 = costs;
        // rightCeiling[i] ： i右侧，第一个大于等于nums[i]的数的索引，若不存在，则为-1
        this.rightCeiling2297 = new int[n2297];
        // rightLower[i] ： i右侧，第一个小于nums[i]的数的索引，若不存在，则为-1
        this.rightLower2297 = new int[n2297];
        Arrays.fill(rightCeiling2297, -1);
        Arrays.fill(rightLower2297, -1);
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < n2297; ++i) {
            while (!stack.isEmpty() && nums[stack.peek()] <= nums[i]) {
                rightCeiling2297[stack.pop()] = i;
            }
            stack.push(i);
        }
        stack.clear();
        for (int i = 0; i < n2297; ++i) {
            while (!stack.isEmpty() && nums[stack.peek()] > nums[i]) {
                rightLower2297[stack.pop()] = i;
            }
            stack.push(i);
        }
        memo2297 = new long[n2297];
        Arrays.fill(memo2297, Long.MAX_VALUE);
        return dfs2297(0);

    }

    private long dfs2297(int i) {
        if (i >= n2297 - 1) {
            return 0;
        }
        if (memo2297[i] != Long.MAX_VALUE) {
            return memo2297[i];
        }
        long res = Long.MAX_VALUE;
        res = Math.min(res, dfs2297(i + 1) + cost2297[i + 1]);
        int index = nums2297[i + 1] < nums2297[i] ? rightCeiling2297[i] : rightLower2297[i];
        if (index != -1) {
            res = Math.min(res, dfs2297(index) + cost2297[index]);
        }
        return memo2297[i] = res;
    }

    // 265. 粉刷房子 II (Paint House II) --plus
    private int[][] costs265;
    private int n265;
    private int k265;
    private int[][] memo265;

    public int minCostII(int[][] costs) {
        this.n265 = costs.length;
        this.k265 = costs[0].length;
        this.costs265 = costs;
        memo265 = new int[n265][k265 + 1];
        for (int i = 0; i < n265; ++i) {
            Arrays.fill(memo265[i], Integer.MAX_VALUE);
        }
        return dfs265(0, k265);

    }

    private int dfs265(int i, int lastColor) {
        if (i == n265) {
            return 0;
        }
        if (memo265[i][lastColor] != Integer.MAX_VALUE) {
            return memo265[i][lastColor];
        }
        int min = Integer.MAX_VALUE;
        for (int j = 0; j < k265; ++j) {
            if (lastColor != j) {
                min = Math.min(min, dfs265(i + 1, j) + costs265[i][j]);
            }
        }
        return memo265[i][lastColor] = min;
    }

    // 1216. 验证回文字符串 III (Valid Palindrome III) --plus
    private boolean[][] isPalindrome1216;
    private int n1216;
    private String s1216;
    private int[][] memo1216;

    public boolean isValidPalindrome(String s, int k) {
        this.n1216 = s.length();
        this.s1216 = s;
        isPalindrome1216 = new boolean[n1216][n1216];
        for (int i = n1216 - 1; i >= 0; --i) {
            for (int j = i; j < n1216; ++j) {
                if (s.charAt(i) == s.charAt(j) && (j - i < 2 || isPalindrome1216[i + 1][j - 1])) {
                    isPalindrome1216[i][j] = true;
                }
            }
        }
        memo1216 = new int[n1216][n1216];
        for (int i = 0; i < n1216; ++i) {
            Arrays.fill(memo1216[i], n1216);
        }
        return dfs1216(0, n1216 - 1) <= k;
    }

    private int dfs1216(int i, int j) {
        if (i > j) {
            return 0;
        }
        if (isPalindrome1216[i][j]) {
            return 0;
        }
        if (memo1216[i][j] != n1216) {
            return memo1216[i][j];
        }
        int min = n1216;
        if (s1216.charAt(i) == s1216.charAt(j)) {
            min = Math.min(min, dfs1216(i + 1, j - 1));
        } else {
            min = Math.min(min, dfs1216(i, j - 1) + 1);
            min = Math.min(min, dfs1216(i + 1, j) + 1);
        }
        return memo1216[i][j] = min;
    }

    // 2378. 选择边来最大化树的得分 (Choose Edges to Maximize Score in a Tree) --plus
    private Map<Integer, List<int[]>> g2378;

    public long maxScore(int[][] edges) {
        g2378 = new HashMap<>();
        int n = edges.length;
        for (int i = 0; i < n; ++i) {
            int fa = edges[i][0];
            int weight = edges[i][1];
            if (fa == -1) {
                continue;
            }
            g2378.computeIfAbsent(fa, k -> new ArrayList<>()).add(new int[] { i, weight });
        }
        // dp[0] = 孩子节点都没选
        // dp[1] = 孩子节点选了一个
        long[] dp = dfs2378(0);
        return Math.max(0, Math.max(dp[0], dp[1]));

    }

    private long[] dfs2378(int x) {
        long max1 = 0l;
        long max2 = 0l;
        List<long[]> list = new ArrayList<>();
        for (int[] children : g2378.getOrDefault(x, new ArrayList<>())) {
            int y = children[0];
            long[] cur = dfs2378(y);
            list.add(new long[] { cur[0], cur[1] });
            max1 += Math.max(cur[0], cur[1]);
        }
        int index = 0;
        for (int[] children : g2378.getOrDefault(x, new ArrayList<>())) {
            long[] item = list.get(index++);
            max2 = Math.max(max2, max1 - Math.max(item[0], item[1]) + item[0] + children[1]);
        }
        return new long[] { max1, max2 };
    }

    // 723. 粉碎糖果 (Candy Crush) --plus
    public int[][] candyCrush(int[][] board) {
        int m = board.length;
        int n = board[0].length;
        boolean cycle = true;
        while (cycle) {
            cycle = false;
            boolean[][] delete = new boolean[m][n];
            for (int i = 0; i < m; ++i) {
                for (int j = 1; j < n - 1; ++j) {
                    if (board[i][j] != 0 && board[i][j - 1] == board[i][j] && board[i][j + 1] == board[i][j]) {
                        delete[i][j - 1] = delete[i][j] = delete[i][j + 1] = true;
                        cycle = true;
                    }
                }
            }
            for (int j = 0; j < n; ++j) {
                for (int i = 1; i < m - 1; ++i) {
                    if (board[i][j] != 0 && board[i - 1][j] == board[i][j] && board[i + 1][j] == board[i][j]) {
                        delete[i - 1][j] = delete[i][j] = delete[i + 1][j] = true;
                        cycle = true;
                    }
                }
            }
            for (int i = m - 1; i >= 0; --i) {
                for (int j = 0; j < n; ++j) {
                    if (delete[i][j]) {
                        int index = i;
                        while (index >= 0 && delete[index][j] == true && board[index][j] != 0) {
                            --index;
                        }
                        if (index >= 0) {
                            board[i][j] = board[index][j];
                            delete[i][j] = false;
                            delete[index][j] = true;
                        }
                    }
                }
            }
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (delete[i][j]) {
                        board[i][j] = 0;
                    }
                }
            }
        }
        return board;

    }

    // 1087. 花括号展开 (Brace Expansion)
    List<String> list1087;

    public String[] expand(String s) {
        list1087 = new ArrayList<>();
        dfs1087(s, 0, new StringBuilder());
        int n = list1087.size();
        Collections.sort(list1087);
        return list1087.toArray(new String[n]);

    }

    private void dfs1087(String s, int start, StringBuilder builder) {
        if (start == s.length()) {
            list1087.add(builder.toString());
            return;
        }
        int i = start;
        while (i < s.length() && s.charAt(i) == ',') {
            ++i;
        }
        if (i < s.length()) {
            if (Character.isLetter(s.charAt(i))) {
                builder.append(s.charAt(i));
                dfs1087(s, i + 1, builder);
                builder.deleteCharAt(builder.length() - 1);
            } else {
                List<Character> list = new ArrayList<>();
                while (s.charAt(i) != '}') {
                    if (Character.isLetter(s.charAt(i))) {
                        list.add(s.charAt(i));
                    }
                    ++i;
                }
                for (char c : list) {
                    builder.append(c);
                    dfs1087(s, i + 1, builder);
                    builder.deleteCharAt(builder.length() - 1);
                }
            }
        }
    }

    // 666. 路径总和 IV (Path Sum IV) --plus
    public int pathSum(int[] nums) {
        Map<Integer, Integer> counts = new HashMap<>();
        int res = 0;
        for (int i = nums.length - 1; i >= 0; --i) {
            int key = nums[i] / 10;
            res += (nums[i] % 10) * counts.getOrDefault(key, 1);
            int fa = (nums[i] / 100 - 1) * 10 + (nums[i] / 10 % 10 + 1) / 2;
            counts.merge(fa, counts.getOrDefault(key, 1), Integer::sum);
        }
        return res;
    }

    // 2052. 将句子分隔成行的最低成本 (Minimum Cost to Separate Sentence Into Rows) --plus
    private int n2052;
    private int[] arr2052;
    private int k2052;
    private int[] memo2052;
    private int last2052;

    public int minimumCost(String sentence, int k) {
        String[] sentences = sentence.split("\\s+");
        this.n2052 = sentences.length;
        arr2052 = new int[n2052];
        for (int i = 0; i < n2052; ++i) {
            arr2052[i] = sentences[i].length();
        }
        last2052 = n2052 - 1;
        int suf = sentences[n2052 - 1].length();
        for (int i = n2052 - 2; i >= 0; --i) {
            if (sentences[i].length() + suf + 1 > k) {
                break;
            }
            suf += sentences[i].length() + 1;
            last2052 = i;
        }

        this.k2052 = k;
        this.memo2052 = new int[n2052];
        Arrays.fill(memo2052, Integer.MAX_VALUE);
        return dfs2052(0);

    }

    private int dfs2052(int i) {
        if (i >= last2052) {
            return 0;
        }
        if (memo2052[i] != Integer.MAX_VALUE) {
            return memo2052[i];
        }
        int min = Integer.MAX_VALUE;
        min = Math.min(min, dfs2052(i + 1) + (k2052 - arr2052[i]) * (k2052 - arr2052[i]));
        int len = arr2052[i];
        int j = i + 1;
        while (j < n2052 && len + arr2052[j] + 1 <= k2052) {
            len += arr2052[j] + 1;
            min = Math.min(min, dfs2052(j + 1) + (k2052 - len) * (k2052 - len));
            ++j;
        }
        return memo2052[i] = min;
    }

    // 296. 最佳的碰头地点 (Best Meeting Point) --中位数 plus
    public int minTotalDistance(int[][] grid) {
        List<Integer> rows = getRowsIndex(grid);
        List<Integer> cols = getColsIndex(grid);
        return getTotalCosts(rows) + getTotalCosts(cols);

    }

    private int getTotalCosts(List<Integer> list) {
        int i = 0;
        int j = list.size() - 1;
        int cost = 0;
        while (i < j) {
            cost += list.get(j--) - list.get(i++);
        }
        return cost;
    }

    private List<Integer> getColsIndex(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        List<Integer> list = new ArrayList<>();
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                if (grid[i][j] == 1) {
                    list.add(j);
                }
            }
        }
        return list;
    }

    private List<Integer> getRowsIndex(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    list.add(i);
                }
            }
        }
        return list;
    }

    // 562. 矩阵中最长的连续1线段 (Longest Line of Consecutive One in Matrix) --plus
    public int longestLine(int[][] mat) {
        int m = mat.length;
        int n = mat[0].length;
        Map<Integer, Integer> map = new HashMap<>();
        Map<Integer, Integer> counterMap = new HashMap<>();
        int res = 0;
        int max = 0;
        // row
        for (int i = 0; i < m; ++i) {
            max = 0;
            for (int j = 0; j < n; ++j) {
                // row
                if (mat[i][j] == 1) {
                    ++max;
                } else {
                    max = 0;
                }
                res = Math.max(res, max);
                // mainDiagonal
                if (map.containsKey(i - j)) {
                    if (mat[i][j] == 1) {
                        map.merge(i - j, 1, Integer::sum);
                    } else {
                        map.put(i - j, 0);
                    }
                } else {
                    map.put(i - j, mat[i][j]);
                }
                res = Math.max(map.get(i - j), res);
            }
            // counterDiagonal
            for (int j = n - 1; j >= 0; --j) {
                if (counterMap.containsKey(i + j)) {
                    if (mat[i][j] == 1) {
                        counterMap.merge(i + j, 1, Integer::sum);
                    } else {
                        counterMap.put(i + j, 0);
                    }
                } else {
                    counterMap.put(i + j, mat[i][j]);
                }
                res = Math.max(counterMap.get(i + j), res);
            }
        }
        // col
        for (int j = 0; j < n; ++j) {
            max = 0;
            for (int i = 0; i < m; ++i) {
                if (mat[i][j] == 1) {
                    ++max;
                } else {
                    max = 0;
                }
                res = Math.max(res, max);
            }
        }
        return res;

    }

    // 1231. 分享巧克力 (Divide Chocolate) --plus
    public int maximizeSweetness(int[] sweetness, int k) {
        int left = Integer.MAX_VALUE;
        int right = 0;
        for (int s : sweetness) {
            left = Math.min(left, s);
            right += s;
        }
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (check1231(sweetness, mid) < k + 1) {
                right = mid - 1;
            } else {
                res = mid;
                left = mid + 1;
            }
        }
        return res;

    }

    private int check1231(int[] sweetness, int target) {
        int count = 0;
        int sum = 0;
        for (int s : sweetness) {
            sum += s;
            if (sum >= target) {
                ++count;
                sum = 0;
            }
        }
        return count;
    }

    // 2387. 行排序矩阵的中位数 (Median of a Row Wise Sorted Matrix) --plus
    public int matrixMedian(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        Queue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(grid[o1[0]][o1[1]], grid[o2[0]][o2[1]]);
            }

        });

        for (int i = 0; i < m; ++i) {
            q.offer(new int[] { i, 0 });
        }
        int cur = 0;
        while (!q.isEmpty()) {
            int[] p = q.poll();
            int x = p[0];
            int y = p[1];

            if (cur == m * n / 2) {
                return grid[x][y];
            }
            if (++y < n) {
                q.offer(new int[] { x, y });
            }
            ++cur;
        }
        return -1;

    }

    // 6315. 统计范围内的元音字符串数 (Count the Number of Vowel Strings in Range)
    public int vowelStrings(String[] words, int left, int right) {
        int res = 0;
        for (int i = left; i <= right; ++i) {
            String word = words[i];
            if (check6315(word.charAt(0)) && check6315(word.charAt(word.length() - 1))) {
                ++res;
            }
        }
        return res;

    }

    private boolean check6315(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }

    // 6316. 重排数组以得到最大前缀分数 (Rearrange Array to Maximize Prefix Score)
    public int maxScore(int[] nums) {
        Arrays.sort(nums);
        int res = 0;
        long pre = 0l;
        for (int i = nums.length - 1; i >= 0; --i) {
            pre += nums[i];
            if (pre <= 0) {
                break;
            }
            ++res;
        }
        return res;
    }

    // 6317. 统计美丽子数组数目 (Count the Number of Beautiful Subarrays)
    public long beautifulSubarrays(int[] nums) {
        long res = 0l;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int mask = 0;
        for (int num : nums) {
            mask ^= num;
            res = res + map.getOrDefault(mask, 0);
            map.merge(mask, 1, Integer::sum);
        }
        return res;
    }

    // 1416. 恢复数组 (Restore The Array)
    private int n1416;
    private int[] memo1416;
    private int k1416;
    private String s1416;

    public int numberOfArrays(String s, int k) {
        this.n1416 = s.length();
        this.memo1416 = new int[n1416];
        Arrays.fill(memo1416, -1);
        this.k1416 = k;
        this.s1416 = s;
        return dfs1416(0);

    }

    private int dfs1416(int i) {
        if (i == n1416) {
            return 1;
        }
        if (s1416.charAt(i) == '0') {
            return 0;
        }
        if (memo1416[i] != -1) {
            return memo1416[i];
        }
        int res = 0;
        long num = 0l;
        final int MOD = (int) (1e9 + 7);
        for (int j = i; j < n1416; ++j) {
            num = num * 10 + s1416.charAt(j) - '0';
            if (num > k1416) {
                break;
            }
            res = (res + dfs1416(j + 1)) % MOD;
        }

        return memo1416[i] = res;
    }

    // 1770. 执行乘法运算的最大分数 (Maximum Score from Performing Multiplication Operations)
    private int[][] memo1770;
    private int m1770;
    private int n1770;
    private int[] nums1770;
    private int[] multipliers1770;

    public int maximumScore(int[] nums, int[] multipliers) {
        this.n1770 = nums.length;
        this.m1770 = multipliers.length;
        this.nums1770 = nums;
        this.multipliers1770 = multipliers;
        // memo[i][j] : 左边选i个数，右边选j个数的最大值
        memo1770 = new int[m1770 + 1][m1770 + 1];
        for (int i = 0; i < m1770 + 1; ++i) {
            Arrays.fill(memo1770[i], Integer.MIN_VALUE);
        }
        return dfs1770(0, 0);

    }

    private int dfs1770(int i, int j) {
        if (i + j >= m1770) {
            return 0;
        }
        if (memo1770[i][j] != Integer.MIN_VALUE) {
            return memo1770[i][j];
        }
        return memo1770[i][j] = Math.max(dfs1770(i + 1, j) + nums1770[i] * multipliers1770[i + j],
                dfs1770(i, j + 1) + nums1770[n1770 - j - 1] * multipliers1770[i + j]);
    }

    // 1197. 进击的骑士 (Minimum Knight Moves)
    public int minKnightMoves(int x, int y) {
        if (x == 0 && y == 0) {
            return 0;
        }
        x = Math.abs(x);
        y = Math.abs(y);
        Set<Integer> visited = new HashSet<>();
        Queue<int[]> q = new LinkedList<>();
        q.offer(new int[] { 0, 0 });
        visited.add(0);
        int res = 0;
        int[][] directions = { { -2, 1 }, { -2, -1 }, { -1, 2 }, { -1, -2 }, { 1, 2 }, { 1, -2 }, { 2, 1 }, { 2, -1 } };
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                int[] cur = q.poll();
                int x0 = cur[0];
                int y0 = cur[1];
                if (x0 == x && y0 == y) {
                    return res;
                }
                for (int[] d : directions) {
                    int nx = x0 + d[0];
                    int ny = y0 + d[1];
                    if (nx >= -5 && nx <= x + 5 && ny >= -5 && ny <= y + 5 && !visited.contains(nx * 310 + ny)) {
                        visited.add(nx * 310 + ny);
                        q.offer(new int[] { nx, ny });
                    }
                }
            }
            ++res;
        }
        return -1;
    }

    // 2589. 完成所有任务的最少时间 (Minimum Time to Complete All Tasks)
    public int findMinimumTime(int[][] tasks) {
        int res = 0;
        int[] idle = new int[2001];
        Arrays.sort(tasks, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[1], o2[1]);
            }

        });

        for (int[] task : tasks) {
            int start = task[0];
            int end = task[1];
            int duration = task[2];
            for (int i = start; i <= end; ++i) {
                duration -= idle[i];
            }
            for (int i = end; i >= start && duration > 0; --i) {
                if (idle[i] == 0) {
                    idle[i] = 1;
                    ++res;
                    --duration;
                }

            }
        }
        return res;

    }

    // 1847. 最近的房间 (Closest Room)
    public int[] closestRoom(int[][] rooms, int[][] queries) {
        int n = rooms.length;
        int k = queries.length;
        int[] res = new int[k];
        Arrays.fill(res, -1);
        Integer[] ids = IntStream.range(0, k).boxed().toArray(Integer[]::new);
        Arrays.sort(ids, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(queries[o2][1], queries[o1][1]);
            }

        });

        Arrays.sort(rooms, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o2[1], o1[1]);
            }

        });

        TreeSet<Integer> set = new TreeSet<>();
        int i = 0;
        for (int id : ids) {
            int minSize = queries[id][1];
            while (i < n && rooms[i][1] >= minSize) {
                set.add(rooms[i][0]);
                ++i;
            }
            int preferId = queries[id][0];
            Integer floorId = set.floor(preferId);
            Integer ceilingId = set.ceiling(preferId);
            if (floorId != null || ceilingId != null) {
                if (floorId == null) {
                    res[id] = ceilingId;
                } else if (ceilingId == null) {
                    res[id] = floorId;
                } else {
                    if (Math.abs(floorId - preferId) <= Math.abs(ceilingId - preferId)) {
                        res[id] = floorId;
                    } else {
                        res[id] = ceilingId;
                    }
                }
            }
        }
        return res;

    }

    // 1320. 二指输入的的最小距离 (Minimum Distance to Type a Word Using Two Fingers)
    private int[][] memo1320;
    private String word1320;
    private int res1320;
    private int n1320;

    public int minimumDistance(String word) {
        this.n1320 = word.length();
        // memo[i][j] : 不指向当前字符的另一个手指的位置（因为其中一个手指一定指向的是当前字符），输入了第i个字符后的最短移动距离
        memo1320 = new int[n1320][26];
        for (int i = 0; i < n1320; ++i) {
            Arrays.fill(memo1320[i], -1);
        }
        this.word1320 = word;
        this.res1320 = Integer.MAX_VALUE;
        for (int j = 0; j < 26; ++j) {
            res1320 = Math.min(res1320, dfs1320(1, j));
        }
        return res1320;
    }

    private int dfs1320(int i, int j) {
        if (i == n1320) {
            return 0;
        }
        if (memo1320[i][j] != -1) {
            return memo1320[i][j];
        }
        int cur = word1320.charAt(i) - 'A';
        int pre = word1320.charAt(i - 1) - 'A';
        return memo1320[i][j] = Math.min(
                dfs1320(i + 1, j) + getDis1320(pre, cur),
                dfs1320(i + 1, pre) + getDis1320(j, cur));
    }

    private int getDis1320(int pos1, int pos2) {
        int row1 = pos1 / 6;
        int col1 = pos1 % 6;
        int row2 = pos2 / 6;
        int col2 = pos2 % 6;
        return Math.abs(row1 - row2) + Math.abs(col1 - col2);
    }

    // 1931. 用三种不同颜色为网格涂色 (Painting a Grid With Three Different Colors)
    private int[][] memo1931;
    private int m1931;
    private int n1931;
    private Set<Integer> set1931;

    public int colorTheGrid(int m, int n) {
        this.memo1931 = new int[n][(int) Math.pow(3, m)];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(memo1931[i], -1);
        }
        this.m1931 = m;
        this.n1931 = n;
        this.set1931 = new HashSet<>();
        for (int i = 0; i < (int) Math.pow(3, m); ++i) {
            if (check1931(i)) {
                set1931.add(i);
            }
        }
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        for (int i : set1931) {
            res = (res + dfs1931(1, i)) % MOD;
        }
        return res;
    }

    private int dfs1931(int j, int i) {
        if (j == n1931) {
            return 1;
        }
        if (memo1931[j][i] != -1) {
            return memo1931[j][i];
        }
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        for (int k : set1931) {
            if (legal1931(k, i)) {
                res = (res + dfs1931(j + 1, k)) % MOD;
            }
        }
        return memo1931[j][i] = res;
    }

    private boolean legal1931(int a, int b) {
        int cnt = m1931;
        while (cnt-- > 0) {
            if (a % 3 == b % 3) {
                return false;
            }
            a /= 3;
            b /= 3;
        }
        return true;
    }

    private boolean check1931(int i) {
        int pre = -1;
        int cnt = m1931;
        while (cnt-- > 0) {
            if (pre == i % 3) {
                return false;
            }
            pre = i % 3;
            i /= 3;
        }
        return true;
    }

    // 549. 二叉树中最长的连续序列 --plus
    private int res549;

    public int longestConsecutive(TreeNode root) {
        dfs549(root);
        return res549;
    }

    public int[] dfs549(TreeNode root) {
        if (root == null) {
            return new int[] { 0, 0 };
        }
        int inr = 1, dcr = 1;
        if (root.left != null) {
            int[] l = dfs549(root.left);
            if (root.val == root.left.val + 1) {
                dcr = l[1] + 1;
            } else if (root.val == root.left.val - 1) {
                inr = l[0] + 1;
            }
        }
        if (root.right != null) {
            int[] r = dfs549(root.right);
            if (root.val == root.right.val + 1) {
                dcr = Math.max(dcr, r[1] + 1);
            } else if (root.val == root.right.val - 1) {
                inr = Math.max(inr, r[0] + 1);
            }
        }
        res549 = Math.max(res549, dcr + inr - 1);
        return new int[] { inr, dcr };
    }

}
