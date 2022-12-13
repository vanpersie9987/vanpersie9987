import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.IntStream;

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

    // LCP 56. 信物传送
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
        Queue<int[]> queue = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return steps[o1[0]][o1[1]] - steps[o2[0]][o2[1]];
            }

        });
        queue.offer(start);
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0];
            int y = cur[1];
            int signDir = map.get(matrix[x].charAt(y));
            int step = steps[x][y];
            for (int i = 0; i < 4; ++i) {
                int nStep = step + (i == signDir ? 0 : 1);
                int nx = x + directions[i][0];
                int ny = y + directions[i][1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && nStep < steps[nx][ny]) {
                    steps[nx][ny] = nStep;
                    queue.offer(new int[] { nx, ny });
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
    private int res2065;
    private int maxTime2065;
    private int[] values2065;
    private Map<Integer, List<int[]>> graph2065;

    public int maximalPathQuality(int[] values, int[][] edges, int maxTime) {
        Map<Integer, List<int[]>> graph = new HashMap<>();
        for (int[] edge : edges) {
            graph.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(new int[] { edge[1], edge[2] });
            graph.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(new int[] { edge[0], edge[2] });
        }
        maxTime2065 = maxTime;
        values2065 = values;
        graph2065 = graph;
        int n = values.length;
        boolean[] visited = new boolean[n];
        visited[0] = true;
        dfs2065(0, values[0], 0, visited);
        return res2065;
    }

    private void dfs2065(int node, int curVal, int curTime, boolean[] visited) {
        if (node == 0) {
            res2065 = Math.max(res2065, curVal);
        }
        for (int[] neighbor : graph2065.getOrDefault(node, new ArrayList<>())) {
            int nNode = neighbor[0];
            int time = neighbor[1];
            if (curTime + time > maxTime2065) {
                continue;
            }
            if (!visited[nNode]) {
                visited[nNode] = true;
                dfs2065(nNode, curVal + values2065[nNode], curTime + time, visited);
                visited[nNode] = false;
            } else {
                dfs2065(nNode, curVal, curTime + time, visited);
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
    private Map<Integer, List<Integer>> graph2246;
    private char[] char2246;
    private int res2246;

    public int longestPath(int[] parent, String s) {
        graph2246 = new HashMap<>();
        for (int i = 0; i < parent.length; ++i) {
            graph2246.computeIfAbsent(i, k -> new ArrayList<>()).add(parent[i]);
            graph2246.computeIfAbsent(parent[i], k -> new ArrayList<>()).add(i);
        }
        res2246 = 1;
        char2246 = s.toCharArray();
        dfs2246(0, -1);
        return res2246;
    }

    private int dfs2246(int x, int fa) {
        if (graph2246.getOrDefault(x, new ArrayList<>()).size() == 1) {
            return 1;
        }
        int max1 = 0;
        int max2 = 0;
        for (int y : graph2246.getOrDefault(x, new ArrayList<>())) {
            if (y != fa) {
                int cnt = dfs2246(y, x);
                if (char2246[y] != char2246[x]) {
                    if (cnt >= max1) {
                        max2 = max1;
                        max1 = cnt;
                    } else if (cnt >= max2) {
                        max2 = cnt;
                    }
                }

            }
        }
        res2246 = Math.max(res2246, 1 + max1 + max2);
        return max1 + 1;
    }

    // 2250. 统计包含每个点的矩形数目 (Count Number of Rectangles Containing Each Point)
    // public int[] countRectangles(int[][] rectangles, int[][] points) {

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

    // 2466. 统计构造好字符串的方案数 (Count Ways To Build Good Strings)
    // public int countGoodStrings(int low, int high, int zero, int one) {

    // }

}
