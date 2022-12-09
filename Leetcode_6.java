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

    // 2467. 树上最大得分和路径 (Most Profitable Path in a Tree)
    // public int mostProfitablePath(int[][] edges, int bob, int[] amount) {

    // }

    // 2466. 统计构造好字符串的方案数 (Count Ways To Build Good Strings)
    // public int countGoodStrings(int low, int high, int zero, int one) {

    // }

}
