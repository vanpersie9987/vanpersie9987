import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

public class Leetcode_3 {
    public static void main(String[] args) {

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
        public int val;
        public TreeNode left;
        public TreeNode right;

        public TreeNode(int val) {
            this.val = val;
        }

        public TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }

    }

    // 111. 二叉树的最小深度 (Minimum Depth of Binary Tree) --dfs
    public int minDepth2(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return 1;
        }
        int depth = Integer.MAX_VALUE;
        if (root.left != null) {
            depth = Math.min(depth, minDepth2(root.left));
        }
        if (root.right != null) {
            depth = Math.min(depth, minDepth2(root.right));
        }
        return depth + 1;

    }

    // 111. 二叉树的最小深度 (Minimum Depth of Binary Tree) --bfs
    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int res = 1;
        while (root != null) {
            int size = queue.size();
            while (size > 0) {
                TreeNode node = queue.poll();
                if (node.left == null && node.right == null) {
                    return res;
                }
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
                --size;
            }
            ++res;
        }
        return res;

    }

    // 104. 二叉树的最大深度 (Maximum Depth of Binary Tree) --bfs
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int res = 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            while (size > 0) {
                TreeNode node = queue.poll();
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
                --size;
            }
            ++res;
        }
        return res;

    }

    // 104. 二叉树的最大深度 (Maximum Depth of Binary Tree) --dfs
    public int maxDepth2(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int leftDepth = maxDepth2(root.left);
        int rightDepth = maxDepth2(root.right);
        return Math.max(leftDepth, rightDepth) + 1;
    }

    // 1380. 矩阵中的幸运数 (Lucky Numbers in a Matrix)
    public List<Integer> luckyNumbers(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        int[] minRow = new int[m];
        Arrays.fill(minRow, Integer.MAX_VALUE);
        int[] maxCol = new int[n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                minRow[i] = Math.min(minRow[i], matrix[i][j]);
                maxCol[j] = Math.max(maxCol[j], matrix[i][j]);
            }
        }
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (minRow[i] == maxCol[j]) {
                    res.add(minRow[i]);
                }
            }
        }
        return res;
    }

    // 103. 二叉树的锯齿形层序遍历 (Binary Tree Zigzag Level Order Traversal)
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new LinkedList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        boolean reverse = false;
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> list = new LinkedList<>();
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                if (reverse) {
                    list.add(0, node.val);
                } else {
                    list.add(node.val);
                }

                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            reverse = !reverse;
            res.add(list);
        }
        return res;

    }

    // 147. 对链表进行插入排序 (Insertion Sort List)
    public ListNode insertionSortList(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode dummy = new ListNode(0, head);
        ListNode lastSorted = head;
        ListNode cur = head.next;
        while (cur != null) {
            if (lastSorted.val <= cur.val) {
                lastSorted = lastSorted.next;
            } else {
                ListNode pre = dummy;
                while (pre.next.val <= cur.val) {
                    pre = pre.next;
                }
                lastSorted.next = cur.next;
                cur.next = pre.next;
                pre.next = cur;
            }
            cur = lastSorted.next;
        }
        return dummy.next;
    }

    // 1367. 二叉树中的列表 (Linked List in Binary Tree)
    public boolean isSubPath(ListNode head, TreeNode root) {
        if (root == null) {
            return false;
        }
        return judgeSubPath(head, root) || isSubPath(head, root.left) || isSubPath(head, root.right);
    }

    private boolean judgeSubPath(ListNode head, TreeNode root) {
        if (head == null) {
            return true;
        }
        if (root == null || root.val != head.val) {
            return false;
        }
        return judgeSubPath(head.next, root.left) || judgeSubPath(head.next, root.right);
    }

    // 23. 合并K个升序链表 (Merge k Sorted Lists) --暴力
    // 剑指 Offer II 078. 合并排序链表
    public ListNode mergeKLists(ListNode[] lists) {
        ListNode res = null;
        for (ListNode list : lists) {
            res = merge23(res, list);
        }
        return res;

    }

    private ListNode merge23(ListNode head1, ListNode head2) {
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        while (head1 != null && head2 != null) {
            if (head1.val < head2.val) {
                cur.next = head1;
                head1 = head1.next;
            } else {
                cur.next = head2;
                head2 = head2.next;
            }
            cur = cur.next;
        }
        cur.next = head1 != null ? head1 : head2;
        return dummy.next;
    }

    // 23. 合并K个升序链表 (Merge k Sorted Lists) --分治
    // 剑指 Offer II 078. 合并排序链表
    public ListNode mergeKLists2(ListNode[] lists) {
        return getLists(lists, 0, lists.length - 1);
    }

    private ListNode getLists(ListNode[] lists, int left, int right) {
        if (left > right) {
            return null;
        }
        if (left == right) {
            return lists[left];
        }
        int mid = left + ((right - left) >>> 1);
        return merge(getLists(lists, left, mid), getLists(lists, mid + 1, right));
    }

    private ListNode merge(ListNode head1, ListNode head2) {
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        while (head1 != null && head2 != null) {
            if (head1.val < head2.val) {
                cur.next = head1;
                head1 = head1.next;
            } else {
                cur.next = head2;
                head2 = head2.next;
            }
            cur = cur.next;
        }
        cur.next = head1 != null ? head1 : head2;
        return dummy.next;
    }

    public class Status implements Comparable<Status> {
        public int val;
        public ListNode node;

        public Status(int val, ListNode node) {
            this.val = val;
            this.node = node;
        }

        @Override
        public int compareTo(Status s) {
            return this.val - s.val;
        }
    }

    // 23. 合并K个升序链表 (Merge k Sorted Lists) --优先队列
    // 剑指 Offer II 078. 合并排序链表
    public ListNode mergeKLists3(ListNode[] lists) {
        PriorityQueue<Status> priorityQueue = new PriorityQueue<>();
        for (ListNode list : lists) {
            if (list != null) {
                priorityQueue.offer(new Status(list.val, list));
            }
        }
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        while (!priorityQueue.isEmpty()) {
            Status pop = priorityQueue.poll();
            cur.next = pop.node;
            cur = cur.next;
            if (pop.node.next != null) {
                priorityQueue.offer(new Status(pop.node.next.val, pop.node.next));
            }
        }
        return dummy.next;
    }

    // 25. K 个一组翻转链表 (Reverse Nodes in k-Group)
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode cur = head;
        int count = 0;
        while (cur != null) {
            cur = cur.next;
            ++count;
        }
        int parts = count / k;
        if (parts == 0) {
            return head;
        }
        ListNode dummy = new ListNode(0, head);
        int index = 0;
        ListNode pre = head;
        while (index < parts) {
            ListNode[] nexts = reverseLinkedList25(head, k);
            if (index == 0) {
                dummy.next = nexts[0];
            } else {
                pre.next = nexts[0];
            }
            pre = head;
            head.next = nexts[1];
            head = nexts[1];
            ++index;
        }
        return dummy.next;

    }

    private ListNode[] reverseLinkedList25(ListNode head, int k) {
        ListNode[] res = new ListNode[2];
        ListNode cur = head;
        ListNode pre = null;
        while (k-- > 0) {
            ListNode temp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = temp;
        }
        res[0] = pre;
        res[1] = cur;
        return res;
    }

    // 2099. 找到和最大的长度为 K 的子序列 (Find Subsequence of Length K With the Largest Sum)
    public int[] maxSubsequence(int[] nums, int k) {
        Element[] elements = new Element[nums.length];
        for (int i = 0; i < elements.length; ++i) {
            elements[i] = new Element(nums[i], i);
        }
        Arrays.sort(elements, (o1, o2) -> o2.val - o1.val);

        Arrays.sort(elements, 0, k, (o1, o2) -> o1.index - o2.index);
        int[] res = new int[k];
        for (int i = 0; i < k; ++i) {
            res[i] = elements[i].val;
        }
        return res;

    }

    public class Element {
        public int val;
        public int index;

        public Element(int val, int index) {
            this.val = val;
            this.index = index;
        }

    }

    // 2100. 适合打劫银行的日子 (Find Good Days to Rob the Bank)
    public List<Integer> goodDaysToRobBank(int[] security, int time) {
        int n = security.length;
        int[] left = new int[n];
        int[] right = new int[n];
        for (int i = 1; i < n; ++i) {
            if (security[i - 1] >= security[i]) {
                left[i] = left[i - 1] + 1;
            }
        }
        for (int i = n - 2; i >= 0; --i) {
            if (security[i] <= security[i + 1]) {
                right[i] = right[i + 1] + 1;
            }
        }
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            if (left[i] >= time && right[i] >= time) {
                res.add(i);
            }
        }
        return res;

    }

    // 1219. 黄金矿工 (Path with Maximum Gold)
    private int res1219;

    public int getMaximumGold(int[][] grid) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int m = grid.length;
        int n = grid[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] != 0) {
                    getMaxGold(grid, i, j, directions, 0);
                }
            }
        }
        return res1219;
    }

    private void getMaxGold(int[][] grid, int i, int j, int[][] directions, int gold) {
        gold += grid[i][j];
        res1219 = Math.max(res1219, gold);
        int temp = grid[i][j];
        grid[i][j] = 0;
        for (int[] direction : directions) {
            int newI = i + direction[0];
            int newJ = j + direction[1];
            if (newI >= 0 && newI < grid.length && newJ >= 0 && newJ < grid[0].length && grid[newI][newJ] != 0) {
                getMaxGold(grid, i + direction[0], j + direction[1], directions, gold);
            }
        }
        grid[i][j] = temp;
    }

    // 2176. 统计数组中相等且可以被整除的数对 (Count Equal and Divisible Pairs in an Array)
    public int countPairs(int[] nums, int k) {
        int res = 0;
        for (int i = 0; i < nums.length; ++i) {
            for (int j = i + 1; j < nums.length; ++j) {
                if (nums[i] == nums[j] && ((i * j % k) == 0)) {
                    ++res;
                }
            }
        }
        return res;

    }

    // 2177. 找到和为给定整数的三个连续整数 (Find Three Consecutive Integers That Sum to a Given
    // Number)
    public long[] sumOfThree(long num) {
        long mid = num / 3;
        if (mid * 3 == num) {
            return new long[] { mid - 1, mid, mid + 1 };
        }
        return new long[] {};

    }

    // 2178. 拆分成最多数目的偶整数之和 (Maximum Split of Positive Even Integers)
    public List<Long> maximumEvenSplit(long finalSum) {
        List<Long> res = new ArrayList<>();
        if (finalSum % 2 == 1) {
            return res;
        }
        int cur = 2;
        while (finalSum > 0) {
            if (!res.isEmpty() && finalSum <= res.get(res.size() - 1)) {
                break;
            }
            res.add((long) cur);
            finalSum -= cur;
            cur += 2;
        }
        res.set(res.size() - 1, res.get(res.size() - 1) + finalSum);
        return res;

    }

    // 2180. 统计各位数字之和为偶数的整数个数 (Count Integers With Even Digit Sum)
    public int countEven(int num) {
        int res = 0;
        for (int i = 1; i <= num; ++i) {
            if (isEven(i)) {
                ++res;
            }
        }
        return res;
    }

    private boolean isEven(int num) {
        int sum = 0;
        while (num != 0) {
            sum += num % 10;
            num /= 10;
        }
        return sum % 2 == 0;
    }

    // 2181. 合并零之间的节点 (Merge Nodes in Between Zeros)
    public ListNode mergeNodes(ListNode head) {
        ListNode res = head;
        ListNode cur = head.next;
        int sum = 0;
        while (cur != null) {
            if (cur.val != 0) {
                sum += cur.val;
            } else {
                cur.val = sum;
                sum = 0;
                head.next = cur;
                head = head.next;
            }
            cur = cur.next;
        }
        return res.next;

    }

    // 382. 链表随机节点 (Linked List Random Node) --水塘抽样
    class Solution {
        private Random random;
        private ListNode head;

        public Solution(ListNode head) {
            this.head = head;
            random = new Random();

        }

        public int getRandom() {
            int i = 1;
            int res = 0;
            ListNode cur = head;
            while (cur != null) {
                if (random.nextInt(i) == 0) {
                    res = cur.val;
                }
                cur = cur.next;
                ++i;
            }
            return res;
        }
    }

    // 398. 随机数索引 (Random Pick Index) --水塘抽样
    class Solution398 {
        private Random random;
        private int[] nums;

        public Solution398(int[] nums) {
            this.nums = nums;
            this.random = new Random();
        }

        public int pick(int target) {
            int res = 0;
            int count = 0;
            for (int i = 0; i < nums.length; ++i) {
                if (nums[i] == target) {
                    ++count;
                    if (random.nextInt(count) == 0) {
                        res = i;
                    }
                }
            }
            return res;
        }
    }

    // 2182. 构造限制重复的字符串 (Construct String With Repeat Limit) --贪心
    public String repeatLimitedString(String s, int repeatLimit) {
        int[] counts = new int[26];
        for (char c : s.toCharArray()) {
            ++counts[c - 'a'];
        }
        StringBuilder res = new StringBuilder();
        for (int i = counts.length - 1; i >= 0; --i) {
            while (counts[i] > 0) {
                if (counts[i] <= repeatLimit) {
                    while (counts[i]-- > 0) {
                        res.append((char) (i + 'a'));
                    }
                    break;
                } else {
                    for (int j = 0; j < repeatLimit; ++j) {
                        res.append((char) (i + 'a'));
                    }
                    counts[i] -= repeatLimit;
                    char c = find2182(counts, i - 1);
                    if (c == 'X') {
                        return res.toString();
                    }
                    res.append(c);
                }
            }
        }
        return res.toString();

    }

    private char find2182(int[] counts, int i) {
        while (i >= 0) {
            if (counts[i] > 0) {
                --counts[i];
                return (char) (i + 'a');
            }
            --i;
        }
        return 'X';
    }

    // 199. 二叉树的右视图 (Binary Tree Right Side View) --广度优先 bfs
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                TreeNode cur = queue.poll();
                if (i == size - 1) {
                    res.add(cur.val);
                }
                if (cur.left != null) {
                    queue.offer(cur.left);
                }
                if (cur.right != null) {
                    queue.offer(cur.right);
                }
            }
        }
        return res;

    }

    // 199. 二叉树的右视图 (Binary Tree Right Side View) --深度优先 dfs
    private List<Integer> res199;

    public List<Integer> rightSideView2(TreeNode root) {
        res199 = new ArrayList<>();
        dfs199(root, 0);
        return res199;
    }

    private void dfs199(TreeNode root, int depth) {
        if (root == null) {
            return;
        }
        if (res199.size() == depth) {
            res199.add(root.val);
        }
        ++depth;
        dfs199(root.right, depth);
        dfs199(root.left, depth);
    }

    // 965. 单值二叉树 (Univalued Binary Tree) --莫里斯+中序遍历
    public boolean isUnivalTree(TreeNode root) {
        TreeNode pre = null;
        int uniVal = root.val;
        while (root != null) {
            if (root.left != null) {
                pre = root.left;
                while (pre.right != null && pre.right != root) {
                    pre = pre.right;
                }
                if (pre.right == null) {
                    pre.right = root;
                    root = root.left;
                } else {
                    pre.right = null;
                    if (root.val != uniVal) {
                        return false;
                    }
                    root = root.right;
                }
            } else {
                if (root.val != uniVal) {
                    return false;
                }
                root = root.right;
            }
        }
        return true;

    }

    // 965. 单值二叉树 (Univalued Binary Tree) --递归
    public boolean isUnivalTree2(TreeNode root) {
        if (root == null) {
            return true;
        }
        if (root.left != null && root.left.val != root.val) {
            return false;
        }
        if (root.right != null && root.right.val != root.val) {
            return false;
        }
        return isUnivalTree2(root.left) && isUnivalTree2(root.right);
    }

    // 面试题 08.06. 汉诺塔问题 (Hanota LCCI)
    public void hanota(List<Integer> A, List<Integer> B, List<Integer> C) {
        getHanota(A, B, C, A.size());
    }

    private void getHanota(List<Integer> a, List<Integer> b, List<Integer> c, int n) {
        if (n == 0) {
            return;
        }
        getHanota(a, c, b, n - 1);
        c.add(a.remove(a.size() - 1));
        getHanota(b, a, c, n - 1);
    }

    // 1162. 地图分析 (As Far from Land as Possible) --bfs 超时
    public int maxDistance(int[][] grid) {
        int res = -1;
        int[][] directions = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
        int n = grid.length;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 0) {
                    res = Math.max(res, getMax1162(grid, directions, i, j));
                }
            }
        }
        return res;

    }

    private int getMax1162(int[][] grid, int[][] directions, int i, int j) {
        int n = grid.length;
        boolean[][] visited = new boolean[n][n];
        visited[i][j] = true;
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[] { i, j, 0 });
        while (!queue.isEmpty()) {
            int[] f = queue.poll();
            for (int[] direction : directions) {
                int nx = direction[0] + f[0];
                int ny = direction[1] + f[1];
                if (nx < 0 || nx >= n || ny < 0 || ny >= n) {
                    continue;
                }
                if (!visited[nx][ny]) {
                    visited[nx][ny] = true;
                    queue.offer(new int[] { nx, ny, f[2] + 1 });
                    if (grid[nx][ny] == 1) {
                        return f[2] + 1;
                    }
                }
            }
        }
        return -1;
    }

    // 1162. 地图分析 (As Far from Land as Possible) --图的bfs
    public int maxDistance2(int[][] grid) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int n = grid.length;
        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    queue.offer(new int[] { i, j });
                }
            }
        }
        boolean hasOcean = false;
        int[] f = null;
        while (!queue.isEmpty()) {
            f = queue.poll();
            for (int[] direction : directions) {
                int nx = f[0] + direction[0];
                int ny = f[1] + direction[1];
                if (nx < 0 || nx >= n || ny < 0 || ny >= n || grid[nx][ny] != 0) {
                    continue;
                }
                hasOcean = true;
                grid[nx][ny] = grid[f[0]][f[1]] + 1;
                queue.offer(new int[] { nx, ny });
            }
        }
        if (!hasOcean || f == null) {
            return -1;
        }
        return grid[f[0]][f[1]] - 1;

    }

    // 1302. 层数最深叶子节点的和 (Deepest Leaves Sum) --bfs
    public int deepestLeavesSum(TreeNode root) {
        int res = 0;
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            res = 0;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
                res += node.val;
            }
        }
        return res;

    }

    // 1302. 层数最深叶子节点的和 (Deepest Leaves Sum) --dfs
    private int maxDepth1302 = -1;
    private int res1302;

    public int deepestLeavesSum2(TreeNode root) {
        dfs1302(root, 0);
        return res1302;
    }

    private void dfs1302(TreeNode root, int depth) {
        if (root == null) {
            return;
        }
        if (depth > maxDepth1302) {
            maxDepth1302 = depth;
            res1302 = root.val;
        } else if (depth == maxDepth1302) {
            res1302 += root.val;
        }
        dfs1302(root.left, depth + 1);
        dfs1302(root.right, depth + 1);
    }

    // 112. 路径总和 (Path Sum) --bfs
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return false;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        Queue<Integer> queueVal = new LinkedList<>();
        queue.offer(root);
        queueVal.offer(root.val);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            int val = queueVal.poll();
            if (node.left == null && node.right == null) {
                if (targetSum == val) {
                    return true;
                }
                continue;
            }
            if (node.left != null) {
                queue.offer(node.left);
                queueVal.offer(val + node.left.val);
            }
            if (node.right != null) {
                queue.offer(node.right);
                queueVal.offer(val + node.right.val);
            }
        }
        return false;

    }

    // 112. 路径总和 (Path Sum) --dfs
    public boolean hasPathSum2(TreeNode root, int targetSum) {
        if (root == null) {
            return false;
        }
        if (root.left == null && root.right == null) {
            return targetSum == root.val;
        }
        return hasPathSum2(root.left, targetSum - root.val) || hasPathSum2(root.right, targetSum - root.val);

    }

    // 257. 二叉树的所有路径 (Binary Tree Paths) --深度优先dfs
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> res = new ArrayList<>();
        dfs257(root, "", res);
        return res;

    }

    private void dfs257(TreeNode root, String path, List<String> paths) {
        if (root == null) {
            return;
        }
        StringBuilder sb = new StringBuilder(path);
        sb.append(root.val);
        if (root.left == null && root.right == null) {
            paths.add(sb.toString());
            return;
        }
        sb.append("->");
        dfs257(root.left, sb.toString(), paths);
        dfs257(root.right, sb.toString(), paths);
    }

    // 257. 二叉树的所有路径 (Binary Tree Paths) --广度优先bfs
    public List<String> binaryTreePaths2(TreeNode root) {
        List<String> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        Queue<String> path = new LinkedList<>();
        queue.offer(root);
        path.offer(String.valueOf(root.val));
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            String string = path.poll();
            if (node.left == null && node.right == null) {
                res.add(string);
            }
            if (node.left != null) {
                queue.offer(node.left);
                path.offer(string + "->" + node.left.val);
            }
            if (node.right != null) {
                queue.offer(node.right);
                path.offer(string + "->" + node.right.val);
            }
        }
        return res;

    }

    // 314. 二叉树的垂直遍历 (Binary Tree Vertical Order Traversal) --plus
    public List<List<Integer>> verticalOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        int minPos = Integer.MAX_VALUE;
        Queue<TreeNode> queue = new LinkedList<>();
        Queue<Integer> queuePos = new LinkedList<>();
        Map<Integer, List<Integer>> map = new HashMap<>();
        queue.offer(root);
        queuePos.offer(0);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            int pos = queuePos.poll();
            map.computeIfAbsent(pos, k -> new LinkedList<>()).add(node.val);
            minPos = Math.min(minPos, pos);
            if (node.left != null) {
                queue.offer(node.left);
                queuePos.offer(pos - 1);
            }
            if (node.right != null) {
                queue.offer(node.right);
                queuePos.offer(pos + 1);
            }
        }
        for (int i = minPos; i < minPos + map.size(); ++i) {
            res.add(map.get(i));
        }
        return res;

    }

    // 298. 二叉树最长连续序列 (Binary Tree Longest Consecutive Sequence) --dfs
    private int res298;

    public int longestConsecutive(TreeNode root) {
        dfs298(root, root.val, 0);
        return res298;

    }

    private void dfs298(TreeNode root, int longestConsecutive, int cur) {
        if (root == null) {
            return;
        }
        if (root.val == longestConsecutive) {
            res298 = Math.max(res298, ++cur);
        } else {
            cur = 0;
            longestConsecutive = root.val;
        }
        dfs298(root.left, longestConsecutive + 1, cur);
        dfs298(root.right, longestConsecutive + 1, cur);

    }

    // 298. 二叉树最长连续序列 (Binary Tree Longest Consecutive Sequence) --bfs
    public int longestConsecutive2(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int res = 1;
        Queue<TreeNode> queue = new LinkedList<>();
        Queue<Integer> queueConsective = new LinkedList<>();
        queue.offer(root);
        queueConsective.offer(1);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            int val = queueConsective.poll();
            int temp = val;
            if (node.left != null) {
                if (node.left.val == node.val + 1) {
                    ++temp;
                } else {
                    temp = 1;
                }
                res = Math.max(res, temp);
                queueConsective.offer(temp);
                queue.offer(node.left);
            }
            temp = val;
            if (node.right != null) {
                if (node.right.val == node.val + 1) {
                    ++temp;
                } else {
                    temp = 1;
                }
                res = Math.max(res, temp);
                queueConsective.offer(temp);
                queue.offer(node.right);
            }
        }
        return res;
    }

    // 250. 统计同值子树 (Count Univalue Subtrees)
    private int res250;

    public int countUnivalSubtrees(TreeNode root) {
        dfs250(root, 0);
        return res250;
    }

    private boolean dfs250(TreeNode root, int val) {
        if (root == null) {
            return true;
        }
        if (!dfs250(root.left, root.val) | !dfs250(root.right, root.val)) {
            return false;
        }
        ++res250;
        return root.val == val;
    }

    // 1119. 删去字符串中的元音 (Remove Vowels from a String) -- plus
    public String removeVowels(String s) {
        StringBuilder res = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (!isVowel(c)) {
                res.append(c);
            }
        }
        return res.toString();

    }

    private boolean isVowel(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }

    // 1874. 两个数组的最小乘积和 (Minimize Product Sum of Two Arrays) --plus
    public int minProductSum(int[] nums1, int[] nums2) {
        int res = 0;
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int i = 0;
        while (i < nums1.length) {
            res += nums1[i] * nums2[nums2.length - i - 1];
            ++i;
        }
        return res;

    }

    // 1165. 单行键盘 (Single-Row Keyboard) --哈希表 plus
    public int calculateTime(String keyboard, String word) {
        int res = 0;
        Map<Character, Integer> map = new HashMap<>();
        char[] keyboardChars = keyboard.toCharArray();
        for (int i = 0; i < keyboardChars.length; ++i) {
            map.put(keyboardChars[i], i);
        }
        char pre = 'X';
        for (char c : word.toCharArray()) {
            if (pre == 'X') {
                res += map.get(c);
            } else {
                res += Math.abs(map.get(c) - map.get(pre));
            }
            pre = c;
        }
        return res;

    }

    // 760. 找出变位映射 (Find Anagram Mappings) --plus
    public int[] anagramMappings(int[] nums1, int[] nums2) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < nums2.length; ++i) {
            map.computeIfAbsent(nums2[i], k -> new LinkedList<>()).add(i);
        }
        int[] res = new int[nums1.length];
        for (int i = 0; i < nums1.length; ++i) {
            List<Integer> list = map.get(nums1[i]);
            res[i] = list.remove(0);
        }
        return res;

    }

    // 1469. 寻找所有的独生节点 (Find All The Lonely Nodes) --plus bfs
    public List<Integer> getLonelyNodes(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node.left != null && node.right != null) {
                queue.offer(node.left);
                queue.offer(node.right);
            } else if (node.left != null) {
                res.add(node.left.val);
                queue.offer(node.left);
            } else if (node.right != null) {
                res.add(node.right.val);
                queue.offer(node.right);
            }
        }
        return res;
    }

    // 1469. 寻找所有的独生节点 (Find All The Lonely Nodes) --plus dfs
    public List<Integer> getLonelyNodes2(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        dfs1469(root, res);
        return res;
    }

    private void dfs1469(TreeNode root, List<Integer> res) {
        if (root == null) {
            return;
        }
        if (root.left != null && root.right != null) {
            dfs1469(root.left, res);
            dfs1469(root.right, res);
        } else if (root.left != null) {
            res.add(root.left.val);
            dfs1469(root.left, res);
        } else if (root.right != null) {
            res.add(root.right.val);
            dfs1469(root.right, res);
        }
    }

    // 513. 找树左下角的值 (Find Bottom Left Tree Value)
    // 剑指 Offer II 045. 二叉树最底层最左边的值 --dfs plus
    private int res513;
    private int maxDepth = -1;

    public int findBottomLeftValue(TreeNode root) {
        dfs513(root, 0);
        return res513;
    }

    private void dfs513(TreeNode root, int depth) {
        if (root == null) {
            return;
        }
        if (depth > maxDepth) {
            maxDepth = depth;
            res513 = root.val;
        }
        dfs513(root.left, depth + 1);
        dfs513(root.right, depth + 1);
    }

    // 513. 找树左下角的值 (Find Bottom Left Tree Value)
    // 剑指 Offer II 045. 二叉树最底层最左边的值 --bfs plus
    public int findBottomLeftValue2(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        int res = 0;
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                if (i == 0) {
                    res = node.val;
                }
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
        }
        return res;

    }

    class Node {
        public int val;
        public Node left;
        public Node right;
        public Node parent;
    };

    // 1650. 二叉树的最近公共祖先 III (Lowest Common Ancestor of a Binary Tree III) --plus
    // 类似于找链表公共节点
    public Node lowestCommonAncestor(Node p, Node q) {
        Node curP = p;
        Node curQ = q;
        while (curP != curQ) {
            curP = curP == null ? q : curP.parent;
            curQ = curQ == null ? p : curQ.parent;
        }
        return curP;

    }

    // 1213. 三个有序数组的交集 (Intersection of Three Sorted Arrays) --plus 三指针+贪心
    public List<Integer> arraysIntersection(int[] arr1, int[] arr2, int[] arr3) {
        List<Integer> res = new ArrayList<>();
        int n = Math.min(Math.min(arr1.length, arr2.length), arr3.length);
        int i = 0;
        int j = 0;
        int k = 0;
        while (i < n && j < n && k < n) {
            if (arr1[i] == arr2[j] && arr2[j] == arr3[k]) {
                res.add(arr1[i]);
                ++i;
                ++j;
                ++k;
            } else {
                int max = Math.max(Math.max(arr1[i], arr2[j]), arr3[k]);
                if (arr1[i] < max) {
                    ++i;
                }
                if (arr2[j] < max) {
                    ++j;
                }
                if (arr3[k] < max) {
                    ++k;
                }
            }
        }
        return res;

    }

    // 366. Find Leaves of Binary Tree --plus dfs后序遍历
    private Map<Integer, List<Integer>> res366;

    public List<List<Integer>> findLeaves(TreeNode root) {
        res366 = new TreeMap<>();
        dfs366(root);
        return new ArrayList<>(res366.values());
    }

    private int dfs366(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = dfs366(root.left);
        int right = dfs366(root.right);
        int max = Math.max(left, right) + 1;
        res366.computeIfAbsent(max, k -> new LinkedList<>()).add(root.val);
        return max;
    }

    // 1180. 统计只含单一字母的子串 (Count Substrings with Only One Distinct Letter) --plus
    public int countLetters(String s) {
        int res = 1;
        int count = 1;
        char[] chars = s.toCharArray();
        for (int i = 1; i < chars.length; ++i) {
            if (chars[i] == chars[i - 1]) {
                ++count;
            } else {
                count = 1;
            }
            res += count;
        }
        return res;

    }

    // 1134. 阿姆斯特朗数 (Armstrong Number) --plus 模拟
    public boolean isArmstrong(int n) {
        int bit = (int) (Math.log(n) / Math.log(10)) + 1;
        int num = n;
        int cur = 0;
        while (num != 0) {
            int val = num % 10;
            cur += Math.pow(val, bit);
            num /= 10;
        }
        return cur == n;

    }

    // 1085. 最小元素各数位之和 (Sum of Digits in the Minimum Number) --plus 模拟
    public int sumOfDigits(int[] nums) {
        int min = Arrays.stream(nums).min().getAsInt();
        int sum = 0;
        while (min != 0) {
            sum += min % 10;
            min /= 10;
        }
        return 1 - sum % 2;

    }

    // 186. 翻转字符串里的单词 II (Reverse Words in a String II) --plus 双指针
    public void reverseWords(char[] s) {
        int left = 0;
        int right = s.length - 1;
        while (left < right) {
            char c = s[left];
            s[left] = s[right];
            s[right] = c;
            ++left;
            --right;
        }
        left = 0;
        right = 0;
        while (right < s.length) {
            while (right < s.length && s[right] != ' ') {
                ++right;
            }
            int curRight = right - 1;
            while (left < curRight) {
                char c = s[left];
                s[left] = s[curRight];
                s[curRight] = c;
                ++left;
                --curRight;
            }
            left = right + 1;
            right = right + 1;
        }

    }

    // 281. 锯齿迭代器 (Zigzag Iterator) --模拟
    // 扩展：K路输入？
    public class ZigzagIterator {
        private List<Integer> v1;
        private List<Integer> v2;
        private int index1;
        private int index2;
        private boolean takeFromV1;

        public ZigzagIterator(List<Integer> v1, List<Integer> v2) {
            this.v1 = v1;
            this.v2 = v2;
            this.takeFromV1 = true;
        }

        public int next() {
            if (hasNext()) {
                if (index1 < v1.size() && index2 < v2.size()) {
                    if (takeFromV1) {
                        takeFromV1 = false;
                        return v1.get(index1++);
                    } else {
                        takeFromV1 = true;
                        return v2.get(index2++);
                    }
                } else if (index1 < v1.size()) {
                    return v1.get(index1++);
                } else {
                    return v2.get(index2++);
                }
            }
            return -1;
        }

        public boolean hasNext() {
            return index1 < v1.size() || index2 < v2.size();
        }
    }

}
