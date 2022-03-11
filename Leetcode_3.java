import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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

    // 542. 01 矩阵 (01 Matrix) --bfs
    public int[][] updateMatrix(int[][] mat) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int m = mat.length;
        int n = mat[0].length;
        boolean[][] seen = new boolean[m][n];
        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (mat[i][j] == 0) {
                    queue.offer(new int[] { i, j, 0 });
                    seen[i][j] = true;
                }
            }
        }
        if (queue.size() == m * n) {
            return mat;
        }
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            for (int[] direction : directions) {
                int nx = cur[0] + direction[0];
                int ny = cur[1] + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && !seen[nx][ny]) {
                    seen[nx][ny] = true;
                    mat[nx][ny] = cur[2] + 1;
                    queue.offer(new int[] { nx, ny, mat[nx][ny] });
                }
            }
        }
        return mat;
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

    // 1940. 排序数组之间的最长公共子序列 (Longest Common Subsequence Between Sorted Arrays)
    // --plus 计数
    public List<Integer> longestCommonSubsequence(int[][] arrays) {
        List<Integer> res = new ArrayList<>();
        int n = arrays.length;
        int[] counts = new int[101];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < arrays[i].length; ++j) {
                ++counts[arrays[i][j]];
            }
        }
        for (int i = 1; i < counts.length; ++i) {
            if (counts[i] == n) {
                res.add(i);
            }
        }
        return res;

    }

    // 1602. 找到二叉树中最近的右侧节点 (Find Nearest Right Node in Binary Tree) --plus bfs
    public TreeNode findNearestRightNode(TreeNode root, TreeNode u) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            TreeNode pre = null;
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                if (pre != null && pre == u) {
                    return node;
                }
                pre = node;
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
        }
        return null;

    }

    // 370. 区间加法 (Range Addition) --plus 差分数组
    public int[] getModifiedArray(int length, int[][] updates) {
        int[] diff = new int[length];
        for (int[] update : updates) {
            diff[update[0]] += update[2];
            if (update[1] + 1 < length) {
                diff[update[1] + 1] -= update[2];
            }
        }
        for (int i = 1; i < diff.length; ++i) {
            diff[i] += diff[i - 1];
        }
        return diff;

    }

    // 1198. 找出所有行中最小公共元素 (Find Smallest Common Element in All Rows) --plus
    public int smallestCommonElement(int[][] mat) {
        int[] counts = new int[10001];
        int m = mat.length;
        int n = mat[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                ++counts[mat[i][j]];
            }
        }
        for (int i = 1; i < counts.length; ++i) {
            if (counts[i] == m) {
                return i;
            }
        }
        return -1;
    }

    // 1836. 从未排序的链表中移除重复元素 (Remove Duplicates From an Unsorted Linked List) --plus
    public ListNode deleteDuplicatesUnsorted(ListNode head) {
        ListNode dummy = new ListNode(0, head);
        ListNode cur = head;
        Map<Integer, Integer> map = new HashMap<>();
        while (cur != null) {
            map.put(cur.val, map.getOrDefault(cur.val, 0) + 1);
            cur = cur.next;
        }
        Set<Integer> set = new HashSet<>();
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (entry.getValue() > 1) {
                set.add(entry.getKey());
            }
        }
        cur = dummy;
        while (cur != null) {
            while (cur.next != null && set.contains(cur.next.val)) {
                cur.next = cur.next.next;
            }
            cur = cur.next;
        }
        return dummy.next;
    }

    // 2046. 给按照绝对值排序的链表排序 (Sort Linked List Already Sorted Using Absolute Values)
    // --plus
    public ListNode sortLinkedList(ListNode head) {
        ListNode pre = head;
        ListNode cur = head.next;
        while (cur != null) {
            if (cur.val >= 0) {
                cur = cur.next;
                pre = pre.next;
            } else {
                pre.next = cur.next;
                cur.next = head;
                head = cur;
                cur = pre.next;
            }
        }
        return head;

    }

    // 1762. 能看到海景的建筑物 (Buildings With an Ocean View) --优化
    public int[] findBuildings(int[] heights) {
        Stack<Integer> stack = new Stack<>();
        stack.push(heights.length - 1);
        for (int i = heights.length - 2; i >= 0; --i) {
            while (i >= 0 && heights[i] <= heights[stack.peek()]) {
                --i;
            }
            if (i >= 0) {
                stack.push(i);
            }
        }
        int[] res = new int[stack.size()];
        int index = 0;
        while (!stack.isEmpty()) {
            res[index++] = stack.pop();
        }
        return res;

    }

    class Node2 {
        char val;
        Node2 left;
        Node2 right;

        Node2() {
            this.val = ' ';
        }

        Node2(char val) {
            this.val = val;
        }

        Node2(char val, Node2 left, Node2 right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    // 1612. 检查两棵二叉表达式树是否等价 (Check If Two Expression Trees are Equivalent) --plus
    // dfs 莫里斯中序遍历
    public boolean checkEquivalence(Node2 root1, Node2 root2) {
        int[] counts1 = getResult1612(root1);
        int[] counts2 = getResult1612(root2);
        return Arrays.equals(counts1, counts2);

    }

    private int[] getResult1612(Node2 root) {
        int[] counts = new int[26];
        int sign = 1;
        Node2 pre = null;
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
                    if (root.val == '-') {
                        sign *= -1;
                    } else if (Character.isLetter(root.val)) {
                        counts[root.val - 'a'] += sign;
                    }
                    root = root.right;
                }

            } else {
                if (root.val == '-') {
                    sign *= -1;
                } else if (Character.isLetter(root.val)) {
                    counts[root.val - 'a'] += sign;
                }
                root = root.right;
            }
        }
        return counts;
    }

    // 1708. 长度为 K 的最大子数组 (Largest Subarray Length K) --plus
    // 元素不重复
    public int[] largestSubarray(int[] nums, int k) {
        int maxIndex = 0;
        int maxNum = 0;
        for (int i = 0; i < nums.length - k + 1; ++i) {
            if (nums[i] > maxNum) {
                maxIndex = i;
                maxNum = nums[i];
            }
        }
        return Arrays.copyOfRange(nums, maxIndex, maxIndex + k);

    }

    // 348. 设计井字棋 (Design Tic-Tac-Toe) --plus
    class TicTacToe {
        private int[][] counts;
        private int n;

        public TicTacToe(int n) {
            this.counts = new int[2][2 * n + 2];
            this.n = n;
        }

        public int move(int row, int col, int player) {
            ++counts[player - 1][row];
            ++counts[player - 1][n + col];
            if (row == col) {
                ++counts[player - 1][2 * n];
            }
            if (row + col == n - 1) {
                ++counts[player - 1][2 * n + 1];
            }
            if (counts[player - 1][row] == n
                    || counts[player - 1][n + col] == n
                    || counts[player - 1][2 * n] == n
                    || counts[player - 1][2 * n + 1] == n) {
                return player;
            }
            return 0;
        }
    }

    // 66. 加一 (Plus One)
    public int[] plusOne(final int[] digits) {
        for (int i = digits.length - 1; i >= 0; --i) {
            ++digits[i];
            if (digits[i] / 10 == 0) {
                return digits;
            }
            digits[i] %= 10;
        }
        int[] res = new int[digits.length + 1];
        res[0] = 1;
        return res;

    }

    // 66. 加一 (Plus One) --从左往右查找第一个不是9的位置，将其位加1、并将其后位置0
    public int[] plusOne2(int[] digits) {
        int firstNoNineIndex = -1;
        int index = 0;
        while (index < digits.length) {
            if (digits[index] != 9) {
                firstNoNineIndex = index;
            }
            ++index;
        }
        if (firstNoNineIndex == -1) {
            int[] res = new int[digits.length + 1];
            res[0] = 1;
            return res;
        }
        ++digits[firstNoNineIndex];
        ++firstNoNineIndex;
        while (firstNoNineIndex < digits.length) {
            digits[firstNoNineIndex++] = 0;
        }
        return digits;

    }

    // 369. 给单链表加一 (Plus One Linked List) --plus
    public ListNode plusOne(ListNode head) {
        ListNode dummy = new ListNode(0, head);
        ListNode noNine = dummy;
        while (head != null) {
            if (head.val != 9) {
                noNine = head;
            }
            head = head.next;
        }
        ++noNine.val;
        noNine = noNine.next;
        while (noNine != null) {
            noNine.val = 0;
            noNine = noNine.next;
        }
        return dummy.val != 0 ? dummy : dummy.next;

    }

    // 379. 电话目录管理系统 (Design Phone Directory) --plus
    class PhoneDirectory {
        private int[] numbers;

        public PhoneDirectory(int maxNumbers) {
            numbers = new int[maxNumbers];
            for (int i = 0; i < numbers.length; ++i) {
                numbers[i] = i;
            }
        }

        public int get() {
            for (int i = 0; i < numbers.length; ++i) {
                if (numbers[i] != -1) {
                    numbers[i] = -1;
                    return i;
                }
            }
            return -1;

        }

        public boolean check(int number) {
            return numbers[number] != -1;

        }

        public void release(int number) {
            numbers[number] = number;
        }
    }

    // 1474. 删除链表 M 个节点之后的 N 个节点 (Delete N Nodes After M Nodes of a Linked List)
    // --plus
    public ListNode deleteNodes(ListNode head, int m, int n) {
        ListNode dummy = new ListNode(0, head);
        ListNode cur = dummy;
        boolean stay = true;
        int mCount = 0;
        int nCount = 0;
        while (cur != null) {
            if (stay) {
                stay = false;
                while (cur != null && mCount++ != m) {
                    cur = cur.next;
                }
                mCount = 0;
            } else {
                stay = true;
                ListNode tryIt = cur;
                while (tryIt != null && nCount++ != n) {
                    tryIt = tryIt.next;
                }
                nCount = 0;
                if (tryIt == null) {
                    cur.next = null;
                    return dummy.next;
                } else {
                    cur.next = tryIt.next;
                }
            }
        }
        return dummy.next;

    }

    // 426. 将二叉搜索树转化为排序的双向链表 (Convert Binary Search Tree to Sorted Doubly Linked
    // List) --plus 莫里斯中序遍历
    public Node treeToDoublyList(Node root) {
        Node res = null;
        Node predecessor = null;
        Node pre = null;
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
                    if (predecessor != null) {
                        predecessor.right = root;
                        root.left = predecessor;
                    } else {
                        res = root;
                    }
                    predecessor = root;
                    root = root.right;
                }
            } else {
                if (predecessor != null) {
                    predecessor.right = root;
                    root.left = predecessor;
                } else {
                    res = root;
                }
                predecessor = root;
                root = root.right;
            }
        }
        if (res != null) {
            predecessor.right = res;
            res.left = predecessor;
        }
        return res;

    }

    // 6008. 统计包含给定前缀的字符串
    public int prefixCount(String[] words, String pref) {
        int res = 0;
        for (String word : words) {
            if (word.indexOf(pref) == 0) {
                ++res;
            }
        }
        return res;

    }

    // 6009. 使两字符串互为字母异位词的最少步骤数
    public int minSteps(String s, String t) {
        int res = 0;
        int[] counts1 = new int[26];
        int[] counts2 = new int[26];
        for (char c : s.toCharArray()) {
            ++counts1[c - 'a'];
        }
        for (char c : t.toCharArray()) {
            ++counts2[c - 'a'];
        }
        for (int i = 0; i < counts1.length; ++i) {
            res += Math.abs(counts1[i] - counts2[i]);
        }
        return res;

    }

    // 6010. 完成旅途的最少时间 (Minimum Time to Complete Trips)
    public long minimumTime(int[] time, int totalTrips) {
        Arrays.sort(time);
        long res = -1L;
        long left = 1;
        long right = Long.MAX_VALUE;
        while (left <= right) {
            long mid = left + ((right - left) >> 1);
            if (getRes(time, mid, totalTrips)) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;

    }

    private boolean getRes(int[] time, long second, int totalTrips) {
        long res = 0;
        for (int t : time) {
            res += second / (long) t;
            if (res >= totalTrips) {
                return true;
            }
        }
        return false;
    }

    // 346. 数据流中的移动平均值 (Moving Average from Data Stream) --plus
    class MovingAverage {
        private Queue<Integer> queue;
        private int size;
        private int sum;

        public MovingAverage(int size) {
            this.size = size;
            this.queue = new LinkedList<>();

        }

        public double next(int val) {
            queue.offer(val);
            sum += val;
            while (queue.size() > size) {
                sum -= queue.poll();
            }
            return (double) sum / queue.size();
        }
    }

    // 544. 输出比赛匹配对 (Output Contest Matches) --plus
    public String findContestMatch(int n) {
        String[] strings = new String[n];
        for (int i = 0; i < n; ++i) {
            strings[i] = String.valueOf(i + 1);
        }
        while (n != 0) {
            for (int i = 0; i < n / 2; ++i) {
                strings[i] = "(" + strings[i] + "," + strings[n - i - 1] + ")";
            }
            n /= 2;
        }
        return strings[0];

    }

    // 1022. 从根到叶的二进制数之和 (Sum of Root To Leaf Binary Numbers) --plus dfs
    private int res1022;

    public int sumRootToLeaf(TreeNode root) {
        dfs1022(root, 0);
        return res1022;

    }

    private void dfs1022(TreeNode root, int cur) {
        if (root == null) {
            return;
        }
        int num = (cur << 1) | root.val;
        if (root.left == null && root.right == null) {
            res1022 += num;
            return;
        }
        dfs1022(root.left, num);
        dfs1022(root.right, num);
    }

    // 1022. 从根到叶的二进制数之和 (Sum of Root To Leaf Binary Numbers) --plus bfs
    public int sumRootToLeaf2(TreeNode root) {
        int res = 0;
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        Queue<Integer> sum = new LinkedList<>();
        queue.offer(root);
        sum.offer(root.val);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            int cur = sum.poll();
            if (node.left == null && node.right == null) {
                res += cur;
            }
            if (node.left != null) {
                queue.offer(node.left);
                sum.offer((cur << 1) | node.left.val);
            }
            if (node.right != null) {
                queue.offer(node.right);
                sum.offer((cur << 1) | node.right.val);
            }
        }
        return res;

    }

    // 1554. 只有一个不同字符的字符串 (Strings Differ by One Character) --plus 哈希表
    public boolean differByOne(String[] dict) {
        Set<String> set = new HashSet<>();
        for (String string : dict) {
            char[] chars = string.toCharArray();
            for (int i = 0; i < chars.length; ++i) {
                char temp = chars[i];
                chars[i] = '*';
                if (set.contains(String.valueOf(chars))) {
                    return true;
                }
                set.add(String.valueOf(chars));
                chars[i] = temp;
            }
        }
        return false;
    }

    // 1426. 数元素 (Counting Elements) --plus
    public int countElements(int[] arr) {
        int[] counts = new int[1001];
        for (int num : arr) {
            ++counts[num];
        }
        int res = 0;
        for (int i = 0; i < counts.length - 1; ++i) {
            if (counts[i] != 0 && counts[i + 1] != 0) {
                res += counts[i];
            }
        }
        return res;

    }

    // 1100. 长度为 K 的无重复字符子串 (Find K-Length Substrings With No Repeated Characters)
    public int numKLenSubstrNoRepeats(String s, int k) {
        int res = 0;
        int[] counts = new int[26];
        char[] chars = s.toCharArray();
        int left = 0;
        int right = 0;
        while (right < chars.length) {
            ++counts[chars[right] - 'a'];
            while (counts[chars[right] - 'a'] > 1) {
                --counts[chars[left++] - 'a'];
            }
            if (right - left + 1 >= k) {
                ++res;
            }
            ++right;
        }
        return res;

    }

    // 320. 列举单词的全部缩写 (Generalized Abbreviation) --plus 位运算枚举
    public List<String> generateAbbreviations(String word) {
        List<String> res = new LinkedList<>();
        for (int i = 0; i < (1 << word.length()); ++i) {
            res.add(getAbbr(word, i));
        }
        return res;

    }

    private String getAbbr(String word, int mask) {
        StringBuilder builder = new StringBuilder();
        int k = 0;
        for (int i = 0; i < word.length(); ++i) {
            if ((mask & 1) == 0) {
                if (k != 0) {
                    builder.append(k);
                    k = 0;
                }
                builder.append(word.charAt(i));
            } else {
                ++k;
            }
            mask >>= 1;
        }
        if (k != 0) {
            builder.append(k);
        }
        return builder.toString();
    }

    // 320. 列举单词的全部缩写 (Generalized Abbreviation) --plus 回溯法 待掌握
    public List<String> generateAbbreviations2(String word) {
        List<String> ans = new ArrayList<String>();
        backtrack(ans, new StringBuilder(), word, 0, 0);
        return ans;
    }

    // i is the current position
    // k is the count of consecutive abbreviated characters
    private void backtrack(List<String> ans, StringBuilder builder, String word, int i, int k) {
        int len = builder.length(); // keep the length of builder
        if (i == word.length()) {
            if (k != 0)
                builder.append(k); // append the last k if non zero
            ans.add(builder.toString());
        } else {
            // the branch that word.charAt(i) is abbreviated
            backtrack(ans, builder, word, i + 1, k + 1);

            // the branch that word.charAt(i) is kept
            if (k != 0)
                builder.append(k);
            builder.append(word.charAt(i));
            backtrack(ans, builder, word, i + 1, 0);
        }
        builder.setLength(len); // reset builder to the original state
    }

    // 280. 摆动排序 (Wiggle Sort) --plus
    public void wiggleSort(int[] nums) {
        for (int i = 0; i < nums.length - 1; ++i) {
            if ((i % 2 == 0) == (nums[i] > nums[i + 1])) {
                swap280(nums, i, i + 1);
            }
        }

    }

    private void swap280(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    // 1196. 最多可以买到的苹果数量 (How Many Apples Can You Put into the Basket) --plus
    public int maxNumberOfApples(int[] weight) {
        Arrays.sort(weight);
        int res = 0;
        int sum = 0;
        for (int w : weight) {
            sum += w;
            if (sum > 5000) {
                break;
            }
            ++res;
        }
        return res;

    }

    // 1086. 前五科的均分 (High Five) --plus
    public int[][] highFive(int[][] items) {
        Arrays.sort(items, (o1, o2) -> o1[0] == o2[0] ? o2[1] - o1[1] : o1[0] - o2[0]);

        List<int[]> res = new ArrayList<>();
        int i = 0;
        while (i < items.length) {
            int count = 0;
            int[] item = new int[2];
            item[0] = items[i][0];
            while (count++ < 5) {
                item[1] += items[i][1];
                ++i;
            }
            item[1] /= 5;
            res.add(item);
            while (i < items.length && res.get(res.size() - 1)[0] == items[i][0]) {
                ++i;
            }
        }
        return res.toArray(new int[res.size()][2]);

    }

    // 1101. 彼此熟识的最早时间 (The Earliest Moment When Everyone Become Friends) --plus 并查集
    public int earliestAcq(int[][] logs, int n) {
        Arrays.sort(logs, (o1, o2) -> o1[0] - o2[0]);
        Union1101 union = new Union1101(n);
        for (int[] log : logs) {
            union.union(log[1], log[2]);
            if (union.getCount() == 1) {
                return log[0];
            }
        }
        return -1;
    }

    class Union1101 {
        private int[] parent;
        private int[] rank;
        private int count;

        public Union1101(int n) {
            this.count = n;
            this.parent = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
            }
            this.rank = new int[n];
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

    // "T?T:F?T?1:2:F?3:4"
    // 439. 三元表达式解析器 (Ternary Expression Parser) --plus
    public String parseTernary(String expression) {
        Stack<Character> stack = new Stack<>();
        char[] chars = expression.toCharArray();
        for (int i = chars.length - 1; i >= 0; --i) {
            if (chars[i] == '?') {
                char judge = chars[i - 1];
                char c = stack.pop();
                if (judge == 'T') {
                    stack.pop();
                    stack.push(c);
                }
                --i;
            } else if (chars[i] != ':') {
                stack.push(chars[i]);
            }
        }
        return String.valueOf(stack.pop());

    }

    // 2113. Elements in Array After Removing and Replacing Elements --plus
    public int[] elementInNums(int[] nums, int[][] queries) {
        int n = nums.length;
        int T = n * 2;
        int[] res = new int[queries.length];
        for (int i = 0; i < queries.length; ++i) {
            int t = queries[i][0] % T;
            if (t < n) {
                int index = queries[i][1] + t;
                res[i] = index >= n ? -1 : nums[index];
            } else {
                int index = queries[i][1];
                res[i] = index >= (t - n) ? -1 : nums[index];
            }
        }
        return res;

    }

    // 1256. 加密数字 (Encode Number) --plus
    public String encode(int num) {
        String binaryString = Integer.toBinaryString(++num);
        return binaryString.substring(1);

    }

    // 2067. Number of Equal Count Substrings --plus
    public int equalCountSubstrings(String s, int count) {
        int res = 0;
        int[] counts = new int[26];
        for (int i = 1; i <= 26; ++i) {
            if (i * count > s.length()) {
                break;
            }
            Arrays.fill(counts, 0);
            int x = 0;
            for (int j = 0; j < s.length(); ++j) {
                if (++counts[s.charAt(j) - 'a'] == count) {
                    ++x;
                }
                if (counts[s.charAt(j) - 'a'] == count + 1) {
                    --x;
                }
                if (j - i * count >= 0 && --counts[s.charAt(j - i * count) - 'a'] == count) {
                    ++x;
                }
                if (j - i * count >= 0 && counts[s.charAt(j - i * count) - 'a'] == count - 1) {
                    --x;
                }
                res += x == i ? 1 : 0;
            }
        }
        return res;

    }

    // 531. 孤独像素 I (Lonely Pixel I) --plus
    public int findLonelyPixel(char[][] picture) {
        int m = picture.length;
        int n = picture[0].length;
        int[] rowCount = new int[m];
        int[] colCount = new int[n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (picture[i][j] == 'B') {
                    ++rowCount[i];
                    ++colCount[j];
                }
            }
        }
        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (picture[i][j] == 'B' && rowCount[i] == 1 && colCount[j] == 1) {
                    ++res;
                }
            }
        }
        return res;

    }

    // 243. 最短单词距离 (Shortest Word Distance) --plus
    public int shortestDistance(String[] wordsDict, String word1, String word2) {
        int i1 = -1, i2 = -1;
        int minDistance = wordsDict.length;
        for (int i = 0; i < wordsDict.length; i++) {
            if (wordsDict[i].equals(word1)) {
                i1 = i;
            } else if (wordsDict[i].equals(word2)) {
                i2 = i;
            }
            if (i1 != -1 && i2 != -1) {
                minDistance = Math.min(minDistance, Math.abs(i1 - i2));
            }
        }
        return minDistance;

    }

    // 2128. Remove All Ones With Row and Column Flips --plus
    // 思路：看每一行是否可以转换成第一行即可
    public boolean removeOnes(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        for (int i = 1; i < m; ++i) {
            int x = grid[i][0] ^ grid[0][0];
            for (int j = 1; j < n; ++j) {
                if ((grid[i][j] ^ grid[0][j]) != x) {
                    return false;
                }
            }
        }
        return true;

    }

    // 573. 松鼠模拟 (Squirrel Simulation) --plus (未提交)
    public int minDistance(int height, int width, int[] tree, int[] squirrel, int[][] nuts) {
        int res = Integer.MAX_VALUE;
        int sum = 0;
        for (int[] nut : nuts) {
            sum += getDistance573(nut, tree) * 2;
        }
        for (int[] nut : nuts) {
            int cur = sum - getDistance573(nut, tree) + getDistance573(nut, squirrel);
            res = Math.min(cur, res);
        }
        return res;

    }

    private int getDistance573(int[] nut, int[] tree) {
        return Math.abs(nut[0] - tree[0]) + Math.abs(nut[1] - tree[1]);
    }

    // 2190. 数组中紧跟 key 之后出现最频繁的数字 (Most Frequent Number Following Key In an Array)
    public int mostFrequent(int[] nums, int key) {
        int max = -1;
        int res = -1;
        int[] counts = new int[1001];
        for (int i = 0; i < nums.length - 1; ++i) {
            if (nums[i] == key) {
                ++counts[nums[i + 1]];
                if (counts[nums[i + 1]] > max) {
                    max = counts[nums[i + 1]];
                    res = nums[i + 1];
                }
            }
        }
        return res;

    }

    // 2191. 将杂乱无章的数字排序 (Sort the Jumbled Numbers)
    public int[] sortJumbled(int[] mapping, int[] nums) {
        Bean[] beans = new Bean[nums.length];
        for (int i = 0; i < nums.length; ++i) {
            beans[i] = new Bean(nums[i], getTransNum(nums[i], mapping), i);
        }
        Arrays.sort(beans, (o1, o2) -> o1.transNum != o2.transNum ? o1.transNum - o2.transNum : o1.index - o2.index);
        int[] res = new int[nums.length];
        for (int i = 0; i < nums.length; ++i) {
            res[i] = beans[i].originNum;
        }
        return res;

    }

    private int getTransNum(int num, int[] mapping) {
        if (num == 0) {
            return mapping[num];
        }
        int res = 0;
        int carry = 1;
        while (num != 0) {
            int bit = num % 10;
            res += mapping[bit] * carry;
            carry *= 10;
            num /= 10;
        }
        return res;
    }

    public class Bean {
        int originNum;
        int transNum;
        int index;

        public Bean(int originNum, int transNum, int index) {
            this.originNum = originNum;
            this.transNum = transNum;
            this.index = index;

        }
    }

    // 2055. 蜡烛之间的盘子 (Plates Between Candles) --前缀和
    public int[] platesBetweenCandles(String s, int[][] queries) {
        int n = s.length();
        int[] preSum = new int[n];
        int sum = 0;
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == '*') {
                ++sum;
            }
            preSum[i] = sum;
        }
        // 当前位置左侧的第一个蜡烛的位置
        int[] left = new int[n];
        int l = -1;
        for (int i = 0; i < n; ++i) {
            if (s.charAt(i) == '|') {
                l = i;
            }
            left[i] = l;
        }
        // 当前位置右侧的第一个蜡烛的位置
        int[] right = new int[n];
        l = -1;
        for (int i = n - 1; i >= 0; --i) {
            if (s.charAt(i) == '|') {
                l = i;
            }
            right[i] = l;
        }
        int[] res = new int[queries.length];
        for (int i = 0; i < queries.length; ++i) {
            int L = right[queries[i][0]];
            int R = left[queries[i][1]];
            res[i] = L == -1 || R == -1 || L >= R ? 0 : preSum[R] - preSum[L];
        }
        return res;
    }

    // 506. 相对名次 (Relative Ranks)
    public String[] findRelativeRanks(int[] score) {
        int[][] arr = new int[score.length][2];
        for (int i = 0; i < score.length; ++i) {
            arr[i][0] = score[i];
            arr[i][1] = i;
        }
        Arrays.sort(arr, (o1, o2) -> o2[0] - o1[0]);
        String[] firstThree = { "Gold Medal", "Silver Medal", "Bronze Medal" };
        String[] res = new String[score.length];
        for (int i = 0; i < score.length; ++i) {
            if (i < 3) {
                res[arr[i][1]] = firstThree[i];
            } else {
                res[arr[i][1]] = Integer.toString(i + 1);
            }
        }
        return res;

    }

    // 2194. Excel 表中某个范围内的单元格 (Cells in a Range on an Excel Sheet)
    public List<String> cellsInRange(String s) {
        int startCol = s.charAt(0) - 'A';
        int endCol = s.charAt(3) - 'A';
        int startRow = s.charAt(1) - '0';
        int endRow = s.charAt(4) - '0';
        List<String> res = new ArrayList<>();
        for (int i = startCol; i <= endCol; ++i) {
            for (int j = startRow; j <= endRow; ++j) {
                char a1 = (char) (i + 'A');
                res.add(String.valueOf(a1) + j);
            }
        }
        return res;

    }

    // 2195. 向数组中追加 K 个整数 (Append K Integers With Minimal Sum)
    public long minimalKSum(int[] nums, int k) {
        Arrays.sort(nums);
        int preNum = 0;
        long res = 0L;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] - preNum <= 1) {
                preNum = nums[i];
                continue;
            }
            if (nums[i] - preNum - 1 < k) {
                res += ((long) (preNum + nums[i])) * (nums[i] - preNum - 1) / 2;
                k -= nums[i] - preNum - 1;
                preNum = nums[i];
            } else {
                res += (preNum * 2L + 1 + k) * k / 2;
                return res;
            }
        }
        if (k > 0) {
            res += (preNum * 2L + 1 + k) * k / 2;
        }
        return res;
    }

    // 2196. 根据描述创建二叉树 (Create Binary Tree From Descriptions)
    public TreeNode createBinaryTree(int[][] descriptions) {
        Set<Integer> childs = new HashSet<>();
        Map<Integer, TreeNode> map = new HashMap<>();
        for (int[] description : descriptions) {
            int pValue = description[0];
            int cValue = description[1];
            boolean isLeft = description[2] == 1;
            TreeNode pNode = map.get(pValue);
            if (pNode == null) {
                pNode = new TreeNode(pValue);
                map.put(pValue, pNode);
            }

            TreeNode cNode = map.get(cValue);
            if (cNode == null) {
                cNode = new TreeNode(cValue);
                map.put(cValue, cNode);
            }
            if (isLeft) {
                pNode.left = cNode;
            } else {
                pNode.right = cNode;
            }
            childs.add(cValue);
        }
        for (int value : map.keySet()) {
            if (!childs.contains(value)) {
                return map.get(value);
            }
        }
        return null;

    }

    // 面试题32 - I. 从上到下打印二叉树 --bfs
    public int[] levelOrder(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        if (root == null) {
            return new int[0];
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            list.add(node.val);
            if (node.left != null) {
                queue.offer(node.left);
            }
            if (node.right != null) {
                queue.offer(node.right);
            }
        }
        int[] res = new int[list.size()];
        for (int i = 0; i < res.length; ++i) {
            res[i] = list.get(i);
        }
        return res;

    }

    // 面试题 17.22. 单词转换 (Word Transformer LCCI) --bfs
    public List<String> findLadders(String beginWord, String endWord, List<String> wordList) {
        List<String> res = new ArrayList<>();
        if (!wordList.contains(endWord)) {
            return res;
        }
        boolean[] visited = new boolean[wordList.size()];
        Queue<String> queue = new LinkedList<>();
        Map<String, String> map = new HashMap<>();
        boolean legal = false;
        queue.offer(beginWord);
        while (!queue.isEmpty()) {
            String word = queue.poll();
            if (word.equals(endWord)) {
                legal = true;
                break;
            }
            for (int i = 0; i < wordList.size(); ++i) {
                if (!visited[i] && compare17_22(wordList.get(i), word)) {
                    visited[i] = true;
                    queue.offer(wordList.get(i));
                    map.put(wordList.get(i), word);
                }
            }
        }
        if (!legal) {
            return res;
        }
        String word = endWord;
        while (!map.get(word).equals(beginWord)) {
            res.add(word);
            word = map.get(word);
        }
        res.add(word);
        res.add(map.get(word));

        Collections.reverse(res);
        return res;
    }

    private boolean compare17_22(String word1, String word2) {
        int diff = 0;
        for (int i = 0; i < word1.length(); ++i) {
            if (word1.charAt(i) != word2.charAt(i)) {
                if (++diff > 1) {
                    return false;
                }
            }
        }
        return true;
    }

    // 面试题 08.10. 颜色填充 (Color Fill LCCI)
    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        int m = image.length;
        int n = image[0].length;
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int oldColor = image[sr][sc];
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[] { sr, sc });
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            image[cur[0]][cur[1]] = newColor;
            for (int[] direction : directions) {
                int nx = cur[0] + direction[0];
                int ny = cur[1] + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n
                        && image[nx][ny] == oldColor
                        && image[nx][ny] != newColor) {
                    queue.offer(new int[] { nx, ny });
                }
            }
        }
        return image;
    }

    // 面试题 04.01. 节点间通路 (Route Between Nodes LCCI) --bfs
    public boolean findWhetherExistsPath(int n, int[][] graph, int start, int target) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        boolean[] visited = new boolean[n];
        for (int[] g : graph) {
            if (g[0] == g[1]) {
                continue;
            }
            List<Integer> parent = map.get(g[0]);
            if (parent == null) {
                parent = new ArrayList<>();
                parent.add(g[1]);
            } else if (!parent.contains(g[1])) {
                parent.add(g[1]);
            }
            map.put(g[0], parent);
        }
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(start);
        visited[start] = true;
        while (!queue.isEmpty()) {
            int key = queue.poll();
            if (key == target) {
                return true;
            }
            if (map.get(key) == null) {
                continue;
            }
            for (int num : map.get(key)) {
                if (num == target) {
                    return true;
                }
                if (!visited[num]) {
                    visited[num] = true;
                    queue.offer(num);
                }
            }
        }
        return false;
    }

    // 1379. 找出克隆二叉树中的相同节点 (Find a Corresponding Node of a Binary Tree in a Clone of
    // That Tree) --bfs
    public final TreeNode getTargetCopy(final TreeNode original, final TreeNode cloned, final TreeNode target) {
        if (original == null) {
            return null;
        }
        TreeNode originalNode = original;
        TreeNode clonedNode = cloned;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(originalNode);
        queue.offer(clonedNode);
        while (!queue.isEmpty()) {
            TreeNode oNode = queue.poll();
            TreeNode cNode = queue.poll();
            if (oNode == target) {
                return cNode;
            }
            if (oNode.left != null) {
                queue.offer(oNode.left);
                queue.offer(cNode.left);
            }
            if (oNode.right != null) {
                queue.offer(oNode.right);
                queue.offer(cNode.right);
            }
        }
        return null;
    }

    // 515. 在每个树行中找最大值 (Find Largest Value in Each Tree Row) --bfs
    public List<Integer> largestValues(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            int max = Integer.MIN_VALUE;
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                max = Math.max(max, node.val);

                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }

            }
            res.add(max);
        }
        return res;

    }

    class Employee {
        public int id;
        public int importance;
        public List<Integer> subordinates;
    };

    // 690. 员工的重要性 (Employee Importance) --bfs
    public int getImportance(List<Employee> employees, int id) {
        int res = 0;
        int[] importances = new int[2001];
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (Employee employee : employees) {
            importances[employee.id] = employee.importance;
            map.computeIfAbsent(employee.id, k -> new ArrayList<>()).addAll(employee.subordinates);
        }
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(id);
        while (!queue.isEmpty()) {
            int curId = queue.poll();
            res += importances[curId];
            List<Integer> list = map.get(curId);
            if (list == null) {
                continue;
            }
            for (int item : list) {
                queue.offer(item);
            }
        }
        return res;

    }

    // LCP 07. 传递信息 --bfs 还需掌握dfs、动态规划
    public int numWays(int n, int[][] relation, int k) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int[] r : relation) {
            map.computeIfAbsent(r[0], o -> new ArrayList<>()).add(r[1]);
        }
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(0);

        while (!queue.isEmpty() && k != 0) {
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int cur = queue.poll();
                List<Integer> list = map.get(cur);
                if (list == null) {
                    continue;
                }
                for (int item : list) {
                    queue.offer(item);
                }
            }
            --k;
        }
        if (k != 0) {
            return 0;
        }
        int res = 0;
        while (!queue.isEmpty()) {
            res += queue.poll() == n - 1 ? 1 : 0;
        }
        return res;

    }

    // 1765. 地图中的最高点 (Map of Highest Peak) --多源bfs
    public int[][] highestPeak(int[][] isWater) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        Queue<int[]> queue = new LinkedList<>();
        int m = isWater.length;
        int n = isWater[0].length;
        boolean[][] visited = new boolean[m][n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (isWater[i][j] == 1) {
                    isWater[i][j] = 0;
                    visited[i][j] = true;
                    queue.offer(new int[] { i, j });
                }
            }
        }
        if (queue.size() == m * n) {
            return isWater;
        }
        while (!queue.isEmpty()) {
            int[] p = queue.poll();
            for (int[] direction : directions) {
                int nx = p[0] + direction[0];
                int ny = p[1] + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny]) {
                    visited[nx][ny] = true;
                    isWater[nx][ny] = isWater[p[0]][p[1]] + 1;
                    queue.offer(new int[] { nx, ny });
                }
            }
        }
        return isWater;

    }

    // 1992. 找到所有的农场组 (Find All Groups of Farmland)
    public int[][] findFarmland(int[][] land) {
        int m = land.length;
        int n = land[0].length;
        List<int[]> list = new ArrayList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (land[i][j] == 0) {
                    continue;
                }
                int row = i;
                int col = j;
                while (row + 1 < m && land[row + 1][j] == 1) {
                    ++row;
                }
                while (col + 1 < n && land[i][col + 1] == 1) {
                    ++col;
                }

                list.add(new int[] { i, j, row, col });
                for (int k = i; k <= row; ++k) {
                    for (int l = j; l <= col; ++l) {
                        land[k][l] = 0;
                    }
                }
            }
        }
        return list.toArray(new int[list.size()][]);

    }

    // 417. 太平洋大西洋水流问题 (Pacific Atlantic Water Flow) --bfs
    public List<List<Integer>> pacificAtlantic(int[][] heights) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int m = heights.length;
        int n = heights[0].length;
        Queue<int[]> queue = new LinkedList<>();
        // 可流入太平洋的岛屿
        boolean[][] canFlowToPacificOcean = new boolean[m][n];
        for (int j = 0; j < n; ++j) {
            canFlowToPacificOcean[0][j] = true;
            queue.offer(new int[] { 0, j });
        }
        for (int i = 0; i < m; ++i) {
            canFlowToPacificOcean[i][0] = true;
            queue.offer(new int[] { i, 0 });
        }
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            for (int[] direction : directions) {
                int nx = cur[0] + direction[0];
                int ny = cur[1] + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && !canFlowToPacificOcean[nx][ny]
                        && heights[nx][ny] >= heights[cur[0]][cur[1]]) {
                    canFlowToPacificOcean[nx][ny] = true;
                    queue.offer(new int[] { nx, ny });
                }
            }
        }
        // 可流入大西洋的岛屿
        boolean[][] canFlowToAtlanticOcean = new boolean[m][n];
        for (int j = 0; j < n; ++j) {
            canFlowToAtlanticOcean[m - 1][j] = true;
            queue.offer(new int[] { m - 1, j });
        }
        for (int i = 0; i < m; ++i) {
            canFlowToAtlanticOcean[i][n - 1] = true;
            queue.offer(new int[] { i, n - 1 });
        }
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            for (int[] direction : directions) {
                int nx = cur[0] + direction[0];
                int ny = cur[1] + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && !canFlowToAtlanticOcean[nx][ny]
                        && heights[nx][ny] >= heights[cur[0]][cur[1]]) {
                    canFlowToAtlanticOcean[nx][ny] = true;
                    queue.offer(new int[] { nx, ny });
                }
            }
        }
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (canFlowToPacificOcean[i][j] && canFlowToAtlanticOcean[i][j]) {
                    List<Integer> item = new ArrayList<>();
                    item.add(i);
                    item.add(j);
                    res.add(item);
                }
            }
        }
        return res;

    }

    // 407. 接雨水 II (Trapping Rain Water II)
    // public int trapRainWater(int[][] heightMap) {

    // }

    // 207. 课程表 (Course Schedule)
    // public boolean canFinish(int numCourses, int[][] prerequisites) {

    // }

    // 剑指 Offer II 109. 开密码锁
    // 752. 打开转盘锁 (Open the Lock) --bfs
    // public int openLock(String[] deadends, String target) {

    // }

    // 909. 蛇梯棋 (Snakes and Ladders) --bfs
    // public int snakesAndLadders(int[][] board) {

    // }

}
